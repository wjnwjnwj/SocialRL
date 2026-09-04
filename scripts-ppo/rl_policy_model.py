from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from contextlib import nullcontext
import torch
from peft import LoraConfig, get_peft_model, PeftModel
from typing import List, Optional, Tuple
import numpy as np
import os
from utils import ResponseCandidate, logger, call_local_small_model


def _get_autocast_context(model: torch.nn.Module):
    if not torch.cuda.is_available():
        return nullcontext()
    dtype = next(model.parameters()).dtype
    if dtype not in (torch.float16, torch.bfloat16):
        return nullcontext()
    # Keep autocast aligned with the checkpoint's configured weight dtype.
    return torch.amp.autocast(device_type="cuda", dtype=dtype)


class QwenPolicyTorch(torch.nn.Module):
    """
    Wraps a causal LM with LoRA adapters for fine-tuning.
    Only LoRA parameters are trainable.
    """

    def __init__(
        self,
        model_path: str,
        device: torch.device,
        use_lora: bool = True,
        lora_config: Optional[dict] = None,
        enable_gradient_checkpointing: bool = True,
        enable_kv_cache: bool = False,
        use_device_map_auto: bool = True,
        peft_checkpoint_path: Optional[str] = None,
    ):
        super().__init__()
        self.device = device

        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )

        # 检查是否有pad token，如果没有则设置
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 对于decoder-only模型，生成时需要使用left padding
        self.tokenizer.padding_side = "left"

        if use_device_map_auto:
            # 使用 device_map 自动将模型分布到多个 GPU 上（适用于非 DDP）
            try:
                num_gpus = torch.cuda.device_count()
                logger.info(
                    "Found %d GPUs, using device_map='auto' for model distribution",
                    num_gpus,
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    torch_dtype="auto",
                    device_map="auto",
                    low_cpu_mem_usage=True,
                )
                first_param = next(self.model.parameters())
                actual_device = first_param.device
                logger.info(
                    "Model loaded and distributed across devices, first parameter on %s",
                    actual_device,
                )
            except Exception as e:
                logger.warning(
                    "Failed to use device_map='auto': %s, falling back to single-device load",
                    e,
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    torch_dtype="auto",
                    device_map=None,
                    low_cpu_mem_usage=True,
                )
                self.model.to(device)
        else:
            # DDP 要求每个 rank 的模型参数都在当前 rank 的单个设备上
            logger.info("Loading model on single device for DDP: %s", device)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype="auto",
                device_map=None,
                low_cpu_mem_usage=True,
            )
            self.model.to(device)

        logger.info("Policy model loaded with dtype=%s", next(self.model.parameters()).dtype)

        # 应用LoRA/PEFT适配器 或 从 checkpoint 加载
        self.is_lora = False
        if peft_checkpoint_path:
            logger.info("Loading PEFT adapter from checkpoint: %s", peft_checkpoint_path)
            self.model = PeftModel.from_pretrained(
                self.model,
                peft_checkpoint_path,
                is_trainable=True,
            )
            self.is_lora = True
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    peft_checkpoint_path, trust_remote_code=True
                )
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.padding_side = "left"
            except Exception as e:
                logger.warning("Could not load tokenizer from checkpoint, keeping base: %s", e)
        elif use_lora and get_peft_model is not None:
            try:
                # 默认LoRA配置
                if lora_config is None:
                    lora_config = {
                        "r": 8,
                        "lora_alpha": 16,
                        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                        "lora_dropout": 0.0,
                    }

                lora_cfg = LoraConfig(
                    r=lora_config.get("r", 8),
                    lora_alpha=lora_config.get("lora_alpha", 16),
                    target_modules=lora_config.get("target_modules"),
                    lora_dropout=lora_config.get("lora_dropout", 0.0),
                    bias="none",
                    task_type="CAUSAL_LM",
                )
                self.model = get_peft_model(self.model, lora_cfg)
                self.is_lora = True
                logger.info("Applied LoRA adapters to model")

                # 打印可训练参数
                trainable_params = sum(
                    p.numel() for p in self.model.parameters() if p.requires_grad
                )
                total_params = sum(p.numel() for p in self.model.parameters())
                logger.info(
                    f"LoRA parameters: {trainable_params:,} trainable out of {total_params:,} total"
                )

            except Exception as e:
                logger.warning("Failed to apply LoRA adapters: %s", e)
                # 如果没有LoRA，冻结所有参数
                for p in self.model.parameters():
                    p.requires_grad = False

        self.enable_kv_cache = bool(enable_kv_cache)

        # 内存优化：关闭 KV cache，启用梯度检查点以降低显存占用
        try:
            if hasattr(self.model, "config") and hasattr(
                self.model.config, "use_cache"
            ):
                self.model.config.use_cache = bool(enable_kv_cache)
            if enable_gradient_checkpointing and hasattr(
                self.model, "gradient_checkpointing_enable"
            ):
                try:
                    self.model.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={"use_reentrant": False}
                    )
                except TypeError:
                    # Older Transformers versions do not accept checkpoint kwargs.
                    self.model.gradient_checkpointing_enable()
                if hasattr(self.model, "enable_input_require_grads"):
                    self.model.enable_input_require_grads()
                logger.info("Enabled non-reentrant gradient checkpointing")
        except Exception as e:
            logger.warning("Failed to enable memory optimizations: %s", e)

        self.model.eval()  # 设置为评估模式，但在训练时会启用梯度

        # 延迟创建优化器
        self.optimizer = None

    def create_optimizer(self, lr: float = 1e-2):
        """Create optimizer for the LoRA parameters."""
        # 收集所有可训练参数（LoRA参数）
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        if not trainable_params:
            logger.warning("No trainable parameters found! Check LoRA configuration.")
            # 如果没有可训练参数，尝试找到模型中的参数
            trainable_params = list(self.model.parameters())

        logger.info(f"Creating optimizer with {len(trainable_params)} parameter groups")
        logger.info(
            f"Total trainable parameters: {sum(p.numel() for p in trainable_params):,}"
        )

        self.optimizer = torch.optim.Adam(trainable_params, lr=lr)
        return self.optimizer

    def generate_candidates(
        self,
        prompt: str,
        num_candidates: int,
        max_new_tokens: int = 32,
        temperature: float = 1.0,
    ) -> List[ResponseCandidate]:
        # 获取模型（如果是DataParallel，需要获取底层模型）
        if hasattr(self.model, "module"):
            model_for_gen = self.model.module
        else:
            model_for_gen = self.model

        raw = call_local_small_model(
            prompt,
            {"model": model_for_gen, "tokenizer": self.tokenizer},
            temperature,
            num_responses=num_candidates,
            max_new_tokens=max_new_tokens,
        )

        if not raw:
            return []

        if isinstance(raw, str):
            responses = [raw]
        else:
            responses = list(raw)

        candidates: List[ResponseCandidate] = []
        for i, text in enumerate(responses):
            candidates.append(ResponseCandidate(text=text.strip(), index=i))
        return candidates

    def clear_kv_cache(self) -> None:
        model = self.model.module if hasattr(self.model, "module") else self.model
        if hasattr(model, "clear_cache") and callable(getattr(model, "clear_cache")):
            try:
                model.clear_cache()
                return
            except Exception:
                pass

        for attr in (
            "past_key_values",
            "_past_key_values",
            "kv_cache",
            "_kv_cache",
            "_cache",
            "cache",
        ):
            if not hasattr(model, attr):
                continue
            try:
                value = getattr(model, attr)
                if hasattr(value, "clear") and callable(getattr(value, "clear")):
                    value.clear()
                else:
                    setattr(model, attr, None)
            except Exception:
                continue

    def generate_candidates_batch(
        self,
        messages: List,
        max_new_tokens: int = 32,
        temperature: float = 1.0,
    ) -> List[List[ResponseCandidate]]:
        if not messages:
            return []

        # messages 是 List[List[Dict[str,str]]]，使用模型 tokenizer 自带的 chat template 渲染。
        prompts = [
            self.tokenizer.apply_chat_template(
                m,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            if isinstance(m, list)
            else m
            for m in messages
        ]

        if hasattr(self.model, "module"):
            model_for_gen = self.model.module
        else:
            model_for_gen = self.model

        # 避免上一轮遗留 cache 导致 attention_mask 与 cache 形状不一致。
        self.clear_kv_cache()
        was_training = self.model.training
        self.model.eval()

        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True)
        device = next(model_for_gen.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": True if temperature and temperature > 0 else False,
            "temperature": float(temperature),
            "top_k": 50,
            "num_return_sequences": 1,
            "use_cache": self.enable_kv_cache,
            "remove_invalid_values": True,
        }
        try:
            with torch.inference_mode():
                with _get_autocast_context(model_for_gen):
                    generated = model_for_gen.generate(**inputs, **gen_kwargs)
        finally:
            if was_training:
                self.model.train()

        # 使用输入序列长度截取：generate() 在整条输入之后生成，新 token 从 input_length 开始
        # 不能用“非 pad token 数”：left padding 时 prompt 在右侧，非 pad 数不等于截取起点
        input_length = inputs["input_ids"].size(1)
        batch_size = len(messages)
        per_prompt: List[List[ResponseCandidate]] = []
        for i in range(batch_size):
            seq = generated[i]
            gen_ids = seq[input_length:]
            text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            per_prompt.append([ResponseCandidate(text=text, index=0)])
        return per_prompt

    def compute_logprob_of_sequences(
        self, input_prompts: List[str], sequences: List[str]
    ) -> torch.Tensor:
        """
        Compute the log-probability of each full sequence under the current policy
        (prompt + generated text). Returns tensor of shape (batch,).
        """
        batch_log_probs = []

        # 获取正确的模型（处理DataParallel情况）
        if hasattr(self.model, "module"):
            model_for_comp = self.model.module
        else:
            model_for_comp = self.model

        for prompt, seq in zip(input_prompts, sequences):
            combined = (prompt + " " + seq).strip()

            # 获取设备
            if hasattr(model_for_comp, "device"):
                device = model_for_comp.device
            else:
                device = next(model_for_comp.parameters()).device

            tokens = self.tokenizer(combined, return_tensors="pt").to(device)
            input_ids = tokens["input_ids"]

            with torch.enable_grad():
                with _get_autocast_context(model_for_comp):
                    outputs = model_for_comp(input_ids)
                    logits = outputs.logits  # (1, seq_len, vocab)
                    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

                # 计算序列的对数概率
                seq_len = input_ids.shape[1]
                if seq_len <= 1:
                    batch_log_probs.append(torch.tensor(0.0, device=device))
                    continue

                token_log_probs = []
                for t in range(seq_len - 1):
                    target_id = input_ids[0, t + 1]
                    token_log_prob = log_probs[0, t, target_id]
                    token_log_probs.append(token_log_prob)

                seq_logprob = torch.stack(token_log_probs).sum()
                batch_log_probs.append(seq_logprob)

        return torch.stack(batch_log_probs)

    def compute_response_logprob_and_entropy(
        self,
        input_prompts: List[str],
        sequences: List[str],
        require_grad: bool = True,
        with_disabled_adapter: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute log pi(a|s) where action a is the full response text.
        Only response-token logprobs are summed; prompt tokens are excluded.
        Also returns mean token entropy for each sample.

        When `with_disabled_adapter=True`, the PEFT adapter is temporarily
        switched off via `PeftModel.disable_adapter()`, so the returned
        logprobs come from the (frozen) base model — i.e. the reference
        policy used for the PPO KL penalty `β·KL(π_θ ‖ π_ref)`. This costs
        only one extra forward pass and no extra GPU memory, since the base
        weights are already loaded.
        """
        if not input_prompts:
            device = next(self.model.parameters()).device
            empty = torch.zeros(0, device=device, dtype=torch.float32)
            return empty, empty

        # 获取正确的模型（处理 DDP/DataParallel）
        if hasattr(self.model, "module"):
            model_for_comp = self.model
        else:
            model_for_comp = self.model

        # Reference policy = current policy with adapter disabled (= base).
        # disable_adapter() is defined on PeftModel; unwrap DDP first if needed.
        underlying = (
            self.model.module if hasattr(self.model, "module") else self.model
        )
        if with_disabled_adapter and hasattr(underlying, "disable_adapter"):
            adapter_ctx = underlying.disable_adapter()
        else:
            adapter_ctx = nullcontext()

        logprob_list: List[torch.Tensor] = []
        entropy_list: List[torch.Tensor] = []

        grad_ctx = torch.enable_grad() if require_grad else torch.no_grad()
        with grad_ctx, adapter_ctx:
            for prompt, seq in zip(input_prompts, sequences):
                combined = (prompt + " " + seq).strip()
                if hasattr(model_for_comp, "device"):
                    device = model_for_comp.device
                else:
                    device = next(model_for_comp.parameters()).device

                prompt_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(
                    device
                )
                combined_ids = self.tokenizer(
                    combined, return_tensors="pt"
                )["input_ids"].to(device)
                seq_len = int(combined_ids.shape[1])
                prompt_len = int(prompt_ids.shape[1])

                # No response token after tokenization.
                if seq_len <= prompt_len:
                    logprob_list.append(torch.tensor(0.0, device=device))
                    entropy_list.append(torch.tensor(0.0, device=device))
                    continue

                with _get_autocast_context(model_for_comp):
                    outputs = model_for_comp(combined_ids)
                    logits = outputs.logits  # (1, seq_len, vocab)
                    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                    probs = torch.softmax(logits, dim=-1)

                # Score response tokens only:
                # token index j in [prompt_len, seq_len-1] is predicted at logit index (j-1).
                pred_pos = torch.arange(
                    prompt_len - 1, seq_len - 1, device=device, dtype=torch.long
                )
                target_ids = combined_ids[0, prompt_len:seq_len]
                token_log_probs = log_probs[0, pred_pos, target_ids]
                seq_logprob = token_log_probs.sum()
                logprob_list.append(seq_logprob)

                token_entropies = -(
                    probs[0, pred_pos] * log_probs[0, pred_pos]
                ).sum(dim=-1)
                entropy_list.append(token_entropies.mean())

        return torch.stack(logprob_list), torch.stack(entropy_list)

    def step_policy_gradient(
        self,
        input_prompts: List[str],
        sequences: List[str],
        rewards: List[float],
        kl_constraint: float = 0.02,
        batch_size: Optional[int] = None,
    ):
        """
        Perform policy gradient update using LoRA parameters.
        """
        if self.optimizer is None:
            self.create_optimizer(lr=1e-2)

        # 训练阶段开启 train 模式，支持梯度检查点
        was_training = self.model.training
        self.model.train()

        # 保存旧参数用于KL计算
        old_params = {}
        if hasattr(self.model, "module"):
            model_for_state = self.model.module
        else:
            model_for_state = self.model

        # 保存可训练参数的旧状态
        for name, param in model_for_state.named_parameters():
            if param.requires_grad:
                old_params[name] = param.data.clone()

        self.optimizer.zero_grad()

        # 获取设备
        device = next(self.model.parameters()).device

        # 全量奖励张量与基线
        rewards_t_full = torch.tensor(rewards, dtype=torch.float32, device=device)
        n = rewards_t_full.shape[0] if rewards_t_full.ndim > 0 else 0
        baseline = rewards_t_full.mean() if n > 0 else torch.tensor(0.0, device=device)

        # 如果未指定 batch_size 或 batch_size >= 全量大小，则按原方式一次处理
        if not batch_size or batch_size <= 0 or batch_size >= max(1, n):
            log_probs = self.compute_logprob_of_sequences(input_prompts, sequences)
            advantages = rewards_t_full - baseline
            loss = -(advantages * log_probs).mean()
            loss_val = float(loss.detach().cpu().item())

            # 反向传播并更新
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad], max_norm=1.0
            )
            self.optimizer.step()
        else:
            # 分块处理：对每个小块计算损失并调用 backward，按整体样本数进行缩放，避免保留全部激活
            use_ddp_no_sync = hasattr(self.model, "no_sync") and callable(
                getattr(self.model, "no_sync")
            )
            loss_val_accum = torch.tensor(0.0, device=device)
            for i in range(0, n, batch_size):
                logger.info(f"device:{self.device} all:{n} now:{i} batch_size:{batch_size}")
                is_last_chunk = (i + batch_size) >= n
                sync_ctx = nullcontext()
                if use_ddp_no_sync and not is_last_chunk:
                    # DDP 下仅在最后一个 chunk 同步梯度，避免各 rank chunk 数不同导致卡住。
                    sync_ctx = self.model.no_sync()

                with sync_ctx:
                    slice_prompts = input_prompts[i : i + batch_size]
                    slice_sequences = sequences[i : i + batch_size]
                    slice_rewards = rewards_t_full[i : i + batch_size]

                    log_probs_chunk = self.compute_logprob_of_sequences(
                        slice_prompts, slice_sequences
                    )
                    advantages_chunk = slice_rewards - baseline
                    loss_chunk = (
                        -(advantages_chunk * log_probs_chunk).sum() / max(1, n)
                    )
                    loss_val_accum = loss_val_accum + loss_chunk.detach()
                    loss_chunk.backward()

            # 直接汇总 chunk loss 作为日志 loss，避免再做一次全量前向导致显存峰值翻倍。
            loss_val = float(loss_val_accum.detach().cpu().item())

            # 梯度裁剪并更新
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad], max_norm=1.0
            )
            self.optimizer.step()

        # 计算KL散度 - 比较更新前后模型对同一个输入的概率分布
        if input_prompts:
            sample_prompt = input_prompts[0]

            # 获取正确的模型
            if hasattr(self.model, "module"):
                model_for_kl = self.model.module
            else:
                model_for_kl = self.model

            device = next(model_for_kl.parameters()).device

            # 使用旧参数计算旧分布
            with torch.no_grad():
                # 临时保存新参数
                new_params = {}
                for name, param in model_for_kl.named_parameters():
                    if param.requires_grad:
                        new_params[name] = param.data.clone()
                        param.data.copy_(old_params[name])  # 恢复旧参数

                # 用旧参数计算logits（取最后一个时间步）
                inputs = self.tokenizer(sample_prompt, return_tensors="pt").to(device)
                old_logits = model_for_kl(**inputs).logits[:, -1, :]

                # 恢复新参数
                for name, param in model_for_kl.named_parameters():
                    if param.requires_grad and name in new_params:
                        param.data.copy_(new_params[name])

                # 用新参数计算logits（取最后一个时间步）
                new_logits = model_for_kl(**inputs).logits[:, -1, :]
                old_logits_f = old_logits.to(torch.float32)
                new_logits_f = new_logits.to(torch.float32)

                old_logsumexp = torch.logsumexp(old_logits_f, dim=-1)  # shape (batch,)
                new_logsumexp = torch.logsumexp(new_logits_f, dim=-1)

                old_probs = torch.softmax(old_logits_f, dim=-1)
                weighted_diff = (old_probs * (old_logits_f - new_logits_f)).sum(dim=-1)
                kl = (weighted_diff + (new_logsumexp - old_logsumexp)).item()

                # 如果KL太大，回退到旧参数
                if kl > kl_constraint and kl > 0:
                    scale = kl_constraint / kl
                    for name, param in model_for_kl.named_parameters():
                        if param.requires_grad and name in old_params:
                            # 线性插值：new_param = old_param + scale * (new_param - old_param)
                            param.data.copy_(
                                old_params[name]
                                + scale * (param.data - old_params[name])
                            )
                    logger.debug(
                        f"KL exceeded constraint: {kl:.6f}, scaled update by {scale:.4f}"
                    )
                    # 重新计算KL
                    new_logits = model_for_kl(**inputs).logits[:, -1, :]

                    # 使用与上面相同的数值稳定计算方法（将 logits 转为 float32）
                    new_logits_f = new_logits.to(torch.float32)
                    new_logsumexp = torch.logsumexp(new_logits_f, dim=-1)
                    # old_logits_f 和 old_logsumexp 已在上面计算过
                    weighted_diff = (old_probs * (old_logits_f - new_logits_f)).sum(
                        dim=-1
                    )
                    kl = (weighted_diff + (new_logsumexp - old_logsumexp)).item()

        # build reward statistics for logging
        try:
            rewards_arr = (
                np.array([float(x) for x in rewards])
                if len(rewards) > 0
                else np.array([])
            )
            reward_mean = float(rewards_arr.mean()) if rewards_arr.size else 0.0
            reward_std = float(rewards_arr.std()) if rewards_arr.size else 0.0
            reward_min = float(rewards_arr.min()) if rewards_arr.size else 0.0
            reward_max = float(rewards_arr.max()) if rewards_arr.size else 0.0
        except Exception:
            reward_mean = reward_std = reward_min = reward_max = 0.0
            rewards_arr = np.array([])

        result = {
            "loss": loss_val,
            "kl": kl,
            "grad_norm": float(grad_norm) if grad_norm is not None else 0.0,
            "reward_mean": reward_mean,
            "reward_std": reward_std,
            "reward_min": reward_min,
            "reward_max": reward_max,
            "rewards": rewards_arr.tolist()
        }
        # 对话采样阶段依赖 eval()，更新结束后统一切回 eval。
        self.model.eval()
        return result

    def save(self, path: str):
        """Save LoRA adapters."""
        parent_dir = os.path.dirname(path) or "."
        try:
            os.makedirs(parent_dir, exist_ok=True)
        except Exception:
            pass

        # 保存LoRA适配器
        try:
            # 获取底层模型（如果是DataParallel）
            if hasattr(self.model, "module"):
                model_to_save = self.model.module
            else:
                model_to_save = self.model

            # 保存LoRA权重
            if hasattr(model_to_save, "save_pretrained"):
                model_to_save.save_pretrained(path)
                logger.info("Saved LoRA adapters to %s", path)

                # 保存tokenizer
                self.tokenizer.save_pretrained(path)
            else:
                logger.warning("Model does not have save_pretrained method")

        except Exception as e:
            logger.error(f"Failed to save LoRA adapters: {e}")


class _QwenCriticModel(torch.nn.Module):
    """
    Critic model with LoRA-tunable backbone + trainable value head.
    """

    def __init__(self, backbone: torch.nn.Module, hidden_size: int):
        super().__init__()
        self.backbone = backbone
        self.value_head = torch.nn.Linear(hidden_size, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden = outputs.hidden_states[-1]  # (B, L, H)

        last_pos = attention_mask.long().sum(dim=1) - 1
        last_pos = torch.clamp(last_pos, min=0)
        batch_idx = torch.arange(hidden.size(0), device=hidden.device)
        last_hidden = hidden[batch_idx, last_pos]  # (B, H)
        values = self.value_head(last_hidden.float()).squeeze(-1)
        return values


class QwenCriticTorch(torch.nn.Module):
    """
    Separate critic model with its own optimizer.
    """

    def __init__(
        self,
        model_path: str,
        device: torch.device,
        use_device_map_auto: bool = True,
        use_lora: bool = True,
        lora_config: Optional[dict] = None,
        enable_gradient_checkpointing: bool = True,
        checkpoint_path: Optional[str] = None,
    ):
        super().__init__()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        if use_device_map_auto:
            backbone = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype="auto",
                device_map="auto",
                low_cpu_mem_usage=True,
            )
        else:
            backbone = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype="auto",
                device_map=None,
                low_cpu_mem_usage=True,
            )
            backbone.to(device)

        logger.info("Critic backbone loaded with dtype=%s", next(backbone.parameters()).dtype)

        self.is_lora = False
        adapter_config_path = (
            os.path.join(checkpoint_path, "adapter_config.json")
            if checkpoint_path
            else None
        )
        if adapter_config_path and os.path.isfile(adapter_config_path):
            logger.info("Loading critic PEFT adapter from checkpoint: %s", checkpoint_path)
            backbone = PeftModel.from_pretrained(
                backbone,
                checkpoint_path,
                is_trainable=True,
            )
            self.is_lora = True
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    checkpoint_path, trust_remote_code=True
                )
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.padding_side = "left"
            except Exception as e:
                logger.warning(
                    "Could not load critic tokenizer from checkpoint, using base: %s", e
                )
        elif use_lora and get_peft_model is not None:
            if lora_config is None:
                lora_config = {
                    "r": 8,
                    "lora_alpha": 16,
                    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                    "lora_dropout": 0.0,
                }
            lora_cfg = LoraConfig(
                r=lora_config.get("r", 8),
                lora_alpha=lora_config.get("lora_alpha", 16),
                target_modules=lora_config.get("target_modules"),
                lora_dropout=lora_config.get("lora_dropout", 0.0),
                bias="none",
                task_type="CAUSAL_LM",
            )
            backbone = get_peft_model(backbone, lora_cfg)
            self.is_lora = True
            logger.info("Applied LoRA adapters to critic backbone")
        else:
            logger.warning(
                "Critic LoRA is disabled or unavailable; full-backbone critic training may be expensive."
            )
            for p in backbone.parameters():
                p.requires_grad = True

        hidden_size = int(getattr(backbone.config, "hidden_size"))
        self.model = _QwenCriticModel(backbone=backbone, hidden_size=hidden_size)
        if not use_device_map_auto:
            self.model.to(device)
        self.optimizer = None

        try:
            if hasattr(backbone, "config") and hasattr(backbone.config, "use_cache"):
                backbone.config.use_cache = False
            if enable_gradient_checkpointing and hasattr(
                backbone, "gradient_checkpointing_enable"
            ):
                try:
                    backbone.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={"use_reentrant": False}
                    )
                except TypeError:
                    backbone.gradient_checkpointing_enable()
                if hasattr(backbone, "enable_input_require_grads"):
                    backbone.enable_input_require_grads()
                logger.info("Enabled non-reentrant critic gradient checkpointing")
        except Exception as e:
            logger.warning("Failed to configure critic memory options: %s", e)

        # Value head must always be trainable.
        for p in self.model.value_head.parameters():
            p.requires_grad = True

        if checkpoint_path:
            self._load_value_head(checkpoint_path)

    def _load_value_head(self, path: str) -> None:
        head_path = os.path.join(path, "critic_value_head.pt")
        if not os.path.isfile(head_path):
            return
        state = torch.load(head_path, map_location="cpu")
        model_to_load = self.model.module if hasattr(self.model, "module") else self.model
        model_to_load.value_head.load_state_dict(state)
        logger.info("Loaded critic value head from %s", head_path)

    def create_optimizer(self, lr: float = 1e-5):
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        logger.info(
            "Creating critic optimizer with %d trainable tensors (%d params)",
            len(trainable_params),
            sum(p.numel() for p in trainable_params),
        )
        self.optimizer = torch.optim.Adam(trainable_params, lr=lr)
        return self.optimizer

    def evaluate_values(
        self, prompts: List[str], require_grad: bool = False
    ) -> torch.Tensor:
        if not prompts:
            device = next(self.model.parameters()).device
            return torch.zeros(0, device=device, dtype=torch.float32)

        model_for_eval = self.model
        if hasattr(model_for_eval, "device"):
            device = model_for_eval.device
        else:
            device = next(model_for_eval.parameters()).device

        tokens = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)

        grad_ctx = torch.enable_grad() if require_grad else torch.no_grad()
        with grad_ctx:
            values = model_for_eval(input_ids=input_ids, attention_mask=attention_mask)
        return values

    def save(self, path: str):
        parent_dir = os.path.dirname(path) or "."
        try:
            os.makedirs(parent_dir, exist_ok=True)
        except Exception:
            pass
        os.makedirs(path, exist_ok=True)
        self.tokenizer.save_pretrained(path)
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        if hasattr(model_to_save.backbone, "save_pretrained"):
            model_to_save.backbone.save_pretrained(path)
        torch.save(
            model_to_save.value_head.state_dict(),
            os.path.join(path, "critic_value_head.pt"),
        )
        logger.info("Saved critic value head to %s", path)
