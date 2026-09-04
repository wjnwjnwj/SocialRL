from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from contextlib import nullcontext
import torch
from peft import LoraConfig, get_peft_model, PeftModel
from typing import List, Optional
import numpy as np
import os
from utils import ResponseCandidate, logger, call_local_small_model


def _get_autocast_context():
    if not torch.cuda.is_available():
        return nullcontext()
    # 使用新的 API 避免 FutureWarning
    return torch.amp.autocast(device_type="cuda")


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
                    torch_dtype=torch.float16,
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
                    torch_dtype=torch.float16,
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
                torch_dtype=torch.float16,
                device_map=None,
                low_cpu_mem_usage=True,
            )
            self.model.to(device)

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
                self.model.gradient_checkpointing_enable()
                if hasattr(self.model, "enable_input_require_grads"):
                    self.model.enable_input_require_grads()
                logger.info("Enabled gradient checkpointing")
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
        prompts: List[str],
        max_new_tokens: int = 32,
        temperature: float = 1.0,
    ) -> List[List[ResponseCandidate]]:
        if not prompts:
            return []

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
        }
        try:
            with torch.inference_mode():
                with _get_autocast_context():
                    generated = model_for_gen.generate(**inputs, **gen_kwargs)
        finally:
            if was_training:
                self.model.train()

        # 使用输入序列长度截取：generate() 在整条输入之后生成，新 token 从 input_length 开始
        # 不能用“非 pad token 数”：left padding 时 prompt 在右侧，非 pad 数不等于截取起点
        input_length = inputs["input_ids"].size(1)
        batch_size = len(prompts)
        per_prompt: List[List[ResponseCandidate]] = []
        for i in range(batch_size):
            seq = generated[i]
            gen_ids = seq[input_length:]
            text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            per_prompt.append([ResponseCandidate(text=text, index=0)])
        return per_prompt

    def compute_logprob_of_sequences(
        self,
        input_prompts: List[str],
        sequences: List[str],
        return_entropy: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the log-probability of each full sequence under the current policy
        (prompt + generated text). Returns tensor of shape (batch,).
        """
        batch_log_probs = []
        batch_entropies = []

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
                with _get_autocast_context():
                    outputs = model_for_comp(input_ids)
                    logits = outputs.logits  # (1, seq_len, vocab)
                    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                    probs = torch.exp(log_probs)

                # 计算序列的对数概率
                seq_len = input_ids.shape[1]
                if seq_len <= 1:
                    batch_log_probs.append(torch.tensor(0.0, device=device))
                    batch_entropies.append(torch.tensor(0.0, device=device))
                    continue

                token_log_probs = []
                token_entropies = []
                for t in range(seq_len - 1):
                    target_id = input_ids[0, t + 1]
                    token_log_prob = log_probs[0, t, target_id]
                    token_log_probs.append(token_log_prob)
                    # Entropy bonus uses token distribution entropy H(pi(.|s_t)).
                    token_entropy = -(probs[0, t] * log_probs[0, t]).sum()
                    token_entropies.append(token_entropy)

                seq_logprob = torch.stack(token_log_probs).sum()
                seq_entropy = torch.stack(token_entropies).mean()
                batch_log_probs.append(seq_logprob)
                batch_entropies.append(seq_entropy)

        stacked_log_probs = torch.stack(batch_log_probs)
        if return_entropy:
            return stacked_log_probs, torch.stack(batch_entropies)
        return stacked_log_probs

    def compute_response_logprob_and_entropy(
        self,
        input_prompts: List[str],
        sequences: List[str],
        require_grad: bool = True,
    ) -> tuple:
        """Compute per-sequence log-prob and mean token entropy.

        Returns (log_probs, entropies), each a 1-D tensor of shape (N,).
        Used by both the rollout phase (require_grad=False) and the update
        phase (require_grad=True).
        """
        model = self.model.module if hasattr(self.model, "module") else self.model
        device = next(model.parameters()).device

        batch_log_probs = []
        batch_entropies = []
        ctx = torch.enable_grad() if require_grad else torch.no_grad()
        with ctx:
            for prompt, seq in zip(input_prompts, sequences):
                combined = (prompt + " " + seq).strip()
                tokens = self.tokenizer(combined, return_tensors="pt").to(device)
                input_ids = tokens["input_ids"]
                with _get_autocast_context():
                    outputs = model(input_ids)
                    logits = outputs.logits
                    log_probs_all = torch.nn.functional.log_softmax(logits, dim=-1)
                seq_len = input_ids.shape[1]
                if seq_len <= 1:
                    batch_log_probs.append(torch.tensor(0.0, device=device, requires_grad=require_grad))
                    batch_entropies.append(torch.tensor(0.0, device=device))
                    continue
                token_lp = []
                token_ent = []
                for t in range(seq_len - 1):
                    tid = input_ids[0, t + 1]
                    token_lp.append(log_probs_all[0, t, tid])
                    probs_t = torch.exp(log_probs_all[0, t])
                    token_ent.append(-(probs_t * log_probs_all[0, t]).sum())
                batch_log_probs.append(torch.stack(token_lp).sum())
                batch_entropies.append(torch.stack(token_ent).mean())

        return torch.stack(batch_log_probs), torch.stack(batch_entropies)

    def step_policy_gradient(
        self,
        input_prompts: List[str],
        sequences: List[str],
        advantages: List[float],
        old_logprobs: Optional[List[float]] = None,
        kl_constraint: float = 0.02,
        entropy_coef: float = 0.0,
        ppo_clip: float = 0.2,
        batch_size: Optional[int] = None,
    ):
        """Process-GRPO policy update with clipped importance-ratio objective.

        Implements equation (3) from the Process-GRPO formulation:
          J = mean_i { mean_t [ min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)
                                 - β * KL(π_θ || π_ref) ] }
        where r_t = π_θ(o_t|ctx) / π_θ_old(o_t|ctx) and A_t is the
        per-token advantage produced by compute_advantages (backward-cumulative
        sum of standardised process rewards).

        When old_logprobs is None the method falls back to vanilla PG
        (ratio = 1, no clipping) for backward compatibility.
        """
        if self.optimizer is None:
            self.create_optimizer(lr=1e-2)

        self.model.train()
        device = next(self.model.parameters()).device
        n = len(sequences)
        if n == 0:
            self.model.eval()
            return {}

        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=device)
        use_ratio = old_logprobs is not None and len(old_logprobs) == n
        if use_ratio:
            old_lp_t = torch.tensor(old_logprobs, dtype=torch.float32, device=device)

        chunk = int(batch_size) if batch_size and batch_size > 0 else n
        use_ddp_no_sync = hasattr(self.model, "no_sync") and callable(
            getattr(self.model, "no_sync")
        )

        self.optimizer.zero_grad()
        loss_accum = 0.0
        pg_loss_accum = 0.0
        entropy_accum = 0.0
        approx_kl_accum = 0.0
        clipfrac_accum = 0.0

        for i in range(0, n, chunk):
            is_last = (i + chunk) >= n
            sync_ctx = nullcontext()
            if use_ddp_no_sync and not is_last:
                sync_ctx = self.model.no_sync()

            with sync_ctx:
                sl_prompts = input_prompts[i : i + chunk]
                sl_seqs = sequences[i : i + chunk]
                sl_adv = advantages_t[i : i + chunk]
                c = len(sl_seqs)

                new_lp, entropies = self.compute_response_logprob_and_entropy(
                    sl_prompts, sl_seqs, require_grad=True
                )

                if use_ratio:
                    sl_old_lp = old_lp_t[i : i + chunk]
                    log_ratio = new_lp - sl_old_lp
                    ratio = torch.exp(log_ratio)
                    surr1 = ratio * sl_adv
                    surr2 = torch.clamp(ratio, 1.0 - ppo_clip, 1.0 + ppo_clip) * sl_adv
                    pg_loss_chunk = -torch.min(surr1, surr2).mean()
                    with torch.no_grad():
                        approx_kl_accum += float((sl_old_lp - new_lp).mean().item()) * c / n
                        clipfrac_accum += float(((ratio - 1.0).abs() > ppo_clip).float().mean().item()) * c / n
                else:
                    pg_loss_chunk = -(sl_adv * new_lp).mean()

                entropy_chunk = entropies.mean()
                loss_chunk = pg_loss_chunk - float(entropy_coef) * entropy_chunk
                scale = c / n
                (loss_chunk * scale).backward()

                loss_accum += float(loss_chunk.detach().cpu().item()) * scale
                pg_loss_accum += float(pg_loss_chunk.detach().cpu().item()) * scale
                entropy_accum += float(entropy_chunk.detach().cpu().item()) * scale

        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad], max_norm=1.0
        )
        self.optimizer.step()

        # ── KL check against old params (parameter-space rollback) ──────────
        kl = float(approx_kl_accum)
        if (not use_ratio) and kl_constraint > 0 and input_prompts:
            # Legacy fallback: measure KL via logit comparison on one sample.
            model_base = self.model.module if hasattr(self.model, "module") else self.model
            with torch.no_grad():
                inputs = self.tokenizer(input_prompts[0], return_tensors="pt").to(device)
                new_logits = model_base(**inputs).logits[:, -1, :].to(torch.float32)
                kl = 0.0  # no reference; skip rollback

        self.model.eval()
        try:
            adv_arr = np.array([float(x) for x in advantages])
        except Exception:
            adv_arr = np.array([])

        return {
            "loss": loss_accum,
            "pg_loss": pg_loss_accum,
            "entropy": entropy_accum,
            "entropy_coef": float(entropy_coef),
            "approx_kl": approx_kl_accum,
            "clipfrac": clipfrac_accum,
            "kl": approx_kl_accum,
            "grad_norm": float(grad_norm) if grad_norm is not None else 0.0,
            "advantage_mean": float(adv_arr.mean()) if adv_arr.size else 0.0,
            "advantage_std": float(adv_arr.std()) if adv_arr.size else 0.0,
        }

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
