import os
from openai import OpenAI
from typing import Dict, List, Optional, Any, Union
import json
import torch
import torch.distributed as dist
import numpy as np
from dataclasses import dataclass
import logging
import time
import random
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import requests

stop_token = "[LEAVE]"

from prompt import new_system_prompt, system_prompt



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rl_qwen")

# Reduce noisy HTTP client / telemetry logs in terminal output.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("wandb").setLevel(logging.WARNING)
logging.getLogger("weave").setLevel(logging.WARNING)
logging.getLogger("weave.trace").setLevel(logging.WARNING)


def load_backgrounds(path: str) -> List[dict]:
    backgrounds = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            backgrounds.append(json.loads(line))
    return backgrounds



def build_new_system_prompt_with_partner(
    agent_name: str,
    agent_background: str,
    partner_background: str,
    agent_goal: Optional[Union[str, Dict]],
    scenario: Optional[str],
) -> str:
    """
    使用 prompt.new_system_prompt 构建系统提示词（包含 partner_background）。
    保留旧函数 build_new_system_prompt 以兼容历史逻辑。
    """
    extra_info = ""
    clarification_hint = ""
    strategy_hint = ""

    goal_text = agent_goal
    if goal_text and isinstance(goal_text, dict):
        main_text = goal_text.get("main_text", "")
        tags = goal_text.get("tags", {})
        extra_info_items = tags.get("extra_info", [])
        if extra_info_items:
            extra_info = " ".join(
                [item.get("content", "") for item in extra_info_items if isinstance(item, dict)]
            )
        clarification_hint_items = tags.get("clarification_hint", [])
        if clarification_hint_items:
            clarification_hint = " ".join(
                [
                    item.get("content", "")
                    for item in clarification_hint_items
                    if isinstance(item, dict)
                ]
            )
        strategy_hint_items = tags.get("strategy_hint", [])
        if strategy_hint_items:
            strategy_hint = " ".join(
                [item.get("content", "") for item in strategy_hint_items if isinstance(item, dict)]
            )
        goal_text = main_text

    context_blocks = []
    if scenario:
        context_blocks.append(f"Scenario: {scenario}")
    if goal_text:
        context_blocks.append(f"{agent_name}'s social goal: {goal_text}")
    if extra_info:
        context_blocks.append(f"{extra_info}")
    if clarification_hint:
        context_blocks.append(f"Clarification hint: {clarification_hint}")
    if strategy_hint:
        context_blocks.append(f"{strategy_hint}")
    context_section = "\n".join(context_blocks).strip()

    prompt = new_system_prompt.format(
        agent_name=agent_name,
        context_section=context_section or "(No additional context)",
        agent_background=agent_background or "(No profile provided)",
        partner_background=partner_background or "(No partner profile provided)",
        stop_token=stop_token,
    )
    return prompt.strip()

    
def call_api_model(
    messages: List[Dict[str, str]],
    model: str,
    temperature: float,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    enable_thinking: bool = False
) -> Optional[str]:
    """
    通过 requests 直接调用 OpenAI 兼容的 vLLM API。
    """
    # 构造请求 URL（注意：vLLM 的聊天补全端点是 /v1/chat/completions）
    url = f"{base_url.rstrip('/')}/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    chat_template_kwargs = {"enable_thinking": bool(enable_thinking)}

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        # vLLM 直连通常读取顶层 chat_template_kwargs
        "chat_template_kwargs": chat_template_kwargs,
        # 某些 OpenAI 兼容层会从 extra_body 合并参数，保留以增强兼容性
        "extra_body": {"chat_template_kwargs": chat_template_kwargs},
        # 可选参数：max_tokens, top_p, stop 等，根据需要可添加
    }

    max_retries = int(os.environ.get("LLM_API_MAX_RETRIES", "6"))
    base_delay = float(os.environ.get("LLM_API_RETRY_DELAY", "0.8"))

    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            if response.status_code == 200:
                data = response.json()
                # 根据 OpenAI 响应格式提取内容
                return data["choices"][0]["message"]["content"].strip()
            else:
                # 非 200 状态码，记录并重试
                logger.warning(
                    "API returned status %d (attempt %d/%d): %s",
                    response.status_code, attempt + 1, max_retries + 1, response.text[:200]
                )
                last_exc = Exception(f"HTTP {response.status_code}: {response.text}")
        except requests.RequestException as e:
            last_exc = e
            logger.warning(
                "Request failed (attempt %d/%d): %s",
                attempt + 1, max_retries + 1, e
            )

        # 重试前等待（指数退避 + 随机抖动）
        if attempt < max_retries:
            sleep_time = base_delay * (2 ** attempt) + random.uniform(0, 0.25)
            time.sleep(sleep_time)

    logger.error("LLM API call failed after %d retries: %s", max_retries + 1, last_exc)
    return None


# def call_api_model(
#     messages: List[Dict[str, str]],
#     model: str,
#     temperature: float,
#     api_key: Optional[str] = None,
#     base_url: Optional[str] = None,
# ) -> Optional[str]:
#     """
#     Minimal wrapper to call OpenAI Chat Completions via the new OpenAI client using messages-format.
#     `messages` should be a list of {"role": "...", "content": "..."} entries.
#     """
#     resolved_api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
#     if not resolved_api_key:
#         print("OPENAI_API_KEY not set in environment. Aborting model call.")
#         return None

#     resolved_base_url = (
#         base_url if base_url is not None else os.environ.get("OPENAI_BASE_URL")
#     )
#     ## debug
#     # resolved_api_key = "EMPTY"
#     # resolved_base_url = "http://127.0.0.1:8000/v1"
#     # model = "Qwen3.5-27B"
#     client = OpenAI(api_key=resolved_api_key, base_url=resolved_base_url)
#     max_retries = int(os.environ.get("LLM_API_MAX_RETRIES", "6"))
#     base_delay = float(os.environ.get("LLM_API_RETRY_DELAY", "0.8"))
#     last_err: Optional[Exception] = None
#     for attempt in range(max_retries + 1):
#         try:
#             resp = client.chat.completions.create(
#                 model=model, messages=messages, temperature=temperature, extra_body={"chat_template_kwargs": {"enable_thinking": False}}
#             )
#             # New-style response structure
#             return resp.choices[0].message.content.strip()
#         except Exception as e:
#             last_err = e
#             # Exponential backoff with jitter
#             sleep_s = base_delay * (2**attempt) + random.uniform(0, 0.25)
#             logger.warning(
#                 "LLM API call failed (attempt %d/%d): %s; retrying in %.2fs",
#                 attempt + 1,
#                 max_retries + 1,
#                 e,
#                 sleep_s,
#             )
#             time.sleep(sleep_s)
#             continue
#     logger.error("LLM API call failed after retries: %s", last_err)
#     return None


def call_local_small_model(
    prompt: str,
    model_dict: Dict[str, Any],
    temperature: float,
    num_responses: int = 1,
    max_new_tokens=256,
) -> Optional[Union[str, List[str]]]:
    """
    Very small-model adapter:
    - If `small_model_cmd` is provided, run it as a subprocess and pass the last user message on stdin.
      The command should read from stdin and print the reply to stdout.
    - Otherwise, fall back to a deterministic lightweight responder that returns a short in-character reply.

    Added parameter:
    - num_responses: number of distinct responses to generate (default 1). If >1, returns a list of strings.
    """
    qwen_model = model_dict.get("model")
    qwen_tokenizer = model_dict.get("tokenizer")
    if qwen_model is not None and qwen_tokenizer is not None:
        # tokenize and generate
        inputs = qwen_tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.cuda() for k, v in inputs.items()}
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": True if temperature and temperature > 0 else False,
            "temperature": float(temperature),
            "top_k": 50,
            "num_return_sequences": max(1, int(num_responses)),
        }
        generated = qwen_model.generate(**inputs, **gen_kwargs)

        # Decode the prompt without special tokens for safe prefix comparison
        try:
            prompt_ids = qwen_tokenizer(prompt)["input_ids"]
            prompt_clean = qwen_tokenizer.decode(prompt_ids, skip_special_tokens=True)
        except Exception:
            prompt_clean = prompt

        # When multiple responses are requested, decode each generated sequence separately.
        if int(num_responses) > 1:
            responses: List[str] = []
            for seq in generated:
                gen_text = qwen_tokenizer.decode(seq, skip_special_tokens=True).strip()
                if prompt_clean and gen_text.startswith(prompt_clean):
                    responses.append(gen_text[len(prompt_clean) :].strip())
                else:
                    responses.append(gen_text)
            return responses

        # Single response (backwards compatible)
        generated_text = qwen_tokenizer.decode(generated[0], skip_special_tokens=True)
        # Return only the newly generated portion if possible (compare using cleaned prompt)
        if prompt_clean and generated_text.startswith(prompt_clean):
            return generated_text[len(prompt_clean) :].strip()
        return generated_text.strip()


@dataclass
class ResponseCandidate:
    text: str
    index: int


def format_conversation_to_prompt(
    conversation: List[Dict[str, str]], model_type: str = "qwen"
) -> str:
    """
    将对话历史转换为适合本地大模型的提示词格式

    Args:
        conversation: 对话历史列表，每个元素包含"role"和"content"
        model_type: 模型类型，可选 "llama", "chatglm", "qwen", "vicuna"等

    Returns:
        格式化后的提示词字符串
    """
    formatted_prompt = ""

    if model_type == "llama":
        # LLaMA2/3 Chat格式
        system_content = ""
        messages = []

        for turn in conversation:
            if turn["role"] == "system":
                system_content = turn["content"]
            else:
                messages.append(
                    f"<|start_header_id|>{turn['role']}<|end_header_id|>\n\n{turn['content']}<|eot_id|>"
                )

        if system_content:
            formatted_prompt = (
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_content}<|eot_id|>"
                + "".join(messages)
            )
        else:
            formatted_prompt = "<|begin_of_text|>" + "".join(messages)

        formatted_prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"

    elif model_type == "chatglm":
        # ChatGLM格式
        for i, turn in enumerate(conversation):
            if turn["role"] == "system":
                formatted_prompt += f"[Round {i}]\n\n问：{turn['content']}\n\n答："
            elif turn["role"] == "user":
                formatted_prompt += f"[Round {i}]\n\n问：{turn['content']}\n\n答："
            elif turn["role"] == "assistant":
                formatted_prompt += f"{turn['content']}\n\n"
        formatted_prompt += f"[Round {len(conversation)}]\n\n问："

    elif model_type == "qwen":
        # Qwen-Chat格式
        formatted_prompt = "<|im_start|>system\n"
        for turn in conversation:
            if turn["role"] == "system":
                formatted_prompt += turn["content"] + "<|im_end|>\n"

        for turn in conversation:
            if turn["role"] != "system":
                formatted_prompt += (
                    f"<|im_start|>{turn['role']}\n{turn['content']}<|im_end|>\n"
                )

        formatted_prompt += "<|im_start|>assistant\n"

    elif model_type == "vicuna":
        # Vicuna格式（v1.1）
        system_content = ""
        for turn in conversation:
            if turn["role"] == "system":
                system_content = turn["content"]

        formatted_prompt = f"{system_content}\n\n"

        for turn in conversation:
            if turn["role"] == "user":
                formatted_prompt += f"USER: {turn['content']}\n"
            elif turn["role"] == "assistant":
                formatted_prompt += f"ASSISTANT: {turn['content']}</s>\n"

        formatted_prompt += "ASSISTANT: "

    elif model_type == "simple":
        # 简单通用格式
        for turn in conversation:
            if turn["role"] == "system":
                formatted_prompt += f"System: {turn['content']}\n\n"
            elif turn["role"] == "user":
                formatted_prompt += f"Human: {turn['content']}\n\n"
            elif turn["role"] == "assistant":
                formatted_prompt += f"Assistant: {turn['content']}\n\n"
        formatted_prompt += "Assistant: "

    else:
        # 默认格式：只拼接用户和助手消息
        for turn in conversation:
            if turn["role"] == "user":
                formatted_prompt += f"User: {turn['content']}\n"
            elif turn["role"] == "assistant":
                formatted_prompt += f"Assistant: {turn['content']}\n"
        formatted_prompt += "Assistant: "

    return formatted_prompt


def append_histories_to_file(histories_path: str, batch_state) -> None:
    try:
        os.makedirs(os.path.dirname(histories_path) or ".", exist_ok=True)
    except Exception:
        pass
    records = []
    for i in range(len(batch_state.history_messages_all)):
        scene = None
        if hasattr(batch_state, "scenes") and batch_state.scenes and i < len(batch_state.scenes):
            scene = batch_state.scenes[i]
        record = {
            "batch_idx": batch_state.batch_idx,
            "round_idx": batch_state.round_idx,
            "scene_id": _extract_scene_id(scene),
            "swap_agents": (
                bool(scene.get("_swap_agents", False))
                if isinstance(scene, dict)
                else False
            ),
            "agent0": batch_state.agent0_list[i],
            "agent1": batch_state.agent1_list[i],
            "history": batch_state.history_messages_all[i],
        }
        records.append(record)
    with open(histories_path, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def append_histories_with_rewards(
    histories_path: str,
    all_groups: List[tuple],
    goal_scores_per_group: List[float],
    goal_reasons_per_group: List[str],
) -> None:
    try:
        os.makedirs(os.path.dirname(histories_path) or ".", exist_ok=True)
    except Exception:
        pass
    with open(histories_path, "a", encoding="utf-8") as f:
        for idx, item in enumerate(all_groups):
            try:
                batch_idx, b, g, group, _scene, _small_agent_name = item
                histories_all = group.get("histories_all") or {}
                agent_names = list(histories_all.keys())
                agent0 = agent_names[0] if len(agent_names) > 0 else ""
                agent1 = agent_names[1] if len(agent_names) > 1 else ""
                score = float(goal_scores_per_group[idx]) if idx < len(goal_scores_per_group) else None
                reason = goal_reasons_per_group[idx] if idx < len(goal_reasons_per_group) else ""
                rec = {
                    "batch_idx": batch_idx,
                    "scene_idx": b,
                    "scene_id": _extract_scene_id(_scene),
                    "swap_agents": (
                        bool(_scene.get("_swap_agents", False))
                        if isinstance(_scene, dict)
                        else False
                    ),
                    "group_idx": g,
                    "agent0": agent0,
                    "agent1": agent1,
                    "history": histories_all,
                    "reward": {
                        "goal_score": score,
                        "goal_reason": reason,
                    },
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            except Exception:
                continue


def _extract_scene_id(scene: Optional[dict]) -> Optional[str]:
    if not isinstance(scene, dict):
        return None
    for key in ("scene_id", "episode_id", "id"):
        value = scene.get(key)
        if value is not None and str(value).strip():
            return str(value)
    meta = scene.get("meta")
    if isinstance(meta, dict):
        for key in ("scene_id", "episode_id", "id"):
            value = meta.get(key)
            if value is not None and str(value).strip():
                return str(value)
    return None


def _rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def _ddp_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def _ddp_reduce_scalar(value: float, op: dist.ReduceOp) -> float:
    if not (dist.is_available() and dist.is_initialized()):
        return float(value)
    t = torch.tensor([float(value)], device=_ddp_device(), dtype=torch.float64)
    dist.all_reduce(t, op=op)
    return float(t.item())


def _ddp_aggregate_stats(values: List[float]) -> tuple[float, float, float, float, int]:
    arr = np.array(values, dtype=float) if values else np.array([])
    local_count = int(arr.size)
    local_sum = float(arr.sum()) if local_count else 0.0
    local_sumsq = float((arr * arr).sum()) if local_count else 0.0
    local_min = float(arr.min()) if local_count else float("inf")
    local_max = float(arr.max()) if local_count else float("-inf")

    if dist.is_available() and dist.is_initialized():
        device = _ddp_device()
        sum_t = torch.tensor(
            [float(local_count), local_sum, local_sumsq],
            device=device,
            dtype=torch.float64,
        )
        min_t = torch.tensor([local_min], device=device, dtype=torch.float64)
        max_t = torch.tensor([local_max], device=device, dtype=torch.float64)
        dist.all_reduce(sum_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(min_t, op=dist.ReduceOp.MIN)
        dist.all_reduce(max_t, op=dist.ReduceOp.MAX)
        global_count = int(sum_t[0].item())
        if global_count <= 0:
            return 0.0, 0.0, 0.0, 0.0, 0
        mean = float(sum_t[1].item() / global_count)
        var = max(float(sum_t[2].item() / global_count) - mean * mean, 0.0)
        std = float(np.sqrt(var))
        return mean, std, float(min_t.item()), float(max_t.item()), global_count

    if local_count <= 0:
        return 0.0, 0.0, 0.0, 0.0, 0
    mean = local_sum / local_count
    var = max(local_sumsq / local_count - mean * mean, 0.0)
    return (
        float(mean),
        float(np.sqrt(var)),
        float(local_min),
        float(local_max),
        local_count,
    )


def _append_local_metrics(path: Optional[str], payload: Dict[str, object]) -> None:
    if not path or _rank() != 0:
        return
    try:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")
    except Exception as e:
        logger.warning(
            "[rank %d] Failed to append local metrics to %s: %s",
            _rank(),
            path,
            e,
        )


def clone_history(history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    return [dict(m) for m in (history or [])]


def build_tasks(
    scene_groups: List[List[dict]],
    agent0_list: List[str],
    agent1_list: List[str],
    round_idx: int,
    is_large_turn: bool,
    active_mask: Optional[List[bool]] = None,
    group_active_mask: Optional[List[List[bool]]] = None,
) -> List[tuple]:
    tasks = []
    for b in range(len(scene_groups)):
        if active_mask is not None and not active_mask[b]:
            continue
        speaker = agent0_list[b] if round_idx % 2 == 0 else agent1_list[b]
        for g in range(len(scene_groups[b])):
            if (
                group_active_mask is not None
                and b < len(group_active_mask)
                and g < len(group_active_mask[b])
                and not group_active_mask[b][g]
            ):
                continue
            group = scene_groups[b][g]
            history_for_speaker = group["histories_all"][speaker]
            prompt = format_conversation_to_prompt(history_for_speaker)
            if not is_large_turn:
                group["turn_histories"].append(clone_history(history_for_speaker))
                group["prompts"].append(prompt)
            tasks.append((b, g, speaker, history_for_speaker, prompt))
    return tasks


def apply_reply(
    scene_groups: List[List[dict]],
    is_large_turn: bool,
    b: int,
    g: int,
    speaker: str,
    reply: str,
) -> None:
    group = scene_groups[b][g]
    # Snapshot history BEFORE appending the new reply, so turn_histories[i]
    # captures the context that led to responses[i].
    if not is_large_turn:
        snapshot = [dict(m) for m in group["histories_all"][speaker]]
        group["turn_histories"].append(snapshot)
    for participant, history in group["histories_all"].items():
        role = "assistant" if participant == speaker else "user"
        history.append({"role": role, "content": reply})
    group["last_reply"] = reply
    group["all_replies"].append(reply)
    group["all_speakers"].append(speaker)
    if not is_large_turn:
        group["responses"].append(reply)
        group["all_small_indices"].append(len(group["responses"]) - 1)
    else:
        group["all_small_indices"].append(None)

def call_api_for_task(
    item: tuple,
    partner_model: str,
    partner_base_url: str,
    partner_api_key: str,
) -> tuple:
    b, g, _speaker, hist, _prompt = item
    r = call_api_model(
        hist,
        model=partner_model,
        temperature=0.7,
        api_key=partner_api_key,
        base_url=partner_base_url,
    )
    return (b, g, r)


def run_dialogue_round(
    policy: object,
    batch_state: object,
    rollout: dict,
    max_new_tokens: int,
    active_mask: Optional[List[bool]] = None,
    group_active_mask: Optional[List[List[bool]]] = None,
    partner_base_url: str = "http://127.0.0.1:8000/v1",
    partner_model: str = "Qwen3.5-35B-A3B",
    partner_api_key: str = "EMPTY",
) -> tuple:
    scene_groups = rollout["scene_groups"]
    agent0_list = batch_state.agent0_list
    agent1_list = batch_state.agent1_list
    is_large_turn = batch_state.is_large
    round_idx = batch_state.round_idx
    speaker_by_b = [
        agent0_list[b] if round_idx % 2 == 0 else agent1_list[b]
        for b in range(len(agent0_list))
    ]

    tasks = build_tasks(
        scene_groups,
        agent0_list,
        agent1_list,
        round_idx,
        is_large_turn,
        active_mask=active_mask,
        group_active_mask=group_active_mask,
    )

    reply_texts = [None] * len(batch_state.prompts)
    has_none_reply = False
    stop_flags = [[False] * len(scene_groups[b]) for b in range(len(scene_groups))]
    scene_replied = [False] * len(batch_state.prompts)

    if is_large_turn:
        workers = max(1, min(8, len(tasks)))
        call_fn = partial(
            call_api_for_task,
            partner_model=partner_model,
            partner_base_url=partner_base_url,
            partner_api_key=partner_api_key,
        )
        with ThreadPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(call_fn, tasks))
        for b, g, reply in results:
            if reply is None:
                has_none_reply = True
                continue
            speaker = speaker_by_b[b]
            apply_reply(scene_groups, is_large_turn, b, g, speaker, reply)
            if stop_token and (stop_token in reply):
                stop_flags[b][g] = True
            if not scene_replied[b]:
                reply_texts[b] = reply
                scene_replied[b] = True
    else:
        prompts_only = [prompt for (_b, _g, _speaker, _hist, prompt) in tasks]
        policy_impl = policy.module if hasattr(policy, "module") else policy
        responses_batch = policy_impl.generate_candidates_batch(
            prompts_only,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
        )
        for item, responses in zip(tasks, responses_batch):
            b, g, speaker, _hist, _prompt = item
            reply = responses[0].text if responses else None
            if reply is None:
                has_none_reply = True
                continue
            apply_reply(scene_groups, is_large_turn, b, g, speaker, reply)
            if stop_token and (stop_token in reply):
                stop_flags[b][g] = True
            if not scene_replied[b]:
                reply_texts[b] = reply
                scene_replied[b] = True

    return reply_texts, has_none_reply, stop_flags


def score_goal_item(
    item: tuple,
    llm_model: str,
    llm_base_url: Optional[str],
    llm_api_key: Optional[str],
) -> tuple:
    from rl_reward import evaluate_social_goal_achievement_with_reason_continuous

    group_idx, scene, history, agent_name = item
    score, reason = evaluate_social_goal_achievement_with_reason_continuous(
        scene,
        history,
        agent_name,
        ranker_model=llm_model,
        temperature=0.7,
        base_url=llm_base_url,
        api_key=llm_api_key,
    )
    score = float(score)
    if score > 1.0:
        score = score / 100.0
    score = max(0.0, min(1.0, score))
    return (group_idx, score, str(reason or ""))


def _find_partner_reply_for_small_step(group: dict, step_idx: int) -> str:
    """Return the large-model reply that immediately followed small step_idx."""
    all_replies = group.get("all_replies") or []
    all_small_indices = group.get("all_small_indices") or []
    # Find the position of this small step in the full reply sequence.
    small_count = -1
    for pos, small_idx in enumerate(all_small_indices):
        if small_idx is not None:
            small_count += 1
            if small_count == step_idx:
                # The next reply after position `pos` that belongs to the large model.
                for nxt in range(pos + 1, len(all_small_indices)):
                    if all_small_indices[nxt] is None and nxt < len(all_replies):
                        return str(all_replies[nxt] or "")
                break
    return ""


def score_process_reward_item(
    item: tuple,
    llm_model: str,
    llm_base_url: Optional[str],
    llm_api_key: Optional[str],
) -> tuple:
    """Score one (group, step) using the dynamic rubric process reward.

    Returns (group_idx, step_idx, score, reason, extra).
    """
    group_idx, step_idx, scene, history, agent_name, small_response = item

    from rl_reward import evaluate_rubric_dynamic_with_reason
    from dynamic_reward_config import (
        aggregate_rubric_weighted_score,
        aggregate_with_explicit_weights,
        compute_rubric_weights,
        get_phase,
        mix_rubric_with_baseline,
        render_rubric_baseline_for_prompt,
        resolve_actor_role,
        ROLE_TO_FAMILY,
    )

    try:
        expected_horizon = int(os.environ.get("EXPECTED_HORIZON", "5"))
    except Exception:
        expected_horizon = 5
    try:
        use_llm_weights = bool(int(os.environ.get("DYNAMIC_WEIGHTS_FROM_JUDGE", "0")))
    except Exception:
        use_llm_weights = False
    try:
        mix_ratio = float(os.environ.get("DYNAMIC_WEIGHTS_MIX_RATIO", "0.5"))
    except Exception:
        mix_ratio = 0.5

    actor_role = resolve_actor_role(scene, agent_name)
    baseline_text = render_rubric_baseline_for_prompt(
        actor_role, int(step_idx), expected_horizon
    )

    dim_scores, rubrics, reason, llm_weights, weight_reasoning = (
        evaluate_rubric_dynamic_with_reason(
            scene,
            history,
            agent_name,
            small_response,
            ranker_model=llm_model,
            temperature=0.7,
            base_url=llm_base_url,
            api_key=llm_api_key,
            baseline_weights_text=baseline_text,
        )
    )

    used_llm_flag = False
    baseline_w = compute_rubric_weights(actor_role, int(step_idx), expected_horizon)
    if use_llm_weights and llm_weights:
        final_w, used_llm_flag = mix_rubric_with_baseline(
            llm_weights, actor_role, int(step_idx), expected_horizon, mix_ratio
        )
        weighted, used_w = aggregate_with_explicit_weights(dim_scores, final_w)
        family = ROLE_TO_FAMILY.get(actor_role, "cooperative")
        phase = get_phase(int(step_idx), expected_horizon)
    else:
        weighted, used_w, family, phase = aggregate_rubric_weighted_score(
            dim_scores, actor_role, int(step_idx), expected_horizon
        )

    extra = {
        "reward_kind": "rubric_dynamic",
        "actor_role": actor_role,
        "family": family,
        "phase": phase,
        "dim_scores": {k: float(v) for k, v in dim_scores.items()},
        "rubrics": rubrics,
        "baseline_weights": {k: float(v) for k, v in baseline_w.items()},
        "weights": {k: float(v) for k, v in used_w.items()},
        "weighted_score": float(weighted),
        "expected_horizon": int(expected_horizon),
        "used_llm_weights": bool(used_llm_flag),
        "mix_ratio": float(mix_ratio) if use_llm_weights else None,
    }
    if llm_weights:
        extra["llm_weights_raw"] = {k: float(v) for k, v in llm_weights.items()}
    if weight_reasoning:
        extra["weight_reasoning"] = weight_reasoning

    return (int(group_idx), int(step_idx), float(weighted), str(reason or ""), extra)


def collect_rewards(
    all_groups: List[tuple],
    llm_model: str,
    llm_base_url: Optional[str] = None,
    llm_api_key: Optional[str] = None,
) -> tuple:
    """Compute terminal goal reward + per-turn process rewards for all groups.

    After this call each group dict will have:
      group["process_rewards"]  — List[float] of length = number of small-model turns
      group["rubric_dynamic_per_turn"] — List[Optional[dict]], one entry per small turn
    """
    goal_scores_per_group: List[float] = [0.0 for _ in range(len(all_groups))]
    goal_reasons_per_group: List[str] = ["" for _ in range(len(all_groups))]

    # ── Terminal reward (end-of-dialogue goal achievement) ──────────────────
    goal_tasks = []
    for group_idx, (_batch_idx, _b, _g, group, scene, agent_name) in enumerate(all_groups):
        history = group["histories_all"][agent_name]
        goal_tasks.append((group_idx, scene, history, agent_name))
    if goal_tasks:
        score_fn = partial(
            score_goal_item,
            llm_model=llm_model,
            llm_base_url=llm_base_url,
            llm_api_key=llm_api_key,
        )
        with ThreadPoolExecutor(max_workers=max(1, min(8, len(goal_tasks)))) as ex:
            goal_results = list(ex.map(score_fn, goal_tasks))
        for group_idx, score, reason in goal_results:
            goal_scores_per_group[group_idx] = float(score)
            goal_reasons_per_group[group_idx] = str(reason or "")

    # ── Per-turn process reward (dynamic rubric) ────────────────────────────
    group_step_horizons: List[int] = [0] * len(all_groups)
    progress_tasks = []
    for group_idx, (_batch_idx, _b, _g, group, scene, agent_name) in enumerate(all_groups):
        turn_histories = list(group.get("turn_histories") or [])
        responses = list(group.get("responses") or [])
        horizon = min(len(turn_histories), len(responses))
        group_step_horizons[group_idx] = horizon
        for step_idx in range(horizon):
            progress_tasks.append(
                (
                    group_idx,
                    step_idx,
                    scene,
                    turn_histories[step_idx],
                    agent_name,
                    responses[step_idx],
                )
            )

    process_scores: List[List[float]] = [[0.0] * h for h in group_step_horizons]
    process_extras: List[List[Optional[dict]]] = [[None] * h for h in group_step_horizons]

    if progress_tasks:
        pr_fn = partial(
            score_process_reward_item,
            llm_model=llm_model,
            llm_base_url=llm_base_url,
            llm_api_key=llm_api_key,
        )
        with ThreadPoolExecutor(max_workers=max(1, min(8, len(progress_tasks)))) as ex:
            pr_results = list(ex.map(pr_fn, progress_tasks))
        for group_idx, step_idx, score, _reason, extra in pr_results:
            if 0 <= group_idx < len(process_scores) and 0 <= step_idx < len(process_scores[group_idx]):
                process_scores[group_idx][step_idx] = float(max(0.0, min(1.0, score)))
                process_extras[group_idx][step_idx] = extra

    # Write results back into each group dict.
    for group_idx, (_batch_idx, _b, _g, group, _scene, _agent_name) in enumerate(all_groups):
        group["process_rewards"] = process_scores[group_idx]
        if any(process_extras[group_idx]):
            group["rubric_dynamic_per_turn"] = process_extras[group_idx]

    return (
        goal_scores_per_group,
        goal_reasons_per_group,
    )


def score_goal_item_test(
    item: tuple,
    llm_model: str,
    llm_base_url: Optional[str],
    llm_api_key: Optional[str],
) -> tuple:
    # Test-time scorer uses dedicated binary goal-achievement prompt.
    from rl_reward import evaluate_social_goal_achievement_with_reason_binary

    group_idx, scene, history, agent_name = item
    score, reason = evaluate_social_goal_achievement_with_reason_binary(
        scene,
        history,
        agent_name,
        ranker_model=llm_model,
        temperature=0.7,
        base_url=llm_base_url,
        api_key=llm_api_key,
    )
    score = float(score)
    # Normalize any accidental 0-100 output into [0, 1].
    if score > 1.0:
        score = score / 100.0
    score = max(0.0, min(1.0, score))
    return (group_idx, score, str(reason or ""))


def collect_test_rewards(
    all_groups: List[tuple],
    llm_model: str,
    llm_base_url: Optional[str] = None,
    llm_api_key: Optional[str] = None,
) -> tuple:
    goal_scores_per_group: List[float] = [0.0 for _ in range(len(all_groups))]
    goal_reasons_per_group: List[str] = ["" for _ in range(len(all_groups))]

    goal_tasks = []
    for group_idx, (_batch_idx, _b, _g, group, scene, agent_name) in enumerate(
        all_groups
    ):
        history = group["histories_all"][agent_name]
        goal_tasks.append((group_idx, scene, history, agent_name))
    if goal_tasks:
        score_fn = partial(
            score_goal_item_test,
            llm_model=llm_model,
            llm_base_url=llm_base_url,
            llm_api_key=llm_api_key,
        )
        with ThreadPoolExecutor(max_workers=max(1, min(8, len(goal_tasks)))) as ex:
            goal_results = list(ex.map(score_fn, goal_tasks))
        for group_idx, score, reason in goal_results:
            goal_scores_per_group[group_idx] = float(score)
            goal_reasons_per_group[group_idx] = str(reason or "")

    return (
        goal_scores_per_group,
        goal_reasons_per_group,
    )
