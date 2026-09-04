"""
Policy and partner model clients for AgentSense evaluation.

Both the policy model and the partner model share the same HTTP client
(AgentModel).  The caller decides which system prompt to pass, making the
same client reusable for any character in a multi-person scene.

Concurrency
-----------
Single-turn generation is synchronous (one HTTP call).  Batch generation
over multiple run items uses a thread pool and is exposed via
generate_batch(), which is used by the runner for parallel rollouts.
"""
from __future__ import annotations

import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import requests

logger = logging.getLogger("agentsense_model")

_DEFAULT_MAX_WORKERS = int(os.environ.get("AGENTSENSE_MODEL_MAX_WORKERS", "32"))

# Model name substrings whose chat templates reject the "system" role.
# The system prompt is folded into the first user message instead.
_NO_SYSTEM_ROLE_MODELS = ("gemma",)


def _resolve_workers(task_count: int) -> int:
    cap = max(1, _DEFAULT_MAX_WORKERS)
    return min(max(1, task_count), cap)


def _needs_no_system_role(model_name: str) -> bool:
    name_lower = model_name.lower()
    return any(pat in name_lower for pat in _NO_SYSTEM_ROLE_MODELS)


def _fold_system_into_user(messages: list) -> list:
    """Merge a leading system turn into the first user turn.

    Gemma3 (and similar models) only accept user/assistant alternation with
    no system role.  This function converts:

        [{"role": "system", "content": S},
         {"role": "user",   "content": U}, ...]

    into:

        [{"role": "user", "content": S + "\\n\\n" + U}, ...]

    If there is no system message the list is returned unchanged.
    If the list contains only a system message (no user turn yet), the system
    content is rewritten as a user turn so the sequence is still valid.
    """
    if not messages or messages[0].get("role") != "system":
        return messages

    system_content = messages[0]["content"]
    rest = messages[1:]

    if rest and rest[0].get("role") == "user":
        merged_first = {
            "role": "user",
            "content": system_content + "\n\n" + rest[0]["content"],
        }
        return [merged_first] + rest[1:]

    # No user turn follows the system turn — replace system with user directly.
    return [{"role": "user", "content": system_content}] + rest


# ---------------------------------------------------------------------------
# Low-level HTTP call (no shared dependency on scripts-ppo utils)
# ---------------------------------------------------------------------------

def _call_api(
    messages: list,
    model: str,
    base_url: str,
    api_key: str,
    temperature: float = 0.7,
    max_tokens: int = 128,
) -> Optional[str]:
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }

    max_retries = int(os.environ.get("AGENTSENSE_API_MAX_RETRIES", "6"))
    base_delay = float(os.environ.get("AGENTSENSE_API_RETRY_DELAY", "0.8"))
    last_exc: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            if resp.status_code == 200:
                data = resp.json()
                if "choices" in data:
                    return data["choices"][0]["message"]["content"].strip()
                last_exc = Exception(f"No 'choices' in response: {data}")
            else:
                last_exc = Exception(f"HTTP {resp.status_code}: {resp.text[:200]}")
        except requests.RequestException as exc:
            last_exc = exc

        if attempt < max_retries:
            sleep_s = base_delay * (2 ** attempt) + random.uniform(0, 0.25)
            logger.warning(
                "API call failed (attempt %d/%d): %s; retrying in %.2fs",
                attempt + 1, max_retries + 1, last_exc, sleep_s,
            )
            time.sleep(sleep_s)

    logger.error("API call failed after %d retries: %s", max_retries + 1, last_exc)
    return None


# ---------------------------------------------------------------------------
# AgentModel: shared client for policy and partner
# ---------------------------------------------------------------------------

class AgentModel:
    """HTTP client wrapping a vLLM OpenAI-compatible server.

    Used for both the policy model and the partner model.  The caller
    is responsible for building the correct messages list (system prompt +
    chat history) before calling generate().

    Parameters
    ----------
    base_url : str
        Base URL of the vLLM server, e.g. ``http://127.0.0.1:9000/v1``.
    model_name : str
        Model name as registered on the server (--served-model-name).
    api_key : str
        Bearer token (default "EMPTY" for local servers).
    temperature : float
        Sampling temperature.
    max_tokens : int
        Max tokens per reply.
    """

    def __init__(
        self,
        base_url: str,
        model_name: str,
        api_key: str = "EMPTY",
        temperature: float = 0.7,
        max_tokens: int = 128,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._no_system_role = _needs_no_system_role(model_name)
        if self._no_system_role:
            logger.info(
                "AgentModel: model='%s' — system role not supported, "
                "will fold system prompt into first user turn.",
                model_name,
            )
        logger.info(
            "AgentModel: model='%s' at %s  temperature=%.2f  max_tokens=%d",
            model_name, base_url, temperature, max_tokens,
        )

    def generate(self, messages: list) -> Optional[str]:
        """Generate one reply given a full messages list."""
        if self._no_system_role:
            messages = _fold_system_into_user(messages)
        return _call_api(
            messages,
            model=self.model_name,
            base_url=self.base_url,
            api_key=self.api_key,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

    def generate_batch(
        self,
        messages_list: List[list],
    ) -> List[Optional[str]]:
        """Generate replies for a batch of conversations concurrently."""
        if not messages_list:
            return []

        results: List[Optional[str]] = [None] * len(messages_list)

        def _call_one(idx_msgs: tuple) -> tuple:
            idx, msgs = idx_msgs
            return idx, self.generate(msgs)

        workers = _resolve_workers(len(messages_list))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for idx, text in pool.map(_call_one, enumerate(messages_list)):
                results[idx] = text
        return results

    def wait_until_ready(self, timeout: float = 300.0, poll_interval: float = 3.0) -> bool:
        """Block until the vLLM server is responding, or timeout."""
        models_url = f"{self.base_url}/models"
        deadline = time.monotonic() + timeout
        attempt = 0
        while time.monotonic() < deadline:
            attempt += 1
            try:
                resp = requests.get(
                    models_url,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=10,
                )
                if resp.status_code == 200:
                    logger.info(
                        "AgentModel '%s': server ready after %d probe(s)",
                        self.model_name, attempt,
                    )
                    return True
            except Exception:
                pass
            logger.debug(
                "AgentModel '%s': waiting for server (attempt %d)…",
                self.model_name, attempt,
            )
            time.sleep(poll_interval)
        logger.error(
            "AgentModel '%s': server at %s did not become ready within %.0fs",
            self.model_name, self.base_url, timeout,
        )
        return False
