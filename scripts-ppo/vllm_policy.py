"""vLLM-backed policy that satisfies the same interface as QwenPolicyTorch.

Only the subset of the interface used during *testing* is implemented:
    policy.generate_candidates_batch(prompts, max_new_tokens, temperature)
        -> List[List[ResponseCandidate]]

The policy talks to any OpenAI-compatible server via /v1/chat/completions.
This works with local vLLM servers and external APIs (e.g. api.getoken.tech).
The rendered prompt string is sent as the content of a single "user" message.

LoRA adapter support
--------------------
For vLLM's OpenAI-compatible server, LoRA adapters must be registered on the
server at startup via ``--lora-modules name=path``.  The request should then use
``model=name`` to select that adapter.  The OpenAI request schema does *not*
accept a per-request ``lora_request`` field; if such a field is sent, vLLM logs
"ignored: {'lora_request'}" and the base model is used.

Therefore this client never sends ``lora_request``.  It only sends the model
name passed by the caller.  When testing a LoRA checkpoint, run_policy_server.sh
registers the adapter under ``POLICY_LORA_NAME`` and run_test_only_vllm.sh sets
``POLICY_MODEL`` to that same name.

Concurrency
-----------
generate_candidates_batch issues all prompt requests concurrently using a
thread pool (matching the concurrency model already used in utils.py for
partner calls).  Pool size is controlled by the env var
``VLLM_POLICY_MAX_WORKERS`` (default: 32).
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import requests

from utils import ResponseCandidate, logger, call_api_model

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_DEFAULT_MAX_WORKERS = 32


def _resolve_workers(task_count: int) -> int:
    raw = os.environ.get("VLLM_POLICY_MAX_WORKERS", str(_DEFAULT_MAX_WORKERS))
    try:
        cap = int(raw)
    except ValueError:
        cap = _DEFAULT_MAX_WORKERS
    return min(max(1, task_count), cap)


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


class VllmPolicy:
    """Thin HTTP client that wraps a running vLLM server as a policy.

    Parameters
    ----------
    base_url:
        Base URL of the vLLM OpenAI-compatible server, e.g.
        ``http://127.0.0.1:9000/v1``.
    model_name:
        Served model name to use in the OpenAI request.  For base-model tests,
        this is the ``--served-model-name`` passed to vLLM.  For LoRA tests,
        this must be the adapter name registered by ``--lora-modules``.
    lora_path:
        Optional adapter path kept only for logging/auditing.  It is not sent
        in the OpenAI request because vLLM ignores per-request ``lora_request``
        fields on this endpoint.
    api_key:
        Bearer token (default ``"EMPTY"`` for local servers without auth).
    request_timeout:
        Per-request HTTP timeout in seconds.
    """

    def __init__(
        self,
        base_url: str,
        model_name: str,
        lora_path: Optional[str] = None,
        api_key: str = "EMPTY",
        request_timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.lora_path = lora_path
        self.api_key = api_key
        self.request_timeout = request_timeout

        if lora_path:
            logger.info(
                "VllmPolicy: LoRA adapter expected via model='%s'; adapter path=%s",
                model_name,
                lora_path,
            )
        else:
            logger.info(
                "VllmPolicy: using served model '%s' at %s",
                model_name,
                base_url,
            )

    # ------------------------------------------------------------------
    # Core interface (must match what run_dialogue_round expects)
    # ------------------------------------------------------------------

    def generate_candidates_batch(
        self,
        messages: List[List[dict]],
        max_new_tokens: int = 128,
        temperature: float = 0.7,
    ) -> List[List[ResponseCandidate]]:
        """Generate one response per conversation, returned as List[List[ResponseCandidate]].

        Each element of *messages* is a full conversation history in
        OpenAI message-list format (same as what call_api_model receives):
            [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, ...]
        """
        if not messages:
            return []

        enable_thinking = False

        def _call_one(idx_msgs: tuple) -> tuple:
            idx, msgs = idx_msgs
            text = call_api_model(
                messages=msgs,
                model=self.model_name,
                temperature=temperature,
                api_key=self.api_key,
                base_url=self.base_url,
                enable_thinking=enable_thinking,
                max_tokens=max_new_tokens,
                top_k=50,
                top_p=0.95,
            )
            return idx, text

        workers = _resolve_workers(len(messages))
        results: List[Optional[str]] = [None] * len(messages)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for idx, text in pool.map(_call_one, enumerate(messages)):
                results[idx] = text

        per_conv: List[List[ResponseCandidate]] = []
        for text in results:
            if text is None:
                per_conv.append([])
            else:
                per_conv.append([ResponseCandidate(text=text.strip(), index=0)])
        return per_conv

    # ------------------------------------------------------------------
    # Health check utility (used by the launch script)
    # ------------------------------------------------------------------

    def wait_until_ready(self, timeout: float = 300.0, poll_interval: float = 3.0) -> bool:
        """Block until the vLLM server responds to /v1/models, or timeout.

        Returns True if the server became ready within *timeout* seconds.
        """
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
                        "VllmPolicy: server ready after %d probe(s) – %s",
                        attempt,
                        self.base_url,
                    )
                    return True
            except Exception:
                pass
            logger.debug(
                "VllmPolicy: waiting for server at %s (attempt %d)…",
                self.base_url,
                attempt,
            )
            time.sleep(poll_interval)
        logger.error(
            "VllmPolicy: server at %s did not become ready within %.0fs",
            self.base_url,
            timeout,
        )
        return False
