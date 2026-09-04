#!/usr/bin/env bash
# Step 3 of 4: start the policy model vLLM server (with LoRA support).
#
# Run after run_judge_server.sh and run_partner_server.sh.
# This script blocks (foreground). Run in a separate terminal.
#
# Usage:
#   bash scripts-ppo-third-dynamic-reward-rubric/run_policy_server.sh
#
# Override via env, e.g.:
#   POLICY_MODEL_PATH=./model/Qwen/Qwen2.5-7B-Instruct \
#   PEFT_CHECKPOINT_PATH=./checkpoints/20260626_153150/checkpoints_epoch6 \
#     bash scripts-ppo-third-dynamic-reward-rubric/run_policy_server.sh

set -euo pipefail

# ==========================================================================
# Configuration
# ==========================================================================

POLICY_MODEL_PATH="${POLICY_MODEL_PATH:-/fudan_university_cfs/wjn/sotopia-dia/model/Qwen/Qwen2.5-7B-Instruct}"
# POLICY_MODEL_PATH="${POLICY_MODEL_PATH:-/fudan_university_cos/models/LLM-Research/gemma-3-4b-it}"
# POLICY_MODEL_PATH="${POLICY_MODEL_PATH:-/fudan_university_cos/models/Qwen/Qwen3-8B}"
# POLICY_MODEL_PATH="${POLICY_MODEL_PATH:-/fudan_university_cos/models/meta-llama/Meta-Llama-3.1-8B-Instruct}"
POLICY_SERVED_MODEL_NAME="${POLICY_SERVED_MODEL_NAME:-Qwen2.5-7B-Instruct}"
PEFT_CHECKPOINT_PATH="${PEFT_CHECKPOINT_PATH:-/fudan_university_cfs/wjn/sotopia-dia/save-ckpt/archer}"
POLICY_HOST="${POLICY_HOST:-127.0.0.3}"
POLICY_PORT="${POLICY_PORT:-8000}"
POLICY_API_KEY="${POLICY_API_KEY:-EMPTY}"
POLICY_CUDA_VISIBLE_DEVICES="${POLICY_CUDA_VISIBLE_DEVICES:-4}"
POLICY_GPU_MEMORY_UTILIZATION="${POLICY_GPU_MEMORY_UTILIZATION:-0.9}"
POLICY_MAX_MODEL_LEN="${POLICY_MAX_MODEL_LEN:-8192}"
POLICY_DTYPE="${POLICY_DTYPE:-auto}"
# LoRA settings – must be >= the rank used during training (e.g. 64)
POLICY_MAX_LORA_RANK="${POLICY_MAX_LORA_RANK:-64}"
POLICY_MAX_LORAS="${POLICY_MAX_LORAS:-4}"

# ==========================================================================
# Derived
# ==========================================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

mkdir -p logs
LOG_FILE="logs/policy_server_$(date +%Y%m%d_%H%M%S).log"

DEV_STR="${POLICY_CUDA_VISIBLE_DEVICES// /}"
if [[ -z "${DEV_STR}" ]]; then
    TP_SIZE=1
else
    TP_SIZE="$(awk -F',' '{print NF}' <<< "${DEV_STR}")"
fi

# ==========================================================================
# Validation
# ==========================================================================

if [[ -z "${POLICY_MODEL_PATH}" ]]; then
    echo "[ERROR] POLICY_MODEL_PATH is required."
    exit 1
fi

if [[ ! -d "${POLICY_MODEL_PATH}" ]]; then
    echo "[ERROR] POLICY_MODEL_PATH does not exist: ${POLICY_MODEL_PATH}"
    exit 1
fi

if [[ -n "${PEFT_CHECKPOINT_PATH}" && ! -d "${PEFT_CHECKPOINT_PATH}" ]]; then
    echo "[ERROR] PEFT_CHECKPOINT_PATH does not exist: ${PEFT_CHECKPOINT_PATH}"
    exit 1
fi

# ==========================================================================
# Launch
# ==========================================================================

echo "=================================================================="
echo "  [Step 3/4] Policy model vLLM server (LoRA-enabled)"
echo "=================================================================="
echo "  POLICY_MODEL_PATH=${POLICY_MODEL_PATH}"
echo "  POLICY_SERVED_MODEL_NAME=${POLICY_SERVED_MODEL_NAME}"
echo "  PEFT_CHECKPOINT_PATH=${PEFT_CHECKPOINT_PATH:-<none, base model only>}"
echo "  POLICY_HOST=${POLICY_HOST}  POLICY_PORT=${POLICY_PORT}"
echo "  POLICY_CUDA_VISIBLE_DEVICES=${POLICY_CUDA_VISIBLE_DEVICES}  TP=${TP_SIZE}"
echo "  POLICY_GPU_MEMORY_UTILIZATION=${POLICY_GPU_MEMORY_UTILIZATION}"
echo "  POLICY_MAX_LORA_RANK=${POLICY_MAX_LORA_RANK}  POLICY_MAX_LORAS=${POLICY_MAX_LORAS}"
echo "  LOG_FILE=${LOG_FILE}"
echo "=================================================================="
echo "  Policy base URL will be: http://${POLICY_HOST}:${POLICY_PORT}/v1"
echo "  Pass to step 4:"
echo "    POLICY_BASE_URL=http://${POLICY_HOST}:${POLICY_PORT}/v1"
echo "    POLICY_MODEL=${POLICY_SERVED_MODEL_NAME}"
echo "=================================================================="

ARGS=(
    --model                  "${POLICY_MODEL_PATH}"
    --served-model-name      "${POLICY_SERVED_MODEL_NAME}"
    --host                   "${POLICY_HOST}"
    --port                   "${POLICY_PORT}"
    --api-key                "${POLICY_API_KEY}"
    --tensor-parallel-size   "${TP_SIZE}"
    --gpu-memory-utilization "${POLICY_GPU_MEMORY_UTILIZATION}"
    --max-model-len          "${POLICY_MAX_MODEL_LEN}"
    --dtype                  "${POLICY_DTYPE}"
    --trust-remote-code
    --enable-lora
    --max-lora-rank          "${POLICY_MAX_LORA_RANK}"
    --max-loras              "${POLICY_MAX_LORAS}"
)

if [[ -n "${PEFT_CHECKPOINT_PATH}" ]]; then
    # OpenAI-compatible vLLM serves LoRA as a separate model name.
    # Requests must set model=${POLICY_LORA_NAME}; per-request lora_request is ignored.
    ARGS+=(--lora-modules "${POLICY_SERVED_MODEL_NAME}=${PEFT_CHECKPOINT_PATH}")
fi

CUDA_VISIBLE_DEVICES="${POLICY_CUDA_VISIBLE_DEVICES}" \
python -m vllm.entrypoints.openai.api_server \
    "${ARGS[@]}" \
    2>&1 | tee "${LOG_FILE}"
