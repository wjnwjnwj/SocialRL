#!/usr/bin/env bash
# Step 2 of 4: start the partner model vLLM server.
#
# Run after run_judge_server.sh, before run_policy_server.sh.
# This script blocks (foreground). Run in a separate terminal.
#
# Usage:
#   bash scripts-ppo-third-dynamic-reward-rubric/run_partner_server.sh
#
# Override via env, e.g.:
#   PARTNER_MODEL_PATH=/path/to/model PARTNER_PORT=8000 \
#     bash scripts-ppo-third-dynamic-reward-rubric/run_partner_server.sh

set -euo pipefail

# ==========================================================================
# Configuration
# ==========================================================================

PARTNER_MODEL_PATH="${PARTNER_MODEL_PATH:-/fudan_university_cfs/wjn/sotopia-dia/model/Qwen/Qwen2.5-7B-Instruct}"
# PARTNER_MODEL_PATH="${PARTNER_MODEL_PATH:-/fudan_university_cos/models/Qwen/Qwen3-8B}"
# PARTNER_MODEL_PATH="${PARTNER_MODEL_PATH:-/fudan_university_cos/models/meta-llama/Meta-Llama-3.1-8B-Instruct}"

PARTNER_SERVED_MODEL_NAME="${PARTNER_SERVED_MODEL_NAME:-Qwen3.5-35B-A3B}"
PARTNER_HOST="${PARTNER_HOST:-127.0.0.2}"
PARTNER_PORT="${PARTNER_PORT:-8000}"
PARTNER_API_KEY="${PARTNER_API_KEY:-EMPTY}"
PARTNER_CUDA_VISIBLE_DEVICES="${PARTNER_CUDA_VISIBLE_DEVICES:-5}"
PARTNER_GPU_MEMORY_UTILIZATION="${PARTNER_GPU_MEMORY_UTILIZATION:-0.9}"
PARTNER_MAX_MODEL_LEN="${PARTNER_MAX_MODEL_LEN:-8192}"
PARTNER_DTYPE="${PARTNER_DTYPE:-auto}"

# ==========================================================================
# Derived
# ==========================================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

mkdir -p logs
LOG_FILE="logs/partner_server_$(date +%Y%m%d_%H%M%S).log"

DEV_STR="${PARTNER_CUDA_VISIBLE_DEVICES// /}"
if [[ -z "${DEV_STR}" ]]; then
    TP_SIZE=1
else
    TP_SIZE="$(awk -F',' '{print NF}' <<< "${DEV_STR}")"
fi

# ==========================================================================
# Validation
# ==========================================================================

if [[ ! -d "${PARTNER_MODEL_PATH}" ]]; then
    echo "[ERROR] PARTNER_MODEL_PATH does not exist: ${PARTNER_MODEL_PATH}"
    exit 1
fi

# ==========================================================================
# Launch
# ==========================================================================

echo "=================================================================="
echo "  [Step 2/4] Partner model vLLM server"
echo "=================================================================="
echo "  PARTNER_MODEL_PATH=${PARTNER_MODEL_PATH}"
echo "  PARTNER_SERVED_MODEL_NAME=${PARTNER_SERVED_MODEL_NAME}"
echo "  PARTNER_HOST=${PARTNER_HOST}  PARTNER_PORT=${PARTNER_PORT}"
echo "  PARTNER_CUDA_VISIBLE_DEVICES=${PARTNER_CUDA_VISIBLE_DEVICES}  TP=${TP_SIZE}"
echo "  PARTNER_GPU_MEMORY_UTILIZATION=${PARTNER_GPU_MEMORY_UTILIZATION}"
echo "  LOG_FILE=${LOG_FILE}"
echo "=================================================================="
echo "  Partner base URL will be: http://${PARTNER_HOST}:${PARTNER_PORT}/v1"
echo "  Pass to step 4:  PARTNER_BASE_URL=http://${PARTNER_HOST}:${PARTNER_PORT}/v1"
echo "=================================================================="

CUDA_VISIBLE_DEVICES="${PARTNER_CUDA_VISIBLE_DEVICES}" \
python -m vllm.entrypoints.openai.api_server \
    --model                  "${PARTNER_MODEL_PATH}" \
    --served-model-name      "${PARTNER_SERVED_MODEL_NAME}" \
    --host                   "${PARTNER_HOST}" \
    --port                   "${PARTNER_PORT}" \
    --api-key                "${PARTNER_API_KEY}" \
    --tensor-parallel-size   "${TP_SIZE}" \
    --gpu-memory-utilization "${PARTNER_GPU_MEMORY_UTILIZATION}" \
    --max-model-len          "${PARTNER_MAX_MODEL_LEN}" \
    --dtype                  "${PARTNER_DTYPE}" \
    --trust-remote-code \
    2>&1 | tee "${LOG_FILE}"
