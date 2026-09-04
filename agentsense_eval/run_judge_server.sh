#!/usr/bin/env bash
# Step 1 of 4: start the judge model vLLM server.
#
# Run this first, then run_partner_server.sh, run_policy_server.sh,
# and finally run_test_only_vllm.sh.
#
# This script blocks (foreground) so the server stays alive.
# Run it in a separate terminal or with tmux/screen/nohup.
#
# Usage:
#   bash scripts-ppo-third-dynamic-reward-rubric/run_judge_server.sh
#
# Override any variable via env, e.g.:
#   JUDGE_MODEL_PATH=/path/to/model JUDGE_PORT=8001 \
#     bash scripts-ppo-third-dynamic-reward-rubric/run_judge_server.sh

set -euo pipefail

# ==========================================================================
# Configuration
# ==========================================================================

JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:-/fudan_university_cos/models/Qwen/Qwen3.5-35B-A3B}"
JUDGE_SERVED_MODEL_NAME="${JUDGE_SERVED_MODEL_NAME:-Qwen3.5-35B-A3B}"
JUDGE_HOST="${JUDGE_HOST:-127.0.0.8}"
JUDGE_PORT="${JUDGE_PORT:-8000}"
JUDGE_API_KEY="${JUDGE_API_KEY:-EMPTY}"
JUDGE_CUDA_VISIBLE_DEVICES="${JUDGE_CUDA_VISIBLE_DEVICES:-6,7}"
JUDGE_GPU_MEMORY_UTILIZATION="${JUDGE_GPU_MEMORY_UTILIZATION:-0.7}"
JUDGE_MAX_MODEL_LEN="${JUDGE_MAX_MODEL_LEN:-8192}"
JUDGE_DTYPE="${JUDGE_DTYPE:-auto}"

# ==========================================================================
# Derived
# ==========================================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

mkdir -p logs
LOG_FILE="logs/judge_server_$(date +%Y%m%d_%H%M%S).log"

DEV_STR="${JUDGE_CUDA_VISIBLE_DEVICES// /}"
if [[ -z "${DEV_STR}" ]]; then
    TP_SIZE=1
else
    TP_SIZE="$(awk -F',' '{print NF}' <<< "${DEV_STR}")"
fi

# ==========================================================================
# Validation
# ==========================================================================

if [[ ! -d "${JUDGE_MODEL_PATH}" ]]; then
    echo "[ERROR] JUDGE_MODEL_PATH does not exist: ${JUDGE_MODEL_PATH}"
    exit 1
fi

# ==========================================================================
# Launch
# ==========================================================================

echo "=================================================================="
echo "  [Step 1/4] Judge model vLLM server"
echo "=================================================================="
echo "  JUDGE_MODEL_PATH=${JUDGE_MODEL_PATH}"
echo "  JUDGE_SERVED_MODEL_NAME=${JUDGE_SERVED_MODEL_NAME}"
echo "  JUDGE_HOST=${JUDGE_HOST}  JUDGE_PORT=${JUDGE_PORT}"
echo "  JUDGE_CUDA_VISIBLE_DEVICES=${JUDGE_CUDA_VISIBLE_DEVICES}  TP=${TP_SIZE}"
echo "  JUDGE_GPU_MEMORY_UTILIZATION=${JUDGE_GPU_MEMORY_UTILIZATION}"
echo "  LOG_FILE=${LOG_FILE}"
echo "=================================================================="
echo "  Judge base URL will be: http://${JUDGE_HOST}:${JUDGE_PORT}/v1"
echo "  Pass to step 4:  JUDGE_BASE_URL=http://${JUDGE_HOST}:${JUDGE_PORT}/v1"
echo "=================================================================="

CUDA_VISIBLE_DEVICES="${JUDGE_CUDA_VISIBLE_DEVICES}" \
python -m vllm.entrypoints.openai.api_server \
    --model                  "${JUDGE_MODEL_PATH}" \
    --served-model-name      "${JUDGE_SERVED_MODEL_NAME}" \
    --host                   "${JUDGE_HOST}" \
    --port                   "${JUDGE_PORT}" \
    --api-key                "${JUDGE_API_KEY}" \
    --tensor-parallel-size   "${TP_SIZE}" \
    --gpu-memory-utilization "${JUDGE_GPU_MEMORY_UTILIZATION}" \
    --max-model-len          "${JUDGE_MAX_MODEL_LEN}" \
    --dtype                  "${JUDGE_DTYPE}" \
    --trust-remote-code \
    2>&1 | tee "${LOG_FILE}"
