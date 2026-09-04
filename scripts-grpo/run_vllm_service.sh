#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_vllm_service.sh
#
# Optional overrides (env):
VLLM_MODEL_PATH="/fudan_university_cos/models/Qwen/Qwen3.5-35B-A3B"                   # required (fallback: MODEL_PATH)
VLLM_HOST=127.0.0.1                              # fallback: HOST
VLLM_PORT=8000                                   # fallback: PORT
VLLM_API_KEY=EMPTY                               # fallback: API_KEY
VLLM_SERVED_MODEL_NAME=Qwen3.5-35B-A3B      # fallback: SERVED_MODEL_NAME
CUDA_VISIBLE_DEVICES=6,7
# VLLM_TENSOR_PARALLEL_SIZE=2                      # fallback: TENSOR_PARALLEL_SIZE
VLLM_GPU_MEMORY_UTILIZATION=0.9                  # fallback: GPU_MEMORY_UTILIZATION
VLLM_MAX_MODEL_LEN=8192                         # fallback: MAX_MODEL_LEN
VLLM_DTYPE=auto                                  # fallback: DTYPE
VLLM_TRUST_REMOTE_CODE=1                         # fallback: TRUST_REMOTE_CODE
VLLM_LOG_DIR=logs
# VLLM_EXTRA_ARGS="--disable-log-requests"
#
# This script starts vLLM OpenAI-compatible server at:
#   http://${HOST}:${PORT}/v1

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

MODEL_PATH_EFFECTIVE="${VLLM_MODEL_PATH:-${VLLM_MODEL_PATH:-}}"
if [[ -z "${MODEL_PATH_EFFECTIVE}" ]]; then
  echo "Error: VLLM_MODEL_PATH (or MODEL_PATH) is not set. Example:"
  echo "  export VLLM_MODEL_PATH=/path/to/Qwen3.5-27B"
  exit 1
fi

export VLLM_MODEL_PATH="${MODEL_PATH_EFFECTIVE}"
export VLLM_HOST="${VLLM_HOST:-${HOST:-127.0.0.1}}"
export VLLM_PORT="${VLLM_PORT:-${PORT:-8000}}"
export VLLM_API_KEY="${VLLM_API_KEY:-${API_KEY:-EMPTY}}"
export VLLM_SERVED_MODEL_NAME="${VLLM_SERVED_MODEL_NAME:-${SERVED_MODEL_NAME:-$(basename "${VLLM_MODEL_PATH}")}}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-${GPU_MEMORY_UTILIZATION:-0.9}}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-${MAX_MODEL_LEN:-4096}}"
export VLLM_DTYPE="${VLLM_DTYPE:-${DTYPE:-auto}}"
export VLLM_TRUST_REMOTE_CODE="${VLLM_TRUST_REMOTE_CODE:-${TRUST_REMOTE_CODE:-1}}"
export VLLM_LOG_DIR="${VLLM_LOG_DIR:-logs}"
export VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"

if [[ -n "${VLLM_TENSOR_PARALLEL_SIZE:-}" ]]; then
  TP_SIZE="${VLLM_TENSOR_PARALLEL_SIZE}"
elif [[ -n "${TENSOR_PARALLEL_SIZE:-}" ]]; then
  TP_SIZE="${TENSOR_PARALLEL_SIZE}"
else
  DEV_STR="${CUDA_VISIBLE_DEVICES:-}"
  DEV_STR="${DEV_STR// /}"
  if [[ -z "${DEV_STR}" ]]; then
    TP_SIZE=1
  else
    TP_SIZE="$(awk -F',' '{print NF}' <<< "${DEV_STR}")"
  fi
fi

mkdir -p "${VLLM_LOG_DIR}"
LOG_FILE="${VLLM_LOG_DIR}/vllm_$(date +%Y%m%d_%H%M%S).log"

echo "Starting vLLM service..."
echo "VLLM_MODEL_PATH=${VLLM_MODEL_PATH}"
echo "VLLM_SERVED_MODEL_NAME=${VLLM_SERVED_MODEL_NAME}"
echo "VLLM_HOST=${VLLM_HOST}"
echo "VLLM_PORT=${VLLM_PORT}"
echo "TENSOR_PARALLEL_SIZE=${TP_SIZE}"
echo "VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION}"
echo "VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN}"
echo "VLLM_DTYPE=${VLLM_DTYPE}"
echo "LOG_FILE=${LOG_FILE}"

ARGS=(
  --model "${VLLM_MODEL_PATH}"
  --served-model-name "${VLLM_SERVED_MODEL_NAME}"
  --host "${VLLM_HOST}"
  --port "${VLLM_PORT}"
  --api-key "${VLLM_API_KEY}"
  --tensor-parallel-size "${TP_SIZE}"
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}"
  --max-model-len "${VLLM_MAX_MODEL_LEN}"
  --dtype "${VLLM_DTYPE}"
)

if [[ "${VLLM_TRUST_REMOTE_CODE}" == "1" ]]; then
  ARGS+=(--trust-remote-code)
fi

# shellcheck disable=SC2206
EXTRA_ARR=(${VLLM_EXTRA_ARGS})
ARGS+=("${EXTRA_ARR[@]}")

python -m vllm.entrypoints.openai.api_server \
  "${ARGS[@]}" 2>&1 | tee "${LOG_FILE}"

