#!/usr/bin/env bash
# AgentSense multi-turn social dialogue evaluation launcher.
#
# Prerequisites (run each in a separate terminal first):
#   Step 1: start the judge vLLM server
#   Step 2: start the policy vLLM server  (the model being evaluated)
#   Step 3: start the partner vLLM server (can be the same server as policy)
#
# Then run:
#   bash Benchmarks/AgentSense/eval/run_agentsense_eval.sh
#
# Override any variable via env, e.g.:
#   POLICY_MODEL=my-finetuned-model \
#   MAX_SCENES=100 \
#     bash Benchmarks/AgentSense/eval/run_agentsense_eval.sh

set -euo pipefail

# ==========================================================================
# Configuration – edit here or override via env
# ==========================================================================

# --- Judge server ---
JUDGE_BASE_URL="${JUDGE_BASE_URL:-http://127.0.0.1:8000/v1}"
JUDGE_MODEL="${JUDGE_MODEL:-Qwen3.5-35B-A3B}"
JUDGE_API_KEY="${JUDGE_API_KEY:-EMPTY}"

# --- Partner server (drives all non-policy characters) ---
# Defaults to the same server as the policy model.
PARTNER_BASE_URL="${PARTNER_BASE_URL:-http://127.0.0.2:8000/v1}"
PARTNER_MODEL="${PARTNER_MODEL:-Qwen3.5-35B-A3B}"
PARTNER_API_KEY="${PARTNER_API_KEY:-EMPTY}"
PARTNER_TEMPERATURE="${PARTNER_TEMPERATURE:-0.7}"
PARTNER_MAX_TOKENS="${PARTNER_MAX_TOKENS:-128}"


# --- Policy server (model being evaluated) ---
POLICY_BASE_URL="${POLICY_BASE_URL:-http://127.0.0.3:8000/v1}"
POLICY_MODEL="${POLICY_MODEL:-Qwen2.5-7B-Instruct}"
POLICY_API_KEY="${POLICY_API_KEY:-EMPTY}"
POLICY_TEMPERATURE="${POLICY_TEMPERATURE:-0.7}"
POLICY_MAX_TOKENS="${POLICY_MAX_TOKENS:-128}"

# POLICY_BASE_URL="${POLICY_BASE_URL:-https://api.getoken.tech/v1}"
# POLICY_MODEL="${POLICY_MODEL:-"gpt-5.5"}"
# POLICY_API_KEY="${POLICY_API_KEY:-"sk-getoken-e35cf6dd61f16f717c1ae7a3e26f8f13"}"
# POLICY_TEMPERATURE="${POLICY_TEMPERATURE:-0.7}"
# POLICY_MAX_TOKENS="${POLICY_MAX_TOKENS:-128}" 

# --- Data ---
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
DATA_PATH="${DATA_PATH:-${ROOT_DIR}/data/agentsense_final_data.jsonl}"  

# --- Evaluation ---
MAX_SCENES="${MAX_SCENES:-0}"        # 0 = all scenes
EVAL_WORKERS="${EVAL_WORKERS:-32}"   # concurrent run items

# --- Output ---
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/Benchmarks/AgentSense/eval/output}"

# ==========================================================================
# Validation
# ==========================================================================

if [[ -z "${POLICY_MODEL}" ]]; then
    echo "[ERROR] POLICY_MODEL is required."
    exit 1
fi

if [[ -z "${JUDGE_MODEL}" ]]; then
    echo "[ERROR] JUDGE_MODEL is required."
    exit 1
fi

if [[ ! -f "${DATA_PATH}" ]]; then
    echo "[ERROR] Data file not found: ${DATA_PATH}"
    exit 1
fi

# ==========================================================================
# Setup
# ==========================================================================

cd "${ROOT_DIR}"
EVAL_DIR="Benchmarks/AgentSense/eval"
mkdir -p "${OUTPUT_DIR}" logs

LOG_FILE="logs/agentsense_eval_$(date +%Y%m%d_%H%M%S).log"

# ==========================================================================
# Print config
# ==========================================================================

echo "=================================================================="
echo "  AgentSense Evaluation"
echo "=================================================================="
echo "  POLICY_BASE_URL    = ${POLICY_BASE_URL}"
echo "  POLICY_MODEL       = ${POLICY_MODEL}"
echo "  POLICY_TEMPERATURE = ${POLICY_TEMPERATURE}"
echo "  POLICY_MAX_TOKENS  = ${POLICY_MAX_TOKENS}"
echo "  PARTNER_BASE_URL   = ${PARTNER_BASE_URL}"
echo "  PARTNER_MODEL      = ${PARTNER_MODEL}"
echo "  JUDGE_BASE_URL     = ${JUDGE_BASE_URL}"
echo "  JUDGE_MODEL        = ${JUDGE_MODEL}"
echo "  DATA_PATH          = ${DATA_PATH}"
echo "  MAX_SCENES         = ${MAX_SCENES}  (0 = all)"
echo "  EVAL_WORKERS       = ${EVAL_WORKERS}"
echo "  OUTPUT_DIR         = ${OUTPUT_DIR}"
echo "  LOG_FILE           = ${LOG_FILE}"
echo "=================================================================="

# ==========================================================================
# Run
# ==========================================================================

python "${EVAL_DIR}/run_agentsense_eval.py" \
    --policy-base-url              "${POLICY_BASE_URL}"    \
    --policy-model                 "${POLICY_MODEL}"       \
    --policy-api-key               "${POLICY_API_KEY}"     \
    --policy-temperature           "${POLICY_TEMPERATURE}" \
    --policy-max-tokens            "${POLICY_MAX_TOKENS}"  \
    --policy-server-ready-timeout  "0"                     \
    --partner-base-url             "${PARTNER_BASE_URL}"   \
    --partner-model                "${PARTNER_MODEL}"      \
    --partner-api-key              "${PARTNER_API_KEY}"    \
    --partner-temperature          "${PARTNER_TEMPERATURE}"\
    --partner-max-tokens           "${PARTNER_MAX_TOKENS}" \
    --judge-base-url               "${JUDGE_BASE_URL}"     \
    --judge-model                  "${JUDGE_MODEL}"        \
    --judge-api-key                "${JUDGE_API_KEY}"      \
    --data-path                    "${DATA_PATH}"          \
    --max-scenes                   "${MAX_SCENES}"         \
    --eval-workers                 "${EVAL_WORKERS}"       \
    --output-dir                   "${OUTPUT_DIR}"         \
    2>&1 | tee "${LOG_FILE}"

echo ""
echo "[INFO] Evaluation complete. Artifacts saved under: ${OUTPUT_DIR}"
echo "[INFO] Log: ${LOG_FILE}"
