#!/usr/bin/env bash
# Step 4 of 4: run the concurrent dialogue test.
#
# Prerequisites (run each in a separate terminal first):
#   Step 1: bash scripts-ppo-third-dynamic-reward-rubric/run_judge_server.sh
#   Step 2: bash scripts-ppo-third-dynamic-reward-rubric/run_partner_server.sh
#   Step 3: bash scripts-ppo-third-dynamic-reward-rubric/run_policy_server.sh
#
# Usage:
#   bash scripts-ppo-third-dynamic-reward-rubric/run_test_only_vllm.sh
#
# Override any variable via env, e.g.:
#   PEFT_CHECKPOINT_PATH=./checkpoints/20260626_153150/checkpoints_epoch6 \
#   TEST_MAX_SCENES=100 \
#     bash scripts-ppo-third-dynamic-reward-rubric/run_test_only_vllm.sh

set -euo pipefail

# ==========================================================================
# Configuration – edit here or override via env
# ==========================================================================

# --- Judge server (Step 1) ---
JUDGE_BASE_URL="${JUDGE_BASE_URL:-http://127.0.0.1:8000/v1}"
JUDGE_MODEL="${JUDGE_MODEL:-Qwen3.5-35B-A3B}"
JUDGE_API_KEY="${JUDGE_API_KEY:-EMPTY}"

# --- Partner server (Step 2) ---
# PARTNER_BASE_URL="${PARTNER_BASE_URL:-https://api.getoken.tech/v1}"
# PARTNER_MODEL="${PARTNER_MODEL:-"gpt-5.5"}"
# PARTNER_API_KEY="${PARTNER_API_KEY:-"sk-getoken-e35cf6dd61f16f717c1ae7a3e26f8f13"}"

PARTNER_BASE_URL="${PARTNER_BASE_URL:-http://127.0.0.2:8000/v1}"
PARTNER_MODEL="${PARTNER_MODEL:-Qwen3.5-35B-A3B}"
PARTNER_API_KEY="${PARTNER_API_KEY:-EMPTY}"

# --- Policy server (Step 3) ---
# POLICY_BASE_URL="${POLICY_BASE_URL:-https://api.getoken.tech/v1}"
# POLICY_MODEL="${POLICY_MODEL:-"gpt-5.5"}"
# POLICY_API_KEY="${POLICY_API_KEY:-"sk-getoken-e35cf6dd61f16f717c1ae7a3e26f8f13"}"

POLICY_BASE_URL="${POLICY_BASE_URL:-http://127.0.0.3:8000/v1}"
POLICY_MODEL="${POLICY_MODEL:-Qwen2.5-7B-Instruct}"
POLICY_API_KEY="${POLICY_API_KEY:-EMPTY}"

# --- Data ---
BACKGROUNDS_PATH="${BACKGROUNDS_PATH:-./data/backgrounds_true_data_subgoal.jsonl}"
TEST_SPLIT="${TEST_SPLIT:-val}"
VAL_RATIO="${VAL_RATIO:-0.146}"
TEST_MAX_SCENES="${TEST_MAX_SCENES:-0}"

# --- Dialogue ---
SCENE_BATCH_SIZE="${SCENE_BATCH_SIZE:-130}"
NUM_GROUPS="${NUM_GROUPS:-2}"
MAX_ROUNDS="${MAX_ROUNDS:-20}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"

# --- Output ---
TEST_OUTPUT_DIR="${TEST_OUTPUT_DIR:-./checkpoints}"

# --- Misc ---
WANDB="${WANDB:-0}"
PROJECT_NAME="${PROJECT_NAME:-sotopia-test-only-vllm}"

# ==========================================================================
# Derived paths
# ==========================================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

SCRIPT_DIR="scripts-ppo-third-dynamic-reward-rubric"
mkdir -p logs "${TEST_OUTPUT_DIR}"
LOG_FILE="logs/test_vllm_$(date +%Y%m%d_%H%M%S).log"

# ==========================================================================
# Validation
# ==========================================================================

if [[ -z "${POLICY_MODEL}" ]]; then
    echo "[ERROR] POLICY_MODEL is required (must match --served-model-name of the policy server)."
    exit 1
fi

# ==========================================================================
# Print config
# ==========================================================================

echo "=================================================================="
echo "  [Step 4/4] Running concurrent dialogue test"
echo "=================================================================="
echo "  POLICY_BASE_URL=${POLICY_BASE_URL}"
echo "  POLICY_MODEL=${POLICY_MODEL}"
echo "  PARTNER_BASE_URL=${PARTNER_BASE_URL}"
echo "  PARTNER_MODEL=${PARTNER_MODEL}"
echo "  JUDGE_BASE_URL=${JUDGE_BASE_URL}"
echo "  JUDGE_MODEL=${JUDGE_MODEL}"
echo "  BACKGROUNDS_PATH=${BACKGROUNDS_PATH}"
echo "  TEST_SPLIT=${TEST_SPLIT}  VAL_RATIO=${VAL_RATIO}"
echo "  TEST_MAX_SCENES=${TEST_MAX_SCENES}"
echo "  SCENE_BATCH_SIZE=${SCENE_BATCH_SIZE}  NUM_GROUPS=${NUM_GROUPS}"
echo "  MAX_ROUNDS=${MAX_ROUNDS}  MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"
echo "  TEST_OUTPUT_DIR=${TEST_OUTPUT_DIR}"
echo "  LOG_FILE=${LOG_FILE}"
echo "=================================================================="

# ==========================================================================
# Build argument list
# ==========================================================================

ARGS=(
    --policy-base-url              "${POLICY_BASE_URL}"
    --policy-model                 "${POLICY_MODEL}"
    --policy-api-key               "${POLICY_API_KEY}"
    --policy-server-ready-timeout  "0"
    --partner-base-url             "${PARTNER_BASE_URL}"
    --partner-model                "${PARTNER_MODEL}"
    --partner-api-key              "${PARTNER_API_KEY}"
    --judge-base-url               "${JUDGE_BASE_URL}"
    --judge-model                  "${JUDGE_MODEL}"
    --judge-api-key                "${JUDGE_API_KEY}"
    --backgrounds                  "${BACKGROUNDS_PATH}"
    --test-split                   "${TEST_SPLIT}"
    --val-ratio                    "${VAL_RATIO}"
    --max-rounds                   "${MAX_ROUNDS}"
    --max-new-tokens               "${MAX_NEW_TOKENS}"
    --scene-batch-size             "${SCENE_BATCH_SIZE}"
    --num-groups                   "${NUM_GROUPS}"
    --test-max-scenes              "${TEST_MAX_SCENES}"
    --output-dir                   "${TEST_OUTPUT_DIR}"
)

if [[ "${WANDB}" == "1" ]]; then
    ARGS+=(--wandb --wandb-project "${PROJECT_NAME}")
fi

# ==========================================================================
# Run
# ==========================================================================

python "${SCRIPT_DIR}/rl_test_only_vllm.py" "${ARGS[@]}" \
    2>&1 | tee "${LOG_FILE}"

echo "[INFO] Test complete. Artifacts in ${TEST_OUTPUT_DIR}"
