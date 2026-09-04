#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_rl_qwen_origin_new_prompt_ddp.sh
# Optional overrides (env):
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
# PROJECT_PATH=/path/to/sotopia-dia
# PROJECT_NAME=sotopia-back-mid
MODEL_PATH=/fudan_university_cfs/wjn/sotopia-dia/model/Qwen/Qwen2.5-7B-Instruct
# TRAIN_NAME=back-mid  # used for SAVE_PATH and PROJECT_NAME
SAVE_PATH=./checkpoints/
BACKGROUNDS_PATH=data/sotopia_backgrounds_clean.jsonl
MASTER_PORT=29501
WANDB=0  # default enabled; set 0 to disable
VAL_RATIO=0.146
DISABLE_TEST_EVAL=0  # set 1 to disable epoch-end test eval
TEST_MAX_SCENES=0    # 0 means evaluate all held-out scenes
TEST_EVERY_BATCHES=1 # run test every N train batches (1 = every batch, 0 = disable)
TEST_SCENE_BATCH_SIZE=65  # scenes per test batch
TEST_NUM_GROUPS=2         # groups per scene during test
NCCL_TIMEOUT_MINUTES=120  # NCCL collective timeout in minutes (default in script: 120)
ENTROPY_COEF=0.001  # entropy regularization coefficient
WANDB_HTTP_TIMEOUT=600  # wandb API/network timeout in seconds
PARTNER_BASE_URL=http://127.0.0.1:8000/v1  # used when source=service
PARTNER_MODEL=Qwen3.5-35B-A3B
PARTNER_API_KEY=EMPTY
REWARD_BASE_URL=http://127.0.0.1:8000/v1
REWARD_MODEL=Qwen3.5-35B-A3B
REWARD_API_KEY=EMPTY
JUDGE_BASE_URL=http://127.0.0.1:8000/v1
JUDGE_MODEL=Qwen3.5-35B-A3B
JUDGE_API_KEY=EMPTY

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# 项目根目录，未设置时默认为脚本所在仓库根
export PROJECT_PATH="${PROJECT_PATH:-${ROOT_DIR}}"
export TRAIN_NAME="${TRAIN_NAME:-back-mid}"
export PROJECT_NAME="${PROJECT_NAME:-sotopia-${TRAIN_NAME}}"
# 训练用小模型（policy）路径，需自行设置或复制 env.example 为 .env 后配置 MODEL_PATH
if [[ -z "${MODEL_PATH:-}" ]]; then
  echo "Error: MODEL_PATH is not set. Set it to your Qwen model path (e.g. export MODEL_PATH=/path/to/Qwen2.5-7B-Instruct). See env.example."
  exit 1
fi
if [[ -z "${SAVE_PATH:-}" ]]; then
  echo "Error: SAVE_PATH is not set. Set it to your checkpoint output directory (e.g. export SAVE_PATH=/path/to/checkpoints). See env.example."
  exit 1
fi
# export RESUME_FROM="${RESUME_FROM:-checkpoints/20260325_235736/checkpoints_epoch3}"  # 可选：从此 checkpoint 恢复，如 RESUME_FROM=${SAVE_PATH}_epoch1_batch10
export BACKGROUNDS_PATH="${BACKGROUNDS_PATH:-${PROJECT_PATH}/data/backgrounds_mid.jsonl}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5,6,7}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29501}"
export WANDB="${WANDB:-1}"
export VAL_RATIO="${VAL_RATIO:-0.146}"
export DISABLE_TEST_EVAL="${DISABLE_TEST_EVAL:-0}"
export TEST_MAX_SCENES="${TEST_MAX_SCENES:-0}"
export TEST_EVERY_BATCHES="${TEST_EVERY_BATCHES:-1}"
export TEST_SCENE_BATCH_SIZE="${TEST_SCENE_BATCH_SIZE:-64}"
export TEST_NUM_GROUPS="${TEST_NUM_GROUPS:-4}"
export ENTROPY_COEF="${ENTROPY_COEF:-0.0}"
export PPO_CLIP="${PPO_CLIP:-0.2}"
export PROCESS_REWARD_WEIGHT="${PROCESS_REWARD_WEIGHT:-0.1}"
export GOAL_REWARD_WEIGHT="${GOAL_REWARD_WEIGHT:-1.0}"
export GAMMA="${GAMMA:-1.0}"
export EXPECTED_HORIZON="${EXPECTED_HORIZON:-5}"
export DYNAMIC_WEIGHTS_FROM_JUDGE="${DYNAMIC_WEIGHTS_FROM_JUDGE:-0}"
export DYNAMIC_WEIGHTS_MIX_RATIO="${DYNAMIC_WEIGHTS_MIX_RATIO:-0.5}"
export PARTNER_BASE_URL="${PARTNER_BASE_URL:-http://127.0.0.1:8000/v1}"
export PARTNER_MODEL="${PARTNER_MODEL:-Qwen3.5-35B-A3B}"
export PARTNER_API_KEY="${PARTNER_API_KEY:-EMPTY}"
export REWARD_BASE_URL="${REWARD_BASE_URL:-http://127.0.0.1:8000/v1}"
export REWARD_MODEL="${REWARD_MODEL:-Qwen3.5-35B-A3B}"
export REWARD_API_KEY="${REWARD_API_KEY:-EMPTY}"
export JUDGE_BASE_URL="${JUDGE_BASE_URL:-${REWARD_BASE_URL}}"
export JUDGE_MODEL="${JUDGE_MODEL:-${REWARD_MODEL}}"
export JUDGE_API_KEY="${JUDGE_API_KEY:-${REWARD_API_KEY}}"
# export WANDB_HTTP_TIMEOUT="${WANDB_HTTP_TIMEOUT:-600}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


mkdir -p "${SAVE_PATH}"
mkdir -p "${ROOT_DIR}/logs"
 
# Infer process count from CUDA_VISIBLE_DEVICES unless user overrides.
if [[ -n "${NPROC_PER_NODE:-}" ]]; then
  PROC_NUM="${NPROC_PER_NODE}"
else
  DEV_STR="${CUDA_VISIBLE_DEVICES// /}"
  if [[ -z "${DEV_STR}" ]]; then
    PROC_NUM=1
  else
    PROC_NUM="$(awk -F',' '{print NF}' <<< "${DEV_STR}")"
  fi
fi

COMMON_ARGS=(
  --use-ddp
  --ddp-backend nccl
  --nccl-timeout-minutes "${NCCL_TIMEOUT_MINUTES:-120}"
  --policy-model-path "${MODEL_PATH}"
  --save-path "${SAVE_PATH}"
  --backgrounds "${BACKGROUNDS_PATH}"
  --learning-rate 1e-6
  --kl-constraint 0.02
  --entropy-coef "${ENTROPY_COEF}"
  --ppo-clip "${PPO_CLIP}"
  --process-reward-weight "${PROCESS_REWARD_WEIGHT}"
  --goal-reward-weight "${GOAL_REWARD_WEIGHT}"
  --gamma "${GAMMA}"
  --expected-horizon "${EXPECTED_HORIZON}"
  --dynamic-weights-mix-ratio "${DYNAMIC_WEIGHTS_MIX_RATIO}"
  --epochs 15
  --max-rounds 20
  --scene-batch-size 16
  --gradient-batch-size 16
  --num-groups 16
  --max-new-tokens 128
  --checkpoint-every 1
  --seed 42
  --val-ratio "${VAL_RATIO}"
  --test-max-scenes "${TEST_MAX_SCENES}"
  --test-every-batches "${TEST_EVERY_BATCHES}"
  --test-scene-batch-size "${TEST_SCENE_BATCH_SIZE}"
  --test-num-groups "${TEST_NUM_GROUPS}"
)

COMMON_ARGS+=(
  --partner-base-url "${PARTNER_BASE_URL}"
  --partner-model "${PARTNER_MODEL}"
  --partner-api-key "${PARTNER_API_KEY}"
  --reward-base-url "${REWARD_BASE_URL}"
  --reward-model "${REWARD_MODEL}"
  --reward-api-key "${REWARD_API_KEY}"
  --judge-base-url "${JUDGE_BASE_URL}"
  --judge-model "${JUDGE_MODEL}"
  --judge-api-key "${JUDGE_API_KEY}"
)

if [[ "${DYNAMIC_WEIGHTS_FROM_JUDGE}" == "1" ]]; then
  COMMON_ARGS+=(--dynamic-weights-from-judge)
fi

if [[ -n "${RESUME_FROM:-}" ]]; then
  COMMON_ARGS+=(--resume-from "${RESUME_FROM}")
fi

if [[ "${WANDB}" == "1" ]]; then
  COMMON_ARGS+=(--wandb --wandb-project "${PROJECT_NAME}")
fi

if [[ "${DISABLE_TEST_EVAL}" == "1" ]]; then
  COMMON_ARGS+=(--disable-test-eval)
fi

echo "Launching DDP training with ${PROC_NUM} processes"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "TRAIN_NAME=${TRAIN_NAME}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "SAVE_PATH=${SAVE_PATH}"
echo "BACKGROUNDS_PATH=${BACKGROUNDS_PATH}"
echo "PROJECT_NAME=${PROJECT_NAME}"
echo "WANDB=${WANDB}"
echo "VAL_RATIO=${VAL_RATIO}"
echo "DISABLE_TEST_EVAL=${DISABLE_TEST_EVAL}"
echo "TEST_MAX_SCENES=${TEST_MAX_SCENES}"
echo "TEST_EVERY_BATCHES=${TEST_EVERY_BATCHES}"
echo "TEST_SCENE_BATCH_SIZE=${TEST_SCENE_BATCH_SIZE}"
echo "TEST_NUM_GROUPS=${TEST_NUM_GROUPS}"
echo "ENTROPY_COEF=${ENTROPY_COEF}"
echo "PARTNER_BASE_URL=${PARTNER_BASE_URL}"
echo "PARTNER_MODEL=${PARTNER_MODEL}"
echo "PARTNER_API_KEY=${PARTNER_API_KEY}"
echo "REWARD_BASE_URL=${REWARD_BASE_URL}"
echo "REWARD_MODEL=${REWARD_MODEL}"
echo "REWARD_API_KEY=${REWARD_API_KEY}"
echo "JUDGE_BASE_URL=${JUDGE_BASE_URL}"
echo "JUDGE_MODEL=${JUDGE_MODEL}"
echo "JUDGE_API_KEY=${JUDGE_API_KEY}"
# echo "WANDB_HTTP_TIMEOUT=${WANDB_HTTP_TIMEOUT}"
if [[ -n "${RESUME_FROM:-}" ]]; then
  echo "RESUME_FROM=${RESUME_FROM}"
fi

python -m torch.distributed.run \
  --nproc_per_node "${PROC_NUM}" \
  --master_addr "${MASTER_ADDR}" \
  --master_port "${MASTER_PORT}" \
  scripts-grpo/rl_qwen_origin_new_prompt_ddp.py \
  "${COMMON_ARGS[@]}" 2>&1 | tee "${ROOT_DIR}/logs/train_ddp_$(date +%Y%m%d_%H%M%S).log"
