#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts-ppo-third-new-para/run_rl_qwen_origin_new_prompt_ddp.sh
#
# 与 scripts-ppo-third-save 的差别（针对 52pp 上限的诊断）：
#   1) MODEL_PATH 默认指向 merged SFT (base + SFT LoRA 合并后的 full model)，
#      让 SFT 知识进入 base，新建的 LoRA 从 zero-init 学 RL 增量。
#   2) RESUME_FROM 默认为空：新 LoRA 配置 (r=16 + MLP) 与 SFT LoRA shape 不兼容。
#   3) LR 1e-6 -> 5e-6，让 LoRA 真正动起来。
#   4) LoRA r 8->16, alpha 16->32, target 加 gate/up/down_proj (MLP)。
#   5) GAMMA 1.0 -> 0.95，多轮对话需要 discounting。
#   6) PROCESS_REWARD_WEIGHT 0.3 -> 0.05，让 goal reward 真正主导。
#   7) KL_BETA = 0.02，加上 reference-KL 防止高 lr 下 collapse；
#      ref policy 通过 PEFT disable_adapter() 自动得到 base = merged SFT，零显存开销。
#   8) PROCESS_REWARD_KIND: "approval"(默认, partner 视角的认可度) | "goal_directed"
#      (第三方视角的 goal 推进度)。negotiation/boundary 这类 inverted-role 桶,
#      partner approval 与 goal 方向反, 切换到 goal_directed 才能给出正确信号。
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
POLICY_GPU_IDS=0,1,2,3
CRITIC_DEVICE_MAP=4,5,6,7
# PROJECT_PATH=/path/to/sotopia-dia
# PROJECT_NAME=sotopia-back-mid

# ---------- 模型路径 ----------
# Policy base 默认用 merged SFT (base + SFT LoRA 已 merge)。如果该目录不存在，
# 可以把它改回原来的 base Qwen2.5-7B-Instruct，但会失去 SFT 已学到的对话能力，
# 收益会差一些。
# MODEL_PATH=/fudan_university_cos/models/meta-llama/Meta-Llama-3.1-8B-Instruct
MODEL_PATH=/fudan_university_cos/models/LLM-Research/gemma-3-4b-it
# MODEL_PATH=/fudan_university_cos/models/Qwen/Qwen3-8B
# Critic 仍然从 base 起步即可（critic 学的是 V(s)，不需要 SFT 知识）。
CRITIC_MODEL_PATH=/fudan_university_cfs/wjn/sotopia-dia/model/Qwen/Qwen2.5-7B-Instruct

# TRAIN_NAME=back-mid  # used for SAVE_PATH and PROJECT_NAME
SAVE_PATH=./checkpoints/
BACKGROUNDS_PATH=./data/sotopia_backgrounds_clean.jsonl
MASTER_PORT=29501
WANDB=0  # default enabled; set 0 to disable
VAL_RATIO=0.146
DISABLE_TEST_EVAL=0  # set 1 to disable epoch-end test eval
TEST_MAX_SCENES=0    # 0 means evaluate all held-out scenes
TEST_EVERY_BATCHES=1 # run test every N train batches (1 = every batch, 0 = disable)
NCCL_TIMEOUT_MINUTES=60  # NCCL collective timeout in minutes (default in script: 60)
WANDB_HTTP_TIMEOUT=600  # wandb API/network timeout in seconds

# ---------- 优化器 ----------
LR=1e-5              # policy LR (was 1e-6 in third-save; 太小 LoRA 几乎不动)
CRITIC_LR=2e-5

# ---------- PPO ----------
GAMMA=0.95           # was 1.0; multi-turn dialogue 必须有 discounting
GAE_LAMBDA=0.95
PPO_CLIP=0.2
PPO_EPOCHS=4
PPO_MINIBATCH_SIZE=128
VF_COEF=0.5
ENT_COEF=0.001

# ---------- Reward shaping ----------
GOAL_REWARD_WEIGHT=1.0
PROCESS_REWARD_WEIGHT=0.3   # was 0.3; 累加压倒 goal -> 把权重压低让 goal 主导
KL_BETA=0.02                 # 0 关闭，>0 开启 PPO ref-KL penalty (k3 estimator)
# 过程奖励类型:
#   "approval"        = partner 视角认可度 (默认 / 原 third-save 设置)
#   "goal_directed"   = 第三方视角 goal 推进度 (third-new-para 实验)
#   "multi_dim_dynamic" = 13 LLM 维度 + 动态/预设权重
#   "rubric_dynamic" = 6 个正交维度，每 turn 先生成 rubric，再独立调用评分并聚合奖励
PROCESS_REWARD_KIND=rubric_dynamic
# 仅 dynamic/rubric 过程奖励用到：
#   TRAJECTORY_NORMALIZE=1  把每 turn 的 process reward 除以 EXPECTED_HORIZON，
#                           防止 PPO 学到 "拉长 horizon 累积 per-turn reward" 的廉价策略。
#   EXPECTED_HORIZON=5      用于 (a) 上面的 normalize 除数, (b) 判断 early/mid/late phase。
#   DYNAMIC_WEIGHTS_FROM_JUDGE=1 让 reward model (judge LLM) 自己输出每个 turn 的权重，
#                                 与 baseline schedule 按 mix_ratio 混合。0=纯 schedule。
#   DYNAMIC_WEIGHTS_MIX_RATIO=0.5 LLM 权重和 schedule baseline 的混合比例。
#                                 0.0=纯 baseline, 1.0=纯 LLM, 0.5=折中 (推荐)。
TRAJECTORY_NORMALIZE=1
EXPECTED_HORIZON=5
DYNAMIC_WEIGHTS_FROM_JUDGE=1
DYNAMIC_WEIGHTS_MIX_RATIO=1

# ---------- LoRA ----------
LORA_R=16                    # was 8
LORA_ALPHA=32                # was 16
LORA_TARGETS="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"  # 加上 MLP

PARTNER_BASE_URL=http://127.0.0.1:8000/v1  # used when source=service
PARTNER_MODEL=Qwen3.5-35B-A3B
PARTNER_API_KEY=EMPTY
REWARD_BASE_URL=http://127.0.0.1:8000/v1
REWARD_MODEL=Qwen3.5-35B-A3B
REWARD_API_KEY=EMPTY
JUDGE_BASE_URL=http://127.0.0.1:8000/v1
JUDGE_MODEL=Qwen3.5-35B-A3B
JUDGE_API_KEY=EMPTY
RL_PARTNER_MAX_WORKERS=8
RL_REWARD_MAX_WORKERS=8
RL_PROGRESS_MAX_WORKERS=6
RL_TEST_REWARD_MAX_WORKERS=8

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
# RESUME_FROM 默认空：新 LoRA 配置 (r=16 + MLP modules) 与 SFT LoRA (r=16 q/k/v/o) shape 不兼容，
# 不能 resume；SFT 的知识已经通过 MODEL_PATH (merged SFT) 进入 base。
# 如果你确实想从某个 RL checkpoint 续训，把同样配置的 ckpt 路径放进来。
export RESUME_FROM="${RESUME_FROM:-checkpoints/20260813_164246/checkpoints_epoch5}"
export BACKGROUNDS_PATH="${BACKGROUNDS_PATH:-${PROJECT_PATH}/data/backgrounds_mid.jsonl}"
export CRITIC_MODEL_PATH="${CRITIC_MODEL_PATH:-${MODEL_PATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export POLICY_GPU_IDS="${POLICY_GPU_IDS:-0,1}"
export CRITIC_DEVICE_MAP="${CRITIC_DEVICE_MAP:-2,3}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29501}"
export WANDB="${WANDB:-1}"
export VAL_RATIO="${VAL_RATIO:-0.146}"
export DISABLE_TEST_EVAL="${DISABLE_TEST_EVAL:-0}"
export TEST_MAX_SCENES="${TEST_MAX_SCENES:-0}"
export TEST_EVERY_BATCHES="${TEST_EVERY_BATCHES:-1}"
export LR="${LR:-5e-6}"
export CRITIC_LR="${CRITIC_LR:-1e-5}"
export GAMMA="${GAMMA:-0.95}"
export GAE_LAMBDA="${GAE_LAMBDA:-0.95}"
export PPO_CLIP="${PPO_CLIP:-0.2}"
export PPO_EPOCHS="${PPO_EPOCHS:-4}"
export PPO_MINIBATCH_SIZE="${PPO_MINIBATCH_SIZE:-128}"
export VF_COEF="${VF_COEF:-0.5}"
export ENT_COEF="${ENT_COEF:-0.001}"
export GOAL_REWARD_WEIGHT="${GOAL_REWARD_WEIGHT:-1.0}"
export PROCESS_REWARD_WEIGHT="${PROCESS_REWARD_WEIGHT:-0.05}"
export PROCESS_REWARD_KIND="${PROCESS_REWARD_KIND:-rubric_dynamic}"
export TRAJECTORY_NORMALIZE="${TRAJECTORY_NORMALIZE:-0}"
export EXPECTED_HORIZON="${EXPECTED_HORIZON:-5}"
export DYNAMIC_WEIGHTS_FROM_JUDGE="${DYNAMIC_WEIGHTS_FROM_JUDGE:-0}"
export DYNAMIC_WEIGHTS_MIX_RATIO="${DYNAMIC_WEIGHTS_MIX_RATIO:-0.5}"
export KL_BETA="${KL_BETA:-0.02}"
export LORA_R="${LORA_R:-16}"
export LORA_ALPHA="${LORA_ALPHA:-32}"
export LORA_TARGETS="${LORA_TARGETS:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}"
export PARTNER_BASE_URL="${PARTNER_BASE_URL:-http://127.0.0.1:8000/v1}"
export PARTNER_MODEL="${PARTNER_MODEL:-Qwen3.5-35B-A3B}"
export PARTNER_API_KEY="${PARTNER_API_KEY:-EMPTY}"
export REWARD_BASE_URL="${REWARD_BASE_URL:-http://127.0.0.1:8000/v1}"
export REWARD_MODEL="${REWARD_MODEL:-Qwen3.5-35B-A3B}"
export REWARD_API_KEY="${REWARD_API_KEY:-EMPTY}"
export JUDGE_BASE_URL="${JUDGE_BASE_URL:-${REWARD_BASE_URL}}"
export JUDGE_MODEL="${JUDGE_MODEL:-${REWARD_MODEL}}"
export JUDGE_API_KEY="${JUDGE_API_KEY:-${REWARD_API_KEY}}"
export RL_PARTNER_MAX_WORKERS="${RL_PARTNER_MAX_WORKERS:-8}"
export RL_REWARD_MAX_WORKERS="${RL_REWARD_MAX_WORKERS:-4}"
export RL_PROGRESS_MAX_WORKERS="${RL_PROGRESS_MAX_WORKERS:-4}"
export RL_TEST_REWARD_MAX_WORKERS="${RL_TEST_REWARD_MAX_WORKERS:-4}"
# export WANDB_HTTP_TIMEOUT="${WANDB_HTTP_TIMEOUT:-600}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


mkdir -p "${SAVE_PATH}"
mkdir -p "${ROOT_DIR}/logs"

# Sanity check: warn if MODEL_PATH points at a directory that doesn't exist.
if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "WARNING: MODEL_PATH does not exist: ${MODEL_PATH}"
  echo "         If you don't have the merged-SFT model yet, either run the"
  echo "         SDPO/scripts/step0_merge_sft_lora.sh helper, or set"
  echo "         MODEL_PATH=/fudan_university_cfs/wjn/sotopia-dia/model/Qwen/Qwen2.5-7B-Instruct"
  echo "         (this falls back to plain base, losing SFT knowledge)."
fi

# Infer process count from POLICY_GPU_IDS unless user overrides.
if [[ -n "${NPROC_PER_NODE:-}" ]]; then
  PROC_NUM="${NPROC_PER_NODE}"
else
  DEV_STR="${POLICY_GPU_IDS// /}"
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
  --critic-model-path "${CRITIC_MODEL_PATH}"
  --critic-device-map "${CRITIC_DEVICE_MAP}"
  --save-path "${SAVE_PATH}"
  --backgrounds "${BACKGROUNDS_PATH}"
  --learning-rate "${LR}"
  --critic-learning-rate "${CRITIC_LR}"
  --gamma "${GAMMA}"
  --gae-lambda "${GAE_LAMBDA}"
  --ppo-clip "${PPO_CLIP}"
  --ppo-epochs "${PPO_EPOCHS}"
  --ppo-minibatch-size "${PPO_MINIBATCH_SIZE}"
  --vf-coef "${VF_COEF}"
  --ent-coef "${ENT_COEF}"
  --goal-reward-weight "${GOAL_REWARD_WEIGHT}"
  --process-reward-weight "${PROCESS_REWARD_WEIGHT}"
  --kl-beta "${KL_BETA}"
  --lora-r "${LORA_R}"
  --lora-alpha "${LORA_ALPHA}"
  --lora-target-modules "${LORA_TARGETS}"
  --epochs 15
  --max-rounds 20
  --scene-batch-size 65
  --gradient-batch-size 16
  --num-groups 2
  --max-new-tokens 64
  --checkpoint-every 1
  --seed 42
  --val-ratio "${VAL_RATIO}"
  --test-max-scenes "${TEST_MAX_SCENES}"
  --test-every-batches "${TEST_EVERY_BATCHES}"
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
echo "POLICY_GPU_IDS=${POLICY_GPU_IDS}"
echo "CRITIC_DEVICE_MAP=${CRITIC_DEVICE_MAP}"
echo "TRAIN_NAME=${TRAIN_NAME}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "CRITIC_MODEL_PATH=${CRITIC_MODEL_PATH}"
echo "SAVE_PATH=${SAVE_PATH}"
echo "BACKGROUNDS_PATH=${BACKGROUNDS_PATH}"
echo "PROJECT_NAME=${PROJECT_NAME}"
echo "WANDB=${WANDB}"
echo "VAL_RATIO=${VAL_RATIO}"
echo "DISABLE_TEST_EVAL=${DISABLE_TEST_EVAL}"
echo "TEST_MAX_SCENES=${TEST_MAX_SCENES}"
echo "TEST_EVERY_BATCHES=${TEST_EVERY_BATCHES}"
echo "LR=${LR}"
echo "CRITIC_LR=${CRITIC_LR}"
echo "GAMMA=${GAMMA}"
echo "GOAL_REWARD_WEIGHT=${GOAL_REWARD_WEIGHT}"
echo "PROCESS_REWARD_WEIGHT=${PROCESS_REWARD_WEIGHT}"
echo "PROCESS_REWARD_KIND=${PROCESS_REWARD_KIND}"
echo "TRAJECTORY_NORMALIZE=${TRAJECTORY_NORMALIZE}"
echo "EXPECTED_HORIZON=${EXPECTED_HORIZON}"
echo "DYNAMIC_WEIGHTS_FROM_JUDGE=${DYNAMIC_WEIGHTS_FROM_JUDGE}"
echo "DYNAMIC_WEIGHTS_MIX_RATIO=${DYNAMIC_WEIGHTS_MIX_RATIO}"
echo "KL_BETA=${KL_BETA}"
echo "LORA_R=${LORA_R}  LORA_ALPHA=${LORA_ALPHA}  LORA_TARGETS=${LORA_TARGETS}"
echo "PARTNER_BASE_URL=${PARTNER_BASE_URL}"
echo "PARTNER_MODEL=${PARTNER_MODEL}"
echo "REWARD_BASE_URL=${REWARD_BASE_URL}"
echo "REWARD_MODEL=${REWARD_MODEL}"
echo "JUDGE_BASE_URL=${JUDGE_BASE_URL}"
echo "JUDGE_MODEL=${JUDGE_MODEL}"
echo "RL_PARTNER_MAX_WORKERS=${RL_PARTNER_MAX_WORKERS}"
echo "RL_REWARD_MAX_WORKERS=${RL_REWARD_MAX_WORKERS}"
echo "RL_PROGRESS_MAX_WORKERS=${RL_PROGRESS_MAX_WORKERS}"
echo "RL_TEST_REWARD_MAX_WORKERS=${RL_TEST_REWARD_MAX_WORKERS}"
# echo "WANDB_HTTP_TIMEOUT=${WANDB_HTTP_TIMEOUT}"
if [[ -n "${RESUME_FROM:-}" ]]; then
  echo "RESUME_FROM=${RESUME_FROM}"
fi

python -m torch.distributed.run \
  --nproc_per_node "${PROC_NUM}" \
  --master_addr "${MASTER_ADDR}" \
  --master_port "${MASTER_PORT}" \
  scripts-ppo/rl_qwen_origin_new_prompt_ddp.py \
  "${COMMON_ARGS[@]}" 2>&1 | tee "${ROOT_DIR}/logs/train_ddp_$(date +%Y%m%d_%H%M%S).log"
