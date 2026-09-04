# SocialRL: Refining LLMs' Social Intelligence through Multi-turn Reinforcement Learning and Reward Design

SocialRL trains a smaller policy model to pursue explicit social goals while interacting with a model-driven conversation partner. Training-time rewards combine terminal goal achievement with per-turn process feedback, and the resulting policies can be evaluated with the AgentSense multi-character benchmark included in this repository.

The repository currently contains two reinforcement-learning implementations:

- **PPO**: actor-critic PPO with LoRA policy tuning, a learned critic, optional reference-policy KL regularization, and configurable process rewards.
- **Process-GRPO**: group-based policy optimization with multiple dialogue rollouts per scene, dynamic rubric rewards, global reward normalization, and backward-cumulative advantages.


## Highlights

- Goal-oriented social dialogue environments based on Sotopia-style background scenes.
- Terminal goal rewards in `[0, 1]` for training and binary goal-success scores for evaluation.
- Dynamic process rewards based on six high-level dimensions:
  `goal_advancement`, `strategic_positioning`, `relational_attunement`,
  `persona_consistency`, `contextual_coherence`, and `turn_quality`.
- Six-dimensional dynamic rubric process rewards for PPO and Process-GRPO.

## Repository Layout

```text
SocialRL/
├── data/
│   ├── sotopia_backgrounds_clean.jsonl
│   ├── sotopia_backgrounds_flat_tags_clean.jsonl
│   ├── sotopia_characters_backgrounds_chinese.jsonl
│   ├── agentsense_final_data.jsonl
│   ├── env_ids.txt
│   └── form_ids.txt
├── scripts-grpo/
│   ├── rl_qwen_origin_new_prompt_ddp.py     # Process-GRPO entry point
│   ├── rl_training_loop_goal_single.py      # Rollouts, rewards, and updates
│   ├── rl_testing_loop_goal_single.py       # Held-out evaluation loop
│   ├── rl_policy_model.py                   # LoRA policy model
│   ├── rl_reward.py                         # Goal and rubric rewards
│   ├── rl_dataset.py                        # Scene loading and dialogue state
│   ├── dynamic_reward_config.py             # Role/phase weight schedules
│   ├── prompt.py                            # Dialogue and judge prompts
│   ├── utils.py                             # API and rollout utilities
│   ├── run_rl_qwen_origin_new_prompt_ddp.sh
│   ├── run_vllm_service.sh
│   └── rl_test_only.py
├── scripts-ppo/
│   ├── rl_qwen_origin_new_prompt_ddp.py     # PPO entry point
│   ├── rl_training_loop_goal_single.py      # PPO rollout and update loop
│   ├── rl_testing_loop_goal_single.py       # Held-out evaluation loop
│   ├── rl_policy_model.py                   # LoRA policy and critic models
│   ├── rl_reward.py                         # PPO reward implementations
│   ├── rl_dataset.py                        # Scene loading and dialogue state
│   ├── dynamic_reward_config.py
│   ├── prompt.py
│   ├── utils.py
│   ├── vllm_policy.py                       # vLLM policy client
│   ├── score_relation_delta.py              # Relation-change post-scoring
│   ├── run_rl_qwen_origin_new_prompt_ddp.sh
│   ├── run_vllm_service.sh
│   └── rl_test_only_vllm.py
├── agentsense_eval/
│   ├── run_agentsense_eval.py               # AgentSense entry point
│   ├── agentsense_runner.py                 # Multi-character evaluation loop
│   ├── agentsense_model.py                  # OpenAI-compatible model client
│   ├── agentsense_judge.py                  # Goal and relation judges
│   ├── agentsense_utils.py                  # Data and prompt helpers
│   ├── agentsense_prompt.py
│   ├── rejudge_agentsense_repeated.py       # Repeat/re-score existing results
│   └── run_*.sh                             # vLLM and evaluation launchers
└── README.md
```

The two training directories share the same overall scene format and many conceptual components, but their implementations and default hyperparameters are independent.

- [`scripts-grpo/README.md`](scripts-grpo/README.md)
- [`scripts-ppo/README.md`](scripts-ppo/README.md)

## How the System Works

Each training scene contains a scenario, character backgrounds, and one or more social goals. A typical rollout has the following components:

```text
Scene JSONL
    |
    v
Role-aware system prompts and dialogue state
    |
    +--> Policy model and Partner model generates the character's reply alternately
    |
    +--> Judge/reward model scores process and terminal rewards
    |
    v
LoRA policy update with PPO or Process-GRPO
```

## Data

### Sotopia Data

The primary training files are JSONL files in `data/`:

- `sotopia_backgrounds_clean.jsonl`: cleaned English social scenarios with `scenario`, `agents_background`, and `social_goals` fields.
- `sotopia_backgrounds_flat_tags_clean.jsonl`: a flattened-tag variant of the Sotopia-style scenes.
- `sotopia_characters_backgrounds_chinese.jsonl`: Chinese character background records.


### AgentSense Data

`data/agentsense_final_data.jsonl` contains multi-character scenes. Each scene has a `sample_idx`, a `background`, a `description`, and a `characters` list.

## Released LoRA Models

The trained SocialRL LoRA adapters are available on Hugging Face:

- [wjnwjnwj/SocialRL](https://huggingface.co/wjnwjnwj/SocialRL)

When serving an adapter with vLLM, provide the base model through `POLICY_MODEL_PATH` and the adapter through `PEFT_CHECKPOINT_PATH`, as shown in the AgentSense evaluation instructions above.

## Setup

The repository provides a pinned dependency specification in [`requirements.txt`](requirements.txt).

Create and activate a Python environment:

```bash
conda create -n socialrl python=3.11
conda activate social
pip install -r requirements.txt
```


### Serving a Model with vLLM

The training scripts expect an OpenAI-compatible endpoint whose base URL ends in `/v1`. Start a partner/reward/judge model with:

```bash
VLLM_MODEL_PATH=/path/to/large-model \
VLLM_SERVED_MODEL_NAME=large-model \
CUDA_VISIBLE_DEVICES=0,1 \
bash scripts-grpo/run_vllm_service.sh
```

### Training with Process-GRPO

Process-GRPO is the recommended entry point for experiments using the dynamic six-dimensional rubric reward.

Launch distributed training:

```bash
MODEL_PATH=/path/to/policy-model \
BACKGROUNDS_PATH=./data/sotopia_backgrounds_clean.jsonl \
SAVE_PATH=./checkpoints \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash scripts-grpo/run_rl_qwen_origin_new_prompt_ddp.sh
```

### Training with PPO

PPO uses a trainable LoRA policy and a separate critic model. Its process reward is based on the six-dimensional dynamic rubric described in [`scripts-ppo/README.md`](scripts-ppo/README.md). Each policy turn receives binary rubric judgments that are aggregated into a weighted process reward.

Launch PPO with:

```bash
MODEL_PATH=/path/to/policy-model \
CRITIC_MODEL_PATH=/path/to/critic-model \
BACKGROUNDS_PATH=./data/sotopia_backgrounds_clean.jsonl \
SAVE_PATH=./checkpoints-ppo \
PARTNER_BASE_URL=http://127.0.0.1:8000/v1 \
PARTNER_MODEL=large-model \
REWARD_BASE_URL=http://127.0.0.1:8000/v1 \
REWARD_MODEL=large-model \
JUDGE_BASE_URL=http://127.0.0.1:8000/v1 \
JUDGE_MODEL=large-model \
WANDB=0 \
bash scripts-ppo/run_rl_qwen_origin_new_prompt_ddp.sh
```

The PPO implementation's details are in [`scripts-ppo/README.md`](scripts-ppo/README.md).

## Evaluation-Agentsense

The `agentsense_eval/` pipeline evaluates a served policy model against a served partner model and a separate judge model. It runs sequential multi-turn conversations within each item and evaluates multiple items concurrently.

### Start the Three Services

The repository provides separate launchers so the services can use different loopback addresses and GPUs. Set model paths and GPU assignments explicitly:

```bash
JUDGE_MODEL_PATH=/path/to/judge-model \
JUDGE_SERVED_MODEL_NAME=judge-model \
JUDGE_HOST=127.0.0.1 \
JUDGE_PORT=8000 \
JUDGE_CUDA_VISIBLE_DEVICES=0 \
bash agentsense_eval/run_judge_server.sh
```

```bash
PARTNER_MODEL_PATH=/path/to/partner-model \
PARTNER_SERVED_MODEL_NAME=partner-model \
PARTNER_HOST=127.0.0.2 \
PARTNER_PORT=8000 \
PARTNER_CUDA_VISIBLE_DEVICES=1 \
bash agentsense_eval/run_partner_server.sh
```

```bash
POLICY_MODEL_PATH=/path/to/policy-base-model \
POLICY_SERVED_MODEL_NAME=policy-model \
PEFT_CHECKPOINT_PATH=./checkpoints/<run>/checkpoints_final \
POLICY_HOST=127.0.0.3 \
POLICY_PORT=8000 \
POLICY_CUDA_VISIBLE_DEVICES=2 \
bash agentsense_eval/run_policy_server.sh
```

The policy launcher serves a PEFT adapter through vLLM's `--lora-modules` option. The requested policy model name must match `POLICY_SERVED_MODEL_NAME`.

### Run AgentSense

With the three services running in separate terminals, run the Python entry point from the repository root:

```bash
python agentsense_eval/run_agentsense_eval.py \
  --policy-base-url http://127.0.0.3:8000/v1 \
  --policy-model policy-model \
  --partner-base-url http://127.0.0.2:8000/v1 \
  --partner-model partner-model \
  --judge-base-url http://127.0.0.1:8000/v1 \
  --judge-model judge-model \
  --data-path ./data/agentsense_final_data.jsonl \
  --output-dir ./agentsense-output \
  --max-scenes 0 \
  --eval-workers 32
```

## Evaluation-Sotopia


### Start the Three Services

The repository provides separate launchers so the services can use different loopback addresses and GPUs. Set model paths and GPU assignments explicitly:

```bash
JUDGE_MODEL_PATH=/path/to/judge-model \
JUDGE_SERVED_MODEL_NAME=judge-model \
JUDGE_HOST=127.0.0.1 \
JUDGE_PORT=8000 \
JUDGE_CUDA_VISIBLE_DEVICES=0 \
bash scripts-ppo/run_judge_server.sh
```

```bash
PARTNER_MODEL_PATH=/path/to/partner-model \
PARTNER_SERVED_MODEL_NAME=partner-model \
PARTNER_HOST=127.0.0.2 \
PARTNER_PORT=8000 \
PARTNER_CUDA_VISIBLE_DEVICES=1 \
bash scripts-ppo/run_partner_server.sh
```

```bash
POLICY_MODEL_PATH=/path/to/policy-base-model \
POLICY_SERVED_MODEL_NAME=policy-model \
PEFT_CHECKPOINT_PATH=./checkpoints/<run>/checkpoints_final \
POLICY_HOST=127.0.0.3 \
POLICY_PORT=8000 \
POLICY_CUDA_VISIBLE_DEVICES=2 \
bash scripts-ppo/run_policy_server.sh
```

### Run sotopia-pi-benchmark

With the three services running in separate terminals, run the Python entry point from the repository root:

```bash
python scripts-ppo/run_test_only_vllm.sh \
  --policy-base-url http://127.0.0.3:8000/v1 \
  --policy-model policy-model \
  --partner-base-url http://127.0.0.2:8000/v1 \
  --partner-model partner-model \
  --judge-base-url http://127.0.0.1:8000/v1 \
  --judge-model judge-model \
  --data-path ./data/sotopia_backgrounds_clean.jsonl \
```