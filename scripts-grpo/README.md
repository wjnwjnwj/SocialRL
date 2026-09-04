# Process-GRPO Training

This directory implements Process-GRPO for goal-oriented social dialogue. A local LoRA policy model interacts with a partner model served through an OpenAI-compatible endpoint. A reward model scores each policy turn and the completed dialogue; the resulting rewards drive group-based policy optimization.

## Reward

Every policy turn receives a dynamic rubric reward in `[0, 1]`. The judge creates two to four binary items for each dimension, computes the pass rate per dimension, and combines them with role- and phase-aware weights.

| Dimension | Meaning |
| --- | --- |
| `goal_advancement` | Advances the policy character's own social goal. |
| `strategic_positioning` | Uses an appropriate strategy for the current interaction. |
| `relational_attunement` | Handles emotion, dignity, and relationship tension well. |
| `persona_consistency` | Remains faithful to the assigned character. |
| `contextual_coherence` | Follows the dialogue naturally and relevantly. |
| `turn_quality` | Produces a concise, natural, stage-appropriate reply. |

The last policy turn also receives a continuous terminal goal-achievement reward. Process-GRPO shapes each rollout as:

```text
reward[t] = PROCESS_REWARD_WEIGHT * rubric_reward[t]
reward[last] += GOAL_REWARD_WEIGHT * terminal_goal_reward
```

Rewards are globally standardized across all group-turn pairs in a batch, then converted into backward-cumulative discounted advantages. Set `DYNAMIC_WEIGHTS_FROM_JUDGE=1` to blend judge-proposed dimension weights with the stable role-family and dialogue-phase schedule.

## Quick Start

Before running a launcher, update the machine-specific configuration block near the top of the corresponding shell script. Set local model paths, data paths, served model names, service URLs, and GPU assignments. One vLLM service can provide the partner, reward, and judge roles:

```bash
bash scripts-grpo/run_vllm_service.sh
```

After configuring `scripts-grpo/run_rl_qwen_origin_new_prompt_ddp.sh`, launch distributed training from the repository root:

```bash
bash scripts-grpo/run_rl_qwen_origin_new_prompt_ddp.sh
```

The wrapper launches `torch.distributed.run` with NCCL. It derives the number of processes from `CUDA_VISIBLE_DEVICES`; set `NPROC_PER_NODE` to override it. Use `python scripts-grpo/rl_qwen_origin_new_prompt_ddp.py --help` for all command-line arguments.

### Resume and Test

To resume, set `RESUME_FROM` to a compatible adapter checkpoint in the training launcher configuration block:

```bash
bash scripts-grpo/run_rl_qwen_origin_new_prompt_ddp.sh
```

Evaluate an adapter without training:

```bash
python scripts-grpo/rl_test_only.py \
  --policy-model-path /path/to/policy-model \
  --peft-checkpoint-path ./checkpoints/<run>/checkpoints_final \
  --backgrounds ./data/sotopia_backgrounds_clean.jsonl \
  --partner-base-url http://127.0.0.1:8000/v1 \
  --partner-model large-model \
  --judge-base-url http://127.0.0.1:8000/v1 \
  --judge-model large-model \
  --output-dir ./test-results
```

## Configuration

| Variable | Purpose |
| --- | --- |
| `MODEL_PATH` | Local policy base model. |
| `SAVE_PATH` | Root directory for timestamped checkpoints. |
| `BACKGROUNDS_PATH` | Training-scene JSONL file. |
| `CUDA_VISIBLE_DEVICES` | GPUs used by DDP training. |
| `PARTNER_*`, `REWARD_*`, `JUDGE_*` | Endpoint, served model name, and API key for each service role. |
| `PROCESS_REWARD_WEIGHT` | Scale for per-turn rubric rewards. |
| `GOAL_REWARD_WEIGHT` | Scale for the final goal reward. |
| `GAMMA` | Discount factor for advantages. |
| `EXPECTED_HORIZON` | Expected policy-turn count used for reward-phase scheduling. |
| `DYNAMIC_WEIGHTS_FROM_JUDGE` | Enable judge-proposed rubric weights. |
| `DYNAMIC_WEIGHTS_MIX_RATIO` | Judge-weight contribution from `0` to `1`. |
| `VAL_RATIO` | Held-out scene fraction. |
| `WANDB` | Set to `1` to enable Weights & Biases logging. |

The launcher's current workload defaults, including epochs, group count, scene batch size, and token limit, are defined in `run_rl_qwen_origin_new_prompt_ddp.sh`. Adjust them before a full run to match available compute.

## Outputs

Each run creates a timestamped directory under `SAVE_PATH` containing training metrics, dialogue histories, test histories, epoch checkpoints, and `checkpoints_final`. LoRA checkpoints include `adapter_model.safetensors`, adapter configuration, tokenizer files, and `train_state.json` for resumption.

## Notes

- Role-swap expansion lets the policy play both characters in each two-person training scene.
- The `[LEAVE]` stop token ends a dialogue rollout early.
- Judge-produced rewards are model-dependent and should be audited through the saved JSONL histories.
- The shell scripts contain machine-specific defaults. Update their configuration blocks with model paths, data paths, service endpoints, and GPU assignments before running them.
