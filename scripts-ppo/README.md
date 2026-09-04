# PPO Training

This directory implements actor-critic PPO for goal-oriented social dialogue. A local LoRA policy model interacts with a partner model served through an OpenAI-compatible endpoint. A separate critic estimates state values, while reward and judge models provide process and terminal feedback.

## Reward

PPO uses the six-dimensional `rubric_dynamic` process reward. Every policy turn is evaluated through two to four binary rubric items per dimension. Pass rates are aggregated with role- and phase-aware weights into a process reward in `[0, 1]`.

| Dimension | Meaning |
| --- | --- |
| `goal_advancement` | Advances the policy character's own social goal. |
| `strategic_positioning` | Uses an appropriate strategy for the current interaction. |
| `relational_attunement` | Handles emotion, dignity, and relationship tension well. |
| `persona_consistency` | Remains faithful to the assigned character. |
| `contextual_coherence` | Follows the dialogue naturally and relevantly. |
| `turn_quality` | Produces a concise, natural, stage-appropriate reply. |

The process reward and binary terminal goal reward are shaped as:

```text
reward[t] = PROCESS_REWARD_WEIGHT * rubric_reward[t]
reward[last] += GOAL_REWARD_WEIGHT * terminal_goal_reward
```

With `TRAJECTORY_NORMALIZE=1`, each per-turn process reward is divided by `EXPECTED_HORIZON` to discourage extending a dialogue only to accumulate reward. PPO uses GAE and can add a reference-policy KL penalty through `KL_BETA`.

## Quick Start

Before running a launcher, update the machine-specific configuration block near the top of the corresponding shell script. Set local model paths, data paths, served model names, service URLs, and GPU assignments. One vLLM service can provide the partner, reward, and judge roles:

```bash
bash scripts-ppo/run_vllm_service.sh
```

After configuring `scripts-ppo/run_rl_qwen_origin_new_prompt_ddp.sh`, launch distributed training from the repository root:

```bash
bash scripts-ppo/run_rl_qwen_origin_new_prompt_ddp.sh
```

The wrapper launches `torch.distributed.run` with NCCL. It derives the number of policy processes from `POLICY_GPU_IDS`; set `NPROC_PER_NODE` to override it. Use `python scripts-ppo/rl_qwen_origin_new_prompt_ddp.py --help` for all command-line arguments.

### Resume and Test

To resume, set `RESUME_FROM` to a checkpoint with LoRA settings compatible with the current run in the training launcher configuration block:

```bash
bash scripts-ppo/run_rl_qwen_origin_new_prompt_ddp.sh
```

The PPO directory also includes `rl_test_only_vllm.py` for evaluating a policy served by vLLM. Run `python scripts-ppo/rl_test_only_vllm.py --help` and provide the policy, partner, judge, data, and output settings explicitly.

## Configuration

| Variable | Purpose |
| --- | --- |
| `MODEL_PATH` | Local policy base model. |
| `CRITIC_MODEL_PATH` | Local critic base model; falls back to `MODEL_PATH` when unset. |
| `SAVE_PATH` | Root directory for timestamped checkpoints. |
| `BACKGROUNDS_PATH` | Training-scene JSONL file. |
| `CUDA_VISIBLE_DEVICES` | GPUs visible to the training process. |
| `POLICY_GPU_IDS` | GPU IDs allocated to DDP policy workers. |
| `CRITIC_DEVICE_MAP` | Devices used by the critic. |
| `PARTNER_*`, `REWARD_*`, `JUDGE_*` | Endpoint, served model name, and API key for each service role. |
| `PROCESS_REWARD_WEIGHT` | Scale for per-turn rubric rewards. |
| `GOAL_REWARD_WEIGHT` | Scale for the final goal reward. |
| `TRAJECTORY_NORMALIZE` | Divide process rewards by `EXPECTED_HORIZON`. |
| `EXPECTED_HORIZON` | Expected policy-turn count used for normalization and phase scheduling. |
| `DYNAMIC_WEIGHTS_FROM_JUDGE` | Enable judge-proposed rubric weights. |
| `DYNAMIC_WEIGHTS_MIX_RATIO` | Judge-weight contribution from `0` to `1`. |
| `KL_BETA` | Reference-policy KL penalty coefficient. |
| `WANDB` | Set to `1` to enable Weights & Biases logging. |

The launcher's current workload and optimizer defaults, including epochs, batch sizes, LoRA targets, PPO epochs, learning rates, and token limit, are defined in `run_rl_qwen_origin_new_prompt_ddp.sh`. Adjust them before a full run to match available compute.

## Outputs

Each run creates a timestamped directory under `SAVE_PATH` containing training metrics, dialogue histories, test histories, epoch checkpoints, and `checkpoints_final`. LoRA checkpoints include `adapter_model.safetensors`, adapter configuration, tokenizer files, and `train_state.json` for resumption.

Test histories can include `relation_delta` scores in `[-1, 1]`, representing the judge-estimated change in the partner's favorability toward the policy character. Use `score_relation_delta.py` to re-score existing test histories.

## Notes

- Role-swap expansion lets the policy play both characters in each two-person training scene.
- The `[LEAVE]` stop token ends a dialogue rollout early.
- Judge-produced rewards are model-dependent and should be audited through the saved JSONL histories.
- The shell scripts contain machine-specific defaults. Update their configuration blocks with model paths, data paths, service endpoints, and GPU assignments before running them.
