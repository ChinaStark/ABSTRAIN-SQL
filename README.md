# REIN-SQL v3: Boundary-Aware Text-to-SQL GRPO

This project trains a Text-to-SQL model with a fixed-rollout REIN-SQL reward on
top of the `verl 0.9.0.dev0` package installed in:

```bash
/zhiliang/conda_env/verl_dynamic
```

It does not require editing verl source code and does not require adding
`/zhiliang/verl` to `PYTHONPATH`.

## What It Does

The model must generate:

```text
<draft_sql>...</draft_sql>
<reflection>...</reflection>
<answer>SQL or ABSTAIN</answer>
```

For each prompt, the trainer generates exactly
`actor_rollout_ref.rollout.n` trajectories. If any final SQL execution matches
the gold SQL, the prompt is classified as `WITHIN`; if all final SQL executions
are wrong, it is classified as `BEYOND`.

All generated trajectories remain in their original GRPO group. The trainer does
not do adaptive sampling, online filtering, or correct/wrong quota selection.

## Main Files

- `train_rein_sql.py`: verl 0.9 launcher using `BaseTaskRunner` and `run_ppo`
- `rein_sql_trainer.py`: custom fixed-rollout REIN-SQL trainer
- `boundary.py`: `WITHIN` / `BEYOND` reward values and reward calculation
- `output_protocol.py`: response parser and format validation
- `sql_reward.py`: verl custom reward entry point
- `exec_sql.py`: killable SQLite execution checker
- `configs/rein_sql_v3_h100_1gpu.yaml`: 1 x H100 config
- `configs/rein_sql_v3_h100_2gpu.yaml`: 2 x H100 config

## Data And Model

Current paths used by the configs:

```text
train parquet : /zhiliang/data/train_v3.parquet
val parquet   : /zhiliang/data/val_v3.parquet
train DB root : /zhiliang/data/train_databases
dev DB root   : /zhiliang/data/dev_databases
model         : /zhiliang/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554
```

The parquet prompt column is `messages`. Rows also need `data_source`,
`reward_model`, and `extra_info`. Database files should follow:

```text
<DB_ROOT>/<db_id>/<db_id>.sqlite
```

## Reward Shape

The trainer first checks whether the response has the required sections. Full
format receives the base format reward, then SQL correctness and abstention are
scored according to the boundary:

```text
WITHIN : reward correct SQL, penalize ABSTAIN
BEYOND : reward ABSTAIN, give wrong SQL no extra correctness reward
```

The current ladder is configured under
`algorithm.adaptive_rollout.reward_values` in the YAML files. The H100 configs
disable abstention warmup and use a cap of eight, equal to the fixed rollout
group size, so abstention reward is fully active from the first training step
and no rollout is excluded by a cap. SQL execution arguments are under
`reward.custom_reward_function.reward_kwargs`, which is the verl 0.9 config
layout.

## Run

1 GPU:

```bash
cd /zhiliang/rein_sql_v3_final
./train_h100.sh
```

The single-H100 launcher starts fresh (`resume_mode=disable`), samples at
temperature 1.0, and writes checkpoints and rollout data to directories named
after `RUN_NAME`.

2 GPUs:

```bash
cd /zhiliang/rein_sql_v3_final
CUDA_VISIBLE_DEVICES=0,1 \
PYTHONPATH=/zhiliang/rein_sql_v3_final \
/zhiliang/conda_env/verl_dynamic/bin/python train_rein_sql.py \
  --config-name rein_sql_v3_h100_2gpu
```

2 x 48GB GPUs (conservative memory profile):

```bash
cd /zhiliang/rein_sql_v3_final
./train_2gpu_48gb.sh
```

Small smoke test:

```bash
cd /zhiliang/rein_sql_v3_final
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/zhiliang/rein_sql_v3_final \
/zhiliang/conda_env/verl_dynamic/bin/python train_rein_sql.py \
  --config-name rein_sql_v3_h100_1gpu \
  data.train_batch_size=1 \
  data.gen_batch_size=1 \
  data.train_max_samples=1 \
  data.val_max_samples=1 \
  actor_rollout_ref.rollout.n=2 \
  actor_rollout_ref.rollout.agent.num_workers=1 \
  actor_rollout_ref.actor.ppo_mini_batch_size=2 \
  trainer.total_epochs=1 \
  trainer.total_training_steps=1 \
  trainer.val_before_train=false \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.logger='[console]'
```

## Checks

Compile project files against the env's installed verl:

```bash
cd /zhiliang/rein_sql_v3_final
PYTHONPATH=/zhiliang/rein_sql_v3_final \
/zhiliang/conda_env/verl_dynamic/bin/python -m py_compile \
  output_protocol.py boundary.py exec_sql.py sql_reward.py \
  build_v3_train_data.py rein_sql_trainer.py train_rein_sql.py
```

Confirm which verl is being imported:

```bash
PYTHONPATH=/zhiliang/rein_sql_v3_final \
/zhiliang/conda_env/verl_dynamic/bin/python - <<'PY'
import verl
print(verl.__version__)
print(verl.__file__)
PY
```

Expected result: `0.9.0.dev0` from the conda env's `site-packages`.

## Notes

- Keep `trainer.use_v1=false`; the custom trainer subclasses the legacy
  `RayPPOTrainer` path still exposed by verl 0.9.
- `actor_rollout_ref.actor.use_dynamic_bsz` and log-prob dynamic batch settings
  are verl batching controls, not adaptive sampling.
- Rollout text samples are logged best-effort to `rollouts.log` next to
  `trainer.logger_path`.
