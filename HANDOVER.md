# REIN-SQL v3 Handover

## Current State

This project is adapted to run with the conda environment at:

```bash
/zhiliang/conda_env/verl_dynamic
```

The launcher now uses the environment's installed `verl 0.9.0.dev0` package. It
does not require `/zhiliang/verl` in `PYTHONPATH` and does not modify verl
source code.

The trainer keeps REIN-SQL's boundary-aware Text-to-SQL reward:

```text
<draft_sql>...</draft_sql>
<reflection>...</reflection>
<answer>SQL or ABSTAIN</answer>
```

For each prompt, the trainer generates exactly `actor_rollout_ref.rollout.n`
trajectories. If any final SQL is execution-correct, the prompt is classified as
`WITHIN`; otherwise it is `BEYOND`. All generated trajectories remain in the
original GRPO group. There is no adaptive sampling stage.

## Data And Model

- Train parquet: `/zhiliang/data/train_v3.parquet`
- Val parquet: `/zhiliang/data/val_v3.parquet`
- Train DB root: `/zhiliang/data/train_databases`
- Dev DB root: `/zhiliang/data/dev_databases`
- Model snapshot:
  `/zhiliang/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554`

The parquet files use `messages` as the prompt column and include
`data_source`, `reward_model`, and `extra_info`.

## Main Files

- `train_rein_sql.py`: verl 0.9 launcher using `BaseTaskRunner` and `run_ppo`
- `rein_sql_trainer.py`: custom fixed-rollout REIN-SQL trainer
- `configs/rein_sql_v3_h100_1gpu.yaml`: 1 x H100 config
- `configs/rein_sql_v3_h100_2gpu.yaml`: 2 x H100 config
- `boundary.py`: WITHIN/BEYOND reward ladder
- `output_protocol.py`: output parser and format checks
- `sql_reward.py`: verl custom reward entry point
- `exec_sql.py`: killable SQLite execution checker

## Run

Use the project directory plus the conda env's Python:

```bash
cd /zhiliang/rein_sql_v3_final
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/zhiliang/rein_sql_v3_final \
/zhiliang/conda_env/verl_dynamic/bin/python train_rein_sql.py \
  --config-name rein_sql_v3_h100_1gpu
```

For 2 GPUs:

```bash
cd /zhiliang/rein_sql_v3_final
CUDA_VISIBLE_DEVICES=0,1 \
PYTHONPATH=/zhiliang/rein_sql_v3_final \
/zhiliang/conda_env/verl_dynamic/bin/python train_rein_sql.py \
  --config-name rein_sql_v3_h100_2gpu
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

```bash
cd /zhiliang/rein_sql_v3_final
PYTHONPATH=/zhiliang/rein_sql_v3_final \
/zhiliang/conda_env/verl_dynamic/bin/python -m py_compile \
  output_protocol.py boundary.py exec_sql.py sql_reward.py \
  build_v3_train_data.py rein_sql_trainer.py train_rein_sql.py
```

The config uses `reward.custom_reward_function`, which is the verl 0.9 layout.
Keep `trainer.use_v1=false` because this project subclasses the legacy
`RayPPOTrainer` path exposed by verl 0.9.
