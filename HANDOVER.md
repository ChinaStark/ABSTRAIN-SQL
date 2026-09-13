# REIN-SQL v3 Handover

## 当前状态

项目是基于 verl 的 Text-to-SQL GRPO/REIN 训练方案，模型输出协议为：

```text
<draft_sql>...</draft_sql>
<reflection>...</reflection>
<answer>...</answer>
```

最终答案可以是 SQL 或 `ABSTAIN`。SQL 正确性通过 SQLite 执行结果比较得到，训练奖励由 `boundary.py` 中的 `WITHIN` / `BEYOND` 规则计算。

## 环境

- Conda 环境：`verl`
- Python：`/zhiliang/conda_env/verl/bin/python`
- verl：`0.7.0.dev0`
- GPU 配置：2 张 H100 80GB

推荐进入项目目录后使用显式 Python 路径，避免误用 base 环境：

```bash
cd /zhiliang/rein_sql_v3_final
```

## 数据

已完成 v3 格式转换：

- 训练集：`/zhiliang/data/train_v3.parquet`，8929 条
- 验证集：`/zhiliang/data/val_v3.parquet`，1534 条
- 训练数据库：`/zhiliang/data/train_databases`
- 验证数据库：`/zhiliang/data/dev_databases`

两份 Parquet 均包含：

```text
messages
data_source
reward_model
extra_info
```

数据库布局要求：

```text
<DB_ROOT>/<db_id>/<db_id>.sqlite
```

## 当前配置

主配置：[configs/rein_sql_v3_h100_2gpu.yaml](configs/rein_sql_v3_h100_2gpu.yaml)

当前关键设置：

- `train_files=/zhiliang/data/train_v3.parquet`
- `val_files=/zhiliang/data/val_v3.parquet`
- `db_root=/zhiliang/data/train_databases`
- `dev_db_root=/zhiliang/data/dev_databases`
- `rollout.n=8`
- `n_correct+n_wrong=8`
- `train_batch_size=4`
- `adaptive_rollout.enable=false`
- rollout tensor parallelism：`1`

当前模型配置指向 HuggingFace cache 根目录。若 verl 无法从该路径识别模型，应改成实际 snapshot 目录：

```text
/zhiliang/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554
```

## 启动入口

[train_rein_sql.py](train_rein_sql.py) 会复用当前 verl 的 `main_ppo`，但将普通 `RayPPOTrainer` 替换为项目中的 `RayREINSQLTrainer`。

先进行小规模冒烟测试：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
PYTHONPATH=/zhiliang/rein_sql_v3_final:/zhiliang/verl \
/zhiliang/conda_env/verl/bin/python train_rein_sql.py \
  --config-path configs \
  --config-name rein_sql_v3_h100_2gpu \
  data.train_max_samples=20 \
  data.val_max_samples=20 \
  trainer.total_epochs=1 \
  trainer.logger='[console]'
```

确认能完成初始化、rollout、SQL reward 和至少一个 actor update 后，再运行完整训练：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
PYTHONPATH=/zhiliang/rein_sql_v3_final:/zhiliang/verl \
/zhiliang/conda_env/verl/bin/python train_rein_sql.py \
  --config-path configs \
  --config-name rein_sql_v3_h100_2gpu
```

## 关键文件

- `rein_sql_trainer.py`：自定义 REIN/GRPO trainer
- `boundary.py`：边界分类和奖励梯度
- `output_protocol.py`：模型输出解析与格式校验
- `sql_reward.py`：verl custom reward 入口
- `exec_sql.py`：可终止 SQLite 执行器
- `build_v3_train_data.py`：原始 Parquet/JSON 转 v3 Parquet
- `train_rein_sql.py`：训练启动入口

## 已知注意点

1. 不要直接运行 `verl.trainer.main_ppo`，否则会实例化普通 `RayPPOTrainer`，不会执行 REIN-SQL 的自定义训练循环。
2. 自定义 reward 配置必须位于顶层 `custom_reward_function`，当前 verl 版本不是旧 README 中的 `reward.custom_reward_function` 结构。
3. `sql_reward.py` 会同时执行 final SQL 和 draft SQL；训练速度可能受 SQLite 子进程评分影响。
4. 首次运行建议保持 fixed rollout，确认数据、模型、数据库和输出协议都正常后，再打开 adaptive rollout。
5. 训练结果默认写入 `checkpoints/rein_sql_v3/rein_sql_v3_h100_2gpu`，具体以 verl 合并后的 trainer 配置为准。

## 交接前检查

```bash
/zhiliang/conda_env/verl/bin/python -m py_compile \
  output_protocol.py boundary.py exec_sql.py sql_reward.py \
  build_v3_train_data.py rein_sql_trainer.py train_rein_sql.py
```

若冒烟测试失败，优先检查：模型 snapshot 路径、Hydra 配置合并结果、`DB_ROOT`/`DEV_DB_ROOT`、Ray GPU 数量，以及 `rollout.n == n_correct + n_wrong`。
