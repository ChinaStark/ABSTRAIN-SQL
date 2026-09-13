# REIN-SQL v3: Boundary-Aware Text-to-SQL GRPO

This directory packages the v3 Text-to-SQL RL setup with:

- draft SQL -> reflection -> final SQL / `ABSTAIN` protocol;
- execution-based SQL verification on SQLite/BIRD-style databases;
- `WITHIN` / `BEYOND` empirical capability boundary;
- boundary-aware reward shaping;
- fixed or adaptive GRPO rollout collection;
- a conservative **2 x H100 80GB** training configuration.

The default GPU configuration targets a **Qwen3-4B-class model**. For an 8B-class
model, start by changing rollout tensor parallelism from 1 to 2.

---

## 1. Files

```text
rein_sql_v3_h100/
├── README.md
├── boundary.py
├── output_protocol.py
├── sql_reward.py
├── exec_sql.py
├── rein_sql_trainer.py
├── build_v3_train_data.py
├── requirements.txt
├── configs/
│   └── rein_sql_v3_h100_2gpu.yaml
└── examples/
    ├── raw_input.json
    └── valid_model_response.txt
```

### Main components

`rein_sql_trainer.py`
: Custom `RayPPOTrainer` subclass. It generates and scores final/draft SQL,
classifies prompts as `WITHIN` or `BEYOND`, constructs GRPO groups, caches
trajectory rewards, computes GRPO advantages, and updates the actor.

`boundary.py`
: Defines the reward values and `trajectory_reward()`.

`output_protocol.py`
: Parses and strictly validates `<draft_sql>`, `<reflection>`, and `<answer>`.

`sql_reward.py`
: verl custom reward entry point. It executes the predicted SQL and returns
execution correctness.

`exec_sql.py`
: Killable SQLite subprocess executor used by `sql_reward.py`.

`build_v3_train_data.py`
: Converts older JSON/parquet prompts into the v3 chat-message parquet format.

`configs/rein_sql_v3_h100_2gpu.yaml`
: 2 x H100 80GB baseline.

---

## 2. Required model output protocol

Every rollout must have exactly these three top-level sections:

```text
<draft_sql>
```sql
SELECT ...;
```
</draft_sql>

<reflection>
Non-empty reflection.
</reflection>

<answer>
```sql
SELECT ...;
```
</answer>
```

or, for abstention:

```text
<draft_sql>
```sql
SELECT ...;
```
</draft_sql>

<reflection>
Non-empty reflection.
</reflection>

<answer>
ABSTAIN
</answer>
```

Rules:

1. `<draft_sql>` must contain exactly one `sql` Markdown code block.
2. `<reflection>` must be non-empty.
3. `<answer>` must contain either:
   - exactly one `sql` Markdown code block; or
   - exactly `ABSTAIN`.
4. No text is allowed outside these three tags.

See `examples/valid_model_response.txt`.

---

## 3. Reward ladder

When the complete three-section format is present:

| Case | Total reward |
|---|---:|
| Final SQL correct + draft SQL correct | 1.2 |
| Final SQL correct + draft SQL wrong | 1.0 |
| `ABSTAIN` + `BEYOND` | 0.6 |
| Final SQL wrong + draft SQL correct | 0.4 |
| Final SQL wrong + draft SQL wrong | 0.3 |
| `ABSTAIN` + `WITHIN` | 0.3 |

Partial format:

| Present required sections | Reward |
|---:|---:|
| 2 | 0.2 |
| 1 | 0.1 |
| 0 | 0.0 |

The `0.1` signal for "draft correct but final wrong" is intentionally small but
prevents an all-wrong GRPO group from becoming completely zero-variance.

---

## 4. Capability boundary

For a prompt, execution correctness is used as the empirical verifier.

```text
at least one correct probe rollout -> WITHIN
all probe rollouts wrong          -> BEYOND
```

Default group/probe settings:

```yaml
n_correct: 4
n_wrong: 4
probe_n: 8
rollout.n: 8
```

Therefore:

```text
group_size = n_correct + n_wrong = 8
```

`actor_rollout_ref.rollout.n` must remain equal to this group size.

### Fixed mode

The H100 baseline starts with:

```yaml
algorithm:
  adaptive_rollout:
    enable: false
```

Every prompt receives exactly 8 rollouts. This is the recommended first step for
checking the data, reward, SQLite paths, memory usage, and two-GPU training.

### Adaptive mode

After the pipeline is stable:

```yaml
algorithm:
  adaptive_rollout:
    enable: true
    probe_n: 8
    micro_rollout_n: 8
    b_max: 32
```

This permits up to four 8-sample generation rounds for a prompt.

Do **not** use `probe_n=8` together with `b_max=8` if you expect additional
adaptive collection: the complete generation budget would already be consumed by
the probe.

---

# 5. Training input data

The final training/validation files supplied to verl are **Parquet** files.

The provided builder writes the prompt into a column called:

```text
messages
```

Therefore the training YAML explicitly sets:

```yaml
data:
  prompt_key: messages
```

This is important because standard verl RL configs commonly default to a prompt
column named `prompt`.

---

## 5.1 Final parquet schema

Each row should conceptually look like:

```python
{
    "messages": [
        {
            "role": "user",
            "content": "<schema + question + v3 instructions>"
        }
    ],

    "data_source": "bird_sql_v3",

    "reward_model": {
        "ground_truth": {
            "db_id": "california_schools",
            "sql": "SELECT ..."
        }
    },

    "extra_info": {
        "db_id": "california_schools",
        "question": "Natural-language question",
        "idx": 123,
        "difficulty": "moderate",
        "evidence": ""
    }
}
```

### Required fields

#### `messages`

A HuggingFace-style chat message list.

Minimum:

```python
[
    {
        "role": "user",
        "content": "..."
    }
]
```

The model chat template is applied by the verl dataset/tokenizer path. Do not
manually store `<|im_start|>` / `<|im_end|>` tokens in the final prompt.

#### `data_source`

Use:

```text
bird_sql_v3
```

It identifies the task/reward source.

#### `reward_model.ground_truth.db_id`

Database identifier.

Example:

```text
california_schools
```

#### `reward_model.ground_truth.sql`

Gold SQL used by the execution verifier.

Example:

```sql
SELECT COUNT(*) FROM schools;
```

#### `extra_info.db_id`

Should match the `db_id` in `reward_model.ground_truth`.

The SQL reward uses it to locate the SQLite database.

### Optional metadata

These are useful for debugging/logging but are not required for SQL execution:

```text
extra_info.question
extra_info.idx
extra_info.difficulty
extra_info.evidence
```

---

# 6. Raw data accepted by the builder

`build_v3_train_data.py` accepts either JSON or an older parquet.

## 6.1 Raw JSON format

Example:

```json
[
  {
    "idx": 0,
    "db_id": "example_db",
    "question": "How many users are active?",
    "gold_sql": "SELECT COUNT(*) FROM users WHERE active = 1;",
    "prompt": "Database Schema:\nusers(id INTEGER, name TEXT, active INTEGER)\n\nQuestion: How many users are active?\n\nInstructions:\nOld instructions",
    "difficulty": "simple",
    "evidence": ""
  }
]
```

Required for training correctness:

```text
prompt
db_id
gold_sql
```

The builder preserves `question`, `idx`, `difficulty`, and `evidence` when
available.

The raw prompt must contain one of these markers so that the old instruction
section can be replaced:

```text
Instructions:
Take a deep breath
Output Format
```

Normally `Instructions:` should be present.

---

## 6.2 Older parquet format

The builder also accepts an existing parquet containing:

```text
prompt
reward_model
extra_info
```

`prompt` may be:

- a plain string; or
- a chat-message list.

Expected nested metadata:

```python
reward_model = {
    "ground_truth": {
        "db_id": "...",
        "sql": "..."
    }
}

extra_info = {
    "db_id": "...",
    "question": "...",
    "index": 0,
    "difficulty": "...",
    "evidence": "..."
}
```

`ground_truth.query` is also accepted as a fallback for `ground_truth.sql`.

---

# 7. Build the v3 parquet

From JSON:

```bash
python3 build_v3_train_data.py \
  examples/raw_input.json \
  data/train_v3.parquet
```

From an older parquet:

```bash
python3 build_v3_train_data.py \
  data/all_merged_reflection_v2.parquet \
  data/train_v3.parquet
```

Build validation data in the same way:

```bash
python3 build_v3_train_data.py \
  data/dev_old.parquet \
  data/val_v3.parquet
```

Then inspect a row:

```python
import pandas as pd
import pprint

df = pd.read_parquet("data/train_v3.parquet")

print(df.columns)
pprint.pp(df.iloc[0]["messages"])
pprint.pp(df.iloc[0]["reward_model"])
pprint.pp(df.iloc[0]["extra_info"])
```

You should see these top-level columns:

```text
messages
data_source
reward_model
extra_info
```

---

# 8. SQLite database layout

The execution reward expects this directory structure:

```text
DB_ROOT/
├── database_a/
│   └── database_a.sqlite
├── database_b/
│   └── database_b.sqlite
└── ...
```

For example:

```text
/data/bird/train_databases/
└── california_schools/
    └── california_schools.sqlite
```

Validation databases can be in another root:

```text
DEV_DB_ROOT/
└── dev_database/
    └── dev_database.sqlite
```

The lookup logic is:

```text
DB_ROOT/<db_id>/<db_id>.sqlite
```

and if it is not found there:

```text
DEV_DB_ROOT/<db_id>/<db_id>.sqlite
```

---

# 9. Environment variables

Example:

```bash
export DB_ROOT=/data/bird/train_databases
export DEV_DB_ROOT=/data/bird/dev_databases
export EXEC_TIMEOUT=30

export CUDA_VISIBLE_DEVICES=0,1
```

You can alternatively place `db_root` and `dev_db_root` directly in the YAML:

```yaml
reward:
  custom_reward_function:
    reward_kwargs:
      db_root: /data/bird/train_databases
      dev_db_root: /data/bird/dev_databases
      exec_timeout: 30.0
```

---

# 10. 2 x H100 80GB baseline

Edit:

```text
configs/rein_sql_v3_h100_2gpu.yaml
```

At minimum change:

```yaml
data:
  train_files: /absolute/path/to/train_v3.parquet
  val_files: /absolute/path/to/val_v3.parquet

actor_rollout_ref:
  model:
    path: /absolute/path/to/your/model
```

Default 4B settings:

```yaml
trainer:
  nnodes: 1
  n_gpus_per_node: 2

data:
  train_batch_size: 4

actor_rollout_ref:
  actor:
    use_dynamic_bsz: true
    ppo_mini_batch_size: 16
    ppo_max_token_len_per_gpu: 8192

  rollout:
    n: 8
    tensor_model_parallel_size: 1
    gpu_memory_utilization: 0.55
    max_num_batched_tokens: 8192
    max_num_seqs: 64
```

With:

```text
train_batch_size = 4 prompts
rollout.n        = 8 trajectories/prompt
```

the policy update contains:

```text
4 x 8 = 32 trajectories
```

This is a conservative first configuration for two H100 80GB GPUs.

---

## 10.1 Qwen3-4B

Start with:

```yaml
actor_rollout_ref:
  rollout:
    tensor_model_parallel_size: 1
    gpu_memory_utilization: 0.55
```

With two available GPUs and TP=1, the rollout pool can use independent replicas
rather than forcing one request across both GPUs.

---

## 10.2 Qwen3-8B

First try:

```yaml
actor_rollout_ref:
  model:
    path: Qwen/Qwen3-8B

  rollout:
    tensor_model_parallel_size: 2
    gpu_memory_utilization: 0.55
```

After confirming memory headroom, TP=1 can also be benchmarked for higher
multi-request throughput.

---

# 11. Recommended tuning order

First make one complete step run reliably.

Then increase utilization in this order:

```text
1. gpu_memory_utilization
   0.55 -> 0.60 -> 0.65

2. ppo_max_token_len_per_gpu
   8192 -> 12288 -> 16384

3. data.train_batch_size
   4 -> 8

4. adaptive_rollout.enable
   false -> true
```

If OOM occurs, reverse the direction:

```text
gpu_memory_utilization down
        ->
train_batch_size down
        ->
rollout TP 1 -> 2
        ->
optimizer offload
        ->
parameter offload
```

Do not increase every knob at once; otherwise it becomes hard to identify the
actual memory bottleneck.

---

# 12. SQL reward execution cost

Each trajectory may execute both:

```text
final SQL
draft SQL
```

For:

```text
train_batch_size = 4
rollout.n = 8
```

there can be:

```text
32 trajectories x 2 SQL checks = 64 SQL checks
```

The custom trainer currently executes scoring with up to 16 worker threads.

On H100, GPU generation can become fast enough that SQLite/subprocess reward
evaluation is the training bottleneck. Watch for the pattern:

```text
GPU high utilization during rollout
-> GPU mostly idle
-> CPU / SQLite busy
-> next rollout starts
```

If this happens, optimize the verifier rather than increasing GPU batch size.

Possible later optimizations:

1. cache the gold SQL execution result for `(db_id, gold_sql)`;
2. keep databases on local NVMe;
3. benchmark 16 vs 32 scoring workers;
4. use the exact official BIRD evaluator if leaderboard equivalence is required.

---

# 13. Important note about `exec_sql.py`

The supplied `exec_sql.py` is a lightweight SQLite result comparator suitable for
training experiments.

It:

- opens the database read-only;
- executes prediction and gold SQL;
- compares normalized result rows;
- ignores row order by sorting result rows.

If you need exact equivalence with a specific BIRD leaderboard/evaluation release,
replace `exec_sql.py` with the exact evaluator from that release. Do not assume a
small local comparator is definitionally identical to every BIRD evaluation
version.

---

# 14. Trainer integration

`rein_sql_trainer.py` defines:

```python
RayREINSQLTrainer
```

It is a custom replacement/subclass of verl's `RayPPOTrainer`.

Use the same custom launcher/entry point you currently use to instantiate this
trainer, and point Hydra at:

```text
configs/rein_sql_v3_h100_2gpu.yaml
```

The exact main/launcher file depends on the verl revision in your training image,
so this package intentionally does not invent a version-specific replacement for
`verl.trainer.main_ppo`.

The custom trainer expects these files importable from the working directory or
`PYTHONPATH`:

```text
boundary.py
output_protocol.py
sql_reward.py
```

A simple setup is:

```bash
cd /path/to/rein_sql_v3_h100
export PYTHONPATH="$PWD:$PYTHONPATH"
```

Then run your existing REIN-SQL launcher with the H100 config.

---

# 15. Pre-flight checklist

Before a real run, verify all of the following:

```text
[ ] n_correct + n_wrong == rollout.n == 8
[ ] data.prompt_key == messages
[ ] train and val parquet files exist
[ ] reward_model.ground_truth.db_id exists
[ ] reward_model.ground_truth.sql exists
[ ] extra_info.db_id matches ground_truth.db_id
[ ] DB_ROOT/<db_id>/<db_id>.sqlite exists
[ ] CUDA_VISIBLE_DEVICES=0,1
[ ] trainer.n_gpus_per_node=2
[ ] model path exists / is downloadable
[ ] one sample response passes output_protocol.parse_response(...)
[ ] one gold SQL can be executed by exec_sql.py
```

Quick parser check:

```bash
python3 - <<'PY'
from pathlib import Path
from output_protocol import parse_response

text = Path("examples/valid_model_response.txt").read_text()
print(parse_response(text))
PY
```

Quick SQL executor check:

```bash
echo '{
  "db_file": "/path/to/example_db.sqlite",
  "gold_sql": "SELECT COUNT(*) FROM users WHERE active = 1;",
  "pred_sqls": ["SELECT COUNT(*) FROM users WHERE active = 1;"]
}' | python3 exec_sql.py
```

Expected:

```json
{"any_correct": true, "results": [true]}
```

---

# 16. One recommended first experiment

For a Qwen3-4B-class actor on 2 x H100 80GB:

```text
adaptive rollout   = false
train_batch_size   = 4
rollout.n          = 8
rollout TP         = 1
gpu mem util       = 0.55
prompt max         = 6144
response max       = 2048
dynamic batching   = on
actor offload      = off
KL reward/loss     = off
```

Run a small subset first (for example 20-100 prompts) and inspect:

```text
format-valid rate
ABSTAIN rate
final SQL execution accuracy
draft SQL execution accuracy
WITHIN/BEYOND counts
reward distribution
response lengths
rollout time
SQL scoring time
GPU memory peak
GPU utilization
```

Only after these are sane should you enable adaptive rollout or increase the
batch/token budgets.
