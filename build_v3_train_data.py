#!/usr/bin/env python3
"""Build v3 REIN Text-to-SQL parquet data from JSON or an older parquet.

Supported raw JSON item:
{
  "prompt": "...existing schema/question prompt...",
  "db_id": "california_schools",
  "gold_sql": "SELECT ...",
  "question": "...",
  "idx": 0
}

Supported older parquet:
- prompt: string OR chat-message list
- reward_model.ground_truth.db_id/sql
- extra_info.db_id/question/index/question_id/difficulty/evidence

Output parquet columns:
- messages
- data_source
- reward_model
- extra_info
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import pandas as pd


V3_INSTRUCTIONS = """- Carefully reason about the database schema, required tables and columns, join paths, predicates, aggregation, grouping, ordering, and nesting before making the final decision.
- The goal is not to guess a SQL query at all costs. If you cannot solve the query reliably, you should explicitly abstain instead of producing a plausible but unreliable SQL query.
- You must first produce one candidate SQL query, then critically inspect that candidate, and finally decide whether to output a corrected final SQL query or ABSTAIN.
- The reflection should focus on whether the draft SQL truly answers the question and whether there are mistakes in table/column selection, joins, predicates, aggregation, grouping, ordering, nesting, or subqueries.
- If the draft SQL is incorrect but you can identify and fix the problem, output the corrected SQL as the final answer.
- If you still cannot determine a reliable SQL query after reflection, output exactly ABSTAIN as the final answer."""

V3_OUTPUT_FORMAT = """Output Format:
You must follow exactly the following structure:

<draft_sql>
```sql
-- Write exactly one draft SQL query here.
```
</draft_sql>

<reflection>
Critically inspect the draft SQL.

Check whether it correctly answers the question and whether there are mistakes involving:
- table or column selection;
- join paths and join conditions;
- predicates and filtering conditions;
- aggregation and GROUP BY;
- ordering;
- nested queries or set operations;
- any other SQL logic.

Explain any issue you find and how it should be corrected before making the final decision.
</reflection>

<answer>
Output ONLY one of the following:

1. A final SQL query enclosed in a SQL Markdown code block:

```sql
-- Your SQL
```

2. Or exactly:

ABSTAIN
</answer>

Important:
- <draft_sql> must always contain exactly one SQL attempt enclosed in a ```sql ... ``` Markdown code block.
- <reflection> must be non-empty.
- If <answer> contains SQL, the SQL must be enclosed in a ```sql ... ``` Markdown code block.
- If the final decision is abstention, the stripped textual content inside <answer> must be exactly ABSTAIN.
- ABSTAIN must NOT be enclosed in a Markdown code block.
- Do not put both SQL and ABSTAIN inside <answer>.
- Do not output any content outside <draft_sql>, <reflection>, and <answer>."""


def strip_chat_template(text: str) -> str:
    text = re.sub(r"<\|im_start\|>\s*user\s*\n?", "", text)
    text = re.sub(r"<\|im_end\|>\s*", "", text)
    text = re.sub(r"<\|im_start\|>\s*assistant\s*\n?", "", text)
    return text.strip()


def replace_prompt(original_prompt: str) -> str | None:
    prompt = strip_chat_template(original_prompt)

    idx = prompt.find("Instructions:")
    if idx == -1:
        for marker in ("Take a deep breath", "Output Format"):
            idx = prompt.find(marker)
            if idx != -1:
                break
    if idx == -1:
        return None

    base = prompt[:idx].rstrip()
    deep_breath = (
        "Take a deep breath and think step by step to find the correct SQL query."
    )

    return (
        base
        + "\n\n"
        + V3_INSTRUCTIONS
        + "\n\n"
        + deep_breath
        + "\n\n"
        + V3_OUTPUT_FORMAT
    )


def _message_content(value) -> str:
    if hasattr(value, "tolist"):
        value = value.tolist()

    if isinstance(value, list):
        for m in value:
            if isinstance(m, dict) and m.get("role") == "user":
                return str(m.get("content", ""))
        return ""
    return str(value or "")


def _normalize_old_parquet_row(row, idx: int) -> dict | None:
    prompt_value = row.get("prompt", row.get("messages", ""))
    prompt_text = _message_content(prompt_value)
    new_prompt = replace_prompt(prompt_text)
    if new_prompt is None:
        return None

    extra = row.get("extra_info", {})
    if not isinstance(extra, dict):
        extra = {}

    rm = row.get("reward_model", {})
    if not isinstance(rm, dict):
        rm = {}

    gt = rm.get("ground_truth", {})
    if not isinstance(gt, dict):
        gt = {}

    db_id = extra.get("db_id", gt.get("db_id", ""))
    gold_sql = gt.get("sql", gt.get("query", ""))

    return {
        "messages": [{"role": "user", "content": new_prompt}],
        "data_source": "bird_sql_v3",
        "reward_model": {
            "ground_truth": {"db_id": db_id, "sql": gold_sql}
        },
        "extra_info": {
            "db_id": db_id,
            "question": extra.get("question", ""),
            "idx": extra.get(
                "idx",
                extra.get("index", extra.get("question_id", idx)),
            ),
            "difficulty": extra.get("difficulty", ""),
            "evidence": extra.get("evidence", ""),
        },
    }


def _normalize_json_item(item: dict, idx: int) -> dict | None:
    new_prompt = replace_prompt(str(item.get("prompt", "")))
    if new_prompt is None:
        return None

    db_id = item.get("db_id", "")
    gold_sql = item.get("gold_sql", item.get("sql", ""))

    return {
        "messages": [{"role": "user", "content": new_prompt}],
        "data_source": "bird_sql_v3",
        "reward_model": {
            "ground_truth": {"db_id": db_id, "sql": gold_sql}
        },
        "extra_info": {
            "db_id": db_id,
            "question": item.get("question", ""),
            "idx": item.get("idx", idx),
            "difficulty": item.get("difficulty", ""),
            "evidence": item.get("evidence", ""),
        },
    }


def main():
    if len(sys.argv) != 3:
        print(
            "Usage: python3 build_v3_train_data.py "
            "<input.json|input.parquet> <output.parquet>"
        )
        raise SystemExit(1)

    input_file = Path(sys.argv[1])
    output_file = Path(sys.argv[2])

    rows = []
    skipped = 0

    if input_file.suffix.lower() == ".json":
        data = json.loads(input_file.read_text(encoding="utf-8"))
        for idx, item in enumerate(data):
            out = _normalize_json_item(item, idx)
            if out is None:
                skipped += 1
            else:
                rows.append(out)

    elif input_file.suffix.lower() == ".parquet":
        df = pd.read_parquet(input_file)
        for idx, row in df.iterrows():
            out = _normalize_old_parquet_row(row, idx)
            if out is None:
                skipped += 1
            else:
                rows.append(out)
    else:
        raise ValueError("Input must be .json or .parquet")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame(rows)
    df_out.to_parquet(output_file, index=False)

    print(f"input={input_file}")
    print(f"output={output_file}")
    print(f"converted={len(rows)}, skipped={skipped}")

    if rows:
        sample = rows[0]
        print("\nFirst output row:")
        print(json.dumps(sample, ensure_ascii=False, indent=2)[:4000])


if __name__ == "__main__":
    main()
