#!/usr/bin/env python3
"""Small killable SQLite execution comparator used by sql_reward.py.

Input on stdin:
{
  "db_file": "/path/to/db.sqlite",
  "gold_sql": "SELECT ...",
  "pred_sqls": ["SELECT ...", "..."]
}

Output:
{
  "any_correct": true/false,
  "results": [true/false, ...]
}

This is intentionally simple. For leaderboard-grade BIRD evaluation, replace this
file with the exact evaluator used by your benchmark version.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
from typing import Any


def _run_query(db_file: str, sql: str) -> list[tuple[Any, ...]]:
    uri = f"file:{os.path.abspath(db_file)}?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    try:
        cur = con.execute(sql)
        return cur.fetchall()
    finally:
        con.close()


def _norm_value(v: Any) -> str:
    # repr keeps None/int/float/string distinctions reasonably stable.
    return repr(v)


def _normalize(rows: list[tuple[Any, ...]]) -> list[tuple[str, ...]]:
    return sorted(tuple(_norm_value(v) for v in row) for row in rows)


def _same_result(db_file: str, pred_sql: str, gold_sql: str) -> bool:
    try:
        pred_rows = _run_query(db_file, pred_sql)
        gold_rows = _run_query(db_file, gold_sql)
        return _normalize(pred_rows) == _normalize(gold_rows)
    except Exception:
        return False


def main() -> None:
    payload = json.load(sys.stdin)
    db_file = payload["db_file"]
    gold_sql = payload["gold_sql"]
    pred_sqls = payload.get("pred_sqls", [])

    results = [_same_result(db_file, sql, gold_sql) for sql in pred_sqls]
    print(json.dumps({"any_correct": any(results), "results": results}))


if __name__ == "__main__":
    main()
