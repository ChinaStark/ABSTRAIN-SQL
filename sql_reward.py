"""BIRD/SQLite execution reward for the v3 reflection protocol."""
from __future__ import annotations

import json
import os
import subprocess
import sys

try:
    from output_protocol import parse_response
except ImportError:
    from adaptive_sql_rein.output_protocol import parse_response


def _unpack_ground_truth(ground_truth, extra_info):
    db_id = None
    gold_sql = None
    if isinstance(ground_truth, dict):
        db_id = ground_truth.get("db_id")
        gold_sql = ground_truth.get("sql") or ground_truth.get("query")
    elif isinstance(ground_truth, str):
        gold_sql = ground_truth
    if isinstance(extra_info, dict):
        db_id = db_id or extra_info.get("db_id")
    return db_id, gold_sql


def _find_exec_sql():
    d = os.path.dirname(os.path.abspath(__file__))
    p = os.path.join(d, "exec_sql.py")
    return p if os.path.exists(p) else None


def _execute_sql_subprocess(
    db_path: str,
    pred_sql: str,
    gold_sql: str,
    timeout: float,
) -> bool:
    """Execute prediction and gold SQL in a killable subprocess."""
    exec_sql_path = _find_exec_sql()
    if not exec_sql_path:
        raise FileNotFoundError("exec_sql.py must be placed beside sql_reward.py")

    payload = json.dumps(
        {
            "db_file": db_path,
            "gold_sql": gold_sql,
            "pred_sqls": [pred_sql],
        }
    )

    try:
        result = subprocess.run(
            [sys.executable, exec_sql_path],
            input=payload,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            return False
        out = json.loads(result.stdout.strip())
        return bool(out.get("any_correct", False))
    except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception):
        return False


def check_format(solution_str: str) -> dict:
    """Parse response and expose fields used by the trainer/reward."""
    parsed = parse_response(solution_str)

    return {
        "has_draft": parsed.has_draft,
        "has_reflection": parsed.reflection_present,
        "has_answer_tag": parsed.has_answer,
        "is_abstain": parsed.abstained,
        "has_sql_fence": bool(parsed.final_sql) if not parsed.abstained else False,
        "format_valid": parsed.format_valid,
        "final_sql": parsed.final_sql,
        "draft_sql": parsed.draft_sql,
        "reflection": parsed.reflection,
        "parsed": parsed,
    }


def score_sql(data_source, sql: str, ground_truth, extra_info=None, **reward_kwargs):
    """Execute SQL and compare with gold. Returns 1.0 or 0.0."""
    del data_source  # kept for verl custom reward API compatibility

    sql = (sql or "").strip()
    if not sql or sql.upper() == "ABSTAIN":
        return 0.0

    db_root = reward_kwargs.get("db_root") or os.environ.get("DB_ROOT")
    dev_db_root = reward_kwargs.get("dev_db_root") or os.environ.get("DEV_DB_ROOT")
    timeout = float(
        reward_kwargs.get("exec_timeout")
        or os.environ.get("EXEC_TIMEOUT", 30.0)
    )

    db_id, gold_sql = _unpack_ground_truth(ground_truth, extra_info or {})

    if not db_root:
        print("[sql_reward] DB_ROOT not set", flush=True)
        return 0.0
    if not db_id:
        print(
            f"[sql_reward] db_id missing: ground_truth={ground_truth}, extra_info={extra_info}",
            flush=True,
        )
        return 0.0
    if not gold_sql:
        print(f"[sql_reward] gold_sql missing for db_id={db_id}", flush=True)
        return 0.0

    db_path = os.path.join(db_root, str(db_id), f"{db_id}.sqlite")
    if not os.path.exists(db_path) and dev_db_root:
        candidate = os.path.join(
            dev_db_root, str(db_id), f"{db_id}.sqlite"
        )
        if os.path.exists(candidate):
            db_path = candidate

    if not os.path.exists(db_path):
        print(
            f"[sql_reward] DB file not found: {db_path}; db_id={db_id}",
            flush=True,
        )
        return 0.0

    return 1.0 if _execute_sql_subprocess(
        db_path, sql, gold_sql, timeout
    ) else 0.0


def score_draft_sql(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    **reward_kwargs,
):
    """Score only <draft_sql>."""
    fmt = check_format(solution_str)
    if not fmt["draft_sql"]:
        return 0.0
    return score_sql(
        data_source=data_source,
        sql=fmt["draft_sql"],
        ground_truth=ground_truth,
        extra_info=extra_info,
        **reward_kwargs,
    )


def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    **reward_kwargs,
):
    """Verl custom reward entry point.

    This function supplies execution correctness to the trainer:
      - invalid protocol -> score 0
      - ABSTAIN -> execution score 0
      - final SQL result equals gold result -> score 1
      - otherwise -> score 0

    Boundary-aware scalar rewards are applied later by rein_sql_trainer.py.
    """
    fmt = check_format(solution_str)

    if not fmt["format_valid"]:
        return {
            "score": 0.0,
            "format_valid": False,
            "is_abstain": False,
            "has_draft": fmt["has_draft"],
        }

    if fmt["is_abstain"]:
        return {
            "score": 0.0,
            "format_valid": True,
            "is_abstain": True,
            "has_draft": True,
        }

    score = score_sql(
        data_source=data_source,
        sql=fmt["final_sql"],
        ground_truth=ground_truth,
        extra_info=extra_info,
        **reward_kwargs,
    )

    return {
        "score": score,
        "format_valid": True,
        "is_abstain": False,
        "has_draft": True,
    }
