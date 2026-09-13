"""Parser for the v3 draft -> reflection -> answer protocol."""
from __future__ import annotations

from dataclasses import dataclass
import re

_TAG_RE = {
    "draft": re.compile(r"<draft_sql>(.*?)</draft_sql>", re.IGNORECASE | re.DOTALL),
    "reflection": re.compile(r"<reflection>(.*?)</reflection>", re.IGNORECASE | re.DOTALL),
    "answer": re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL),
}
_SQL_FENCE_RE = re.compile(r"```sql\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def extract_sql_fence(text: str) -> str:
    """Extract the first ```sql ... ``` body, or return an empty string."""
    if not text:
        return ""
    m = _SQL_FENCE_RE.search(text)
    return m.group(1).strip() if m else ""


def _tag_body(text: str, name: str) -> tuple[bool, str]:
    m = _TAG_RE[name].search(text or "")
    if not m:
        return False, ""
    return True, m.group(1).strip()


def _is_exact_single_sql_fence(text: str) -> bool:
    """True only when the whole body is one SQL Markdown fence."""
    if not text:
        return False
    matches = list(_SQL_FENCE_RE.finditer(text))
    if len(matches) != 1:
        return False
    before = text[: matches[0].start()].strip()
    after = text[matches[0].end() :].strip()
    return not before and not after and bool(matches[0].group(1).strip())


@dataclass(frozen=True)
class ParsedSQLResponse:
    draft_sql: str
    reflection: str
    answer: str
    final_sql: str
    abstained: bool
    has_draft: bool
    reflection_present: bool
    has_answer: bool
    format_valid: bool


def parse_response(solution_str: str) -> ParsedSQLResponse:
    """Parse and strictly validate one model response."""
    text = solution_str or ""

    draft_tag, draft_body = _tag_body(text, "draft")
    reflection_tag, reflection = _tag_body(text, "reflection")
    answer_tag, answer = _tag_body(text, "answer")

    draft_sql = extract_sql_fence(draft_body)
    final_sql = extract_sql_fence(answer)

    has_draft = draft_tag and _is_exact_single_sql_fence(draft_body)
    reflection_present = reflection_tag and bool(reflection.strip())
    has_answer = answer_tag and bool(answer.strip())
    abstained = has_answer and answer.strip().upper() == "ABSTAIN"

    answer_valid = abstained or _is_exact_single_sql_fence(answer)

    # Also require that there is no non-whitespace output outside the 3 tags.
    stripped = text
    for name in ("draft", "reflection", "answer"):
        stripped = _TAG_RE[name].sub("", stripped, count=1)
    no_outside_text = not stripped.strip()

    format_valid = (
        has_draft
        and reflection_present
        and has_answer
        and answer_valid
        and no_outside_text
    )

    return ParsedSQLResponse(
        draft_sql=draft_sql,
        reflection=reflection,
        answer=answer,
        final_sql="" if abstained else final_sql,
        abstained=abstained,
        has_draft=has_draft,
        reflection_present=reflection_present,
        has_answer=has_answer,
        format_valid=format_valid,
    )
