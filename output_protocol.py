"""Parsing utilities for the v3 draft -> reflection -> answer protocol."""
from __future__ import annotations

from dataclasses import dataclass
import re


def _tag(text: str, name: str) -> str:
    """Extract content between a pair of protocol tags."""
    match = re.search(
        rf"<{name}\s*>(.*?)</{name}\s*>",
        text or "",
        flags=re.IGNORECASE | re.DOTALL,
    )
    return match.group(1).strip() if match else ""


def extract_sql_fence(text: str) -> str:
    """Extract SQL from a SQL fence, accepting a generic fence as fallback."""
    if not text:
        return ""
    match = re.search(r"```sql\s*(.*?)\s*```", text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
    return match.group(1).strip() if match else ""


@dataclass(frozen=True)
class ParsedSQLResponse:
    draft_section: str
    reflection: str
    answer: str

    @property
    def draft_sql(self) -> str:
        return extract_sql_fence(self.draft_section)

    @property
    def reflection_present(self) -> bool:
        return bool(self.reflection)

    @property
    def abstained(self) -> bool:
        return self.answer.strip().upper() == "ABSTAIN"

    @property
    def final_sql(self) -> str:
        return "" if self.abstained else extract_sql_fence(self.answer)

    @property
    def has_draft(self) -> bool:
        return bool(self.draft_sql)

    @property
    def has_answer(self) -> bool:
        return bool(self.answer)

    @property
    def is_invalid(self) -> bool:
        return not self.abstained and not bool(self.final_sql)

    @property
    def format_valid(self) -> bool:
        return (
            self.has_draft
            and self.reflection_present
            and self.has_answer
            and not self.is_invalid
        )


def parse_response(text: str) -> ParsedSQLResponse:
    """Parse model output into draft, reflection, and final-answer sections."""
    return ParsedSQLResponse(
        draft_section=_tag(text, "draft_sql"),
        reflection=_tag(text, "reflection"),
        answer=_tag(text, "answer"),
    )


def is_abstain_response(text: str) -> bool:
    return parse_response(text).abstained
