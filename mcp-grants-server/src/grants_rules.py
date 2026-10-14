"""Hybrid AI validation: LLM-extracted rules become deterministic checks.

An AI assistant reads a funding opportunity's instructions and proposes
rules like "budget must not exceed $750,000". The model's output is never
executed as code or trusted as-is. Instead each proposed rule must:

1. parse against a small, closed schema (``parse_rule``) -- unknown rule
   types, extra fields, or wrong value types are rejected;
2. be grounded (``check_grounding``) -- the rule cites a verbatim quote
   from the instructions, and the rule's value must appear in that quote,
   which catches hallucinated limits;
3. be approved by a person (``review_rules.py``) before it runs.

Approved rules are then evaluated deterministically (``evaluate``), so the
same proposal always gets the same, explainable result.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date
from typing import Any, Mapping

RULE_FIELDS: dict[str, dict[str, type]] = {
    "max_amount": {"limit": int},   # requested_amount <= limit (USD)
    "max_pages": {"limit": int},    # page_count <= limit
    "submit_by": {"date": str},     # submit_date <= date (ISO yyyy-mm-dd)
}

MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]


class RuleError(ValueError):
    """A proposed rule failed schema or grounding validation."""


@dataclass(frozen=True)
class Rule:
    type: str
    value: int | str

    def to_json(self) -> str:
        key = "date" if self.type == "submit_by" else "limit"
        return json.dumps({"type": self.type, key: self.value}, sort_keys=True)

    def describe(self) -> str:
        if self.type == "max_amount":
            return f"requested amount <= ${self.value:,}"
        if self.type == "max_pages":
            return f"page count <= {self.value}"
        return f"submitted on or before {self.value}"


def parse_rule(spec: Any) -> Rule:
    if not isinstance(spec, Mapping):
        raise RuleError("rule must be an object")
    rtype = spec.get("type")
    if rtype not in RULE_FIELDS:
        raise RuleError(f"unknown rule type: {rtype!r}; allowed: {sorted(RULE_FIELDS)}")
    expected = RULE_FIELDS[rtype]
    extra = set(spec) - {"type"} - set(expected)
    if extra:
        raise RuleError(f"unexpected fields for {rtype}: {sorted(extra)}")
    (field, ftype), = expected.items()
    if field not in spec:
        raise RuleError(f"{rtype} requires '{field}'")
    value = spec[field]
    if ftype is int:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise RuleError(f"{rtype}.{field} must be a positive integer")
    else:
        if not isinstance(value, str):
            raise RuleError(f"{rtype}.{field} must be an ISO date string")
        try:
            date.fromisoformat(value)
        except ValueError as exc:
            raise RuleError(f"{rtype}.{field} is not a valid ISO date: {value!r}") from exc
    return Rule(type=rtype, value=value)


def _numbers_in(text: str) -> set[int]:
    return {int(m.replace(",", "")) for m in re.findall(r"\d[\d,]*", text)}


def check_grounding(rule: Rule, source_quote: str, instructions: str) -> None:
    """Raise RuleError unless the quote is verbatim and supports the value."""
    quote = " ".join(source_quote.split())
    if len(quote) < 10:
        raise RuleError("source_quote is too short to justify a rule")
    if quote not in " ".join(instructions.split()):
        raise RuleError("source_quote does not appear verbatim in the opportunity instructions")
    if rule.type in ("max_amount", "max_pages"):
        if rule.value not in _numbers_in(quote):
            raise RuleError(f"value {rule.value} is not stated in the source_quote")
    else:
        d = date.fromisoformat(str(rule.value))
        forms = {d.isoformat(), f"{MONTHS[d.month - 1]} {d.day}, {d.year}"}
        if not any(f in quote for f in forms):
            raise RuleError(f"date {rule.value} is not stated in the source_quote")


def evaluate(rule: Rule, proposal: Mapping[str, Any]) -> dict[str, Any]:
    """Deterministic check of one rule against one proposal row."""
    if rule.type == "max_amount":
        actual = proposal["requested_amount"]
        passed: bool | None = actual <= rule.value
    elif rule.type == "max_pages":
        actual = proposal["page_count"]
        passed = actual <= rule.value
    else:
        actual = proposal["submit_date"]
        passed = None if actual is None else date.fromisoformat(actual) <= date.fromisoformat(str(rule.value))
    result = "pass" if passed else ("not_applicable" if passed is None else "fail")
    return {"rule": rule.describe(), "actual": actual, "result": result}
