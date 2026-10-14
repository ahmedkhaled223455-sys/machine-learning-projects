import pytest

from grants_rules import Rule, RuleError, check_grounding, evaluate, parse_rule
from grants_store import OPPORTUNITIES

NSF_INSTR = OPPORTUNITIES[0][3]


def test_parse_valid_rules():
    assert parse_rule({"type": "max_amount", "limit": 750000}) == Rule("max_amount", 750000)
    assert parse_rule({"type": "submit_by", "date": "2026-11-15"}).value == "2026-11-15"


@pytest.mark.parametrize("spec", [
    "max_amount 750000",
    {"type": "run_python", "code": "import os"},
    {"type": "max_amount", "limit": 750000, "note": "x"},
    {"type": "max_amount"},
    {"type": "max_amount", "limit": "750000"},
    {"type": "max_amount", "limit": True},
    {"type": "max_pages", "limit": -3},
    {"type": "submit_by", "date": "Nov 15"},
])
def test_parse_rejects_malformed(spec):
    with pytest.raises(RuleError):
        parse_rule(spec)


def test_grounding_accepts_stated_values():
    check_grounding(Rule("max_amount", 750000), "Requested budget must not exceed $750,000.", NSF_INSTR)
    check_grounding(Rule("max_pages", 15), "Project description limited to 15 pages.", NSF_INSTR)
    check_grounding(Rule("submit_by", "2026-11-15"),
                    "Proposals must be submitted by 5:00 PM on November 15, 2026.", NSF_INSTR)


@pytest.mark.parametrize("rule,quote", [
    (Rule("max_amount", 900000), "Requested budget must not exceed $750,000."),        # wrong value
    (Rule("max_amount", 750000), "Requested budget must not exceed $750,000 total."),  # not verbatim
    (Rule("max_pages", 15), "15 pages."),                                               # too short
    (Rule("submit_by", "2026-11-16"),
     "Proposals must be submitted by 5:00 PM on November 15, 2026."),                  # wrong date
])
def test_grounding_rejects_unsupported_rules(rule, quote):
    with pytest.raises(RuleError):
        check_grounding(rule, quote, NSF_INSTR)


def test_evaluate_is_deterministic():
    row = {"requested_amount": 800000, "page_count": 12, "submit_date": None}
    assert evaluate(Rule("max_amount", 750000), row)["result"] == "fail"
    assert evaluate(Rule("max_pages", 15), row)["result"] == "pass"
    assert evaluate(Rule("submit_by", "2026-11-15"), row)["result"] == "not_applicable"
    row["submit_date"] = "2026-11-15"
    assert evaluate(Rule("submit_by", "2026-11-15"), row)["result"] == "pass"
    row["submit_date"] = "2026-11-16"
    assert evaluate(Rule("submit_by", "2026-11-15"), row)["result"] == "fail"
