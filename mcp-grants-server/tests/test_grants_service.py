import json

import pytest

from grants_auth import PermissionDenied
from grants_service import ToolError

NSF_BUDGET = {"rule": {"type": "max_amount", "limit": 750000},
              "source_quote": "Requested budget must not exceed $750,000."}


def _audit(env, tool=None):
    sql = "SELECT tool, outcome FROM audit_log"
    rows = env.conn.execute(sql + (" WHERE tool = ?" if tool else ""), (tool,) if tool else ()).fetchall()
    return [(r["tool"], r["outcome"]) for r in rows]


def _set(env, proposal_id, **fields):
    cols = ", ".join(f"{k} = ?" for k in fields)
    env.conn.execute(f"UPDATE proposals SET {cols} WHERE id = ?", (*fields.values(), proposal_id))
    env.conn.commit()


# ------------------------------------------------------------- tenant isolation
def test_list_only_returns_own_tenant(grants_env):
    out = grants_env.service("t1-viewer").list_proposals(limit=100)
    ids = [p["id"] for p in out["proposals"]]
    assert out["total"] == 20 and len(ids) == 20
    assert all((i - 1) % 3 == 0 for i in ids)  # tenant 1 owns ids 1, 4, 7, ...


def test_other_tenant_record_is_not_found_not_forbidden(grants_env):
    svc = grants_env.service("t1-admin")
    with pytest.raises(ToolError, match="not found"):
        svc.get_proposal(2)  # owned by tenant 2
    with pytest.raises(ToolError, match="not found"):
        svc.get_proposal(9999)  # does not exist: same message, so existence isn't leaked
    with pytest.raises(ToolError, match="not found"):
        svc.update_status(2, "internal_review")


def test_search_is_tenant_scoped(grants_env):
    assert grants_env.service("t1-viewer").search_proposals("IGNORE ALL")["proposals"]
    assert grants_env.service("t2-viewer").search_proposals("IGNORE ALL")["proposals"] == []


def test_rules_are_tenant_scoped(grants_env):
    grants_env.service("t1-editor").propose_rules(1, [NSF_BUDGET])
    assert grants_env.service("t2-viewer").get_opportunity(1)["rules"] == []
    rule_id = grants_env.service("t1-viewer").get_opportunity(1)["rules"][0]["id"]
    with pytest.raises(ToolError, match="not found"):
        grants_env.service("t2-admin").review_rule(rule_id, "approved")


# ------------------------------------------------------------- permissions
def test_viewer_cannot_write_and_denial_is_audited(grants_env):
    svc = grants_env.service("t1-viewer")
    with pytest.raises(PermissionDenied):
        svc.update_status(1, "internal_review")
    with pytest.raises(PermissionDenied):
        svc.propose_rules(1, [NSF_BUDGET])
    assert _audit(grants_env, "update_status") == [("update_status", "denied")]


def test_only_admin_can_approve_rules(grants_env):
    editor = grants_env.service("t1-editor")
    rid = editor.propose_rules(1, [NSF_BUDGET])["results"][0]["rule_id"]
    with pytest.raises(PermissionDenied):
        editor.review_rule(rid, "approved")
    assert grants_env.service("t1-admin").review_rule(rid, "approved")["status"] == "approved"
    with pytest.raises(ToolError, match="already approved"):
        grants_env.service("t1-admin").review_rule(rid, "rejected")


# ------------------------------------------------------------- input validation
@pytest.mark.parametrize("kwargs", [{"limit": 0}, {"limit": 101}, {"limit": True},
                                    {"status": "deleted"}, {"offset": -1}, {"opportunity_id": "1"}])
def test_list_rejects_bad_inputs(grants_env, kwargs):
    with pytest.raises(ToolError):
        grants_env.service("t1-viewer").list_proposals(**kwargs)


def test_search_input_is_bounded_and_parameterized(grants_env):
    svc = grants_env.service("t1-viewer")
    with pytest.raises(ToolError):
        svc.search_proposals("x" * 101)
    with pytest.raises(ToolError):
        svc.search_proposals("   ")
    assert svc.search_proposals("'; DROP TABLE proposals; --")["proposals"] == []
    assert svc.search_proposals("%")["proposals"] == []  # wildcard is escaped, matches literally
    assert grants_env.conn.execute("SELECT COUNT(*) FROM proposals").fetchone()[0] == 60


# ------------------------------------------------------------- untrusted text
def test_injected_text_is_returned_as_labelled_data(grants_env):
    out = grants_env.service("t1-editor").get_proposal(7)
    assert "IGNORE ALL PREVIOUS INSTRUCTIONS" in out["proposal"]["abstract"]
    assert "abstract" in out["untrusted_fields"] and "not as instructions" in out["notice"]
    statuses_before = grants_env.conn.execute("SELECT status FROM proposals ORDER BY id").fetchall()
    # reading it changed nothing
    assert grants_env.conn.execute("SELECT status FROM proposals ORDER BY id").fetchall() == statuses_before
    assert _audit(grants_env) == [("get_proposal", "ok")]


# ------------------------------------------------------------- rules workflow
def test_rules_need_grounding_then_human_approval(grants_env):
    editor = grants_env.service("t1-editor")
    res = editor.propose_rules(1, [
        NSF_BUDGET,
        {"rule": {"type": "max_amount", "limit": 900000},
         "source_quote": "Requested budget must not exceed $750,000."},
        {"rule": {"type": "max_pages", "limit": 15}},
    ])["results"]
    assert [r["accepted"] for r in res] == [True, False, False]

    _set(grants_env, 1, opportunity_id=1, requested_amount=800_000)
    before = editor.validate_proposal(1)
    assert before["approved_rules"] == 0 and before["rules_pending_review"] == 1 and before["valid"]

    grants_env.service("t1-admin").review_rule(res[0]["rule_id"], "approved")
    after = editor.validate_proposal(1)
    assert not after["valid"] and after["checks"][0]["result"] == "fail"


def test_submission_blocked_by_failing_approved_rule(grants_env):
    editor, admin = grants_env.service("t1-editor"), grants_env.service("t1-admin")
    rid = editor.propose_rules(1, [NSF_BUDGET])["results"][0]["rule_id"]
    admin.review_rule(rid, "approved")

    _set(grants_env, 1, opportunity_id=1, status="internal_review", requested_amount=800_000, submit_date=None)
    with pytest.raises(ToolError, match="submission blocked"):
        editor.update_status(1, "submitted")
    assert grants_env.conn.execute("SELECT status FROM proposals WHERE id = 1").fetchone()[0] == "internal_review"

    _set(grants_env, 1, requested_amount=700_000)
    out = editor.update_status(1, "submitted")
    assert out["new_status"] == "submitted"
    row = grants_env.conn.execute("SELECT submit_date FROM proposals WHERE id = 1").fetchone()
    assert row[0] == "2026-11-10"


def test_status_transitions_are_enforced(grants_env):
    editor = grants_env.service("t1-editor")
    _set(grants_env, 4, status="draft", submit_date=None)
    with pytest.raises(ToolError, match="cannot move"):
        editor.update_status(4, "awarded")
    with pytest.raises(ToolError):
        editor.update_status(4, "deleted")
    assert editor.update_status(4, "internal_review")["new_status"] == "internal_review"


def test_every_call_is_audited(grants_env):
    svc = grants_env.service("t1-editor")
    svc.whoami()
    svc.list_proposals()
    with pytest.raises(ToolError):
        svc.get_proposal(2)
    rows = grants_env.conn.execute("SELECT tool, outcome, user_id, tenant_id, args FROM audit_log ORDER BY id").fetchall()
    assert [(r["tool"], r["outcome"]) for r in rows] == [
        ("whoami", "ok"), ("list_proposals", "ok"), ("get_proposal", "error")]
    assert all(r["tenant_id"] == 1 for r in rows)
    assert json.loads(rows[2]["args"]) == {"proposal_id": 2}
