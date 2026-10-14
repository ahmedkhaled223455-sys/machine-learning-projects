"""Human review of AI-proposed rules (deliberately not an MCP tool).

    GRANTS_API_KEY=<admin key> python src/review_rules.py list
    GRANTS_API_KEY=<admin key> python src/review_rules.py approve 3
    GRANTS_API_KEY=<admin key> python src/review_rules.py reject 4
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from grants_auth import authenticate  # noqa: E402
from grants_service import GrantsService  # noqa: E402
from grants_store import connect  # noqa: E402


def main(argv: list[str]) -> int:
    db = os.environ.get("GRANTS_DB", str(Path(__file__).resolve().parent.parent / "data" / "grants.db"))
    conn = connect(db)
    principal = authenticate(conn, os.environ.get("GRANTS_API_KEY"))
    if not argv or argv[0] == "list":
        rows = conn.execute(
            "SELECT r.id, r.opportunity_id, r.spec, r.source_quote, r.status FROM rules r "
            "WHERE r.tenant_id = ? ORDER BY r.id", (principal.tenant_id,)).fetchall()
        for r in rows:
            print(f"[{r['id']}] opp {r['opportunity_id']} {r['status']:9s} {r['spec']}\n      quote: {r['source_quote']}")
        return 0
    action, rule_id = argv[0], int(argv[1])
    decision = {"approve": "approved", "reject": "rejected"}[action]
    print(json.dumps(GrantsService(conn, principal).review_rule(rule_id, decision)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
