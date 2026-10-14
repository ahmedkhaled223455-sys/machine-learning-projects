"""End-to-end check over real stdio: launch the server as a subprocess with
the official MCP client, list its tools, and exercise the main guardrails.

    python src/smoke_client.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import anyio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

from grants_store import build_db  # noqa: E402


def _json(result) -> dict:
    return json.loads(result.content[0].text)


async def run(db: str, key: str) -> None:
    params = StdioServerParameters(command=sys.executable, args=[str(SRC / "grants_mcp_server.py")],
                                   env={**os.environ, "GRANTS_DB": db, "GRANTS_API_KEY": key})
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = sorted(t.name for t in (await session.list_tools()).tools)
            print("tools:", tools)
            assert "review_rule" not in tools, "rule approval must not be exposed to the model"

            me = _json(await session.call_tool("whoami", {}))
            print("whoami:", me)

            page = _json(await session.call_tool("list_proposals", {"limit": 5}))
            print(f"list_proposals: {page['total']} proposals visible to this tenant")

            other = await session.call_tool("get_proposal", {"proposal_id": 2})  # tenant 2's record
            assert other.isError, "cross-tenant read must fail"
            print("cross-tenant get_proposal -> error:", other.content[0].text)

            bad = await session.call_tool("propose_rules", {"opportunity_id": 1, "rules": [
                {"rule": {"type": "max_amount", "limit": 900000},
                 "source_quote": "Requested budget must not exceed $750,000."}]})
            verdict = _json(bad)["results"][0]
            assert not verdict["accepted"]
            print("hallucinated rule rejected:", verdict["reason"])
    print("SMOKE OK")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db = str(Path(tmp) / "grants.db")
        keys = build_db(db)
        anyio.run(run, db, keys["t1-editor"])


if __name__ == "__main__":
    main()
