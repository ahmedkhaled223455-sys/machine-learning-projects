"""Protocol-level tests: a real MCP client talks to the server in memory."""

import json

import pytest

pytest.importorskip("mcp")
import anyio  # noqa: E402
from mcp.shared.memory import create_connected_server_and_client_session  # noqa: E402

from grants_auth import AuthError  # noqa: E402
from grants_mcp_server import create_server  # noqa: E402

EXPECTED_TOOLS = {"whoami", "list_proposals", "search_proposals", "get_proposal", "list_opportunities",
                  "get_opportunity", "propose_rules", "validate_proposal", "update_status"}


def _run(server, body):
    async def main():
        async with create_connected_server_and_client_session(server._mcp_server) as client:
            return await body(client)
    return anyio.run(main)


def _json(result):
    return json.loads(result.content[0].text)


def test_bad_key_fails_at_startup(grants_env):
    with pytest.raises(AuthError):
        create_server(grants_env.db, "not-a-key")


def test_tool_list_and_schemas(grants_env):
    server = create_server(grants_env.db, grants_env.keys["t1-viewer"])

    async def body(client):
        return (await client.list_tools()).tools
    tools = {t.name: t for t in _run(server, body)}
    assert set(tools) == EXPECTED_TOOLS  # review_rule is human-only
    assert tools["get_proposal"].inputSchema["properties"]["proposal_id"]["type"] == "integer"
    assert all(t.description for t in tools.values())


def test_calls_round_trip_and_enforce_guardrails(grants_env):
    server = create_server(grants_env.db, grants_env.keys["t1-viewer"])

    async def body(client):
        me = _json(await client.call_tool("whoami", {}))
        listed = _json(await client.call_tool("list_proposals", {"limit": 3}))
        cross = await client.call_tool("get_proposal", {"proposal_id": 2})
        write = await client.call_tool("update_status", {"proposal_id": 1, "new_status": "draft"})
        bad_type = await client.call_tool("get_proposal", {"proposal_id": "abc"})
        return me, listed, cross, write, bad_type

    me, listed, cross, write, bad_type = _run(server, body)
    assert me["role"] == "viewer" and me["institution"] == "Gulf Coast University"
    assert listed["total"] == 20 and len(listed["proposals"]) == 3
    assert cross.isError and "not found" in cross.content[0].text
    assert write.isError and "lacks permission" in write.content[0].text
    assert bad_type.isError
