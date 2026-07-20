"""Test the full agent orchestration loop with a scripted MockLLM.

No API key, no network: the mock plays the role Claude would -- first ask
for the schema, then run SQL, then give a final answer -- and we verify
the agent actually executed those tools against the real database and
threaded the results back correctly.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent import run_agent
from llm import LLMResponse, MockLLM


def test_agent_runs_tools_and_returns_final_answer(db_path):
    scripted = [
        # 1. model asks for the schema
        LLMResponse("tool_use", [
            {"type": "tool_use", "id": "t1", "name": "get_schema", "input": {}}]),
        # 2. model runs a SQL query
        LLMResponse("tool_use", [
            {"type": "tool_use", "id": "t2", "name": "run_sql",
             "input": {"query": "SELECT COUNT(*) FROM students"}}]),
        # 3. model gives the final answer
        LLMResponse("end_turn", [
            {"type": "text", "text": "There are 1000 students.\nFINAL ANSWER: 1000"}]),
    ]
    llm = MockLLM(scripted)
    answer, messages = run_agent("How many students?", llm, db_path)

    assert "FINAL ANSWER: 1000" in answer
    # the agent made three model calls (schema, sql, final)
    assert len(llm.calls) == 3
    # tool_result blocks were produced and fed back
    tool_results = [b for m in messages if m["role"] == "user"
                    and isinstance(m["content"], list)
                    for b in m["content"] if b.get("type") == "tool_result"]
    assert len(tool_results) == 2
    # the run_sql tool_result actually carries the real count from the DB
    assert any("1000" in tr["content"] for tr in tool_results)


def test_agent_stops_at_step_budget(db_path):
    # A model that always asks for a tool and never finishes must not loop
    # forever -- the step budget caps it.
    loop = [LLMResponse("tool_use", [
        {"type": "tool_use", "id": f"t{i}", "name": "get_schema", "input": {}}])
        for i in range(10)]
    llm = MockLLM(loop)
    answer, _ = run_agent("never ends", llm, db_path, max_steps=3)
    assert llm.calls and len(llm.calls) == 3   # capped, didn't exhaust all 10
