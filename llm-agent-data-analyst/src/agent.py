"""The agent loop: an LLM orchestrating tools to answer data questions.

Standard tool-use loop -- send the conversation + tool schemas to the
model; if it returns tool_use blocks, execute them and feed the results
back; repeat until it returns a final text answer (or a step budget is
hit). Model-agnostic: it works with AnthropicLLM (real Claude) or MockLLM
(scripted) because both return the same LLMResponse.

The system prompt asks for a `FINAL ANSWER: <value>` line so the eval
harness can grade answers deterministically -- a small but real piece of
eval-oriented prompt design.

Run one question against real Claude (needs ANTHROPIC_API_KEY):
    python src/agent.py "How many students completed test prep?"
"""

from __future__ import annotations

from pathlib import Path

from tools import TOOL_SCHEMAS, dispatch_tool

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "students.db"

SYSTEM = """You are a data analyst agent with tools to inspect and query a
SQLite database containing a 'students' table of exam results.

Answer the user's question by:
1. Calling get_schema to see the columns.
2. Writing and running a SQL SELECT with run_sql to compute the answer.
3. Replying with a short explanation and a final line in EXACTLY this form:
   FINAL ANSWER: <value>
where <value> is the number or short phrase that answers the question
(a number should be plain, e.g. "358" or "66.1", no extra words).
Round averages to one decimal place. Do not guess -- compute with SQL."""


def run_agent(question: str, llm, db_path=DB, max_steps: int = 6):
    """Return (final_answer_text, message_history)."""
    messages = [{"role": "user", "content": question}]
    resp = None
    for _ in range(max_steps):
        resp = llm.create(SYSTEM, messages, TOOL_SCHEMAS)
        messages.append({"role": "assistant", "content": resp.content})

        tool_uses = resp.tool_uses()
        if not tool_uses:
            return resp.text(), messages

        tool_results = []
        for tu in tool_uses:
            result = dispatch_tool(tu["name"], tu.get("input", {}), db_path)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tu["id"],
                "content": result,
            })
        messages.append({"role": "user", "content": tool_results})

    # Step budget exhausted; return whatever text the last response had.
    return (resp.text() if resp else ""), messages


def main():
    import sys

    from llm import AnthropicLLM

    if not DB.exists():
        from data import build_db
        build_db()

    question = (" ".join(sys.argv[1:])
                or "How many students are in the dataset?")
    answer, _ = run_agent(question, AnthropicLLM())
    print(f"Q: {question}\n{answer}")


if __name__ == "__main__":
    main()
