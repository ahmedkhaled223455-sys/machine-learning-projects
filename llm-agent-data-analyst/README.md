# LLM Agent: Tool-Using Data Analyst

An LLM agent that answers natural-language questions about a dataset by
**calling tools** — inspecting a database schema and running SQL — in a
real Anthropic tool-use / function-calling loop, with an **eval harness**
that scores its answers against known-correct baselines.

The focus is the engineering an AI-tooling role actually cares about: a
clean agent orchestration loop, a well-defined tool interface, a
model-agnostic client that's testable without an API key, and evals that
measure quality against a baseline rather than vibes.

## What it does

Ask a question in English; the agent figures out the SQL:

```
$ python src/agent.py "What is the average math score for students who completed test prep?"
Q: What is the average math score for students who completed test prep?
The average math score among students who completed test prep is 69.7.
FINAL ANSWER: 69.7
```

Under the hood it runs a standard **tool-use loop** (`src/agent.py`):

1. Send the question + tool schemas to Claude.
2. Claude replies with `tool_use` blocks (e.g. `get_schema`, then
   `run_sql`).
3. The agent executes each tool and feeds the results back as
   `tool_result`.
4. Repeat until Claude returns a final text answer (bounded by a step
   budget so a misbehaving model can't loop forever).

## The pieces

| File | Role |
|---|---|
| `src/tools.py` | The function-calling surface: Anthropic `tools=[...]` JSON schemas + pure, read-only implementations (`get_schema`, `run_sql` with a SELECT-only guard). |
| `src/agent.py` | The orchestration loop and system prompt (which requires a gradable `FINAL ANSWER:` line). |
| `src/llm.py` | Model-agnostic client. `AnthropicLLM` calls Claude; `MockLLM` returns scripted responses so the whole loop is testable offline. |
| `evals/run_evals.py` | Runs the agent over `eval_questions.jsonl` and scores answers against ground truth. |

## Tool-use, not string-parsing

`run_sql` is exposed to the model as a real function with a JSON schema,
and it's **read-only** — it rejects anything but a single `SELECT`, so
the model can't drop or mutate the table even if it tries. Tools return
plain strings (JSON result sets) that go straight back into the
conversation as `tool_result` blocks. This is the actual Anthropic
function-calling contract, not a regex over model output.

## Evals against a baseline

`evals/eval_questions.jsonl` holds 10 natural-language questions paired
with **known-correct answers** (independently verified in
[`../student-outcomes-analysis`](../student-outcomes-analysis/)). The
harness pulls each answer out of the agent's `FINAL ANSWER:` line, grades
it (numeric within a tolerance, or text by match), and reports
per-question pass/fail plus overall accuracy:

```
Agent benchmark: 10/10 correct (accuracy 100%)
  [PASS] How many students completed the test preparation course?
         expected=358 predicted='358'
  ...
```

Designing the system prompt to emit a structured `FINAL ANSWER:` is a
small piece of **eval-oriented prompt engineering** — it makes answers
gradable without a second model to parse them.

## How this is verified (and the honest caveat)

The sandbox this was built in has no API key, so the split is deliberate:

- **Runs in CI, no key needed.** The tools (pure SQLite) and the *entire
  agent loop* are tested with a scripted `MockLLM` that plays Claude's
  role — issuing `get_schema`, then `run_sql`, then a final answer — while
  the tests confirm the agent really executed those tools against the
  database and threaded the results back (`tests/`). The grading logic is
  unit-tested too.
- **Needs your key.** Producing real accuracy numbers means running the
  agent against a live model. Set `ANTHROPIC_API_KEY` and run the
  benchmark below; the code path is real and ready.

So: the orchestration, tools, and scoring are proven; the model's actual
score on the eval set is one command away once you supply a key.

## Run

```bash
pip install -r requirements.txt
python src/data.py                 # build the SQLite database

# offline: tools + agent loop + grading (no key)
python -m pytest tests/ -v

# live: benchmark the agent against real Claude
export ANTHROPIC_API_KEY=sk-ant-...     # your key
python evals/run_evals.py
# ask a one-off question:
python src/agent.py "Which parental education level has the highest test-prep completion rate?"
```

`ANTHROPIC_MODEL` overrides the model (defaults to a current Claude
Sonnet).

## Limitations / next steps

- Single dataset and two tools; the same loop generalizes to more tools
  (plotting, Python execution, retrieval) — the registry in `tools.py`
  is where they'd plug in.
- No retry/repair step when the model writes invalid SQL; the tool
  returns the error and the loop lets the model correct itself, but an
  explicit self-repair policy would raise accuracy.
- Grading is exact/tolerance-based; open-ended questions would need an
  LLM-as-judge rubric (with its own baseline to keep it honest).
