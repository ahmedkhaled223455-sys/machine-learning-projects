"""Benchmark the agent against expert-curated ground truth.

Each line in eval_questions.jsonl is a natural-language question plus the
known-correct answer (verified independently in
../student-outcomes-analysis). The harness runs the agent on each, pulls
the `FINAL ANSWER:` value out of its reply, grades it, and reports
per-question pass/fail and overall accuracy -- i.e. tool quality measured
against a baseline, which is the point.

Run against real Claude (needs `anthropic` + ANTHROPIC_API_KEY):
    python evals/run_evals.py

The grading functions are deliberately import-friendly and model-free so
they can be unit-tested without any API (see tests/test_evals.py).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = ROOT / "evals" / "eval_questions.jsonl"


def extract_final_answer(text: str) -> str:
    """Pull the value after 'FINAL ANSWER:'; fall back to the whole text."""
    m = re.search(r"FINAL ANSWER:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    return (m.group(1) if m else text).strip()


def _first_number(text: str):
    m = re.search(r"-?\d[\d,]*\.?\d*", text.replace(",", ""))
    return float(m.group(0)) if m else None


def grade(prediction: str, expected, tolerance=None) -> bool:
    """Numeric answers compare within tolerance; text answers by substring."""
    if isinstance(expected, bool):  # guard: bool is a subclass of int
        expected = int(expected)
    if isinstance(expected, (int, float)):
        num = _first_number(prediction)
        if num is None:
            return False
        tol = 0.1 if tolerance is None else tolerance
        return abs(num - float(expected)) <= tol
    return str(expected).strip().lower() in prediction.lower()


def load_questions(path: Path = QUESTIONS) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]


def run_suite(llm, db_path=None) -> dict:
    """Run the agent over every eval question; return a results summary."""
    from agent import DB, run_agent
    db_path = db_path or DB
    questions = load_questions()
    results, n_pass = [], 0
    for item in questions:
        answer, _ = run_agent(item["question"], llm, db_path)
        pred = extract_final_answer(answer)
        ok = grade(pred, item["expected"], item.get("tolerance"))
        n_pass += ok
        results.append({"question": item["question"], "expected": item["expected"],
                        "predicted": pred, "pass": ok})
    return {"n": len(questions), "n_pass": n_pass,
            "accuracy": round(n_pass / len(questions), 3), "results": results}


def main():
    from agent import DB
    from data import build_db
    from llm import AnthropicLLM

    if not DB.exists():
        build_db()

    try:
        llm = AnthropicLLM()
    except Exception as e:  # missing SDK or key
        print(f"Cannot start real Claude client: {e}\n"
              "Install `anthropic` and set ANTHROPIC_API_KEY to run the "
              "benchmark against a live model.")
        raise SystemExit(1)

    summary = run_suite(llm)
    print(f"\nAgent benchmark: {summary['n_pass']}/{summary['n']} correct "
          f"(accuracy {summary['accuracy']:.0%})\n")
    for r in summary["results"]:
        mark = "PASS" if r["pass"] else "FAIL"
        print(f"  [{mark}] {r['question']}")
        print(f"         expected={r['expected']!r} predicted={r['predicted']!r}")


if __name__ == "__main__":
    main()
