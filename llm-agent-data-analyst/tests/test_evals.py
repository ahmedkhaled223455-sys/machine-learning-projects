"""Unit tests for the eval harness's grading logic -- no model needed."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))

from run_evals import extract_final_answer, grade, load_questions


def test_extract_final_answer():
    assert extract_final_answer("blah\nFINAL ANSWER: 358") == "358"
    assert extract_final_answer("FINAL ANSWER: some high school") == "some high school"
    # case-insensitive, tolerates surrounding text
    assert extract_final_answer("Final Answer:  66.1  ") == "66.1"


def test_grade_numeric_within_tolerance():
    assert grade("358", 358, 0) is True
    assert grade("The answer is 66.1", 66.1, 0.15) is True
    assert grade("66.3", 66.1, 0.15) is False        # outside tolerance
    assert grade("1,000 students", 1000, 0) is True   # comma handling


def test_grade_text_substring():
    assert grade("some high school", "some high school") is True
    assert grade("It is 'some high school'.", "some high school") is True
    assert grade("bachelor's degree", "some high school") is False


def test_grade_handles_missing_number():
    assert grade("no idea", 42, 0) is False


def test_eval_questions_file_is_well_formed():
    qs = load_questions()
    assert len(qs) >= 8
    for q in qs:
        assert "question" in q and "expected" in q
