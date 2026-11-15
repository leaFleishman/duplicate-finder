"""Offline tests for the experiment evaluators."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval"))

import evaluators  # noqa: E402

IDENTICAL = {"expected_verdict": "identical", "acceptable_keeps": [["B"], ["C"]]}
DIFFERENT = {"expected_verdict": "different", "acceptable_keeps": [[]]}


def answer(verdict, keep, confidence=0.9):
    return {"judgement": {"verdict": verdict, "keep": keep, "confidence": confidence, "reason": "x"},
            "failures": []}


FAILED = {"judgement": None, "failures": ["timeout: ReadTimeout", "quota: ChatGoogleGenerativeAIError"]}


def score(evaluator, outputs, reference):
    return evaluator(outputs, reference)["score"]


def test_verdict_and_keep_with_acceptable_alternatives():
    assert score(evaluators.verdict_correct, answer("identical", ["C"]), IDENTICAL) == 1
    assert score(evaluators.keep_correct, answer("identical", ["C"]), IDENTICAL) == 1
    assert score(evaluators.keep_correct, answer("identical", ["B", "C"]), IDENTICAL) == 0
    assert score(evaluators.keep_correct, answer("identical", ["A"]), IDENTICAL) == 0
    assert score(evaluators.keep_correct, answer("different", []), DIFFERENT) is None  # Skipped


def test_false_duplicate_only_on_different_examples():
    assert score(evaluators.false_duplicate, answer("identical", ["A"]), DIFFERENT) == 1
    assert score(evaluators.false_duplicate, answer("copy_with_changes", ["A"]), DIFFERENT) == 1
    assert score(evaluators.false_duplicate, answer("different", []), DIFFERENT) == 0
    assert score(evaluators.false_duplicate, answer("different", []), IDENTICAL) is None


def test_failed_call_is_not_an_error():
    for evaluator in (evaluators.verdict_correct, evaluators.keep_correct, evaluators.false_duplicate):
        assert score(evaluator, FAILED, IDENTICAL) is None
    assert evaluators.failed_call(FAILED, IDENTICAL) == {"key": "failed_call", "score": 1, "comment": "quota"}
    assert evaluators.is_correct(FAILED, IDENTICAL) is None


def test_summaries():
    outputs = [answer("identical", ["B"], 0.95), answer("identical", ["A"], 0.4),
               answer("identical", ["A"], 0.9), FAILED, answer("different", [], 0.8)]
    references = [IDENTICAL, IDENTICAL, DIFFERENT, IDENTICAL, DIFFERENT]
    summary = evaluators.summarize(outputs, references)
    assert summary["verdict_correct"] == 0.75 and summary["verdict_correct_n"] == 4
    assert summary["keep_correct"] == 0.5
    assert summary["false_duplicate"] == 0.5
    assert summary["failed_calls"] == 1 and summary["failed_calls_by_type"] == {"quota": 1}
    assert summary["avg_confidence_correct"] == 0.875  # (0.95 + 0.8) / 2
    assert summary["avg_confidence_incorrect"] == 0.65  # (0.4 + 0.9) / 2
    assert summary["errors_with_low_confidence"] == 0.5
    keys = [r["key"] for r in evaluators.failed_calls(outputs, references)["results"]]
    assert keys == ["failed_calls", "failed_calls_quota"]
