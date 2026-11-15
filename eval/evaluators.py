"""Evaluators for the duplicate-judge experiments.

The target returns {"judgement": dict | None, "failures": ["category: ExceptionName", ...]}.
A failed call (judgement is None) is not counted as an error: per-example metrics return
score None for it, and it is counted separately in failed_calls, by failure type.

Per-example evaluators (signature: outputs, reference_outputs):
    verdict_correct, keep_correct, false_duplicate, failed_call
Summary evaluators (lists of outputs and reference_outputs):
    confidence_calibration, failed_calls
"""

import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402

LOW_CONFIDENCE = config.DEFAULT_CONFIDENCE_THRESHOLD


def _judgement(outputs: dict | None) -> dict | None:
    return (outputs or {}).get("judgement")


def failure_type(outputs: dict | None) -> str:
    """Category of the last failed attempt, for example "timeout"."""
    failures = (outputs or {}).get("failures") or []
    return failures[-1].split(":", 1)[0] if failures else "other"


def _skipped(key: str, reason: str) -> dict:
    return {"key": key, "score": None, "comment": reason}


# -------------------------------------------------------------- Per example


def verdict_correct(outputs: dict, reference_outputs: dict) -> dict:
    judgement = _judgement(outputs)
    if judgement is None:
        return _skipped("verdict_correct", "failed call")
    return {"key": "verdict_correct",
            "score": int(judgement["verdict"] == reference_outputs["expected_verdict"])}


def keep_correct(outputs: dict, reference_outputs: dict) -> dict:
    """Correct if the keep set equals one of the acceptable keeps. Skipped for "different"."""
    judgement = _judgement(outputs)
    if reference_outputs["expected_verdict"] == "different":
        return _skipped("keep_correct", "correct answer is different")
    if judgement is None:
        return _skipped("keep_correct", "failed call")
    acceptable = [set(keep) for keep in reference_outputs["acceptable_keeps"]]
    return {"key": "keep_correct", "score": int(set(judgement["keep"]) in acceptable)}


def false_duplicate(outputs: dict, reference_outputs: dict) -> dict:
    """1 if a "different" pair was classified as a duplicate (the most dangerous error).

    Scored only on examples whose correct answer is "different", so the mean is the
    false-duplicate rate among those examples.
    """
    judgement = _judgement(outputs)
    if reference_outputs["expected_verdict"] != "different":
        return _skipped("false_duplicate", "correct answer is not different")
    if judgement is None:
        return _skipped("false_duplicate", "failed call")
    return {"key": "false_duplicate", "score": int(judgement["verdict"] != "different")}


def failed_call(outputs: dict, reference_outputs: dict) -> dict:
    if _judgement(outputs) is None:
        return {"key": "failed_call", "score": 1, "comment": failure_type(outputs)}
    return {"key": "failed_call", "score": 0}


PER_EXAMPLE_EVALUATORS = [verdict_correct, keep_correct, false_duplicate, failed_call]


def is_correct(outputs: dict, reference_outputs: dict) -> bool | None:
    """Verdict correct and (when it applies) keep correct; None for a failed call."""
    verdict = verdict_correct(outputs, reference_outputs)["score"]
    if verdict is None:
        return None
    keep = keep_correct(outputs, reference_outputs)["score"]
    return bool(verdict) and keep != 0


# ------------------------------------------------------------------ Summary


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def confidence_calibration(outputs: list[dict], reference_outputs: list[dict]) -> dict:
    """Are errors concentrated among low-confidence answers?

    Reports the mean confidence of correct and incorrect answers, and the share of
    incorrect answers that had low confidence (below the UI threshold, so the user
    would have seen the "uncertain" banner).
    """
    correct, incorrect = [], []
    for out, ref in zip(outputs, reference_outputs):
        result = is_correct(out, ref)
        if result is not None:
            (correct if result else incorrect).append(_judgement(out)["confidence"])
    low_share = _mean([float(c < LOW_CONFIDENCE) for c in incorrect])
    return {"results": [
        {"key": "avg_confidence_correct", "score": _mean(correct)},
        {"key": "avg_confidence_incorrect", "score": _mean(incorrect)},
        {"key": "errors_with_low_confidence", "score": low_share,
         "comment": f"{len(incorrect)} errors, threshold {LOW_CONFIDENCE}"},
    ]}


def failed_calls(outputs: list[dict], reference_outputs: list[dict]) -> dict:
    by_type = Counter(failure_type(out) for out in outputs if _judgement(out) is None)
    results = [{"key": "failed_calls", "score": sum(by_type.values()),
                "comment": ", ".join(f"{k}: {v}" for k, v in sorted(by_type.items())) or "none"}]
    results += [{"key": f"failed_calls_{kind}", "score": count} for kind, count in sorted(by_type.items())]
    return {"results": results}


SUMMARY_EVALUATORS = [confidence_calibration, failed_calls]


def summarize(outputs: list[dict], reference_outputs: list[dict]) -> dict:
    """All metrics for one experiment, computed locally (for the printed comparison)."""
    def mean_of(evaluator):
        scores = [evaluator(o, r)["score"] for o, r in zip(outputs, reference_outputs)]
        scored = [s for s in scores if s is not None]
        return _mean(scored), len(scored)

    summary = {}
    for evaluator in (verdict_correct, keep_correct, false_duplicate):
        summary[evaluator.__name__], summary[f"{evaluator.__name__}_n"] = mean_of(evaluator)
    for result in confidence_calibration(outputs, reference_outputs)["results"]:
        summary[result["key"]] = result["score"]
    by_type = Counter(failure_type(o) for o in outputs if _judgement(o) is None)
    summary["failed_calls"] = sum(by_type.values())
    summary["failed_calls_by_type"] = dict(by_type)
    return summary
