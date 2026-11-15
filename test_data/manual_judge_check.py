"""Manual check: run judge with v1 and v2 on the fixture groups, side by side.

Usage (from the project folder):
    python test_data/manual_judge_check.py
Requires GOOGLE_API_KEY in .env. Calls are traced to LangSmith when tracing is on.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
import llm_judge  # noqa: E402
import scanner  # noqa: E402
from make_fixtures import create_fixtures  # noqa: E402

EXPECTED = {
    "01_copy_suffix": "identical",
    "02_three_copies": "identical",
    "03_newer_copy": "copy_with_changes, keep תזה - עותק",
    "04_similar_names": "different",
    "05_different_names": "identical",
    "06_nested": "identical",
    "10_containment": "copy_with_changes, keep מצגת - מורחבת",
    "11_projects": "identical, keep both (different projects), low confidence",
}


def summarize(judgement, group) -> str:
    if judgement is None:
        return llm_judge.NO_RECOMMENDATION_MESSAGE
    names = {m["id"]: m["name"] for m in group["members"]}
    keep = ", ".join(f"{k}={names[k]}" for k in judgement.keep) or "-"
    return f"{judgement.verdict} ({judgement.confidence:.2f}) keep: {keep}"


def main() -> None:
    root = create_fixtures(tempfile.mkdtemp(prefix="duplicate_finder_"))
    result = scanner.scan([root])
    candidates = scanner.find_candidates(
        result, config.DEFAULT_OVERLAP_THRESHOLD, config.DEFAULT_MIN_FOLDER_SIZE_MB
    )
    groups = [item for item in candidates["queue"] if item["type"] == "duplicate"]

    rows = []
    for group in groups:
        case = os.path.relpath(group["members"][0]["path"], root).split(os.sep)[0]
        v1 = llm_judge.judge(group, "v1")
        v2 = llm_judge.judge(group, "v2")
        rows.append((case, EXPECTED.get(case, "?"), summarize(v1, group), summarize(v2, group), v1, v2))

    print(f"\nModel: {config.GEMINI_MODEL}   Fixtures: {root}\n")
    for case, expected, v1_text, v2_text, v1, v2 in rows:
        print(f"[{case}]  expected: {expected}")
        print(f"   v1: {v1_text}")
        print(f"   v2: {v2_text}")
        if v1:
            print(f"   v1 reason: {v1.reason}")
        if v2:
            print(f"   v2 reason: {v2.reason}")
        print()


if __name__ == "__main__":
    main()
