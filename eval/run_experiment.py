"""Run the before/after experiments (v1 vs v2) on the LangSmith dataset and print a comparison.

Usage (from the project folder):
    python eval/run_experiment.py                      Run v1 and v2
    python eval/run_experiment.py --run v3 \\
        --reuse v1=<v1 experiment name> v2=<v2 experiment name>
                                                       Run only v3, compare with existing experiments

Requires the dataset uploaded by: python eval/make_dataset.py --upload
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from langsmith import Client, evaluate  # noqa: E402

import config  # noqa: E402
import evaluators  # noqa: E402
import llm_judge  # noqa: E402
from make_dataset import DATASET_NAME, format_keeps  # noqa: E402

EXPERIMENTS = [  # (prompt version, experiment prefix)
    ("v1", "v1-names-only"),
    ("v2", "v2-full-context"),
    ("v3", "v3-reason-first"),
]
DELAY_SECONDS = 4  # Between calls, to stay within the Gemini quota


def make_target(prompt_version: str):
    def target(inputs: dict) -> dict:
        time.sleep(DELAY_SECONDS)
        result = llm_judge.judge_with_details(inputs["group"], prompt_version)
        return {
            "judgement": result.judgement.model_dump() if result.judgement else None,
            "failures": result.failures,
        }
    target.__name__ = f"judge_{prompt_version}"
    return target


def run(prompt_version: str, prefix: str):
    print(f"\nRunning {prefix} ...", flush=True)
    results = evaluate(
        make_target(prompt_version),
        data=DATASET_NAME,
        evaluators=evaluators.PER_EXAMPLE_EVALUATORS,
        summary_evaluators=evaluators.SUMMARY_EVALUATORS,
        experiment_prefix=prefix,
        metadata={"prompt_version": prompt_version, "model": config.GEMINI_MODEL},
        max_concurrency=1,  # One call at a time, with DELAY_SECONDS between calls
    )
    rows = {}
    for row in results:
        example = row["example"]
        rows[example.metadata["number"]] = {
            "outputs": row["run"].outputs or {},
            "reference": example.outputs,
            "description": example.metadata["description"],
        }
    return results.experiment_name, rows


def answer_text(outputs: dict) -> str:
    judgement = outputs.get("judgement")
    if judgement is None:
        return f"FAILED ({evaluators.failure_type(outputs)})"
    return f"{judgement['verdict']} keep={','.join(judgement['keep']) or '-'} ({judgement['confidence']:.2f})"


def fmt(value) -> str:
    return "-" if value is None else f"{value:.3f}" if isinstance(value, float) else str(value)


def load_rows(client: Client, experiment_name: str) -> dict:
    """Rows of an existing experiment, in the same shape run() returns."""
    examples = {e.id: e for e in client.list_examples(dataset_name=DATASET_NAME)}
    rows = {}
    for run_ in client.list_runs(project_name=experiment_name, is_root=True):
        example = examples[run_.reference_example_id]
        rows[example.metadata["number"]] = {
            "outputs": run_.outputs or {},
            "reference": example.outputs,
            "description": example.metadata["description"],
        }
    return rows


def print_comparison(names: dict, summaries: dict) -> None:
    versions = list(summaries)
    print("\n=== Comparison ===")
    print(f"{'metric':<30}" + "".join(f" {names[v][:24]:>25}" for v in versions))
    for key, label in [
        ("verdict_correct", "verdict_correct"),
        ("keep_correct", "keep_correct"),
        ("false_duplicate", "false_duplicate (lower=better)"),
        ("avg_confidence_correct", "avg confidence | correct"),
        ("avg_confidence_incorrect", "avg confidence | incorrect"),
        ("errors_with_low_confidence", "errors with low confidence"),
        ("failed_calls", "failed_calls"),
    ]:
        print(f"{label:<30}" + "".join(f" {fmt(summaries[v][key]):>25}" for v in versions))
    print(f"{'failed_calls by type':<30}"
          + "".join(f" {str(summaries[v]['failed_calls_by_type'] or '-'):>25}" for v in versions))


def print_differences(all_rows: dict, first: str, second: str) -> None:
    print(f"\n=== Examples where {first} and {second} answered differently ===")
    found = False
    for number in sorted(all_rows[first]):
        a, b = all_rows[first][number], all_rows[second][number]
        if answer_text(a["outputs"]).rsplit(" (", 1)[0] == answer_text(b["outputs"]).rsplit(" (", 1)[0]:
            continue
        found = True
        reference = a["reference"]
        mark = lambda row: {True: "✓", False: "✗", None: "?"}[evaluators.is_correct(row["outputs"], reference)]
        print(f"#{number} {a['description']}")
        print(f"    expected: {reference['expected_verdict']} keep={format_keeps(reference['acceptable_keeps'])}")
        print(f"    {first} {mark(a)} {answer_text(a['outputs'])}")
        print(f"    {second} {mark(b)} {answer_text(b['outputs'])}")
    if not found:
        print("(none)")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", nargs="+", default=["v1", "v2"], choices=[v for v, _ in EXPERIMENTS],
                        help="Prompt versions to run as new experiments")
    parser.add_argument("--reuse", nargs="*", default=[], metavar="VERSION=EXPERIMENT_NAME",
                        help="Existing experiments to include in the comparison without rerunning them")
    args = parser.parse_args()

    client = Client()
    names, all_rows = {}, {}
    for item in args.reuse:
        version, name = item.split("=", 1)
        names[version], all_rows[version] = name, load_rows(client, name)
    for version, prefix in EXPERIMENTS:
        if version in args.run:
            names[version], all_rows[version] = run(version, prefix)

    versions = sorted(all_rows)
    summaries = {}
    for version in versions:
        rows = [all_rows[version][n] for n in sorted(all_rows[version])]
        summaries[version] = evaluators.summarize([r["outputs"] for r in rows], [r["reference"] for r in rows])

    print_comparison(names, summaries)
    for first, second in zip(versions, versions[1:]):
        print_differences(all_rows, first, second)

    print("\n=== LangSmith ===")
    dataset = client.read_dataset(dataset_name=DATASET_NAME)
    sessions = [client.read_project(project_name=names[v]) for v in versions]
    for version, session in zip(versions, sessions):
        print(f"{names[version]}: {session.url}")
    print(f"compare: {dataset.url}/compare?selectedSessions={'%2C'.join(str(s.id) for s in sessions)}")


if __name__ == "__main__":
    main()
