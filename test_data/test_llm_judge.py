"""Offline tests for llm_judge: validation, prompts and retry (no API calls)."""

import pytest
from pydantic import ValidationError

import llm_judge
from llm_judge import Judgement

GROUP = {
    "id": "g1",
    "jaccard": 0.8,
    "containment": 1.0,
    "members": [
        {"id": "A", "name": "תזה", "path": r"C:\x\תזה", "size_bytes": 1_260_000, "file_count": 4,
         "latest_modified": "2024-01-15 10:00", "unique_files": [], "unique_count": 0},
        {"id": "B", "name": "תזה - עותק", "path": r"C:\x\תזה - עותק", "size_bytes": 1_610_000,
         "file_count": 5, "latest_modified": "2025-06-01 12:00",
         "unique_files": [{"relative_path": "פרק 5 - חדש.docx", "size_bytes": 350_000}], "unique_count": 1},
    ],
}


def make(verdict="copy_with_changes", keep=("B",), confidence=0.9):
    return Judgement(verdict=verdict, confidence=confidence, keep=list(keep), reason="הסבר")


def test_valid_judgements_pass():
    llm_judge.validate_judgement(make(), GROUP)
    llm_judge.validate_judgement(make("different", keep=()), GROUP)


@pytest.mark.parametrize("judgement", [
    make(keep=("C",)),                 # Unknown member ID
    make("different", keep=("A",)),    # different must have empty keep
    make("identical", keep=()),        # Other verdicts need a non-empty keep
])
def test_invalid_keep_is_rejected(judgement):
    with pytest.raises(ValueError):
        llm_judge.validate_judgement(judgement, GROUP)


@pytest.mark.parametrize("fields", [
    {"verdict": "maybe", "confidence": 0.5, "keep": [], "reason": "x"},
    {"verdict": "identical", "confidence": 1.5, "keep": ["A"], "reason": "x"},
    {"verdict": "identical", "confidence": -0.1, "keep": ["A"], "reason": "x"},
])
def test_model_rejects_bad_verdict_and_confidence(fields):
    with pytest.raises(ValidationError):
        Judgement(**fields)


def test_v1_prompt_has_names_and_paths_only():
    system, human = (m.content for m in llm_judge.build_messages(GROUP, "v1"))
    assert "תזה - עותק" in human and r"C:\x\תזה" in human
    assert "פרק 5" not in human and "80%" not in human and "MB" not in human
    assert "Decision criteria" not in system


def test_v2_prompt_has_full_context_and_criteria():
    system, human = (m.content for m in llm_judge.build_messages(GROUP, "v2"))
    assert "80%" in human and "100%" in human
    assert "פרק 5 - חדש.docx" in human and "2025-06-01" in human
    assert "Decision criteria" in system and "גיבוי" in system


def test_v1_prompt_has_no_location_analysis():
    system, human = (m.content for m in llm_judge.build_messages(GROUP, "v1"))
    assert "distinct part" not in human and "common location" not in human
    assert "סוג מיקום" not in system


def test_projects_only_in_v2():
    projects = [r"C:\p\shop", r"C:\p\blog"]
    members = [dict(m, project=p) for m, p in zip(GROUP["members"], projects)]
    group = dict(GROUP, members=members)
    v1_system, v1_human = (m.content for m in llm_judge.build_messages(group, "v1"))
    v2_system, v2_human = (m.content for m in llm_judge.build_messages(group, "v2"))
    assert "project" not in v1_human.lower() and "project" not in v1_system.lower()
    assert r"inside project: C:\p\shop" in v2_human and r"inside project: C:\p\blog" in v2_human
    assert "different projects" in v2_system
    no_project = llm_judge.build_messages(GROUP, "v2")[1].content
    assert "inside project: none" in no_project


def test_v2_prompt_has_common_location_and_distinct_parts():
    system, human = (m.content for m in llm_judge.build_messages(GROUP, "v2"))
    assert "common location of all members: C:\\x" in human
    assert "distinct part of the path: תזה - עותק" in human
    assert "אם שתי התיקיות נמצאות באותו סוג מיקום, המיקום אינו שיקול" in system


@pytest.mark.parametrize("paths, prefix, distinct", [
    ([r"C:\Temp\run\a\site", r"C:\temp\RUN\a\site backup"], r"C:\Temp\run\a", ["site", "site backup"]),
    ([r"C:\Users\u\Documents\x", r"C:\Users\u\Downloads\x"], r"C:\Users\u", [r"Documents\x", r"Downloads\x"]),
    ([r"C:\a", r"C:\b", r"C:\c"], "C:\\", ["a", "b", "c"]),
    ([r"C:\data\x", r"D:\data\x"], "", [r"C:\data\x", r"D:\data\x"]),
    ([r"C:\p\site", r"C:\p\site backup\images"], r"C:\p", ["site", r"site backup\images"]),
])
def test_split_common_location(paths, prefix, distinct):
    assert llm_judge.split_common_location(paths) == (prefix, distinct)


def test_unique_files_are_truncated(monkeypatch):
    monkeypatch.setattr(llm_judge.config, "MAX_UNIQUE_FILES_TO_LLM", 2)
    member = dict(GROUP["members"][1], unique_count=5,
                  unique_files=[{"relative_path": f"f{i}.txt", "size_bytes": 1000} for i in range(5)])
    group = dict(GROUP, members=[GROUP["members"][0], member])
    human = llm_judge.build_messages(group, "v2")[1].content
    assert "showing 2 of 5" in human and "f1.txt" in human and "f2.txt" not in human


class FakeModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def invoke(self, messages, config=None):
        self.calls += 1
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


@pytest.mark.parametrize("responses, expected_calls, succeeds", [
    ([make()], 1, True),
    ([ConnectionError("network"), make()], 2, True),
    ([make(keep=("Z",)), make()], 2, True),                    # Invalid output, then valid
    ([TimeoutError("timeout"), ConnectionError("again")], 2, False),
    ([None, make("identical", keep=())], 2, False),            # Both outputs invalid
])
def test_one_retry_then_none(monkeypatch, responses, expected_calls, succeeds):
    fake = FakeModel(responses)
    monkeypatch.setattr(llm_judge, "_structured_model", lambda *_: fake)
    result = llm_judge.judge(GROUP, "v2")
    assert fake.calls == expected_calls
    assert (result is not None) == succeeds


class ReadTimeout(Exception):
    """Same class name as httpx.ReadTimeout."""


class ChatGoogleGenerativeAIError(Exception):
    pass


@pytest.mark.parametrize("error, category", [
    (ReadTimeout("The read operation timed out"), "timeout"),
    (TimeoutError(), "timeout"),
    (ChatGoogleGenerativeAIError("429 RESOURCE_EXHAUSTED: quota exceeded"), "quota"),
    (ChatGoogleGenerativeAIError("400 API key not valid"), "auth"),
    (ConnectionError("network unreachable"), "connection"),
    (ValueError("keep contains unknown member IDs"), "invalid_output"),
    (RuntimeError("something odd"), "other"),
])
def test_classify_error(error, category):
    assert llm_judge.classify_error(error) == category


def test_failure_types_are_reported(monkeypatch):
    fake = FakeModel([ReadTimeout("timed out"), make(keep=("Z",))])
    monkeypatch.setattr(llm_judge, "_structured_model", lambda *_: fake)
    result = llm_judge.judge_with_details(GROUP, "v2")
    assert result.judgement is None
    assert result.failures == ["timeout: ReadTimeout", "invalid_output: ValueError"]
    assert llm_judge.describe_failure(result.failures) == "פלט לא תקין מהמודל (ValueError)"


def test_unknown_prompt_version():
    with pytest.raises(ValueError):
        llm_judge.judge(GROUP, "v9")


def test_v3_schema_puts_reason_before_keep():
    v2_fields = list(llm_judge.OUTPUT_SCHEMAS["v2"].model_json_schema()["properties"])
    v3_fields = list(llm_judge.OUTPUT_SCHEMAS["v3"].model_json_schema()["properties"])
    assert v2_fields == ["verdict", "confidence", "keep", "reason"]  # v1/v2 unchanged
    assert v3_fields == ["verdict", "confidence", "reason", "keep"]
    assert llm_judge.OUTPUT_SCHEMAS["v1"] is llm_judge.Judgement


def test_v3_prompt_is_v2_plus_one_instruction():
    v2_system, v2_human = (m.content for m in llm_judge.build_messages(GROUP, "v2"))
    v3_system, v3_human = (m.content for m in llm_judge.build_messages(GROUP, "v3"))
    assert v3_human == v2_human
    assert v3_system == f"{v2_system}\n\n{llm_judge.V3_INSTRUCTION}"
    assert "לשמור את A, למחוק את B" in v3_system and "לשמור את A, למחוק את B" not in v2_system


def test_each_version_uses_its_schema(monkeypatch):
    used = []

    def fake_model(schema=llm_judge.Judgement):
        used.append(schema)
        return FakeModel([llm_judge.JudgementReasonFirst(
            verdict="copy_with_changes", confidence=0.9, reason="לשמור את B, למחוק את A", keep=["B"])])

    monkeypatch.setattr(llm_judge, "_structured_model", fake_model)
    result = llm_judge.judge(GROUP, "v3")
    assert used == [llm_judge.JudgementReasonFirst]
    assert isinstance(result, llm_judge.Judgement) and result.keep == ["B"]
