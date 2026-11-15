"""Graph tests with a fake judge and a fake Recycle Bin (no API calls, no real trash)."""

import csv
import os
import shutil

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

import actions
import config
import graph
import llm_judge
from make_fixtures import create_fixtures


def fake_judge(group, prompt_version=None):
    if any(m["name"].startswith("דוח") for m in group["members"]):
        return llm_judge.Judgement(verdict="different", confidence=0.9, keep=[], reason="שונות")
    return llm_judge.Judgement(verdict="identical", confidence=0.95, keep=["A"], reason="זהות")


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Fresh fixtures, fake judge, fake trash and a temporary log file."""
    root = create_fixtures(str(tmp_path))
    trashed = []

    def fake_trash(path):
        trashed.append(path)
        shutil.rmtree(path) if os.path.isdir(path) else os.remove(path)

    monkeypatch.setattr(graph.llm_judge, "judge_with_details",
                        lambda group, prompt_version=None: llm_judge.JudgeResult(fake_judge(group), []))
    monkeypatch.setattr(actions, "send2trash", fake_trash)
    monkeypatch.setattr(config, "ACTIONS_LOG_PATH", str(tmp_path / "log.csv"))
    return {"root": root, "trashed": trashed, "log": str(tmp_path / "log.csv")}


def start(app, root, thread="t1"):
    cfg = graph.thread_config(thread)
    app.invoke(graph.initial_state([root]), cfg)
    return cfg


def current_payload(app, cfg):
    interrupts = app.get_state(cfg).interrupts
    return interrupts[0].value if interrupts else None


def run_to_end(app, cfg, decide):
    """Answer every interrupt with decide(payload); return the payloads seen."""
    seen = []
    while (payload := current_payload(app, cfg)) is not None:
        seen.append(payload)
        app.invoke(Command(resume=decide(payload)), cfg)
    return seen


def read_log(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def names(payload):
    return [m["name"] for m in payload["item"]["members"]]


def test_full_run(env):
    app = graph.build_graph(InMemorySaver())
    cfg = start(app, env["root"])
    copy_path = os.path.join(env["root"], "01_copy_suffix", "מסמכים - עותק")

    def decide(payload):
        if payload["type"] == "empty_folders":
            return {"delete": payload["folders"][:1]}
        if payload["type"] == "duplicate" and "מסמכים - עותק" in names(payload):
            return {"action": "delete", "paths": [copy_path]}
        return {"action": "skip"}

    seen = run_to_end(app, cfg, decide)
    state = app.get_state(cfg).values

    # 04 was classified "different" and never shown to the user.
    assert not any(p["type"] == "duplicate" and "דוח 2024" in names(p) for p in seen)
    # Skipping 06 reveals its hidden children immediately after it.
    index = next(i for i, p in enumerate(seen) if p["type"] == "duplicate" and names(p) == ["אתר", "אתר גיבוי"])
    child = seen[index + 1]
    assert child["type"] == "duplicate" and names(child) != ["אתר", "אתר גיבוי"]
    assert all(os.sep + "06_nested" + os.sep in m["path"] for m in child["item"]["members"])
    shown_paths = [m["path"] for p in seen if p["type"] == "duplicate" for m in p["item"]["members"]]
    assert any(path.endswith(os.path.join("אתר", "תמונות")) for path in shown_paths)
    # ZIP items and the empty-folder screen were reached.
    assert sum(p["type"] == "zip" for p in seen) == 3
    assert seen[-1]["type"] == "empty_folders"

    assert env["trashed"][0] == copy_path and not os.path.exists(copy_path)
    assert copy_path in state["deleted_paths"]
    assert state["freed_bytes"] == 1_500_000
    rows = read_log(env["log"])
    deleted_row = next(r for r in rows if r["path"] == copy_path)
    assert deleted_row["action"] == actions.ACTION_MOVED
    assert deleted_row["llm_verdict"] == "identical"
    assert deleted_row["recommendation_outcome"] == actions.OUTCOME_FOLLOWED
    different_row = next(r for r in rows if r["action"] == actions.ACTION_DIFFERENT)
    assert different_row["recommendation_outcome"] == ""  # No user decision
    skipped_groups = [r for r in rows if r["item_type"] == "duplicate" and r["user_decision"] == "skip"]
    assert skipped_groups and all(r["recommendation_outcome"] == actions.OUTCOME_SKIPPED for r in skipped_groups)
    assert all(r["recommendation_outcome"] == "" for r in rows if r["item_type"] in ("zip", "empty_folder"))
    assert state["summary"]["deleted"] == 2  # The copy and one empty folder
    assert state["summary"]["classified_different"] == 1


def test_no_side_effects_before_interrupt(env):
    app = graph.build_graph(InMemorySaver())
    cfg = start(app, env["root"])
    assert current_payload(app, cfg) is not None
    assert read_log(env["log"]) == []  # Waiting at an interrupt wrote nothing
    app.invoke(Command(resume={"action": "skip"}), cfg)
    assert len(read_log(env["log"])) == 1  # Exactly one row, even though human_review re-ran


def test_outcome_overridden_and_no_recommendation(env, monkeypatch):
    app = graph.build_graph(InMemorySaver())
    cfg = start(app, env["root"])
    original = current_payload(app, cfg)["item"]["members"][0]["path"]  # Recommendation keeps A
    app.invoke(Command(resume={"action": "delete", "paths": [original]}), cfg)
    assert read_log(env["log"])[-1]["recommendation_outcome"] == actions.OUTCOME_OVERRIDDEN

    monkeypatch.setattr(graph.llm_judge, "judge_with_details",  # LLM failure after the retry
                        lambda group, prompt_version=None: llm_judge.JudgeResult(
                            None, ["timeout: ReadTimeout", "quota: ChatGoogleGenerativeAIError"]))
    app.invoke(Command(resume={"action": "skip"}), cfg)  # Skips group 2 (judged before the patch)
    app.invoke(Command(resume={"action": "skip"}), cfg)  # Group 3, judged with the failing LLM
    assert current_payload(app, cfg)["recommendation"] is None
    assert read_log(env["log"])[-1]["recommendation_outcome"] == ""
    # The failure type of the last attempt is reported to the user.
    warnings = app.get_state(cfg).values["warnings"]
    assert any("לא התקבלה המלצה מהמערכת עבור קבוצה" in w and "חריגה ממכסת השימוש" in w
               and "ChatGoogleGenerativeAIError" in w for w in warnings)


def test_thread_id_in_run_metadata():
    assert graph.thread_config("abc")["metadata"]["thread_id"] == "abc"


def test_cannot_delete_all_members(env):
    app = graph.build_graph(InMemorySaver())
    cfg = start(app, env["root"])
    payload = current_payload(app, cfg)
    all_paths = [m["path"] for m in payload["item"]["members"]]
    app.invoke(Command(resume={"action": "delete", "paths": all_paths}), cfg)
    state = app.get_state(cfg).values
    assert env["trashed"] == [] and all(os.path.exists(p) for p in all_paths)
    assert any("אסור למחוק את כל העותקים" in w for w in state["warnings"])


def test_changed_folder_is_not_deleted(env):
    app = graph.build_graph(InMemorySaver())
    cfg = start(app, env["root"])
    copy_path = current_payload(app, cfg)["item"]["members"][1]["path"]
    with open(os.path.join(copy_path, "חדש.txt"), "w", encoding="utf-8") as handle:
        handle.write("נוסף אחרי הסריקה")
    app.invoke(Command(resume={"action": "delete", "paths": [copy_path]}), cfg)
    state = app.get_state(cfg).values
    assert os.path.exists(copy_path) and env["trashed"] == []
    assert any("השתנתה מאז הסריקה" in w for w in state["warnings"])
    assert read_log(env["log"])[0]["action"] == actions.ACTION_NOT_DELETED


def test_resume_after_restart_with_sqlite(env, tmp_path):
    db = str(tmp_path / "checkpoints.sqlite")
    first = graph.open_checkpointer(db)
    app = graph.build_graph(first)
    cfg = start(app, env["root"], thread="restart")
    app.invoke(Command(resume={"action": "skip"}), cfg)
    before = current_payload(app, cfg)
    first.conn.close()  # Simulate closing the process

    app = graph.build_graph(graph.open_checkpointer(db))
    assert current_payload(app, cfg) == before  # Same pending item after the restart
    seen = run_to_end(app, cfg, lambda p: {"delete": []} if p["type"] == "empty_folders" else {"action": "skip"})
    assert seen[0] == before
    assert app.get_state(cfg).values["summary"] is not None


def test_pick_next_skips_items_inside_deleted_paths():
    deleted = r"C:\data\a"
    queue = [
        {"id": "g1", "type": "duplicate", "members": [{"path": r"C:\data\a\sub"}, {"path": r"C:\data\b"}]},
        {"id": "g2", "type": "duplicate", "members": [{"path": r"C:\data\ab"}, {"path": r"C:\data\c"}]},
    ]
    state = graph.initial_state([r"C:\data"])
    state.update(queue=queue, deleted_paths=[deleted])
    update = graph.pick_next(state)
    assert update["current"]["id"] == "g2"  # "C:\data\ab" is not inside "C:\data\a"
