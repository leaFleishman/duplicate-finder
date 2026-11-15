"""Both features together: "delete only the shared files" with cross-project groups hidden."""

import os
import shutil

import pytest
import streamlit as st
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from streamlit.testing.v1 import AppTest

import actions
import config
import graph
import llm_judge
from make_fixtures import create_fixtures

APP_FILE = os.path.join(config.PROJECT_DIR, "app.py")
BIDI = dict.fromkeys(map(ord, "⁦⁧⁨⁩"))


def fake_judge(group, prompt_version=None):
    if any(m["name"].startswith("דוח") for m in group["members"]):
        return llm_judge.Judgement(verdict="different", confidence=0.9, keep=[], reason="שונות")
    return llm_judge.Judgement(verdict="identical", confidence=0.95, keep=["A"], reason="זהות")


@pytest.fixture
def env(tmp_path, monkeypatch):
    root = create_fixtures(str(tmp_path))
    trashed = []

    def fake_trash(path):
        trashed.append(path)
        shutil.rmtree(path) if os.path.isdir(path) else os.remove(path)

    monkeypatch.setattr(graph.llm_judge, "judge_with_details",
                        lambda g, prompt_version=None: llm_judge.JudgeResult(fake_judge(g), []))
    monkeypatch.setattr(actions, "send2trash", fake_trash)
    monkeypatch.setattr(config, "ACTIONS_LOG_PATH", str(tmp_path / "log.csv"))
    monkeypatch.setattr(config, "CHECKPOINT_DB_PATH", str(tmp_path / "checkpoints.sqlite"))
    st.cache_resource.clear()
    yield {"root": root, "trashed": trashed}
    st.cache_resource.clear()


def names(payload):
    return [m["name"] for m in payload["item"]["members"]]


def run(root, include_cross_project, decide):
    app = graph.build_graph(InMemorySaver())
    cfg = graph.thread_config("t")
    settings = {**graph.default_settings(), "include_cross_project": include_cross_project}
    app.invoke(graph.initial_state([root], settings), cfg)
    seen = []
    while app.get_state(cfg).interrupts:
        payload = app.get_state(cfg).interrupts[0].value
        seen.append(payload)
        app.invoke(Command(resume=decide(payload)), cfg)
    return seen, app.get_state(cfg).values


def test_shared_delete_in_same_project_with_cross_project_hidden(env):
    """Case 12 (same project) is shown and can be thinned; cases 06/11 (cross-project) stay hidden."""
    root = env["root"]
    copy = os.path.join(root, "12_same_project", "אתר משרד", "lib", "bootstrap - Copy")
    keeper = os.path.join(root, "12_same_project", "אתר משרד", "lib", "bootstrap")

    def decide(payload):
        if payload["type"] == "empty_folders":
            return {"delete": []}
        if payload["type"] == "duplicate" and "bootstrap - Copy" in names(payload):
            assert "delete_shared" in payload["actions"]
            return {"action": "delete_shared", "from": "B", "keep_in": "A"}
        return {"action": "skip"}

    seen, state = run(root, include_cross_project=False, decide=decide)
    shown = [n for p in seen if p["type"] == "duplicate" for n in names(p)]
    assert "fontawesome" not in shown and "bootstrap - Copy" in shown
    assert os.listdir(copy) == [] and len(os.listdir(keeper)) == 2
    assert state["protected_paths"] == [keeper]
    # The emptied copy is offered in the empty-folders step; the hidden library folders (with files) are not.
    empty_step = next(p for p in seen if p["type"] == "empty_folders")
    assert copy in empty_step["folders"]
    assert not any("11_projects" in f for f in empty_step["folders"])
    # The summary reports both features.
    assert state["summary"]["shared_files_deleted"] == 2
    assert state["summary"]["skipped_cross_project"] == 3


def test_protection_applies_to_groups_that_are_shown(env):
    """After thinning a group, a later group whose members include the protected folder cannot delete it."""
    root = env["root"]
    keeper = os.path.join(root, "06_nested", "אתר")

    def decide(payload):
        if payload["type"] == "empty_folders":
            return {"delete": []}
        if payload["type"] != "duplicate":
            return {"action": "skip"}
        members = {m["name"]: m for m in payload["item"]["members"]}
        if set(members) == {"אתר", "אתר גיבוי"}:
            return {"action": "delete_shared", "from": members["אתר גיבוי"]["id"], "keep_in": members["אתר"]["id"]}
        return {"action": "skip"}

    seen, state = run(root, include_cross_project=True, decide=decide)
    assert keeper in state["protected_paths"]
    # With cross-project shown, try to delete the protected folder's images via a direct execute call.
    images = os.path.join(keeper, "תמונות")
    item = {"id": "gx", "type": "duplicate", "members": [
        {"id": "A", "path": images, "size_bytes": 0, "file_count": 0},
        {"id": "B", "path": os.path.join(root, "06_nested", "אתר גיבוי", "תמונות"), "size_bytes": 0, "file_count": 0},
    ]}
    update = graph.execute({**state, "current": item, "decision": {"action": "delete", "paths": [images]},
                            "judgement": None})
    assert os.path.isdir(images)
    assert any("העותק היחיד שנשאר" in w for w in update["warnings"])


def test_ui_summary_shows_both_numbers(env):
    root = env["root"]
    at = AppTest.from_file(APP_FILE, default_timeout=60)
    at.run()
    at.text_area(key="roots_text").input(root)
    at.button(key="start_button").click().run()
    while at.subheader and at.subheader[0].value.translate(BIDI) != "סיכום":
        heading = at.subheader[0].value
        shared = [b for b in at.button if ":shared:B:A" in b.key]
        current = str(at.session_state["current_state"]["current"])
        if heading.startswith("תיקיות ריקות"):
            next(b for b in at.button if b.label == "מחק את המסומנות").click().run()
        elif "bootstrap - Copy" in current and shared:
            shared[0].click().run()
        else:
            next(b for b in at.button if b.label == "דלג").click().run()
        assert not at.exception
    captions = " ".join(c.value.translate(BIDI) for c in at.caption)
    assert "בנוסף נמחקו 2 קבצים משותפים" in captions
    assert "קבוצות כפילות בין פרויקטים שונים שלא הוצגו: 3" in captions
