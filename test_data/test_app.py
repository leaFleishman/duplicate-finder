"""Streamlit UI tests with AppTest (fake judge, fake Recycle Bin, temporary checkpoint DB)."""

import os
import shutil

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import actions
import config
import graph
import llm_judge
from make_fixtures import create_fixtures

APP_FILE = os.path.join(config.PROJECT_DIR, "app.py")


def fake_judge(group, prompt_version=None):
    if any(m["name"].startswith("דוח") for m in group["members"]):
        return llm_judge.Judgement(verdict="different", confidence=0.9, keep=[], reason="שונות")
    if any(m["name"] == "random_stuff_17" for m in group["members"]):
        return llm_judge.Judgement(verdict="identical", confidence=0.4, keep=["B"], reason="לא בטוח")
    return llm_judge.Judgement(verdict="identical", confidence=0.95, keep=["A"], reason="זהות")


@pytest.fixture
def env(tmp_path, monkeypatch):
    root = create_fixtures(str(tmp_path))
    trashed = []

    def fake_trash(path):
        trashed.append(path)
        shutil.rmtree(path) if os.path.isdir(path) else os.remove(path)

    monkeypatch.setattr(graph.llm_judge, "judge_with_details",
                        lambda group, prompt_version=None: llm_judge.JudgeResult(fake_judge(group), []))
    monkeypatch.setattr(actions, "send2trash", fake_trash)
    monkeypatch.setattr(config, "ACTIONS_LOG_PATH", str(tmp_path / "log.csv"))
    monkeypatch.setattr(config, "CHECKPOINT_DB_PATH", str(tmp_path / "checkpoints.sqlite"))
    st.cache_resource.clear()  # Each test gets a graph bound to its own temporary DB
    yield {"root": root, "trashed": trashed}
    st.cache_resource.clear()


def new_session() -> AppTest:
    at = AppTest.from_file(APP_FILE, default_timeout=60)
    at.run()
    return at


def start_scan(at: AppTest, root: str) -> None:
    at.text_area(key="roots_text").input(root)
    at.button(key="start_button").click().run()


def button(at: AppTest, label: str):
    return next(b for b in at.button if b.label == label)


BIDI_MARKS = dict.fromkeys(map(ord, "⁦⁧⁨⁩"))


def heading(at: AppTest) -> str:
    """The first subheader, without the Unicode direction-isolation marks."""
    return at.subheader[0].value.translate(BIDI_MARKS) if at.subheader else ""


def test_invalid_paths_show_errors(env):
    at = new_session()
    at.text_area(key="roots_text").input(r"C:\no\such\folder" + "\n" + os.path.join(env["root"], "x.txt"))
    at.button(key="start_button").click().run()
    assert any("הנתיב לא קיים" in e.value for e in at.error)
    assert "thread_id" not in at.session_state


def test_skip_list_root_is_rejected(env, tmp_path):
    skipped = tmp_path / "node_modules"
    skipped.mkdir()
    at = new_session()
    start_scan(at, str(skipped))
    assert any("ברשימת הדילוג" in e.value for e in at.error)


def test_full_flow_in_the_ui(env):
    at = new_session()
    start_scan(at, env["root"])
    assert not at.exception
    assert heading(at).startswith("קבוצת כפילויות g1")

    # Deleting every member is blocked.
    for box in at.checkbox:
        box.check()
    at.run()
    assert button(at, "מחק את המסומנים").disabled
    assert any("אי אפשר למחוק את כל התיקיות" in e.value for e in at.error)

    # Delete only the copy (B).
    at.checkbox[0].uncheck()
    at.run()
    copy_path = os.path.join(env["root"], "01_copy_suffix", "מסמכים - עותק")
    button(at, "מחק את המסומנים").click().run()
    assert env["trashed"] == [copy_path]

    seen = []
    while at.subheader and heading(at) != "סיכום":
        seen.append(heading(at))
        if heading(at).startswith("תיקיות ריקות"):
            button(at, "בחר הכל").click().run()
            assert all(box.value for box in at.checkbox)
            button(at, "נקה הכל").click().run()
            at.checkbox[0].check()
            at.run()
            button(at, "מחק את המסומנות").click().run()
        else:
            if "random_stuff_17" in str(at.session_state["current_state"]["current"]):
                assert any("המערכת לא בטוחה, כדאי לבדוק" in e.value for e in at.error)
            if "תזה" in str(at.session_state["current_state"]["current"]):
                assert any("קבצים אלה יימחקו אם תמחקי תיקייה זו" in w.value for w in at.warning)
            button(at, "דלג").click().run()
        assert not at.exception

    assert sum(h.startswith("קובץ ZIP") for h in seen) == 3
    assert any("חולץ חלקית" in h for h in seen)
    assert heading(at) == "סיכום"
    assert len(env["trashed"]) == 2  # The copy and one empty folder


def test_double_click_does_not_resume_twice(env):
    at = new_session()
    start_scan(at, env["root"])
    interrupt_id = button(at, "דלג").key.rsplit(":", 1)[0]
    copy_path = os.path.join(env["root"], "01_copy_suffix", "מסמכים - עותק")

    button(at, "דלג").click().run()
    assert heading(at).startswith("קבוצת כפילויות g2")

    # The same button clicked again (stale interrupt id), this time even asking to delete.
    at.session_state["request"] = {"kind": "resume", "interrupt_id": interrupt_id,
                                   "decision": {"action": "delete", "paths": [copy_path]}}
    at.run()
    assert env["trashed"] == [] and os.path.exists(copy_path)
    assert heading(at).startswith("קבוצת כפילויות g2")  # Still waiting on the same item


def test_start_token_is_rotated_after_use(env):
    at = new_session()
    token = at.session_state["start_token"]
    start_scan(at, env["root"])
    assert at.session_state["start_token"] != token  # A stale second click carries the old token


def test_resume_in_new_session_after_restart(env):
    at = new_session()
    start_scan(at, env["root"])
    button(at, "דלג").click().run()
    thread_id = at.session_state["thread_id"]
    waiting_on = heading(at)

    st.cache_resource.clear()  # Like restarting streamlit: new graph and DB connection
    at = new_session()         # Like a new browser session: empty session_state
    assert "thread_id" not in at.session_state
    at.text_input(key="thread_input").input(thread_id)
    at.button(key="open_thread_button").click().run()
    assert heading(at) == waiting_on

    at = new_session()
    at.text_input(key="thread_input").input("no-such-thread")
    at.button(key="open_thread_button").click().run()
    assert any("לא נמצאה הרצה" in e.value for e in at.error)
