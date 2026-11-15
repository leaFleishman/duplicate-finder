"""Cross-project groups, even when shown, can only be skipped (UI, CLI, graph and execute)."""

import os
import shutil

import pytest
import streamlit as st
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from streamlit.testing.v1 import AppTest

import actions
import cli
import config
import graph
import llm_judge
from make_fixtures import create_fixtures

APP_FILE = os.path.join(config.PROJECT_DIR, "app.py")
MESSAGE = "התיקיות שייכות לפרויקטים שונים. מחיקה עלולה לשבור את הפרויקט, ולכן אינה אפשרית כאן."
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


def start(root):
    app = graph.build_graph(InMemorySaver())
    cfg = graph.thread_config("t")
    app.invoke(graph.initial_state([root], {**graph.default_settings(), "include_cross_project": True}), cfg)
    return app, cfg


def payload_of(app, cfg):
    interrupts = app.get_state(cfg).interrupts
    return interrupts[0].value if interrupts else None


def advance_to_library(app, cfg):
    """Skip until the fontawesome group (case 11, members in two different projects)."""
    while True:
        payload = payload_of(app, cfg)
        assert payload is not None
        if payload["type"] == "duplicate" and {m["name"] for m in payload["item"]["members"]} == {"fontawesome"}:
            return payload
        app.invoke(Command(resume={"action": "skip"} if payload["type"] != "empty_folders" else {"delete": []}), cfg)


# --------------------------------------------------------------------- Graph


def test_payload_offers_only_skip_with_the_message(env):
    app, cfg = start(env["root"])
    payload = advance_to_library(app, cfg)
    assert payload["actions"] == ["skip"]
    assert payload["blocked_reason"] == MESSAGE


def test_regular_groups_are_not_blocked(env):
    app, cfg = start(env["root"])
    payload = payload_of(app, cfg)  # Case 01, no projects
    assert payload["blocked_reason"] is None and "delete" in payload["actions"]


@pytest.mark.parametrize("decision", [
    lambda item: {"action": "delete", "paths": [item["members"][1]["path"]]},
    lambda item: {"action": "delete_shared", "from": "B", "keep_in": "A"},
])
def test_any_deletion_request_becomes_a_skip(env, decision):
    app, cfg = start(env["root"])
    payload = advance_to_library(app, cfg)
    item = payload["item"]
    app.invoke(Command(resume=decision(item)), cfg)
    state = app.get_state(cfg).values
    assert env["trashed"] == []
    assert all(os.path.isdir(m["path"]) and os.listdir(m["path"]) for m in item["members"])
    assert any(MESSAGE in w for w in state["warnings"])


def test_execute_refuses_even_if_a_deletion_arrives(env):
    """Extra guard: execute itself refuses, without relying on human_review."""
    app, cfg = start(env["root"])
    item = advance_to_library(app, cfg)["item"]
    state = app.get_state(cfg).values
    for decision in ({"action": "delete", "paths": [item["members"][0]["path"]]},
                     {"action": "delete_shared", "from": "A", "keep_in": "B"}):
        update = graph.execute({**state, "current": item, "decision": decision, "judgement": None})
        assert env["trashed"] == []
        assert any(MESSAGE in w for w in update["warnings"])
        assert update["log"][-1]["action"] == actions.ACTION_NOT_DELETED
        assert "deleted_paths" not in update and "protected_paths" not in update
    assert all(os.listdir(m["path"]) for m in item["members"])


# ----------------------------------------------------------------------- CLI


def test_cli_only_accepts_skip(monkeypatch, capsys):
    answers = iter(["d B", "x B A", "s"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    payload = {"item": {"members": [{"id": "A", "path": "a"}, {"id": "B", "path": "b"}]},
               "actions": ["skip"], "blocked_reason": MESSAGE}
    assert cli.read_duplicate_decision(payload) == {"action": "skip"}
    output = capsys.readouterr().out
    assert MESSAGE in output and output.count("אפשר רק לדלג") == 2


# ------------------------------------------------------------------------ UI


def test_ui_disables_deletion_and_keeps_skip(env):
    at = AppTest.from_file(APP_FILE, default_timeout=60)
    at.run()
    at.checkbox(key="include_cross_project").check().run()
    at.text_area(key="roots_text").input(env["root"])
    at.button(key="start_button").click().run()
    while "fontawesome" not in str(at.session_state["current_state"]["current"]):
        assert at.subheader[0].value.translate(BIDI) != "סיכום", "library group not reached"
        next(b for b in at.button if b.label == "דלג").click().run()
    assert any(MESSAGE in w.value for w in at.warning)
    delete = next(b for b in at.button if b.label == "מחק את המסומנים")
    shared = [b for b in at.button if ":shared:" in b.key]
    skip = next(b for b in at.button if b.label == "דלג")
    assert delete.disabled and shared and all(b.disabled for b in shared)
    assert all(c.disabled for c in at.checkbox if c.key and c.key.endswith((":A", ":B")))
    assert not skip.disabled
    blocked_id = at.session_state["current_state"]["current"]["id"]
    skip.click().run()
    assert not at.exception
    # Skipping moves on (here to its hidden webfonts group, which is also cross-project and also blocked).
    current = at.session_state["current_state"]["current"]
    assert current["id"] != blocked_id
    assert any(MESSAGE in w.value for w in at.warning)
    assert env["trashed"] == []
