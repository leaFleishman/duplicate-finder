"""Duplicates whose members are all in different projects are hidden by default."""

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
import scanner
from make_fixtures import create_fixtures

APP_FILE = os.path.join(config.PROJECT_DIR, "app.py")
LIBRARY = {r"11_projects\אתר בלוג\fontawesome", r"11_projects\אתר חנות\fontawesome"}
LIBRARY_WEBFONTS = {r"11_projects\אתר בלוג\fontawesome\webfonts", r"11_projects\אתר חנות\fontawesome\webfonts"}
NESTED_IMAGES = {r"06_nested\אתר\תמונות", r"06_nested\אתר גיבוי\תמונות"}
SAME_PROJECT = {r"12_same_project\אתר משרד\lib\bootstrap", r"12_same_project\אתר משרד\lib\bootstrap - Copy"}


@pytest.fixture(scope="module")
def scanned(tmp_path_factory):
    root = create_fixtures(str(tmp_path_factory.mktemp("data")))
    return root, scanner.scan([root])


def find(scanned, include):
    root, result = scanned
    return scanner.find_candidates(result, config.DEFAULT_OVERLAP_THRESHOLD, config.DEFAULT_MIN_FOLDER_SIZE_MB,
                                   include_cross_project=include)


def member_sets(root, groups):
    return [{os.path.relpath(m["path"], root) for m in g["members"]} for g in groups if g["type"] == "duplicate"]


def all_groups(candidates):
    hidden = [g for children in candidates["hidden_children"].values() for g in children]
    return [g for g in candidates["queue"] if g["type"] == "duplicate"] + hidden


# ------------------------------------------------------------------ Scanner


def test_library_in_two_projects_is_hidden_by_default(scanned):
    root = scanned[0]
    candidates = find(scanned, include=False)
    shown = member_sets(root, all_groups(candidates))
    assert LIBRARY not in shown and LIBRARY_WEBFONTS not in shown
    assert NESTED_IMAGES not in shown  # Also inside two (site) projects
    assert candidates["skipped_cross_project"] == 3


def test_library_in_two_projects_is_shown_when_enabled(scanned):
    root = scanned[0]
    candidates = find(scanned, include=True)
    assert LIBRARY in member_sets(root, candidates["queue"])
    assert NESTED_IMAGES in member_sets(root, all_groups(candidates))
    assert candidates["skipped_cross_project"] == 0


def test_duplicates_inside_the_same_project_are_still_shown(scanned):
    root = scanned[0]
    candidates = find(scanned, include=False)
    group = next(g for g in candidates["queue"]
                 if g["type"] == "duplicate" and member_sets(root, [g])[0] == SAME_PROJECT)
    projects = {m["project"] for m in group["members"]}
    assert len(projects) == 1 and os.path.relpath(projects.pop(), root) == r"12_same_project\אתר משרד"


def test_other_groups_are_unaffected(scanned):
    root = scanned[0]
    default, everything = find(scanned, include=False), find(scanned, include=True)
    removed = [s for s in member_sets(root, all_groups(everything)) if s not in member_sets(root, all_groups(default))]
    assert sorted(map(sorted, removed)) == sorted(map(sorted, [LIBRARY, LIBRARY_WEBFONTS, NESTED_IMAGES]))
    # ZIP items are never affected.
    assert [i for i in default["queue"] if i["type"] == "zip"] == [i for i in everything["queue"] if i["type"] == "zip"]


def test_empty_folders_do_not_depend_on_the_filter(scanned):
    root = scanned[0]
    default, everything = find(scanned, include=False), find(scanned, include=True)
    assert default["empty_folders"] == everything["empty_folders"]
    # Folders with files (the hidden library folders) are never listed as empty.
    for path in default["empty_folders"]:
        assert not path.startswith(os.path.join(root, "11_projects"))
        assert scanner.is_empty_on_disk(path)


# ------------------------------------------------------------ Rule itself


def group(*projects):
    return {"id": "g", "type": "duplicate",
            "members": [{"id": chr(65 + n), "path": f"p{n}", "project": p} for n, p in enumerate(projects)]}


@pytest.mark.parametrize("projects, cross", [
    ((r"C:\a", r"C:\b"), True),               # Every member in a different project
    ((r"C:\a", r"C:\b", r"C:\c"), True),
    ((r"C:\a", r"C:\a"), False),              # Same project
    ((r"C:\a", r"C:\A"), False),              # Same project, different case (Windows)
    ((r"C:\a", r"C:\a", r"C:\b"), False),     # At least two members in the same project
    ((r"C:\a", None), False),                 # One member is not in a project
    ((None, None), False),
])
def test_is_cross_project(projects, cross):
    assert scanner.is_cross_project(group(*projects)) is cross


def test_hidden_groups_of_a_removed_group_move_up():
    """If a cross-project group is removed, its hidden groups that are not cross-project take its place."""
    parent = {**group(r"C:\a", r"C:\b"), "id": "g1"}
    child = {**group(r"C:\a", None), "id": "g2"}       # Not cross-project
    grandchild = {**group(r"C:\x", r"C:\y"), "id": "g3"}  # Cross-project
    other = {**group(None, None), "id": "g4"}
    top, hidden, skipped = scanner.drop_cross_project([parent, other], {"g1": [child], "g2": [grandchild]})
    assert [g["id"] for g in top] == ["g2", "g4"]
    assert hidden == {} and skipped == 2


# --------------------------------------------------------------- Graph and UI


def fake_judge(group_, prompt_version=None):
    if any(m["name"].startswith("דוח") for m in group_["members"]):
        return llm_judge.Judgement(verdict="different", confidence=0.9, keep=[], reason="שונות")
    return llm_judge.Judgement(verdict="identical", confidence=0.95, keep=["A"], reason="זהות")


@pytest.fixture
def env(tmp_path, monkeypatch):
    root = create_fixtures(str(tmp_path))
    monkeypatch.setattr(graph.llm_judge, "judge_with_details",
                        lambda g, prompt_version=None: llm_judge.JudgeResult(fake_judge(g), []))
    monkeypatch.setattr(actions, "send2trash", lambda path: None)
    monkeypatch.setattr(config, "ACTIONS_LOG_PATH", str(tmp_path / "log.csv"))
    monkeypatch.setattr(config, "CHECKPOINT_DB_PATH", str(tmp_path / "checkpoints.sqlite"))
    st.cache_resource.clear()
    yield root
    st.cache_resource.clear()


def run_all(root, include):
    app = graph.build_graph(InMemorySaver())
    cfg = graph.thread_config("t")
    app.invoke(graph.initial_state([root], {**graph.default_settings(), "include_cross_project": include}), cfg)
    shown = []
    while app.get_state(cfg).interrupts:
        payload = app.get_state(cfg).interrupts[0].value
        if payload["type"] == "duplicate":
            shown.append({os.path.relpath(m["path"], root) for m in payload["item"]["members"]})
        decision = {"delete": []} if payload["type"] == "empty_folders" else {"action": "skip"}
        app.invoke(Command(resume=decision), cfg)
    return shown, app.get_state(cfg).values["summary"]


def test_graph_default_hides_and_counts(env):
    shown, summary = run_all(env, include=False)
    assert LIBRARY not in shown and SAME_PROJECT in shown
    assert summary["skipped_cross_project"] == 3


def test_graph_with_checkbox_shows_the_library(env):
    shown, summary = run_all(env, include=True)
    assert LIBRARY in shown and SAME_PROJECT in shown
    assert summary["skipped_cross_project"] == 0


def test_default_setting_is_off():
    assert graph.default_settings()["include_cross_project"] is False


BIDI = dict.fromkeys(map(ord, "\u2066\u2067\u2068\u2069"))


def walk_ui(root, include):
    at = AppTest.from_file(APP_FILE, default_timeout=60)
    at.run()
    assert at.checkbox(key="include_cross_project").value is False  # Off by default
    if include:
        at.checkbox(key="include_cross_project").check().run()
    at.text_area(key="roots_text").input(root)
    at.button(key="start_button").click().run()
    seen = []
    while at.subheader and at.subheader[0].value.translate(BIDI) != "סיכום":
        current = at.session_state["current_state"]["current"]
        if current:
            seen.append(str(current))
        if at.subheader[0].value.startswith("תיקיות ריקות"):
            next(b for b in at.button if b.label == "מחק את המסומנות").click().run()
        else:
            next(b for b in at.button if b.label == "דלג").click().run()
        assert not at.exception
    captions = " ".join(c.value.translate(BIDI) for c in at.caption)
    return seen, captions


def test_ui_default_hides_and_reports_the_count(env):
    seen, captions = walk_ui(env, include=False)
    assert not any("fontawesome" in s for s in seen)
    assert "קבוצות כפילות בין פרויקטים שונים שלא הוצגו: 3" in captions


def test_ui_checkbox_shows_the_library(env):
    seen, captions = walk_ui(env, include=True)
    assert any("fontawesome" in s for s in seen)
    assert "שלא הוצגו" not in captions
