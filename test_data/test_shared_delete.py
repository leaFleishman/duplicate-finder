"""Tests for "delete only the shared files" (fake judge, fake Recycle Bin, temporary log)."""

import csv
import os
import shutil

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from streamlit.testing.v1 import AppTest
import streamlit as st

import actions
import config
import graph
import llm_judge
import scanner
from make_fixtures import KB, _write_files, create_fixtures

APP_FILE = os.path.join(config.PROJECT_DIR, "app.py")
SHARED = {"action": "delete_shared", "from": "B", "keep_in": "A"}


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
                        lambda group, prompt_version=None: llm_judge.JudgeResult(fake_judge(group), []))
    monkeypatch.setattr(actions, "send2trash", fake_trash)
    monkeypatch.setattr(config, "ACTIONS_LOG_PATH", str(tmp_path / "log.csv"))
    monkeypatch.setattr(config, "CHECKPOINT_DB_PATH", str(tmp_path / "checkpoints.sqlite"))
    st.cache_resource.clear()
    yield {"root": root, "trashed": trashed, "log": str(tmp_path / "log.csv")}
    st.cache_resource.clear()


def payload_of(app, cfg):
    interrupts = app.get_state(cfg).interrupts
    return interrupts[0].value if interrupts else None


def names(payload):
    return [m["name"] for m in payload["item"]["members"]]


def advance_to(app, cfg, member_name):
    """Skip groups until the one that contains the given member."""
    while True:
        payload = payload_of(app, cfg)
        assert payload is not None, f"group with {member_name} not found"
        if payload["type"] == "duplicate" and member_name in names(payload):
            return payload
        app.invoke(Command(resume={"action": "skip"}), cfg)


def start(app, root):
    cfg = graph.thread_config("t1")
    app.invoke(graph.initial_state([root]), cfg)
    return cfg


def read_log(path):
    with open(path, encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def shared_rows(path):
    return [r for r in read_log(path) if r["action"] == actions.ACTION_SHARED_DELETED]


# ------------------------------------------------------------------ Scanner


@pytest.fixture(scope="module")
def groups(tmp_path_factory):
    root = create_fixtures(str(tmp_path_factory.mktemp("data")))
    result = scanner.scan([root])
    candidates = scanner.find_candidates(result, config.DEFAULT_OVERLAP_THRESHOLD, config.DEFAULT_MIN_FOLDER_SIZE_MB)
    return root, {m["name"]: g for g in candidates["queue"] if g["type"] == "duplicate" for m in g["members"]}


def test_scanner_lists_shared_files_per_member(groups):
    root, by_name = groups
    group = by_name["תזה - עותק"]
    copy = next(m for m in group["members"] if m["name"] == "תזה - עותק")
    original = next(m for m in group["members"] if m["name"] == "תזה")
    assert copy["shared_with"][original["id"]]["count"] == 4
    assert copy["shared_with"][original["id"]]["size_bytes"] == 1_260_000
    # The unique file is never listed as shared, in either direction.
    for member in (copy, original):
        for info in member["shared_with"].values():
            assert "פרק 5 - חדש.docx" not in [f["relative_path"] for f in info["files"]]
    assert original["shared_with"][copy["id"]]["count"] == 4


def test_shared_entries_point_to_an_identical_file(groups):
    root, by_name = groups
    group = by_name["תזה - עותק"]
    members = {m["id"]: m for m in group["members"]}
    for member in members.values():
        for other_id, info in member["shared_with"].items():
            for entry in info["files"]:
                path = os.path.join(member["path"], entry["relative_path"])
                copy = os.path.join(members[other_id]["path"], entry["other_relative_path"])
                assert scanner.sha256_file(path) == scanner.sha256_file(copy) == entry["sha256"]


def test_shared_files_found_across_different_structures(tmp_path):
    """Same content, different file names and folders: the copy path is mapped correctly."""
    _write_files(str(tmp_path / "X"), {"docs/one.bin": ("s1", 600 * KB), "two.bin": ("s2", 600 * KB)})
    _write_files(str(tmp_path / "Y"), {"other/renamed.bin": ("s1", 600 * KB), "deep/er/two2.bin": ("s2", 600 * KB)})
    candidates = scanner.find_candidates(scanner.scan([str(tmp_path)]), 0.7, 0)
    group = next(g for g in candidates["queue"] if g["type"] == "duplicate")
    x = next(m for m in group["members"] if m["name"] == "X")
    y = next(m for m in group["members"] if m["name"] == "Y")
    entries = {e["relative_path"]: e["other_relative_path"] for e in x["shared_with"][y["id"]]["files"]}
    assert entries == {os.path.join("docs", "one.bin"): os.path.join("other", "renamed.bin"),
                       "two.bin": os.path.join("deep", "er", "two2.bin")}


# ------------------------------------------------------------- Revalidation


def make_pair(tmp_path, content=b"same content"):
    a, b = tmp_path / "a.bin", tmp_path / "b.bin"
    a.write_bytes(content)
    b.write_bytes(content)
    return str(a), str(b), scanner.sha256_file(str(a))


def test_verify_shared_file_ok(tmp_path):
    path, copy, digest = make_pair(tmp_path)
    assert actions.verify_shared_file(path, copy, digest) is None


@pytest.mark.parametrize("change, expected", [
    (lambda path, copy: os.remove(path), "כבר לא קיים"),
    (lambda path, copy: os.remove(copy), "העותק בתיקייה השנייה כבר לא קיים"),
    (lambda path, copy: open(path, "ab").write(b"x"), "הקובץ השתנה מאז הסריקה"),
    (lambda path, copy: open(copy, "ab").write(b"x"), "העותק בתיקייה השנייה השתנה"),
])
def test_verify_shared_file_rejects_any_change(tmp_path, change, expected):
    path, copy, digest = make_pair(tmp_path)
    change(path, copy)
    assert expected in actions.verify_shared_file(path, copy, digest)


# --------------------------------------------------------------------- Graph


def test_delete_shared_keeps_unique_files(env):
    app = graph.build_graph(InMemorySaver())
    cfg = start(app, env["root"])
    payload = advance_to(app, cfg, "תזה - עותק")
    assert "delete_shared" in payload["actions"]
    original = os.path.join(env["root"], "03_newer_copy", "תזה")
    copy = os.path.join(env["root"], "03_newer_copy", "תזה - עותק")

    app.invoke(Command(resume=SHARED), cfg)
    state = app.get_state(cfg).values

    assert os.listdir(copy) == ["פרק 5 - חדש.docx"]  # Only the unique file is left
    assert len(os.listdir(original)) == 4  # The keeper is untouched
    assert len(env["trashed"]) == 4 and all(p.startswith(copy) for p in env["trashed"])
    assert state["freed_bytes"] == 1_260_000 and state["shared_files_deleted"] == 4
    assert state["protected_paths"] == [original]
    assert state["deleted_paths"] == []  # No whole folder was deleted


def test_overriding_the_recommendation_is_logged(env):
    """Thinning the member the LLM wanted to keep is logged as overridden."""
    app = graph.build_graph(InMemorySaver())
    cfg = start(app, env["root"])
    advance_to(app, cfg, "תזה - עותק")
    app.invoke(Command(resume={"action": "delete_shared", "from": "A", "keep_in": "B"}), cfg)
    assert shared_rows(env["log"])[0]["recommendation_outcome"] == actions.OUTCOME_OVERRIDDEN


def test_log_has_one_row_per_action(env):
    app = graph.build_graph(InMemorySaver())
    cfg = start(app, env["root"])
    advance_to(app, cfg, "תזה - עותק")
    app.invoke(Command(resume=SHARED), cfg)
    rows = shared_rows(env["log"])
    assert len(rows) == 1
    row = rows[0]
    assert row["files_deleted"] == "4" and row["size_bytes"] == "1260000" and row["skipped_files"] == ""
    assert row["user_decision"] == "delete_shared" and row["item_type"] == "duplicate"
    assert row["path"].endswith("תזה - עותק") and row["llm_verdict"] == "identical"
    # The fake judge recommends keeping A, and the files were removed from B: matches the recommendation.
    assert row["recommendation_outcome"] == actions.OUTCOME_FOLLOWED


@pytest.mark.parametrize("change, skipped, freed, warning", [
    # A file changed between the scan and the deletion (in the folder being thinned).
    (lambda a, b: open(os.path.join(b, "פרק 2.docx"), "ab").write(b"edited"), "פרק 2.docx", 950_000,
     "הקובץ השתנה מאז הסריקה"),
    # The copy in the other folder changed.
    (lambda a, b: open(os.path.join(a, "פרק 1.docx"), "ab").write(b"edited"), "פרק 1.docx", 960_000,
     "העותק בתיקייה השנייה השתנה"),
    # The copy in the other folder disappeared.
    (lambda a, b: os.remove(os.path.join(a, "פרק 3.docx")), "פרק 3.docx", 940_000,
     "העותק בתיקייה השנייה כבר לא קיים"),
])
def test_file_changed_after_scan_is_not_deleted(env, change, skipped, freed, warning):
    app = graph.build_graph(InMemorySaver())
    cfg = start(app, env["root"])
    advance_to(app, cfg, "תזה - עותק")
    original = os.path.join(env["root"], "03_newer_copy", "תזה")
    copy = os.path.join(env["root"], "03_newer_copy", "תזה - עותק")
    change(original, copy)

    app.invoke(Command(resume=SHARED), cfg)
    state = app.get_state(cfg).values

    assert os.path.exists(os.path.join(copy, skipped))  # Not deleted
    assert len(env["trashed"]) == 3 and state["shared_files_deleted"] == 3
    assert state["freed_bytes"] == freed
    assert any(warning in w and skipped in w for w in state["warnings"])
    row = shared_rows(env["log"])[0]
    assert row["files_deleted"] == "3" and row["skipped_files"] == skipped


def test_nothing_deleted_when_every_file_changed(env):
    app = graph.build_graph(InMemorySaver())
    cfg = start(app, env["root"])
    advance_to(app, cfg, "תזה - עותק")
    original = os.path.join(env["root"], "03_newer_copy", "תזה")
    for name in os.listdir(original):
        with open(os.path.join(original, name), "ab") as handle:
            handle.write(b"edited")
    app.invoke(Command(resume=SHARED), cfg)
    state = app.get_state(cfg).values
    assert env["trashed"] == [] and state["shared_files_deleted"] == 0 and state["protected_paths"] == []
    assert shared_rows(env["log"])[0]["files_deleted"] == "0"


def test_emptied_folder_appears_in_the_empty_folders_step(env):
    app = graph.build_graph(InMemorySaver())
    cfg = start(app, env["root"])
    advance_to(app, cfg, "מסמכים - עותק")  # Case 01: all three files are shared
    copy = os.path.join(env["root"], "01_copy_suffix", "מסמכים - עותק")
    app.invoke(Command(resume=SHARED), cfg)
    assert os.listdir(copy) == []

    payload = payload_of(app, cfg)
    while payload["type"] != "empty_folders":
        app.invoke(Command(resume={"action": "skip"}), cfg)
        payload = payload_of(app, cfg)
    assert copy in payload["folders"]
    assert os.path.join(env["root"], "01_copy_suffix", "מסמכים") not in payload["folders"]


def test_folder_with_remaining_files_is_not_listed_as_empty(env):
    app = graph.build_graph(InMemorySaver())
    cfg = start(app, env["root"])
    advance_to(app, cfg, "תזה - עותק")  # Keeps its unique file
    copy = os.path.join(env["root"], "03_newer_copy", "תזה - עותק")
    app.invoke(Command(resume=SHARED), cfg)
    payload = payload_of(app, cfg)
    while payload["type"] != "empty_folders":
        app.invoke(Command(resume={"action": "skip"}), cfg)
        payload = payload_of(app, cfg)
    assert copy not in payload["folders"]


# ------------------------------------------------------- Protection and rules


def state_for_execute(item, decision, protected):
    state = graph.initial_state(["x"])
    state.update(current=item, decision=decision, judgement=None, protected_paths=protected)
    return state


def test_protected_folder_is_never_deleted(env, tmp_path):
    keeper, other = tmp_path / "keeper", tmp_path / "other"
    keeper.mkdir()
    other.mkdir()
    (keeper / "f.bin").write_bytes(b"data")
    item = {"id": "g9", "type": "duplicate", "members": [
        {"id": "A", "path": str(keeper), "size_bytes": 4, "file_count": 1},
        {"id": "B", "path": str(other), "size_bytes": 0, "file_count": 0},
    ]}
    update = graph.execute(state_for_execute(item, {"action": "delete", "paths": [str(keeper)]}, [str(keeper)]))
    assert env["trashed"] == [] and keeper.exists()
    assert any("העותק היחיד שנשאר" in w for w in update["warnings"])
    assert update["log"][0]["action"] == actions.ACTION_NOT_DELETED


@pytest.mark.parametrize("path, blocked", [
    (r"C:\data\keeper", True),             # The protected folder itself
    (r"C:\data\keeper\sub\f.bin", True),   # Inside it
    (r"C:\data", True),                    # A folder that contains it
    (r"C:\data\keeper2", False),           # Similar name, unrelated
    (r"C:\other", False),
])
def test_protection_rule(path, blocked):
    assert (graph._protection_problem(path, [r"C:\data\keeper"]) is not None) == blocked


def shared_item():
    info = {"count": 1, "size_bytes": 4, "files": [
        {"relative_path": "f", "size_bytes": 4, "sha256": "h", "other_relative_path": "f"}]}
    return {"id": "g1", "type": "duplicate", "members": [
        {"id": "A", "path": "pa", "shared_with": {"B": info}},
        {"id": "B", "path": "pb", "shared_with": {"A": info}},
    ]}


IDENTICAL = {"verdict": "identical", "keep": ["A"], "confidence": 0.9, "reason": "x"}


def test_sanitize_accepts_a_valid_request():
    decision, warning = graph._sanitize_decision(shared_item(), SHARED, IDENTICAL)
    assert decision == SHARED and warning is None


@pytest.mark.parametrize("request_, judgement", [
    (SHARED, None),                                                        # No recommendation (LLM failed)
    (SHARED, {**IDENTICAL, "verdict": "different"}),                       # Not copies
    ({"action": "delete_shared", "from": "B", "keep_in": "B"}, IDENTICAL),  # Same member
    ({"action": "delete_shared", "from": "B", "keep_in": "Z"}, IDENTICAL),  # Unknown member
    ({"action": "delete_shared", "from": "Q", "keep_in": "A"}, IDENTICAL),
    ({"action": "delete_shared"}, IDENTICAL),
])
def test_sanitize_rejects_invalid_requests(request_, judgement):
    decision, warning = graph._sanitize_decision(shared_item(), request_, judgement)
    assert decision == {"action": "skip"} and "קבצים משותפים" in warning


def test_action_is_offered_only_for_copies(env):
    app = graph.build_graph(InMemorySaver())
    cfg = start(app, env["root"])
    payload = payload_of(app, cfg)
    assert payload["actions"] == ["delete", "delete_shared", "skip"]
    assert graph.shared_delete_allowed(payload["item"], None) is False
    assert graph.shared_delete_allowed(payload["item"], {**IDENTICAL, "verdict": "different"}) is False


def test_llm_input_has_no_shared_file_lists():
    group = shared_item()
    cleaned = graph._for_llm(group)
    assert all("shared_with" not in m for m in cleaned["members"])
    assert "shared_with" in group["members"][0]  # The original is not modified


# ---------------------------------------------------------------------- Log


def test_old_log_header_is_upgraded(tmp_path):
    log = tmp_path / "old.csv"
    old_columns = actions.LOG_COLUMNS[:-2]
    with open(log, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(old_columns)
        writer.writerow(["2026-01-01T00:00:00", "duplicate", "דילוג", "p", "1", "", "", "", "skip", ""])
    row = actions.make_log_row("g1", "duplicate", actions.ACTION_SHARED_DELETED, "q", 5, None, "delete_shared", "",
                               files_deleted=2, skipped_files=["a", "b"])
    actions.append_log([row], str(log))
    rows = read_log(str(log))
    assert list(rows[0].keys()) == actions.LOG_COLUMNS
    assert rows[0]["files_deleted"] == "" and rows[0]["path"] == "p"  # Old row kept, new columns empty
    assert rows[1]["files_deleted"] == "2" and rows[1]["skipped_files"] == "a; b"


# ----------------------------------------------------------------------- UI


def test_ui_shows_the_shared_delete_button_and_runs_it(env):
    at = AppTest.from_file(APP_FILE, default_timeout=60)
    at.run()
    at.text_area(key="roots_text").input(env["root"])
    at.button(key="start_button").click().run()
    assert not at.exception
    button = next(b for b in at.button if ":shared:B:A" in b.key)  # Case 01: from B, keep in A
    assert "3 קבצים" in button.label and "1.43 MB" in button.label.replace("\u2066", "").replace("\u2069", "")
    copy = os.path.join(env["root"], "01_copy_suffix", "מסמכים - עותק")
    button.click().run()
    assert not at.exception
    assert len(env["trashed"]) == 3 and os.listdir(copy) == []
    heading = at.subheader[0].value.translate(dict.fromkeys(map(ord, "⁦⁧⁨⁩")))
    assert heading.startswith("קבוצת כפילויות g2")  # Moved on to the next group
