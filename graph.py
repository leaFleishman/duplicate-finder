"""LangGraph workflow: scan, find candidates, then review each item with a human.

Flow:
    scan -> find_candidates -> pick_next
    pick_next -> judge | human_review | review_empty_folders | summary
    judge -> pick_next (verdict "different") | human_review
    human_review (interrupt) -> execute | expand_children | pick_next
    expand_children -> pick_next,  execute -> pick_next
    review_empty_folders (interrupt) -> summary -> END

Nodes that call interrupt() have no side effects before it, because they
restart from the beginning on resume. Files are deleted and log rows are
written only in execute and review_empty_folders, and (for skip / "different"
decisions) at the start of the following pick_next.

"Delete only shared files" (action delete_shared) removes from one member only the
files that have an identical copy in another member. Every file is rechecked (both
copies exist, both hashes unchanged) before send2trash, and the member that keeps the
copies becomes protected: it is never deleted as a whole folder later in the run.
"""

import os
import sqlite3
from typing import Annotated, Any, TypedDict

from langgraph.channels import UntrackedValue
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

import actions
import config
import llm_judge
import scanner

RECURSION_LIMIT = 10_000  # The review loop can run many steps between interrupts

DUPLICATE_ACTIONS = ["delete", "skip"]
ZIP_ACTIONS = ["delete_zip", "delete_folder", "skip"]
# "Delete only the shared files" is offered only when the LLM judged the folders to be copies.
SHARED_DELETE_VERDICTS = ("identical", "copy_with_changes")


class State(TypedDict):
    roots: list[str]
    settings: dict  # overlap_threshold, min_folder_size_mb, confidence_threshold
    queue: list[dict]
    hidden_children: dict
    empty_folders: list[str]
    current: dict | None
    judgement: dict | None
    decision: dict | None
    deleted_paths: list[str]
    freed_bytes: int
    warnings: list[str]
    log: list[dict]
    summary: dict | None
    # Folders that hold the only remaining copy of files deleted by "delete only shared files".
    # They are never deleted as a whole folder (or as part of one) later in the run.
    protected_paths: list[str]
    shared_files_deleted: int
    skipped_cross_project: int  # Groups not shown because their members are in different projects
    # Full scan data, passed from scan to find_candidates in memory only.
    # It is large, so it is never written to the checkpoint.
    scan_result: Annotated[Any, UntrackedValue(object)]


def default_settings() -> dict:
    return {
        "overlap_threshold": config.DEFAULT_OVERLAP_THRESHOLD,
        "min_folder_size_mb": config.DEFAULT_MIN_FOLDER_SIZE_MB,
        "confidence_threshold": config.DEFAULT_CONFIDENCE_THRESHOLD,
        # Duplicates whose members are all in different projects are hidden unless this is on.
        "include_cross_project": False,
    }


def initial_state(roots: list[str], settings: dict | None = None) -> dict:
    return {
        "roots": roots,
        "settings": settings or default_settings(),
        "queue": [],
        "hidden_children": {},
        "empty_folders": [],
        "current": None,
        "judgement": None,
        "decision": None,
        "deleted_paths": [],
        "freed_bytes": 0,
        "warnings": [],
        "log": [],
        "summary": None,
        "protected_paths": [],
        "shared_files_deleted": 0,
        "skipped_cross_project": 0,
        "scan_result": None,
    }


def thread_config(thread_id: str) -> dict:
    # metadata.thread_id groups a run and its later resumes in the LangSmith Threads view.
    return {
        "configurable": {"thread_id": thread_id},
        "metadata": {"thread_id": thread_id},
        "recursion_limit": RECURSION_LIMIT,
    }


# ---------------------------------------------------------------- Helpers


def item_paths(item: dict) -> list[str]:
    if item["type"] == "duplicate":
        return [member["path"] for member in item["members"]]
    return [item["zip_path"], item["folder_path"]]


def _recommendation_outcome(item: dict, decision: dict | None, judgement: dict | None) -> str:
    """followed / overridden / skipped, or "" when there was no recommendation or no user decision."""
    if item["type"] != "duplicate" or judgement is None or decision is None:
        return ""
    if decision["action"] == "skip":
        return actions.OUTCOME_SKIPPED
    if decision["action"] == "delete_shared":
        # Followed if files were removed from a member the recommendation did not keep.
        return actions.OUTCOME_OVERRIDDEN if decision["from"] in judgement["keep"] else actions.OUTCOME_FOLLOWED
    recommended = {m["path"] for m in item["members"] if m["id"] not in judgement["keep"]}
    if set(decision["paths"]) == recommended:
        return actions.OUTCOME_FOLLOWED
    return actions.OUTCOME_OVERRIDDEN


def _close_previous_item(state: State) -> list[dict]:
    """Log the previous item if it ended without execute (skip or "different")."""
    item, decision, judgement = state["current"], state["decision"], state["judgement"]
    if item is None:
        return []
    if decision is None and judgement and judgement["verdict"] == "different":
        action, user_decision = actions.ACTION_DIFFERENT, ""
    elif decision is not None and decision["action"] == "skip":
        action, user_decision = actions.ACTION_SKIPPED, "skip"
    else:
        return []
    paths = item_paths(item)
    size = (
        sum(m["size_bytes"] for m in item["members"]) if item["type"] == "duplicate"
        else item["zip_size_bytes"] + item["folder_size_bytes"]
    )
    row = actions.make_log_row(
        item["id"], item["type"], action, "; ".join(paths), size, judgement, user_decision,
        _recommendation_outcome(item, decision, judgement),
    )
    actions.append_log([row])
    return [row]


CROSS_PROJECT_BLOCK_MESSAGE = (
    "התיקיות שייכות לפרויקטים שונים. מחיקה עלולה לשבור את הפרויקט, ולכן אינה אפשרית כאן."
)


def deletion_blocked(item: dict | None) -> bool:
    """A group whose members are all in different projects can only be skipped (even when shown)."""
    return bool(item) and item["type"] == "duplicate" and scanner.is_cross_project(item)


def shared_delete_allowed(item: dict, judgement: dict | None) -> bool:
    return (item["type"] == "duplicate" and judgement is not None
            and judgement["verdict"] in SHARED_DELETE_VERDICTS and not deletion_blocked(item))


def _sanitize_decision(item: dict, decision: Any, judgement: dict | None = None) -> tuple[dict, str | None]:
    """Turn the resume value into a valid decision; anything invalid becomes skip."""
    action = decision.get("action") if isinstance(decision, dict) else None
    if deletion_blocked(item) and action not in (None, "skip"):
        return {"action": "skip"}, f"{CROSS_PROJECT_BLOCK_MESSAGE} הפריט דולג."
    if action == "delete_shared":
        members = {m["id"]: m for m in item.get("members", [])}
        source, keep_in = decision.get("from"), decision.get("keep_in")
        if (shared_delete_allowed(item, judgement) and source in members and keep_in in members
                and source != keep_in and keep_in in members[source].get("shared_with", {})):
            return {"action": "delete_shared", "from": source, "keep_in": keep_in}, None
        return {"action": "skip"}, "בקשה לא תקינה למחיקת קבצים משותפים, הפריט דולג"
    if item["type"] == "duplicate" and action == "delete":
        member_paths = [m["path"] for m in item["members"]]
        paths = [p for p in member_paths if p in decision.get("paths", [])]
        if 0 < len(paths) < len(member_paths):
            return {"action": "delete", "paths": paths}, None
        return {"action": "skip"}, "בחירה לא תקינה (אסור למחוק את כל העותקים או אף אחד), הפריט דולג"
    if item["type"] == "zip" and action in ("delete_zip", "delete_folder"):
        return {"action": action}, None
    if action != "skip":
        return {"action": "skip"}, f"החלטה לא מוכרת ({action}), הפריט דולג"
    return {"action": "skip"}, None


def _deletion_targets(item: dict, decision: dict) -> list[dict]:
    if item["type"] == "duplicate":
        members = {m["path"]: m for m in item["members"]}
        return [
            {"path": p, "size": members[p]["size_bytes"], "file_count": members[p]["file_count"], "is_file": False}
            for p in decision["paths"]
        ]
    if decision["action"] == "delete_zip":
        return [{"path": item["zip_path"], "size": item["zip_size_bytes"], "file_count": 1, "is_file": True}]
    return [{"path": item["folder_path"], "size": item["folder_size_bytes"],
             "file_count": item["folder_file_count"], "is_file": False}]


# ------------------------------------------------------------------ Nodes


def scan(state: State) -> dict:
    return {"scan_result": scanner.scan(state["roots"])}


def find_candidates(state: State) -> dict:
    # After a restart the in-memory scan data is gone, so scan again.
    scan_result = state.get("scan_result") or scanner.scan(state["roots"])
    settings = state["settings"]
    candidates = scanner.find_candidates(
        scan_result, settings["overlap_threshold"], settings["min_folder_size_mb"],
        include_cross_project=settings.get("include_cross_project", False),
    )
    return {
        "queue": candidates["queue"],
        "hidden_children": candidates["hidden_children"],
        "empty_folders": candidates["empty_folders"],
        "skipped_cross_project": candidates["skipped_cross_project"],
        "warnings": state["warnings"] + scan_result.warnings,
        "scan_result": None,
    }


def pick_next(state: State) -> dict:
    update = {"log": state["log"] + _close_previous_item(state), "judgement": None, "decision": None}
    queue = list(state["queue"])
    while queue:
        item = queue.pop(0)
        already_deleted = any(
            scanner.is_within(path, deleted)
            for path in item_paths(item)
            for deleted in state["deleted_paths"]
        )
        if not already_deleted:
            return {**update, "queue": queue, "current": item}
    return {**update, "queue": [], "current": None}


def route_after_pick(state: State) -> str:
    item = state["current"]
    if item is None:
        # Deletions may have created new empty folders, so check the disk in that case too.
        if state["empty_folders"] or state["deleted_paths"] or state["shared_files_deleted"]:
            return "review_empty_folders"
        return "summary"
    return "judge" if item["type"] == "duplicate" else "human_review"


def _for_llm(group: dict) -> dict:
    """The group without the shared-file lists, which the LLM does not need (and traces stay small)."""
    return {**group, "members": [{k: v for k, v in m.items() if k != "shared_with"} for m in group["members"]]}


def judge(state: State) -> dict:
    result = llm_judge.judge_with_details(_for_llm(state["current"]))
    if result.judgement is not None:
        return {"judgement": result.judgement.model_dump()}
    warning = (f"{llm_judge.NO_RECOMMENDATION_MESSAGE} עבור קבוצה {state['current']['id']}: "
               f"{llm_judge.describe_failure(result.failures)}")
    return {"judgement": None, "warnings": state["warnings"] + [warning]}


def route_after_judge(state: State) -> str:
    judgement = state["judgement"]
    if judgement and judgement["verdict"] == "different":
        return "pick_next"
    return "human_review"


def human_review(state: State) -> dict:
    item, judgement = state["current"], state["judgement"]
    threshold = state["settings"]["confidence_threshold"]
    payload = {
        "type": item["type"],
        "item": item,
        "recommendation": judgement,
        "no_recommendation_message": (
            llm_judge.NO_RECOMMENDATION_MESSAGE if item["type"] == "duplicate" and judgement is None else None
        ),
        "low_confidence": judgement is not None and judgement["confidence"] < threshold,
        "actions": (
            ZIP_ACTIONS if item["type"] == "zip"
            else ["skip"] if deletion_blocked(item)
            else ["delete", "delete_shared", "skip"] if shared_delete_allowed(item, judgement)
            else DUPLICATE_ACTIONS
        ),
        "blocked_reason": CROSS_PROJECT_BLOCK_MESSAGE if deletion_blocked(item) else None,
    }
    decision, warning = _sanitize_decision(item, interrupt(payload), judgement)
    update = {"decision": decision}
    if warning:
        update["warnings"] = state["warnings"] + [warning]
    return update


def route_after_human_review(state: State) -> str:
    item = state["current"]
    if state["decision"]["action"] != "skip":
        return "execute"
    if item["type"] == "duplicate" and item["id"] in state["hidden_children"]:
        return "expand_children"
    return "pick_next"


def expand_children(state: State) -> dict:
    children = state["hidden_children"][state["current"]["id"]]
    return {"queue": children + state["queue"]}


def _protection_problem(path: str, protected_paths: list[str]) -> str | None:
    """A protected folder holds the only remaining copy of deleted files: never delete it,
    anything inside it, or a folder that contains it."""
    for protected in protected_paths:
        if scanner.is_within(path, protected) or scanner.is_within(protected, path):
            return (f"לא נמחק: {path} מכיל את העותק היחיד שנשאר של קבצים שנמחקו קודם "
                    f"(מחיקת קבצים משותפים בלבד)")
    return None


def _execute_shared(state: State) -> dict:
    """Delete from one member only the files that have an identical copy in another member."""
    item, decision, judgement = state["current"], state["decision"], state["judgement"]
    members = {m["id"]: m for m in item["members"]}
    source, keep_in = members[decision["from"]], members[decision["keep_in"]]
    warnings = list(state["warnings"])
    deleted_count, freed, skipped = 0, 0, []
    for entry in source["shared_with"][keep_in["id"]]["files"]:
        path = os.path.join(source["path"], entry["relative_path"])
        copy_path = os.path.join(keep_in["path"], entry["other_relative_path"])
        problem = (_protection_problem(path, state["protected_paths"])
                   or actions.verify_shared_file(path, copy_path, entry["sha256"])
                   or actions.move_to_recycle_bin(path))
        if problem:
            warnings.append(problem)
            skipped.append(entry["relative_path"])
        else:
            deleted_count += 1
            freed += entry["size_bytes"]
    row = actions.make_log_row(
        item["id"], item["type"], actions.ACTION_SHARED_DELETED, source["path"], freed, judgement,
        "delete_shared", _recommendation_outcome(item, decision, judgement),
        files_deleted=deleted_count, skipped_files=skipped,
    )
    actions.append_log([row])
    protected = list(state["protected_paths"])
    if deleted_count and keep_in["path"] not in protected:
        protected.append(keep_in["path"])
    return {
        "freed_bytes": state["freed_bytes"] + freed,
        "shared_files_deleted": state["shared_files_deleted"] + deleted_count,
        "protected_paths": protected,
        "warnings": warnings,
        "log": state["log"] + [row],
    }


def execute(state: State) -> dict:
    item, decision, judgement = state["current"], state["decision"], state["judgement"]
    if deletion_blocked(item):
        # Extra guard: human_review never lets a deletion through for such a group.
        warning = f"{CROSS_PROJECT_BLOCK_MESSAGE} לא נמחק דבר (קבוצה {item['id']})."
        row = actions.make_log_row(
            item["id"], item["type"], actions.ACTION_NOT_DELETED, "; ".join(item_paths(item)), 0,
            judgement, decision["action"], _recommendation_outcome(item, decision, judgement),
        )
        actions.append_log([row])
        return {"warnings": state["warnings"] + [warning], "log": state["log"] + [row]}
    if decision["action"] == "delete_shared":
        return _execute_shared(state)
    outcome = _recommendation_outcome(item, decision, judgement)
    deleted_paths = list(state["deleted_paths"])
    freed_bytes = state["freed_bytes"]
    warnings = list(state["warnings"])
    rows = []
    for target in _deletion_targets(item, decision):
        path = target["path"]
        problem = (_protection_problem(path, state["protected_paths"])
                   or actions.revalidate(path, target["size"], target["file_count"], target["is_file"]))
        if problem is None:
            problem = actions.move_to_recycle_bin(path)
        if problem:
            warnings.append(problem)
            action = actions.ACTION_NOT_DELETED
        else:
            deleted_paths.append(path)
            freed_bytes += target["size"]
            action = actions.ACTION_MOVED
        rows.append(actions.make_log_row(
            item["id"], item["type"], action, path, target["size"], judgement, decision["action"], outcome
        ))
    actions.append_log(rows)
    return {
        "deleted_paths": deleted_paths,
        "freed_bytes": freed_bytes,
        "warnings": warnings,
        "log": state["log"] + rows,
    }


def review_empty_folders(state: State) -> dict:
    # Recomputed from disk: includes folders that became empty after deletions.
    folders = scanner.find_empty_folders_on_disk(state["roots"])
    if not folders:
        return {"empty_folders": []}
    answer = interrupt({"type": "empty_folders", "folders": folders})

    requested = answer.get("delete", []) if isinstance(answer, dict) else []
    deleted_paths = list(state["deleted_paths"])
    warnings = list(state["warnings"])
    rows = []
    for path in folders:
        if path not in requested:
            rows.append(actions.make_log_row(
                f"empty:{path}", "empty_folder", actions.ACTION_SKIPPED, path, 0, None, "skip", ""
            ))
            continue
        problem = None if scanner.is_empty_on_disk(path) else f"התיקייה כבר לא ריקה, לא נמחקה: {path}"
        if problem is None:
            problem = actions.move_to_recycle_bin(path)
        if problem:
            warnings.append(problem)
            action = actions.ACTION_NOT_DELETED
        else:
            deleted_paths.append(path)
            action = actions.ACTION_MOVED
        rows.append(actions.make_log_row(f"empty:{path}", "empty_folder", action, path, 0, None, "delete", ""))
    actions.append_log(rows)
    return {
        "empty_folders": folders,
        "deleted_paths": deleted_paths,
        "warnings": warnings,
        "log": state["log"] + rows,
    }


def summary(state: State) -> dict:
    log = state["log"]
    reviewed = {row["item_id"] for row in log if row["user_decision"]}
    skipped = {row["item_id"] for row in log if row["user_decision"] == "skip"}
    different = {row["item_id"] for row in log if row["action"] == actions.ACTION_DIFFERENT}
    return {"summary": {
        "reviewed": len(reviewed),
        "deleted": len(state["deleted_paths"]),
        "shared_files_deleted": state["shared_files_deleted"],
        "skipped": len(skipped),
        "classified_different": len(different),
        "skipped_cross_project": state.get("skipped_cross_project", 0),
        "freed_bytes": state["freed_bytes"],
        "warnings": state["warnings"],
        "log_file": config.ACTIONS_LOG_PATH,
    }}


# ------------------------------------------------------------------ Graph


def build_graph(checkpointer):
    builder = StateGraph(State)
    builder.add_node("scan", scan)
    builder.add_node("find_candidates", find_candidates)
    builder.add_node("pick_next", pick_next)
    builder.add_node("judge", judge)
    builder.add_node("human_review", human_review)
    builder.add_node("expand_children", expand_children)
    builder.add_node("execute", execute)
    builder.add_node("review_empty_folders", review_empty_folders)
    builder.add_node("summary", summary)

    builder.add_edge(START, "scan")
    builder.add_edge("scan", "find_candidates")
    builder.add_edge("find_candidates", "pick_next")
    builder.add_conditional_edges(
        "pick_next", route_after_pick, ["judge", "human_review", "review_empty_folders", "summary"]
    )
    builder.add_conditional_edges("judge", route_after_judge, ["pick_next", "human_review"])
    builder.add_conditional_edges(
        "human_review", route_after_human_review, ["execute", "expand_children", "pick_next"]
    )
    builder.add_edge("expand_children", "pick_next")
    builder.add_edge("execute", "pick_next")
    builder.add_edge("review_empty_folders", "summary")
    builder.add_edge("summary", END)
    return builder.compile(checkpointer=checkpointer)


def open_checkpointer(db_path: str | None = None) -> SqliteSaver:
    connection = sqlite3.connect(db_path or config.CHECKPOINT_DB_PATH, check_same_thread=False)
    return SqliteSaver(connection)
