"""Revalidation before deletion, moving to the Recycle Bin, and the action log."""

import csv
import os
from datetime import datetime

from send2trash import send2trash

import config
import scanner

LOG_COLUMNS = [
    "timestamp", "item_type", "action", "path", "size_bytes",
    "llm_verdict", "llm_keep", "llm_confidence", "user_decision", "recommendation_outcome",
]

# Values of the "recommendation_outcome" column ("" = there was no recommendation:
# ZIP items, empty folders, LLM failure, or no user decision).
OUTCOME_FOLLOWED = "followed"      # The decision matches the recommendation
OUTCOME_OVERRIDDEN = "overridden"  # Different members were deleted than recommended
OUTCOME_SKIPPED = "skipped"        # The user skipped a group that had a recommendation

# Values of the "action" column (shown to the user, so in Hebrew).
ACTION_MOVED = "הועבר לסל המחזור"
ACTION_NOT_DELETED = "לא נמחק"
ACTION_SKIPPED = "דילוג"
ACTION_DIFFERENT = "סווג כשונה"


def revalidate(path: str, expected_size: int, expected_file_count: int, is_file: bool) -> str | None:
    """Return a warning if the path changed since the scan, otherwise None."""
    if is_file:
        if not os.path.isfile(path):
            return f"הקובץ כבר לא קיים, לא נמחק: {path}"
        if os.path.getsize(path) != expected_size:
            return f"הקובץ השתנה מאז הסריקה, לא נמחק: {path}"
        return None
    if not os.path.isdir(path):
        return f"התיקייה כבר לא קיימת, לא נמחקה: {path}"
    if scanner.measure_folder(path) != (expected_size, expected_file_count):
        return f"התיקייה השתנתה מאז הסריקה (גודל או מספר קבצים), לא נמחקה: {path}"
    return None


def move_to_recycle_bin(path: str) -> str | None:
    """Return a warning on failure, otherwise None."""
    try:
        send2trash(path)
    except OSError as error:
        return f"ההעברה לסל המחזור נכשלה, לא נמחק: {path} ({error})"
    return None


def make_log_row(
    item_id: str,
    item_type: str,
    action: str,
    path: str,
    size_bytes: int,
    judgement: dict | None,
    user_decision: str,
    recommendation_outcome: str,
) -> dict:
    """A log row. item_id is kept in the graph state only, not in the CSV."""
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "item_type": item_type,
        "action": action,
        "path": path,
        "size_bytes": size_bytes,
        "llm_verdict": judgement["verdict"] if judgement else "",
        "llm_keep": ",".join(judgement["keep"]) if judgement else "",
        "llm_confidence": judgement["confidence"] if judgement else "",
        "user_decision": user_decision,
        "recommendation_outcome": recommendation_outcome,
        "item_id": item_id,
    }


def append_log(rows: list[dict], log_path: str | None = None) -> None:
    if not rows:
        return
    log_path = log_path or config.ACTIONS_LOG_PATH
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    is_new = not os.path.exists(log_path)
    # utf-8-sig so that Excel displays Hebrew correctly.
    with open(log_path, "a", newline="", encoding="utf-8-sig" if is_new else "utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_COLUMNS, extrasaction="ignore")
        if is_new:
            writer.writeheader()
        writer.writerows(rows)
