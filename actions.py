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
    "files_deleted", "skipped_files",  # Filled only for "delete only shared files"
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
ACTION_SHARED_DELETED = "נמחקו קבצים משותפים"


# בדיקה לפני מחיקת קובץ משותף: הוא והעותק קיימים וה-hash שלהם לא השתנה
def verify_shared_file(path: str, copy_path: str, expected_sha256: str) -> str | None:
    """Before deleting one shared file: the file and its copy in the other member must both
    still exist and still have the scanned hash. Return a warning, or None if it is safe."""
    if not os.path.isfile(path):
        return f"הקובץ כבר לא קיים: {path}"
    if not os.path.isfile(copy_path):
        return f"העותק בתיקייה השנייה כבר לא קיים, הקובץ לא נמחק: {path}"
    if scanner.sha256_file(path) != expected_sha256:
        return f"הקובץ השתנה מאז הסריקה, לא נמחק: {path}"
    if scanner.sha256_file(copy_path) != expected_sha256:
        return f"העותק בתיקייה השנייה השתנה מאז הסריקה, הקובץ לא נמחק: {path}"
    return None


# בדיקה לפני מחיקה: הנתיב קיים, והגודל ומספר הקבצים לא השתנו מאז הסריקה
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


# העברה לסל המחזור; מחזיר אזהרה אם נכשל
def move_to_recycle_bin(path: str) -> str | None:
    """Return a warning on failure, otherwise None."""
    try:
        send2trash(path)
    except OSError as error:
        return f"ההעברה לסל המחזור נכשלה, לא נמחק: {path} ({error})"
    return None


# בניית שורה ליומן הפעולות
def make_log_row(
    item_id: str,
    item_type: str,
    action: str,
    path: str,
    size_bytes: int,
    judgement: dict | None,
    user_decision: str,
    recommendation_outcome: str,
    files_deleted: int | str = "",
    skipped_files: list[str] | None = None,
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
        "files_deleted": files_deleted,
        "skipped_files": "; ".join(skipped_files or []),
        "item_id": item_id,
    }


# עדכון כותרת של יומן ישן לעמודות הנוכחיות, בלי לאבד שורות
def _upgrade_log_header(log_path: str) -> None:
    """A log written by an older version has fewer columns: rewrite it with the current
    header (old rows get empty values in the new columns), so that new rows line up."""
    with open(log_path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames == LOG_COLUMNS:
            return
        rows = list(reader)
    temporary = log_path + ".tmp"
    with open(temporary, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, log_path)


# הוספת שורות לקובץ היומן (CSV בקידוד שמתאים ל-Excel)
def append_log(rows: list[dict], log_path: str | None = None) -> None:
    if not rows:
        return
    log_path = log_path or config.ACTIONS_LOG_PATH
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    is_new = not os.path.exists(log_path)
    if not is_new:
        _upgrade_log_header(log_path)
    # utf-8-sig so that Excel displays Hebrew correctly.
    with open(log_path, "a", newline="", encoding="utf-8-sig" if is_new else "utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_COLUMNS, extrasaction="ignore")
        if is_new:
            writer.writeheader()
        writer.writerows(rows)
