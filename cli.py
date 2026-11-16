"""Run the graph in the terminal (for debugging before the Streamlit UI).

Usage:
    python cli.py PATH [PATH ...]     Start a new run
    python cli.py --thread ID         Continue a stopped run from its checkpoint
"""

import argparse
import sys
import uuid

from langgraph.types import Command

import graph
import llm_judge
import scanner

VERDICT_LABELS = {"identical": "זהות", "copy_with_changes": "עותק עם שינויים", "different": "שונות"}
MAX_UNIQUE_FILES_TO_PRINT = 10


# חריגה לעצירה מסודרת כשהקלט נגמר (ההרצה נשמרת)
class StopRun(Exception):
    """Input ended (Ctrl+Z / closed terminal): stop and keep the checkpoint."""


# הצגת גודל ב-KB או MB
def format_size(size_bytes: int) -> str:
    if size_bytes >= 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    return f"{size_bytes / 1024:.0f} KB"


# קריאת שורה מהמשתמשת; בסוף הקלט עוצרים ושומרים
def ask(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except EOFError:
        raise StopRun from None


# ------------------------------------------------------------------ Items


# הדפסת קבוצת כפילויות: חברים, קבצים ייחודיים והמלצה
def print_duplicate(payload: dict) -> None:
    group = payload["item"]
    print(f"\n=== קבוצה {group['id']}: {len(group['members'])} תיקיות | "
          f"חפיפה (Jaccard) {group['jaccard'] * 100:.0f}% | הכלה {group['containment'] * 100:.0f}% ===")
    for member in group["members"]:
        print(f"  {member['id']}  {member['path']}")
        print(f"     {format_size(member['size_bytes'])} | {member['file_count']} קבצים | "
              f"שינוי אחרון: {member['latest_modified']}")
        if member.get("project"):
            print(f"     בתוך פרויקט: {member['project']}")
        for other_id, info in member.get("shared_with", {}).items():
            print(f"     {info['count']} קבצים משותפים עם {other_id} ({format_size(info['size_bytes'])})")
        if member["unique_count"]:
            print(f"     קבצים ייחודיים ({member['unique_count']}) - "
                  "קבצים אלה יימחקו אם תמחקי תיקייה זו:")
            for unique in member["unique_files"][:MAX_UNIQUE_FILES_TO_PRINT]:
                print(f"       - {unique['relative_path']} ({format_size(unique['size_bytes'])})")
    recommendation = payload["recommendation"]
    if recommendation is None:
        print(f"  {payload['no_recommendation_message']}")
    else:
        keep = ", ".join(recommendation["keep"])
        print(f"  המלצה: {VERDICT_LABELS[recommendation['verdict']]} "
              f"(ביטחון {recommendation['confidence']:.2f}) | לשמור: {keep}")
        print(f"  נימוק: {recommendation['reason']}")
        if payload["low_confidence"]:
            print("  !!! המערכת לא בטוחה, כדאי לבדוק !!!")


# קריאת ההחלטה לקבוצת כפילויות (מחיקה, קבצים משותפים או דילוג)
def read_duplicate_decision(payload: dict) -> dict:
    members = {m["id"]: m["path"] for m in payload["item"]["members"]}
    if payload.get("blocked_reason"):
        print(f"  {payload['blocked_reason']}")
        while ask("  פקודה [s = דלג]: ").lower() != "s":
            print("  אפשר רק לדלג על הקבוצה הזו.")
        return {"action": "skip"}
    while True:
        shared = "delete_shared" in payload["actions"]
        prompt = "  פקודה [d A,B = מחק את המסומנים | " + ("x A B = מחק מ-A רק קבצים שקיימים גם ב-B | " if shared else "") + "s = דלג]: "
        answer = ask(prompt)
        if answer.lower() == "s":
            return {"action": "skip"}
        parts = answer.upper().split()
        if shared and len(parts) == 3 and parts[0] == "X":
            source, other = parts[1], parts[2]
            info = next((m["shared_with"].get(other) for m in payload["item"]["members"] if m["id"] == source), None)
            if info:
                print(f"  יימחקו {info['count']} קבצים מ-{source}, יתפנו {format_size(info['size_bytes'])}")
                return {"action": "delete_shared", "from": source, "keep_in": other}
        if answer.lower().startswith("d "):
            ids = [i.strip().upper() for i in answer[2:].split(",") if i.strip()]
            if ids and all(i in members for i in ids) and len(set(ids)) < len(members):
                return {"action": "delete", "paths": [members[i] for i in ids]}
        print("  פקודה לא תקינה. יש לבחור מזהים קיימים, ולא את כל התיקיות בקבוצה.")


# הדפסת פריט ZIP וקבצים חסרים
def print_zip(payload: dict) -> None:
    item = payload["item"]
    status = "חולץ במלואו" if item["fully_extracted"] else "חולץ חלקית"
    print(f"\n=== ZIP {item['id']}: {status} ===")
    print(f"  ZIP:     {item['zip_path']} ({format_size(item['zip_size_bytes'])}, "
          f"{item['zip_file_count']} קבצים)")
    print(f"  תיקייה:  {item['folder_path']} ({format_size(item['folder_size_bytes'])}, "
          f"{item['folder_file_count']} קבצים)")
    if item["missing_files"]:
        print("  אזהרה: הקבצים הבאים חסרים בתיקייה ויאבדו אם ה-ZIP יימחק:")
        for missing in item["missing_files"]:
            print(f"    - {missing}")


# קריאת ההחלטה לפריט ZIP
def read_zip_decision(payload: dict) -> dict:
    commands = {"z": "delete_zip", "f": "delete_folder", "s": "skip"}
    while True:
        answer = ask("  פקודה [z = מחק את ה-ZIP | f = מחק את התיקייה | s = דלג]: ").lower()
        if answer in commands:
            return {"action": commands[answer]}
        print("  פקודה לא תקינה.")


# הדפסת רשימת התיקיות הריקות
def print_empty_folders(payload: dict) -> None:
    print(f"\n=== תיקיות ריקות ({len(payload['folders'])}) ===")
    for number, path in enumerate(payload["folders"], start=1):
        print(f"  {number}. {path}")


# קריאת התיקיות הריקות שנבחרו למחיקה
def read_empty_folders_decision(payload: dict) -> dict:
    folders = payload["folders"]
    while True:
        answer = ask("  פקודה [מספרים מופרדים בפסיק | all = בחר הכל | s = אל תמחק]: ").lower()
        if answer == "s":
            return {"delete": []}
        if answer == "all":
            return {"delete": list(folders)}
        try:
            numbers = {int(n) for n in answer.split(",") if n.strip()}
        except ValueError:
            numbers = set()
        if numbers and all(1 <= n <= len(folders) for n in numbers):
            return {"delete": [folders[n - 1] for n in sorted(numbers)]}
        print("  פקודה לא תקינה.")


# הצגת הפריט שממתין לאישור וקריאת ההחלטה לפי סוגו
def handle_interrupt(payload: dict) -> dict:
    if payload["type"] == "duplicate":
        print_duplicate(payload)
        return read_duplicate_decision(payload)
    if payload["type"] == "zip":
        print_zip(payload)
        return read_zip_decision(payload)
    print_empty_folders(payload)
    return read_empty_folders_decision(payload)


# -------------------------------------------------------------------- Run


# הדפסת הסיכום בסוף ההרצה
def print_summary(summary: dict) -> None:
    print("\n=== סיכום ===")
    print(f"  פריטים שנבדקו: {summary['reviewed']}")
    print(f"  נמחקו (לסל המחזור): {summary['deleted']}")
    print(f"  דולגו: {summary['skipped']}")
    print(f"  סווגו כשונים: {summary['classified_different']}")
    if summary.get("shared_files_deleted"):
        print(f"  קבצים משותפים שנמחקו (בלי התיקייה): {summary['shared_files_deleted']}")
    if summary.get("skipped_cross_project"):
        print(f"  קבוצות בין פרויקטים שונים שלא הוצגו: {summary['skipped_cross_project']} "
              "(להצגה: --include-cross-project)")
    print(f"  מקום שהתפנה: {format_size(summary['freed_bytes'])}")
    print(f"  אזהרות ({len(summary['warnings'])}):")
    for warning in summary["warnings"]:
        print(f"    - {warning}")
    print(f"  יומן פעולות: {summary['log_file']}")


# הפעלת הגרף ומענה על כל עצירה עד שההרצה מסתיימת
def drive(app, config: dict, start_input: dict | None) -> None:
    """Invoke the graph and answer interrupts until it finishes."""
    pending = start_input
    while True:
        snapshot = app.get_state(config)
        if snapshot.interrupts:
            decision = handle_interrupt(snapshot.interrupts[0].value)
            app.invoke(Command(resume=decision), config)
        elif pending is not None or snapshot.next:
            if pending is not None:
                print("סורק... (זה עשוי לקחת זמן)")
            app.invoke(pending, config)  # None continues a run stopped mid-node
            pending = None
        else:
            break
    print_summary(app.get_state(config).values["summary"])


# נקודת הכניסה: הרצה חדשה עם נתיבים או המשך עם --thread
def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="איתור תיקיות כפולות (מצב טרמינל)")
    parser.add_argument("paths", nargs="*", help="תיקיות לסריקה")
    parser.add_argument("--thread", help="מזהה הרצה קיימת להמשך")
    parser.add_argument("--include-cross-project", action="store_true",
                        help="הצג גם כפילויות שכל החברים בהן בפרויקטים שונים")
    args = parser.parse_args()

    app = graph.build_graph(graph.open_checkpointer())
    if args.thread:
        thread_id, start_input = args.thread, None
        if not app.get_state(graph.thread_config(thread_id)).values:
            sys.exit(f"לא נמצאה הרצה עם המזהה {thread_id}")
    else:
        if not args.paths:
            parser.error("יש לציין לפחות תיקייה אחת לסריקה, או --thread להמשך הרצה")
        errors = [e for e in (scanner.validate_root(p) for p in args.paths) if e]
        if errors:
            sys.exit("\n".join(errors))
        thread_id = uuid.uuid4().hex[:8]
        settings = {**graph.default_settings(), "include_cross_project": args.include_cross_project}
        start_input = graph.initial_state(args.paths, settings)

    print(f"מזהה הרצה: {thread_id}  (להמשך אחרי עצירה: python cli.py --thread {thread_id})")
    try:
        drive(app, graph.thread_config(thread_id), start_input)
    except StopRun:
        print(f"\nהקלט הסתיים. ההרצה נשמרה; להמשך: python cli.py --thread {thread_id}")
    except KeyboardInterrupt:
        print(f"\nנעצר. ההרצה נשמרה; להמשך: python cli.py --thread {thread_id}")


if __name__ == "__main__":
    main()
