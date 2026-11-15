"""Streamlit interface (Hebrew, right-to-left).

Run:  streamlit run app.py

Every decision button carries the id of the interrupt it was rendered for.
Before resuming, the pending interrupt in the checkpoint is checked under a
lock, so a double click, a stale page or a refresh never resumes twice.
"""

import html
import os
import threading
import uuid

import streamlit as st
from langgraph.types import Command

import config
import graph
import scanner
from cli import VERDICT_LABELS, format_size

st.set_page_config(page_title="איתור תיקיות כפולות", page_icon="🗂️", layout="wide")

RTL_CSS = """
<style>
.stApp, [data-testid="stSidebar"], [data-testid="stMarkdownContainer"],
[data-testid="stExpander"], [data-testid="stAlert"], [data-testid="stMetric"] {
    direction: rtl; text-align: right;
}
[data-testid="stSlider"], textarea, input { direction: ltr; }
[data-testid="stWidgetLabel"] { direction: rtl; text-align: right; }
table.rtl-table { direction: rtl; width: 100%; border-collapse: collapse; margin-bottom: 1rem; }
table.rtl-table th, table.rtl-table td {
    text-align: right; padding: 6px 10px; border-bottom: 1px solid rgba(128, 128, 128, 0.3);
}
.ltr { direction: ltr; unicode-bidi: isolate; display: inline-block; }
ul.rtl-list { direction: rtl; text-align: right; padding-right: 1.2rem; }
</style>
"""

PATH_INSTRUCTIONS = (
    "איך מעתיקים נתיב מסייר הקבצים: פותחים את התיקייה, לוחצים על שורת הכתובת למעלה "
    "ומעתיקים (Ctrl+C). אפשר גם ללחוץ על תיקייה עם לחצן ימני ולבחור \"העתק כנתיב\" – "
    "המירכאות יוסרו אוטומטית."
)


# ----------------------------------------------------------- Shared resources


@st.cache_resource
def get_app():
    """The compiled graph with its SqliteSaver connection, built once per process."""
    return graph.build_graph(graph.open_checkpointer())


@st.cache_resource
def get_lock() -> threading.Lock:
    """Serializes graph invocations, so a resume can never run twice in parallel."""
    return threading.Lock()


# ------------------------------------------------------------------ Helpers


def path_html(path: str) -> str:
    return f'<span class="ltr">{html.escape(path)}</span>'


def bdi(text: str) -> str:
    """Isolate a left-to-right value (such as "1.25 MB") inside right-to-left HTML."""
    return f"<bdi>{html.escape(text)}</bdi>"


def ltr(text: str) -> str:
    """Same as bdi() for plain-text labels: Unicode LEFT-TO-RIGHT ISOLATE ... POP DIRECTIONAL ISOLATE."""
    return f"⁦{text}⁩"


def isolate(text: str) -> str:
    """FIRST STRONG ISOLATE: for names that may be Hebrew or English."""
    return f"⁨{text}⁩"


def size_html(size_bytes: int) -> str:
    return bdi(format_size(size_bytes))


def html_table(headers: list[str], rows: list[list[str]]) -> str:
    """Cells must already be HTML-escaped."""
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
    return f'<table class="rtl-table"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def html_list(items: list[str]) -> str:
    return '<ul class="rtl-list">' + "".join(f"<li>{item}</li>" for item in items) + "</ul>"


def parse_roots(text: str) -> list[str]:
    roots = []
    for line in text.splitlines():
        path = line.strip().strip('"').strip()
        if path and path not in roots:
            roots.append(path)
    return roots


def current_config() -> dict:
    return graph.thread_config(st.session_state.thread_id)


def set_thread(thread_id: str) -> None:
    st.session_state.thread_id = thread_id
    st.query_params["thread"] = thread_id


# ---------------------------------------------------- Requests (callbacks)
# Callbacks only record the request; it is executed once in process_request().


def request_start(token: str) -> None:
    if token != st.session_state.get("start_token"):
        return  # A second click on a button rendered before the first click was handled
    roots = parse_roots(st.session_state.get("roots_text", ""))
    errors = [e for e in (scanner.validate_root(r) for r in roots) if e]
    if not roots:
        errors = ["יש להזין לפחות תיקייה אחת לסריקה."]
    st.session_state.form_errors = errors
    if errors:
        return
    st.session_state.start_token = uuid.uuid4().hex  # The old button can no longer start a scan
    st.session_state.request = {"kind": "start", "roots": roots, "settings": {
        "overlap_threshold": st.session_state.overlap_threshold,
        "min_folder_size_mb": st.session_state.min_folder_size_mb,
        "confidence_threshold": st.session_state.confidence_threshold,
    }}


def request_resume(interrupt_id: str, decision: dict) -> None:
    st.session_state.request = {"kind": "resume", "interrupt_id": interrupt_id, "decision": decision}


def request_open_thread() -> None:
    thread_id = st.session_state.get("thread_input", "").strip()
    if not thread_id or not get_app().get_state(graph.thread_config(thread_id)).values:
        st.session_state.thread_error = f"לא נמצאה הרצה עם המזהה {thread_id}"
        return
    st.session_state.thread_error = None
    set_thread(thread_id)


def set_checkboxes(keys: list[str], value: bool) -> None:
    for key in keys:
        st.session_state[key] = value


def process_request(app) -> None:
    request = st.session_state.pop("request", None)
    if request is None:
        return
    with get_lock():
        if request["kind"] == "start":
            thread_id = uuid.uuid4().hex[:8]
            set_thread(thread_id)
            with st.spinner("סורק את התיקיות ומנתח את הממצאים... בתיקיות גדולות זה עשוי לקחת כמה דקות."):
                app.invoke(graph.initial_state(request["roots"], request["settings"]), current_config())
            return
        snapshot = app.get_state(current_config())
        if not snapshot.interrupts or snapshot.interrupts[0].id != request["interrupt_id"]:
            return  # Already handled (double click, refresh or a stale page)
        with st.spinner("מבצע את ההחלטה ומכין את הפריט הבא..."):
            app.invoke(Command(resume=request["decision"]), current_config())


def continue_stopped_run(app) -> None:
    """A run stopped in the middle of a node (not at an interrupt) continues from its checkpoint."""
    with get_lock():
        snapshot = app.get_state(current_config())
        if snapshot.next and not snapshot.interrupts:
            with st.spinner("ממשיך את ההרצה מהנקודה שבה נעצרה..."):
                app.invoke(None, current_config())


# -------------------------------------------------------------------- Sidebar


def sidebar() -> None:
    with st.sidebar:
        st.header("סריקה חדשה")
        st.text_area("תיקיות לסריקה (נתיב אחד בכל שורה)", key="roots_text", height=130,
                     placeholder=r"C:\Users\...\Documents")
        st.caption(PATH_INSTRUCTIONS)
        st.slider("סף חפיפה", 0.5, 1.0, config.DEFAULT_OVERLAP_THRESHOLD, 0.05, key="overlap_threshold")
        st.slider("גודל תיקייה מינימלי (MB)", 0, 100, config.DEFAULT_MIN_FOLDER_SIZE_MB, 1,
                  key="min_folder_size_mb")
        st.slider("סף ביטחון", 0.0, 1.0, config.DEFAULT_CONFIDENCE_THRESHOLD, 0.05,
                  key="confidence_threshold")
        st.session_state.setdefault("start_token", uuid.uuid4().hex)
        st.button("התחל סריקה", type="primary", on_click=request_start,
                  args=(st.session_state.start_token,), key="start_button", use_container_width=True)
        for error in st.session_state.get("form_errors") or []:
            st.error(error)

        st.divider()
        st.header("המשך הרצה קודמת")
        st.text_input("מזהה הרצה", key="thread_input")
        st.button("המשך", on_click=request_open_thread, key="open_thread_button")
        if st.session_state.get("thread_error"):
            st.error(st.session_state.thread_error)
        if st.session_state.get("thread_id"):
            st.caption("מזהה ההרצה הנוכחית (לשמירה, להמשך אחרי סגירה):")
            st.code(st.session_state.thread_id, language=None)


# ---------------------------------------------------------------------- Cards


def duplicate_card(payload: dict, interrupt_id: str) -> None:
    group = payload["item"]
    members = group["members"]
    st.subheader(f"קבוצת כפילויות {ltr(group['id'])} · {len(members)} תיקיות")
    left, right = st.columns(2)
    left.metric("חפיפה (Jaccard)", ltr(f"{group['jaccard'] * 100:.0f}%"))
    right.metric("הכלה (containment)", ltr(f"{group['containment'] * 100:.0f}%"))

    st.markdown(html_table(
        ["מזהה", "נתיב", "גודל", "קבצים", "שינוי אחרון", "פרויקט"],
        [[bdi(m["id"]), path_html(m["path"]), size_html(m["size_bytes"]), str(m["file_count"]),
          bdi(m["latest_modified"]), path_html(m["project"]) if m.get("project") else "—"]
         for m in members],
    ), unsafe_allow_html=True)

    recommendation = payload["recommendation"]
    if recommendation is None:
        st.info(payload["no_recommendation_message"])
    else:
        if payload["low_confidence"]:
            st.error("⚠️ המערכת לא בטוחה, כדאי לבדוק")
        names = {m["id"]: m["name"] for m in members}
        keep = ", ".join(f"{ltr(k)} ({isolate(names[k])})" for k in recommendation["keep"])
        st.markdown(
            f"**המלצה:** {VERDICT_LABELS[recommendation['verdict']]} · "
            f"ביטחון {ltr(format(recommendation['confidence'], '.2f'))} · לשמור: {keep}"
        )
        st.markdown(f"**נימוק:** {recommendation['reason']}")

    for member in members:
        if not member["unique_count"]:
            continue
        label = (f"קבצים ייחודיים בתיקייה {ltr(member['id'])} ({isolate(member['name'])}): "
                 f"{member['unique_count']}")
        with st.expander(label):
            st.warning("קבצים אלה יימחקו אם תמחקי תיקייה זו")
            st.markdown(html_list([
                f"{path_html(u['relative_path'])} ({size_html(u['size_bytes'])})"
                for u in member["unique_files"]
            ]), unsafe_allow_html=True)

    st.markdown("**סימון תיקיות למחיקה:**")
    selected = [
        m["path"] for m in members
        if st.checkbox(f"תיקייה {ltr(m['id'])} · {isolate(m['name'])}", key=f"{interrupt_id}:{m['id']}")
    ]
    deletes_all = len(selected) == len(members)
    if deletes_all:
        st.error("אי אפשר למחוק את כל התיקיות בקבוצה – לפחות אחת חייבת להישאר.")
    delete_column, skip_column, _ = st.columns([1, 1, 3])
    delete_column.button(
        "מחק את המסומנים", type="primary", disabled=not selected or deletes_all,
        on_click=request_resume, args=(interrupt_id, {"action": "delete", "paths": selected}),
        key=f"{interrupt_id}:delete",
    )
    skip_column.button("דלג", on_click=request_resume, args=(interrupt_id, {"action": "skip"}),
                       key=f"{interrupt_id}:skip")

    if "delete_shared" in payload["actions"]:
        shared_delete_buttons(members, interrupt_id)


def shared_delete_buttons(members: list[dict], interrupt_id: str) -> None:
    """For each member and each other member: delete only the files that also exist there."""
    options = [
        (member, other_id, info)
        for member in members
        for other_id, info in member.get("shared_with", {}).items()
    ]
    if not options:
        return
    names = {m["id"]: m["name"] for m in members}
    st.markdown("**מחיקת קבצים משותפים בלבד** (התיקייה עצמה נשארת, וגם הקבצים הייחודיים שבה):")
    st.caption("רק קבצים שיש להם עותק זהה בתיקייה השנייה יימחקו, אחרי בדיקה חוזרת של כל קובץ. "
               "התיקייה שהעותקים נשארים בה תהיה מוגנת ולא תימחק בהמשך ההרצה.")
    for member, other_id, info in options:
        label = (f"מחק מתיקייה {ltr(member['id'])} רק את הקבצים שקיימים גם ב-{ltr(other_id)} "
                 f"({isolate(names[other_id])}): {info['count']} קבצים, "
                 f"יתפנו {ltr(format_size(info['size_bytes']))}")
        st.button(
            label, on_click=request_resume,
            args=(interrupt_id, {"action": "delete_shared", "from": member["id"], "keep_in": other_id}),
            key=f"{interrupt_id}:shared:{member['id']}:{other_id}",
        )


def zip_card(payload: dict, interrupt_id: str) -> None:
    item = payload["item"]
    status = "חולץ במלואו" if item["fully_extracted"] else "חולץ חלקית"
    st.subheader(f"קובץ {ltr('ZIP')} שחולץ · {status}")
    st.markdown(html_table(
        ["", "נתיב", "גודל", "קבצים"],
        [[bdi("ZIP"), path_html(item["zip_path"]), size_html(item["zip_size_bytes"]), str(item["zip_file_count"])],
         ["תיקייה", path_html(item["folder_path"]), size_html(item["folder_size_bytes"]),
          str(item["folder_file_count"])]],
    ), unsafe_allow_html=True)
    if item["missing_files"]:
        st.warning(f"החילוץ חלקי: {len(item['missing_files'])} קבצים מה-ZIP חסרים בתיקייה, "
                   "והם יאבדו אם ה-ZIP יימחק.")
        st.markdown(html_list([path_html(p) for p in item["missing_files"]]), unsafe_allow_html=True)
    columns = st.columns([1, 1, 1, 2])
    for column, (label, action) in zip(columns, [
        ("מחק את ה-ZIP", "delete_zip"), ("מחק את התיקייה", "delete_folder"), ("דלג", "skip"),
    ]):
        column.button(label, on_click=request_resume, args=(interrupt_id, {"action": action}),
                      key=f"{interrupt_id}:{action}")


def empty_folders_card(payload: dict, interrupt_id: str) -> None:
    folders = payload["folders"]
    st.subheader(f"תיקיות ריקות · {len(folders)}")
    st.caption("תיקיות שאין בהן אף קובץ, בשום עומק. מוצגת רק התיקייה העליונה בכל ענף.")
    keys = [f"{interrupt_id}:empty:{n}" for n in range(len(folders))]
    select_column, clear_column, _ = st.columns([1, 1, 3])
    select_column.button("בחר הכל", on_click=set_checkboxes, args=(keys, True), key=f"{interrupt_id}:all")
    clear_column.button("נקה הכל", on_click=set_checkboxes, args=(keys, False), key=f"{interrupt_id}:none")
    selected = [path for path, key in zip(folders, keys) if st.checkbox(ltr(path), key=key)]
    st.button(
        "מחק את המסומנות", type="primary", on_click=request_resume,
        args=(interrupt_id, {"delete": selected}), key=f"{interrupt_id}:delete",
    )
    if not selected:
        st.caption("אם לא מסומנת אף תיקייה, הלחיצה תמשיך לסיכום בלי למחוק.")


def summary_card(summary: dict) -> None:
    st.subheader("סיכום")
    columns = st.columns(5)
    columns[0].metric("פריטים שנבדקו", summary["reviewed"])
    columns[1].metric("נמחקו (לסל המחזור)", summary["deleted"])
    if summary.get("shared_files_deleted"):
        st.caption(f"בנוסף נמחקו {ltr(str(summary['shared_files_deleted']))} קבצים משותפים "
                   "(מחיקה של קבצים בלבד, בלי התיקייה).")
    columns[2].metric("דולגו", summary["skipped"])
    columns[3].metric("סווגו כשונים", summary["classified_different"])
    columns[4].metric("מקום שהתפנה", ltr(format_size(summary["freed_bytes"])))
    if summary["warnings"]:
        with st.expander(f"אזהרות ({len(summary['warnings'])})", expanded=True):
            st.markdown(html_list([html.escape(w) for w in summary["warnings"]]), unsafe_allow_html=True)
    log_file = summary["log_file"]
    st.markdown(f"**יומן הפעולות:** {path_html(log_file)}", unsafe_allow_html=True)
    if os.path.exists(log_file):
        with open(log_file, "rb") as handle:
            st.download_button("הורדת יומן הפעולות (CSV)", handle.read(), file_name="actions_log.csv",
                               mime="text/csv")
    st.caption("לסריקה חדשה, יש להזין תיקיות בסרגל הצד וללחוץ על \"התחל סריקה\".")


# ----------------------------------------------------------------------- Main


def main() -> None:
    st.markdown(RTL_CSS, unsafe_allow_html=True)
    app = get_app()
    st.session_state.app = app
    if "thread_id" not in st.session_state and st.query_params.get("thread"):
        st.session_state.thread_id = st.query_params["thread"]  # Restored after a refresh

    sidebar()
    st.title("🗂️ איתור תיקיות כפולות")
    process_request(app)

    if not st.session_state.get("thread_id"):
        st.info("כדי להתחיל, יש להזין תיקיות בסרגל הצד וללחוץ על \"התחל סריקה\". "
                "שום דבר לא נמחק בלי אישור, וכל מחיקה עוברת לסל המחזור.")
        return

    continue_stopped_run(app)
    snapshot = app.get_state(current_config())
    st.session_state.current_state = snapshot.values
    if not snapshot.values:
        st.error(f"לא נמצאה הרצה עם המזהה {st.session_state.thread_id}")
        return

    values = snapshot.values
    st.caption(f"מזהה הרצה: {ltr(st.session_state.thread_id)} · ממתינים בתור: {len(values.get('queue', []))} · "
               f"נמחקו עד כה: {len(values.get('deleted_paths', []))}")
    if snapshot.interrupts:
        interrupt = snapshot.interrupts[0]
        card = {"duplicate": duplicate_card, "zip": zip_card, "empty_folders": empty_folders_card}
        card[interrupt.value["type"]](interrupt.value, interrupt.id)
    elif values.get("summary"):
        summary_card(values["summary"])


main()
