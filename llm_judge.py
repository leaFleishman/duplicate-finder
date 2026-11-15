"""LLM judge for duplicate folder groups (Gemini with structured output).

Public API:
    judge(group, prompt_version=None) -> Judgement | None
    judge_with_details(group, prompt_version=None) -> JudgeResult (judgement + failure types)
    Judgement, NO_RECOMMENDATION_MESSAGE, describe_failure
"""

import logging
import os
from functools import lru_cache
from typing import Literal, NamedTuple

from dotenv import load_dotenv
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langsmith import traceable
from pydantic import BaseModel, Field, ValidationError

import config

load_dotenv()
logger = logging.getLogger(__name__)

NO_RECOMMENDATION_MESSAGE = "לא התקבלה המלצה מהמערכת"
PROMPT_VERSIONS = ("v1", "v2", "v3")
MAX_ATTEMPTS = 2  # The first attempt plus one retry
REQUEST_TIMEOUT_SECONDS = 60


class Judgement(BaseModel):
    verdict: Literal["identical", "copy_with_changes", "different"] = Field(
        description="identical, copy_with_changes or different"
    )
    confidence: float = Field(ge=0, le=1, description="Confidence from 0 to 1")
    keep: list[str] = Field(
        description="IDs of members recommended for retention; empty if verdict is different"
    )
    reason: str = Field(description="Short explanation in Hebrew, 1-3 sentences")


class JudgementReasonFirst(BaseModel):
    """v3 output schema: the same fields as Judgement, with reason before keep,
    so the model explains before it chooses. Converted to Judgement after parsing."""
    verdict: Literal["identical", "copy_with_changes", "different"] = Field(
        description="identical, copy_with_changes or different"
    )
    confidence: float = Field(ge=0, le=1, description="Confidence from 0 to 1")
    reason: str = Field(description="Short explanation in Hebrew, 1-3 sentences")
    keep: list[str] = Field(
        description="IDs of members recommended for retention; empty if verdict is different"
    )


OUTPUT_SCHEMAS = {"v1": Judgement, "v2": Judgement, "v3": JudgementReasonFirst}


# ---------------------------------------------------------------- Prompts

# Shared by both versions: the task and the meaning of each output field.
OUTPUT_INSTRUCTIONS = """\
You help a user clean up duplicate folders on their Windows computer.
A scanner flagged a group of folders as possible copies of each other. The user \
decides what to move to the Recycle Bin; your recommendation helps them decide.

Return:
- verdict: "identical" if the folders are copies with the same content; \
"copy_with_changes" if they are copies of the same folder but some copies have \
added or changed files; "different" if they are not copies of the same folder.
- confidence: a number from 0 to 1.
- keep: the IDs of the members you recommend keeping, for example ["A"]. \
Use only IDs from the input. Must be empty if the verdict is "different", \
and must not be empty otherwise.
- reason: a short explanation in Hebrew, 1-3 sentences."""

V2_CRITERIA = """\
How to read the input:
- Files are compared by content hash, so file names do not affect the overlap.
- overlap_jaccard = shared files / all files in both folders.
- containment = shared files / files in the smaller folder.
- unique files = files that exist only in that member and would be lost if it were deleted.

Decision criteria:
1. Names such as "- עותק" ("- Copy"), "- Copy", "Copy of", "(2)", "_old", "גיבוי" \
("backup") and "ישן" ("old") suggest a copy, but are not conclusive. \
The name alone is insufficient; rely on the overlap and the unique files.
2. A copy is not always the redundant version. If it contains newer unique files, \
it may be the most up-to-date version and should be kept.
3. Similar names with different content, such as "דוח 2024" ("Report 2024") and \
"דוח 2025" ("Report 2025"), should be classified as different.
4. Prefer keeping a copy in an organized location, such as Documents or Projects, \
over a temporary location, such as Downloads or Desktop. Judge the location only by the \
distinct part of each path: the common location is shared by all members, so it never \
makes one member more temporary than another. If both folders are in the same kind of \
location, location is not a consideration \
("אם שתי התיקיות נמצאות באותו סוג מיקום, המיקום אינו שיקול").
5. For copy_with_changes, the reason must say which unique files would be lost \
if either side were deleted.
6. Assign low confidence when the indicators conflict.
7. "inside project" is the nearest enclosing folder that contains project files \
(html, .csproj, .sln, package.json or requirements.txt). If members are inside \
different projects, each project may use its own copy (for example a library folder \
such as css or fontawesome), and deleting one copy can break that project. In that case \
keep every member, give low confidence (below 0.5), and say in the reason that both \
copies may be needed by different projects."""

V1_QUESTION = "Are these copies of the same folder, and which one should be kept?"
V2_QUESTION = "Judge this group according to the criteria."

# v3 = v2 plus this instruction (and the reason-before-keep schema).
V3_INSTRUCTION = """\
In the reason, name explicitly the IDs of the members to keep and the members to delete, \
for example "לשמור את A, למחוק את B" ("keep A, delete B")."""


def _format_size(size_bytes: int) -> str:
    if size_bytes >= 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    return f"{size_bytes / 1024:.0f} KB"


def _describe_v1(group: dict) -> str:
    lines = ["Folders:"]
    for member in group["members"]:
        lines.append(f'{member["id"]}: name="{member["name"]}", path="{member["path"]}"')
    return "\n".join(lines)


def split_common_location(paths: list[str]) -> tuple[str, list[str]]:
    """Return (common parent folder, distinct part of each path), compared case-insensitively."""
    split = [os.path.normpath(p).split(os.sep) for p in paths]
    common = 0
    while all(len(parts) > common + 1 for parts in split) and \
            len({parts[common].casefold() for parts in split}) == 1:
        common += 1
    prefix = os.sep.join(split[0][:common])
    if common == 1:
        prefix += os.sep  # A drive root such as "C:\"
    return prefix, [os.sep.join(parts[common:]) for parts in split]


def _describe_v2(group: dict) -> str:
    members = group["members"]
    common_location, distinct_parts = split_common_location([m["path"] for m in members])
    lines = [
        f'overlap_jaccard: {group["jaccard"] * 100:.0f}%',
        f'containment: {group["containment"] * 100:.0f}%',
        f'common location of all members: {common_location or "(none)"}',
        "",
        "Folders:",
    ]
    for member, distinct in zip(members, distinct_parts):
        lines += [
            f'{member["id"]}: name="{member["name"]}"',
            f'   path: {member["path"]}',
            f'   distinct part of the path: {distinct}',
            f'   inside project: {member.get("project") or "none"}',
            f'   size: {_format_size(member["size_bytes"])}, files: {member["file_count"]}, '
            f'latest modification: {member["latest_modified"]}',
        ]
        unique = member["unique_files"][: config.MAX_UNIQUE_FILES_TO_LLM]
        if not unique:
            lines.append("   unique files: none")
            continue
        lines.append(f'   unique files (showing {len(unique)} of {member["unique_count"]}):')
        lines += [f'     - {u["relative_path"]} ({_format_size(u["size_bytes"])})' for u in unique]
    return "\n".join(lines)


def build_messages(group: dict, prompt_version: str) -> list:
    if prompt_version == "v1":
        system = OUTPUT_INSTRUCTIONS
        human = f"{_describe_v1(group)}\n\n{V1_QUESTION}"
    else:
        system = f"{OUTPUT_INSTRUCTIONS}\n\n{V2_CRITERIA}"
        if prompt_version == "v3":
            system += f"\n\n{V3_INSTRUCTION}"
        human = f"{_describe_v2(group)}\n\n{V2_QUESTION}"
    return [SystemMessage(content=system), HumanMessage(content=human)]


# ------------------------------------------------------------- Validation


def validate_judgement(judgement: Judgement, group: dict) -> None:
    """Raise ValueError if keep does not match the group or the verdict.

    verdict and the confidence range are already enforced by the Pydantic model.
    """
    member_ids = {member["id"] for member in group["members"]}
    unknown = set(judgement.keep) - member_ids
    if unknown:
        raise ValueError(f"keep contains unknown member IDs: {sorted(unknown)}")
    if judgement.verdict == "different" and judgement.keep:
        raise ValueError("keep must be empty when verdict is 'different'")
    if judgement.verdict != "different" and not judgement.keep:
        raise ValueError("keep must not be empty when verdict is not 'different'")


# ------------------------------------------------------------- Invocation


@lru_cache(maxsize=None)
def _structured_model(schema: type[BaseModel] = Judgement):
    model = ChatGoogleGenerativeAI(
        model=config.GEMINI_MODEL,
        temperature=0,
        max_retries=0,  # Retries are handled below: exactly one retry
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    return model.with_structured_output(schema)


FAILURE_LABELS = {  # Hebrew labels, shown to the user in warnings
    "timeout": "תם זמן ההמתנה לתשובה",
    "quota": "חריגה ממכסת השימוש",
    "auth": "בעיה במפתח ה-API או בהרשאות",
    "connection": "בעיית חיבור לרשת",
    "invalid_output": "פלט לא תקין מהמודל",
    "other": "שגיאה אחרת",
}


class JudgeResult(NamedTuple):
    judgement: Judgement | None
    failures: list[str]  # One "category: ExceptionName" entry per failed attempt


def classify_error(error: Exception) -> str:
    """Map an exception to one of the FAILURE_LABELS categories."""
    name = type(error).__name__.lower()
    text = str(error).lower()
    if "resource_exhausted" in text or "resourceexhausted" in name or "429" in text or "quota" in text \
            or "rate limit" in text:
        return "quota"
    if "api key" in text or "api_key" in text or "permission" in text or "401" in text or "403" in text \
            or "unauthenticated" in text:
        return "auth"
    if "timeout" in name or "timed out" in text or isinstance(error, TimeoutError):
        return "timeout"
    if "connect" in name or "network" in name or isinstance(error, ConnectionError):
        return "connection"
    if isinstance(error, (ValidationError, OutputParserException, ValueError)):
        return "invalid_output"
    return "other"


def describe_failure(failures: list[str]) -> str:
    """Hebrew description of the last failure, for a user-facing warning."""
    if not failures:
        return FAILURE_LABELS["other"]
    category, _, exception_name = failures[-1].partition(": ")
    return f"{FAILURE_LABELS.get(category, FAILURE_LABELS['other'])} ({exception_name})"


def judge(group: dict, prompt_version: str | None = None) -> Judgement | None:
    """Return a validated Judgement, or None if both attempts failed."""
    return judge_with_details(group, prompt_version).judgement


def judge_with_details(group: dict, prompt_version: str | None = None) -> JudgeResult:
    """Like judge(), but also report the type of each failed attempt."""
    prompt_version = prompt_version or config.PROMPT_VERSION
    if prompt_version not in PROMPT_VERSIONS:
        raise ValueError(f"Unknown prompt version: {prompt_version}")
    trace_info = {
        "tags": [f"prompt-{prompt_version}"],
        "metadata": {
            "prompt_version": prompt_version,
            "model": config.GEMINI_MODEL,
            "group_id": group.get("id"),
        },
    }
    return _judge_traced(group, prompt_version, trace_info, langsmith_extra=trace_info)


@traceable(run_type="chain", name="judge")
def _judge_traced(group: dict, prompt_version: str, trace_info: dict) -> JudgeResult:
    messages = build_messages(group, prompt_version)
    failures = []
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            output = _structured_model(OUTPUT_SCHEMAS[prompt_version]).invoke(messages, config=trace_info)
            if output is None:
                raise ValueError("The model returned no structured output")
            judgement = Judgement(**output.model_dump())  # Same fields for every version
            validate_judgement(judgement, group)
            return JudgeResult(judgement, failures)
        except Exception as error:  # Connection, quota, timeout or invalid output
            category = classify_error(error)
            failures.append(f"{category}: {type(error).__name__}")
            logger.warning("judge attempt %d/%d failed [%s] %s: %s",
                           attempt, MAX_ATTEMPTS, category, type(error).__name__, error)
    return JudgeResult(None, failures)
