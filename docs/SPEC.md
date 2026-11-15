Project Specification: Duplicate Folder Finder with Human Approval (LangGraph)
Background and Goal
A course submission project: a local Windows application that scans folders selected by the user, identifies duplicate folders, ZIP files that have already been extracted, and empty folders, and requests human approval for each item before moving it to the Recycle Bin.

Mandatory assignment requirements that the project must meet:
1. Use the LangGraph library.
2. A graph with at least one loop and one conditional branch.
3. At least one interrupt (human-in-the-loop).
4. A dataset and automated evaluation in LangSmith, with an experiment before and after an improvement.
5. A README with trace screenshots and one insight discovered from them.
Working Method
- Build the project in stages, following the order at the end of this document. Stop at the end of each stage and explain what was built and how to test it before continuing.
- Use English names for variables, functions, and files. User interface text must be in Hebrew.
- Keep the code simple and readable. Do not add features that are not specified here.
Technologies
- Python 3.11 or later; Windows only; local execution only.
- langgraph, langgraph-checkpoint-sqlite, langchain-google-genai, langsmith, streamlit, send2trash, python-dotenv, pydantic.
- Model: Gemini through ChatGoogleGenerativeAI, using with_structured_output. Define the model name in config.py (GEMINI_MODEL), rather than hard-coding it in the implementation.
- Environment variables in .env: GOOGLE_API_KEY, LANGSMITH_API_KEY, LANGSMITH_TRACING=true, LANGSMITH_PROJECT=duplicate-finder. Add .env to .gitignore.
Project Structure
duplicate-finder/
├── config.py          # Constants and defaults
├── scanner.py         # Scanning, hashing, candidate detection, ZIPs, empty folders
├── llm_judge.py       # Pydantic schemas, three prompt versions (v1, v2, v3), Gemini invocation, retry and failure types
├── graph.py           # State, nodes, routing, interrupt, checkpointer
├── actions.py         # Revalidation, moving to the Recycle Bin, action logging
├── app.py             # Streamlit interface
├── cli.py             # Terminal execution for debugging
├── eval/
│   ├── make_dataset.py
│   ├── dataset.json   # The examples, saved for review before uploading
│   ├── evaluators.py
│   └── run_experiment.py
├── test_data/
│   ├── make_fixtures.py
│   ├── manual_judge_check.py   # Runs judge on the fixture groups with every prompt version
│   └── test_*.py      # pytest tests (no API calls, no real Recycle Bin)
├── docs/
│   ├── SPEC.md        # This document
│   ├── findings.md    # Findings from development, the basis for the README insights
│   └── images/        # Screenshots used in the README
├── logs/              # Action log (actions_log.csv)
└── README.md
config.py
- SKIP_DIR_NAMES: Windows, Program Files, Program Files (x86), ProgramData, $Recycle.Bin, System Volume Information, AppData, .git, .svn, .vs, .idea, .vscode, node_modules, bin, obj, venv, .venv, __pycache__.
- PROJECT_MARKER_NAMES = {"package.json", "requirements.txt", "index.html"} and PROJECT_MARKER_EXTENSIONS = {".csproj", ".sln"}: files that mark a project root (see "Project Detection").
- DEFAULT_OVERLAP_THRESHOLD = 0.7: Minimum overlap for a pair to qualify as a candidate.
- DEFAULT_MIN_FOLDER_SIZE_MB = 1
- DEFAULT_CONFIDENCE_THRESHOLD = 0.7
- MAX_UNIQUE_FILES_TO_LLM = 20
- Settings (per run, from the sidebar): overlap_threshold, min_folder_size_mb, confidence_threshold, include_cross_project (default False).
- PROMPT_VERSION = "v3" (can be changed to "v1" or "v2"). v3 was chosen as the default after the experiments; see "Prompt Versions".
- GEMINI_MODEL
- CHECKPOINT_DB_PATH and ACTIONS_LOG_PATH: relative to the project folder, regardless of the working directory.
Scanning (scanner.py), Without an LLM
- Scan using os.scandir or os.walk, without following shortcuts, symlinks, or junctions.
- Skip:
  - Folders whose names appear in SKIP_DIR_NAMES (case-insensitive comparison).
  - Folders and files with the Windows Hidden or System attributes.
  - OneDrive files that are not stored locally: attributes FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS = 0x400000, FILE_ATTRIBUTE_RECALL_ON_OPEN = 0x40000, FILE_ATTRIBUTE_OFFLINE = 0x1000. Do not read these files, because reading them would download them from the cloud.
- Permission errors, locked files, and long paths: skip and add an entry to the warnings list. The scan must not crash.
- Fully support Hebrew names.
- Hashing: SHA-256, reading in blocks. Compute hashes only for files that have at least one other file of exactly the same size anywhere in the scan.
- For each folder, store its path, name, total size, file count, latest modification date (the latest among its files), and a multiset of hashes of all files at every depth.
Folder Comparison (Based on Content Only)
- Two folders are identical if their hash multisets are identical, even if filenames or internal directory structures differ.
- Overlap = the number of shared hashes divided by the number of hashes in the union (multiset Jaccard).
- To avoid comparing every possible pair, build an inverted index from hashes to folders, and compare only folders that share at least one hash.
- Ignore empty folders (handled separately) and folders smaller than min_folder_size_mb.
- Do not compare a folder with any of its ancestors or descendants.
- Best partner only: each folder stays only in a pair with its best partner. A pair that is not identical is dropped if the smaller folder in the pair (fewer files) has another candidate pair with a higher Jaccard. Example: if "Site" and "Site backup" are identical, the pair "Site" ↔ "Site backup\Images" is dropped, because "Site backup\Images" has an identical partner "Site\Images". This also applies when the cross pair passes on Jaccard: if most of a library's files are in its "webfonts" subfolder, "fontawesome" ↔ "other\fontawesome\webfonts" can reach Jaccard 0.75, and it is dropped because "webfonts" has an identical partner. Identical pairs are never dropped by this rule.
Groups
- Completely identical folders (overlap 1.0) are merged into a single group, which may contain more than two copies.
- Pairs with overlap between the threshold and 1.0 form two-member groups.
- For each group, calculate the overlap percentage and, for each member, a list of unique files (files not present in any other member of the group), including relative paths and sizes.
- A folder that contains nothing except a single subfolder with the same content (for example "X\X") is represented by the outer folder only, so the same content never appears twice in one group.
Shared Files (for "delete only the shared files")
- For each member of a group, and for each other member, the scanner stores in member["shared_with"][other_id]: the number of files, their total size, and a list of entries {relative_path, size_bytes, sha256, other_relative_path}. An entry is a file in this member that has an identical copy (same SHA-256) in the other member; other_relative_path is where that copy is (it may have a different name or folder). Files that were not hashed (unique size or unreadable) are never listed.
- This data is not sent to the LLM.
Project Detection
- For each group member, find the nearest ancestor folder (not the member itself) that directly contains a project marker file: a file named package.json, requirements.txt or index.html, or any .csproj or .sln file (names compared case-insensitively). Other .html files are not enough: saved web pages and exported reports are common in ordinary folders and would turn them into "projects".
- Store its path in the member's "project" field (None if there is none). Pass it to the LLM (v2 and v3) and show it in the interface ("פרויקט" column in the CLI and Streamlit).
- Reason: identical library folders (for example css or fontawesome) inside two different projects are both needed; deleting one breaks that project. This case was discovered only in real use.
Cross-Project Groups (hidden by default)
- A duplicate group is cross-project if every member is inside a project and no two members share a project (project paths compared case-insensitively).
- Cross-project groups are not shown by default, at any level (top-level groups and hidden_children). They are shown when the setting include_cross_project is on (sidebar checkbox "הצג גם כפילויות בין פרויקטים שונים", off by default; CLI flag --include-cross-project).
- A group where at least two members are in the same project, or at least one member is not in a project, is shown as usual.
- If a removed group had hidden groups that are not cross-project, they move up and take its place, so no other content disappears.
- The number of groups not shown for this reason (at every level) is kept in state.skipped_cross_project and shown on the summary screen.
- Interaction with "delete only the shared files": the filter runs once, in find_candidates, before any deletion. Protection (state.protected_paths) is checked in execute for every group that is shown, including hidden groups that moved up after their cross-project parent was removed. When include_cross_project is on, cross-project groups are shown, but only for review: they can only be skipped (see the next item).
- Deletion is blocked in cross-project groups, even when they are shown: the payload offers only "skip" and carries blocked_reason "התיקיות שייכות לפרויקטים שונים. מחיקה עלולה לשבור את הפרויקט, ולכן אינה אפשרית כאן." ("The folders belong to different projects. Deleting could break the project, so it is not possible here."). In Streamlit the message is shown, the delete checkboxes, "מחק את המסומנים" and the shared-files buttons are disabled, and "דלג" stays enabled. The CLI shows the message and accepts only "s". Any deletion decision for such a group (delete or delete_shared) becomes a skip with a warning, and execute refuses it as well (extra guard): nothing is deleted, a warning is added and a "לא נמחק" log row is written.
- The filter applies only to duplicate groups. ZIP items and empty folders are not affected: the empty-folder list is computed from the scan independently (and recomputed from disk in review_empty_folders), so a folder that contains files is never listed as empty.
Show Only the Top-Level Group
- Hide a group if each of its members is located inside a member of another candidate group. Store these hidden groups separately (hidden_children), associated with their parent group, so they can be revealed if the user skips the top-level group.
Extracted ZIP Files
- For each .zip file, check whether a folder with the same name as the ZIP filename without its extension exists in the same directory (see clarification 5).
- Compare the archive's file list (zipfile, relative paths and uncompressed sizes) with the folder's contents.
- Fully extracted: all files exist. Partially extracted: some files exist; retain the list of missing files. If none of the files exist, the item is not a candidate.
- Corrupted or password-protected ZIP files: skip and record a warning.
Empty Folders
- An empty folder is a folder with no files at any depth, including a folder that contains only empty folders.
- A folder that contains a hidden file, a cloud-only file, a link, or a skipped subfolder is not empty (so that deleting it never removes content the scan did not see).
- Show only the highest empty folder in each branch. A scan root is never listed.
The LLM (llm_judge.py)
Run the LLM only on duplicate groups. Do not run it on ZIP files or empty folders.
Structured Output (Pydantic)
class Judgement(BaseModel):
    verdict: Literal["identical", "copy_with_changes", "different"]
    confidence: float          # From 0 to 1
    keep: list[str]            # IDs of members recommended for retention;
                               # empty if verdict == "different"
    reason: str                # Short explanation in Hebrew, 1–3 sentences
v3 output schema (same fields, reason before keep, so the model explains before it chooses):
class JudgementReasonFirst(BaseModel):
    verdict: Literal["identical", "copy_with_changes", "different"]
    confidence: float
    reason: str
    keep: list[str]
The v3 output is converted to Judgement after parsing, so the rest of the system uses a single type. v1 and v2 use Judgement directly (keep before reason).
Validation (all versions):
- verdict is one of the three values, and confidence is between 0 and 1 (enforced by Pydantic).
- Every ID in keep must identify an existing member of the group.
- keep must be empty if verdict == "different", and must not be empty otherwise.
Input to the LLM
For each group member: ID (A, B, C, ...), folder name, full path, size, file count, latest modification date, and up to MAX_UNIQUE_FILES_TO_LLM unique files (relative paths and sizes), together with the total number of unique files. Also include the group's overlap percentage (Jaccard and containment). In v2 and v3, also include the common location of all members, the distinct part of each member's path, and the member's project ("inside project", or "none"). Do not send file contents; send only names and metadata.
Prompt Versions (for the Before-and-After Experiments)
All versions share the same task description and the meaning of each output field. The differences are only in the information and the criteria.
- v1 (baseline): The model receives only folder names and paths. Instruction: "Are these copies of the same folder, and which one should be kept?"
- v2 (full context): All the input described above, plus explicit criteria:
  - Names such as "- עותק" ("- Copy"), "- Copy", "Copy of", "(2)", "_old", "גיבוי" ("backup"), and "ישן" ("old") suggest a copy, but are not conclusive. The name alone is insufficient.
  - A copy is not always the redundant version. If it contains newer unique files, it may be the most up-to-date version.
  - Similar names with different content, such as "דוח 2024" ("Report 2024") and "דוח 2025" ("Report 2025"), should be classified as different.
  - Prefer retaining a copy in an organized location, such as Documents or Projects, over a temporary location, such as Downloads or Desktop. The location is judged only by the part of the path that differs between members: the code computes the common prefix of the member paths (component by component, case-insensitive) and sends it, together with each member's distinct path part, to the LLM (v2 and v3). Explicit instruction: "אם שתי התיקיות נמצאות באותו סוג מיקום, המיקום אינו שיקול" ("If both folders are in the same kind of location, location is not a consideration").
  - For a copy_with_changes group, explain which unique files would be lost if either side were deleted.
  - Assign low confidence when the indicators conflict.
  - If members are inside different projects, each project may need its own copy: keep every member, give low confidence (below 0.5), and say so in the reason.
- v3 (default): Identical to v2, with exactly two changes:
  - The output schema is JudgementReasonFirst (reason before keep).
  - An added instruction: the reason must explicitly name the IDs to keep and to delete, for example "לשמור את A, למחוק את B" ("keep A, delete B").
  - Why v3 is the default although its accuracy is similar to v2: in v2 the reason and keep could contradict each other (dataset example 15: the reason says to keep the folder in Documents, keep points to the one in Downloads, confidence 1.0). In v3 the reason names the IDs, so it can be checked against keep, and no contradiction was found. In addition, the case where v2 silently skipped a group ("different" used to mean "keep both") became a visible validation failure in v3, so the group is shown to the user without a recommendation instead of being skipped. The full analysis is in docs/findings.md (finding D).
Failure Handling
- On a connection failure (network, API key, quota, or timeout) or invalid output, retry once. The model's built-in retries are disabled (max_retries=0) so that exactly one retry happens; each call has a 60-second timeout.
- Classify each failed attempt into one of: timeout, quota, auth, connection, invalid_output, other. Log the category and the exception type in the warning.
- judge(group, prompt_version=None) returns Judgement or None. judge_with_details(...) also returns the list of failures ("category: ExceptionName" per failed attempt).
- If the retry also fails, return None. Display the item without a recommendation and with the message "לא התקבלה המלצה מהמערכת" ("No recommendation was received from the system"), and add a Hebrew warning to the run that names the failure type of the last attempt.
- Every judge call is traced to LangSmith as "judge", tagged prompt-v1 / prompt-v2 / prompt-v3, with prompt_version in its metadata. The code never enables or disables tracing itself; that is controlled only by LANGSMITH_TRACING in .env.
The Graph (graph.py)
State
class State(TypedDict):
    roots: list[str]
    settings: dict                 # overlap_threshold, min_folder_size_mb, confidence_threshold, include_cross_project
    queue: list[dict]              # Pending items: {"id", "type": "duplicate"|"zip", ...}
    hidden_children: dict          # group_id -> list of hidden groups
    empty_folders: list[str]
    current: dict | None
    judgement: dict | None
    decision: dict | None
    deleted_paths: list[str]
    freed_bytes: int
    warnings: list[str]
    log: list[dict]
    summary: dict | None           # Set by the summary node
    skipped_cross_project: int     # Groups not shown because all members are in different projects
    scan_result: Annotated[Any, UntrackedValue(object)]
                                   # Full scan data passed from scan to find_candidates in memory only;
                                   # it is large and never written to the checkpoint. If a run resumes
                                   # between these two nodes, find_candidates scans again.
Nodes and Flow
1. scan: Scan and build the data structures.
2. find_candidates: Build the queue (top-level duplicate groups and ZIP items) and the empty-folder list. Unless include_cross_project is on, remove cross-project groups and count them.
3. pick_next: First, close the previous item: if it ended without execute (a skip, or a group classified as different), write its log row here, because human_review must not write anything. Then remove the next item from the queue. Skip items if any of their paths has been deleted or is located inside a deleted path.
4. Routing after pick_next:
   - Queue empty and empty folders exist (or something was deleted, which may have created new empty folders): review_empty_folders.
   - Queue empty and no empty folders exist: summary.
   - Duplicate item: judge.
   - ZIP item: human_review.
5. judge: Invoke the LLM (judge_with_details with PROMPT_VERSION). If there is no recommendation, add a warning with the failure type.
6. Routing after judge:
   - verdict == "different": Return to pick_next and log "סווג כשונה" ("Classified as different").
   - Otherwise: human_review.
7. human_review: Call interrupt() with a payload containing the item, the recommendation (if available), a low_confidence flag, and the available actions.
   Resume values:
   - Duplicate: {"action": "delete", "paths": [...]}, {"action": "skip"}, or {"action": "delete_shared", "from": id, "keep_in": id}. The last one is offered only when the verdict is identical or copy_with_changes, and only for a pair with shared files; otherwise it becomes a skip with a warning.
   - ZIP: {"action": "delete_zip"}, {"action": "delete_folder"}, or {"action": "skip"}.
   An invalid resume value (unknown paths, deleting every member, or deleting none) becomes a skip with a warning.
8. Routing after human_review:
   - Delete: execute.
   - Skip a duplicate group that has hidden_children: expand_children.
   - Skip without children: pick_next.
9. expand_children: Add the hidden groups to the front of the queue so they are displayed immediately, then proceed to pick_next.
10. execute: For each path, recheck that it still exists and that its size and file count have not changed since the scan. If anything has changed, do not delete it and display a warning. Otherwise, call send2trash, update deleted_paths and freed_bytes, and record a log entry. Then proceed to pick_next.
    delete_shared: delete from member "from" only the files that have an identical copy in member "keep_in". Per file, before send2trash: (1) the file and its copy must both still exist, (2) both must still have the scanned SHA-256 (they are rehashed), and (3) the member that holds the copy must not be missing. A file that fails any check is not deleted and a warning is added; the rest continue. A file with no identical copy in another remaining member is never deleted. The folder itself is never deleted by this action (it may become empty, and is then handled by review_empty_folders). The member "keep_in" becomes protected (state.protected_paths) if at least one file was deleted: for the rest of the run it, anything inside it, and any folder that contains it, are never deleted as a whole, because it holds the only remaining copy. The action writes one log row.
11. review_empty_folders: A single interrupt containing all empty folders. Resume value: {"delete": [...paths]}. Delete the selected folders after rechecking that they are still empty, then proceed to summary.
12. summary: Counts of items reviewed, deleted, and skipped; space freed; warnings; the number of cross-project groups not shown.
Checkpointer
Use SqliteSaver with a local file (checkpoints.sqlite) so that a workflow can be resumed after restarting Streamlit.
Every graph invocation is sent with metadata {"thread_id": ...}, so a run and its later resumes are grouped together in the LangSmith Threads view. The recursion limit is raised (10,000) because the review loop can run many steps between interrupts.
Action Log (actions.py)
Append one row for every decision to logs/actions_log.csv, with these columns: timestamp, item_type, action, path, size_bytes, llm_verdict, llm_keep, llm_confidence, user_decision, recommendation_outcome, files_deleted, skipped_files.
files_deleted and skipped_files are filled only for delete_shared: the number of files deleted, and the relative paths of files that were skipped (separated by "; "). For delete_shared, size_bytes is the total size actually freed and path is the member the files were deleted from. A log written by an older version (fewer columns) is upgraded in place on the next write.
recommendation_outcome values:
- followed: the decision matches the recommendation (exactly the members not in keep were deleted; for delete_shared, files were removed from a member that is not in keep).
- overridden: different members were deleted than recommended (for delete_shared, files were removed from a member that is in keep).
- skipped: the user skipped a group that had a recommendation.
- empty: there was no recommendation (ZIP items, empty folders, LLM failure), or there was no user decision (an item classified as different).
- The log file is written with a UTF-8 BOM so that Excel shows Hebrew correctly. It contains real, private paths and is excluded from git, together with checkpoints.sqlite.
Streamlit Interface (app.py)
- Sidebar:
  - A text box for paths, one path per line. Validate that each path exists, is a folder, and is not on the skip list. Display a clear error message otherwise.
  - Brief instructions for copying a path from File Explorer.
  - Sliders: overlap threshold, minimum folder size, confidence threshold.
  - A checkbox "הצג גם כפילויות בין פרויקטים שונים" ("Also show duplicates between different projects"), off by default.
  - A "התחל סריקה" ("Start scan") button.
  - A "thread ID" field with a "המשך" ("Continue") button, to resume a previous run (like cli.py --thread).
- Store the graph, thread_id, and current state in st.session_state. The thread_id is also kept in the URL (?thread=...), so a page refresh returns to the same run. Resume with Command(resume=...) on each decision-button click.
- No double resume: every decision button carries the id of the interrupt it was rendered for. Before resuming, the pending interrupt in the checkpoint is checked under a lock; a double click, a refresh, or a stale page does nothing. The "Start scan" button uses a one-time token.
- During scanning: display a progress message.
- Duplicate item card:
  - A table of group members: path, size, file count, modification date, project.
  - Overlap percentage.
  - The recommendation and explanation, with a prominent banner saying "המערכת לא בטוחה, כדאי לבדוק" ("The system is uncertain; please review") if confidence is below the threshold.
  - For each member with unique files, show an expandable list and the warning "קבצים אלה יימחקו אם תמחקי תיקייה זו" ("These files will be deleted if you delete this folder").
  - Checkboxes for selecting members to delete (none checked by default), and buttons "מחק את המסומנים" ("Delete selected") and "דלג" ("Skip"). Do not allow deletion of every member of a group.
  - When delete_shared is offered: for each member and each other member that has shared files, a button "מחק מתיקייה X רק את הקבצים שקיימים גם ב-Y" ("Delete from folder X only the files that also exist in Y"), with the number of files and the size that would be freed. A short note explains that each file is rechecked and that the folder that keeps the copies becomes protected.
- ZIP card: ZIP path and folder path, sizes, whether extraction is complete, and a list of missing files if extraction is partial, with a warning. No recommendation. Buttons: "מחק את ה-ZIP" ("Delete ZIP"), "מחק את התיקייה" ("Delete folder"), and "דלג" ("Skip").
- Empty folders screen: A list with a checkbox for each folder, buttons "בחר הכל" ("Select all") and "נקה הכל" ("Clear all"), and a "מחק את המסומנות" ("Delete selected") button.
- Summary screen: Include a link to the log file; the number of shared files deleted by "delete only the shared files" (when there were any); and the number of cross-project groups that were not shown (when there were any). Both numbers are shown together when both apply.
- The interface must be in Hebrew and use right-to-left (RTL) layout. Tables and lists are HTML (st.dataframe does not support RTL). Left-to-right values inside Hebrew text (sizes such as "1.25 MB", IDs, paths) are isolated with <bdi> in HTML and with Unicode isolate marks in plain-text labels, so they are not reordered.
cli.py
Run the same graph in the terminal: accept paths, print each item, and read decisions from the keyboard. Intended for debugging before implementing Streamlit.
Test Folders (test_data/make_fixtures.py)
A script that creates a directory structure under a temporary folder with these known cases:
- A folder and an identical copy, with the suffix "- עותק" ("- Copy").
- Three identical copies: "פרויקט", "פרויקט - עותק", "פרויקט (2)" ("Project", "Project - Copy", "Project (2)").
- A copy with newer files added.
- Similar names with different contents: "דוח 2024", "דוח 2025" ("Report 2024", "Report 2025").
- Identical folders with completely different names.
- A duplicate top-level pair that also contains duplicate subfolders.
- A fully extracted ZIP, a partially extracted ZIP, and a ZIP without an extracted folder.
- Empty folders, including a folder containing only empty folders.
- A folder on the skip list containing duplicates, to verify that it is not scanned, and a hidden folder with a copy of another case, to verify that hidden folders are not scanned.
- Hebrew names.
- A copy with the same 3 files plus 4 new ones (Jaccard about 0.43, containment 1.0), detected only thanks to containment (case 10).
- The same library folder (fontawesome) inside two different website projects (case 11), to verify project detection. It is hidden by default and shown when include_cross_project is on.
- Two identical copies of a library inside the same project (case 12), which must still be shown by default.
- The script takes an optional target folder (python test_data\make_fixtures.py C:\demo) and never overwrites an existing fixtures folder.
LangSmith Evaluation (eval/)
- Evaluation scope: Only the judge function (input: group data; output: Judgement). Do not evaluate the entire graph or real folders.
- make_dataset.py: Create a duplicate-judge dataset with at least 35 examples using synthetic data in the LLM input format. Each example must include expected_verdict and acceptable_keeps (a list of acceptable keep lists; most examples have one, ambiguous ones have several). Save the examples to eval/dataset.json for review before uploading. Include at least 12 examples whose correct verdict is different. Include all case types from the test folders, especially tricky cases: a copy that is actually the most up-to-date version, similar names with different contents, and different names with identical contents.
- evaluators.py:
  - verdict_correct: Whether the verdict matches the expected verdict.
  - keep_correct: Whether the model's keep (as a set) equals one of the acceptable_keeps. Skip (return None) when the correct answer is different.
  - false_duplicate: Return 1 if the correct classification is different but the model classified the item as a duplicate. This is the most dangerous error. Scored only on examples whose correct answer is different, so its mean is the false-duplicate rate among them.
  - failed_calls: When judge returns None (failure after the retry), the example is not counted as an error in the other metrics; it is counted separately in failed_calls, with a breakdown by failure type (timeout, quota, auth, connection, invalid_output, other). Otherwise network failures would distort the comparison between versions.
  - Wait briefly between calls during an experiment to stay within the Gemini quota.
  - confidence_calibration: Whether errors are concentrated among low-confidence items. Report the average confidence for correct and incorrect answers, and the share of errors with confidence below DEFAULT_CONFIDENCE_THRESHOLD.
- The final dataset (duplicate-judge) has 46 examples: 22 identical, 10 copy_with_changes, 14 different. Each example's metadata has a description, a category and a running number, so results can be reported by example number. It includes the fixture cases, two folders in the same temporary location, a library inside two different projects (and a control case inside the same project), and cases inspired by the errors seen during development.
- run_experiment.py: Run evaluate() for each prompt version with experiment_prefix values v1-names-only, v2-full-context and v3-reason-first, one call at a time (max_concurrency=1). Print a comparison table of all metrics (including failed_calls by type), the examples where consecutive versions answered differently, and the LangSmith links. Options: --run selects which versions to run, and --reuse VERSION=EXPERIMENT_NAME includes an existing experiment in the comparison without running it again.
README.md
- Brief description and goal.
- Graph diagram using Mermaid, or an export from get_graph().draw_mermaid().
- Installation and local execution instructions, explicitly stating that execution is local only.
- Explanation of safety measures: skip list, Recycle Bin instead of permanent deletion, revalidation, approval for every item, and logging.
- Privacy: which data is sent to Gemini—names and metadata only, not file contents.
- Known limitations: individual duplicate files are not handled in this version; Windows only.
- Trace screenshots in docs/images, each with a caption that describes exactly what it shows: the interface duplicate card, pausing at an interrupt (human_review), the experiments page with v1, v2 and v3, and the v2 vs v3 answers for dataset example 15. No browser address bar, LangSmith organization ID, or personal paths in any screenshot, and no LangSmith links in the README (they are private).
- A table of the three experiments, including the caveat that v3's verdict_correct of 1.000 comes from one example moving from "error" to "failed call", not from a real improvement.
- The central insight from the traces (finding D in docs/findings.md): a prompt fix solved the problem it targeted but created a new failure mode, and the automated evaluation is what exposed it. Additional findings (A, C, E) and the evaluation limitations.
- Links to docs/SPEC.md and docs/findings.md; no links to files outside the project folder.
Build Order (Stop After Each Stage)
1. Setup: folder structure, config.py, requirements.txt, .env.example, and .gitignore.
2. scanner.py and test_data/make_fixtures.py, verifying that the scan finds exactly the expected cases.
3. llm_judge.py, with manual testing on 3–4 groups.
4. actions.py and graph.py, with a complete run through cli.py on the test folders.
5. app.py.
6. eval/: create the dataset, review it before uploading, and run the experiments (v1 and v2, then v3 after analyzing the v2 errors).
7. README.md.
Do not implement at this stage: duplicate detection for individual files. This is planned for the next version.
Additional Clarifications (Mandatory)
1. interrupt: In any node containing interrupt(), do not perform any side effects—deletion, file writes, or log writes—before calling interrupt(), because the node restarts from the beginning when execution resumes after a pause. Perform all such actions only in the execute and review_empty_folders nodes, after receiving the decision.
2. SqliteSaver in Streamlit: Create the connection using sqlite3.connect("checkpoints.sqlite", check_same_thread=False), pass it to SqliteSaver(conn), and cache the compiled graph using @st.cache_resource so it is not rebuilt on every click.
3. Overlap metric: In addition to Jaccard, calculate containment = the number of shared hashes divided by the number of hashes in the smaller folder. A pair is a candidate if either metric exceeds the threshold, subject to the "best partner only" rule in "Folder Comparison". Pass both values to the LLM and display them in the interface.
4. Empty folders: Recompute the list at the beginning of review_empty_folders by checking the disk directly, so that it includes folders that became empty as a result of deletions during the workflow.
5. ZIP: An extracted folder is a folder in the same parent directory as the ZIP file, whose name matches the ZIP filename without its extension, using a case-insensitive comparison. If it contains only a single folder with the same name—a common double-nesting pattern after extraction—compare against that inner folder.
6. Root folders: Always scan a folder explicitly entered by the user, even if it is marked as hidden. Skip rules apply only to folders inside it. If the entered path itself is on the skip list, display an error in the interface.
7. Tests: In stage 2, write pytest tests that run the scanner on the test folders and verify the expected results for every case.