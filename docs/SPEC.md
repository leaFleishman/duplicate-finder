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
├── llm_judge.py       # Pydantic model, two prompt versions, Gemini invocation
├── graph.py           # State, nodes, routing, interrupt, checkpointer
├── actions.py         # Revalidation, moving to the Recycle Bin, action logging
├── app.py             # Streamlit interface
├── cli.py             # Terminal execution for debugging
├── eval/
│   ├── make_dataset.py
│   ├── evaluators.py
│   └── run_experiment.py
├── test_data/
│   └── make_fixtures.py
├── logs/              # Action log (actions_log.csv)
└── README.md
config.py
- SKIP_DIR_NAMES: Windows, Program Files, Program Files (x86), ProgramData, $Recycle.Bin, System Volume Information, AppData, .git, .svn, .vs, .idea, .vscode, node_modules, bin, obj, venv, .venv, pycache.
- DEFAULT_OVERLAP_THRESHOLD = 0.7: Minimum overlap for a pair to qualify as a candidate.
- DEFAULT_MIN_FOLDER_SIZE_MB = 1
- DEFAULT_CONFIDENCE_THRESHOLD = 0.7
- MAX_UNIQUE_FILES_TO_LLM = 20
- PROMPT_VERSION = "v2" (can be changed to "v1")
- GEMINI_MODEL
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
Show Only the Top-Level Group
- Hide a group if each of its members is located inside a member of another candidate group. Store these hidden groups separately (hidden_children), associated with their parent group, so they can be revealed if the user skips the top-level group.
Extracted ZIP Files
- For each .zip file, check whether a folder with the same name as the ZIP filename without its extension exists in the same directory or in an adjacent directory.
- Compare the archive's file list (zipfile, relative paths and uncompressed sizes) with the folder's contents.
- Fully extracted: all files exist. Partially extracted: some files exist; retain the list of missing files. If none of the files exist, the item is not a candidate.
- Corrupted or password-protected ZIP files: skip and record a warning.
Empty Folders
- An empty folder is a folder with no files at any depth, including a folder that contains only empty folders.
- Show only the highest empty folder in each branch.
The LLM (llm_judge.py)
Run the LLM only on duplicate groups. Do not run it on ZIP files or empty folders.
Structured Output (Pydantic)
class Judgement(BaseModel):
    verdict: Literal["identical", "copy_with_changes", "different"]
    confidence: float          # From 0 to 1
    keep: list[str]            # IDs of members recommended for retention;
                               # empty if verdict == "different"
    reason: str                # Short explanation in Hebrew, 1–3 sentences
Validation: every ID in keep must identify an existing member of the group.
Input to the LLM
For each group member: ID (A, B, C, ...), folder name, full path, size, file count, latest modification date, and up to MAX_UNIQUE_FILES_TO_LLM unique files (relative paths and sizes), together with the total number of unique files. Also include the group's overlap percentage. Do not send file contents; send only names and metadata.
Two Prompt Versions (for the Before-and-After Experiment)
- v1 (baseline): The model receives only folder names and paths. Instruction: "Are these copies of the same folder, and which one should be kept?"
- v2 (improved): All the input described above, plus explicit criteria:
  - Names such as "- עותק" ("- Copy"), "- Copy", "Copy of", "(2)", "_old", "גיבוי" ("backup"), and "ישן" ("old") suggest a copy, but are not conclusive. The name alone is insufficient.
  - A copy is not always the redundant version. If it contains newer unique files, it may be the most up-to-date version.
  - Similar names with different content, such as "דוח 2024" ("Report 2024") and "דוח 2025" ("Report 2025"), should be classified as different.
  - Prefer retaining a copy in an organized location, such as Documents or Projects, over a temporary location, such as Downloads or Desktop. The location is judged only by the part of the path that differs between members: the code computes the common prefix of the member paths and sends it, together with each member's distinct path part, to the LLM (v2 only). Explicit instruction: "אם שתי התיקיות נמצאות באותו סוג מיקום, המיקום אינו שיקול" ("If both folders are in the same kind of location, location is not a consideration").
  - For a copy_with_changes group, explain which unique files would be lost if either side were deleted.
  - Assign low confidence when the indicators conflict.
Failure Handling
- On a connection failure (network, API key, quota, or timeout) or invalid output, retry once.
- If the retry also fails, return None. Display the item without a recommendation and with the message "לא התקבלה המלצה מהמערכת" ("No recommendation was received from the system").
The Graph (graph.py)
State
class State(TypedDict):
    roots: list[str]
    settings: dict                 # overlap_threshold, min_folder_size_mb, confidence_threshold
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
Nodes and Flow
1. scan: Scan and build the data structures.
2. find_candidates: Build the queue (top-level duplicate groups and ZIP items) and the empty-folder list.
3. pick_next: Remove the next item from the queue. Skip items if any of their paths has been deleted or is located inside a deleted path.
4. Routing after pick_next:
   - Queue empty and empty folders exist: review_empty_folders.
   - Queue empty and no empty folders exist: summary.
   - Duplicate item: judge.
   - ZIP item: human_review.
5. judge: Invoke the LLM.
6. Routing after judge:
   - verdict == "different": Return to pick_next and log "סווג כשונה" ("Classified as different").
   - Otherwise: human_review.
7. human_review: Call interrupt() with a payload containing the item, the recommendation (if available), a low_confidence flag, and the available actions.
   Resume values:
   - Duplicate: {"action": "delete", "paths": [...]} or {"action": "skip"}.
   - ZIP: {"action": "delete_zip"}, {"action": "delete_folder"}, or {"action": "skip"}.
8. Routing after human_review:
   - Delete: execute.
   - Skip a duplicate group that has hidden_children: expand_children.
   - Skip without children: pick_next.
9. expand_children: Add the hidden groups to the front of the queue so they are displayed immediately, then proceed to pick_next.
10. execute: For each path, recheck that it still exists and that its size and file count have not changed since the scan. If anything has changed, do not delete it and display a warning. Otherwise, call send2trash, update deleted_paths and freed_bytes, and record a log entry. Then proceed to pick_next.
11. review_empty_folders: A single interrupt containing all empty folders. Resume value: {"delete": [...paths]}. Delete the selected folders after rechecking that they are still empty, then proceed to summary.
12. summary: Counts of items reviewed, deleted, and skipped; space freed; warnings.
Checkpointer
Use SqliteSaver with a local file (checkpoints.sqlite) so that a workflow can be resumed after restarting Streamlit.
Action Log (actions.py)
Append one row for every decision to logs/actions_log.csv, with these columns: timestamp, item_type, action, path, size_bytes, llm_verdict, llm_keep, llm_confidence, user_decision, recommendation_outcome.
recommendation_outcome values:
- followed: the decision matches the recommendation (exactly the members not in keep were deleted).
- overridden: different members were deleted than recommended.
- skipped: the user skipped a group that had a recommendation.
- empty: there was no recommendation (ZIP items, empty folders, LLM failure), or there was no user decision (an item classified as different).
Streamlit Interface (app.py)
- Sidebar:
  - A text box for paths, one path per line. Validate that each path exists, is a folder, and is not on the skip list. Display a clear error message otherwise.
  - Brief instructions for copying a path from File Explorer.
  - Sliders: overlap threshold, minimum folder size, confidence threshold.
  - A "התחל סריקה" ("Start scan") button.
- Store the graph, thread_id, and current state in st.session_state. Resume with Command(resume=...) on each decision-button click.
- During scanning: display a progress message.
- Duplicate item card:
  - A table of group members: path, size, file count, modification date.
  - Overlap percentage.
  - The recommendation and explanation, with a prominent banner saying "המערכת לא בטוחה, כדאי לבדוק" ("The system is uncertain; please review") if confidence is below the threshold.
  - For each member with unique files, show an expandable list and the warning "קבצים אלה יימחקו אם תמחקי תיקייה זו" ("These files will be deleted if you delete this folder").
  - Checkboxes for selecting members to delete, and buttons "מחק את המסומנים" ("Delete selected") and "דלג" ("Skip"). Do not allow deletion of every member of a group.
- ZIP card: ZIP path and folder path, sizes, whether extraction is complete, and a list of missing files if extraction is partial, with a warning. No recommendation. Buttons: "מחק את ה-ZIP" ("Delete ZIP"), "מחק את התיקייה" ("Delete folder"), and "דלג" ("Skip").
- Empty folders screen: A list with a checkbox for each folder, buttons "בחר הכל" ("Select all") and "נקה הכל" ("Clear all"), and a "מחק את המסומנות" ("Delete selected") button.
- Summary screen: Include a link to the log file.
- The interface must be in Hebrew and use right-to-left (RTL) layout.
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
- A folder on the skip list containing duplicates, to verify that it is not scanned.
- Hebrew names.
LangSmith Evaluation (eval/)
- Evaluation scope: Only the judge function (input: group data; output: Judgement). Do not evaluate the entire graph or real folders.
- make_dataset.py: Create a duplicate-judge dataset with at least 35 examples using synthetic data in the LLM input format. Each example must include expected_verdict and acceptable_keeps (a list of acceptable keep lists; most examples have one, ambiguous ones have several). Save the examples to eval/dataset.json for review before uploading. Include at least 12 examples whose correct verdict is different. Include all case types from the test folders, especially tricky cases: a copy that is actually the most up-to-date version, similar names with different contents, and different names with identical contents.
- evaluators.py:
  - verdict_correct: Whether the verdict matches the expected verdict.
  - keep_correct: Whether the model's keep (as a set) equals one of the acceptable_keeps. Skip (return None) when the correct answer is different.
  - false_duplicate: Return 1 if the correct classification is different but the model classified the item as a duplicate. This is the most dangerous error.
  - failed_calls: When judge returns None (failure after the retry), the example is not counted as an error in the other metrics; it is counted separately in failed_calls, with a breakdown by failure type (timeout, quota, auth, connection, invalid_output, other). Otherwise network failures would distort the comparison between versions.
  - Wait briefly between calls during an experiment to stay within the Gemini quota.
  - confidence_calibration: Whether errors are concentrated among low-confidence items. Report the average confidence for correct and incorrect answers.
- run_experiment.py: Run evaluate() twice, using experiment_prefix values v1-names-only and v2-full-context, and print a summary of both experiments.
README.md
- Brief description and goal.
- Graph diagram using Mermaid, or an export from get_graph().draw_mermaid().
- Installation and local execution instructions, explicitly stating that execution is local only.
- Explanation of safety measures: skip list, Recycle Bin instead of permanent deletion, revalidation, approval for every item, and logging.
- Privacy: which data is sent to Gemini—names and metadata only, not file contents.
- Known limitations: individual duplicate files are not handled in this version; Windows only.
- Placeholders for trace screenshots: pausing at an interrupt, the expand_children loop, and a comparison of the v1 and v2 experiments.
- A table of experiment results and a placeholder for an insight.
Build Order (Stop After Each Stage)
1. Setup: folder structure, config.py, requirements.txt, .env.example, and .gitignore.
2. scanner.py and test_data/make_fixtures.py, verifying that the scan finds exactly the expected cases.
3. llm_judge.py, with manual testing on 3–4 groups.
4. actions.py and graph.py, with a complete run through cli.py on the test folders.
5. app.py.
6. eval/: create the dataset and run both experiments.
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