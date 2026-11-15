"""Constants and defaults for the duplicate folder finder."""

import os

# Folder names that are never scanned (compared case-insensitively).
SKIP_DIR_NAMES = {
    "windows",
    "program files",
    "program files (x86)",
    "programdata",
    "$recycle.bin",
    "system volume information",
    "appdata",
    ".git",
    ".svn",
    ".vs",
    ".idea",
    ".vscode",
    "node_modules",
    "bin",
    "obj",
    "venv",
    ".venv",
    "__pycache__",
}

# Windows file attributes.
FILE_ATTRIBUTE_HIDDEN = 0x2
FILE_ATTRIBUTE_SYSTEM = 0x4
FILE_ATTRIBUTE_REPARSE_POINT = 0x400
FILE_ATTRIBUTE_OFFLINE = 0x1000
FILE_ATTRIBUTE_RECALL_ON_OPEN = 0x40000
FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS = 0x400000
CLOUD_ONLY_ATTRIBUTES = (
    FILE_ATTRIBUTE_OFFLINE
    | FILE_ATTRIBUTE_RECALL_ON_OPEN
    | FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS
)
IO_REPARSE_TAG_MOUNT_POINT = 0xA0000003  # Junction

# Paths this long are skipped with a warning (classic Windows MAX_PATH).
MAX_PATH_LENGTH = 260

# A folder is a project root if it directly contains one of these files.
# Any .html file is not enough (saved web pages, exported reports); only index.html counts.
PROJECT_MARKER_NAMES = {"package.json", "requirements.txt", "index.html"}
PROJECT_MARKER_EXTENSIONS = {".csproj", ".sln"}

# Detection defaults (adjustable from the UI).
DEFAULT_OVERLAP_THRESHOLD = 0.7
DEFAULT_MIN_FOLDER_SIZE_MB = 1
DEFAULT_CONFIDENCE_THRESHOLD = 0.7

# LLM settings.
MAX_UNIQUE_FILES_TO_LLM = 20
PROMPT_VERSION = "v3"  # "v1" names only, "v2" full context, "v3" = v2 with reason before keep
GEMINI_MODEL = "gemini-3.1-flash-lite"

# Hashing.
HASH_BLOCK_SIZE = 1024 * 1024  # 1 MB

# Local files (relative to the project folder, regardless of the working directory).
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DB_PATH = os.path.join(PROJECT_DIR, "checkpoints.sqlite")
ACTIONS_LOG_PATH = os.path.join(PROJECT_DIR, "logs", "actions_log.csv")
