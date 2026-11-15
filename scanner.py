"""Folder scanning and candidate detection (no LLM involved).

Public API:
    scan(roots) -> ScanResult
    find_candidates(scan_result, overlap_threshold, min_folder_size_mb) -> dict
    is_within(path, parent) -> bool
    validate_root(path) -> error message or None
    measure_folder(path), is_empty_on_disk(path), find_empty_folders_on_disk(roots), sha256_file(path)
"""

import hashlib
import os
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from itertools import combinations

import config

# Hash key prefix for files that were not hashed (unique size or unreadable).
# Such a key never matches another file, so the file always counts as unique.
UNIQUE_PREFIX = "unique:"

ZIP_FLAG_ENCRYPTED = 0x1
ZIP_FLAG_UTF8 = 0x800


@dataclass
class FileInfo:
    path: str
    size: int
    mtime: float
    hash_key: str = ""


@dataclass
class FolderInfo:
    path: str
    name: str
    parent: str | None  # None for a scan root
    files: list[FileInfo] = field(default_factory=list)  # Direct files only
    subfolders: list[str] = field(default_factory=list)
    # True if the folder directly contains any file or any skipped entry.
    has_direct_content: bool = False
    # Aggregated over every depth:
    total_size: int = 0
    file_count: int = 0
    latest_mtime: float = 0.0
    hashes: Counter = field(default_factory=Counter)
    is_empty: bool = False


@dataclass
class ScanResult:
    roots: list[str]
    folders: dict[str, FolderInfo] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)  # Parents before children
    warnings: list[str] = field(default_factory=list)
    project_cache: dict[str, bool] = field(default_factory=dict)  # folder -> is project root


# ---------------------------------------------------------------- Helpers


def is_within(path: str, parent: str) -> bool:
    """True if path equals parent or is located inside it."""
    path = os.path.normcase(os.path.normpath(path))
    parent = os.path.normcase(os.path.normpath(parent))
    return path == parent or path.startswith(parent.rstrip(os.sep) + os.sep)


def _format_time(timestamp: float) -> str:
    if not timestamp:
        return ""
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")


def _access_warning(path: str, error: OSError) -> str:
    if isinstance(error, PermissionError):
        return f"אין הרשאת גישה, דולג: {path}"
    return f"לא ניתן לגשת, דולג: {path} ({error.strerror or error})"


def _iter_files(scan_result: ScanResult, folder_path: str):
    """Yield all scanned files under a folder, at every depth."""
    stack = [folder_path]
    while stack:
        folder = scan_result.folders[stack.pop()]
        yield from folder.files
        stack.extend(folder.subfolders)


def validate_root(path: str) -> str | None:
    """Return a Hebrew error message if a user-entered root is not valid."""
    if not os.path.exists(path):
        return f"הנתיב לא קיים: {path}"
    if not os.path.isdir(path):
        return f"הנתיב אינו תיקייה: {path}"
    name = os.path.basename(os.path.normpath(path))
    if name.casefold() in config.SKIP_DIR_NAMES:
        return f"תיקייה זו נמצאת ברשימת הדילוג ולא תיסרק: {path}"
    return None


# ------------------------------------------------------------------- Scan


def scan(roots: list[str]) -> ScanResult:
    result = ScanResult(roots=_normalize_roots(roots))
    for root in result.roots:
        _walk(root, result)
    all_files = [f for folder in result.folders.values() for f in folder.files]
    _hash_files(all_files, result.warnings)
    _aggregate(result)
    return result


def _walk_without_hashing(roots: list[str]) -> ScanResult:
    """Same traversal and skip rules as scan(), without reading file contents."""
    result = ScanResult(roots=_normalize_roots(roots))
    for root in result.roots:
        _walk(root, result)
    _aggregate(result)
    return result


def measure_folder(path: str) -> tuple[int, int]:
    """Current (total size, file count) of a folder, counted like the scan."""
    folder = _walk_without_hashing([path]).folders[os.path.abspath(path)]
    return folder.total_size, folder.file_count


def is_empty_on_disk(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    return _walk_without_hashing([path]).folders[os.path.abspath(path)].is_empty


def find_empty_folders_on_disk(roots: list[str]) -> list[str]:
    """Recompute the empty-folder list directly from the disk."""
    existing = [root for root in roots if os.path.isdir(root)]
    return find_empty_folders(_walk_without_hashing(existing))


def _normalize_roots(roots: list[str]) -> list[str]:
    """Absolute paths, without duplicates or roots nested inside other roots."""
    kept = []
    for root in sorted({os.path.abspath(r) for r in roots}, key=len):
        if not any(is_within(root, other) for other in kept):
            kept.append(root)
    return kept


def _walk(root: str, result: ScanResult) -> None:
    # The root itself is always scanned; skip rules apply only inside it.
    stack = [(root, None)]
    while stack:
        path, parent = stack.pop()
        folder = FolderInfo(path=path, name=os.path.basename(path) or path, parent=parent)
        result.folders[path] = folder
        result.order.append(path)
        if parent is not None:
            result.folders[parent].subfolders.append(path)
        try:
            with os.scandir(path) as iterator:
                entries = list(iterator)
        except OSError as error:
            folder.has_direct_content = True
            result.warnings.append(_access_warning(path, error))
            continue
        for entry in entries:
            subfolder = _process_entry(entry, folder, result.warnings)
            if subfolder:
                stack.append((subfolder, path))


def _process_entry(entry: os.DirEntry, folder: FolderInfo, warnings: list[str]) -> str | None:
    """Record a file in the folder, or return a subfolder path to scan."""
    try:
        if len(entry.path) >= config.MAX_PATH_LENGTH:
            folder.has_direct_content = True
            warnings.append(f"נתיב ארוך מדי, דולג: {entry.path}")
            return None
        stat = entry.stat(follow_symlinks=False)
        attributes = getattr(stat, "st_file_attributes", 0)
        if _is_link(entry, attributes):
            folder.has_direct_content = True
            return None
        hidden_or_system = attributes & (config.FILE_ATTRIBUTE_HIDDEN | config.FILE_ATTRIBUTE_SYSTEM)
        if entry.is_dir(follow_symlinks=False):
            if hidden_or_system or entry.name.casefold() in config.SKIP_DIR_NAMES:
                folder.has_direct_content = True
                return None
            return entry.path
        # Any file, even a skipped one, means the folder is not empty.
        folder.has_direct_content = True
        if hidden_or_system or attributes & config.CLOUD_ONLY_ATTRIBUTES:
            return None  # Cloud-only files are never read (would trigger download)
        folder.files.append(FileInfo(entry.path, stat.st_size, stat.st_mtime))
    except OSError as error:
        folder.has_direct_content = True
        warnings.append(_access_warning(entry.path, error))
    return None


def _is_link(entry: os.DirEntry, attributes: int) -> bool:
    """Symlinks and junctions are never followed."""
    if entry.is_symlink():
        return True
    if attributes & config.FILE_ATTRIBUTE_REPARSE_POINT:
        tag = getattr(os.lstat(entry.path), "st_reparse_tag", 0)
        return tag == config.IO_REPARSE_TAG_MOUNT_POINT
    return False


def _hash_files(files: list[FileInfo], warnings: list[str]) -> None:
    """Hash only files that share their exact size with another file."""
    by_size = defaultdict(list)
    for file in files:
        by_size[file.size].append(file)
    for same_size in by_size.values():
        if len(same_size) == 1:
            same_size[0].hash_key = UNIQUE_PREFIX + same_size[0].path
            continue
        for file in same_size:
            digest = sha256_file(file.path)
            if digest is None:
                warnings.append(f"לא ניתן לקרוא את הקובץ (ייתכן שהוא נעול), דולג: {file.path}")
                digest = UNIQUE_PREFIX + file.path
            file.hash_key = digest


def sha256_file(path: str) -> str | None:
    """SHA-256 of a file, read in blocks; None if it cannot be read."""
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            while block := handle.read(config.HASH_BLOCK_SIZE):
                digest.update(block)
    except OSError:
        return None
    return digest.hexdigest()


def _aggregate(result: ScanResult) -> None:
    """Compute per-folder totals bottom-up (children before parents)."""
    for path in reversed(result.order):
        folder = result.folders[path]
        folder.total_size = sum(f.size for f in folder.files)
        folder.file_count = len(folder.files)
        folder.latest_mtime = max((f.mtime for f in folder.files), default=0.0)
        folder.hashes = Counter(f.hash_key for f in folder.files)
        folder.is_empty = not folder.has_direct_content
        for sub_path in folder.subfolders:
            sub = result.folders[sub_path]
            folder.total_size += sub.total_size
            folder.file_count += sub.file_count
            folder.latest_mtime = max(folder.latest_mtime, sub.latest_mtime)
            folder.hashes.update(sub.hashes)
            folder.is_empty = folder.is_empty and sub.is_empty


# ------------------------------------------------------------- Candidates


def find_candidates(scan_result: ScanResult, overlap_threshold: float, min_folder_size_mb: float) -> dict:
    groups, hidden_children = find_duplicate_groups(scan_result, overlap_threshold, min_folder_size_mb)
    return {
        "queue": groups + find_zip_items(scan_result),
        "hidden_children": hidden_children,
        "empty_folders": find_empty_folders(scan_result),
    }


# ------------------------------------------------------ Duplicate folders


def find_duplicate_groups(
    scan_result: ScanResult, overlap_threshold: float, min_folder_size_mb: float
) -> tuple[list[dict], dict[str, list[dict]]]:
    """Return (top-level groups, hidden_children by parent group id)."""
    folders = scan_result.folders
    min_bytes = min_folder_size_mb * 1024 * 1024
    eligible = [
        f for f in folders.values()
        if f.file_count > 0 and f.total_size >= min_bytes and not _same_content_as_parent(scan_result, f)
    ]

    # Inverted index: hash -> folders containing it. Only folders sharing a hash are compared.
    index = defaultdict(list)
    for folder in eligible:
        for key in folder.hashes:
            if not key.startswith(UNIQUE_PREFIX):
                index[key].append(folder.path)
    pairs = set()
    for paths in index.values():
        for a, b in combinations(paths, 2):
            pairs.add((a, b) if a < b else (b, a))

    union_parent = {}

    def find_root(path):
        while union_parent.get(path, path) != path:
            path = union_parent[path]
        return path

    # Pass 1: every pair that passes the threshold on either metric.
    candidate_pairs = []
    for path_a, path_b in sorted(pairs):
        if is_within(path_a, path_b) or is_within(path_b, path_a):
            continue
        jaccard, containment = _overlap(folders[path_a].hashes, folders[path_b].hashes)
        if max(jaccard, containment) >= overlap_threshold:
            candidate_pairs.append((path_a, path_b, jaccard, containment))

    # Pass 2: each folder stays only with its best partner. A pair that is not identical
    # is dropped if its smaller folder has another pair with a higher Jaccard.
    best_jaccard = defaultdict(float)
    for path_a, path_b, jaccard, _ in candidate_pairs:
        best_jaccard[path_a] = max(best_jaccard[path_a], jaccard)
        best_jaccard[path_b] = max(best_jaccard[path_b], jaccard)

    partial_groups = []
    for path_a, path_b, jaccard, containment in candidate_pairs:
        if folders[path_a].hashes == folders[path_b].hashes:
            root_a, root_b = find_root(path_a), find_root(path_b)
            union_parent.setdefault(root_b, root_b)
            if root_a != root_b:
                union_parent[root_a] = root_b
            continue
        smaller = min(path_a, path_b, key=lambda p: (folders[p].file_count, folders[p].total_size))
        if best_jaccard[smaller] > jaccard:
            continue
        partial_groups.append(_build_group(scan_result, [path_a, path_b], jaccard, containment))

    components = defaultdict(list)
    for path in union_parent:
        components[find_root(path)].append(path)
    identical_groups = [
        _build_group(scan_result, sorted(paths), 1.0, 1.0) for paths in components.values()
    ]

    groups = sorted(
        identical_groups + partial_groups,
        key=lambda g: [os.path.normcase(m["path"]) for m in g["members"]],
    )
    for number, group in enumerate(groups, start=1):
        group["id"] = f"g{number}"
    return _split_top_level(groups)


def _same_content_as_parent(scan_result: ScanResult, folder: FolderInfo) -> bool:
    """A folder whose parent holds nothing else is represented by the parent."""
    parent = scan_result.folders.get(folder.parent)
    return parent is not None and parent.file_count == folder.file_count


def _overlap(hashes_a: Counter, hashes_b: Counter) -> tuple[float, float]:
    """Multiset Jaccard and containment (shared / smaller folder)."""
    shared = sum((hashes_a & hashes_b).values())
    union = sum((hashes_a | hashes_b).values())
    smaller = min(sum(hashes_a.values()), sum(hashes_b.values()))
    return shared / union, shared / smaller


def _is_project_root(path: str) -> bool:
    try:
        with os.scandir(path) as iterator:
            for entry in iterator:
                name = entry.name.casefold()
                if name in config.PROJECT_MARKER_NAMES or os.path.splitext(name)[1] in config.PROJECT_MARKER_EXTENSIONS:
                    if entry.is_file(follow_symlinks=False):
                        return True
    except OSError:
        pass
    return False


def find_project(path: str, cache: dict | None = None) -> str | None:
    """The nearest ancestor folder (not the folder itself) that is a project root, or None."""
    cache = {} if cache is None else cache
    current = os.path.dirname(os.path.abspath(path))
    while True:
        if current not in cache:
            cache[current] = _is_project_root(current)
        if cache[current]:
            return current
        parent = os.path.dirname(current)
        if parent == current:
            return None
        current = parent


def _shared_files(files: list[FileInfo], path: str, other_files: list[FileInfo], other_path: str) -> list[dict]:
    """Files of one member that have an identical copy (same SHA-256) in another member."""
    copy_by_hash = {}
    for file in other_files:
        if not file.hash_key.startswith(UNIQUE_PREFIX):
            copy_by_hash.setdefault(file.hash_key, file)
    shared = [
        {
            "relative_path": os.path.relpath(f.path, path),
            "size_bytes": f.size,
            "sha256": f.hash_key,
            "other_relative_path": os.path.relpath(copy_by_hash[f.hash_key].path, other_path),
        }
        for f in files
        if f.hash_key in copy_by_hash
    ]
    return sorted(shared, key=lambda s: s["relative_path"])


def _build_group(scan_result: ScanResult, member_paths: list[str], jaccard: float, containment: float) -> dict:
    folders = scan_result.folders
    project_cache = scan_result.project_cache
    files_by_member = {path: list(_iter_files(scan_result, path)) for path in member_paths}
    ids = {path: chr(ord("A") + number) for number, path in enumerate(member_paths)}
    members = []
    for path in member_paths:
        other_hashes = set()
        for other in member_paths:
            if other != path:
                other_hashes.update(folders[other].hashes)
        unique_files = sorted(
            (
                {"relative_path": os.path.relpath(f.path, path), "size_bytes": f.size}
                for f in files_by_member[path]
                if f.hash_key not in other_hashes
            ),
            key=lambda u: u["relative_path"],
        )
        # For each other member: the files here that also exist there (for "delete only shared files").
        shared_with = {}
        for other in member_paths:
            if other == path:
                continue
            shared = _shared_files(files_by_member[path], path, files_by_member[other], other)
            if shared:
                shared_with[ids[other]] = {
                    "count": len(shared),
                    "size_bytes": sum(s["size_bytes"] for s in shared),
                    "files": shared,
                }
        folder = folders[path]
        members.append({
            "id": ids[path],
            "path": path,
            "name": folder.name,
            "size_bytes": folder.total_size,
            "file_count": folder.file_count,
            "latest_modified": _format_time(folder.latest_mtime),
            "unique_files": unique_files,
            "unique_count": len(unique_files),
            "project": find_project(path, project_cache),  # Path of the enclosing project, or None
            "shared_with": shared_with,  # other member ID -> {"count", "size_bytes", "files"}
        })
    return {
        "type": "duplicate",
        "jaccard": round(jaccard, 3),
        "containment": round(containment, 3),
        "members": members,
    }


def _covers(outer: dict, inner: dict) -> bool:
    """True if every member of inner is located inside a member of outer."""
    return all(
        any(is_within(m["path"], o["path"]) for o in outer["members"])
        for m in inner["members"]
    )


def _split_top_level(groups: list[dict]) -> tuple[list[dict], dict[str, list[dict]]]:
    top_level = []
    hidden_children = defaultdict(list)
    for group in groups:
        covering = [other for other in groups if other is not group and _covers(other, group)]
        if not covering:
            top_level.append(group)
            continue
        # Attach to the closest (smallest) covering group, so skipping reveals one level at a time.
        parent = min(covering, key=lambda g: (sum(m["size_bytes"] for m in g["members"]), g["id"]))
        hidden_children[parent["id"]].append(group)
    return top_level, dict(hidden_children)


# -------------------------------------------------------------- ZIP files


def find_zip_items(scan_result: ScanResult) -> list[dict]:
    items = []
    for path in scan_result.order:
        folder = scan_result.folders[path]
        for file in folder.files:
            if file.path.casefold().endswith(".zip"):
                item = _check_zip(scan_result, folder, file)
                if item:
                    items.append(item)
    for number, item in enumerate(items, start=1):
        item["id"] = f"z{number}"
    return items


def _check_zip(scan_result: ScanResult, parent: FolderInfo, zip_file: FileInfo) -> dict | None:
    stem = os.path.splitext(os.path.basename(zip_file.path))[0].casefold()
    target = next(
        (scan_result.folders[s] for s in parent.subfolders if scan_result.folders[s].name.casefold() == stem),
        None,
    )
    if target is None:
        return None
    entries = _read_zip_entries(zip_file.path, scan_result.warnings)
    if not entries:
        return None
    compared = _unwrap_double_nesting(scan_result, target)
    existing = {
        os.path.relpath(f.path, compared.path).casefold(): f.size
        for f in _iter_files(scan_result, compared.path)
    }
    missing = [rel for rel, size in entries if existing.get(rel.casefold()) != size]
    if len(missing) == len(entries):
        return None  # Nothing was extracted: not a candidate
    return {
        "type": "zip",
        "zip_path": zip_file.path,
        "zip_size_bytes": zip_file.size,
        "folder_path": target.path,
        "folder_size_bytes": target.total_size,
        "folder_file_count": target.file_count,
        "zip_file_count": len(entries),
        "fully_extracted": not missing,
        "missing_files": missing,
    }


def _read_zip_entries(path: str, warnings: list[str]) -> list[tuple[str, int]] | None:
    """Return (relative path, uncompressed size) for each file in the archive."""
    try:
        with zipfile.ZipFile(path) as archive:
            infos = [info for info in archive.infolist() if not info.is_dir()]
    except (zipfile.BadZipFile, zipfile.LargeZipFile, OSError, ValueError, EOFError):
        warnings.append(f"קובץ ZIP פגום או לא קריא, דולג: {path}")
        return None
    if any(info.flag_bits & ZIP_FLAG_ENCRYPTED for info in infos):
        warnings.append(f"קובץ ZIP מוגן בסיסמה, דולג: {path}")
        return None
    return [(os.path.normpath(_zip_entry_name(info)), info.file_size) for info in infos]


def _zip_entry_name(info: zipfile.ZipInfo) -> str:
    """Recover Hebrew names from archives that do not set the UTF-8 flag."""
    name = info.filename
    if info.flag_bits & ZIP_FLAG_UTF8:
        return name
    raw = name.encode("cp437")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        pass
    hebrew = raw.decode("cp862", errors="replace")
    if any("א" <= ch <= "ת" for ch in hebrew):
        return hebrew
    return name


def _unwrap_double_nesting(scan_result: ScanResult, folder: FolderInfo) -> FolderInfo:
    """Handle 'name/name/...' produced by some extraction tools."""
    if not folder.files and len(folder.subfolders) == 1:
        inner = scan_result.folders[folder.subfolders[0]]
        if inner.name.casefold() == folder.name.casefold():
            return inner
    return folder


# ---------------------------------------------------------- Empty folders


def find_empty_folders(scan_result: ScanResult) -> list[str]:
    """Highest empty folder in each branch. Scan roots themselves are never listed."""
    folders = scan_result.folders
    empty = []
    for path in scan_result.order:
        folder = folders[path]
        if folder.parent is None or not folder.is_empty:
            continue
        parent = folders[folder.parent]
        if parent.parent is None or not parent.is_empty:
            empty.append(path)
    return sorted(empty, key=os.path.normcase)
