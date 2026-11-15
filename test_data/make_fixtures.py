"""Create a folder tree with known duplicate, ZIP and empty-folder cases.

Usage:
    python test_data/make_fixtures.py [target_dir]

Without target_dir, a new temporary folder is created. The fixtures are
written to <target_dir>/fixtures and that path is printed.
File contents are deterministic pseudo-random bytes, so equal (seed, size)
pairs produce identical files.
"""

import ctypes
import os
import random
import sys
import tempfile
import zipfile
from datetime import datetime

KB = 1000
OLD_TIME = datetime(2024, 1, 15, 10, 0).timestamp()
NEW_TIME = datetime(2025, 6, 1, 12, 0).timestamp()
FILE_ATTRIBUTE_HIDDEN = 0x2


def _content(seed: str, size: int) -> bytes:
    return random.Random(seed).randbytes(size)


def _write_files(folder: str, files: dict, mtime: float = OLD_TIME) -> None:
    """files: {"relative/path": (seed, size)}"""
    os.makedirs(folder, exist_ok=True)
    for relative, (seed, size) in files.items():
        path = os.path.join(folder, *relative.split("/"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as handle:
            handle.write(_content(seed, size))
        os.utime(path, (mtime, mtime))


def _make_zip(path: str, files: dict) -> None:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for relative, (seed, size) in files.items():
            archive.writestr(relative, _content(seed, size))


def _set_hidden(path: str) -> None:
    if not ctypes.windll.kernel32.SetFileAttributesW(path, FILE_ATTRIBUTE_HIDDEN):
        raise OSError(f"Could not set hidden attribute on {path}")


def create_fixtures(target_dir: str) -> str:
    root = os.path.join(target_dir, "fixtures")
    os.makedirs(root)  # Fails if it already exists, so nothing is overwritten

    # 1. A folder and an identical copy with "- עותק".
    case = os.path.join(root, "01_copy_suffix")
    documents = {
        "סיכום.docx": ("docs-1", 500 * KB),
        "תמונה.jpg": ("docs-2", 520 * KB),
        "נתונים.xlsx": ("docs-3", 480 * KB),
    }
    for name in ("מסמכים", "מסמכים - עותק"):
        _write_files(os.path.join(case, name), documents)

    # 2. Three identical copies.
    case = os.path.join(root, "02_three_copies")
    project = {
        "main.py": ("project-1", 400 * KB),
        "נתונים/קלט.csv": ("project-2", 450 * KB),
        "README.md": ("project-3", 300 * KB),
    }
    for name in ("פרויקט", "פרויקט - עותק", "פרויקט (2)"):
        _write_files(os.path.join(case, name), project)

    # 3. A copy with a newer file added (the copy is the up-to-date version).
    case = os.path.join(root, "03_newer_copy")
    thesis = {
        "פרק 1.docx": ("thesis-1", 300 * KB),
        "פרק 2.docx": ("thesis-2", 310 * KB),
        "פרק 3.docx": ("thesis-3", 320 * KB),
        "פרק 4.docx": ("thesis-4", 330 * KB),
    }
    _write_files(os.path.join(case, "תזה"), thesis)
    _write_files(os.path.join(case, "תזה - עותק"), thesis)
    _write_files(os.path.join(case, "תזה - עותק"), {"פרק 5 - חדש.docx": ("thesis-new", 350 * KB)}, NEW_TIME)

    # 4. Similar names, different main content (shared template files only).
    case = os.path.join(root, "04_similar_names")
    common = {
        "תבנית.dotx": ("report-common-1", 150 * KB),
        "לוגו.png": ("report-common-2", 160 * KB),
        "נספח א.pdf": ("report-common-3", 170 * KB),
        "נספח ב.pdf": ("report-common-4", 180 * KB),
    }
    _write_files(os.path.join(case, "דוח 2024"), {**common, "דוח 2024.docx": ("report-2024", 600 * KB)})
    _write_files(os.path.join(case, "דוח 2025"), {**common, "דוח 2025.docx": ("report-2025", 610 * KB)})

    # 5. Identical folders with completely different names.
    case = os.path.join(root, "05_different_names")
    course = {
        "הרצאה 1.pptx": ("course-1", 400 * KB),
        "הרצאה 2.pptx": ("course-2", 420 * KB),
        "תרגיל.pdf": ("course-3", 380 * KB),
    }
    for name in ("חומרי קורס", "random_stuff_17"):
        _write_files(os.path.join(case, name), course)

    # 6. A duplicate top-level pair that also contains duplicate subfolders.
    case = os.path.join(root, "06_nested")
    site = {
        "index.html": ("site-1", 50 * KB),
        "תמונות/רקע.jpg": ("site-img-1", 600 * KB),
        "תמונות/לוגו.png": ("site-img-2", 620 * KB),
        "קוד/app.js": ("site-code-1", 200 * KB),
        "קוד/style.css": ("site-code-2", 100 * KB),
    }
    for name in ("אתר", "אתר גיבוי"):
        _write_files(os.path.join(case, name), site)

    # 7. ZIP files.
    case = os.path.join(root, "07_zips")
    full = {"קובץ 1.txt": ("zip-full-1", 40 * KB), "תיקייה פנימית/קובץ 2.txt": ("zip-full-2", 50 * KB)}
    _write_files(os.path.join(case, "ארכיון מלא"), full)
    _make_zip(os.path.join(case, "ארכיון מלא.zip"), full)

    partial = {"א.txt": ("zip-part-1", 30 * KB), "ב.txt": ("zip-part-2", 31 * KB), "ג.txt": ("zip-part-3", 32 * KB)}
    _write_files(os.path.join(case, "חלקי"), {k: partial[k] for k in ("א.txt", "ב.txt")})
    _make_zip(os.path.join(case, "חלקי.zip"), partial)

    _make_zip(os.path.join(case, "בודד.zip"), {"לבד.txt": ("zip-alone-1", 20 * KB)})

    nested = {"מסמך.txt": ("zip-nest-1", 25 * KB), "עוד/מסמך 2.txt": ("zip-nest-2", 26 * KB)}
    _write_files(os.path.join(case, "כפול", "כפול"), nested)
    _make_zip(os.path.join(case, "כפול.zip"), nested)

    with open(os.path.join(case, "פגום.zip"), "wb") as handle:
        handle.write(_content("zip-broken", 5 * KB))
    _write_files(os.path.join(case, "פגום"), {"משהו.txt": ("zip-broken-1", 10 * KB)})

    # 8. Empty folders, including a folder that contains only empty folders.
    case = os.path.join(root, "08_empty")
    _write_files(case, {"הערה.txt": ("empty-note", 1 * KB)})
    os.makedirs(os.path.join(case, "ריקה"))
    os.makedirs(os.path.join(case, "רק ריקות", "תת ריקה"))
    os.makedirs(os.path.join(case, "רק ריקות", "עוד אחת", "עמוקה"))

    # 9. Skip-list folder (mixed case) and a hidden folder, both containing duplicates.
    case = os.path.join(root, "09_skipped")
    package = {"index.js": ("pkg-1", 600 * KB), "lib.js": ("pkg-2", 610 * KB)}
    for name in ("חבילה", "חבילה - עותק"):
        _write_files(os.path.join(case, "Node_Modules", name), package)
    hidden = os.path.join(case, "מוסתרת")
    _write_files(hidden, documents)  # Same content as case 1
    _set_hidden(hidden)

    # 10. A copy with the same 3 files plus 4 new ones: Jaccard 3/7 ≈ 0.43, containment 1.0.
    case = os.path.join(root, "10_containment")
    slides = {
        "מצגת ראשית.pptx": ("slides-1", 400 * KB),
        "תמונות רקע.zip": ("slides-2", 410 * KB),
        "הערות דובר.docx": ("slides-3", 420 * KB),
    }
    _write_files(os.path.join(case, "מצגת"), slides)
    _write_files(os.path.join(case, "מצגת - מורחבת"), slides)
    _write_files(
        os.path.join(case, "מצגת - מורחבת"),
        {f"שקף נוסף {n}.pptx": (f"slides-extra-{n}", (100 + n) * KB) for n in range(1, 5)},
        NEW_TIME,
    )

    # 11. The same library folder inside two different projects (each project needs its copy).
    case = os.path.join(root, "11_projects")
    library = {
        "css/all.min.css": ("fa-1", 100 * KB),
        "webfonts/fa-solid.woff2": ("fa-2", 450 * KB),
        "webfonts/fa-regular.woff2": ("fa-3", 460 * KB),
        "webfonts/fa-brands.woff2": ("fa-4", 470 * KB),
    }
    for project, seed in (("אתר חנות", "shop"), ("אתר בלוג", "blog")):
        _write_files(os.path.join(case, project), {
            "index.html": (f"{seed}-index", 20 * KB + len(seed)),
            "about.html": (f"{seed}-about", 15 * KB + len(seed)),
            "app.js": (f"{seed}-app", 60 * KB + len(seed)),
            "logo.png": (f"{seed}-logo", 80 * KB + len(seed)),
            "style.css": (f"{seed}-style", 30 * KB + len(seed)),
        })
        _write_files(os.path.join(case, project, "fontawesome"), library)

    # 12. Two identical copies of a library inside the same project (one can be deleted).
    case = os.path.join(root, "12_same_project", "אתר משרד")
    _write_files(case, {"index.html": ("office-index", 22 * KB), "app.js": ("office-app", 70 * KB)})
    bootstrap = {"bootstrap.min.css": ("bs-1", 550 * KB), "bootstrap.min.js": ("bs-2", 560 * KB)}
    for name in ("bootstrap", "bootstrap - Copy"):
        _write_files(os.path.join(case, "lib", name), bootstrap)

    return root


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix="duplicate_finder_")
    print(create_fixtures(target))
