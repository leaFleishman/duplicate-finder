"""Build the duplicate-judge evaluation dataset (synthetic groups in the LLM input format).

Usage (from the project folder):
    python eval/make_dataset.py            Write eval/dataset.json and print a review table
    python eval/make_dataset.py --upload   Also create the dataset in LangSmith

Each example has:
    inputs:  {"group": <group dict exactly as the scanner builds it>}
    outputs: {"expected_verdict": ..., "acceptable_keeps": [[...], ...]}
             keep_correct is true if the model's keep (as a set) equals one of the acceptable keeps.
    metadata: {"description": ..., "category": ...}
"""

import argparse
import json
import os
import sys

DATASET_NAME = "duplicate-judge"
DATASET_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset.json")

HOME = r"C:\Users\user"
DOCS = HOME + r"\Documents"
DOWNLOADS = HOME + r"\Downloads"
DESKTOP = HOME + r"\Desktop"
PICTURES = HOME + r"\Pictures"
PROJECTS = HOME + r"\Projects"
TEMP = HOME + r"\AppData\Local\Temp\extract_8812"

OLD = "2024-01-15 10:00"
MID = "2024-09-02 14:30"
NEW = "2025-06-01 12:00"
MB = 1024 * 1024
KB = 1024


def member(path, size_mb, files, modified, unique=(), project=None):
    """unique: [(relative_path, size_kb), ...]; project: enclosing project folder, as the scanner reports it."""
    unique_files = [{"relative_path": rel, "size_bytes": int(kb * KB)} for rel, kb in unique]
    return {
        "path": path,
        "name": path.rsplit("\\", 1)[-1],
        "size_bytes": int(size_mb * MB),
        "file_count": files,
        "latest_modified": modified,
        "unique_files": unique_files,
        "unique_count": len(unique_files),
        "project": project,
    }


def group(jaccard, containment, *members):
    for index, m in enumerate(members):
        m["id"] = chr(ord("A") + index)
    return {"id": "g1", "type": "duplicate", "jaccard": jaccard, "containment": containment,
            "members": [{"id": m.pop("id"), **m} for m in members]}


def identical(*paths, size_mb=12.4, files=37, modified=MID):
    return group(1.0, 1.0, *(member(p, size_mb, files, modified) for p in paths))


EXAMPLES = []


def add(category, description, grp, verdict, keep, also_acceptable=()):
    """keep: the preferred answer; also_acceptable: other keep lists that are also correct."""
    EXAMPLES.append({
        "inputs": {"group": grp},
        "outputs": {"expected_verdict": verdict, "acceptable_keeps": [keep, *also_acceptable]},
        "metadata": {"description": description, "category": category},
    })


# ------------------------------------------------------------- identical

add("fixture-01", "עותק זהה עם הסיומת '- עותק' באותה תיקייה",
    identical(DOCS + r"\מסמכים", DOCS + r"\מסמכים - עותק"), "identical", ["A"])
add("fixture-02", "שלושה עותקים זהים: פרויקט, '- עותק', '(2)'",
    identical(DOCS + r"\פרויקט", DOCS + r"\פרויקט (2)", DOCS + r"\פרויקט - עותק"), "identical", ["A"])
add("fixture-05", "תוכן זהה, שמות שונים לגמרי: 'חומרי קורס' מול 'random_stuff_17'",
    identical(DOCS + r"\random_stuff_17", DOCS + r"\חומרי קורס"), "identical", ["B"])
add("names", "תוכן זהה, שמות שונים: 'New folder (3)' מול 'מתכונים'",
    identical(DOCS + r"\New folder (3)", DOCS + r"\מתכונים", size_mb=3.1, files=22), "identical", ["B"])
add("names", "'Copy of Photos 2023' מול 'Photos 2023' (באנגלית)",
    identical(PICTURES + r"\Copy of Photos 2023", PICTURES + r"\Photos 2023", size_mb=840, files=612),
    "identical", ["B"])
add("names", "'Report final' מול 'Report final - Copy'",
    identical(DOCS + r"\Report final", DOCS + r"\Report final - Copy", size_mb=4.2, files=9), "identical", ["A"])
add("names", "סיומת _old: 'קוד_old' מול 'קוד', תוכן זהה",
    identical(PROJECTS + r"\קוד", PROJECTS + r"\קוד_old", size_mb=18, files=240), "identical", ["A"])
add("names", "המילה 'ישן': 'תזה ישן' מול 'תזה', תוכן זהה",
    identical(DOCS + r"\תזה", DOCS + r"\תזה ישן", size_mb=6.5, files=14), "identical", ["A"])
add("names", "'Screenshots' מול 'Screenshots (2)' באותה תיקייה",
    identical(PICTURES + r"\Screenshots", PICTURES + r"\Screenshots (2)", size_mb=95, files=180),
    "identical", ["A"])
add("location", "Documents מול Downloads, אותו שם",
    identical(DOCS + r"\חוזים", DOWNLOADS + r"\חוזים", size_mb=2.3, files=11), "identical", ["A"])
add("location", "Desktop מול Documents\\לימודים, אותו שם",
    identical(DESKTOP + r"\סיכומים", DOCS + r"\לימודים\סיכומים", size_mb=7.7, files=31), "identical", ["B"])
add("location", "Projects מול Desktop עם '- Copy'",
    identical(DESKTOP + r"\app - Copy", PROJECTS + r"\app", size_mb=55, files=410), "identical", ["B"])
add("location", "שלושה עותקים: Documents, Downloads, ו-Desktop עם '- עותק'",
    identical(DESKTOP + r"\מתכונים - עותק", DOCS + r"\מתכונים", DOWNLOADS + r"\מתכונים", size_mb=3.0, files=20),
    "identical", ["B"], also_acceptable=[["C"]])  # Downloads holds the original name
add("location", "'גיבוי טלפון 2022' ב-Documents מול 'טלפון 2022' ב-Pictures",
    identical(DOCS + r"\גיבוי טלפון 2022", PICTURES + r"\טלפון 2022", size_mb=2100, files=1850),
    "identical", ["B"])
add("location-conflict", "שם של עותק במקום מסודר מול שם מקורי ב-Downloads: המיקום גובר",
    identical(DOCS + r"\חשבוניות - עותק", DOWNLOADS + r"\חשבוניות", size_mb=1.8, files=24), "identical", ["A"])
add("same-location", "שתי התיקיות ב-Downloads: 'תמונות טיול' מול 'תמונות טיול (1)'",
    identical(DOWNLOADS + r"\תמונות טיול", DOWNLOADS + r"\תמונות טיול (1)", size_mb=320, files=150),
    "identical", ["A"])
add("same-location", "שתי התיקיות בתיקייה זמנית (Temp): 'אתר' מול 'אתר גיבוי'",
    identical(TEMP + r"\אתר", TEMP + r"\אתר גיבוי", size_mb=1.5, files=5), "identical", ["A"])
add("same-location", "שתי התיקיות ב-Temp, עמוק: 'אתר\\תמונות' מול 'אתר גיבוי\\תמונות'",
    identical(TEMP + r"\אתר גיבוי\תמונות", TEMP + r"\אתר\תמונות", size_mb=1.2, files=2), "identical", ["B"])
add("same-location", "שתי התיקיות על שולחן העבודה: 'מצגת' מול 'מצגת - עותק'",
    identical(DESKTOP + r"\מצגת", DESKTOP + r"\מצגת - עותק", size_mb=25, files=8), "identical", ["A"])

add("fixture-11", "ספריית fontawesome זהה בתוך שני פרויקטים שונים: כל פרויקט צריך את העותק שלו",
    group(1.0, 1.0,
          member(PROJECTS + r"\אתר בלוג\fontawesome", 1.4, 4, MID, project=PROJECTS + r"\אתר בלוג"),
          member(PROJECTS + r"\אתר חנות\fontawesome", 1.4, 4, MID, project=PROJECTS + r"\אתר חנות")),
    "identical", ["A", "B"])
add("projects", "תיקיית css זהה בשני פרויקטים של לקוחות שונים",
    group(1.0, 1.0,
          member(DOCS + r"\עבודות\לקוח כהן - אתר\css", 2.6, 14, OLD, project=DOCS + r"\עבודות\לקוח כהן - אתר"),
          member(DOCS + r"\עבודות\לקוח לוי - אתר\css", 2.6, 14, OLD, project=DOCS + r"\עבודות\לקוח לוי - אתר")),
    "identical", ["A", "B"])
add("projects-control", "ביקורת: שני עותקים של ספרייה באותו פרויקט ('- Copy'): אפשר למחוק אחד",
    group(1.0, 1.0,
          member(PROJECTS + r"\shop\lib\bootstrap", 3.2, 30, MID, project=PROJECTS + r"\shop"),
          member(PROJECTS + r"\shop\lib\bootstrap - Copy", 3.2, 30, MID, project=PROJECTS + r"\shop")),
    "identical", ["A"])

# ----------------------------------------------------- copy_with_changes

add("fixture-03", "העותק מכיל קובץ חדש ועדכני: העותק הוא הגרסה העדכנית",
    group(0.8, 1.0,
          member(DOCS + r"\תזה", 1.2, 4, OLD),
          member(DOCS + r"\תזה - עותק", 1.54, 5, NEW, [("פרק 5 - חדש.docx", 342)])),
    "copy_with_changes", ["B"])
add("fixture-10", "עותק מורחב: אותם 3 קבצים ועוד 4 חדשים (Jaccard 0.43, containment 1.0)",
    group(0.429, 1.0,
          member(DOCS + r"\מצגת", 1.18, 3, OLD),
          member(DOCS + r"\מצגת - מורחבת", 1.59, 7, NEW,
                 [(f"שקף נוסף {n}.pptx", 100 + n) for n in range(1, 5)])),
    "copy_with_changes", ["B"])
add("newer-copy", "שם '_old' אבל דווקא הוא מכיל את הקבצים העדכניים",
    group(0.85, 1.0,
          member(DOCS + r"\דוח שנתי", 3.0, 17, OLD),
          member(DOCS + r"\דוח שנתי_old", 3.4, 20, NEW,
                 [("טיוטה 4.docx", 210), ("טיוטה 5.docx", 220), ("נתונים מעודכנים.xlsx", 80)])),
    "copy_with_changes", ["B"])
add("newer-copy", "העותק ב-Downloads מכיל קבצים חדשים; המקור ב-Documents ישן: התוכן גובר על המיקום",
    group(0.75, 1.0,
          member(DOCS + r"\תקציב משפחתי", 0.9, 12, OLD),
          member(DOWNLOADS + r"\תקציב משפחתי", 1.3, 16, NEW,
                 [("תקציב 2025.xlsx", 95), ("קבלות ינואר.pdf", 180), ("קבלות פברואר.pdf", 150), ("הערות.txt", 2)])),
    "copy_with_changes", ["B"])
add("newer-copy", "'Copy of thesis' עם פרק נוסף ועדכני",
    group(0.88, 1.0,
          member(DOCS + r"\thesis", 8.1, 22, MID),
          member(DOCS + r"\Copy of thesis", 8.9, 25, NEW,
                 [("chapter6.docx", 410), ("figures\\fig12.png", 300), ("refs_updated.bib", 40)])),
    "copy_with_changes", ["B"])
add("older-copy", "המקור מכיל קבצים חדשים; 'גיבוי ישן' הוא תמונת מצב ישנה",
    group(0.8, 1.0,
          member(PROJECTS + r"\אתר העסק", 40, 120, NEW, [("blog\\post-31.html", 18), ("img\\banner2.jpg", 640)]),
          member(PROJECTS + r"\אתר העסק - גיבוי ישן", 39, 118, OLD)),
    "copy_with_changes", ["A"])
add("older-copy", "תיקיית backup ישנה; התיקייה הראשית עדכנית יותר",
    group(0.9, 1.0,
          member(DOCS + r"\recipes", 2.2, 30, NEW, [("pasta.docx", 30), ("bread.docx", 25), ("cake.docx", 28)]),
          member(DOCS + r"\recipes backup", 2.1, 27, OLD)),
    "copy_with_changes", ["A"])
add("both-unique", "לשני הצדדים קבצים ייחודיים: שומרים את שניהם",
    group(0.78, 0.86,
          member(PICTURES + r"\תמונות 2024", 1200, 700, MID, [(f"IMG_{n}.jpg", 3500) for n in range(4101, 4113)]),
          member(PICTURES + r"\תמונות 2024 - עותק", 1180, 694, NEW, [(f"IMG_{n}.jpg", 3400) for n in range(5201, 5207)])),
    "copy_with_changes", ["A", "B"])
add("both-unique", "לשני הצדדים קבצים ייחודיים, אחד מהם ב-Downloads: שומרים את שניהם",
    group(0.8, 0.9,
          member(DOCS + r"\עבודה סמינריונית", 5.0, 18, MID, [("ראיון 3.docx", 60), ("ראיון 4.docx", 64)]),
          member(DOWNLOADS + r"\עבודה סמינריונית", 4.8, 17, NEW, [("הערות המנחה.pdf", 120)])),
    "copy_with_changes", ["A", "B"])
add("containment", "חלק קטן של אלבום (20 קבצים) נמצא במלואו בתוך האלבום המלא (60 קבצים)",
    group(0.333, 1.0,
          member(DOWNLOADS + r"\photos_part", 60, 20, MID),
          member(PICTURES + r"\Trip 2023", 180, 60, MID,
                 [(f"DSC_{n}.jpg", 3000) for n in range(1, 41)])),
    "copy_with_changes", ["B"])

# -------------------------------------------------------------- different

add("fixture-04", "שמות דומים, תוכן שונה: 'דוח 2024' מול 'דוח 2025' (רק תבניות משותפות)",
    group(0.667, 0.8,
          member(DOCS + r"\דוח 2024", 1.24, 5, OLD, [("דוח 2024.docx", 586)]),
          member(DOCS + r"\דוח 2025", 1.25, 5, NEW, [("דוח 2025.docx", 596)])),
    "different", [])
add("similar-names", "'Invoices 2023' מול 'Invoices 2024' עם לוגו ותבנית משותפים",
    group(0.5, 0.75,
          member(DOCS + r"\Invoices 2023", 2.0, 8, OLD, [("inv-2023-01.pdf", 300), ("inv-2023-02.pdf", 310)]),
          member(DOCS + r"\Invoices 2024", 2.1, 8, MID, [("inv-2024-01.pdf", 320), ("inv-2024-02.pdf", 330)])),
    "different", [])
add("similar-names", "'תרגיל 1' מול 'תרגיל 2' עם קובץ הנחיות משותף",
    group(0.6, 0.75,
          member(DOCS + r"\קורס\תרגיל 1", 1.1, 4, OLD, [("פתרון.py", 12)]),
          member(DOCS + r"\קורס\תרגיל 2", 1.3, 4, MID, [("פתרון.py", 15)])),
    "different", [])
add("similar-names", "'לקוח - כהן' מול 'לקוח - לוי' עם תבנית חוזה משותפת",
    group(0.6, 0.75,
          member(DOCS + r"\לקוחות\לקוח - כהן", 1.6, 4, MID, [("חוזה חתום.pdf", 400)]),
          member(DOCS + r"\לקוחות\לקוח - לוי", 1.7, 4, NEW, [("חוזה חתום.pdf", 420)])),
    "different", [])
add("similar-names", "'סמסטר א' מול 'סמסטר ב' עם סילבוס ותקנון משותפים",
    group(0.5, 0.75,
          member(DOCS + r"\תואר\סמסטר א", 30, 40, OLD, [("מבחן לינארית.pdf", 900), ("סיכום חדו\"א.docx", 400)]),
          member(DOCS + r"\תואר\סמסטר ב", 34, 44, MID, [("מבחן הסתברות.pdf", 950), ("סיכום אלגו.docx", 420)])),
    "different", [])
add("similar-names", "'מצגת כנס 2023' מול 'מצגת כנס 2024' עם תמונות רקע משותפות",
    group(0.55, 0.8,
          member(DOCS + r"\כנסים\מצגת כנס 2023", 22, 9, OLD, [("מצגת.pptx", 9000)]),
          member(DOCS + r"\כנסים\מצגת כנס 2024", 25, 10, MID, [("מצגת.pptx", 10500), ("וידאו.mp4", 3000)])),
    "different", [])
add("same-name", "אותו שם 'תמונות' אצל שני ילדים, חלק מתמונות המשפחה משותפות",
    group(0.5, 0.75,
          member(PICTURES + r"\ילדים\נועה\תמונות", 400, 120, MID, [(f"נועה_{n}.jpg", 3000) for n in range(1, 31)]),
          member(PICTURES + r"\ילדים\יואב\תמונות", 380, 120, MID, [(f"יואב_{n}.jpg", 3000) for n in range(1, 31)])),
    "different", [])
add("similar-names", "'Final exam 2023' מול 'Final exam 2024' עם דף נוסחאות משותף",
    group(0.55, 0.75,
          member(DOCS + r"\Teaching\Final exam 2023", 1.6, 6, OLD, [("exam.docx", 220), ("solutions.docx", 180)]),
          member(DOCS + r"\Teaching\Final exam 2024", 1.7, 6, MID, [("exam.docx", 230), ("solutions.docx", 190)])),
    "different", [])
add("collections", "'דפי עבודה כיתה ד' מול 'דפי עבודה כיתה ה': רוב הדפים משותפים, אבל אלה שני אוספים נפרדים",
    group(0.6, 0.8,
          member(DOCS + r"\הוראה\דפי עבודה כיתה ד", 12, 40, MID,
                 [(f"דף עבודה ד - {n}.pdf", 300) for n in range(1, 9)]),
          member(DOCS + r"\הוראה\דפי עבודה כיתה ה", 13, 42, NEW,
                 [(f"דף עבודה ה - {n}.pdf", 310) for n in range(1, 11)])),
    "different", [])
add("collections", "'מתכונים של סבתא' מול 'מתכונים של אמא': כמה מתכונים משפחתיים משותפים",
    group(0.45, 0.72,
          member(DOCS + r"\מתכונים של סבתא", 2.0, 25, OLD, [(f"סבתא - מתכון {n}.docx", 45) for n in range(1, 8)]),
          member(DOCS + r"\מתכונים של אמא", 2.4, 30, NEW, [(f"אמא - מתכון {n}.docx", 48) for n in range(1, 13)])),
    "different", [])
add("template", "'הזמנה - חתונת דנה ויוסי' מול 'הזמנה - בר מצווה של איתי': גופנים ותבנית משותפים",
    group(0.5, 0.75,
          member(DOCS + r"\עיצובים\הזמנה - חתונת דנה ויוסי", 48, 8, MID, [("הזמנה סופית.psd", 21000), ("הזמנה.pdf", 900)]),
          member(DOCS + r"\עיצובים\הזמנה - בר מצווה של איתי", 52, 8, NEW, [("הזמנה סופית.psd", 23000), ("הזמנה.pdf", 950)])),
    "different", [])
add("two-projects", "שני פרויקטים שונים ב-Projects עם מעט קבצי הגדרות משותפים (eslint, prettier, LICENSE)",
    group(0.25, 0.7,
          member(PROJECTS + r"\weather-app", 1.3, 10, MID, [("src\\weather.js", 40), ("src\\api.js", 22), ("index.html", 6)]),
          member(PROJECTS + r"\todo-app", 2.8, 31, NEW,
                 [(f"src\\components\\Todo{n}.jsx", 12) for n in range(1, 21)] + [("index.html", 5)])),
    "different", [])
add("two-projects", "שני פתרונות Visual Studio שונים עם אותה ספריית NuGet בתיקיית packages",
    group(0.3, 0.75,
          member(PROJECTS + r"\Calculator", 6.0, 16, OLD, [("Calculator.sln", 2), ("Calculator\\Program.cs", 9),
                                                          ("Calculator\\Calculator.csproj", 3), ("Calculator\\Ops.cs", 14)]),
          member(PROJECTS + r"\InventoryApp", 14.0, 40, NEW,
                 [("InventoryApp.sln", 2), ("InventoryApp\\InventoryApp.csproj", 4)]
                 + [(f"InventoryApp\\Models\\Item{n}.cs", 6) for n in range(1, 27)])),
    "different", [])
add("same-name", "אותו שם בדיוק ('מסמכים') בשני מקומות שונים, תוכן שונה לגמרי (חפיפה 10%)",
    group(0.1, 0.12,
          member(DOCS + r"\עבודה\מסמכים", 9.0, 48, MID,
                 [(f"פרוטוקול ישיבה {n}.docx", 60) for n in range(1, 31)]),
          member(DESKTOP + r"\ארכיון משפחתי\מסמכים", 11.0, 52, OLD,
                 [(f"תעודה סרוקה {n}.pdf", 180) for n in range(1, 31)])),
    "different", [])

for number, example in enumerate(EXAMPLES, start=1):
    example["metadata"]["number"] = number


# ------------------------------------------------------------------- Output


def save(path: str = DATASET_FILE) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(EXAMPLES, handle, ensure_ascii=False, indent=2)


def format_keeps(acceptable_keeps: list[list[str]]) -> str:
    return " / ".join(",".join(keep) or "-" for keep in acceptable_keeps)


def print_table() -> None:
    print(f"{'#':>3}  {'verdict':<18} {'acceptable_keeps':<17} description")
    for number, example in enumerate(EXAMPLES, start=1):
        outputs, metadata = example["outputs"], example["metadata"]
        print(f"{number:>3}  {outputs['expected_verdict']:<18} {format_keeps(outputs['acceptable_keeps']):<17} "
              f"{metadata['description']}")


def upload() -> None:
    from dotenv import load_dotenv
    from langsmith import Client

    load_dotenv()
    client = Client()
    if client.has_dataset(dataset_name=DATASET_NAME):
        sys.exit(f"Dataset '{DATASET_NAME}' already exists in LangSmith; delete it first to re-upload.")
    dataset = client.create_dataset(DATASET_NAME, description="Duplicate folder judge: synthetic groups")
    client.create_examples(
        dataset_id=dataset.id,
        inputs=[e["inputs"] for e in EXAMPLES],
        outputs=[e["outputs"] for e in EXAMPLES],
        metadata=[e["metadata"] for e in EXAMPLES],
    )
    print(f"Uploaded {len(EXAMPLES)} examples to LangSmith dataset '{DATASET_NAME}'.")


def validate() -> None:
    for example in EXAMPLES:
        grp, outputs = example["inputs"]["group"], example["outputs"]
        ids = {m["id"] for m in grp["members"]}
        description = example["metadata"]["description"]
        assert outputs["acceptable_keeps"], description
        for keep in outputs["acceptable_keeps"]:
            assert set(keep) <= ids, description
            assert (outputs["expected_verdict"] == "different") == (not keep), description
    assert len(EXAMPLES) >= 35
    assert sum(e["outputs"]["expected_verdict"] == "different" for e in EXAMPLES) >= 12


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--upload", action="store_true", help="Create the dataset in LangSmith")
    args = parser.parse_args()
    validate()
    save()
    print(f"Saved {len(EXAMPLES)} examples to {DATASET_FILE}\n")
    print_table()
    if args.upload:
        upload()
