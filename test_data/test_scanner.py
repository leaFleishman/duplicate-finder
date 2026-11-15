"""Run the scanner on the fixture tree and verify every known case."""

import os

import pytest

import config
import scanner
from make_fixtures import create_fixtures


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    return create_fixtures(str(tmp_path_factory.mktemp("data")))


@pytest.fixture(scope="module")
def scan_result(root):
    return scanner.scan([root])


@pytest.fixture(scope="module")
def candidates(scan_result):
    return scanner.find_candidates(
        scan_result, config.DEFAULT_OVERLAP_THRESHOLD, config.DEFAULT_MIN_FOLDER_SIZE_MB
    )


def rel(root, path):
    return os.path.relpath(path, root)


def members(root, group):
    return frozenset(rel(root, m["path"]) for m in group["members"])


def top_groups(root, candidates):
    return {members(root, g): g for g in candidates["queue"] if g["type"] == "duplicate"}


def all_hidden_groups(candidates):
    return [g for children in candidates["hidden_children"].values() for g in children]


def zip_items(candidates):
    return {os.path.basename(i["zip_path"]): i for i in candidates["queue"] if i["type"] == "zip"}


def by_name(group, name):
    return next(m for m in group["members"] if m["name"] == name)


COPY_SUFFIX = frozenset({r"01_copy_suffix\מסמכים", r"01_copy_suffix\מסמכים - עותק"})
THREE_COPIES = frozenset({r"02_three_copies\פרויקט", r"02_three_copies\פרויקט - עותק", r"02_three_copies\פרויקט (2)"})
NEWER_COPY = frozenset({r"03_newer_copy\תזה", r"03_newer_copy\תזה - עותק"})
SIMILAR_NAMES = frozenset({r"04_similar_names\דוח 2024", r"04_similar_names\דוח 2025"})
DIFFERENT_NAMES = frozenset({r"05_different_names\חומרי קורס", r"05_different_names\random_stuff_17"})
NESTED_TOP = frozenset({r"06_nested\אתר", r"06_nested\אתר גיבוי"})
NESTED_IMAGES = frozenset({r"06_nested\אתר\תמונות", r"06_nested\אתר גיבוי\תמונות"})
CONTAINMENT_ONLY = frozenset({r"10_containment\מצגת", r"10_containment\מצגת - מורחבת"})
LIBRARY_IN_PROJECTS = frozenset({r"11_projects\אתר בלוג\fontawesome", r"11_projects\אתר חנות\fontawesome"})


def test_exactly_the_expected_top_level_groups(root, candidates):
    assert set(top_groups(root, candidates)) == {
        COPY_SUFFIX, THREE_COPIES, NEWER_COPY, SIMILAR_NAMES, DIFFERENT_NAMES, NESTED_TOP, CONTAINMENT_ONLY,
        LIBRARY_IN_PROJECTS,
    }
    assert len(candidates["queue"]) == 8 + 3  # 8 duplicate groups + 3 ZIP items


def test_library_in_two_projects_reports_each_project(root, candidates):
    """Case 11: identical library folders, each inside a different project."""
    group = top_groups(root, candidates)[LIBRARY_IN_PROJECTS]
    assert group["jaccard"] == 1.0
    assert {rel(root, m["project"]) for m in group["members"]} == {
        r"11_projects\אתר בלוג", r"11_projects\אתר חנות",
    }
    # The project folders themselves are not paired (they share only the library).
    assert not any(m["name"] in ("אתר בלוג", "אתר חנות") for g in all_hidden_groups(candidates) for m in g["members"])
    # No cross pairs (library <-> other webfonts, Jaccard 0.75); only webfonts <-> webfonts is hidden.
    assert [members(root, g) for g in candidates["hidden_children"][group["id"]]] == [frozenset({
        r"11_projects\אתר בלוג\fontawesome\webfonts", r"11_projects\אתר חנות\fontawesome\webfonts",
    })]


def test_members_outside_projects_have_no_project(root, candidates):
    group = top_groups(root, candidates)[COPY_SUFFIX]
    assert all(m["project"] is None for m in group["members"])


def test_nested_images_belong_to_their_site_projects(root, candidates):
    """Case 06: the sites contain index.html, so the images folders are inside two projects."""
    images = next(g for g in all_hidden_groups(candidates) if members(root, g) == NESTED_IMAGES)
    assert {rel(root, m["project"]) for m in images["members"]} == {r"06_nested\אתר", r"06_nested\אתר גיבוי"}


def test_copy_with_many_new_files_detected_by_containment(root, candidates):
    """Case 10: Jaccard 3/7 is far below the threshold; containment 1.0 qualifies the pair."""
    group = top_groups(root, candidates)[CONTAINMENT_ONLY]
    assert group["jaccard"] == pytest.approx(3 / 7, abs=1e-3)
    assert group["jaccard"] < config.DEFAULT_OVERLAP_THRESHOLD
    assert group["containment"] == 1.0
    extended = by_name(group, "מצגת - מורחבת")
    assert extended["unique_count"] == 4
    assert by_name(group, "מצגת")["unique_count"] == 0


def test_identical_copy_with_suffix(root, candidates):
    group = top_groups(root, candidates)[COPY_SUFFIX]
    assert group["jaccard"] == 1.0 and group["containment"] == 1.0
    assert all(m["unique_count"] == 0 for m in group["members"])


def test_three_identical_copies_form_one_group(root, candidates):
    group = top_groups(root, candidates)[THREE_COPIES]
    assert len(group["members"]) == 3
    assert [m["id"] for m in group["members"]] == ["A", "B", "C"]
    assert group["jaccard"] == 1.0


def test_copy_with_newer_files(root, candidates):
    group = top_groups(root, candidates)[NEWER_COPY]
    assert group["jaccard"] == pytest.approx(0.8)
    assert group["containment"] == 1.0
    original, copy = by_name(group, "תזה"), by_name(group, "תזה - עותק")
    assert original["unique_files"] == []
    assert copy["unique_files"] == [{"relative_path": "פרק 5 - חדש.docx", "size_bytes": 350_000}]
    assert copy["latest_modified"] > original["latest_modified"]


def test_similar_names_with_different_content(root, candidates):
    group = top_groups(root, candidates)[SIMILAR_NAMES]
    assert group["jaccard"] == pytest.approx(4 / 6, abs=1e-3)  # Below the threshold...
    assert group["containment"] == pytest.approx(0.8)  # ...but containment qualifies it
    assert [u["relative_path"] for u in by_name(group, "דוח 2024")["unique_files"]] == ["דוח 2024.docx"]
    assert [u["relative_path"] for u in by_name(group, "דוח 2025")["unique_files"]] == ["דוח 2025.docx"]


def test_identical_content_with_different_names(root, candidates):
    group = top_groups(root, candidates)[DIFFERENT_NAMES]
    assert group["jaccard"] == 1.0


def test_nested_duplicates_are_hidden_under_top_group(root, candidates):
    hidden = all_hidden_groups(candidates)
    assert NESTED_IMAGES in {members(root, g) for g in hidden}
    assert NESTED_IMAGES not in top_groups(root, candidates)
    # Hidden groups exist only in the cases with nested duplicates (06 and the library in 11).
    assert all(path.startswith(("06_nested", "11_projects")) for g in hidden for path in members(root, g))
    top_id = top_groups(root, candidates)[NESTED_TOP]["id"]
    assert top_id in candidates["hidden_children"]


def test_nested_case_has_no_cross_pairs(root, candidates):
    """Case 06: only the top pair and the images pair hidden directly under it."""
    nested_top = [g for g in candidates["queue"] if g["type"] == "duplicate"
                  and members(root, g) <= NESTED_TOP | NESTED_IMAGES]
    nested_hidden = [g for g in all_hidden_groups(candidates)
                     if all(p.startswith("06_nested") for p in members(root, g))]
    assert [members(root, g) for g in nested_top] == [NESTED_TOP]
    assert [members(root, g) for g in nested_hidden] == [NESTED_IMAGES]
    assert candidates["hidden_children"][nested_top[0]["id"]] == nested_hidden


def test_newer_copy_still_detected_with_full_containment(root, candidates):
    """Case 03: the copy fully contains the original (containment 1.0)."""
    group = top_groups(root, candidates)[NEWER_COPY]
    assert group["containment"] == 1.0


def test_containment_only_pair_kept_without_better_partner(root, candidates):
    """Case 04 passes only thanks to containment and has no better partner, so it stays."""
    group = top_groups(root, candidates)[SIMILAR_NAMES]
    assert group["jaccard"] < config.DEFAULT_OVERLAP_THRESHOLD <= group["containment"]


def test_zip_items(candidates):
    items = zip_items(candidates)
    assert set(items) == {"ארכיון מלא.zip", "חלקי.zip", "כפול.zip"}  # "בודד.zip" has no folder
    assert items["ארכיון מלא.zip"]["fully_extracted"] is True
    assert items["חלקי.zip"]["fully_extracted"] is False
    assert items["חלקי.zip"]["missing_files"] == ["ג.txt"]
    assert items["כפול.zip"]["fully_extracted"] is True  # Double nesting handled
    assert os.path.basename(items["כפול.zip"]["folder_path"]) == "כפול"


def test_corrupted_zip_produces_warning(scan_result):
    assert any("פגום.zip" in warning for warning in scan_result.warnings)


def test_empty_folders_highest_only(root, candidates):
    assert [rel(root, p) for p in candidates["empty_folders"]] == [
        r"08_empty\ריקה",
        r"08_empty\רק ריקות",
    ]


def test_skip_list_and_hidden_folders_not_scanned(root, scan_result, candidates):
    skipped = os.path.join(root, "09_skipped")
    scanned_inside = [p for p in scan_result.folders if scanner.is_within(p, skipped) and p != skipped]
    assert scanned_inside == []
    all_groups = [g for g in candidates["queue"] if g["type"] == "duplicate"] + all_hidden_groups(candidates)
    assert not any(scanner.is_within(m["path"], skipped) for g in all_groups for m in g["members"])
    # Contains skipped content, so it is not reported as empty either.
    assert skipped not in candidates["empty_folders"]


def test_no_unexpected_warnings(scan_result):
    assert len(scan_result.warnings) == 1  # Only the corrupted ZIP
