#!/usr/bin/env python3
from __future__ import annotations
"""
Organize the messy Winter 2026 documents folder into a clean Obsidian vault.

This script:
1. Scans ~/Documents/Winter 2026/ for study-relevant files
2. Filters out build artifacts, node_modules, CMake dirs, etc.
3. Maps files to course + category (Lecture/Assignment/Lab/Exam)
4. Copies files with clean, consistent names
5. Deduplicates identical files (hash-based)
6. Generates _course-index.md files with links to all materials

IMPORTANT: This COPIES files — originals are never modified or deleted.
"""

import hashlib
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

# ============================================================
# Configuration
# ============================================================

SOURCE_DIR = Path.home() / "Documents" / "Winter 2026"
VAULT_DIR = Path.home() / "Documents" / "StudyVault"

# Course name mapping: folder name → clean name
COURSE_NAMES = {
    "CMPE 223": "CMPE 223 - Software Specification",
    "ELEC 472": "ELEC 472 - Artificial Intelligence",
    "ELEC 477": "ELEC 477 - Distributed Systems",
}

# Directories/patterns to skip entirely
SKIP_PATTERNS = {
    "node_modules",
    "build",
    "CMakeFiles",
    "cyclonedds",
    "cyclonedds-cxx",
    ".git",
    ".claude",
    "__MACOSX",
    ".next",
    ".DS_Store",
    "CAPSTONE",  # Capstone is a code project, not study notes
}

# File extensions to include
INCLUDE_EXTENSIONS = {
    ".pdf", ".txt", ".md", ".docx", ".html", ".csv", ".pub",
}

# Extensions to skip (build artifacts, code files, etc.)
SKIP_EXTENSIONS = {
    ".o", ".bin", ".out", ".a", ".so", ".dylib",
    ".cmake", ".make", ".log", ".sh", ".py", ".cpp", ".c", ".h",
    ".idl", ".rst", ".zip", ".gitkeep", ".json",
    ".png", ".jpg", ".jpeg", ".gif", ".svg",
}


# ============================================================
# Data types
# ============================================================

@dataclass
class FileEntry:
    """Represents a file to be organized."""
    source_path: Path
    course: str
    category: str  # Lectures, Assignments, Labs, Exams, Resources
    clean_name: str
    file_hash: str = ""


# ============================================================
# Helpers
# ============================================================

def file_hash(path: Path) -> str:
    """Compute SHA-256 hash of a file for deduplication."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def should_skip_path(path: Path) -> bool:
    """Check if any component of the path matches a skip pattern."""
    parts = set(path.parts)
    return bool(parts & SKIP_PATTERNS)


def detect_course(path: Path) -> str | None:
    """Detect which course a file belongs to based on its path."""
    path_str = str(path)
    for folder_name in COURSE_NAMES:
        if folder_name in path_str:
            return folder_name
    return None


def detect_category(path: Path, filename: str) -> str:
    """Detect the category (Lectures, Assignments, Labs, Exams, Resources)."""
    path_str = str(path).lower()
    filename_lower = filename.lower()

    # Check path components for category hints
    if "final exam" in path_str or "final_exam" in path_str:
        # Files in Final Exam folder need sub-classification
        return classify_final_exam_file(filename)
    elif "printable exam" in path_str:
        return "Exams"
    elif "quiz" in path_str:
        return "Exams"
    elif "midterm" in path_str:
        return "Exams"
    elif "lab" in path_str:
        return "Labs"
    elif "assignment" in path_str or path_str.endswith("/a1") or path_str.endswith("/a3"):
        return "Assignments"
    elif "textbook" in path_str or "notes" in path_str:
        return "Resources"
    elif "cisc 223" in path_str or "cmpe223" in path_str:
        return "Resources"

    # Fallback: check filename
    if "assign" in filename_lower or filename_lower.startswith("a") and "sol" in filename_lower:
        return "Assignments"
    elif "lab" in filename_lower:
        return "Labs"
    elif "slide" in filename_lower or "lecture" in filename_lower:
        return "Lectures"
    elif "exam" in filename_lower or "midterm" in filename_lower:
        return "Exams"

    return "Resources"


def classify_final_exam_file(filename: str) -> str:
    """
    Files inside 'Final Exam' folders are a mix of lectures, assignments,
    labs, and past exams. Classify them based on filename patterns.
    """
    fn = filename.lower()

    # ELEC 477 patterns: E477W1C1 = Week 1 Class 1
    if re.match(r"e477w\d+c\d+", fn) or re.match(r"e4775c\d+", fn) or re.match(r"e4777c\d+", fn):
        return "Lectures"

    # Caption files for lectures
    if "captions" in fn or fn.endswith("_captions_english (united states).txt"):
        return "Lectures"

    # ELEC 472 patterns
    if fn.startswith("slides") or fn.startswith("ch"):
        return "Lectures"

    # Assignment files
    if "assign" in fn or re.match(r"a\d+", fn):
        return "Assignments"

    # Lab files
    if "lab" in fn:
        return "Labs"

    # Week scans (CMPE 223)
    if fn.startswith("scansweek") or fn.startswith("scan"):
        return "Lectures"

    # Module/chapter files (CMPE 223: m104.pdf, m114.pdf, etc.)
    if re.match(r"m\d+\.pdf", fn):
        return "Lectures"

    # Formula sheets
    if "formula" in fn:
        return "Resources"

    # Crowdmark (past exams)
    if "crowdmark" in fn:
        return "Exams"

    # Past exam papers (ELEC477.pdf, ELEC472.pdf, CISC_CMPE223.pdf, etc.)
    if re.match(r"(elec|cisc|cmpe)\d*", fn) or "cisc_cmpe" in fn:
        return "Exams"

    # Practice/sample exams
    if "practice" in fn or "sample" in fn:
        return "Exams"

    # Cheat sheets
    if "cheatsheet" in fn or "cheat_sheet" in fn:
        return "Resources"

    # Text files that look like course outlines
    if fn.endswith(".pdf") and re.match(r"\d+\.", fn):
        return "Lectures"

    return "Resources"


def clean_filename_elec477(filename: str) -> str:
    """Clean ELEC 477 lecture filenames: E477W1C1 → Week 01 - Lecture 1"""
    # Match patterns like E477W1C1, E477W10C2, E4775C1(4C2)
    m = re.match(r"E477W(\d+)[Cc](\d+)", filename, re.IGNORECASE)
    if m:
        week = int(m.group(1))
        lecture = int(m.group(2))
        return f"Week {week:02d} - Lecture {lecture}"

    # Caption files
    m = re.match(r"E477W(\d+)[Cc](\d+)_Captions", filename, re.IGNORECASE)
    if m:
        week = int(m.group(1))
        lecture = int(m.group(2))
        return f"Week {week:02d} - Lecture {lecture} - Captions"

    return None


def clean_filename_elec472(filename: str) -> str:
    """Clean ELEC 472 filenames."""
    fn = filename.lower()

    # Slides - ChX - Title
    m = re.match(r"slides\s*-\s*(ch\d+)\s*-\s*(.+?)(?:\s*-\s*annotated)?\.pdf", fn, re.IGNORECASE)
    if m:
        chapter = m.group(1).upper().replace("CH", "Ch")
        title = m.group(2).strip().title()
        return f"{chapter} - {title}"

    return None


def clean_filename_cmpe223(filename: str) -> str:
    """Clean CMPE 223 filenames."""
    fn = filename.lower()

    # Weekly scans: scansWeek1W26 → Week 01 - Lecture Notes
    m = re.match(r"scansweek(\d+)w\d+(?:part(\d+))?", fn, re.IGNORECASE)
    if m:
        week = int(m.group(1))
        part = m.group(2)
        suffix = f" Part {part}" if part else ""
        return f"Week {week:02d} - Lecture Notes{suffix}"

    return None


def clean_filename(entry: FileEntry) -> str:
    """Generate a clean filename for the destination."""
    filename = entry.source_path.name
    stem = entry.source_path.stem
    ext = entry.source_path.suffix

    # Try course-specific cleaners
    if entry.course == "ELEC 477":
        cleaned = clean_filename_elec477(stem)
        if cleaned:
            return cleaned + ext

        # Caption files
        if "captions" in filename.lower():
            m = re.match(r"E477W(\d+)[Cc](\d+)", stem, re.IGNORECASE)
            if m:
                week = int(m.group(1))
                lecture = int(m.group(2))
                return f"Week {week:02d} - Lecture {lecture} - Captions.txt"

    elif entry.course == "ELEC 472":
        cleaned = clean_filename_elec472(filename)
        if cleaned:
            return cleaned + ext

    elif entry.course == "CMPE 223":
        cleaned = clean_filename_cmpe223(stem)
        if cleaned:
            return cleaned + ext

    # Generic cleaning: remove redundant course codes from filename
    # Only strip if the result is still a meaningful name
    clean = filename
    
    # For CMPE 223 exam files with CISC_CMPE prefixes, give them proper names
    cisc_cmpe_match = re.match(r"CISC_?CMPE_?223_?(.*)", stem, re.IGNORECASE)
    if cisc_cmpe_match:
        remainder = cisc_cmpe_match.group(1).strip(" _-()")
        if remainder:
            return f"Past Exam - {remainder}{ext}"
        else:
            return f"Past Exam - CISC CMPE 223{ext}"
    
    # Strip common course code prefixes, but only if result is meaningful
    prefixes_to_strip = [
        "ELEC472_", "ELEC472 ", "ELEC 472 - ",
        "ELEC477_", "ELEC 477 - ",
        "CMPE223_", "CMPE 223_", "CMPE 223 - ",
    ]
    for code in prefixes_to_strip:
        if clean.startswith(code):
            candidate = clean[len(code):]
            if candidate.strip(" _-"):  # only strip if something remains
                clean = candidate
                break

    # Clean up leading/trailing spaces and underscores
    clean = clean.strip(" _-")
    if not clean or clean == ext:
        clean = filename

    return clean


# ============================================================
# Main pipeline
# ============================================================

def scan_source_files() -> list[FileEntry]:
    """Scan the source directory and build a list of files to organize."""
    entries = []

    for root, dirs, files in os.walk(SOURCE_DIR):
        root_path = Path(root)

        # Skip unwanted directories
        dirs[:] = [d for d in dirs if d not in SKIP_PATTERNS]

        if should_skip_path(root_path):
            continue

        for filename in files:
            filepath = root_path / filename

            # Skip hidden files
            if filename.startswith("."):
                continue

            # Check extension
            ext = filepath.suffix.lower()
            if ext in SKIP_EXTENSIONS:
                continue
            if ext not in INCLUDE_EXTENSIONS:
                continue

            # Detect course
            course = detect_course(filepath)
            if not course:
                continue

            # Detect category
            category = detect_category(filepath, filename)

            entry = FileEntry(
                source_path=filepath,
                course=course,
                category=category,
                clean_name=filename,  # will be cleaned later
            )
            entries.append(entry)

    return entries


def deduplicate(entries: list[FileEntry]) -> list[FileEntry]:
    """Remove duplicate files based on content hash."""
    seen_hashes: dict[str, FileEntry] = {}
    unique = []
    duplicates = 0

    for entry in entries:
        h = file_hash(entry.source_path)
        entry.file_hash = h

        key = f"{entry.course}:{h}"
        if key not in seen_hashes:
            seen_hashes[key] = entry
            unique.append(entry)
        else:
            duplicates += 1
            existing = seen_hashes[key]
            print(f"  ⊘ Duplicate: {entry.source_path.name} (same as {existing.clean_name})")

    print(f"\n  Found {duplicates} duplicates, keeping {len(unique)} unique files\n")
    return unique


def organize(entries: list[FileEntry], dry_run: bool = False) -> dict:
    """Copy files to the vault with clean names."""
    stats = {"copied": 0, "skipped": 0, "errors": []}

    for entry in entries:
        # Clean the filename
        entry.clean_name = clean_filename(entry)

        # Build destination path
        course_dir = COURSE_NAMES.get(entry.course, entry.course)
        dest_dir = VAULT_DIR / course_dir / entry.category
        dest_path = dest_dir / entry.clean_name

        # Handle name collisions
        if dest_path.exists():
            stem = dest_path.stem
            ext = dest_path.suffix
            counter = 2
            while dest_path.exists():
                dest_path = dest_dir / f"{stem} ({counter}){ext}"
                counter += 1

        if dry_run:
            print(f"  → {entry.source_path.name}")
            print(f"    ↳ {course_dir}/{entry.category}/{entry.clean_name}")
            stats["copied"] += 1
            continue

        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(entry.source_path, dest_path)
            stats["copied"] += 1
        except Exception as e:
            stats["errors"].append(f"{entry.source_path}: {e}")
            print(f"  ✗ Error copying {entry.source_path.name}: {e}")

    return stats


def generate_course_index(course_code: str, course_name: str):
    """Generate a _course-index.md file for a course."""
    course_dir = VAULT_DIR / course_name

    if not course_dir.exists():
        return

    lines = [f"# {course_name}\n\n"]

    for category in ["Lectures", "Assignments", "Labs", "Exams", "Resources"]:
        cat_dir = course_dir / category
        if not cat_dir.exists():
            continue

        files = sorted(cat_dir.iterdir())
        if not files:
            continue

        lines.append(f"## {category}\n\n")
        for f in files:
            if f.name.startswith("_"):
                continue
            # Obsidian-style wiki link
            rel_path = f.relative_to(course_dir)
            lines.append(f"- [{f.name}]({rel_path})\n")
        lines.append("\n")

    index_path = course_dir / "_course-index.md"
    index_path.write_text("".join(lines))
    print(f"  ✓ Generated {index_path.relative_to(VAULT_DIR)}")


def generate_vault_index():
    """Generate a master _vault-index.md file."""
    lines = [
        "# Study Vault — Winter 2026\n\n",
        "Queen's University — Computer Engineering\n\n",
    ]

    for course_code, course_name in sorted(COURSE_NAMES.items()):
        course_dir = VAULT_DIR / course_name
        if not course_dir.exists():
            continue

        # Count files
        file_count = sum(1 for _ in course_dir.rglob("*") if _.is_file() and not _.name.startswith("_"))
        lines.append(f"## [{course_name}]({course_name}/_course-index.md)\n")
        lines.append(f"*{file_count} files*\n\n")

    index_path = VAULT_DIR / "_vault-index.md"
    index_path.write_text("".join(lines))
    print(f"  ✓ Generated _vault-index.md")


def main():
    import sys

    dry_run = "--dry-run" in sys.argv
    mode = "DRY RUN" if dry_run else "COPY"

    print(f"\n{'='*60}")
    print(f"  Study Vault Organizer ({mode})")
    print(f"{'='*60}")
    print(f"\n  Source: {SOURCE_DIR}")
    print(f"  Vault:  {VAULT_DIR}\n")

    if not SOURCE_DIR.exists():
        print(f"  ✗ Source directory not found: {SOURCE_DIR}")
        return

    # Step 1: Scan
    print("─── Scanning files ───")
    entries = scan_source_files()
    print(f"  Found {len(entries)} study-relevant files\n")

    # Show breakdown by course
    for course in COURSE_NAMES:
        course_files = [e for e in entries if e.course == course]
        categories = {}
        for e in course_files:
            categories[e.category] = categories.get(e.category, 0) + 1
        cat_str = ", ".join(f"{k}: {v}" for k, v in sorted(categories.items()))
        print(f"  {course}: {len(course_files)} files ({cat_str})")

    print()

    # Step 2: Deduplicate
    print("─── Deduplicating ───")
    entries = deduplicate(entries)

    # Step 3: Organize (copy)
    print(f"─── Organizing ({mode}) ───\n")
    stats = organize(entries, dry_run=dry_run)

    print(f"\n  ✓ {stats['copied']} files {'would be copied' if dry_run else 'copied'}")
    if stats["errors"]:
        print(f"  ✗ {len(stats['errors'])} errors")
        for err in stats["errors"]:
            print(f"    - {err}")

    # Step 4: Generate indexes (skip in dry run)
    if not dry_run:
        print("\n─── Generating indexes ───\n")
        for course_code, course_name in COURSE_NAMES.items():
            generate_course_index(course_code, course_name)
        generate_vault_index()

    print(f"\n{'='*60}")
    if dry_run:
        print("  This was a dry run. No files were copied.")
        print("  Run without --dry-run to actually organize.\n")
    else:
        print(f"  Vault created at: {VAULT_DIR}")
        print("  Open this folder in Obsidian to browse your notes.\n")


if __name__ == "__main__":
    main()
