#!/usr/bin/env python3
"""
build_vault.py — Clean-slate Obsidian vault builder with Gemma4-powered renaming.

This script:
  Phase 0 — LM Studio health check (fail fast if not running)
  Phase 1 — Wipe and recreate the StudyVault directory
  Phase 2 — Scan ~/Documents/Winter 2026, filter, deduplicate
  Phase 3 — Call Gemma4 to generate clean filenames; OCR fallback for scanned PDFs
  Phase 4 — Generate Obsidian _course-index.md and _vault-index.md files
  Phase 5 — Wipe ChromaDB and re-ingest the fresh vault

Run from the project root:
    uv run python scripts/build_vault.py [--dry-run]
"""
from __future__ import annotations

import base64
import hashlib
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
import openai
from dotenv import load_dotenv

# ============================================================
# Configuration
# ============================================================

load_dotenv()

LMSTUDIO_BASE_URL: str = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
LMSTUDIO_MODEL: str = os.getenv("LMSTUDIO_MODEL", "google/gemma-4-26b-a4b")
_vault_env = os.getenv("VAULT_PATH", str(Path.home() / "Documents" / "StudyVault"))
VAULT_DIR = Path(_vault_env).expanduser()

SOURCE_DIR = Path.home() / "Documents" / "Winter 2026"

# Sibling junk-bin directory (same parent as VAULT_DIR)
JUNK_BIN_DIR = VAULT_DIR.parent / "StudyVault_JunkBin"

# Course name mapping: folder name → clean display name
COURSE_NAMES = {
    "CMPE 223": "CMPE 223 - Software Specification",
    "ELEC 472": "ELEC 472 - Artificial Intelligence",
    "ELEC 477": "ELEC 477 - Distributed Systems",
}

# Directories / path components to skip entirely
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
    "CAPSTONE",
}

# File extensions to include
INCLUDE_EXTENSIONS = {
    ".pdf", ".txt", ".md", ".docx", ".html",
}

# Extensions to skip (build artifacts, code files, images, etc.)
SKIP_EXTENSIONS = {
    ".o", ".bin", ".out", ".a", ".so", ".dylib",
    ".cmake", ".make", ".log", ".sh", ".py", ".cpp", ".c", ".h",
    ".idl", ".rst", ".zip", ".gitkeep", ".json",
    ".png", ".jpg", ".jpeg", ".gif", ".svg",
}

# ANSI colour helpers
GREEN = "\033[92m"
RED = "\033[91m"
CYAN = "\033[96m"
RESET = "\033[0m"


# ============================================================
# Data types
# ============================================================

@dataclass
class FileEntry:
    """Represents a source file to be processed and copied into the vault."""
    source_path: Path
    course: str       # e.g. "ELEC 472"
    category: str     # Lectures / Assignments / Labs / Exams / Resources
    file_hash: str = ""


# ============================================================
# Phase 0 — LM Studio health check
# ============================================================

def check_lmstudio(exit_on_fail: bool = True) -> bool:
    """Verify LM Studio is reachable and has a model loaded.

    Returns True if healthy. When *exit_on_fail* is True (default),
    calls sys.exit(1) on failure; otherwise returns False.
    """
    url = f"{LMSTUDIO_BASE_URL}/models"
    try:
        resp = httpx.get(url, timeout=5.0)
        resp.raise_for_status()
        data = resp.json()
        if not data.get("data"):
            if exit_on_fail:
                print(f"{RED}ERROR: No models loaded in LM Studio{RESET}")
                sys.exit(1)
            return False
        return True
    except Exception:
        if exit_on_fail:
            print(
                f"{RED}ERROR: Cannot connect to LM Studio at {LMSTUDIO_BASE_URL}{RESET}\n"
                "Please ensure LM Studio is running with a Gemma model loaded."
            )
            sys.exit(1)
        return False


def wait_for_model(max_wait: int = 60) -> bool:
    """Wait for LM Studio to have a model loaded, polling every 3 seconds."""
    print(f"    {CYAN}Waiting for LM Studio model to reload...{RESET}", end="", flush=True)
    for _ in range(max_wait // 3):
        time.sleep(3)
        print(".", end="", flush=True)
        if check_lmstudio(exit_on_fail=False):
            print(f" {GREEN}OK{RESET}")
            return True
    print(f" {RED}TIMED OUT{RESET}")
    return False


# ============================================================
# Path classification helpers (reused from organize_vault.py)
# ============================================================

def should_skip_path(path: Path) -> bool:
    """Return True if any path component matches a skip pattern."""
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
    """Detect the category: Lectures / Assignments / Labs / Exams / Resources."""
    path_str = str(path).lower()
    filename_lower = filename.lower()

    if "final exam" in path_str or "final_exam" in path_str:
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
    if "assign" in filename_lower or (filename_lower.startswith("a") and "sol" in filename_lower):
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

    # Practice / sample exams
    if "practice" in fn or "sample" in fn:
        return "Exams"

    # Cheat sheets
    if "cheatsheet" in fn or "cheat_sheet" in fn:
        return "Resources"

    # Text files that look like course outlines
    if fn.endswith(".pdf") and re.match(r"\d+\.", fn):
        return "Lectures"

    return "Resources"


# ============================================================
# Phase 2 — scan & deduplication
# ============================================================

def file_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file for content-based deduplication."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def scan_source_files() -> list[FileEntry]:
    """Walk the source directory and return a filtered list of FileEntry objects."""
    entries: list[FileEntry] = []

    for root, dirs, files in os.walk(SOURCE_DIR):
        root_path = Path(root)

        # Prune unwanted subdirectories in-place so os.walk skips them entirely
        dirs[:] = [d for d in dirs if d not in SKIP_PATTERNS]

        if should_skip_path(root_path):
            continue

        for filename in files:
            filepath = root_path / filename

            # Skip hidden files
            if filename.startswith("."):
                continue

            ext = filepath.suffix.lower()
            if ext in SKIP_EXTENSIONS:
                continue
            if ext not in INCLUDE_EXTENSIONS:
                continue

            course = detect_course(filepath)
            if not course:
                continue

            category = detect_category(filepath, filename)
            entries.append(FileEntry(source_path=filepath, course=course, category=category))

    return entries


def deduplicate(entries: list[FileEntry]) -> list[FileEntry]:
    """Remove duplicate files (same course + identical SHA-256)."""
    seen: dict[str, FileEntry] = {}
    unique: list[FileEntry] = []
    dup_count = 0

    for entry in entries:
        h = file_sha256(entry.source_path)
        entry.file_hash = h

        key = f"{entry.course}:{h}"
        if key not in seen:
            seen[key] = entry
            unique.append(entry)
        else:
            dup_count += 1
            existing = seen[key]
            print(f"  Duplicate: {entry.source_path.name} (same as {existing.source_path.name})")

    print(f"\n  Found {dup_count} duplicates, keeping {len(unique)} unique files\n")
    return unique


# ============================================================
# Phase 3 — Gemma renaming helpers
# ============================================================

# Detect course code prefixes that Gemma echoes back (to strip, not just reject)
_COURSE_CODE_RE = re.compile(
    r"^(CMPE\s*223|ELEC\s*472|ELEC\s*477|CISC\s*223)", re.IGNORECASE
)
# Broader pattern to strip leading course codes including cross-listed (CISC/CMPE 223)
_COURSE_CODE_STRIP_RE = re.compile(
    r"^(CISC\s*/?\s*CMPE|CMPE|CISC|ELEC)\s*-?\s*\d{3}\s*[-–—:,]?\s*",
    re.IGNORECASE,
)

_SYSTEM_PROMPT = (
    "You are a file naming assistant. Given the content of a study document, "
    "generate a clean, descriptive filename. Rules:\n"
    "- Output ONLY the filename, nothing else\n"
    "- No file extension (it will be added automatically)\n"
    "- Use title case with spaces (e.g., \"Final Exam - April 2014\")\n"
    "- Include the document type if clear (Lecture, Assignment, Lab, Exam, Midterm, etc.)\n"
    "- Include dates, week numbers, or chapter numbers if present\n"
    "- Keep it concise (under 60 characters)\n"
    "- Do not include the course code (it's already in the folder path)"
)


def validate_filename(name: str, course_code: str) -> bool:
    """
    Validate a Gemma-generated filename stem.

    Strips surrounding whitespace and quotes, then applies several rejection
    rules. Returns True if the name is safe to use, False otherwise.
    """
    name = name.strip().strip("\"'")

    if not name:
        return False
    # Illegal filesystem characters
    if any(ch in name for ch in r'/\:*?"<>|'):
        return False
    # Markdown / prompt artefacts
    if "`" in name or "#" in name or "**" in name:
        return False
    # Multi-line response
    if "\n" in name or "\r" in name:
        return False
    # Too long
    if len(name) > 80:
        return False
    # Gemma just echoed the course code
    if _COURSE_CODE_RE.match(name):
        return False

    return True


def _clean_gemma_response(raw: str) -> str:
    """Extract a usable filename stem from Gemma's raw output."""
    # Remove .pdf suffix if the model appended it
    cleaned = raw.strip().removesuffix(".pdf")
    # Strip surrounding quotes and markdown formatting
    cleaned = cleaned.strip().strip("\"'*_`")
    # If there are multiple lines, take the first non-empty one
    lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
    cleaned = lines[0] if lines else cleaned
    # Strip course code prefixes that Gemma often echoes back
    cleaned = _COURSE_CODE_STRIP_RE.sub("", cleaned).strip(" -–—:,")
    # Replace illegal filesystem characters
    cleaned = re.sub(r'[/\\:*?"<>|]', " - ", cleaned)
    # Collapse multiple spaces/dashes
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def extract_text_preview(path: Path) -> str:
    """
    Extract a short text preview suitable for sending to Gemma.

    PDFs: text from the first 2 pages, capped at 2000 chars.
    Other formats: first 500 chars of file content.
    Returns an empty string on any error.
    """
    ext = path.suffix.lower()

    if ext == ".pdf":
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(str(path))
            text = ""
            for i in range(min(2, len(doc))):
                text += doc[i].get_text("text") + "\n"
            doc.close()
            return text[:2000].strip()
        except Exception as e:
            print(f"  {RED}Warning: could not open PDF {path.name}: {e}{RESET}")
            return ""

    if ext in {".md", ".txt", ".docx", ".html"}:
        try:
            if ext == ".docx":
                from docx import Document as DocxDocument
                doc = DocxDocument(str(path))
                text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
                return text[:500].strip()
            return path.read_text(encoding="utf-8", errors="replace")[:500].strip()
        except Exception:
            return ""

    return ""


def _call_gemma_text(course_name: str, category: str, text_preview: str) -> str | None:
    """
    Send a text-based naming request to Gemma via LM Studio.
    Returns the raw response string, or None on any failure.
    """
    user_content = (
        f"Course: {course_name}, Category: {category}\n"
        f"Content preview:\n{text_preview}"
    )
    try:
        client = openai.OpenAI(base_url=LMSTUDIO_BASE_URL, api_key="lm-studio")
        response = client.chat.completions.create(
            model=LMSTUDIO_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.1,
            max_tokens=60,
            extra_body={"thinking": {"type": "disabled"}},
        )
        return response.choices[0].message.content or None
    except Exception as e:
        msg = str(e)
        print(f"  {RED}Gemma text call failed: {e}{RESET}")
        if "No models loaded" in msg or "WebSocket" in msg:
            wait_for_model()
        return None


def _call_gemma_ocr(course_name: str, category: str, b64_image: str) -> str | None:
    """
    Send a vision-based naming request to Gemma via LM Studio (OCR fallback).
    Returns the raw response string, or None on any failure.
    """
    content = [
        {
            "type": "text",
            "text": (
                f"Course: {course_name}, Category: {category}\n"
                "Look at this scanned document and generate a clean filename."
            ),
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"},
        },
    ]
    try:
        client = openai.OpenAI(base_url=LMSTUDIO_BASE_URL, api_key="lm-studio")
        response = client.chat.completions.create(
            model=LMSTUDIO_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            temperature=0.1,
            max_tokens=60,
            extra_body={"thinking": {"type": "disabled"}},
        )
        return response.choices[0].message.content or None
    except Exception as e:
        msg = str(e)
        print(f"  {RED}Gemma OCR call failed: {e}{RESET}")
        if "No models loaded" in msg or "WebSocket" in msg:
            wait_for_model()
        return None


def rename_entry(entry: FileEntry, index: int, total: int) -> tuple[str, bool, bool]:
    """
    Attempt to generate a clean filename for a file using Gemma.

    Returns (clean_stem, success, used_ocr).
    On total failure, success=False and clean_stem is the original stem.
    """
    course_name = COURSE_NAMES.get(entry.course, entry.course)
    ext = entry.source_path.suffix.lower()

    # Attempt 1 — text preview (with one retry)
    text_preview = extract_text_preview(entry.source_path)
    if len(text_preview) > 50:
        for attempt in range(2):
            raw = _call_gemma_text(course_name, entry.category, text_preview)
            if raw is not None:
                candidate = _clean_gemma_response(raw)
                if validate_filename(candidate, entry.course):
                    return candidate, True, False
                else:
                    print(f"    {RED}Rejected name (attempt {attempt + 1}): {repr(candidate)}{RESET}")

    # Attempt 2 — OCR fallback (PDFs only)
    if ext == ".pdf":
        try:
            import fitz
            doc = fitz.open(str(entry.source_path))
            if len(doc) > 0:
                page = doc[0]
                pix = page.get_pixmap(dpi=100)
                # Use JPEG to reduce payload size (critical for Tailscale)
                b64_image = base64.b64encode(pix.tobytes("jpeg")).decode("utf-8")
                doc.close()

                raw = _call_gemma_ocr(course_name, entry.category, b64_image)
                if raw is not None:
                    candidate = _clean_gemma_response(raw)
                    if validate_filename(candidate, entry.course):
                        return candidate, True, True
            else:
                doc.close()
        except Exception as e:
            print(f"  {RED}Warning: OCR fallback failed for {entry.source_path.name}: {e}{RESET}")

    # Both attempts failed — fall back to original stem
    return entry.source_path.stem, False, False


# ============================================================
# Phase 4 — index generation
# ============================================================

def generate_course_index(course_code: str, course_name: str) -> None:
    """Generate a _course-index.md Obsidian index file for a single course."""
    course_dir = VAULT_DIR / course_name
    if not course_dir.exists():
        return

    lines: list[str] = [f"# {course_name}\n\n"]

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
            rel_path = f.relative_to(course_dir)
            lines.append(f"- [{f.name}]({rel_path})\n")
        lines.append("\n")

    index_path = course_dir / "_course-index.md"
    index_path.write_text("".join(lines), encoding="utf-8")
    print(f"  {GREEN}✓ Generated {index_path.relative_to(VAULT_DIR)}{RESET}")


def generate_vault_index() -> None:
    """Generate the master _vault-index.md file at the vault root."""
    lines: list[str] = [
        "# Study Vault — Winter 2026\n\n",
        "Queen's University — Computer Engineering\n\n",
    ]

    for course_code, course_name in sorted(COURSE_NAMES.items()):
        course_dir = VAULT_DIR / course_name
        if not course_dir.exists():
            continue

        file_count = sum(
            1 for f in course_dir.rglob("*")
            if f.is_file() and not f.name.startswith("_")
        )
        lines.append(f"## [{course_name}]({course_name}/_course-index.md)\n")
        lines.append(f"*{file_count} files*\n\n")

    index_path = VAULT_DIR / "_vault-index.md"
    index_path.write_text("".join(lines), encoding="utf-8")
    print(f"  {GREEN}✓ Generated _vault-index.md{RESET}")


# ============================================================
# Main
# ============================================================

def main() -> None:
    dry_run = "--dry-run" in sys.argv
    mode = "DRY RUN" if dry_run else "FULL BUILD"

    print(f"\n{'=' * 60}")
    print(f"  Study Vault Builder  [{CYAN}{mode}{RESET}]")
    print(f"{'=' * 60}")
    print(f"\n  Source : {SOURCE_DIR}")
    print(f"  Vault  : {VAULT_DIR}")
    print(f"  Model  : {LMSTUDIO_MODEL}")
    print(f"  LM URL : {LMSTUDIO_BASE_URL}\n")

    # ----------------------------------------------------------
    # Phase 0 — LM Studio health check
    # ----------------------------------------------------------
    print(f"{CYAN}─── Phase 0: LM Studio health check ───{RESET}")
    check_lmstudio()
    print(f"  {GREEN}✓ LM Studio reachable{RESET}\n")

    # ----------------------------------------------------------
    # Phase 1 — Clean slate
    # ----------------------------------------------------------
    print(f"{CYAN}─── Phase 1: Clean slate ───{RESET}")

    if dry_run:
        if VAULT_DIR.exists():
            print(f"  [DRY RUN] Would delete  : {VAULT_DIR}")
        if JUNK_BIN_DIR.exists():
            print(f"  [DRY RUN] Would delete  : {JUNK_BIN_DIR}")
        print(f"  [DRY RUN] Would recreate: {VAULT_DIR}\n")
    else:
        answer = input(
            f"\n  This will DELETE {VAULT_DIR} and rebuild from scratch.\n"
            "  Continue? [y/N] "
        ).strip().lower()
        if answer not in {"y", "yes"}:
            print("  Aborted.")
            sys.exit(0)

        if VAULT_DIR.exists():
            shutil.rmtree(VAULT_DIR)
            print(f"  Deleted : {VAULT_DIR}")
        if JUNK_BIN_DIR.exists():
            shutil.rmtree(JUNK_BIN_DIR)
            print(f"  Deleted : {JUNK_BIN_DIR}")

        VAULT_DIR.mkdir(parents=True, exist_ok=True)
        print(f"  {GREEN}✓ Recreated: {VAULT_DIR}{RESET}\n")

    # ----------------------------------------------------------
    # Phase 2 — Scan & filter
    # ----------------------------------------------------------
    print(f"{CYAN}─── Phase 2: Scan & filter ───{RESET}")

    if not SOURCE_DIR.exists():
        print(f"  {RED}ERROR: Source directory not found: {SOURCE_DIR}{RESET}")
        sys.exit(1)

    entries = scan_source_files()
    print(f"  Found {len(entries)} study-relevant files\n")

    # Per-course / per-category summary
    for course_code in COURSE_NAMES:
        course_files = [e for e in entries if e.course == course_code]
        cat_counts: dict[str, int] = {}
        for e in course_files:
            cat_counts[e.category] = cat_counts.get(e.category, 0) + 1
        cat_str = ", ".join(f"{k}: {v}" for k, v in sorted(cat_counts.items()))
        print(f"  {course_code}: {len(course_files)} files  ({cat_str})")

    print()
    entries = deduplicate(entries)

    # ----------------------------------------------------------
    # Phase 3 — Gemma renaming + copy
    # ----------------------------------------------------------
    print(f"{CYAN}─── Phase 3: Gemma renaming + copy ───{RESET}\n")

    total = len(entries)
    needs_attention: list[Path] = []
    course_counts: dict[str, int] = {code: 0 for code in COURSE_NAMES}
    total_copied = 0

    retry_waits = [2, 5, 10]  # seconds to wait between retries

    for idx, entry in enumerate(entries, start=1):
        label = f"[{idx}/{total}]"
        course_name = COURSE_NAMES.get(entry.course, entry.course)
        ext = entry.source_path.suffix.lower()

        # Try up to 3 times, waiting longer each time if model crashes
        clean_stem, success, used_ocr = None, False, False
        for attempt in range(3):
            clean_stem, success, used_ocr = rename_entry(entry, idx, total)
            if success:
                break
            # Check if the model is down (crash during rename)
            if not check_lmstudio(exit_on_fail=False):
                wait_sec = retry_waits[attempt]
                print(
                    f"    {CYAN}Model crashed — retry {attempt + 2}/3 "
                    f"in {wait_sec}s...{RESET}"
                )
                time.sleep(wait_sec)
                if not wait_for_model(max_wait=wait_sec + 30):
                    continue
            else:
                # Model is fine, naming just failed — don't retry
                break

        if not success:
            needs_attention.append(entry.source_path)
            print(
                f"  {label} {entry.source_path.name} "
                f"-> {RED}NEEDS ATTENTION{RESET} (text + OCR naming failed)"
            )
            # Still copy the file using its original stem so nothing is lost
            clean_stem = entry.source_path.stem

        clean_name = clean_stem + ext
        ocr_tag = " (OCR)" if used_ocr else ""
        dest_dir = VAULT_DIR / course_name / entry.category
        dest_path = dest_dir / clean_name

        if success:
            print(
                f"  {label} {CYAN}{entry.source_path.name}{RESET} "
                f"-> {GREEN}{clean_name}{RESET}{ocr_tag}"
            )

        if dry_run:
            short_dest = dest_dir.relative_to(VAULT_DIR.parent)
            print(f"    [DRY RUN] Would copy to: {short_dest}/{clean_name}")
            total_copied += 1
            course_counts[entry.course] = course_counts.get(entry.course, 0) + 1
            continue

        # Handle name collisions
        if dest_path.exists():
            stem = dest_path.stem
            counter = 2
            while dest_path.exists():
                dest_path = dest_dir / f"{stem} ({counter}){ext}"
                counter += 1

        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(entry.source_path, dest_path)
            total_copied += 1
            course_counts[entry.course] = course_counts.get(entry.course, 0) + 1
        except Exception as e:
            print(f"  {RED}Error copying {entry.source_path.name}: {e}{RESET}")

    print()

    # ----------------------------------------------------------
    # Phase 4 — Generate Obsidian indexes
    # ----------------------------------------------------------
    if not dry_run:
        print(f"{CYAN}─── Phase 4: Generating indexes ───{RESET}\n")
        for course_code, course_name in COURSE_NAMES.items():
            generate_course_index(course_code, course_name)
        generate_vault_index()
        print()

    # ----------------------------------------------------------
    # Phase 5 — Wipe ChromaDB + re-ingest
    # ----------------------------------------------------------
    if not dry_run:
        print(f"{CYAN}─── Phase 5: Wipe ChromaDB + re-ingest ───{RESET}\n")

        project_root = Path(__file__).resolve().parent.parent
        data_dir = project_root / "data"
        chroma_dir = data_dir / "chroma"
        manifest_path = data_dir / "manifest.json"

        if chroma_dir.exists():
            shutil.rmtree(chroma_dir)
            print(f"  Deleted : {chroma_dir}")
        if manifest_path.exists():
            manifest_path.unlink()
            print(f"  Deleted : {manifest_path}")

        # Make the backend package importable when run from the project root
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        try:
            from backend.ingestion.pipeline import IngestionPipeline  # noqa: PLC0415
            print("  Starting ingestion...\n")
            result = IngestionPipeline().ingest(force_full=True)
            print(
                f"\n  {GREEN}✓ Ingestion complete — "
                f"added {result.new} chunks, updated {result.updated} chunks{RESET}"
            )
        except Exception as e:
            print(f"  {RED}Ingestion failed: {e}{RESET}")
            print(
                "  (The vault was built successfully — "
                "re-ingest manually with scripts/force_ingest.py)"
            )

        print()

    # ----------------------------------------------------------
    # End of run — summary
    # ----------------------------------------------------------
    print(f"{'=' * 60}")
    print(f"  {'DRY RUN SUMMARY' if dry_run else 'BUILD COMPLETE'}")
    print(f"{'=' * 60}")

    verb = "would be copied" if dry_run else "copied"
    print(f"\n  Total files {verb}: {total_copied}")
    for course_code, course_name in COURSE_NAMES.items():
        count = course_counts.get(course_code, 0)
        print(f"    {course_name}: {count} files")

    if needs_attention:
        print(f"\n  {RED}{len(needs_attention)} files need manual renaming:{RESET}")
        for p in needs_attention:
            print(f"    - {p}")

    if dry_run:
        print(f"\n  {CYAN}This was a dry run. No files were modified.{RESET}")
    else:
        print(f"\n  {GREEN}Vault created at: {VAULT_DIR}{RESET}")
        print("  Open this folder in Obsidian to browse your notes.")

    print()


if __name__ == "__main__":
    main()
