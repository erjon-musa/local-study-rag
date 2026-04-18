"""
One-shot re-OCR for a course's zero-chunk files.

Why this script exists
----------------------
After rebuilding the manifest from ChromaDB, any files on disk that never
produced chunks show up with `chunks == 0` (`status: "missing"`). The most
common cause: scanned PDFs whose pages had no registered images, so the
old OCR trigger skipped them entirely. With the broadened trigger in
`backend/ingestion/loader.py` (pixmap stddev check), those pages now
qualify for OCR — but only if we actually re-run ingestion against them.

This script does exactly that, nothing else:
  1. Read `data/manifest.json`.
  2. Filter to entries whose derived-or-recorded course matches --course
     AND whose `chunks` is 0.
  3. Defensively clear any stragglers in ChromaDB for each `rel_path`.
  4. Call `pipeline._ingest_file(path, force=True)` to re-ingest through
     the existing loader (LightOnOCR on MPS first, Gemma via LM Studio
     as fallback — that chain lives in `loader.py`, don't duplicate it).
  5. Print per-file progress and a final tally.

Exit code 0 when every file produced at least one chunk; exit code 1 if
anything totally failed.

Usage
-----
    python scripts/force_reocr.py --course "ELEC 472"
    python scripts/force_reocr.py --course "ELEC 472" --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

# Make `backend` importable when invoked as `python scripts/force_reocr.py`.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.ingestion.pipeline import IngestionPipeline  # noqa: E402

MANIFEST_PATH = Path(os.getenv("MANIFEST_PATH", "./data/manifest.json"))


def _derive_course(rel_path: str, recorded_course: str = "") -> str:
    """
    Figure out the course name for a manifest entry.

    Entries produced by `rebuild_manifest.py` for files *missing* from
    the index have `course: null` — in that case, the first path segment
    (e.g. "ELEC 472 - Artificial Intelligence") is the vault's own
    convention for a course folder.
    """
    if recorded_course:
        return recorded_course
    parts = Path(rel_path).parts
    return parts[0] if parts else ""


def _course_matches(entry_course: str, requested: str) -> bool:
    """
    Match the user's `--course "ELEC 472"` against the entry's course name.

    Accept substring match (case-insensitive) so the user doesn't have to
    type the full "ELEC 472 - Artificial Intelligence" name.
    """
    if not entry_course or not requested:
        return False
    return requested.strip().lower() in entry_course.lower()


def _pick_targets(manifest: dict, requested_course: str) -> List[Tuple[str, dict]]:
    """Return [(rel_path, entry)] for files in --course with chunks == 0."""
    targets: List[Tuple[str, dict]] = []
    for rel_path, entry in manifest.get("files", {}).items():
        if not isinstance(entry, dict):
            continue
        if entry.get("chunks", 0) != 0:
            continue
        entry_course = _derive_course(rel_path, entry.get("course") or "")
        if _course_matches(entry_course, requested_course):
            targets.append((rel_path, entry))
    return targets


def _format_page_log(rel_path: str, stats, duration_s: float) -> str:
    """Turn PdfLoadStats.pages into human-readable per-page lines.

    Target format (from plan Task 5):
        "[ELEC 472/Lectures/Lecture Chapter 2 Search.pdf] page 3/18: "
        "OCR via LightOnOCR (1.4s, 2341 chars)"
    """
    lines: List[str] = []
    total = len(stats.pages) or 1
    for pr in stats.pages:
        method = pr.extraction_method
        if method in ("ocr_local_lighton", "ocr_gemma4"):
            tag = "OCR via LightOnOCR" if method == "ocr_local_lighton" else "OCR via Gemma"
            lines.append(
                f"[{rel_path}] page {pr.page}/{total}: {tag} "
                f"({pr.duration_s:.1f}s, {pr.text_length} chars)"
            )
        elif method == "failed":
            lines.append(
                f"[{rel_path}] page {pr.page}/{total}: OCR FAILED "
                f"({pr.duration_s:.1f}s, {pr.error})"
            )
        elif method == "skipped_blank":
            lines.append(f"[{rel_path}] page {pr.page}/{total}: skipped (blank)")
        # "text" pages (digital extraction) are the common case; not spammed.
    if duration_s > 0:
        lines.append(f"[{rel_path}] total: {duration_s:.1f}s")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Re-OCR a course's zero-chunk files through the existing OCR chain."
    )
    parser.add_argument(
        "--course",
        required=True,
        help='Course name (substring match, case-insensitive). Example: "ELEC 472"',
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report which files would be processed; do not re-ingest.",
    )
    args = parser.parse_args()

    if "DISABLE_OCR" in os.environ:
        print(
            "ERROR: DISABLE_OCR is set. Re-OCR would be a no-op. "
            "Unset it and re-run.",
            file=sys.stderr,
        )
        return 1

    if not MANIFEST_PATH.exists():
        print(
            f"ERROR: manifest not found at {MANIFEST_PATH}. "
            "Run `python scripts/rebuild_manifest.py` first.",
            file=sys.stderr,
        )
        return 1

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    targets = _pick_targets(manifest, args.course)

    print(
        f"Would process {len(targets)} zero-chunk file(s) matching "
        f"--course {args.course!r}."
    )
    if args.dry_run:
        for rel_path, _entry in targets:
            print(f"  - {rel_path}")
        return 0

    if not targets:
        print("Nothing to do — no zero-chunk files for this course.")
        return 0

    pipeline = IngestionPipeline()
    vault_path = pipeline.vault_path

    total_files = 0
    total_pages_ocr = 0
    total_pages_failed = 0
    total_new_chunks = 0
    totally_failed_files: List[str] = []

    for rel_path, _entry in targets:
        abs_path = vault_path / rel_path
        if not abs_path.exists():
            print(f"  ✗ {rel_path}: file missing on disk — skipping")
            totally_failed_files.append(rel_path)
            continue

        # Defensive cleanup — should be a no-op since chunks == 0, but
        # guards against manifest/Chroma drift leaking stale chunks.
        try:
            pipeline.collection.delete(where={"rel_path": rel_path})
        except Exception as e:
            print(
                f"  ⚠ pre-delete failed for {rel_path}: "
                f"{type(e).__name__}: {e}"
            )

        print(f"\n→ Re-ingesting: {rel_path}")
        start = time.time()
        try:
            chunk_count, chunk_ids, stats = pipeline._ingest_file(
                abs_path, force=True
            )
        except Exception as e:
            err_class = type(e).__name__
            print(f"  ✗ {rel_path}: {err_class}: {e}")
            totally_failed_files.append(rel_path)
            continue

        duration = time.time() - start
        print(_format_page_log(rel_path, stats, duration))

        if chunk_count == 0:
            totally_failed_files.append(rel_path)
            print(
                f"  ✗ {rel_path}: 0 chunks produced "
                f"(ocr_pages={stats.ocr_pages}, failed={stats.ocr_failed_pages})"
            )
        else:
            print(
                f"  ✓ {rel_path}: {chunk_count} chunks "
                f"(ocr_pages={stats.ocr_pages}, failed={stats.ocr_failed_pages})"
            )

        # Record manifest entry exactly as pipeline.ingest() would.
        pipeline.manifest["files"][rel_path] = {
            "hash": pipeline._file_hash(abs_path),
            "chunks": chunk_count,
            "ingested_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "chunk_ids": chunk_ids,
            "ocr_pages": stats.ocr_pages,
            "ocr_failed_pages": stats.ocr_failed_pages,
            "skipped_blank_pages": stats.skipped_blank_pages,
        }

        total_files += 1
        total_pages_ocr += stats.ocr_pages
        total_pages_failed += stats.ocr_failed_pages
        total_new_chunks += chunk_count

    # Persist manifest (atomic write via pipeline helper).
    pipeline._save_manifest()

    print(
        "\nRe-OCR complete: "
        f"{total_files} files, "
        f"{total_pages_ocr} pages OCR'd, "
        f"{total_pages_failed} pages failed, "
        f"{total_new_chunks} new chunks added."
    )
    if totally_failed_files:
        print("\nFiles that produced 0 chunks:")
        for rp in totally_failed_files:
            print(f"  - {rp}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
