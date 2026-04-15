"""
Document management API endpoints.

Sync vault, upload files, list documents, get stats.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, UploadFile
from pydantic import BaseModel

from ..ingestion.pipeline import IngestionPipeline

router = APIRouter(prefix="/api/documents", tags=["documents"])

_pipeline: Optional[IngestionPipeline] = None


def get_pipeline() -> IngestionPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = IngestionPipeline()
    return _pipeline


@router.post("/sync")
async def sync_vault():
    """
    Scan the vault for new/modified/deleted files and process changes.
    
    Returns a summary of what was processed.
    """
    pipeline = get_pipeline()
    result = pipeline.ingest()

    return {
        "new": result.new,
        "updated": result.updated,
        "deleted": result.deleted,
        "skipped": result.skipped,
        "total_chunks": result.total_chunks,
        "errors": result.errors,
        "duration_seconds": round(result.duration_seconds, 2),
    }


@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    course: str = Form(...),
    category: str = Form(default="Resources"),
):
    """
    Upload a file directly to the vault and auto-ingest it.
    
    The file is saved to the vault under the specified course/category,
    then immediately ingested into the index.
    """
    pipeline = get_pipeline()
    vault_path = pipeline.vault_path

    # Determine destination
    dest_dir = vault_path / course / category
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / file.filename

    # Handle name collisions
    if dest_path.exists():
        stem = dest_path.stem
        ext = dest_path.suffix
        counter = 2
        while dest_path.exists():
            dest_path = dest_dir / f"{stem} ({counter}){ext}"
            counter += 1

    # Save the file
    with open(dest_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Trigger ingestion for the new file
    result = pipeline.ingest()

    return {
        "filename": dest_path.name,
        "path": str(dest_path.relative_to(vault_path)),
        "course": course,
        "category": category,
        "ingestion": {
            "new": result.new,
            "total_chunks": result.total_chunks,
        },
    }


@router.get("")
async def list_documents(course: Optional[str] = None):
    """List all indexed documents, optionally filtered by course."""
    pipeline = get_pipeline()
    manifest = pipeline.manifest

    documents = []
    for rel_path, info in manifest.get("files", {}).items():
        parts = Path(rel_path).parts
        doc_course = parts[0] if len(parts) > 0 else "Unknown"
        doc_category = parts[1] if len(parts) > 1 else "Resources"
        doc_name = parts[-1] if parts else rel_path

        if course and doc_course != course:
            continue

        documents.append({
            "name": doc_name,
            "path": rel_path,
            "course": doc_course,
            "category": doc_category,
            "chunks": info.get("chunks", 0),
            "ingested_at": info.get("ingested_at", ""),
        })

    # Sort by course, then category, then name
    documents.sort(key=lambda d: (d["course"], d["category"], d["name"]))

    return {"documents": documents, "total": len(documents)}


@router.delete("/{doc_path:path}")
async def delete_document(doc_path: str):
    """Remove a document from the index (does not delete the file from vault)."""
    pipeline = get_pipeline()

    if doc_path not in pipeline.manifest.get("files", {}):
        return {"error": f"Document not found: {doc_path}"}

    pipeline._remove_file(doc_path)
    pipeline._save_manifest()

    return {"deleted": doc_path}


@router.get("/stats")
async def get_stats():
    """Get statistics about the indexed documents."""
    pipeline = get_pipeline()
    return pipeline.get_stats()
