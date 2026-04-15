"""
Course listing API endpoint.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import APIRouter

from ..ingestion.pipeline import IngestionPipeline

router = APIRouter(prefix="/api/courses", tags=["courses"])

_pipeline: Optional[IngestionPipeline] = None


def get_pipeline() -> IngestionPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = IngestionPipeline()
    return _pipeline


@router.get("")
async def list_courses():
    """List all courses with document counts."""
    pipeline = get_pipeline()
    stats = pipeline.get_stats()

    courses = []
    for course_name, info in stats.get("courses", {}).items():
        courses.append({
            "name": course_name,
            "files": info["files"],
            "chunks": info["chunks"],
        })

    # Sort by name
    courses.sort(key=lambda c: c["name"])

    return {"courses": courses}
