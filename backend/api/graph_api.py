"""
Knowledge Graph API Endpoints

Serves the knowledge graph data to the frontend visualizer
and provides endpoints for graph operations.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from backend.ingestion.graph import GRAPH_OUTPUT_PATH, load_graph

load_dotenv()

router = APIRouter(prefix="/api/graph", tags=["Knowledge Graph"])


@router.get("")
async def get_graph():
    """Return the full knowledge graph."""
    graph = load_graph()
    if not graph:
        raise HTTPException(
            status_code=404,
            detail="Knowledge graph not found. Run `python scripts/build_graph.py` to generate it.",
        )
    return JSONResponse(content=graph)


@router.get("/stats")
async def get_graph_stats():
    """Return graph metadata and statistics."""
    graph = load_graph()
    if not graph:
        raise HTTPException(status_code=404, detail="Knowledge graph not found.")

    metadata = graph.get("metadata", {})
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])

    # Compute additional stats
    type_counts = {}
    course_counts = {}
    for node in nodes:
        t = node.get("type", "Unknown")
        type_counts[t] = type_counts.get(t, 0) + 1
        for course in node.get("courses", []):
            course_counts[course] = course_counts.get(course, 0) + 1

    relation_counts = {}
    for edge in edges:
        r = edge.get("relation", "unknown")
        relation_counts[r] = relation_counts.get(r, 0) + 1

    return JSONResponse(content={
        **metadata,
        "type_breakdown": type_counts,
        "course_breakdown": course_counts,
        "relation_breakdown": relation_counts,
    })


@router.get("/node/{node_id}")
async def get_node(node_id: str):
    """Return a specific node with all its connections."""
    graph = load_graph()
    if not graph:
        raise HTTPException(status_code=404, detail="Knowledge graph not found.")

    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])

    # Find the node
    node = next((n for n in nodes if n["id"] == node_id), None)
    if not node:
        raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found.")

    # Find all connected edges and neighbor nodes
    connected_edges = [
        e for e in edges
        if e["source"] == node_id or e["target"] == node_id
    ]

    neighbor_ids = set()
    for edge in connected_edges:
        neighbor_ids.add(edge["source"])
        neighbor_ids.add(edge["target"])
    neighbor_ids.discard(node_id)

    neighbors = [n for n in nodes if n["id"] in neighbor_ids]

    return JSONResponse(content={
        "node": node,
        "edges": connected_edges,
        "neighbors": neighbors,
    })


@router.get("/search")
async def search_nodes(q: str = Query(..., min_length=1, description="Search query")):
    """Search nodes by label (fuzzy prefix match)."""
    graph = load_graph()
    if not graph:
        raise HTTPException(status_code=404, detail="Knowledge graph not found.")

    query = q.lower().strip()
    nodes = graph.get("nodes", [])

    results = []
    for node in nodes:
        label = node.get("label", "").lower()
        desc = node.get("description", "").lower()
        if query in label or query in desc:
            results.append(node)

    # Sort by relevance: exact label match first, then label contains, then description
    results.sort(key=lambda n: (
        0 if n["label"].lower() == query else
        1 if query in n["label"].lower() else 2
    ))

    return JSONResponse(content={"results": results[:20], "total": len(results)})
