"""
Knowledge Graph Extraction Module

Extracts entities and relationships from document chunks using LM Studio,
then builds and persists a knowledge graph as JSON.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..config import settings
from ..lm_client import get_sync_client

# Aliased here for backwards compatibility — callers can still
# `from backend.ingestion.graph import GRAPH_OUTPUT_PATH`. Source of
# truth lives in `backend.config.settings.graph_output_path`.
GRAPH_OUTPUT_PATH: Path = settings.graph_output_path

# Entity types we tell the LLM to look for
ENTITY_TYPES = [
    "Concept",
    "Algorithm",
    "Theorem",
    "Formula",
    "Tool",
    "Person",
    "Course",
    "Topic",
]

# ── Prompt ──────────────────────────────────────────────────────────

EXTRACTION_PROMPT = """You are a knowledge graph extraction engine for university course materials.

Given the following text from a document, extract entities and relationships.

ENTITY TYPES: {entity_types}

RULES:
1. Extract only meaningful academic entities (concepts, algorithms, theorems, etc.)
2. Entity names should be concise and normalized (e.g., "Binary Search" not "the binary search algorithm")
3. Relationships should use short, clear verbs (e.g., "is_a", "uses", "extends", "part_of", "related_to", "implements", "requires", "compares_to", "proves", "defined_by")
4. Do NOT extract generic/obvious entities like "student", "professor", "class", "assignment"
5. Extract at most 10 entities and 15 relationships per chunk

Respond with ONLY valid JSON in this exact format, no other text:
{{
  "entities": [
    {{"name": "Entity Name", "type": "Concept", "description": "Brief 1-sentence description"}}
  ],
  "relations": [
    {{"source": "Entity A", "target": "Entity B", "relation": "relationship_verb"}}
  ]
}}

If no meaningful entities can be extracted, respond with:
{{"entities": [], "relations": []}}

TEXT:
---
{text}
---

DOCUMENT CONTEXT:
- Course: {course}
- Category: {category}
- File: {filename}"""


# ── Data Structures ─────────────────────────────────────────────────

@dataclass
class GraphNode:
    """A node in the knowledge graph."""
    id: str
    label: str
    type: str
    description: str
    courses: set = field(default_factory=set)
    doc_count: int = 0
    source_files: set = field(default_factory=set)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "type": self.type,
            "description": self.description,
            "courses": sorted(self.courses),
            "doc_count": self.doc_count,
            "source_files": sorted(self.source_files),
        }


@dataclass
class GraphEdge:
    """An edge in the knowledge graph."""
    source: str
    target: str
    relation: str
    weight: int = 1

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "target": self.target,
            "relation": self.relation,
            "weight": self.weight,
        }


@dataclass
class KnowledgeGraph:
    """The full knowledge graph."""
    nodes: dict[str, GraphNode] = field(default_factory=dict)
    edges: dict[str, GraphEdge] = field(default_factory=dict)
    chunks_processed: int = 0
    errors: list[str] = field(default_factory=list)

    def add_node(self, name: str, type_: str, description: str,
                 course: str = "", source_file: str = "") -> str:
        """Add or merge a node. Returns the node ID."""
        node_id = self._normalize_id(name)
        if node_id in self.nodes:
            node = self.nodes[node_id]
            node.doc_count += 1
            if course:
                node.courses.add(course)
            if source_file:
                node.source_files.add(source_file)
            # Keep the longer/better description
            if len(description) > len(node.description):
                node.description = description
        else:
            self.nodes[node_id] = GraphNode(
                id=node_id,
                label=name,
                type=type_,
                description=description,
                courses={course} if course else set(),
                doc_count=1,
                source_files={source_file} if source_file else set(),
            )
        return node_id

    def add_edge(self, source_name: str, target_name: str, relation: str):
        """Add or increment an edge."""
        source_id = self._normalize_id(source_name)
        target_id = self._normalize_id(target_name)

        # Only add edges between existing nodes
        if source_id not in self.nodes or target_id not in self.nodes:
            return

        # Don't create self-loops
        if source_id == target_id:
            return

        edge_key = f"{source_id}|{relation}|{target_id}"
        if edge_key in self.edges:
            self.edges[edge_key].weight += 1
        else:
            self.edges[edge_key] = GraphEdge(
                source=source_id,
                target=target_id,
                relation=relation,
            )

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        return {
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges.values()],
            "metadata": {
                "last_built": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "total_chunks_processed": self.chunks_processed,
                "total_nodes": len(self.nodes),
                "total_edges": len(self.edges),
                "errors": len(self.errors),
            },
        }

    @staticmethod
    def _normalize_id(name: str) -> str:
        """Normalize an entity name to a stable ID."""
        return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


# ── LM Studio Integration ──────────────────────────────────────────────

def call_llm(prompt: str, timeout: float = 120.0) -> str:
    """
    Send a prompt to LM Studio and return the response text.
    All processing happens on the PC GPU via LM Link.
    """
    # Shared client + per-request timeout (entity extraction needs a longer
    # ceiling than chat does).
    client = get_sync_client().with_options(timeout=timeout)

    response = client.chat.completions.create(
        model=settings.lmstudio_model,
        messages=[
            {"role": "system", "content": "You are a knowledge graph extraction engine. Respond only with valid JSON. Do not use internal reasoning. Respond directly."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=2048,
    )

    message = response.choices[0].message
    text = message.content or ""
    if not text:
        text = getattr(message, "reasoning_content", None) or ""
    return text


def parse_llm_response(response_text: str) -> dict | None:
    """Parse the LLM's JSON response, handling common formatting issues."""
    text = response_text.strip()

    # Try to find JSON in the response (LLM might wrap it in markdown code blocks)
    json_match = re.search(r"\{[\s\S]*\}", text)
    if not json_match:
        return None

    try:
        data = json.loads(json_match.group())
        # Validate structure
        if "entities" in data and "relations" in data:
            return data
        return None
    except json.JSONDecodeError:
        return None


# ── Extraction Pipeline ─────────────────────────────────────────────

def extract_from_chunk(
    chunk_text: str,
    course: str = "Unknown",
    category: str = "Unknown",
    filename: str = "Unknown",
) -> dict | None:
    """
    Extract entities and relations from a single text chunk.
    Returns the parsed result dict or None on failure.
    """
    prompt = EXTRACTION_PROMPT.format(
        entity_types=", ".join(ENTITY_TYPES),
        text=chunk_text[:3000],  # Cap input length
        course=course,
        category=category,
        filename=filename,
    )

    response = call_llm(prompt)
    return parse_llm_response(response)


def build_graph(
    chunks: list[dict],
    progress_callback=None,
) -> KnowledgeGraph:
    """
    Build a knowledge graph from a list of document chunks.

    Each chunk should be a dict with:
      - text: str (the chunk content)
      - course: str (e.g., "ELEC 477")
      - category: str (e.g., "Lectures")
      - filename: str (source filename)

    Args:
        chunks: List of chunk dicts
        progress_callback: Optional callable(current, total, message) for progress updates
    """
    graph = KnowledgeGraph()
    total = len(chunks)

    for i, chunk in enumerate(chunks):
        text = chunk.get("text", "")
        course = chunk.get("course", "Unknown")
        category = chunk.get("category", "Unknown")
        filename = chunk.get("filename", "Unknown")

        if progress_callback:
            progress_callback(i + 1, total, f"Processing: {filename}")

        # Skip very short chunks
        if len(text.strip()) < 50:
            continue

        try:
            result = extract_from_chunk(text, course, category, filename)
            if not result:
                graph.errors.append(f"Chunk {i}: Failed to parse LLM response")
                continue

            # Add entities
            for entity in result.get("entities", []):
                name = entity.get("name", "").strip()
                etype = entity.get("type", "Concept")
                desc = entity.get("description", "")

                if not name or len(name) < 2:
                    continue

                # Validate entity type
                if etype not in ENTITY_TYPES:
                    etype = "Concept"

                graph.add_node(name, etype, desc, course, filename)

            # Add relations
            for rel in result.get("relations", []):
                source = rel.get("source", "").strip()
                target = rel.get("target", "").strip()
                relation = rel.get("relation", "related_to").strip()

                if source and target:
                    graph.add_edge(source, target, relation)

            graph.chunks_processed += 1

        except Exception as e:
            error_msg = f"Chunk {i} ({filename}): {str(e)}"
            graph.errors.append(error_msg)
            print(f"  ✗ {error_msg}")

    return graph


# ── Persistence ─────────────────────────────────────────────────────

def save_graph(graph: KnowledgeGraph, path: Path | None = None) -> Path:
    """Save the knowledge graph to a JSON file."""
    path = path or GRAPH_OUTPUT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(graph.to_dict(), f, indent=2)

    print(f"  ✓ Graph saved to {path}")
    print(f"    {len(graph.nodes)} nodes, {len(graph.edges)} edges")
    return path


def load_graph(path: Path | None = None) -> dict | None:
    """Load a knowledge graph from its JSON file."""
    path = path or GRAPH_OUTPUT_PATH
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)
