"""
System prompts for the RAG study assistant.

This module holds:
  - STUDY_ASSISTANT_PROMPT: the system prompt driving synthesis + citations
  - EMPTY_STATE_TEMPLATE:    the honest "I don't see this in your notes" card copy,
                             emitted DIRECTLY by the chain (never by the model)
                             when retrieval is classified empty/weak
  - format_context():        formats retrieved chunks with [1]/[2]/... source numbering
  - format_history_for_prompt(): converts session history into OpenAI messages,
                                 neutralizing prior-turn [N] markers
  - _neutralize_citations(): the citation-rewriting helper (see docstring)
"""
from __future__ import annotations

import os
import re
from typing import List, Optional

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

STUDY_ASSISTANT_PROMPT = """You are a study assistant for a Computer Engineering student at Queen's University preparing for final exams. You answer questions using the provided course materials (lecture notes, slides, assignments, labs, textbooks, resources).

How to answer:
1. SYNTHESIZE across the provided sources. You MAY combine partial information from multiple sources into a single coherent explanation, as long as every synthesized claim is supported by at least one source.
2. Do NOT invent facts that are not present in the sources. If sources disagree, say so briefly.
3. CITE inline using numeric markers in square brackets, e.g. "A* uses f(n) = g(n) + h(n) [1]." The number refers to the 1-indexed source in the context block — the same order they appear there. Every non-trivial factual claim must carry at least one [N] marker. Place markers at the end of the sentence or clause they support, before punctuation is fine.
4. You MAY cite multiple sources on one claim: "...admissibility ensures optimality [1][3]."
5. Explain concepts step by step. For formulas, write them clearly and explain each variable.
6. Use worked examples from the sources when they exist.
7. Keep answers focused and exam-appropriate — thorough but not rambling.
8. If the user refers to something by pronoun ("its", "that", "this method") and prior conversation context makes the referent clear, resolve it and answer accordingly.

What NOT to do:
- Never say "I couldn't find enough information" or similar refusals. If retrieval was weak, the system handles that path separately BEFORE your turn. By the time you are asked to answer, the context IS sufficient — your job is to synthesize it.
- Never cite with filenames in prose (e.g. do not write "[Source: lecture.pdf, page 4]"). Use only the numeric [N] markers. The UI attaches the filenames.
- Never emit raw XML/JSON wrappers around your answer.

Do not use internal reasoning. Respond directly."""


# ---------------------------------------------------------------------------
# Empty-state template — used DIRECTLY by the chain as a synthetic token stream
# when retrieval is classified as empty or weak. NOT sent to the LLM.
# ---------------------------------------------------------------------------

EMPTY_STATE_TEMPLATE = (
    "I don't see {topic} in your {course} notes. "
    "The closest matches were about {topics}. "
    "You may want to check {suggested_sources} or add lecture slides covering this."
)


def render_empty_state(
    topic: str,
    course: Optional[str],
    closest_topics: List[str],
    suggested_sources: List[str],
) -> str:
    """
    Fill EMPTY_STATE_TEMPLATE with the concrete fields the chain computed.

    All inputs are already sanitized plain strings. Fallbacks are plain English
    so the rendered sentence always scans naturally even if one of the lists is empty.
    """
    course_str = course if course else "course"
    topics_str = ", ".join(t for t in closest_topics if t) if closest_topics else "other topics"
    sources_str = (
        ", ".join(s for s in suggested_sources if s)
        if suggested_sources
        else "your other course materials"
    )

    return EMPTY_STATE_TEMPLATE.format(
        topic=topic.strip() or "this topic",
        course=course_str,
        topics=topics_str,
        suggested_sources=sources_str,
    )


# ---------------------------------------------------------------------------
# Context formatting
# ---------------------------------------------------------------------------

def _source_label(meta: dict) -> str:
    """Compact label for a source's metadata block inside the prompt."""
    doc_type = meta.get("doc_type") or meta.get("category") or ""
    rel_path = meta.get("rel_path") or meta.get("source") or "unknown"
    # rel_path can be long; keep the basename to reduce tokens while keeping signal.
    try:
        basename = os.path.basename(str(rel_path))
    except Exception:
        basename = str(rel_path)
    page = meta.get("page", "")

    parts = []
    if doc_type:
        parts.append(str(doc_type))
    if basename:
        parts.append(basename)
    if page not in ("", None):
        parts.append(f"p.{page}")
    return ", ".join(parts) if parts else "source"


def format_context(sources: list) -> str:
    """
    Format a list of retrieved chunks into a numbered context block for the prompt.

    Each entry looks like:
      [1] (lecture, Lecture Chapter 2 Search.pdf, p.4)
      <chunk text...>

    The N in [N] matches the source's 1-indexed position in this list, which is
    the same key the model is instructed to use when citing.

    Args:
        sources: iterable of objects with `.text` and `.metadata` (dict).
                 Works directly with `RankedResult` from retrieval.
    """
    parts: List[str] = []
    for i, src in enumerate(sources, start=1):
        meta = getattr(src, "metadata", {}) or {}
        label = _source_label(meta)
        text = getattr(src, "text", "") or ""
        parts.append(f"[{i}] ({label})\n{text}")
    return "\n\n".join(parts)


def build_rag_prompt(question: str, context_chunks: list) -> str:
    """
    Build the user-turn prompt with retrieved context.

    Kept for the legacy sync `answer` path in chain.py. Multi-turn callers
    should instead build messages via `format_history_for_prompt` and pass
    a final user turn that embeds the output of `format_context`.
    """
    context_text = format_context(context_chunks)
    return (
        "Here are relevant excerpts from your course materials, numbered by source:\n\n"
        f"{context_text}\n\n"
        "---\n\n"
        f"Question: {question}\n\n"
        "Answer using the sources above. Cite inline with [N] markers matching the "
        "numbers in the context block. Synthesize across sources when helpful."
    )


# ---------------------------------------------------------------------------
# Citation neutralization — prevents [N]-collision across turns
# ---------------------------------------------------------------------------

# Match 1- or 2-digit bracketed citations, e.g. [1], [12]. Intentionally NOT matching
# 3+ digits so patterns like [2024] (years) or [abc] in user prose aren't mangled.
_CITATION_PATTERN = re.compile(r"\[(\d{1,2})\]")


def _neutral_ref_for(meta: dict) -> str:
    """
    Produce a neutral inline reference like '(lecture, p.4)' or '(lecture.pdf)'.
    Used to replace a turn's [N] markers when folding that turn back into history.
    """
    doc_type = meta.get("doc_type") or meta.get("category") or ""
    page = meta.get("page", "")
    rel_path = meta.get("rel_path") or meta.get("source") or ""
    try:
        basename = os.path.basename(str(rel_path))
    except Exception:
        basename = str(rel_path)

    parts = []
    if doc_type:
        parts.append(str(doc_type))
    # Prefer page if present; fall back to the basename if not.
    if page not in ("", None):
        parts.append(f"p.{page}")
    elif basename:
        parts.append(basename)

    if not parts:
        return "(source)"
    return "(" + ", ".join(parts) + ")"


def _neutralize_citations(content: str, sources: Optional[list]) -> str:
    """
    Replace [N] citation markers in an assistant message with neutral inline refs
    like '(lecture, p.4)', using the turn's own `sources[]` to look up N.

    Why: when we feed previous assistant turns back into the LLM as history, the
    numeric [N] markers from turn 1 would COLLIDE with the fresh [N] numbering we
    attach to turn 2's retrieved context. Stripping/rewriting the history avoids
    the model confusing which source "[1]" currently refers to.

    Behavior:
      - If `sources` is provided and has `sources[N-1]`: replace [N] with a neutral
        tag derived from that source's metadata.
      - If `sources` is None/empty: strip all [N] markers (just remove them), so the
        prose survives intact without misleading numbers.
      - If N is out of range for the given sources: replace with '' (drop it) — we
        never want to crash or expose a bogus reference to the model.
      - Matches both [1] and [12] via a 1-2 digit class; larger numbers are left
        untouched (e.g. [2024] stays as-is, which is the correct behavior).
    """
    if not content:
        return content

    # No sources → strip all markers cleanly, collapse any double-space left behind.
    if not sources:
        stripped = _CITATION_PATTERN.sub("", content)
        # Tidy up doubled spaces that a removed " [3]" would leave: " ." → "." etc.
        stripped = re.sub(r"\s+([.,;:!?])", r"\1", stripped)
        stripped = re.sub(r"[ \t]{2,}", " ", stripped)
        return stripped

    def _replace(match: re.Match) -> str:
        try:
            idx = int(match.group(1)) - 1
        except ValueError:
            return ""
        if idx < 0 or idx >= len(sources):
            # Out of range → drop the marker rather than lie about its referent.
            return ""
        src = sources[idx]
        # Sources may arrive as dicts (from API history payloads) or as objects
        # with a .metadata attr; be tolerant of both shapes.
        if isinstance(src, dict):
            # A serialized source dict from chat history: we have fields like
            # filename, page, category — fold them into a metadata-like mapping.
            meta = {
                "doc_type": src.get("doc_type") or src.get("category"),
                "category": src.get("category"),
                "page": src.get("page"),
                "rel_path": src.get("rel_path") or src.get("filename"),
                "source": src.get("filename"),
            }
        else:
            meta = getattr(src, "metadata", {}) or {}
        return _neutral_ref_for(meta)

    rewritten = _CITATION_PATTERN.sub(_replace, content)
    # Clean up any resulting whitespace-before-punctuation.
    rewritten = re.sub(r"\s+([.,;:!?])", r"\1", rewritten)
    rewritten = re.sub(r"[ \t]{2,}", " ", rewritten)
    return rewritten


# ---------------------------------------------------------------------------
# History formatting — session history → OpenAI messages list
# ---------------------------------------------------------------------------

# Per-turn content cap (chars, not tokens — rough but sufficient guardrail).
# Keeps a single long assistant turn from eating the whole context budget.
HISTORY_TURN_CHAR_CAP = 400


def _truncate(text: str, cap: int = HISTORY_TURN_CHAR_CAP) -> str:
    """Truncate a turn's content, appending an ellipsis if we cut it."""
    if text is None:
        return ""
    if len(text) <= cap:
        return text
    # Cut to cap-1 so the combined result is <= cap including the ellipsis char.
    return text[: cap - 1].rstrip() + "…"


def format_history_for_prompt(history: list) -> list:
    """
    Convert session history into an OpenAI-format messages list, ready to be
    inserted between the system prompt and the current user turn.

    `history` items look like:
      {"role": "user"|"assistant", "content": str, "sources"?: list}

    For assistant turns, `sources` (if present) is used by `_neutralize_citations`
    to replace [N] markers with neutral refs. If `sources` is missing, markers are
    stripped. User turns pass through unchanged (just truncated).

    Each turn's content is also truncated to HISTORY_TURN_CHAR_CAP chars so a
    single long turn doesn't blow the prompt budget.
    """
    if not history:
        return []

    messages: List[dict] = []
    for turn in history:
        role = turn.get("role")
        content = turn.get("content", "") or ""
        if role not in ("user", "assistant"):
            # Silently skip unknown roles rather than polluting the prompt.
            continue

        if role == "assistant":
            # Neutralize [N] markers before truncation so the cap doesn't slice
            # mid-marker and leave dangling "[" in the history.
            content = _neutralize_citations(content, turn.get("sources"))

        content = _truncate(content)
        messages.append({"role": role, "content": content})

    return messages
