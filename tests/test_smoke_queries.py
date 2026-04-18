"""
End-to-end smoke tests for the RAG chain.

Runs 6 golden queries (2 per course × 3 courses) through
`RAGChain.answer_stream_async` and asserts the response is a real, cited
answer — not a stub, not a refusal, not a mis-categorized exam citation
for a conceptual query.

Requires:
  - LM Studio reachable at LMSTUDIO_BASE_URL (default http://localhost:1234/v1)
  - ChromaDB populated at CHROMA_PERSIST_DIR (default ./data/chroma)

If LM Studio is down, the smoke tests SKIP rather than fail — that's a
local-environment concern, not a correctness regression.

The file is pytest-compatible but also runnable directly as a script,
because the project's requirements.txt does not currently ship pytest.
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

# Ensure project root is importable when run via `python tests/...`
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env")
except Exception:
    pass

from backend.generation.chain import RAGChain  # noqa: E402
from backend.generation.llm import check_health  # noqa: E402

# ---------------------------------------------------------------------------
# 6 golden queries — 2 per course.
#
# ELEC 477 selection rationale: the vault has ~50 ELEC 477 lecture PDFs on
# distributed-systems topics (RPC, replication, RAFT, naming, DDS, real-time
# scheduling, coordination). "remote procedure call" and "primary-backup
# replication" are core concepts with multiple dedicated lectures AND
# assignments/labs — high-confidence retrieval expected.
# ---------------------------------------------------------------------------

GOLDEN_QUERIES = [
    # ELEC 472 — Artificial Intelligence
    ("Explain the A* search algorithm", "ELEC 472"),
    ("What are the core steps in data preprocessing for machine learning?", "ELEC 472"),
    # CMPE 223 — Software Specification
    ("What is a context-free grammar?", "CMPE 223"),
    ("Explain regular expressions and their use in lexical analysis", "CMPE 223"),
    # ELEC 477 — Distributed Systems
    ("What is remote procedure call (RPC) and how does it work?", "ELEC 477"),
    ("Explain primary-backup replication in distributed systems", "ELEC 477"),
]

# Category (folder name) → doc_type (normalized). Source dataclass exposes
# `category` but not `doc_type`, so we normalize here for the assertion.
_CATEGORY_TO_DOCTYPE = {
    "lectures": "lecture",
    "labs": "lab",
    "exams": "exam",
    "assignments": "assignment",
    "resources": "resource",
}


def _normalize_doc_type(category: str) -> str:
    if not category:
        return ""
    return _CATEGORY_TO_DOCTYPE.get(category.strip().lower(), category.strip().lower())


def _is_conceptual(q: str) -> bool:
    low = q.lower()
    return any(s in low for s in ("explain", "what is", "what are", "how does"))


async def _collect_answer(chain: RAGChain, question: str, course: str):
    """Run one query through the async streaming chain and return
    (answer_text, sources, top_source_doc_type)."""
    token_stream, sources = await chain.answer_stream_async(
        question=question,
        course=course,
        top_k=10,
    )

    chunks = []
    async for tok in token_stream:
        chunks.append(tok)
    answer = "".join(chunks)

    top_source_doc_type = ""
    if sources:
        top_source_doc_type = _normalize_doc_type(sources[0].category)

    return answer, sources, top_source_doc_type


def _count_citation_markers(text: str) -> int:
    """Count [N] markers where 1 <= N <= 9. Matches the assertion in the spec."""
    import re
    return len(re.findall(r"\[[1-9]\]", text))


def _lm_studio_up() -> bool:
    health = check_health()
    return health.get("status") == "ok"


# ---------------------------------------------------------------------------
# Test: all 6 golden queries
# ---------------------------------------------------------------------------

def test_golden_smoke_queries():
    """Run each golden query and verify the response quality.

    Soft-pass conditions (per plan Task 10 & AC #11):
      - Good retrieval: response > 300 chars + >= 1 [N] marker +
        top source is lecture/resource for conceptual queries.
      - Weak/empty retrieval: response is the empty-state template
        (detected heuristically by 'I don't see' / 'closest matches')
        → reported as SKIPPED(empty-state), not a failure.
    """
    if not _lm_studio_up():
        _emit("SKIP (LM Studio not reachable) — cannot run smoke queries")
        return

    chain = RAGChain()
    failures = []

    for question, course in GOLDEN_QUERIES:
        try:
            answer, sources, top_dt = asyncio.run(
                _collect_answer(chain, question, course)
            )
        except Exception as exc:  # pragma: no cover
            failures.append(f"{question!r}: exception {type(exc).__name__}: {exc}")
            _emit(f"ERROR  {question[:40]!r:42s} | {type(exc).__name__}: {exc}")
            continue

        markers = _count_citation_markers(answer)

        # Detect empty-state template — it starts with "I don't see" per
        # backend/generation/prompts.py::EMPTY_STATE_TEMPLATE.
        is_empty_state = "i don't see" in answer.lower() and "closest matches" in answer.lower()

        top_label = ""
        if sources:
            fn = sources[0].filename
            pg = sources[0].page
            top_label = f"{top_dt} ({fn}{' p.'+pg if pg else ''})"

        short_q = question if len(question) <= 40 else question[:37] + "..."

        if is_empty_state:
            # AC allows this — the vault may not cover a topic; emit as skipped.
            _emit(f"SKIP(empty-state)  {short_q!r:42s} | {len(answer):4d} chars | top={top_label}")
            continue

        # Real answer path — run the hard assertions.
        problems = []
        if len(answer) <= 300:
            problems.append(f"len={len(answer)} <= 300 (stub/short answer)")
        if markers < 1:
            problems.append(f"markers={markers} (< 1 required [N])")
        if _is_conceptual(question):
            if not sources:
                problems.append("no sources returned for conceptual query")
            elif top_dt in {"exam", "assignment", "lab"}:
                problems.append(
                    f"top source doc_type={top_dt!r} — expected lecture/resource. "
                    "Doc-type boost may not have fired."
                )

        if problems:
            failures.append(f"{question!r}: " + "; ".join(problems))
            _emit(
                f"FAIL  {short_q!r:42s} | {len(answer):4d} chars | "
                f"{markers} markers | top={top_label} | {'; '.join(problems)}"
            )
        else:
            _emit(
                f"PASS  {short_q!r:42s} | {len(answer):4d} chars | "
                f"{markers} markers | source={top_label}"
            )

    assert not failures, (
        "Golden smoke queries failed:\n  - " + "\n  - ".join(failures)
    )


# ---------------------------------------------------------------------------
# Test: multi-turn session memory
# ---------------------------------------------------------------------------

def test_multi_turn_memory():
    """Turn-2 'What's its time complexity?' with 'its' referring to A*
    should produce an answer that references A* / heuristic search /
    branching factor — proving the chain forwarded prior context.

    We do NOT monkey-patch the LLM client to inspect the neutralized prompt
    — that would require invasive surgery in a module we're not allowed to
    modify. Citation neutralization is verified separately by the unit tests
    in tests/test_neutralize_citations.py (AC #12).
    """
    if not _lm_studio_up():
        _emit("SKIP (LM Studio not reachable) — cannot run multi-turn memory test")
        return

    chain = RAGChain()

    # Turn 1: establish the topic.
    # NOTE: use the full form ("Explain the A* search algorithm") not "Explain A*".
    # The short form scores top_sim=0.792 with overlap=0 against the current index
    # and classifies as 'weak' — a known limitation of short/ambiguous queries
    # with this embedding model. The longer form classifies 'good' and is what a
    # NotebookLM-style user would actually type.
    turn1_answer, turn1_sources, _ = asyncio.run(
        _collect_answer(chain, "Explain the A* search algorithm", "ELEC 472")
    )

    # If turn 1 came back empty-state, the vault just doesn't have A*
    # content — skip the follow-up (not a regression).
    if "i don't see" in turn1_answer.lower():
        _emit("SKIP(empty-state) multi-turn: turn-1 'Explain A*' classified weak/empty")
        return

    # Build history in the shape the chain expects
    # (chat.py serializes Source -> dict with filename/page/category/etc.)
    history_sources = [
        {
            "filename": s.filename,
            "page": s.page,
            "course": s.course,
            "category": s.category,
            "relevance_score": s.relevance_score,
            "text_preview": s.text_preview,
        }
        for s in turn1_sources
    ]
    history = [
        {"role": "user", "content": "Explain A*"},
        {"role": "assistant", "content": turn1_answer, "sources": history_sources},
    ]

    # Turn 2: pronoun-ambiguous question — must resolve "its" via history.
    async def _turn2():
        token_stream, _ = await chain.answer_stream_async(
            question="What's its time complexity?",
            course="ELEC 472",
            top_k=10,
            history=history,
        )
        parts = []
        async for tok in token_stream:
            parts.append(tok)
        return "".join(parts)

    turn2_answer = asyncio.run(_turn2())

    low = turn2_answer.lower()
    referent_signals = ("a*", "heuristic search", "branching factor", "heuristic")
    hit = any(sig in low for sig in referent_signals)

    _emit(
        f"multi-turn: len(turn2)={len(turn2_answer)}  "
        f"signals_hit={[s for s in referent_signals if s in low]}"
    )

    assert hit, (
        "Multi-turn memory test: turn-2 answer did not reference A* / "
        "heuristic search / branching factor — chain may not be forwarding "
        f"history. Turn-2 answer starts with: {turn2_answer[:200]!r}"
    )


# ---------------------------------------------------------------------------
# Small helper to emit status both under pytest (-s) and when run directly.
# ---------------------------------------------------------------------------

def _emit(line: str) -> None:
    print(line, flush=True)


# ---------------------------------------------------------------------------
# Plain-Python runner (pytest not in requirements.txt yet).
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    import traceback

    tests = [test_golden_smoke_queries, test_multi_turn_memory]
    failed = 0
    for t in tests:
        print(f"\n=== {t.__name__} ===")
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
        except Exception:
            failed += 1
            print(f"ERROR {t.__name__}:")
            traceback.print_exc()
    sys.exit(1 if failed else 0)
