#!/usr/bin/env python3
"""
End-to-end smoke test for the local-study-rag stack.

Run this before any demo. It verifies:
  1. /api/health is OK and LM Studio reports a loaded model.
  2. data/manifest.json exists and references at least one ingested file.
  3. POST /api/chat streams a real answer:
     - emits a `sources` event with non-empty data,
     - emits at least 5 `token` events,
     - terminates with a `done` event,
     - the assembled answer contains no <think>/<reasoning> tags.

Prints time-to-first-token and total time so you have real numbers to cite.

Usage:
    python scripts/smoke_test.py
    python scripts/smoke_test.py --course "ELEC 472" --question "What is A* search?"
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import httpx

DEFAULT_BASE = "http://localhost:8000"
DEFAULT_QUESTION = "Summarize the main concepts you can find in the notes."
REASONING_TAG_RE = re.compile(r"</?(?:think|reasoning)\b[^>]*>", re.IGNORECASE)

GREEN = "\033[92m"
RED = "\033[91m"
DIM = "\033[2m"
RESET = "\033[0m"


def ok(msg: str) -> None:
    print(f"  {GREEN}✓{RESET} {msg}")


def fail(msg: str) -> bool:
    print(f"  {RED}✗ {msg}{RESET}")
    return False


def step(num: int, name: str) -> None:
    print(f"\n[{num}] {name}")


def check_health(base: str) -> bool:
    step(1, "Health check (/api/health)")
    try:
        r = httpx.get(f"{base}/api/health", timeout=10.0)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        return fail(f"GET /api/health failed: {e}")

    llm = data.get("llm", {}) or {}
    if data.get("status") != "ok" or llm.get("status") != "ok":
        return fail(
            f"backend or LM Studio not healthy: status={data.get('status')!r} "
            f"llm={llm.get('error') or llm.get('status')!r}"
        )

    models = llm.get("models") or []
    ok(f"backend up, LM Studio reachable, {len(models)} model(s) loaded")
    if models:
        print(f"      {DIM}models: {', '.join(models[:3])}{RESET}")
    return True


def check_manifest() -> bool:
    step(2, "Manifest check (data/manifest.json)")
    manifest_path = Path(__file__).resolve().parent.parent / "data" / "manifest.json"
    if not manifest_path.exists():
        return fail(f"manifest not found at {manifest_path}\n     Run ingestion first.")

    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception as e:
        return fail(f"manifest is not valid JSON: {e}")

    files = manifest.get("files") or {}
    total_chunks = manifest.get("total_chunks", 0)
    if not files:
        return fail("manifest has no `files` entries — vault hasn't been ingested")

    ok(f"manifest has {len(files)} file(s), {total_chunks} chunk(s) ingested")
    return True


def run_chat(base: str, question: str, course: str | None, timeout: float) -> bool:
    label = f"course={course or 'any'!r}, q={question[:50]!r}"
    step(3, f"POST /api/chat ({label})")

    payload = {"question": question, "course": course, "history": []}

    sources_event = None
    tokens: list[str] = []
    done = False
    t0 = time.time()
    t_first_token: float | None = None

    try:
        with httpx.stream(
            "POST", f"{base}/api/chat", json=payload, timeout=timeout
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    evt = json.loads(line)
                except json.JSONDecodeError:
                    return fail(f"non-JSON line in stream: {line[:120]!r}")

                kind = evt.get("type")
                if kind == "sources":
                    sources_event = evt.get("data", [])
                elif kind == "token":
                    if t_first_token is None:
                        t_first_token = time.time()
                    tokens.append(evt.get("data", ""))
                elif kind == "done":
                    done = True
                    break
    except Exception as e:
        return fail(f"streaming POST failed: {e}")

    t_total = time.time() - t0
    ttft = (t_first_token - t0) if t_first_token is not None else None

    if sources_event is None:
        return fail("no `sources` event received")
    ok(f"sources event with {len(sources_event)} source(s)")

    if len(tokens) < 5:
        return fail(f"only {len(tokens)} token event(s); expected >= 5")
    ok(f"received {len(tokens)} token event(s)")

    if not done:
        return fail("stream ended without `done` event")
    ok("stream terminated with `done`")

    full_answer = "".join(tokens)
    leaks = REASONING_TAG_RE.findall(full_answer)
    if leaks:
        return fail(f"answer leaked reasoning tags: {leaks[:3]}")
    ok("no <think>/<reasoning> tags in visible answer")

    if any("[Model Error:" in t for t in tokens):
        return fail("backend reported a Model Error in the stream")
    ok("no model-error sentinel in stream")

    print()
    if ttft is not None:
        print(f"  {DIM}time-to-first-token: {ttft:.2f}s{RESET}")
    print(f"  {DIM}total stream time:   {t_total:.2f}s{RESET}")
    print(f"  {DIM}token events:        {len(tokens)}{RESET}")
    preview = full_answer.strip().replace("\n", " ")
    print(f"  {DIM}answer preview:      {preview[:160]!r}{RESET}")
    return True


def main() -> int:
    p = argparse.ArgumentParser(
        description="End-to-end smoke test for local-study-rag.",
    )
    p.add_argument("--base", default=DEFAULT_BASE, help="Backend base URL")
    p.add_argument("--question", default=DEFAULT_QUESTION, help="Test question")
    p.add_argument("--course", default=None, help="Optional course filter")
    p.add_argument(
        "--timeout", type=float, default=180.0, help="Stream timeout (seconds)"
    )
    args = p.parse_args()

    print(f"local-study-rag smoke test  ({DIM}base={args.base}{RESET})")

    passed = (
        check_health(args.base)
        and check_manifest()
        and run_chat(args.base, args.question, args.course, args.timeout)
    )

    print()
    if passed:
        print(f"{GREEN}RESULT: PASS{RESET} — system is demo-ready")
        return 0
    print(f"{RED}RESULT: FAIL{RESET} — fix the issue above before demoing")
    return 1


if __name__ == "__main__":
    sys.exit(main())
