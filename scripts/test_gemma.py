"""
Quick test script to verify Gemma4 works via Ollama.

Tests:
  1. Basic generation (non-streaming)
  2. Streaming generation (token-by-token)
  3. Structured JSON output (the pattern needed for knowledge graph extraction)

Usage:
  cd RAG_System
  source .venv/bin/activate
  python scripts/test_gemma.py
"""
import json
import sys
import time
import httpx

OLLAMA_BASE_URL = "http://localhost:11434"
MODEL = "gemma4:latest"  # swap to "gemma4:e2b" for the smaller 5B variant


def test_health():
    """Check Ollama is up and the model is available."""
    print("=" * 60)
    print("TEST 1: Ollama Health Check")
    print("=" * 60)
    try:
        r = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        r.raise_for_status()
        models = [m["name"] for m in r.json().get("models", [])]
        print(f"✅ Ollama is running. Available models: {models}")
        if MODEL not in models:
            print(f"⚠️  Warning: '{MODEL}' not in model list. It may fail.")
        return True
    except Exception as e:
        print(f"❌ Ollama is not reachable: {e}")
        return False


def test_basic_generation():
    """Non-streaming generation test."""
    print("\n" + "=" * 60)
    print("TEST 2: Basic Generation (non-streaming)")
    print("=" * 60)
    prompt = "Explain what a knowledge graph is in exactly 2 sentences."
    print(f"Prompt: {prompt}")
    print("-" * 40)

    start = time.time()
    r = httpx.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "keep_alive": "0",
            "options": {"temperature": 0.3, "num_predict": 256},
        },
        timeout=120,
    )
    r.raise_for_status()
    data = r.json()
    elapsed = time.time() - start

    response_text = data.get("response", "")
    eval_count = data.get("eval_count", 0)
    eval_duration_ns = data.get("eval_duration", 1)
    tokens_per_sec = eval_count / (eval_duration_ns / 1e9) if eval_duration_ns else 0

    print(f"Response:\n{response_text}")
    print(f"\n⏱  Total time: {elapsed:.1f}s")
    print(f"📊 Tokens generated: {eval_count}")
    print(f"🚀 Speed: {tokens_per_sec:.1f} tokens/sec")
    return True


def test_streaming():
    """Streaming generation test — prints tokens as they arrive."""
    print("\n" + "=" * 60)
    print("TEST 3: Streaming Generation")
    print("=" * 60)
    prompt = "List 3 benefits of using a knowledge graph for studying."
    print(f"Prompt: {prompt}")
    print("-" * 40)

    start = time.time()
    token_count = 0

    with httpx.Client(timeout=120) as client:
        with client.stream(
            "POST",
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": True,
                "keep_alive": "0",
                "options": {"temperature": 0.3, "num_predict": 512},
            },
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        token = data.get("response", "")
                        if token:
                            print(token, end="", flush=True)
                            token_count += 1
                        if data.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue

    elapsed = time.time() - start
    print(f"\n\n⏱  Total time: {elapsed:.1f}s | Tokens: {token_count}")
    return True


def test_json_extraction():
    """
    Test structured JSON output — this is what the knowledge graph
    extractor will rely on. We ask Gemma4 to extract entities and
    relations from a sample text chunk.
    """
    print("\n" + "=" * 60)
    print("TEST 4: Structured JSON Extraction (Knowledge Graph Pattern)")
    print("=" * 60)

    sample_chunk = (
        "Binary Search Trees (BST) are a fundamental data structure in computer "
        "science. Each node has at most two children, with the left subtree "
        "containing values less than the parent, and the right subtree containing "
        "values greater. The time complexity for search, insert, and delete "
        "operations is O(log n) for balanced trees, but degrades to O(n) in the "
        "worst case. AVL Trees and Red-Black Trees are self-balancing variants "
        "that guarantee O(log n) performance."
    )

    system_prompt = """You are a knowledge graph entity extractor. Given a text chunk, extract entities and relations.

Return ONLY valid JSON in this exact format (no markdown, no explanation):
{
  "entities": [
    {"name": "entity name", "type": "Concept|Algorithm|Theorem|Formula|Course", "description": "brief description"}
  ],
  "relations": [
    {"source": "entity1", "target": "entity2", "relation": "relationship description"}
  ]
}"""

    print(f"Sample chunk: {sample_chunk[:80]}...")
    print("-" * 40)

    start = time.time()
    r = httpx.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": MODEL,
            "prompt": sample_chunk,
            "system": system_prompt,
            "stream": False,
            "keep_alive": "0",
            "options": {"temperature": 0.1, "num_predict": 1024},
        },
        timeout=120,
    )
    r.raise_for_status()
    raw = r.json().get("response", "")
    elapsed = time.time() - start

    print(f"Raw response:\n{raw}")
    print(f"\n⏱  Time: {elapsed:.1f}s")

    # Try to parse it as JSON
    try:
        # Strip markdown fences if present
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = "\n".join(cleaned.split("\n")[1:])
        if cleaned.endswith("```"):
            cleaned = "\n".join(cleaned.split("\n")[:-1])
        parsed = json.loads(cleaned.strip())
        print(f"✅ Valid JSON! Entities: {len(parsed.get('entities', []))}, "
              f"Relations: {len(parsed.get('relations', []))}")
        print(json.dumps(parsed, indent=2))
        return True
    except json.JSONDecodeError as e:
        print(f"⚠️  JSON parse failed: {e}")
        print("   This is expected sometimes — the knowledge graph extractor "
              "uses retry logic to handle this.")
        return False


if __name__ == "__main__":
    print("🧪 Gemma4 Model Test Suite")
    print(f"   Model: {MODEL}")
    print(f"   Server: {OLLAMA_BASE_URL}")
    print()

    if not test_health():
        print("\n❌ Ollama is not running. Start it with: ollama serve")
        sys.exit(1)

    test_basic_generation()
    test_streaming()
    json_ok = test_json_extraction()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("✅ Health check passed")
    print("✅ Basic generation works")
    print("✅ Streaming works")
    print(f"{'✅' if json_ok else '⚠️ '} JSON extraction {'passed' if json_ok else 'needs retry logic'}")
    print("\nGemma4 is ready for knowledge graph extraction! 🎉")
