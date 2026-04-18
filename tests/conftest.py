"""
Shared pytest setup: make the project root importable so tests can do
`from backend...` and load the project's .env like the real app does.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path so `backend` imports work.
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Load .env exactly the way backend/main.py does, so LM Studio URL,
# CHROMA_PERSIST_DIR, model IDs etc. are available when the chain is
# constructed during tests.
try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env")
except Exception:
    pass
