"""
Satya web UI launcher.

    python -m UI.run          (from the repo root)
    python UI/run.py          (also fine — the repo root is added to sys.path)

The web UI imports the shared backend (services/, src/), so the repo root must be
importable and the .env at the repo root is the single source of configuration.
Equivalent: uvicorn UI.src.server:app --host 0.0.0.0 --port 8000
"""
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    import uvicorn
    from UI.src.server import app

    host = os.getenv("WEB_HOST", "0.0.0.0")
    port = int(os.getenv("WEB_PORT", "8000"))

    print("🔍 Satya web UI")
    print(f"   → http://localhost:{port}")
    print("   Press Ctrl+C to stop.")

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
