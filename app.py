import os

if os.getenv("VERCEL"):
    os.environ.setdefault("SATYA_UPLOAD_DIR", "/tmp/satya-uploads")
    os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:////tmp/satya.db")

from UI.src.server import app
