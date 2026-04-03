"""
Central settings loaded from environment variables.
Copy .env.example to .env and fill in your values before running.
"""

from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv()

# ── OpenAI ────────────────────────────────────────────────────────────────────
OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o")
OPENAI_TIMEOUT_S: float = float(os.getenv("OPENAI_TIMEOUT_S", "60"))
OPENAI_MAX_RETRIES: int = int(os.getenv("OPENAI_MAX_RETRIES", "3"))

# ── Rate limiting ─────────────────────────────────────────────────────────────
RPM_LIMIT: int = int(os.getenv("RPM_LIMIT", "10"))

# ── Database ──────────────────────────────────────────────────────────────────
DATABASE_URL: str = os.getenv(
    "DATABASE_URL", "postgresql://rca:rca@localhost:5432/rca_db"
)

# ── AWS S3 ────────────────────────────────────────────────────────────────────
AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET: str = os.getenv("S3_BUCKET", "its-rca-knowledge-base")

# ── Log parser ────────────────────────────────────────────────────────────────
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "8192"))
CONTEXT_LINES: int = int(os.getenv("CONTEXT_LINES", "50"))

# ── Embeddings ────────────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", "1536"))
