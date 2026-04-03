"""
PostgreSQL + pgvector schema management.

Run `python main.py init-db` (or `init_db()` directly) once after
`docker-compose up -d` to create the extension and table.

The ivfflat index uses cosine distance, which matches the normalised
embeddings produced by text-embedding-3-small.
"""

from __future__ import annotations

import psycopg2
from pgvector.psycopg2 import register_vector
from psycopg2.extensions import connection as PGConnection

from config import settings

# lists=100 is appropriate for datasets up to ~1 M rows.
# Bump to sqrt(n_rows) if the knowledge base grows much larger.
_DDL = f"""\
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS knowledge_base (
    id          SERIAL      PRIMARY KEY,
    doc_id      TEXT        NOT NULL,
    doc_type    TEXT        NOT NULL,
    title       TEXT        NOT NULL,
    chunk_idx   INTEGER     NOT NULL,
    content     TEXT        NOT NULL,
    s3_key      TEXT        NOT NULL,
    embedding   vector({settings.EMBEDDING_DIM}),
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS knowledge_base_embedding_idx
    ON knowledge_base
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
"""


def get_connection() -> PGConnection:
    """Open a psycopg2 connection with the pgvector adapter registered."""
    conn = psycopg2.connect(settings.DATABASE_URL)
    register_vector(conn)
    return conn


def init_db() -> None:
    """Create the pgvector extension, table, and index (idempotent)."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(_DDL)
        conn.commit()
    finally:
        conn.close()
