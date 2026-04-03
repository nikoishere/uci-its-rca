"""
Document ingestion pipeline: S3 upload → chunk → embed → pgvector.

Usage (via CLI):
    python main.py ingest path/to/runbook.md --type runbook --title "OOM on zone-skims load"
    python main.py ingest path/to/incident.txt --type incident --title "KeyError trip_mode 2024-11"
"""

from __future__ import annotations

import uuid
from pathlib import Path

from rag.db import get_connection
from rag.embedder import Embedder, chunk_text
from rag.s3_store import S3Store


class Ingestor:
    def __init__(self) -> None:
        self._embedder = Embedder()
        self._s3 = S3Store()

    def ingest(self, file_path: Path, doc_type: str, title: str) -> str:
        """
        Full ingestion pipeline for a single document.

        Steps:
          1. Upload the original file to S3.
          2. Read, chunk, and embed the text.
          3. Persist each chunk + vector into the knowledge_base table.

        Returns the doc_id (a hex UUID) that groups all chunks for this doc.
        """
        doc_id = uuid.uuid4().hex

        # 1. Upload raw document to S3 for later full-text access.
        self._s3.ensure_bucket_exists()
        s3_key = self._s3.upload(file_path, doc_type, title)

        # 2. Chunk and embed.
        text = file_path.read_text(errors="replace")
        chunks = chunk_text(text)
        if not chunks:
            raise ValueError(f"Document is empty: {file_path}")
        embeddings = self._embedder.embed(chunks)

        # 3. Bulk-insert into pgvector.
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                    cur.execute(
                        """
                        INSERT INTO knowledge_base
                            (doc_id, doc_type, title, chunk_idx,
                             content, s3_key, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            doc_id,
                            doc_type,
                            title,
                            idx,
                            chunk,
                            s3_key,
                            embedding,
                        ),
                    )
            conn.commit()
        finally:
            conn.close()

        return doc_id
