"""
Semantic retrieval from the pgvector knowledge base.

The query is constructed from the failure line and the tail of the
stack trace so the embedding captures both the error type and the
call-site context.
"""
from __future__ import annotations

from rca.models import LogParseResult, RAGCitation
from rag.db import get_connection
from rag.embedder import Embedder


class Retriever:
    def __init__(
        self,
        top_k: int = 4,
        min_similarity: float = 0.70,
    ) -> None:
        self._top_k = top_k
        self._min_similarity = min_similarity
        self._embedder = Embedder()

    def retrieve(self, parse_result: LogParseResult) -> list[RAGCitation]:
        """
        Embed the failure context and return the top-k knowledge-base
        chunks whose cosine similarity exceeds min_similarity.
        """
        query = self._build_query(parse_result)
        [embedding] = self._embedder.embed([query])

        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        doc_id,
                        doc_type,
                        title,
                        content,
                        s3_key,
                        1 - (embedding <=> %s) AS similarity
                    FROM knowledge_base
                    ORDER BY embedding <=> %s
                    LIMIT %s
                    """,
                    (embedding, embedding, self._top_k),
                )
                rows = cur.fetchall()
        finally:
            conn.close()

        citations: list[RAGCitation] = []
        for doc_id, doc_type, title, content, s3_key, similarity in rows:
            sim = float(similarity)
            if sim < self._min_similarity:
                continue
            citations.append(
                RAGCitation(
                    doc_id=doc_id,
                    title=title,
                    doc_type=doc_type,
                    snippet=content[:600],
                    s3_key=s3_key,
                    similarity=round(sim, 4),
                )
            )
        return citations

    def _build_query(self, parse_result: LogParseResult) -> str:
        parts = [parse_result.failure_line]
        if parse_result.stack_trace:
            # Use only the tail of the stack trace to avoid exceeding
            # the embedding model's token limit on very deep traces.
            parts.append(parse_result.stack_trace[-600:])
        return "\n".join(parts)
