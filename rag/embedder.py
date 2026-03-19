"""
Text chunking and embedding via the OpenAI Embeddings API.

Chunks use a small character-level overlap so sentences that straddle
a boundary are still findable regardless of which chunk is retrieved.
"""
from __future__ import annotations

import time

from openai import OpenAI

from config import settings
from rca.metrics import MetricsTracker, compute_cost
from rca.models import CallMetrics

_CHUNK_CHARS = 1_500
_OVERLAP_CHARS = 150


def chunk_text(
    text: str,
    size: int = _CHUNK_CHARS,
    overlap: int = _OVERLAP_CHARS,
) -> list[str]:
    """Split text into overlapping fixed-size character chunks."""
    chunks: list[str] = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + size])
        start += size - overlap
    return chunks


class Embedder:
    def __init__(
        self, metrics_tracker: MetricsTracker | None = None
    ) -> None:
        self._client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self._metrics = metrics_tracker or MetricsTracker()

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of texts and return a list of float vectors.

        The OpenAI Embeddings API accepts up to 2048 inputs per request,
        so this single call is fine for typical runbook / incident sizes.
        For very large ingestion jobs consider splitting into sub-batches.
        """
        if not texts:
            return []

        t0 = time.monotonic()
        response = self._client.embeddings.create(
            model=settings.EMBEDDING_MODEL,
            input=texts,
        )
        latency = time.monotonic() - t0

        self._metrics.record(
            CallMetrics(
                model=settings.EMBEDDING_MODEL,
                latency_s=round(latency, 3),
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=0,
                cost_usd=compute_cost(
                    settings.EMBEDDING_MODEL,
                    response.usage.prompt_tokens,
                    0,
                ),
            )
        )

        return [item.embedding for item in response.data]
