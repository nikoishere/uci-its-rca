"""
Latency and token-cost tracking for every LLM / embedding API call.

Pricing table is approximate as of early 2026; update _PRICING when
OpenAI publishes new rates.
"""
from __future__ import annotations

from rca.models import CallMetrics, SessionMetrics

# USD per 1 million tokens
_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o":                   {"input": 2.50,  "output": 10.00},
    "gpt-4o-mini":              {"input": 0.15,  "output": 0.60},
    "text-embedding-3-small":   {"input": 0.02,  "output": 0.00},
    "text-embedding-3-large":   {"input": 0.13,  "output": 0.00},
}
_DEFAULT_PRICING: dict[str, float] = {"input": 2.50, "output": 10.00}


def compute_cost(
    model: str, prompt_tokens: int, completion_tokens: int
) -> float:
    """Return the estimated USD cost for a single API call."""
    p = _PRICING.get(model, _DEFAULT_PRICING)
    return (prompt_tokens * p["input"] + completion_tokens * p["output"]) / 1_000_000


class MetricsTracker:
    """Accumulates CallMetrics across the lifetime of one CLI invocation."""

    def __init__(self) -> None:
        self._session = SessionMetrics()

    def record(self, metrics: CallMetrics) -> None:
        self._session.calls.append(metrics)

    def summary(self) -> dict:
        return self._session.summary()

    @property
    def session(self) -> SessionMetrics:
        return self._session
