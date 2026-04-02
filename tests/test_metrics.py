"""Tests for rca.metrics — cost computation and session tracking."""
from __future__ import annotations

from rca.metrics import MetricsTracker, compute_cost
from rca.models import CallMetrics


class TestComputeCost:
    def test_gpt4o_cost(self) -> None:
        # 1M input tokens @ $2.50, 1M output tokens @ $10.00
        cost = compute_cost("gpt-4o", 1_000_000, 1_000_000)
        assert abs(cost - 12.50) < 0.01

    def test_embedding_cost(self) -> None:
        cost = compute_cost("text-embedding-3-small", 1_000, 0)
        assert cost == pytest.approx(0.00002, abs=1e-7)

    def test_unknown_model_uses_default(self) -> None:
        cost = compute_cost("unknown-model", 1_000_000, 0)
        assert cost == pytest.approx(2.50, abs=0.01)


class TestMetricsTracker:
    def test_record_and_summary(self) -> None:
        tracker = MetricsTracker()
        tracker.record(
            CallMetrics(
                model="gpt-4o",
                latency_s=1.5,
                prompt_tokens=500,
                completion_tokens=200,
                cost_usd=0.0033,
            )
        )
        tracker.record(
            CallMetrics(
                model="text-embedding-3-small",
                latency_s=0.3,
                prompt_tokens=100,
                completion_tokens=0,
                cost_usd=0.000002,
            )
        )
        s = tracker.summary()
        assert s["calls"] == 2
        assert s["prompt_tokens"] == 600
        assert s["completion_tokens"] == 200
        assert s["total_latency_s"] == 1.8

    def test_empty_session(self) -> None:
        tracker = MetricsTracker()
        s = tracker.summary()
        assert s["calls"] == 0
        assert s["estimated_cost_usd"] == 0.0

    def test_session_property(self) -> None:
        tracker = MetricsTracker()
        assert tracker.session.total_cost_usd == 0.0


import pytest  # noqa: E402 (needed for approx)
