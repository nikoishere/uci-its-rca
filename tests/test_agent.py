"""Tests for rca.agent — LLM-powered RCA agent (OpenAI mocked)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from rca.agent import RCAAgent
from rca.metrics import MetricsTracker
from rca.models import LogParseResult, RAGCitation, RCAOutput, YAMLContext
from rca.rate_limiter import TokenBucket


def _make_parse_result(found: bool = True) -> LogParseResult:
    if not found:
        return LogParseResult(found=False, log_path="/tmp/clean.log")
    return LogParseResult(
        found=True,
        failure_type="Python Traceback",
        failure_line="KeyError: 'trip_mode'",
        context_window=["line1", "line2", "KeyError: 'trip_mode'"],
        stack_trace="Traceback (most recent call last):\n  ...\nKeyError: 'trip_mode'",
        log_path="/tmp/run.log",
    )


def _mock_rca_output() -> RCAOutput:
    return RCAOutput(
        failure_summary="KeyError in tour_mode_choice",
        root_cause="Column trip_mode missing from tours table.",
        affected_component="tour_mode_choice.py",
        suggested_fixes=["Add trip_mode column"],
        config_issues=[],
        confidence="HIGH",
    )


class TestRCAAgent:
    @patch("rca.agent.OpenAI")
    def test_analyze_success(self, mock_openai_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 500
        mock_usage.completion_tokens = 200

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.parsed = _mock_rca_output()
        mock_response.usage = mock_usage
        mock_client.beta.chat.completions.parse.return_value = mock_response

        agent = RCAAgent(
            metrics_tracker=MetricsTracker(),
            rate_limiter=TokenBucket(rate_per_minute=600),
        )
        report = agent.analyze(_make_parse_result())

        assert report.confidence == "HIGH"
        assert "trip_mode" in report.root_cause
        assert report.log_path == "/tmp/run.log"

    @patch("rca.agent.OpenAI")
    def test_analyze_no_failure(self, mock_openai_cls: MagicMock) -> None:
        agent = RCAAgent()
        report = agent.analyze(_make_parse_result(found=False))
        assert "No failure signal" in report.failure_summary
        assert report.confidence == "LOW"

    @patch("rca.agent.OpenAI")
    def test_analyze_llm_failure_returns_fallback(self, mock_openai_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.beta.chat.completions.parse.side_effect = Exception("API down")

        agent = RCAAgent(
            metrics_tracker=MetricsTracker(),
            rate_limiter=TokenBucket(rate_per_minute=600),
        )
        report = agent.analyze(_make_parse_result())

        assert "unavailable" in report.failure_summary.lower() or "API down" in report.failure_summary
        assert report.confidence == "LOW"

    @patch("rca.agent.OpenAI")
    def test_analyze_with_yaml_and_rag(self, mock_openai_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 800
        mock_usage.completion_tokens = 300

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.parsed = _mock_rca_output()
        mock_response.usage = mock_usage
        mock_client.beta.chat.completions.parse.return_value = mock_response

        yaml_ctxs = [
            YAMLContext(
                config_path="settings.yaml",
                relevant_keys={"chunk_size": 0},
                raw_snippet="chunk_size: 0",
            )
        ]
        rag_cits = [
            RAGCitation(
                doc_id="abc",
                title="OOM Runbook",
                doc_type="runbook",
                snippet="Reduce chunk size...",
                s3_key="runbook/abc.md",
                similarity=0.9,
            )
        ]

        agent = RCAAgent(
            metrics_tracker=MetricsTracker(),
            rate_limiter=TokenBucket(rate_per_minute=600),
        )
        report = agent.analyze(_make_parse_result(), yaml_ctxs, rag_cits)

        assert len(report.rag_citations) == 1
        assert report.rag_citations[0].title == "OOM Runbook"
