"""Tests for rca.models — Pydantic data models."""
from __future__ import annotations

import json

from rca.models import (
    LogParseResult,
    RCAOutput,
    RCAReport,
    RAGCitation,
    SessionMetrics,
    YAMLContext,
)


class TestLogParseResult:
    def test_defaults(self) -> None:
        r = LogParseResult(found=False)
        assert r.failure_type == ""
        assert r.context_window == []
        assert r.stack_trace == ""

    def test_full(self) -> None:
        r = LogParseResult(
            found=True,
            failure_type="Python Traceback",
            failure_line="KeyError: 'trip_mode'",
            context_window=["line1", "line2"],
            stack_trace="Traceback...",
            log_path="/tmp/run.log",
        )
        assert r.found is True
        assert len(r.context_window) == 2


class TestRCAOutput:
    def test_schema_has_all_fields(self) -> None:
        schema = RCAOutput.model_json_schema()
        props = schema["properties"]
        assert "failure_summary" in props
        assert "root_cause" in props
        assert "suggested_fixes" in props
        assert "confidence" in props

    def test_json_roundtrip(self) -> None:
        output = RCAOutput(
            failure_summary="KeyError in tour_mode_choice",
            root_cause="Missing column trip_mode",
            affected_component="tour_mode_choice.py",
            suggested_fixes=["Add trip_mode column"],
            config_issues=[],
            confidence="HIGH",
        )
        data = json.loads(output.model_dump_json())
        restored = RCAOutput(**data)
        assert restored.confidence == "HIGH"


class TestRAGCitation:
    def test_creation(self) -> None:
        c = RAGCitation(
            doc_id="abc123",
            title="OOM Runbook",
            doc_type="runbook",
            snippet="When memory exceeds...",
            s3_key="runbook/abc.md",
            similarity=0.89,
        )
        assert c.similarity == 0.89


class TestSessionMetrics:
    def test_empty(self) -> None:
        s = SessionMetrics()
        assert s.total_cost_usd == 0.0
        assert s.total_latency_s == 0.0
        assert s.total_prompt_tokens == 0
        assert s.total_completion_tokens == 0
