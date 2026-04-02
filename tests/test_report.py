"""Tests for rca.report — Rich terminal + Markdown output."""
from __future__ import annotations

from pathlib import Path

from rca.models import RCAReport, RAGCitation, SessionMetrics, CallMetrics
from rca.report import print_report, save_report_markdown


def _make_report(**overrides) -> RCAReport:
    defaults = dict(
        log_path="/tmp/run.log",
        failure_summary="KeyError in tour_mode_choice",
        root_cause="Missing column trip_mode in the tours DataFrame.",
        affected_component="tour_mode_choice.py",
        stack_trace="Traceback (most recent call last):\n  ...\nKeyError: 'trip_mode'",
        suggested_fixes=["Add trip_mode column to tours table."],
        config_issues=["settings.yaml: households_sample_size may be too large"],
        rag_citations=[],
        confidence="HIGH",
    )
    defaults.update(overrides)
    return RCAReport(**defaults)


class TestPrintReport:
    def test_print_report_does_not_crash(self) -> None:
        """Smoke test — just make sure it runs without exceptions."""
        report = _make_report()
        print_report(report)

    def test_print_report_with_metrics(self) -> None:
        report = _make_report()
        metrics = SessionMetrics(
            calls=[
                CallMetrics(
                    model="gpt-4o",
                    latency_s=1.2,
                    prompt_tokens=400,
                    completion_tokens=150,
                    cost_usd=0.0025,
                )
            ]
        )
        print_report(report, metrics)

    def test_print_report_with_citations(self) -> None:
        report = _make_report(
            rag_citations=[
                RAGCitation(
                    doc_id="abc",
                    title="OOM Runbook",
                    doc_type="runbook",
                    snippet="When memory exceeds limit...",
                    s3_key="runbook/abc.md",
                    similarity=0.92,
                )
            ]
        )
        print_report(report)


class TestSaveReportMarkdown:
    def test_save_creates_file(self, tmp_path: Path) -> None:
        report = _make_report()
        out = tmp_path / "report.md"
        save_report_markdown(report, out)
        assert out.exists()
        content = out.read_text()
        assert "# ActivitySim RCA Report" in content
        assert "KeyError" in content
        assert "trip_mode" in content

    def test_save_includes_citations(self, tmp_path: Path) -> None:
        report = _make_report(
            rag_citations=[
                RAGCitation(
                    doc_id="x",
                    title="Past Incident",
                    doc_type="incident",
                    snippet="Resolved by restarting...",
                    s3_key="incident/x.txt",
                    similarity=0.85,
                )
            ]
        )
        out = tmp_path / "report_cit.md"
        save_report_markdown(report, out)
        content = out.read_text()
        assert "Past Incident" in content
        assert "incident" in content

    def test_save_includes_config_issues(self, tmp_path: Path) -> None:
        report = _make_report()
        out = tmp_path / "report_cfg.md"
        save_report_markdown(report, out)
        content = out.read_text()
        assert "Config Issues" in content
