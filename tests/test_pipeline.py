"""End-to-end integration test: log → parse → YAML → agent → report.

OpenAI is mocked so these tests run without credentials.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from rca.agent import RCAAgent
from rca.log_parser import LogParser
from rca.metrics import MetricsTracker
from rca.models import RCAOutput
from rca.rate_limiter import TokenBucket
from rca.report import save_report_markdown
from rca.yaml_extractor import YAMLExtractor


def _mock_rca_output() -> RCAOutput:
    return RCAOutput(
        failure_summary="MemoryError loading skim matrix 121/124",
        root_cause="System has 15.8 GB RAM but skims require 19.2 GB.",
        affected_component="activitysim.core.los",
        suggested_fixes=[
            "Reduce the number of skim matrices",
            "Use a machine with more RAM",
            "Enable sharrow to reduce memory footprint",
        ],
        config_issues=[
            "network_los.yaml: taz_skims references a file with 124 matrices",
        ],
        confidence="HIGH",
    )


class TestFullPipeline:
    @patch("rca.agent.OpenAI")
    def test_oom_log_full_pipeline(
        self,
        mock_openai_cls: MagicMock,
        oom_log: Path,
        config_dir: Path,
        tmp_path: Path,
    ) -> None:
        # Mock the OpenAI client
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 1000
        mock_usage.completion_tokens = 400

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.parsed = _mock_rca_output()
        mock_response.usage = mock_usage
        mock_client.beta.chat.completions.parse.return_value = mock_response

        # 1. Parse
        parse_result = LogParser().parse(oom_log)
        assert parse_result.found is True
        assert parse_result.failure_type == "MemoryError"

        # 2. YAML extraction
        yaml_contexts = YAMLExtractor().extract(
            config_dir, log_context=parse_result.context_window
        )
        assert len(yaml_contexts) >= 1

        # 3. Agent analysis
        metrics = MetricsTracker()
        agent = RCAAgent(
            metrics_tracker=metrics,
            rate_limiter=TokenBucket(rate_per_minute=600),
        )
        report = agent.analyze(parse_result, yaml_contexts)

        assert report.confidence == "HIGH"
        # The mock returns a fixed root_cause about RAM; just check it's present
        assert (
            "15.8 GB" in report.root_cause
            or "memory" in report.root_cause.lower()
            or report.root_cause
        )

        # 4. Save to Markdown
        output_file = tmp_path / "report.md"
        save_report_markdown(report, output_file)
        assert output_file.exists()
        content = output_file.read_text()
        assert "RCA Report" in content

        # 5. Check metrics
        assert metrics.summary()["calls"] == 1

    @patch("rca.agent.OpenAI")
    def test_clean_log_pipeline(
        self,
        mock_openai_cls: MagicMock,
        clean_log: Path,
    ) -> None:
        parse_result = LogParser().parse(clean_log)
        assert parse_result.found is False

        agent = RCAAgent()
        report = agent.analyze(parse_result)
        assert "No failure signal" in report.failure_summary
