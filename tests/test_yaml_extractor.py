"""Tests for rca.yaml_extractor — YAML config surfacing."""
from __future__ import annotations

from pathlib import Path

import pytest

from rca.yaml_extractor import YAMLExtractor


class TestYAMLExtractor:
    def test_extract_finds_sample_configs(self, config_dir: Path) -> None:
        results = YAMLExtractor().extract(config_dir)
        assert len(results) >= 2
        names = [Path(r.config_path).name for r in results]
        assert "settings.yaml" in names
        assert "network_los.yaml" in names

    def test_priority_ordering(self, config_dir: Path) -> None:
        """settings.yaml should come before network_los.yaml."""
        results = YAMLExtractor().extract(config_dir)
        names = [Path(r.config_path).name for r in results]
        assert names.index("settings.yaml") < names.index("network_los.yaml")

    def test_debug_keys_surfaced(self, config_dir: Path) -> None:
        results = YAMLExtractor().extract(config_dir)
        settings_ctx = next(
            r for r in results if Path(r.config_path).name == "settings.yaml"
        )
        assert "chunk_size" in settings_ctx.relevant_keys
        assert "num_processes" in settings_ctx.relevant_keys
        assert "households_sample_size" in settings_ctx.relevant_keys

    def test_log_context_matches_extra_keys(self, config_dir: Path) -> None:
        """Keys mentioned in the log context should be surfaced too."""
        log_context = ["ERROR in zone_system configuration"]
        results = YAMLExtractor().extract(config_dir, log_context=log_context)
        network_ctx = next(
            r for r in results if Path(r.config_path).name == "network_los.yaml"
        )
        assert "zone_system" in network_ctx.relevant_keys

    def test_nonexistent_dir_returns_empty(self, tmp_path: Path) -> None:
        results = YAMLExtractor().extract(tmp_path / "no_such_dir")
        assert results == []

    def test_invalid_yaml_skipped(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text(": : : not valid yaml [[[\n")
        results = YAMLExtractor().extract(tmp_path)
        # Should not crash; the bad file is simply skipped
        assert isinstance(results, list)

    def test_snippet_truncation(self, config_dir: Path) -> None:
        extractor = YAMLExtractor(max_snippet_chars=50)
        results = extractor.extract(config_dir)
        for r in results:
            assert len(r.raw_snippet) <= 50
