"""
Extracts YAML configuration context from an ActivitySim run directory.

ActivitySim runs are driven by several YAML files (settings.yaml,
network_los.yaml, etc.).  When a simulation fails, the config values
in those files often hold clues — wrong paths, missing keys, bad chunk
sizes, etc.  This module surfaces the most relevant keys without
dumping the entire config into the LLM prompt.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml

from rca.models import YAMLContext

# Checked before all other YAML files; order matters for prompt placement.
_PRIORITY_NAMES = [
    "settings.yaml",
    "network_los.yaml",
    "sharrow.yaml",
    "chunk_and_resume.yaml",
    "settings.yml",
]

# Keys that are almost always worth surfacing regardless of the failure.
_DEBUG_KEYS = {
    "chunk_size",
    "num_processes",
    "multiprocess",
    "memory_limit",
    "input_table_list",
    "data_dir",
    "output_dir",
    "resume_after",
    "households_sample_size",
    "trace_hh_id",
    "sharrow",
    "chunk_training_mode",
}

_MAX_SNIPPET_CHARS = 2_000


class YAMLExtractor:
    def __init__(self, max_snippet_chars: int = _MAX_SNIPPET_CHARS) -> None:
        self._max = max_snippet_chars

    # ── Public ────────────────────────────────────────────────────────────────

    def extract(
        self,
        config_dir: Path,
        log_context: Optional[list[str]] = None,
    ) -> list[YAMLContext]:
        """
        Discover YAML files in config_dir and return parsed contexts ordered
        by relevance (priority files first, then alphabetical).
        """
        config_dir = Path(config_dir)
        if not config_dir.is_dir():
            return []

        yaml_files = self._discover(config_dir)
        results: list[YAMLContext] = []
        for path in yaml_files:
            ctx = self._parse(path, log_context)
            if ctx is not None:
                results.append(ctx)
        return results

    # ── Private ───────────────────────────────────────────────────────────────

    def _discover(self, config_dir: Path) -> list[Path]:
        all_files = list(config_dir.glob("*.yaml")) + list(config_dir.glob("*.yml"))
        priority_set = {n.lower() for n in _PRIORITY_NAMES}
        priority: list[Path] = []
        rest: list[Path] = []
        for p in all_files:
            (priority if p.name.lower() in priority_set else rest).append(p)
        # Sort priority by the canonical order, rest alphabetically.
        priority.sort(
            key=lambda p: (
                _PRIORITY_NAMES.index(p.name.lower())
                if p.name.lower() in _PRIORITY_NAMES
                else 99
            )
        )
        return priority + sorted(rest)

    def _parse(
        self,
        yaml_path: Path,
        log_context: Optional[list[str]],
    ) -> Optional[YAMLContext]:
        try:
            raw = yaml_path.read_text(errors="replace")
            data = yaml.safe_load(raw)
        except Exception:
            return None

        if not isinstance(data, dict):
            return None

        relevant = self._relevant_keys(data, log_context)
        snippet = self._build_snippet(yaml_path, relevant, raw)
        return YAMLContext(
            config_path=str(yaml_path),
            relevant_keys=relevant,
            raw_snippet=snippet,
        )

    def _relevant_keys(
        self,
        data: dict,
        log_context: Optional[list[str]],
    ) -> dict:
        result: dict = {}

        for key in _DEBUG_KEYS:
            if key in data:
                result[key] = data[key]

        if log_context:
            log_text = " ".join(log_context)
            for key, val in data.items():
                if isinstance(key, str) and key in log_text and key not in result:
                    result[key] = val

        return result

    def _build_snippet(self, yaml_path: Path, relevant: dict, raw: str) -> str:
        header = f"# {yaml_path.name}\n"
        body = (
            yaml.dump(relevant, default_flow_style=False, allow_unicode=True)
            if relevant
            else raw
        )
        return (header + body)[: self._max]
