"""Shared fixtures for all tests."""
from __future__ import annotations

from pathlib import Path

import pytest

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"
LOGS_DIR = SAMPLES_DIR / "logs"
CONFIGS_DIR = SAMPLES_DIR / "configs"


@pytest.fixture
def keyerror_log() -> Path:
    return LOGS_DIR / "keyerror_trip_mode.log"


@pytest.fixture
def oom_log() -> Path:
    return LOGS_DIR / "oom_zone_skims.log"


@pytest.fixture
def missing_file_log() -> Path:
    return LOGS_DIR / "missing_input_file.log"


@pytest.fixture
def clean_log() -> Path:
    return LOGS_DIR / "clean_run.log"


@pytest.fixture
def config_dir() -> Path:
    return CONFIGS_DIR
