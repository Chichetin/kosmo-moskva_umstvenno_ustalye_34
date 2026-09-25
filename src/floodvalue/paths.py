"""Filesystem locations shared by all stages. Data never lives in git (docs/data.md)."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    return Path(os.environ.get("FLOODVALUE_DATA_DIR", REPO_ROOT / "data"))


def output_dir() -> Path:
    return Path(os.environ.get("FLOODVALUE_OUTPUT_DIR", REPO_ROOT / "outputs"))


CONFIGS_DIR = REPO_ROOT / "configs"
SPLITS_DIR = REPO_ROOT / "splits"
MODELS_DIR = REPO_ROOT / "models"
