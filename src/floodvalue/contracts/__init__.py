"""floodvalue.contracts — the only shared import between modules (D-005, docs/contracts.md).

- files: names, mandatory fields and raster specs of the 12 required files;
- schemas: pydantic v2 models of rows/documents and configs/run.yaml (D-004);
- geo: ROUND_HALF_UP and EPSG:6933 area;
- io: strict CSV/JSON readers (no NaN/Inf).
"""

from pathlib import Path

import yaml

from floodvalue.contracts.schemas import RunConfig
from floodvalue.paths import CONFIGS_DIR


def load_run_config(path: Path | None = None) -> RunConfig:
    """configs/run.yaml: demo chip, portfolio seed, budget, decision_at, decision_deadline."""
    with open(path or CONFIGS_DIR / "run.yaml", encoding="utf-8") as f:
        return RunConfig.model_validate(yaml.safe_load(f))
