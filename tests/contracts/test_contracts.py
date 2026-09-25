"""Shared arithmetic, schemas and configs/run.yaml of contracts v1."""

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest
import rasterio
from pydantic import ValidationError
from pyproj import Transformer
from shapely.geometry import Polygon

from floodvalue.contracts import files as F
from floodvalue.contracts import load_run_config
from floodvalue.contracts.geo import area_km2, product_half_up, round_half_up
from floodvalue.contracts.schemas import AssetLossRow, StrategyComparisonRow

HERE = Path(__file__).resolve().parent
GOOD = HERE.parent / "fixtures" / "good"


def _close(a, b) -> bool:
    """Structural equality with the same types; floats equal up to 1e-12 relative (PROJ
    last-bit differences). Key order is not compared."""
    if type(a) is not type(b):
        return False
    if isinstance(a, float):
        return math.isclose(a, b, rel_tol=1e-12)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_close(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return isinstance(b, list) and len(a) == len(b) and all(map(_close, a, b))
    return a == b


def test_round_half_up_not_bankers():
    assert round_half_up(0.0005, "0.001") == 0.001
    assert round_half_up(2.5, "1") == 3.0
    assert round_half_up(1703.4624, "0.01") == 1703.46
    assert round_half_up(0.9996, "0.001") == 1.0


@pytest.mark.parametrize("freshness, expected", [(1.8, 1703.46), (1.0, 946.37), (0.6, 567.82)])
def test_price_control_arithmetic(freshness, expected):
    # task.md: Б=788,64; L2 (1,2); internal (1); Р=1; S=1,000 км²
    unit = product_half_up([788.64, 1.2, 1.0, freshness, 1.0], "0.01")
    assert unit == expected
    assert product_half_up([unit, 1.000], "0.01") == expected


def test_area_km2_equal_area_square():
    to_ll = Transformer.from_crs("EPSG:6933", "EPSG:4326", always_xy=True)
    x0, y0 = Transformer.from_crs("EPSG:4326", "EPSG:6933", always_xy=True).transform(80.0, 7.8)
    ring = [to_ll.transform(x0 + dx, y0 + dy) for dx, dy in [(0, 0), (1000, 0), (1000, 1000), (0, 1000)]]
    assert area_km2(Polygon(ring)) == pytest.approx(1.0, abs=1e-9)
    # swapped axes ([lat, lon]) give a different area: order matters
    assert abs(area_km2(Polygon([(y, x) for x, y in ring])) - 1.0) > 1e-3


def test_run_config_loads():
    cfg = load_run_config()
    assert cfg.chip_id == "Sri-Lanka_713926" and cfg.chip_split == "validation"
    assert cfg.decision_deadline >= cfg.decision_at and cfg.budget_rub > 0


def test_empty_cell_is_none_and_nan_rejected():
    row = {"asset_id": "a", "chip_id": "c", "p_flood": "", "expected_loss_rub": "", "uncertainty": "",
               "rank": "", "status": "no_data"}
    assert AssetLossRow.model_validate(row).p_flood is None
    with pytest.raises(ValidationError):
        AssetLossRow.model_validate(dict(row, status="ok", p_flood="nan", expected_loss_rub="1",
                                         rank="1"))


def test_not_estimated_requires_empty_number():
    base = {"strategy": "B", "data_cost_rub": "1", "other_cost_rub": "0", "decision_cost_rub": "1",
                "budget_rub": "5", "budget_feasible": "true", "covered_expected_loss_rub": "0",
                "coverage_share": "", "residual_uncertainty": "", "uncertainty_status": "not_estimated",
                "uncertainty_basis": "x"}
    StrategyComparisonRow.model_validate(base)
    with pytest.raises(ValidationError):
        StrategyComparisonRow.model_validate(dict(base, residual_uncertainty="0"))
    with pytest.raises(ValidationError):
        StrategyComparisonRow.model_validate(dict(base, uncertainty_status="scenario"))


def test_fixture_is_reproduced_by_builder(tmp_path):
    spec = importlib.util.spec_from_file_location("build_fixture", HERE / "build_fixture.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.build(tmp_path)
    for name in F.REQUIRED_FILES:
        if name.endswith(".tif"):
            with rasterio.open(tmp_path / name) as a, rasterio.open(GOOD / name) as b:
                assert np.array_equal(a.read(), b.read()) and a.profile == b.profile
        elif name.endswith(".geojson"):
            # PROJ may differ in the last float bit across platforms: compare coordinates with tolerance.
            a = json.loads((tmp_path / name).read_text(encoding="utf-8"))
            b = json.loads((GOOD / name).read_text(encoding="utf-8"))
            assert _close(a, b), name
        else:
            assert (tmp_path / name).read_bytes() == (GOOD / name).read_bytes(), name
