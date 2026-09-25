"""plan(run_dir, budget): пересчёт без изменения p-растра, маски и портфеля (D-013)."""
import csv
import hashlib
import json

import numpy as np
import rasterio
from rasterio.transform import from_origin

from floodvalue import procurement
from floodvalue.cli import main
from floodvalue.procurement import run as prun

FILES = ["flood_probability.tif", "flood_mask.tif", "assets.csv", "asset_loss.csv"]


def _write_run_dir(d):
    tr = from_origin(80.0, 7.05, 8.983e-05, 8.983e-05)
    prof = {"driver": "GTiff", "height": 512, "width": 512, "count": 1, "crs": "EPSG:4326",
            "transform": tr}
    p = np.random.default_rng(0).random((512, 512)).astype("float32")
    with rasterio.open(d / "flood_probability.tif", "w", dtype="float32", nodata=-9999, **prof) as f:
        f.write(p, 1)
    with rasterio.open(d / "flood_mask.tif", "w", dtype="uint8", nodata=255, **prof) as f:
        f.write((p >= 0.5).astype("uint8"), 1)
    with open(d / "assets.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["asset_id", "chip_id", "asset_class", "asset_value_rub",
                    "vulnerability_coef", "longitude", "latitude"])
        for i in range(1, 11):
            w.writerow([f"A{i:02d}", "X_1", f"c{i}", 1e6, 0.5, 80.002 + 0.0042 * i,
                        7.048 - 0.0042 * i])
    with open(d / "asset_loss.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["asset_id", "chip_id", "p_flood", "expected_loss_rub", "uncertainty",
                    "rank", "status"])
        for i in range(1, 11):
            w.writerow([f"A{i:02d}", "X_1", i / 10, i * 5e4, 0.5, 11 - i, "ok"])


def _hashes(d):
    return {n: hashlib.sha256((d / n).read_bytes()).hexdigest() for n in FILES}


def test_budget_change_keeps_inputs(tmp_path):
    _write_run_dir(tmp_path)
    before = _hashes(tmp_path)
    prun.write_outputs(tmp_path, 5000)
    small = procurement.plan(tmp_path, 1000)
    big = procurement.plan(tmp_path, 100000)
    assert set(small) >= {"plans", "plan_rows", "comparison", "reasons"}
    assert small["plans"]["C"] == [] and len(big["plans"]["C"]) > 1
    b_small = next(r for r in small["comparison"] if r["strategy"] == "B")
    assert b_small["budget_feasible"] is False
    assert _hashes(tmp_path) == before
    cat_hash = hashlib.sha256((tmp_path / "candidate_orders.geojson").read_bytes()).hexdigest()
    prun.write_outputs(tmp_path, 20000)
    assert _hashes(tmp_path) == before
    assert hashlib.sha256((tmp_path / "candidate_orders.geojson").read_bytes()).hexdigest() == cat_hash


def test_cli_procure_writes_contract_files(tmp_path):
    _write_run_dir(tmp_path)
    assert main(["procure", "--budget", "5000", "--out", str(tmp_path)]) == 0
    plans = json.loads((tmp_path / "strategy_plans.json").read_text())
    with open(tmp_path / "procurement_plan.csv") as f:
        rows = list(csv.DictReader(f))
    for s in "BC":
        assert [r["candidate_id"] for r in rows if r["strategy"] == s] == plans[s]
    assert plans["A"] == []
    with open(tmp_path / "strategy_comparison.csv") as f:
        cmp_ = {r["strategy"]: r for r in csv.DictReader(f)}
    assert float(cmp_["C"]["data_cost_rub"]) <= 5000
    assert cmp_["B"]["budget_feasible"] == "false"
    assert cmp_["B"]["residual_uncertainty"] == ""
    assert abs(sum(float(r["cost_rub"]) for r in rows if r["strategy"] == "C")
               - float(cmp_["C"]["data_cost_rub"])) < 1e-9
    with open(tmp_path / "procurement_sensitivity.csv") as f:
        sens = list(csv.DictReader(f))
    assert sum(r["scenario_id"].startswith("budget_") for r in sens) >= 5
    assert {r["scenario_id"] for r in sens if r["scenario_id"].startswith("rate_")} >= {
        "rate_x0.5", "rate_x1.5"}
    meta = json.loads((tmp_path / "procurement_metadata.json").read_text())
    assert "N-OPT-LATE" in meta["zones"]["excluded_from_coverage"]


def test_cli_missing_assets(tmp_path):
    assert main(["procure", "--budget", "5000", "--out", str(tmp_path)]) == 1
