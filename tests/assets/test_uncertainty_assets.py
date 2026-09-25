"""assets reads uncertainty.tif and flood_probability_range.tif written by predict (D-039)."""

import argparse
import csv
import json

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from floodvalue.assets import run as assets_run
from floodvalue.assets.portfolio import (
    PROB_NODATA,
    Asset,
    assess_assets,
    binary_entropy,
    mask_unstable,
)

H = W = 32
TRANSFORM = from_origin(81.0, 7.0, 8.983152841195215e-05, 8.983152841195215e-05)


def _write(path, bands, transform=TRANSFORM):
    bands = np.asarray(bands, dtype="float32")
    if bands.ndim == 2:
        bands = bands[None]
    with rasterio.open(path, "w", driver="GTiff", height=bands.shape[1], width=bands.shape[2],
                       count=bands.shape[0], dtype="float32", crs="EPSG:4326",
                       transform=transform, nodata=PROB_NODATA) as dst:
        dst.write(bands)


@pytest.fixture
def run_dir(tmp_path, monkeypatch):
    data = tmp_path / "data"
    s1_dir = data / "raw" / "HandLabeled" / "S1Hand"
    s1_dir.mkdir(parents=True)
    s1 = np.full((2, H, W), -15.0, dtype="float32")
    with rasterio.open(s1_dir / "T_1_S1Hand.tif", "w", driver="GTiff", height=H, width=W,
                       count=2, dtype="float32", crs="EPSG:4326", transform=TRANSFORM,
                       nodata=np.nan) as dst:
        dst.write(s1)
    out = tmp_path / "out"
    out.mkdir()
    # p grows along columns, uncertainty is unique per pixel, range = p -/+ 0.1
    p = np.tile(np.linspace(0.05, 0.95, W, dtype="float32"), (H, 1))
    _write(out / "flood_probability.tif", p)
    monkeypatch.setenv("FLOODVALUE_DATA_DIR", str(data))
    return argparse.Namespace(chip="T_1", out=str(out), budget=None), out, p


def _read(path):
    with open(path, encoding="utf-8", newline="") as f:
        return {r["asset_id"]: r for r in csv.DictReader(f)}


def _pixel(out, aid):
    a = _read(out / "assets.csv")[aid]
    c, r = ~TRANSFORM @ (float(a["longitude"]), float(a["latitude"]))
    return int(r), int(c), float(a["asset_value_rub"]), float(a["vulnerability_coef"])


def test_asset_uncertainty_and_el_range_come_from_predict_rasters(run_dir):
    args, out, p = run_dir
    u = (np.arange(H * W, dtype="float32").reshape(H, W) + 1) / (H * W)
    lo, hi = np.clip(p - 0.1, 0, 1), np.clip(p + 0.1, 0, 1)
    _write(out / "uncertainty.tif", u)
    _write(out / "flood_probability_range.tif", np.stack([lo, hi]))
    (out / "prediction.json").write_text(json.dumps(
        {"threshold": 0.37, "uncertainty": {"measure": "entropy_p", "units": "биты, 0…1",
                                            "status": "проверено на validation",
                                            "choice": "models/unet/uncertainty.json"}}))
    assert assets_run.run(args) == 0
    loss = _read(out / "asset_loss.csv")
    assert len(loss) == 10
    for aid, r in loss.items():
        row, col, v, q = _pixel(out, aid)
        assert float(r["uncertainty"]) == pytest.approx(float(u[row, col]))
        assert float(r["uncertainty"]) != pytest.approx(binary_entropy(float(r["p_flood"])))
        assert float(r["p_low"]) == pytest.approx(float(lo[row, col]))
        assert float(r["p_high"]) == pytest.approx(float(hi[row, col]))
        assert float(r["el_low_rub"]) == pytest.approx(float(r["p_low"]) * v * q, abs=0.005)
        assert float(r["el_high_rub"]) == pytest.approx(float(r["p_high"]) * v * q, abs=0.005)
        assert float(r["el_low_rub"]) <= float(r["expected_loss_rub"]) <= float(r["el_high_rub"])
        want = float(r["p_low"]) < 0.37 <= float(r["p_high"])
        assert r["mask_unstable"] == ("true" if want else "false")
    meta = json.loads((out / "assets_metadata.json").read_text(encoding="utf-8"))["uncertainty"]
    assert meta["name"] == "entropy_p" and meta["status"] == "проверено на validation"
    assert meta["units"] == "биты, 0…1" and meta["raster"] == "uncertainty.tif"
    assert "not a confidence interval" in meta["el_range_rule"]
    assert meta["el_range_rule"] in meta["rule"]     # export carries `rule` to run_metadata


def test_without_predict_rasters_uncertainty_is_unverified_entropy(run_dir):
    args, out, _ = run_dir
    assert assets_run.run(args) == 0
    for r in _read(out / "asset_loss.csv").values():
        assert float(r["uncertainty"]) == pytest.approx(binary_entropy(float(r["p_flood"])))
        assert r["p_low"] == r["p_high"] == r["el_low_rub"] == r["el_high_rub"] == ""
        assert r["mask_unstable"] == ""
    meta = json.loads((out / "assets_metadata.json").read_text(encoding="utf-8"))["uncertainty"]
    assert meta["name"] == "binary_entropy_bits" and meta["status"] == "temporary_unverified"


def test_uncertainty_nodata_at_ok_asset_is_empty_not_zero(run_dir):
    args, out, _ = run_dir
    _write(out / "uncertainty.tif", np.full((H, W), PROB_NODATA, "float32"))
    (out / "prediction.json").write_text(json.dumps({"uncertainty": {"measure": "entropy_p"}}))
    assert assets_run.run(args) == 0
    for r in _read(out / "asset_loss.csv").values():
        assert r["status"] == "ok" and r["uncertainty"] == ""


def test_uncertainty_raster_on_other_grid_is_refused(run_dir):
    args, out, _ = run_dir
    _write(out / "uncertainty.tif", np.zeros((H, W)), transform=from_origin(0, 0, 1e-4, 1e-4))
    with pytest.raises(ValueError, match="сетка"):
        assets_run.run(args)


def test_no_data_asset_has_empty_range():
    prob = np.array([[0.4, np.nan]], dtype="float32")
    lo, hi = prob - 0.1, prob + 0.1
    assets = [Asset("A01", "x", 1000, 0.5, 0, 0, 0, 0), Asset("A02", "y", 1000, 0.5, 0, 1, 0, 0)]
    rows = {r["asset_id"]: r for r in assess_assets(assets, prob, None, (lo, hi))}
    assert rows["A02"]["status"] == "no_data"
    assert rows["A02"]["p_low"] is rows["A02"]["el_high_rub"] is None
    assert rows["A01"]["el_low_rub"] == pytest.approx(float(lo[0, 0]) * 500)


def test_mask_unstable_boundaries():
    assert mask_unstable({"p_low": 0.3, "p_high": 0.37}, 0.37) is True     # p_high == t: mask 1
    assert mask_unstable({"p_low": 0.37, "p_high": 0.5}, 0.37) is False    # p_low == t: all 1
    assert mask_unstable({"p_low": 0.1, "p_high": 0.2}, 0.37) is False
    assert mask_unstable({"p_low": None, "p_high": None}, 0.37) is None
    assert mask_unstable({"p_low": 0.1, "p_high": 0.5}, None) is None


def test_raster_without_its_measure_in_prediction_is_ignored(run_dir):
    args, out, _ = run_dir
    _write(out / "uncertainty.tif", np.full((H, W), 0.123, "float32"))
    (out / "prediction.json").write_text(json.dumps({"chip_id": "Other_9", "threshold": 0.37,
                                                     "uncertainty": {"measure": "entropy_p"}}))
    assert assets_run.run(args) == 0
    for r in _read(out / "asset_loss.csv").values():
        assert float(r["uncertainty"]) == pytest.approx(binary_entropy(float(r["p_flood"])))
    meta = json.loads((out / "assets_metadata.json").read_text(encoding="utf-8"))
    assert meta["uncertainty"]["status"] == "temporary_unverified"


def test_inspection_threshold_only_for_entropy_measures(run_dir):
    args, out, _ = run_dir
    _write(out / "uncertainty.tif", np.full((H, W), 0.9, "float32"))
    for measure, expect_none in (("member_std", True), ("entropy_p", False)):
        (out / "prediction.json").write_text(json.dumps(
            {"threshold": 0.37, "uncertainty": {"measure": measure, "units": "u",
                                                "status": "s"}}))
        assert assets_run.run(args) == 0
        prio = json.loads((out / "assets_metadata.json").read_text(encoding="utf-8"))[
            "inspection_priority"]
        assert prio["uncertainty_measure"] == measure
        assert (prio["uncertainty_high"] is None) is expect_none
