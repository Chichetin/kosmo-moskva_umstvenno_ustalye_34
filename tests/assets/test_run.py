import argparse
import csv
import json

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from floodvalue.assets import run as assets_run
from floodvalue.assets.portfolio import PROB_NODATA

H = W = 32
TRANSFORM = from_origin(81.0, 7.0, 8.983152841195215e-05, 8.983152841195215e-05)


def write_s1(path):
    s1 = np.random.default_rng(1).normal(-15, 3, (2, H, W)).astype("float32")
    s1[:, :, :8] = np.nan  # S1 nodata strip
    s1[1, 20, 20] = np.nan  # VH only missing
    with rasterio.open(path, "w", driver="GTiff", height=H, width=W, count=2, dtype="float32",
                       crs="EPSG:4326", transform=TRANSFORM, nodata=np.nan) as dst:
        dst.write(s1)
    return s1


def write_prob(path, prob):
    with rasterio.open(path, "w", driver="GTiff", height=H, width=W, count=1, dtype="float32",
                       crs="EPSG:4326", transform=TRANSFORM, nodata=PROB_NODATA) as dst:
        dst.write(prob.astype("float32"), 1)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    data = tmp_path / "data"
    s1_dir = data / "raw" / "HandLabeled" / "S1Hand"
    s1_dir.mkdir(parents=True)
    s1 = write_s1(s1_dir / "Test_1_S1Hand.tif")
    out = tmp_path / "out"
    out.mkdir()
    prob = np.full((H, W), 0.4, dtype="float32")
    prob[~np.isfinite(s1).all(axis=0)] = PROB_NODATA
    write_prob(out / "flood_probability.tif", prob)
    monkeypatch.setenv("FLOODVALUE_DATA_DIR", str(data))
    return argparse.Namespace(chip="Test_1", out=str(out), budget=None), out, s1


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_run_writes_three_files_by_contract(setup):
    args, out, s1 = setup
    assert assets_run.run(args) == 0
    fc = json.loads((out / "assets.geojson").read_text(encoding="utf-8"))
    assert fc["type"] == "FeatureCollection" and len(fc["features"]) == 10
    assets_csv = read_csv(out / "assets.csv")
    loss = read_csv(out / "asset_loss.csv")
    assert list(assets_csv[0]) == ["asset_id", "chip_id", "asset_class", "asset_value_rub",
                                   "vulnerability_coef", "longitude", "latitude"]
    # contract fields first, then the optional EL-range columns of D-038
    assert list(loss[0]) == ["asset_id", "chip_id", "p_flood", "expected_loss_rub",
                             "uncertainty", "rank", "status", "p_low", "p_high", "el_low_rub",
                             "el_high_rub", "mask_unstable"]
    by_id = {r["asset_id"]: r for r in assets_csv}
    for feat in fc["features"]:
        props = feat["properties"]
        lon, lat = feat["geometry"]["coordinates"]
        row = by_id[props["asset_id"]]
        assert props["chip_id"] == "Test_1"
        assert float(row["longitude"]) == lon and float(row["latitude"]) == lat
        assert 81.0 < lon < 81.0 + W * 9e-05 and 7.0 - H * 9e-05 < lat < 7.0
        c, r = ~TRANSFORM @ (lon, lat)
        assert np.isfinite(s1[:, int(r), int(c)]).all()
    for r in loss:
        assert r["status"] == "ok"
        assert r["p_flood"] == "0.4"
        v = float(by_id[r["asset_id"]]["asset_value_rub"])
        q = float(by_id[r["asset_id"]]["vulnerability_coef"])
        assert float(r["expected_loss_rub"]) == pytest.approx(0.4 * v * q, rel=1e-6)
    meta = json.loads((out / "assets_metadata.json").read_text(encoding="utf-8"))
    assert {"seed", "placement_rule", "p_extraction_rule", "uncertainty", "status_rule"} <= set(meta)


def test_run_no_data_asset_has_empty_fields(setup):
    args, out, _ = setup
    assets_run.run(args)
    first = read_csv(out / "assets.csv")[0]
    c, r = ~TRANSFORM @ (float(first["longitude"]), float(first["latitude"]))
    with rasterio.open(out / "flood_probability.tif") as src:
        prob = src.read(1)
    prob[int(r), int(c)] = PROB_NODATA
    write_prob(out / "flood_probability.tif", prob)
    assets_run.run(args)
    loss = {r["asset_id"]: r for r in read_csv(out / "asset_loss.csv")}
    row = loss[first["asset_id"]]
    assert row["status"] == "no_data"
    assert row["p_flood"] == row["expected_loss_rub"] == row["rank"] == row["uncertainty"] == ""
    # placement unchanged by the p-raster
    assert read_csv(out / "assets.csv")[0] == first


def test_run_places_only_where_both_vv_and_vh_finite(tmp_path, monkeypatch):
    data = tmp_path / "data"
    s1_dir = data / "raw" / "HandLabeled" / "S1Hand"
    s1_dir.mkdir(parents=True)
    s1 = np.full((2, H, W), -15.0, dtype="float32")
    s1[1, :, :] = np.nan  # VH missing everywhere ...
    s1[1, 10:14, 10:14] = -20.0  # ... except a 4x4 block
    with rasterio.open(s1_dir / "Vh_1_S1Hand.tif", "w", driver="GTiff", height=H, width=W,
                       count=2, dtype="float32", crs="EPSG:4326", transform=TRANSFORM,
                       nodata=np.nan) as dst:
        dst.write(s1)
    out = tmp_path / "out"
    out.mkdir()
    write_prob(out / "flood_probability.tif", np.full((H, W), 0.2, dtype="float32"))
    monkeypatch.setenv("FLOODVALUE_DATA_DIR", str(data))
    assert assets_run.run(argparse.Namespace(chip="Vh_1", out=str(out), budget=None)) == 0
    for row in read_csv(out / "assets.csv"):
        c, r = ~TRANSFORM @ (float(row["longitude"]), float(row["latitude"]))
        assert 10 <= int(r) < 14 and 10 <= int(c) < 14


def test_run_writes_inspection_priority_and_keeps_el_rank(setup):
    # D-035: inspection order is a separate key in assets_metadata.json; asset_loss.csv rank
    # stays "1 = max EL, ties by asset_id".
    args, out, _ = setup
    assert assets_run.run(args) == 0
    loss = read_csv(out / "asset_loss.csv")
    by_el = sorted(loss, key=lambda r: (-float(r["expected_loss_rub"]), r["asset_id"]))
    assert [int(r["rank"]) for r in by_el] == list(range(1, 11))
    meta = json.loads((out / "assets_metadata.json").read_text(encoding="utf-8"))
    ip = meta["inspection_priority"]
    assert ip["rule"] and ip["p_dry"] == 0.1 and ip["social_max_position"] == 3
    assert ip["uncertainty_measure"] == meta["uncertainty"]["name"]
    order = ip["order"]
    assert sorted(o["asset_id"] for o in order) == [r["asset_id"] for r in loss]
    assert [o["inspection_order"] for o in order] == list(range(1, 11))
    rank = {r["asset_id"]: int(r["rank"]) for r in loss}
    assert all(o["el_rank"] == rank[o["asset_id"]] for o in order)
    assert set(ip["scenario_changes"]) == {
        r["scenario_id"] for r in read_csv(out / "assets_sensitivity.csv")}


@pytest.mark.parametrize("pred_chip, expected", [("Test_1", 0.5), ("Other_2", None)])
def test_run_takes_mask_threshold_only_from_prediction_of_same_chip(setup, pred_chip, expected):
    # A stale prediction.json of another chip must not put its threshold into the reasons.
    args, out, _ = setup
    (out / "prediction.json").write_text(json.dumps({"chip_id": pred_chip, "threshold": 0.5}),
                                         encoding="utf-8")
    assert assets_run.run(args) == 0
    ip = json.loads((out / "assets_metadata.json").read_text(encoding="utf-8"))[
        "inspection_priority"]
    assert ip["mask_threshold"] == expected
    below_map = any("ниже порога карты" in o["reason"] for o in ip["order"])
    assert below_map == (expected is not None)  # p = 0.4 everywhere, below 0.5
