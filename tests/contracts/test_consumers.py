"""Producers already in main write files that pass the contract validator."""

import argparse
import json
import shutil

import numpy as np
import rasterio
from rasterio.transform import from_origin

from floodvalue.assets import run as assets_run
from floodvalue.contracts import files as F
from floodvalue.export import validate as V
from floodvalue.export.validate import FIXTURE_RUN as GOOD

H = W = 48
TRANSFORM = from_origin(81.0, 7.0, 8.983152841195215e-05, 8.983152841195215e-05)
CHIP = "Test_1"


def test_assets_stage_output_passes_validator(tmp_path, monkeypatch):
    data = tmp_path / "data" / "raw" / "HandLabeled" / "S1Hand"
    data.mkdir(parents=True)
    s1 = np.random.default_rng(1).normal(-15, 3, (2, H, W)).astype("float32")
    s1[:, :, :10] = np.nan
    s1[1, 30, 30] = np.nan
    prof = {"driver": "GTiff", "height": H, "width": W, "crs": "EPSG:4326", "transform": TRANSFORM}
    with rasterio.open(data / f"{CHIP}_S1Hand.tif", "w", count=2, dtype="float32", nodata=np.nan,
                       **prof) as dst:
        dst.write(s1)
    valid = np.isfinite(s1).all(axis=0)
    rr, cc = np.mgrid[0:H, 0:W]
    p = np.where(valid, (rr * W + cc) / (H * W), -9999.0).astype("float32")
    mask = np.where(valid, (p.astype(np.float64) >= 0.5).astype(np.uint8), 255).astype(np.uint8)

    out = tmp_path / "run"
    shutil.copytree(GOOD, out)
    with rasterio.open(out / F.FLOOD_PROBABILITY, "w", count=1, dtype="float32", nodata=-9999.0,
                       **prof) as dst:
        dst.write(p, 1)
    with rasterio.open(out / F.FLOOD_MASK, "w", count=1, dtype="uint8", nodata=255, **prof) as dst:
        dst.write(mask, 1)
    meta = json.loads((out / F.RUN_METADATA).read_text(encoding="utf-8"))
    meta["chip_id"] = CHIP
    meta["source"]["s1_files"] = [f"{CHIP}_S1Hand.tif"]
    (out / F.RUN_METADATA).write_text(json.dumps(meta), encoding="utf-8")
    monkeypatch.setenv("FLOODVALUE_DATA_DIR", str(tmp_path / "data"))

    args = argparse.Namespace(chip=CHIP, out=str(out), budget=None)
    assert assets_run.run(args) == 0
    errors = V.validate_run(out)
    for name in (F.FLOOD_PROBABILITY, F.FLOOD_MASK, F.ASSETS_GEOJSON, F.ASSETS_CSV, F.ASSET_LOSS,
                 F.RUN_METADATA):
        assert errors[name] == [], (name, errors[name])


def test_procurement_price_basket_matches_reference_plan():
    """procurement.pricing (D-017) and the validator agree to the kopeck on the reference set."""
    import csv

    from floodvalue.procurement.pricing import load_pricing, price_basket

    cat = {f["properties"]["candidate_id"]: f["properties"] for f in json.loads(
        (GOOD / F.CANDIDATE_ORDERS).read_text(encoding="utf-8"))["features"]}
    with open(GOOD / F.PROCUREMENT_PLAN, encoding="utf-8", newline="") as f:
        plan = list(csv.DictReader(f))
    cfg = load_pricing()
    for strategy in ("B", "C"):
        mine = [r for r in plan if r["strategy"] == strategy]
        theirs = price_basket([cat[r["candidate_id"]] for r in mine], cfg, as_of="2017-05-30")
        for a, b in zip(mine, theirs, strict=True):
            for field in ("discount_coef", "group_area_km2", "unit_price_rub_km2", "cost_rub",
                          "freshness_coef", "discount_group_id"):
                assert str(b[field]) == a[field] or float(b[field]) == float(a[field]), \
                    (strategy, a["candidate_id"], field, a[field], b[field])
