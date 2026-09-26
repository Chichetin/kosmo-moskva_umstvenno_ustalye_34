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
    (out / "baseline_mask.tif").unlink()  # the fixture's baseline mask is on the replaced grid
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


def test_data_s1_source_entries_pass_manifest_schema():
    """floodvalue.data (D-015) entries, copied as is, satisfy SourceManifest and the S1 check."""
    from floodvalue.contracts.schemas import SourceManifest
    from floodvalue.data.fetch import bucket_name, s1_source_entries

    rows = [{"chip_id": "Sri-Lanka_713926", "event_id": "Sri-Lanka",
             "s1_file": "HandLabeled/S1Hand/Sri-Lanka_713926_S1Hand.tif",
             "label_file": "HandLabeled/LabelHand/Sri-Lanka_713926_LabelHand.tif",
             "s1_sha256": "0" * 64, "label_sha256": "1" * 64}]
    bucket = {bucket_name(rows[0]["s1_file"]): {"updated": "2020-10-06T21:13:52.123Z",
                                                "generation": "1"}}
    entries = s1_source_entries(rows, {"Sri-Lanka": "2017-05-30"}, bucket)
    manifest = SourceManifest.model_validate({"sources": entries})
    src = manifest.sources[0]
    assert "Sri-Lanka_713926" in src.id and src.file == "Sri-Lanka_713926_S1Hand.tif"


def test_procurement_geometry_area_matches_validator():
    """procurement.geometry.order_area_km2 = validator's ROUND_HALF_UP(EPSG:6933) on the set."""
    from shapely.geometry import shape

    from floodvalue.contracts.geo import area_km2, round_half_up
    from floodvalue.procurement.geometry import order_area_km2

    feats = json.loads((GOOD / F.CANDIDATE_ORDERS).read_text(encoding="utf-8"))["features"]
    for f in feats:
        geom = shape(f["geometry"])
        assert float(order_area_km2(geom)) == round_half_up(area_km2(geom), "0.001") \
            == f["properties"]["area_km2"]
