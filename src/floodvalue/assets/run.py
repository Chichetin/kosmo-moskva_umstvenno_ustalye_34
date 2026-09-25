"""`python -m floodvalue assets --chip <id> [--out <dir>]`: 10 assets, p, EL, rank.

Reads the S1 chip (only for its validity mask and grid) and flood_probability.tif from the
results directory; writes assets.geojson, assets.csv, asset_loss.csv and
assets_metadata.json (seed and rules for run_metadata.json).
"""

import csv
import json
from pathlib import Path

import numpy as np
import rasterio

from floodvalue.assets.portfolio import (
    P_EXTRACTION_RULE,
    PLACEMENT_RULE,
    STATUS_RULE,
    UNCERTAINTY_RULE,
    assess_assets,
    load_asset_types,
    place_assets,
)
from floodvalue.paths import data_dir, output_dir

# Portfolio seed. TODO: read from configs/run.yaml once it lands in main (contracts §5).
PORTFOLIO_SEED = 20260925

ASSETS_FIELDS = ["asset_id", "chip_id", "asset_class", "asset_value_rub", "vulnerability_coef",
                 "longitude", "latitude"]
LOSS_FIELDS = ["asset_id", "chip_id", "p_flood", "expected_loss_rub", "uncertainty", "rank",
               "status"]


def s1_path(chip_id: str) -> Path:
    return data_dir() / "raw" / "HandLabeled" / "S1Hand" / f"{chip_id}_S1Hand.tif"


def _cell(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return repr(v) if np.isfinite(v) else ""
    return str(v)


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(fields)
        for r in rows:
            w.writerow([_cell(r[k]) for k in fields])


def run(args) -> int:
    chip_id = args.chip
    if not chip_id:
        print("assets: --chip <chip_id> обязателен")
        return 1
    out = Path(args.out) if args.out else output_dir() / chip_id
    prob_path = out / "flood_probability.tif"

    with rasterio.open(s1_path(chip_id)) as src:
        s1 = src.read().astype("float32")
        transform, shape = src.transform, (src.height, src.width)
    with rasterio.open(prob_path) as src:
        if (src.height, src.width) != shape or not np.allclose(
            tuple(src.transform), tuple(transform), rtol=0, atol=1e-12
        ):
            print(f"assets: сетка {prob_path} не совпадает с S1 {chip_id}")
            return 1
        prob = src.read(1).astype("float32")
        if src.nodata is not None and np.isfinite(src.nodata):
            prob[prob == src.nodata] = np.nan

    s1_valid = np.isfinite(s1).all(axis=0)
    assets = place_assets(s1_valid, transform, PORTFOLIO_SEED, load_asset_types())
    loss_rows = assess_assets(assets, prob)

    out.mkdir(parents=True, exist_ok=True)
    asset_rows = [
        {"asset_id": a.asset_id, "chip_id": chip_id, "asset_class": a.asset_class,
         "asset_value_rub": a.asset_value_rub, "vulnerability_coef": a.vulnerability_coef,
         "longitude": a.longitude, "latitude": a.latitude}
        for a in assets
    ]
    fc = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature",
             "geometry": {"type": "Point", "coordinates": [r["longitude"], r["latitude"]]},
             "properties": {k: r[k] for k in ASSETS_FIELDS[:5]}}
            for r in asset_rows
        ],
    }
    (out / "assets.geojson").write_text(json.dumps(fc, ensure_ascii=False, indent=1),
                                        encoding="utf-8")
    _write_csv(out / "assets.csv", ASSETS_FIELDS, asset_rows)
    _write_csv(out / "asset_loss.csv", LOSS_FIELDS, [{**r, "chip_id": chip_id} for r in loss_rows])

    meta = {
        "chip_id": chip_id,
        "s1_file": s1_path(chip_id).name,
        "seed": PORTFOLIO_SEED,
        "seed_source": "floodvalue.assets.run.PORTFOLIO_SEED (configs/run.yaml not yet available)",
        "placement_rule": PLACEMENT_RULE,
        "p_extraction_rule": P_EXTRACTION_RULE,
        "el_model": "EL = p_flood * asset_value_rub * vulnerability_coef",
        "status_rule": STATUS_RULE,
        "uncertainty": {"name": "binary_entropy_bits", "status": "temporary",
                        "rule": UNCERTAINTY_RULE},
        "unassessed_asset_ids": [r["asset_id"] for r in loss_rows if r["status"] != "ok"],
    }
    (out / "assets_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                              encoding="utf-8")
    n_ok = sum(r["status"] == "ok" for r in loss_rows)
    print(f"assets: {len(assets)} объектов, оценено {n_ok}, файлы в {out}")
    return 0
