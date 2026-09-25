"""`python -m floodvalue assets --chip <id> [--out <dir>]`: 10 assets, p, EL, rank.

Reads the S1 chip (only for its validity mask and grid) and flood_probability.tif from the
results directory, plus — if `predict` wrote them — uncertainty.tif, flood_probability_range.tif
and prediction.json (measure, threshold; D-039); writes assets.geojson, assets.csv, asset_loss.csv, assets_sensitivity.csv
(p/V/q scenarios, P18a/D-028; export merges it with procurement_sensitivity.csv into
sensitivity.csv) and assets_metadata.json (seed and rules for run_metadata.json).
"""

import csv
import json
from pathlib import Path

import numpy as np
import rasterio

from floodvalue.assets.portfolio import (
    EL_RANGE_RULE,
    P_EXTRACTION_RULE,
    PLACEMENT_RULE,
    STATUS_RULE,
    UNCERTAINTY_FALLBACK_RULE,
    UNCERTAINTY_NAME,
    UNCERTAINTY_RULE,
    assess_assets,
    load_asset_types,
    mask_unstable,
    place_assets,
)
from floodvalue.assets.priority import (
    P_DRY,
    RULE,
    SOCIAL_CLASSES,
    SOCIAL_MAX_POSITION,
    UNCERTAINTY_HIGH,
    inspection_order,
)
from floodvalue.assets.sensitivity import (
    P_RULE,
    Q_RULE,
    SENSITIVITY_FIELDS,
    V_RULE,
    el_sensitivity,
    inspection_order_changes,
)
from floodvalue.paths import data_dir, output_dir

# Portfolio seed. TODO: read from configs/run.yaml once it lands in main (contracts §5).
PORTFOLIO_SEED = 20260925

ASSETS_FIELDS = ["asset_id", "chip_id", "asset_class", "asset_value_rub", "vulnerability_coef",
                 "longitude", "latitude"]
LOSS_FIELDS = ["asset_id", "chip_id", "p_flood", "expected_loss_rub", "uncertainty", "rank",
               "status", "p_low", "p_high", "el_low_rub", "el_high_rub", "mask_unstable"]


def s1_path(chip_id: str) -> Path:
    return data_dir() / "raw" / "HandLabeled" / "S1Hand" / f"{chip_id}_S1Hand.tif"


def _cell(v) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return repr(v) if np.isfinite(v) else ""
    return str(v)


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(fields)
        for r in rows:
            w.writerow([_cell(r[k]) for k in fields])


def _read_on_grid(path: Path, shape: tuple[int, int], transform) -> np.ndarray:
    """All bands of a float raster that must share the S1 grid; nodata stays -9999/NaN."""
    with rasterio.open(path) as src:
        if (src.height, src.width) != shape or not np.allclose(
            tuple(src.transform), tuple(transform), rtol=0, atol=1e-12
        ):
            raise ValueError(f"assets: сетка {path} не совпадает с S1")
        return src.read().astype("float32")


def uncertainty_meta(has_raster: bool, has_range: bool, pred_unc: dict) -> dict:
    """assets_metadata.uncertainty -> run_metadata.uncertainty (export: name, units, status, rule)."""
    if has_raster and pred_unc.get("measure"):
        meta = {"name": pred_unc["measure"], "units": pred_unc.get("units", "не задано"),
                "status": pred_unc.get("status", "не задано"),
                "rule": UNCERTAINTY_RULE + (" " + EL_RANGE_RULE if has_range else ""),
                "raster": "uncertainty.tif", "choice": pred_unc.get("choice"),
                "validation": pred_unc.get("validation")}
    else:
        meta = {"name": "binary_entropy_bits", "units": "биты, 0…1",
                "status": "temporary_unverified", "rule": UNCERTAINTY_FALLBACK_RULE}
    meta["el_range_rule"] = EL_RANGE_RULE if has_range else "не рассчитан: нет flood_probability_range.tif"
    meta["priority_note"] = ("rank — по EL (обязательное поле); очерёдность осмотра — "
                             "inspection_priority (D-035); el_high_rub и mask_unstable — "
                             "дополнительно: где EL выше при другом seed и где маска неустойчива "
                             "(docs/verification/uncertainty-val.md, раздел 8)")
    return meta


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
        prob = src.read(1).astype("float32")  # nodata -9999/NaN is handled by extract_p

    unc_raster, p_range = None, None
    if (out / "uncertainty.tif").exists():
        unc_raster = _read_on_grid(out / "uncertainty.tif", shape, transform)[0]
    if (out / "flood_probability_range.tif").exists():
        lo, hi = _read_on_grid(out / "flood_probability_range.tif", shape, transform)[:2]
        p_range = (lo, hi)
    # flood_mask.tif threshold and the uncertainty measure of predict; a prediction.json of
    # another chip is ignored (same chip_id rule as export).
    prediction_path = out / "prediction.json"
    prediction = (json.loads(prediction_path.read_text(encoding="utf-8"))
                  if prediction_path.is_file() else {})
    if prediction.get("chip_id") not in (None, chip_id):
        prediction = {}
    mask_threshold = prediction.get("threshold")
    unc_meta = uncertainty_meta(unc_raster is not None, p_range is not None,
                                prediction.get("uncertainty") or {})

    s1_valid = np.isfinite(s1).all(axis=0)
    assets = place_assets(s1_valid, transform, PORTFOLIO_SEED, load_asset_types())
    loss_rows = assess_assets(assets, prob, unc_raster, p_range)
    for r in loss_rows:
        r["mask_unstable"] = mask_unstable(
            r, float(mask_threshold) if mask_threshold is not None else None)

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

    sensitivity_rows = el_sensitivity(assets, loss_rows)
    # UNCERTAINTY_HIGH = H(0.10) in bits (D-035) holds only for an entropy of p in bits
    entropy_bits = unc_meta["name"] in (UNCERTAINTY_NAME, "entropy_p")
    uncertainty_high = UNCERTAINTY_HIGH if entropy_bits else None
    order = inspection_order(loss_rows, {a.asset_id: a.asset_class for a in assets},
                             uncertainty_high=uncertainty_high, mask_threshold=mask_threshold)
    _write_csv(out / "assets_sensitivity.csv", SENSITIVITY_FIELDS, sensitivity_rows)

    meta = {
        "chip_id": chip_id,
        "s1_file": s1_path(chip_id).name,
        "seed": PORTFOLIO_SEED,
        "seed_source": "floodvalue.assets.run.PORTFOLIO_SEED (configs/run.yaml not yet available)",
        "placement_rule": PLACEMENT_RULE,
        "p_extraction_rule": P_EXTRACTION_RULE,
        "el_model": "EL = p_flood * asset_value_rub * vulnerability_coef",
        "status_rule": STATUS_RULE,
        "uncertainty": unc_meta,
        "unassessed_asset_ids": [r["asset_id"] for r in loss_rows if r["status"] != "ok"],
        "sensitivity": {"p_rule": P_RULE, "v_rule": V_RULE, "q_rule": Q_RULE,
                        "n_scenarios": len(sensitivity_rows),
                        "note": "assets_sensitivity.csv only; budget/rate scenarios are "
                                "procurement_sensitivity.csv (P18b)"},
        "inspection_priority": {
            "decision": "D-035", "rule": RULE, "p_dry": P_DRY,
            "uncertainty_measure": unc_meta["name"], "uncertainty_high": uncertainty_high,
            "social_classes": list(SOCIAL_CLASSES), "social_max_position": SOCIAL_MAX_POSITION,
            "mask_threshold": mask_threshold,
            "n_not_assessed": sum(o["group"] == "not_assessed" for o in order),
            "order": order,
            "scenario_changes": inspection_order_changes(assets, loss_rows),
            "scenario_note": "очерёдность пересчитана тем же правилом в каждом сценарии p/V/q "
                             "assets_sensitivity.csv; пустой список — без изменений",
        },
    }
    (out / "assets_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                              encoding="utf-8")
    n_ok = sum(r["status"] == "ok" for r in loss_rows)
    print(f"assets: {len(assets)} объектов, оценено {n_ok}, файлы в {out}")
    return 0
