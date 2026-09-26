"""Build the synthetic reference set tests/fixtures/good (12 files + optional baseline_mask.tif, 64×64 raster).

    uv run --frozen python tests/contracts/build_fixture.py tests/fixtures/good

Synthetic chip `Synthetic_000001` (not Sen1Floods11): p is a diagonal gradient with a 6×6
nodata corner, threshold 0.5; the optional baseline_mask.tif is p >= 0.6 on the same grid. Prices follow configs/pricing.yaml; the arithmetic here is
written independently of the validator on purpose.
"""

import csv
import json
import math
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
import rasterio
import yaml
from pyproj import Transformer
from rasterio.transform import from_origin
from shapely.geometry import Polygon, mapping

ROOT = Path(__file__).resolve().parents[2]
CHIP = "Synthetic_000001"
LON0, LAT0, RES, SIZE = 80.0, 7.80, 0.0005, 64
THRESHOLD = 0.5
BASELINE_CUT = 0.6  # synthetic stand-in for the VH rule of the baseline mask
BUDGET = 5000.0
DEADLINE = "2017-05-31T00:00:00+00:00"

ASSETS = [  # asset_id, class, (row, col)
    ("a01", "warehouse", (2, 2)),  # in the nodata corner -> status no_data
    ("a02", "substation", (10, 50)),
    ("a03", "road_unit", (20, 40)),
    ("a04", "facility", (30, 10)),
    ("a05", "pumping_station", (40, 60)),
    ("a06", "clinic", (50, 20)),
    ("a07", "school", (60, 62)),
    ("a08", "workshop", (15, 15)),
    ("a09", "telecom_node", (33, 33)),
    ("a10", "water_intake", (55, 45)),
]

ZONES = [  # id, centre asset, side m, sensor, res m, acquisition, role, obs, avail, level
    ("Z1", "a02", 1100, "optical", 1, "new", "event_observation",
     "2017-05-30T18:00:00+00:00", "2017-05-30T20:00:00+00:00", "L2"),
    ("Z2", "a09", 1050, "sar", 3, "operational", "event_observation",
     "2017-05-30T00:40:00+00:00", "2017-05-30T06:00:00+00:00", "L1"),
    ("Z3", "a06", 1100, "optical", 1, "archive", "context",
     "2016-12-01T05:00:00+00:00", "2016-12-05T00:00:00+00:00", "L2"),
    ("Z4", "a10", 1100, "optical", 1, "new", "event_observation", None,
     "2017-06-02T00:00:00+00:00", "L2"),
]
PLANS = {"A": [], "B": ["Z1", "Z2", "Z3", "Z4"], "C": ["Z1", "Z2"]}

TO_EA = Transformer.from_crs("EPSG:4326", "EPSG:6933", always_xy=True)
TO_LL = Transformer.from_crs("EPSG:6933", "EPSG:4326", always_xy=True)


def q(value, quantum: str) -> float:
    return float(Decimal(repr(value)).quantize(Decimal(quantum), rounding=ROUND_HALF_UP))


def dmul(*xs) -> Decimal:
    out = Decimal(1)
    for x in xs:
        out *= Decimal(repr(float(x)))
    return out


def lonlat(row: int, col: int) -> tuple[float, float]:
    return LON0 + (col + 0.5) * RES, LAT0 - (row + 0.5) * RES


def square(lon: float, lat: float, side_m: float) -> Polygon:
    x, y = TO_EA.transform(lon, lat)
    h = side_m / 2
    corners = [(x - h, y - h), (x + h, y - h), (x + h, y + h), (x - h, y + h)]
    return Polygon([TO_LL.transform(cx, cy) for cx, cy in corners])


def area_km2(poly: Polygon) -> float:
    xs, ys = TO_EA.transform(*poly.exterior.xy)
    return Polygon(zip(xs, ys)).area / 1e6


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")


def build(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    types = {t["asset_class"]: t for t in
             yaml.safe_load((ROOT / "configs/assets.yaml").read_text(encoding="utf-8"))["asset_types"]}
    pricing = yaml.safe_load((ROOT / "configs/pricing.yaml").read_text(encoding="utf-8"))

    # rasters
    rr, cc = np.mgrid[0:SIZE, 0:SIZE]
    p = np.clip((rr + cc) / (2 * SIZE - 2), 0, 1).astype(np.float32)
    p[:6, :6] = -9999.0
    mask = np.where(p == -9999.0, 255, (p.astype(np.float64) >= THRESHOLD).astype(np.uint8))
    profile = {"driver": "GTiff", "width": SIZE, "height": SIZE, "count": 1, "crs": "EPSG:4326",
                   "transform": from_origin(LON0, LAT0, RES, RES)}
    with rasterio.open(out / "flood_probability.tif", "w", dtype="float32", nodata=-9999.0,
                       **profile) as dst:
        dst.write(p, 1)
    with rasterio.open(out / "flood_mask.tif", "w", dtype="uint8", nodata=255, **profile) as dst:
        dst.write(mask.astype(np.uint8), 1)
    # optional baseline mask (contracts §2, #115): its own rule, 255 exactly where S1 has no data
    base = np.where(p == -9999.0, 255, (p.astype(np.float64) >= BASELINE_CUT).astype(np.uint8))
    with rasterio.open(out / "baseline_mask.tif", "w", dtype="uint8", nodata=255, **profile) as dst:
        dst.write(base.astype(np.uint8), 1)

    # assets and losses
    feats, asset_rows, loss_rows, points = [], [], [], {}
    for aid, cls, (r, c) in ASSETS:
        t = types[cls]
        lon, lat = lonlat(r, c)
        points[aid] = (lon, lat)
        props = {"asset_id": aid, "chip_id": CHIP, "asset_class": cls,
                 "asset_value_rub": t["asset_value_rub"], "vulnerability_coef": t["vulnerability_coef"]}
        feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [lon, lat]},
                      "properties": props})
        asset_rows.append({**props, "longitude": repr(lon), "latitude": repr(lat)})
        pv = float(p[r, c])
        if pv == -9999.0:
            loss_rows.append({"asset_id": aid, "chip_id": CHIP, "p_flood": "",
                              "expected_loss_rub": "", "uncertainty": "", "rank": "",
                              "status": "no_data"})
        else:
            el = q(pv * t["asset_value_rub"] * t["vulnerability_coef"], "0.01")
            loss_rows.append({"asset_id": aid, "chip_id": CHIP, "p_flood": repr(pv),
                              "expected_loss_rub": f"{el:.2f}", "uncertainty": repr(q(pv * (1 - pv), "0.000001")),
                              "rank": "", "status": "ok"})
    ok = sorted((x for x in loss_rows if x["status"] == "ok"),
                key=lambda x: (-float(x["expected_loss_rub"]), x["asset_id"]))
    for i, x in enumerate(ok, start=1):
        x["rank"] = str(i)
    el = {x["asset_id"]: float(x["expected_loss_rub"]) for x in ok}
    write_json(out / "assets.geojson", {"type": "FeatureCollection", "features": feats})
    write_csv(out / "assets.csv", asset_rows)
    write_csv(out / "asset_loss.csv", loss_rows)

    # candidate zones
    rate = pricing["base_rate"]
    cat, polys = {}, {}
    for zid, centre, side, sensor, res, acq, role, obs, avail, level in ZONES:
        poly = square(*points[centre], side)
        polys[zid] = poly
        cat[zid] = {"candidate_id": zid, "sensor_type": sensor, "resolution_m": res,
                    "acquisition_type": acq, "data_role": role, "observation_at": obs,
                    "available_at": avail, "processing_level": level, "usage_type": "internal",
                    "guaranteed_purchase": acq == "new", "area_km2": q(area_km2(poly), "0.001"),
                    "base_rate_rub_km2": rate["rub_per_km2"], "base_rate_status": rate["status"],
                    "base_rate_source": rate["source"]}
    write_json(out / "candidate_orders.geojson", {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": mapping(polys[z]), "properties": cat[z]} for z in cat]})

    # plan: discount per (strategy, sensor, resolution, year) basket, recomputed as a whole
    fresh = {"new": 1.8, "operational": 1.0, "archive": 0.6}
    plan_rows = []
    for s in ("B", "C"):
        groups: dict[tuple, list[str]] = {}
        for zid in PLANS[s]:
            groups.setdefault((cat[zid]["sensor_type"], cat[zid]["resolution_m"]), []).append(zid)
        for (sensor, res), zids in groups.items():
            gid = f"{sensor}_{res}m_2017"  # D-017: family_rm_year
            S = sum(cat[z]["area_km2"] for z in zids)
            k = pricing["discount"][sensor]
            disc = q(min(1.0, max(k["floor"], k["slope"] * math.log(S / res**2) + k["intercept"])),
                     "0.000000001")
            for z in zids:
                c = cat[z]
                o = pricing["processing_coef"][c["processing_level"]]
                u = pricing["usage_coef"][c["usage_type"]]
                t = fresh[c["acquisition_type"]]
                unit = float(dmul(c["base_rate_rub_km2"], o, u, t, disc).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP))
                cost = float(dmul(unit, c["area_km2"]).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
                plan_rows.append({
                    "strategy": s, "candidate_id": z, "area_km2": f"{c['area_km2']:.3f}",
                    "base_rate_rub_km2": c["base_rate_rub_km2"], "base_rate_status": c["base_rate_status"],
                    "processing_level": c["processing_level"], "usage_type": c["usage_type"],
                    "guaranteed_purchase": str(c["guaranteed_purchase"]).lower(),
                    "processing_coef": o, "usage_coef": u, "freshness_coef": t,
                    "discount_coef": f"{disc:.9f}", "discount_group_id": gid,
                    "group_area_km2": f"{S:.3f}", "unit_price_rub_km2": f"{unit:.2f}",
                    "cost_rub": f"{cost:.2f}", "formula": pricing["formula"],
                    "legal_edition": pricing["legal_edition"], "sensor_type": sensor,
                    "resolution_m": res, "acquisition_type": c["acquisition_type"]})
    write_csv(out / "procurement_plan.csv", plan_rows)
    write_json(out / "strategy_plans.json", PLANS)

    # comparison
    total_el = sum(el.values())
    comp = []
    for s in ("A", "B", "C"):
        data_cost = round(sum(float(r["cost_rub"]) for r in plan_rows if r["strategy"] == s), 2)
        other = 1500.0  # analyst time for manual check, disclosed separately
        used = [polys[z] for z in PLANS[s] if cat[z]["data_role"] == "event_observation"
                and cat[z]["available_at"] and cat[z]["available_at"] <= DEADLINE]
        from shapely.geometry import Point
        covered = round(sum(v for a, v in el.items() if any(z.covers(Point(points[a])) for z in used)), 2)
        status = {"A": "baseline", "B": "not_estimated", "C": "scenario"}[s]
        resid = {"A": "0.183", "B": "", "C": "0.150"}[s]
        comp.append({"strategy": s, "data_cost_rub": f"{data_cost:.2f}", "other_cost_rub": f"{other:.2f}",
                     "decision_cost_rub": f"{data_cost + other:.2f}", "budget_rub": f"{BUDGET:.2f}",
                     "budget_feasible": str(data_cost <= BUDGET).lower(),
                     "covered_expected_loss_rub": f"{covered:.2f}",
                     "coverage_share": repr(round(covered / total_el, 6)),
                     "residual_uncertainty": resid, "uncertainty_status": status,
                     "uncertainty_basis": {"A": "средняя p(1−p) по 9 объектам ok",
                                           "B": "эффект новых наблюдений не оценён",
                                           "C": "сценарий: p(1−p) обнуляется у покрытых объектов"}[s]})
    assert comp[1]["budget_feasible"] == "false" and comp[2]["budget_feasible"] == "true", comp
    write_csv(out / "strategy_comparison.csv", comp)

    sens = []
    for i, (key, val) in enumerate([("p_flood", {"delta": 0.1}), ("asset_value_rub", {"factor": 1.2}),
                                    ("vulnerability_coef", {"factor": 0.8}),
                                    ("budget_rub", 10000), ("base_rate_rub_km2", 1000.0)], start=1):
        sens.append({"scenario_id": f"S{i}", "strategy": "C",
                     "changed_inputs_json": json.dumps({key: val}),
                     "decision_cost_rub": comp[2]["decision_cost_rub"],
                     "covered_expected_loss_rub": comp[2]["covered_expected_loss_rub"],
                     "result_status": "recomputed", "interpretation": "синтетический эталон: состав C не меняется"})
    write_csv(out / "sensitivity.csv", sens)

    write_json(out / "run_metadata.json", {
        "contract_version": "1", "run_id": f"{CHIP}_5000_fixture", "chip_id": CHIP,
        "event_id": "Synthetic", "event_date": "2017-05-30", "created_at": "2026-09-25T20:00:00+00:00",
        "commit": "fixture", "threshold": THRESHOLD,
        "source": {"dataset": "синтетический эталон (не Sen1Floods11)", "dataset_url": "https://github.com/cloudtostreet/Sen1Floods11",
                   "s1_files": [f"{CHIP}_S1Hand.tif"], "source_manifest": "source_manifest.json"},
        "split": {"file": "splits/split_v1.json", "sha256": None, "chip_split": "none"},
        "portfolio": {"seed": 20260925, "placement_rule": "фиксированные пиксели эталона",
                      "assets_config": "configs/assets.yaml"},
        "methods": {
            "baseline": {"name": "vh_threshold", "version": "fixture", "weights": [], "threshold": -20.0,
                         "threshold_units": "dB VH", "threshold_source": "эталон"},
            "main": {"name": "gradient", "version": "fixture", "weights": [], "threshold": THRESHOLD,
                     "threshold_units": "p", "threshold_source": "эталон"},
            "evaluation_order": "baseline и main на одной маске валидных пикселей"},
        "expected_loss": {"formula": "EL = p_flood × V × q", "p_source": "flood_probability.tif, пиксель объекта",
                          "status_rules": "ok — пиксель валиден; no_data — nodata; partial — не используется"},
        "budget": {"budget_rub": BUDGET, "decision_at": "2017-05-30T12:00:00+00:00", "decision_deadline": DEADLINE},
        "pricing": {"config": "configs/pricing.yaml", "legal_edition": pricing["legal_edition"],
                    "base_rate_rub_km2": rate["rub_per_km2"], "base_rate_status": rate["status"],
                    "discount_rule": "Р по корзине группы, previous_area_km2 = 0",
                    "grouping": "strategy × sensor_type × resolution_m × calendar_year"},
        "zones": {"construction": "квадраты в EPSG:6933 вокруг объектов", "availability_rule":
                  "охват — только event_observation с available_at ≤ decision_deadline",
                  "area_method": "EPSG:6933, always_xy", "unknown_dates_rule": "observation_at пуст — дата будущей съёмки неизвестна"},
        "strategies": {"rule_B": "все зоны каталога", "rule_C": "зоны event_observation в пределах бюджета",
                       "other_costs": "1500 руб. — ручная проверка аналитиком",
                       "coverage_aggregation": "сумма EL уникальных asset_id под covers", "coverage_share_note": None},
        "uncertainty": {"meaning": "p(1−p) в пикселе объекта", "units": "безразмерная, [0; 0,25]",
                        "method": "аналитически из p", "raster": None},
        "proxy_check": {"status": "not_done", "test_chip": None, "control_points": None, "fixed_v_q": None,
                        "n_points": None, "error_metric": None, "result": None},
        "metrics": {"files": ["metrics.csv"]},
        "reproduce": ["uv run --frozen python tests/contracts/build_fixture.py tests/fixtures/good"],
    })
    h = "0" * 64
    write_csv(out / "metrics.csv", [
        {"method": m, "split": "validation", "event": "ALL", "iou": iou, "f1": f1, "precision": pr,
         "recall": rc, "tp": tp, "fp": fp, "fn": fn, "tn": tn, "n_pixels": 4060,
         "threshold": thr, "valid_mask_sha256": h}
        for m, iou, f1, pr, rc, tp, fp, fn, tn, thr in [
            ("baseline", 0.5, 0.667, 0.6, 0.75, 600, 400, 200, 2860, -20.0),
            ("main", 0.6, 0.75, 0.7, 0.8, 640, 274, 160, 2986, THRESHOLD)]])
    write_json(out / "source_manifest.json", {"sources": [{
        "id": f"{CHIP}_S1Hand", "file": f"{CHIP}_S1Hand.tif", "url": "https://github.com/cloudtostreet/Sen1Floods11",
        "version": "synthetic", "observation_date": "2017-05-30", "available_date": "2017-05-30",
        "license": "CC BY 4.0 (формат как у Sen1Floods11)", "purpose": "S1 демо-чипа эталона",
        "limitations": "синтетика для тестов валидатора", "applies_to": CHIP}]})


if __name__ == "__main__":
    build(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "tests/fixtures/good")
