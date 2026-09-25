"""DEMO kit for the operator UI while the real stages (predict, assets, procurement, export)
are not merged yet.

Everything here is a stand-in, marked `"demo": true` in run_metadata.json and shown as
"ДЕМО" in the interface. It reads only S1 of a validation chip (no labels, D-010), so it
never touches test data. When `floodvalue.procurement.plan` exists, the app uses it
instead of `demo_replan`, and real kits from `make all` replace this one.

    uv run python -m floodvalue.app.demo            # -> outputs/demo_Sri-Lanka_713926/
"""

from __future__ import annotations

import csv
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
import rasterio
import yaml
from pyproj import Transformer
from shapely.geometry import Point, box, mapping, shape
from shapely.ops import transform as shp_transform

from floodvalue.paths import CONFIGS_DIR, data_dir, output_dir

CHIP = "Sri-Lanka_713926"
EVENT = "Sri-Lanka"
S1_DATE = "2017-05-30"
SEED = 20260925
THRESHOLD = 0.5
VH_T, VH_S = -21.69, 1.5  # author VH threshold for the event (metadata), demo slope
BUDGET = 15000.0

_TO_EA = Transformer.from_crs("EPSG:4326", "EPSG:6933", always_xy=True).transform
_FROM_EA = Transformer.from_crs("EPSG:6933", "EPSG:4326", always_xy=True).transform


def q(x, step: str) -> float:
    return float(Decimal(str(x)).quantize(Decimal(step), rounding=ROUND_HALF_UP))


def area_km2(geom) -> float:
    return shp_transform(_TO_EA, geom).area / 1e6


# ---------- pricing (stand-in for P05, follows configs/pricing.yaml) ----------

def load_pricing() -> dict:
    return yaml.safe_load((CONFIGS_DIR / "pricing.yaml").read_text())


def freshness_coef(zone: dict, pr: dict, decision_at: datetime) -> float:
    f = pr["freshness"]
    if zone["acquisition_type"] == "new":
        return f["new_guaranteed"]
    obs = datetime.fromisoformat(zone["observation_at"])
    age = (decision_at - obs).days
    return f["operational"] if age <= f["operational_max_age_days"] else f["archive"]


def price_basket(zones: list[dict], pr: dict, decision_at: datetime, strategy: str) -> list[dict]:
    """Price a whole basket: discount is recomputed per group over the basket."""
    groups: dict[tuple, float] = {}
    for z in zones:
        key = (z["sensor_type"], z["resolution_m"], decision_at.year)
        groups[key] = groups.get(key, 0.0) + z["area_km2"]
    rows = []
    for z in zones:
        key = (z["sensor_type"], z["resolution_m"], decision_at.year)
        s = groups[key] + pr["discount"]["previous_area_km2"]
        d = pr["discount"]["sar" if z["sensor_type"] == "sar" else "optical"]
        disc = min(1.0, max(d["floor"], d["slope"] * math.log(s / z["resolution_m"] ** 2) + d["intercept"]))
        disc = q(disc, pr["rounding"]["discount_coef"])
        o = pr["processing_coef"][z["processing_level"]]
        u = pr["usage_coef"][z["usage_type"]]
        t = freshness_coef(z, pr, decision_at)
        b = pr["base_rate"]["rub_per_km2"]
        unit = q(Decimal(str(b)) * Decimal(str(o)) * Decimal(str(u)) * Decimal(str(t)) * Decimal(str(disc)), "0.01")
        cost = q(Decimal(str(unit)) * Decimal(str(z["area_km2"])), "0.01")
        rows.append({
            "strategy": strategy, "candidate_id": z["candidate_id"], "area_km2": z["area_km2"],
            "base_rate_rub_km2": b, "base_rate_status": pr["base_rate"]["status"],
            "processing_level": z["processing_level"], "usage_type": z["usage_type"],
            "guaranteed_purchase": z["guaranteed_purchase"], "processing_coef": o, "usage_coef": u,
            "freshness_coef": t, "discount_coef": disc,
            "discount_group_id": f"{strategy}-{key[0]}-{key[1]}m-{key[2]}",
            "group_area_km2": q(s, "0.001"), "unit_price_rub_km2": unit, "cost_rub": cost,
            "formula": pr["formula"], "legal_edition": pr["legal_edition"],
            "sensor_type": z["sensor_type"], "resolution_m": z["resolution_m"],
            "acquisition_type": z["acquisition_type"],
        })
    return rows


# ---------- coverage and plan (stand-in for P07) ----------

def covered(zone_ids, zones_by_id, assets, loss, deadline) -> set[str]:
    out = set()
    for cid in zone_ids:
        z = zones_by_id[cid]
        if z["data_role"] != "event_observation" or not z["available_at"]:
            continue
        if datetime.fromisoformat(z["available_at"]) > deadline:
            continue
        for a in assets:
            if z["_geom"].covers(Point(a["longitude"], a["latitude"])) and loss[a["asset_id"]]["status"] == "ok":
                out.add(a["asset_id"])
    return out


def demo_replan(kit: dict, budget: float) -> dict:
    """Greedy C under budget with full basket repricing; B = every event_observation zone."""
    pr, meta = load_pricing(), kit["meta"]
    decision_at = datetime.fromisoformat(meta["decision"]["decision_at"])
    deadline = datetime.fromisoformat(meta["decision"]["decision_deadline"])
    zones = kit["zones"]
    by_id = {z["candidate_id"]: z for z in zones}
    loss = {r["asset_id"]: r for r in kit["loss"]}
    assets = kit["assets"]
    el = {k: (v["expected_loss_rub"] or 0.0) for k, v in loss.items()}
    unc = {k: (v["uncertainty"] or 0.0) for k, v in loss.items()}
    total_el = sum(el.values())

    b_ids = [z["candidate_id"] for z in zones if z["data_role"] == "event_observation"]
    c_ids: list[str] = []
    reasons: dict[str, str] = {}
    while True:
        have = covered(c_ids, by_id, assets, loss, deadline)
        cost_now = sum(r["cost_rub"] for r in price_basket([by_id[i] for i in c_ids], pr, decision_at, "C"))
        best = None
        for z in zones:
            cid = z["candidate_id"]
            if cid in c_ids or z["data_role"] != "event_observation":
                continue
            new = covered([cid], by_id, assets, loss, deadline) - have
            gain = sum(el[a] * (1 + unc[a]) for a in new)
            if gain <= 0:
                continue
            cost = sum(r["cost_rub"] for r in price_basket([by_id[i] for i in c_ids + [cid]], pr, decision_at, "C"))
            if cost > budget:
                continue
            score = gain / max(cost - cost_now, 0.01)
            if best is None or score > best[0]:
                best = (score, cid, new, cost - cost_now)
        if best is None:
            break
        _, cid, new, dc = best
        c_ids.append(cid)
        names = ", ".join(sorted(new))
        reasons[cid] = f"новый охват: {names}; EL {sum(el[a] for a in new):,.0f} руб.; прирост цены корзины {dc:,.2f} руб.".replace(",", " ")

    plan_rows, comparison = [], []
    other = meta["costs"]["other_cost_rub"]
    for s, ids in (("A", []), ("B", b_ids), ("C", c_ids)):
        rows = price_basket([by_id[i] for i in ids], pr, decision_at, s)
        plan_rows += rows
        data_cost = q(sum(r["cost_rub"] for r in rows), "0.01")
        cov = covered(ids, by_id, assets, loss, deadline)
        cov_el = sum(el[a] for a in cov)
        base_unc = sum(unc[a] for a in el)
        comparison.append({
            "strategy": s, "data_cost_rub": data_cost, "other_cost_rub": other,
            "decision_cost_rub": q(data_cost + other, "0.01"), "budget_rub": budget,
            "budget_feasible": data_cost <= budget, "covered_expected_loss_rub": q(cov_el, "0.01"),
            "coverage_share": q(cov_el / total_el, "0.0001") if total_el > 0 else None,
            "residual_uncertainty": q(base_unc, "0.0001") if s == "A" else None,
            "uncertainty_status": "baseline" if s == "A" else "not_estimated",
            "uncertainty_basis": "сумма энтропии p по оценённым объектам" if s == "A"
            else "новых наблюдений нет — снижение не заявляется",
        })
    return {"engine": "demo", "budget_rub": budget, "plans": {"A": [], "B": b_ids, "C": c_ids},
            "plan_rows": plan_rows, "comparison": comparison, "reasons": reasons}


# ---------- kit builder ----------

def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if v is None else str(v).lower() if isinstance(v, bool) else v) for k, v in r.items()})


def build(out: Path | None = None) -> Path:
    s1_rel = f"raw/HandLabeled/S1Hand/{CHIP}_S1Hand.tif"
    s1_path = data_dir() / s1_rel
    out = out or output_dir() / f"demo_{CHIP}"
    out.mkdir(parents=True, exist_ok=True)
    with rasterio.open(s1_path) as r:
        s1 = r.read().astype("float32")
        prof, tr, bounds = r.profile, r.transform, r.bounds
    valid = np.isfinite(s1).all(axis=0)
    p = 1.0 / (1.0 + np.exp((s1[1] - VH_T) / VH_S))
    p = np.where(valid, p, np.nan).astype("float32")
    eps = 1e-6
    ent = -(p * np.log2(p + eps) + (1 - p) * np.log2(1 - p + eps))
    prof.update(count=1, compress="deflate")
    with rasterio.open(out / "flood_probability.tif", "w", **{**prof, "dtype": "float32", "nodata": -9999}) as w:
        w.write(np.where(valid, p, -9999).astype("float32"), 1)
    with rasterio.open(out / "uncertainty.tif", "w", **{**prof, "dtype": "float32", "nodata": -9999}) as w:
        w.write(np.where(valid, ent, -9999).astype("float32"), 1)
    with rasterio.open(out / "flood_mask.tif", "w", **{**prof, "dtype": "uint8", "nodata": 255}) as w:
        w.write(np.where(valid, (p >= THRESHOLD).astype("uint8"), 255).astype("uint8"), 1)

    # portfolio: random valid pixel per class, fixed seed, labels never read
    types = yaml.safe_load((CONFIGS_DIR / "assets.yaml").read_text())["asset_types"]
    rng = np.random.default_rng(SEED)
    assets, loss = [], []
    h, wd = valid.shape
    for i, t in enumerate(types, 1):
        while True:
            row, col = int(rng.integers(0, h)), int(rng.integers(0, wd))
            if valid[row, col]:
                break
        lon, lat = rasterio.transform.xy(tr, row, col)
        aid = f"A{i:02d}_{t['asset_class']}"
        assets.append({"asset_id": aid, "chip_id": CHIP, "asset_class": t["asset_class"],
                       "asset_value_rub": t["asset_value_rub"], "vulnerability_coef": t["vulnerability_coef"],
                       "longitude": round(lon, 7), "latitude": round(lat, 7), "_rc": (row, col)})
    for a in assets:
        row, col = a["_rc"]
        pv = float(p[row, col])
        loss.append({"asset_id": a["asset_id"], "chip_id": CHIP, "p_flood": round(pv, 6),
                     "expected_loss_rub": round(pv * a["asset_value_rub"] * a["vulnerability_coef"], 2),
                     "uncertainty": round(float(ent[row, col]), 6), "rank": None, "status": "ok"})
    for rank, r in enumerate(sorted(loss, key=lambda r: (-r["expected_loss_rub"], r["asset_id"])), 1):
        r["rank"] = rank
    for a in assets:
        a.pop("_rc")

    # candidate zones: square around each asset (~1.2 km side), two cluster zones, one archive context
    decision_at = datetime.fromisoformat(S1_DATE + "T18:00:00+00:00")
    deadline = decision_at + timedelta(hours=12)
    zones = []

    def add(geom, **kw):
        a = area_km2(geom)
        assert a >= 1.0, a
        z = {"candidate_id": f"Z{len(zones) + 1:02d}", **kw, "area_km2": q(a, "0.001"),
             "base_rate_rub_km2": 788.64, "base_rate_status": "scenario",
             "base_rate_source": "Учебная сценарная ставка кейса; не официальный тариф", "_geom": geom}
        zones.append(z)

    new_opt = dict(sensor_type="optical", resolution_m=1.0, acquisition_type="new", data_role="event_observation",
                   observation_at=None, available_at=(decision_at + timedelta(hours=8)).isoformat(),
                   processing_level="L2", usage_type="internal", guaranteed_purchase=True)
    new_sar = dict(sensor_type="sar", resolution_m=3.0, acquisition_type="new", data_role="event_observation",
                   observation_at=None, available_at=(decision_at + timedelta(hours=6)).isoformat(),
                   processing_level="L1", usage_type="internal", guaranteed_purchase=True)
    half = 610.0  # metres in EPSG:6933
    for a in assets:
        x, y = _TO_EA(a["longitude"], a["latitude"])
        geom = shp_transform(_FROM_EA, box(x - half, y - half, x + half, y + half))
        add(geom, **new_opt)
    xs = [a["longitude"] for a in assets]
    ys = [a["latitude"] for a in assets]
    mid = (min(xs) + max(xs)) / 2
    west = box(bounds.left, bounds.bottom, mid, bounds.top)
    east = box(mid, bounds.bottom, bounds.right, bounds.top)
    add(west, **new_sar)
    add(east, **new_sar)
    late = dict(new_opt, available_at=(decision_at + timedelta(hours=30)).isoformat())
    x, y = _TO_EA(assets[0]["longitude"], assets[0]["latitude"])
    add(shp_transform(_FROM_EA, box(x - 900, y - 900, x + 900, y + 900)), **late)
    arch = dict(sensor_type="optical", resolution_m=1.0, acquisition_type="archive", data_role="context",
                observation_at=(decision_at - timedelta(days=200)).isoformat(),
                available_at=decision_at.isoformat(), processing_level="L2", usage_type="internal",
                guaranteed_purchase=False)
    add(box(bounds.left, bounds.bottom, bounds.right, bounds.top), **arch)

    meta = {
        "run_id": f"demo_{CHIP}", "demo": True,
        "demo_note": "ДЕМО-комплект интерфейса: p — сигмоида от VH вокруг порога авторов события, "
                     "зоны и цена — заглушки до P05/P06/P07. Не результат модели.",
        "chip_id": CHIP, "event_id": EVENT, "s1_date": S1_DATE, "s1_file": s1_rel,
        "split": "validation (D-010: до заморозки UI гоняется на validation)",
        "portfolio": {"seed": SEED, "rule": "случайный пиксель с конечными VV и VH, по одному на тип; метки не читаются"},
        "methods": {"main": {"name": "demo-sigmoid-vh", "threshold": THRESHOLD, "vh_threshold_db": VH_T, "slope_db": VH_S},
                    "baseline": {"name": "vh-threshold", "threshold_db": VH_T}},
        "threshold": THRESHOLD, "el_model": "EL = p × V × q, p — значение flood_probability.tif в пикселе объекта",
        "uncertainty": {"asset_field": "энтропия p в битах (0 — уверенно, 1 — p = 0,5)", "raster": "uncertainty.tif"},
        "decision": {"decision_at": decision_at.isoformat(), "decision_deadline": deadline.isoformat(),
                     "user": "инфраструктурный оператор (D-001)"},
        "budget_rub": BUDGET,
        "costs": {"other_cost_rub": 12000.0, "other_cost_basis": "сценарий: 8 ч аналитика × 1 500 руб."},
        "pricing": {"legal_edition": load_pricing()["legal_edition"], "base_rate_status": "scenario"},
        "strategy_rules": {"B": "все зоны event_observation каталога", "C": "жадно: прирост EL×(1+неопределённость) на рубль прироста цены корзины, в пределах бюджета"},
        "proxy_check": {"status": "not_estimated", "reason": "демо-комплект"},
        "reproduce": ["uv run python -m floodvalue.app.demo", "make app"],
    }
    kit = {"meta": meta, "zones": zones, "assets": assets, "loss": loss}
    plan = demo_replan(kit, BUDGET)

    fc = lambda feats: {"type": "FeatureCollection", "features": feats}  # noqa: E731
    (out / "assets.geojson").write_text(json.dumps(fc([
        {"type": "Feature", "geometry": {"type": "Point", "coordinates": [a["longitude"], a["latitude"]]},
         "properties": {k: v for k, v in a.items() if k not in ("longitude", "latitude")}} for a in assets]), ensure_ascii=False, indent=1))
    _write_csv(out / "assets.csv", assets)
    _write_csv(out / "asset_loss.csv", loss)
    (out / "candidate_orders.geojson").write_text(json.dumps(fc([
        {"type": "Feature", "geometry": mapping(z["_geom"]), "properties": {k: v for k, v in z.items() if k != "_geom"}}
        for z in zones]), ensure_ascii=False, indent=1))
    _write_csv(out / "procurement_plan.csv", plan["plan_rows"])
    (out / "strategy_plans.json").write_text(json.dumps(plan["plans"], ensure_ascii=False))
    _write_csv(out / "strategy_comparison.csv", plan["comparison"])

    sens = []
    for b in (3000, 8000, 15000, 30000, 60000):
        r = demo_replan(kit, b)
        c = next(x for x in r["comparison"] if x["strategy"] == "C")
        sens.append({"scenario_id": f"budget_{b}", "strategy": "C", "changed_inputs_json": json.dumps({"budget_rub": b}),
                     "decision_cost_rub": c["decision_cost_rub"], "covered_expected_loss_rub": c["covered_expected_loss_rub"],
                     "result_status": "ok", "interpretation": f"в корзине {len(r['plans']['C'])} зон"})
    top = min(loss, key=lambda r: r["rank"])
    a_top = next(a for a in assets if a["asset_id"] == top["asset_id"])
    for name, mult in (("p", 0.5), ("p", 1.5), ("V", 0.8), ("V", 1.2), ("q", 0.8), ("q", 1.2)):
        pv = min(1.0, top["p_flood"] * mult) if name == "p" else top["p_flood"]
        v = a_top["asset_value_rub"] * (mult if name == "V" else 1)
        qq = a_top["vulnerability_coef"] * (mult if name == "q" else 1)
        sens.append({"scenario_id": f"{name}_x{mult}_{top['asset_id']}", "strategy": "A",
                     "changed_inputs_json": json.dumps({"asset_id": top["asset_id"], name: f"×{mult}"}),
                     "decision_cost_rub": None, "covered_expected_loss_rub": round(pv * v * qq, 2),
                     "result_status": "scenario", "interpretation": f"EL объекта {top['asset_id']} при {name} ×{mult}"})
    _write_csv(out / "sensitivity.csv", sens)
    (out / "run_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    (out / "source_manifest.json").write_text(json.dumps([{
        "id": f"{CHIP}_S1Hand", "file": s1_rel,
        "url": f"https://storage.googleapis.com/sen1floods11/v1.1/data/flood_events/HandLabeled/S1Hand/{CHIP}_S1Hand.tif",
        "version": "Sen1Floods11 v1.1", "observed_at": S1_DATE, "available_at": "2020-10-01",
        "license": "не указана авторами (STAC: proprietary), цитирование Bonafilia et al. 2020",
        "purpose": "S1 VV/VH демо-чипа", "limitations": "validation-чип, только для демонстрации интерфейса"}], ensure_ascii=False, indent=1))
    return out


if __name__ == "__main__":
    print(build(Path(sys.argv[1]) if len(sys.argv) > 1 else None))
