"""`python -m floodvalue validate --out <run_dir>` — check the 12 required files (criterion 12).

Per-file checks follow docs/contracts.md §6. Each error is attributed to one of the 12 files;
the run passes only when all 12 have no errors. Without --out the newest complete set in
FLOODVALUE_OUTPUT_DIR is checked, or tests/fixtures/good when there is none (D-014).
"""

import math
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import yaml
from pydantic import BaseModel, ValidationError
from rasterio.warp import transform as warp_transform
from shapely.geometry import Point, shape

from floodvalue.contracts import files as F
from floodvalue.contracts.geo import MIN_AREA_KM2, area_km2, product_half_up, round_half_up
from floodvalue.contracts.io import read_csv, read_json
from floodvalue.contracts.schemas import (AssetCsvRow, AssetLossRow, AssetProperties,
                                          CandidateProperties, ProcurementPlanRow, RunMetadata,
                                          SensitivityRow, SourceManifest, StrategyComparisonRow,
                                          StrategyPlans)
from floodvalue.paths import CONFIGS_DIR, REPO_ROOT, data_dir, output_dir

FIXTURE_RUN = REPO_ROOT / "tests" / "fixtures" / "good"
MONEY_TOL = 0.005  # half a kopeck: recomputed kopeck amounts must match exactly
EL_TOL = 0.01  # EL written to the kopeck
P_TOL = 1e-6  # p_flood vs float32 raster value
COORD_TOL = 1e-7  # degrees, ~1 cm


def _utc(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=UTC)


def _fmt(exc: ValidationError) -> str:
    return "; ".join(f"{'.'.join(map(str, e['loc'])) or '-'}: {e['msg']}" for e in exc.errors())


class _Report:
    def __init__(self) -> None:
        self.errors: dict[str, list[str]] = {name: [] for name in F.REQUIRED_FILES}

    def add(self, name: str, message: str) -> None:
        self.errors[name].append(message)


def _load_csv(report: _Report, run_dir: Path, name: str, fields: tuple[str, ...],
              model: type[BaseModel]) -> list[Any] | None:
    header, raw, errors = read_csv(run_dir / name)
    for e in errors:
        report.add(name, e)
    missing = [f for f in fields if f not in header]
    if missing:
        report.add(name, f"нет обязательных столбцов: {missing}")
        return None
    rows = []
    for i, row in enumerate(raw, start=2):
        try:
            rows.append(model.model_validate(row))
        except ValidationError as exc:
            report.add(name, f"строка {i}: {_fmt(exc)}")
    return rows if len(rows) == len(raw) else None


def _load_features(report: _Report, run_dir: Path, name: str, geom_type: str,
                   model: type[BaseModel]) -> list[tuple[Any, Any]] | None:
    doc = read_json(run_dir / name)
    if not isinstance(doc, dict) or doc.get("type") != "FeatureCollection":
        report.add(name, "ожидается FeatureCollection")
        return None
    out = []
    for i, feat in enumerate(doc.get("features") or []):
        geom = (feat or {}).get("geometry") or {}
        if geom.get("type") != geom_type:
            report.add(name, f"объект {i}: геометрия {geom.get('type')!r}, ожидается {geom_type}")
            continue
        try:
            props = model.model_validate(feat.get("properties") or {})
            out.append((props, shape(geom)))
        except (ValidationError, ValueError, TypeError) as exc:
            msg = _fmt(exc) if isinstance(exc, ValidationError) else str(exc)
            report.add(name, f"объект {i}: {msg}")
    return out if len(out) == len(doc.get("features") or []) else None


def _lonlat_ok(geom) -> bool:
    xs, ys = np.asarray(geom.exterior.coords if geom.geom_type == "Polygon" else geom.coords).T
    return bool(np.all(np.abs(xs) <= 180) and np.all(np.abs(ys) <= 90))


# --- rasters -----------------------------------------------------------------------------------

def _find_s1(chip_id: str) -> Path | None:
    root = data_dir()
    if not root.is_dir():
        return None
    return next(root.rglob(f"{chip_id}_S1Hand.tif"), None)


def _check_rasters(report: _Report, run_dir: Path, meta: RunMetadata | None):
    """Return (prob array, valid mask, profile) for later sampling, or None."""
    P, M = F.FLOOD_PROBABILITY, F.FLOOD_MASK
    with rasterio.open(run_dir / P) as src:
        p_prof, p = src.profile, src.read()
    with rasterio.open(run_dir / M) as src:
        m_prof, m = src.profile, src.read()
    if p.shape[0] != 1:
        report.add(P, f"каналов {p.shape[0]}, нужен 1")
    if p_prof["dtype"] != F.PROBABILITY_DTYPE:
        report.add(P, f"тип {p_prof['dtype']}, нужен Float32")
    if p_prof.get("nodata") != F.PROBABILITY_NODATA:
        report.add(P, f"nodata {p_prof.get('nodata')}, нужен -9999")
    if m.shape[0] != 1:
        report.add(M, f"каналов {m.shape[0]}, нужен 1")
    if m_prof["dtype"] != F.MASK_DTYPE:
        report.add(M, f"тип {m_prof['dtype']}, нужен UInt8")
    if m_prof.get("nodata") != F.MASK_NODATA:
        report.add(M, f"nodata {m_prof.get('nodata')}, нужен 255")
    p, m = p[0].astype(np.float64), m[0]
    if not np.isfinite(p).all():
        report.add(P, f"NaN/Inf в {int((~np.isfinite(p)).sum())} пикселях (нет данных = -9999)")
    valid = np.isfinite(p) & (p != F.PROBABILITY_NODATA)
    if valid.any() and (p[valid].min() < 0 or p[valid].max() > 1):
        report.add(P, "значения вне [0, 1]")
    bad_values = ~np.isin(m, (0, 1, F.MASK_NODATA))
    if bad_values.any():
        report.add(M, f"значения кроме 0/1/255 в {int(bad_values.sum())} пикселях")
    if p_prof.get("crs") is None:
        report.add(P, "нет CRS")
    same_grid = (p.shape == m.shape and p_prof.get("crs") == m_prof.get("crs")
                 and np.allclose(tuple(p_prof["transform"]), tuple(m_prof["transform"]),
                                 rtol=0, atol=1e-12))
    if not same_grid:
        report.add(M, "сетка (CRS, transform, размер) не совпадает с flood_probability.tif")
        return None
    nodata_mismatch = valid != (m != F.MASK_NODATA)
    if nodata_mismatch.any():
        report.add(M, f"nodata не согласован с вероятностью в {int(nodata_mismatch.sum())} пикселях")
    if meta is not None:
        disagree = valid & ((p >= meta.threshold) != (m == 1))
        if disagree.any():
            report.add(M, f"mask ≠ (p ≥ threshold={meta.threshold}) в {int(disagree.sum())} пикселях")
        s1 = _find_s1(meta.chip_id)
        if s1 is not None:
            with rasterio.open(s1) as src:
                if (src.shape != p.shape or src.crs != p_prof.get("crs")
                        or not np.allclose(tuple(src.transform), tuple(p_prof["transform"]),
                                           rtol=0, atol=1e-9)):
                    report.add(P, f"сетка не совпадает с S1-чипом {s1.name}")
    return p, valid, p_prof


def _sample(raster, lon: float, lat: float) -> tuple[bool, float | None]:
    """(inside chip, p at the asset pixel or None if nodata)."""
    p, valid, prof = raster
    xs, ys = [lon], [lat]
    if prof["crs"] and prof["crs"].to_epsg() != 4326:
        xs, ys = warp_transform("EPSG:4326", prof["crs"], xs, ys)
    col, row = ~prof["transform"] @ (xs[0], ys[0])
    r, c = math.floor(row), math.floor(col)
    if not (0 <= r < p.shape[0] and 0 <= c < p.shape[1]):
        return False, None
    return True, (float(p[r, c]) if valid[r, c] else None)


# --- pricing recomputation (configs/pricing.yaml) ----------------------------------------------

def _family(pricing: dict, sensor_type: str) -> str:
    for family, members in pricing["discount"]["sensor_family"].items():
        if sensor_type in members:
            return family
    raise ValueError(f"sensor_type {sensor_type!r} не найден в pricing.discount.sensor_family")


def _discount(pricing: dict, sensor_type: str, resolution_m: float, group_area: float) -> float:
    k = pricing["discount"][_family(pricing, sensor_type)]
    raw = k["slope"] * math.log(group_area / resolution_m**2) + k["intercept"]
    return round_half_up(min(1.0, max(k["floor"], raw)), pricing["rounding"]["discount_coef"])


def _check_plan(report: _Report, plan: list[ProcurementPlanRow], catalog: dict, pricing: dict):
    name = F.PROCUREMENT_PLAN
    pairs = Counter((r.strategy, r.candidate_id) for r in plan)
    for pair, n in pairs.items():
        if n > 1:
            report.add(name, f"пара (strategy, candidate_id)={pair} повторяется {n} раз")
    fresh = pricing["freshness"]
    fresh_by_type = {"new": fresh["new_guaranteed"], "operational": fresh["operational"],
                     "archive": fresh["archive"]}
    groups: dict[tuple[str, str], list[ProcurementPlanRow]] = defaultdict(list)
    for r in plan:
        where = f"{r.strategy}/{r.candidate_id}"
        c = catalog.get(r.candidate_id)
        if c is None:
            report.add(name, f"{where}: candidate_id нет в candidate_orders.geojson")
            continue
        groups[(r.strategy, r.discount_group_id)].append(r)
        for field in ("area_km2", "base_rate_rub_km2", "base_rate_status", "processing_level",
                      "usage_type", "guaranteed_purchase"):
            if getattr(r, field) != getattr(c, field):
                report.add(name, f"{where}: {field}={getattr(r, field)!r} ≠ каталогу "
                                 f"{getattr(c, field)!r}")
        extra = r.model_extra or {}
        for field in ("sensor_type", "acquisition_type"):
            if extra.get(field) not in (None, "") and extra[field] != getattr(c, field):
                report.add(name, f"{where}: {field} ≠ каталогу")
        if extra.get("resolution_m") not in (None, "") and float(extra["resolution_m"]) != c.resolution_m:
            report.add(name, f"{where}: resolution_m ≠ каталогу")
        expected = {"processing_coef": pricing["processing_coef"][r.processing_level],
                    "usage_coef": pricing["usage_coef"][r.usage_type],
                    "freshness_coef": fresh_by_type[c.acquisition_type]}
        for field, value in expected.items():
            if not math.isclose(getattr(r, field), value, abs_tol=1e-12):
                report.add(name, f"{where}: {field}={getattr(r, field)} ≠ {value} по pricing.yaml")
        unit = product_half_up([r.base_rate_rub_km2, r.processing_coef, r.usage_coef,
                                r.freshness_coef, r.discount_coef],
                               pricing["rounding"]["unit_price_rub_km2"])
        if abs(unit - r.unit_price_rub_km2) > MONEY_TOL:
            report.add(name, f"{where}: unit_price_rub_km2={r.unit_price_rub_km2} ≠ пересчёту {unit}")
        cost = product_half_up([r.unit_price_rub_km2, r.area_km2], pricing["rounding"]["cost_rub"])
        if abs(cost - r.cost_rub) > MONEY_TOL:
            report.add(name, f"{where}: cost_rub={r.cost_rub} ≠ пересчёту {cost}")
        if r.legal_edition != pricing["legal_edition"]:
            report.add(name, f"{where}: legal_edition ≠ configs/pricing.yaml")
    for (strategy, gid), rows in groups.items():
        where = f"{strategy}/группа {gid}"
        try:
            kinds = {(_family(pricing, catalog[r.candidate_id].sensor_type),
                      catalog[r.candidate_id].resolution_m) for r in rows}
        except ValueError as exc:
            report.add(name, f"{where}: {exc}")
            continue
        if len(kinds) > 1:
            report.add(name, f"{where}: разные семейства сенсора/resolution_m в одной группе скидки")
            continue
        _, res = kinds.pop()
        basket = sum(r.area_km2 for r in rows)
        if len({r.group_area_km2 for r in rows}) > 1:
            report.add(name, f"{where}: group_area_km2 различается внутри группы")
        for r in rows:
            # S = basket area of the group + previously ordered area (>= 0, default 0)
            if r.group_area_km2 < basket - 1e-6:
                report.add(name, f"{where}: group_area_km2={r.group_area_km2} меньше площади "
                                 f"корзины группы {round(basket, 6)} — скидка не по корзине целиком")
            disc = _discount(pricing, catalog[r.candidate_id].sensor_type, res, r.group_area_km2)
            if abs(disc - r.discount_coef) > 1.5e-9:
                report.add(name, f"{where}/{r.candidate_id}: discount_coef={r.discount_coef} ≠ "
                                 f"пересчёту по корзине {disc}")


# --- main entry --------------------------------------------------------------------------------

def validate_run(run_dir: Path, assets_config: Path | None = None,
                 pricing_config: Path | None = None) -> dict[str, list[str]]:
    """Validate one result set. Returns {file name: [errors]} for all 12 required files."""
    run_dir = Path(run_dir)
    report = _Report()
    for name in F.REQUIRED_FILES:
        if not (run_dir / name).is_file():
            report.add(name, "файл отсутствует")
    present = {n for n in F.REQUIRED_FILES if not report.errors[n]}

    def guarded(name, fn):
        if name not in present:
            return None
        try:
            return fn()
        except Exception as exc:  # malformed file: report, keep checking the others
            report.add(name, f"не читается: {type(exc).__name__}: {exc}")
            return None

    with open(assets_config or CONFIGS_DIR / "assets.yaml", encoding="utf-8") as f:
        asset_types = {t["asset_class"]: t for t in yaml.safe_load(f)["asset_types"]}
    with open(pricing_config or CONFIGS_DIR / "pricing.yaml", encoding="utf-8") as f:
        pricing = yaml.safe_load(f)

    # run_metadata.json -------------------------------------------------------------------------
    def load_meta():
        try:
            return RunMetadata.model_validate(read_json(run_dir / F.RUN_METADATA))
        except ValidationError as exc:
            report.add(F.RUN_METADATA, _fmt(exc))
            return None
    meta = guarded(F.RUN_METADATA, load_meta)
    if meta is None and F.RUN_METADATA in present and not report.errors[F.RUN_METADATA]:
        report.add(F.RUN_METADATA, "не прочитан")
    if meta is None:
        for n in (F.FLOOD_MASK, F.STRATEGY_COMPARISON, F.SOURCE_MANIFEST):
            report.add(n, "перекрёстная проверка невозможна: run_metadata.json невалиден")
    elif not any(meta.chip_id in s for s in meta.source.s1_files):
        report.add(F.RUN_METADATA, "source.s1_files не содержит S1 чипа chip_id")

    # rasters -----------------------------------------------------------------------------------
    raster = None
    if F.FLOOD_PROBABILITY in present and F.FLOOD_MASK in present:
        try:
            raster = _check_rasters(report, run_dir, meta)
        except Exception as exc:
            report.add(F.FLOOD_PROBABILITY, f"растры не читаются: {type(exc).__name__}: {exc}")

    # assets.geojson / assets.csv ---------------------------------------------------------------
    assets = guarded(F.ASSETS_GEOJSON, lambda: _load_features(
        report, run_dir, F.ASSETS_GEOJSON, "Point", AssetProperties))
    by_id: dict[str, tuple[AssetProperties, Point]] = {}
    if assets is not None:
        name = F.ASSETS_GEOJSON
        ids = Counter(a.asset_id for a, _ in assets)
        for aid, n in ids.items():
            if n > 1:
                report.add(name, f"asset_id {aid} повторяется {n} раз")
        if len(assets) != 10:
            report.add(name, f"объектов {len(assets)}, нужно ровно 10")
        classes = Counter(a.asset_class for a, _ in assets)
        if set(classes) != set(asset_types) or any(n != 1 for n in classes.values()):
            report.add(name, f"нужно по одному объекту каждого типа configs/assets.yaml, есть {dict(classes)}")
        for a, pt in assets:
            by_id.setdefault(a.asset_id, (a, pt))
            t = asset_types.get(a.asset_class)
            if t and (not math.isclose(a.asset_value_rub, t["asset_value_rub"], abs_tol=1e-9)
                      or not math.isclose(a.vulnerability_coef, t["vulnerability_coef"], abs_tol=1e-12)):
                report.add(name, f"{a.asset_id}: V/q не совпадают с configs/assets.yaml")
            if meta and a.chip_id != meta.chip_id:
                report.add(name, f"{a.asset_id}: chip_id {a.chip_id} ≠ run_metadata {meta.chip_id}")
            if not _lonlat_ok(pt):
                report.add(name, f"{a.asset_id}: координаты вне диапазона [lon, lat]")
            elif raster is not None and not _sample(raster, pt.x, pt.y)[0]:
                report.add(name, f"{a.asset_id}: точка вне чипа (порядок [долгота, широта]?)")

    rows = guarded(F.ASSETS_CSV, lambda: _load_csv(
        report, run_dir, F.ASSETS_CSV, F.ASSETS_CSV_FIELDS, AssetCsvRow))
    if rows is not None and assets is not None:
        name = F.ASSETS_CSV
        if len(rows) != len({r.asset_id for r in rows}) or {r.asset_id for r in rows} != set(by_id):
            report.add(name, "набор asset_id не совпадает с assets.geojson (или есть повторы)")
        for r in rows:
            if r.asset_id not in by_id:
                continue
            a, pt = by_id[r.asset_id]
            same = (r.chip_id == a.chip_id and r.asset_class == a.asset_class
                    and math.isclose(r.asset_value_rub, a.asset_value_rub, abs_tol=1e-9)
                    and math.isclose(r.vulnerability_coef, a.vulnerability_coef, abs_tol=1e-12)
                    and abs(r.longitude - pt.x) <= COORD_TOL and abs(r.latitude - pt.y) <= COORD_TOL)
            if not same:
                report.add(name, f"{r.asset_id}: значения или координаты ≠ assets.geojson")

    # asset_loss.csv ----------------------------------------------------------------------------
    losses = guarded(F.ASSET_LOSS, lambda: _load_csv(
        report, run_dir, F.ASSET_LOSS, F.ASSET_LOSS_FIELDS, AssetLossRow))
    el_by_id: dict[str, float] = {}
    if losses is not None:
        name = F.ASSET_LOSS
        ids = Counter(r.asset_id for r in losses)
        for aid, n in ids.items():
            if n > 1:
                report.add(name, f"asset_id {aid} повторяется {n} раз")
        if by_id and set(ids) != set(by_id):
            report.add(name, "набор asset_id не совпадает с assets.geojson")
        for r in losses:
            if meta and r.chip_id != meta.chip_id:
                report.add(name, f"{r.asset_id}: chip_id ≠ run_metadata")
            if r.asset_id not in by_id:
                continue
            a, pt = by_id[r.asset_id]
            p_pix = _sample(raster, pt.x, pt.y)[1] if raster is not None else None
            if r.status == "ok":
                el_by_id[r.asset_id] = r.expected_loss_rub
                expected = r.p_flood * a.asset_value_rub * a.vulnerability_coef
                if abs(expected - r.expected_loss_rub) > EL_TOL + 1e-9 * expected:
                    report.add(name, f"{r.asset_id}: EL={r.expected_loss_rub} ≠ p·V·q={expected:.2f}")
                if raster is not None:
                    if p_pix is None:
                        report.add(name, f"{r.asset_id}: status=ok, но пиксель объекта — nodata")
                    elif abs(p_pix - r.p_flood) > P_TOL:
                        report.add(name, f"{r.asset_id}: p_flood={r.p_flood} ≠ растру {p_pix}")
            elif raster is not None and r.status == "no_data" and p_pix is not None:
                report.add(name, f"{r.asset_id}: status=no_data, но пиксель объекта валиден")
        ok = sorted((r for r in losses if r.status == "ok"),
                    key=lambda r: (-r.expected_loss_rub, r.asset_id))
        for expected_rank, r in enumerate(ok, start=1):
            if r.rank != expected_rank:
                report.add(name, f"{r.asset_id}: rank={r.rank}, по правилу (max EL, затем asset_id) "
                                 f"{expected_rank}")

    # candidate_orders.geojson ------------------------------------------------------------------
    cands = guarded(F.CANDIDATE_ORDERS, lambda: _load_features(
        report, run_dir, F.CANDIDATE_ORDERS, "Polygon", CandidateProperties))
    catalog: dict[str, CandidateProperties] = {}
    zones: dict[str, Any] = {}
    if cands is not None:
        name = F.CANDIDATE_ORDERS
        if not cands:
            report.add(name, "каталог пуст")
        ids = Counter(c.candidate_id for c, _ in cands)
        for cid, n in ids.items():
            if n > 1:
                report.add(name, f"candidate_id {cid} повторяется {n} раз")
        for c, geom in cands:
            cid = c.candidate_id
            catalog.setdefault(cid, c)
            zones.setdefault(cid, geom)
            if not _lonlat_ok(geom):
                report.add(name, f"{cid}: координаты вне диапазона [lon, lat]")
                continue
            if not geom.is_valid:
                report.add(name, f"{cid}: невалидный Polygon")
            if len(geom.interiors) > 0:
                report.add(name, f"{cid}: есть внутренние вырезы (holes)")
            area = area_km2(geom)
            if area < MIN_AREA_KM2:
                report.add(name, f"{cid}: площадь {area:.6f} км² < 1 км² (до округления)")
            if abs(round_half_up(area, "0.001") - c.area_km2) > 1e-9:
                report.add(name, f"{cid}: area_km2={c.area_km2} ≠ ROUND_HALF_UP(EPSG:6933, 0,001)="
                                 f"{round_half_up(area, '0.001')}")

    # procurement_plan.csv ----------------------------------------------------------------------
    plan = guarded(F.PROCUREMENT_PLAN, lambda: _load_csv(
        report, run_dir, F.PROCUREMENT_PLAN, F.PROCUREMENT_PLAN_FIELDS, ProcurementPlanRow))
    if plan is not None and cands is not None:
        _check_plan(report, plan, catalog, pricing)

    # strategy_plans.json -----------------------------------------------------------------------
    def load_plans():
        try:
            return StrategyPlans.model_validate(read_json(run_dir / F.STRATEGY_PLANS))
        except ValidationError as exc:
            report.add(F.STRATEGY_PLANS, _fmt(exc))
            return None
    plans = guarded(F.STRATEGY_PLANS, load_plans)
    if plans is not None:
        name = F.STRATEGY_PLANS
        for s in ("B", "C"):
            ids = getattr(plans, s)
            if cands is not None:
                unknown = [i for i in ids if i not in catalog]
                if unknown:
                    report.add(name, f"{s}: нет в candidate_orders.geojson: {unknown}")
            if plan is not None and set(ids) != {r.candidate_id for r in plan if r.strategy == s}:
                report.add(name, f"{s}: состав ≠ строкам procurement_plan.csv")

    # strategy_comparison.csv -------------------------------------------------------------------
    comp = guarded(F.STRATEGY_COMPARISON, lambda: _load_csv(
        report, run_dir, F.STRATEGY_COMPARISON, F.STRATEGY_COMPARISON_FIELDS, StrategyComparisonRow))
    if comp is not None:
        name = F.STRATEGY_COMPARISON
        counts = Counter(r.strategy for r in comp)
        if sorted(counts.elements()) != list(F.STRATEGIES):
            report.add(name, f"нужно ровно по одной строке A, B, C; есть {dict(counts)}")
        budgets = {r.budget_rub for r in comp}
        if len(budgets) > 1:
            report.add(name, f"budget_rub различается между строками: {sorted(budgets)}")
        if meta and any(abs(b - meta.budget.budget_rub) > MONEY_TOL for b in budgets):
            report.add(name, "budget_rub ≠ run_metadata.budget.budget_rub")
        total_el = sum(el_by_id.values())
        for r in comp:
            s = r.strategy
            if plan is not None:
                data_cost = sum(p.cost_rub for p in plan if p.strategy == s)
                if abs(round(data_cost, 2) - r.data_cost_rub) > MONEY_TOL:
                    report.add(name, f"{s}: data_cost_rub={r.data_cost_rub} ≠ сумме строк плана "
                                     f"{data_cost:.2f}")
            if abs(r.data_cost_rub + r.other_cost_rub - r.decision_cost_rub) > MONEY_TOL:
                report.add(name, f"{s}: decision_cost_rub ≠ data_cost_rub + other_cost_rub")
            within = r.data_cost_rub <= r.budget_rub + 1e-9
            if r.budget_feasible != within:
                report.add(name, f"{s}: budget_feasible={str(r.budget_feasible).lower()}, а "
                                 f"data_cost {'≤' if within else '>'} бюджета")
            if s == "C" and not within:
                report.add(name, f"C: data_cost_rub={r.data_cost_rub} превышает бюджет {r.budget_rub}")
            if s in ("B", "C") and plans is not None and cands is not None and meta and losses:
                deadline = _utc(meta.budget.decision_deadline)
                used = [zones[i] for i in getattr(plans, s)
                        if i in catalog and catalog[i].data_role == "event_observation"
                        and catalog[i].available_at is not None
                        and _utc(catalog[i].available_at) <= deadline]
                covered = covered_expected_loss(used, {a: by_id[a][1] for a in el_by_id}, el_by_id)
                if abs(covered - r.covered_expected_loss_rub) > EL_TOL:
                    report.add(name, f"{s}: covered_expected_loss_rub={r.covered_expected_loss_rub} ≠ "
                                     f"пересчёту {covered:.2f} (covers, event_observation, ≤ deadline)")
            if losses is not None:
                if total_el > 0 and r.coverage_share is None:
                    report.add(name, f"{s}: coverage_share пуст при ненулевом суммарном EL")
                if total_el > 0 and r.coverage_share is not None and \
                        abs(r.coverage_share - r.covered_expected_loss_rub / total_el) > 1e-4:
                    report.add(name, f"{s}: coverage_share ≠ covered_expected_loss_rub / ΣEL")
                if total_el == 0 and r.coverage_share is not None:
                    report.add(name, f"{s}: coverage_share должен быть пуст при ΣEL = 0")

    # sensitivity.csv ---------------------------------------------------------------------------
    sens = guarded(F.SENSITIVITY, lambda: _load_csv(
        report, run_dir, F.SENSITIVITY, F.SENSITIVITY_FIELDS, SensitivityRow))
    if sens is not None:
        name = F.SENSITIVITY
        required = set(F.SENSITIVITY_REQUIRED_KEYS)
        seen = set()
        for r in sens:
            changed = r.changed_keys() & required
            if len(changed) == 1:  # exactly one of p, V, q, budget, rate varied
                seen |= changed
        missing = required - seen
        if missing:
            report.add(name, f"нет сценариев, меняющих по отдельности: {sorted(missing)}")

    # source_manifest.json ----------------------------------------------------------------------
    def load_manifest():
        try:
            return SourceManifest.model_validate(read_json(run_dir / F.SOURCE_MANIFEST))
        except ValidationError as exc:
            report.add(F.SOURCE_MANIFEST, _fmt(exc))
            return None
    manifest = guarded(F.SOURCE_MANIFEST, load_manifest)
    if manifest is not None:
        name = F.SOURCE_MANIFEST
        ids = Counter(s.id for s in manifest.sources)
        for sid, n in ids.items():
            if n > 1:
                report.add(name, f"id {sid} повторяется {n} раз")
        if meta:
            s1 = f"{meta.chip_id}_S1Hand"
            if not any(s1 in s.id or s1 in (s.file or "") for s in manifest.sources):
                report.add(name, f"нет источника S1 демо-чипа ({s1})")
            for s in manifest.sources:
                if s.observation_date and s.observation_date > meta.event_date and not s.after_event_note:
                    report.add(name, f"{s.id}: наблюдение позже даты события без after_event_note")
    return report.errors


def default_run_dir() -> Path:
    """Newest set in FLOODVALUE_OUTPUT_DIR (by run_metadata.json mtime), else the fixture."""
    root = output_dir()
    runs = [d for d in root.iterdir() if (d / F.RUN_METADATA).is_file()] if root.is_dir() else []
    if not runs:
        return FIXTURE_RUN
    return max(runs, key=lambda d: (d / F.RUN_METADATA).stat().st_mtime)


def run(args) -> int:
    run_dir = Path(args.out) if getattr(args, "out", None) else default_run_dir()
    print(f"Комплект: {run_dir}")
    if not run_dir.is_dir():
        print("каталог не найден")
        return 1
    errors = validate_run(run_dir)
    passed = 0
    for name, errs in errors.items():
        if errs:
            print(f"FAIL {name}")
            for e in errs:
                print(f"     - {e}")
        else:
            passed += 1
            print(f"OK   {name}")
    print(f"{passed}/{len(errors)} файлов прошли проверку")
    return 0 if passed == len(errors) else 1
