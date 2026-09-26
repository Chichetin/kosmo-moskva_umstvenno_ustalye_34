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
from rasterio.transform import array_bounds
from rasterio.warp import transform as warp_transform
from rasterio.warp import transform_bounds
from shapely.geometry import Point, box, shape

from floodvalue.contracts import files as F
from floodvalue.contracts.geo import MIN_AREA_KM2, area_km2, product_half_up, round_half_up
from floodvalue.contracts.io import read_csv, read_json
from floodvalue.contracts.schemas import (
    AssetCsvRow,
    AssetLossRow,
    AssetProperties,
    CandidateProperties,
    ProcurementPlanRow,
    RunMetadata,
    SensitivityRow,
    SourceManifest,
    StrategyComparisonRow,
    StrategyPlans,
)
from floodvalue.paths import CONFIGS_DIR, REPO_ROOT, data_dir, output_dir

FIXTURE_RUN = REPO_ROOT / "tests" / "fixtures" / "good"
MONEY_TOL = 0.005  # half a kopeck: recomputed kopeck amounts must match exactly
EL_TOL = 0.01  # EL written to the kopeck
P_TOL = 1e-6  # p_flood vs float32 raster value
COORD_TOL = 1e-7  # degrees, ~1 cm
THRESHOLD_TOL = 1e-9  # main threshold in metrics.csv vs run_metadata.threshold (float noise)
BASELINE_MASK = "baseline_mask.tif"  # optional raster, contracts §2 (#115)


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
        if geom_type == "Polygon":
            open_rings = _open_rings(geom.get("coordinates"))
            if open_rings:
                report.add(name, "; ".join(f"объект {i}: кольцо {k} не замкнуто или короче 4 точек "
                                           f"(RFC 7946 §3.1.6)" for k in open_rings))
                continue
        try:
            props = model.model_validate(feat.get("properties") or {})
            out.append((props, shape(geom)))
        except (ValidationError, ValueError, TypeError) as exc:
            msg = _fmt(exc) if isinstance(exc, ValidationError) else str(exc)
            report.add(name, f"объект {i}: {msg}")
    return out if len(out) == len(doc.get("features") or []) else None


def _open_rings(rings) -> list[int]:
    """Indices of Polygon rings whose first and last positions differ (shapely closes them silently)."""
    return [k for k, ring in enumerate(rings or [])
            if not isinstance(ring, list) or len(ring) < 4 or ring[0] != ring[-1]]


def _chip_lonlat(raster):
    """Chip extent as a WGS 84 [lon, lat] box, from the probability raster grid."""
    p, _, prof = raster
    bounds = array_bounds(p.shape[0], p.shape[1], prof["transform"])  # (west, south, east, north)
    if prof["crs"] and prof["crs"].to_epsg() != 4326:
        bounds = transform_bounds(prof["crs"], "EPSG:4326", *bounds)
    return box(*bounds)


def _lonlat_ok(geom) -> bool:
    xs, ys = np.asarray(geom.exterior.coords if geom.geom_type == "Polygon" else geom.coords).T
    return bool(np.all(np.abs(xs) <= 180) and np.all(np.abs(ys) <= 90))


# --- rasters -----------------------------------------------------------------------------------

def _find_s1(chip_id: str) -> Path | None:
    """S1 of the chip in the raw layout of docs/data.md §3 (raw/HandLabeled/S1Hand/)."""
    path = data_dir() / "raw" / "HandLabeled" / "S1Hand" / f"{chip_id}_S1Hand.tif"
    return path if path.is_file() else None


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


def _check_baseline_mask(report: _Report, run_dir: Path) -> None:
    """Optional baseline_mask.tif (contracts §2): absent is fine; present must sit on the grid of
    flood_probability.tif (itself checked against S1), hold only 0/1/255 and have 255 exactly
    where p is -9999, i.e. where S1 has no data. Errors go to flood_mask.tif, its sibling."""
    path = run_dir / BASELINE_MASK
    if not path.is_file():
        return

    def err(message: str) -> None:
        report.add(F.FLOOD_MASK, f"{BASELINE_MASK} (доп.): {message}")

    try:
        with rasterio.open(run_dir / F.FLOOD_PROBABILITY) as src:
            p_prof, p = src.profile, src.read(1)
        with rasterio.open(path) as src:
            b_prof, b = src.profile, src.read()
    except Exception as exc:  # noqa: BLE001 - malformed file: report, keep checking
        err(f"не читается: {type(exc).__name__}: {exc}")
        return
    if b.shape[0] != 1:
        err(f"каналов {b.shape[0]}, нужен 1")
    if b_prof["dtype"] != F.MASK_DTYPE:
        err(f"тип {b_prof['dtype']}, нужен UInt8")
    if b_prof.get("nodata") != F.MASK_NODATA:
        err(f"nodata {b_prof.get('nodata')}, нужен 255")
    b = b[0]
    bad = ~np.isin(b, (0, 1, F.MASK_NODATA))
    if bad.any():
        err(f"значения кроме 0/1/255 в {int(bad.sum())} пикселях")
    same_grid = (b.shape == p.shape and b_prof.get("crs") == p_prof.get("crs")
                 and np.allclose(tuple(b_prof["transform"]), tuple(p_prof["transform"]),
                                 rtol=0, atol=1e-12))
    if not same_grid:
        err("сетка (CRS, transform, размер) не совпадает с flood_probability.tif / S1-чипом")
        return
    no_s1 = ~np.isfinite(p) | (p == F.PROBABILITY_NODATA)
    mismatch = no_s1 != (b == F.MASK_NODATA)
    if mismatch.any():
        err(f"255 не ровно там, где нет данных S1 (p = -9999), в {int(mismatch.sum())} пикселях")


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


def covered_expected_loss(zones: list, points: dict[str, Point], el: dict[str, float]) -> float:
    """EL of unique asset_id covered by any zone; `covers` counts the boundary (task.md)."""
    return sum(v for aid, v in el.items() if any(z.covers(points[aid]) for z in zones))


def decision_risk_from_el(el: float, p: float) -> float:
    """Expected cost of a wrong decision R = V·q·min(p, 1−p), rub. (scenario, D-049), from the
    written EL and p of asset_loss.csv (V·q = EL/p), exactly as procurement does: EL rounded to
    the kopeck would otherwise drift from V·q·min(p, 1−p) by up to 0,005 per asset (#146)."""
    if p <= 0.5:
        return el
    return el * (1.0 - p) / p


def _optional_number(extra: dict, field: str) -> tuple[bool, float | None]:
    """(present, value) of an optional CSV column; empty cell -> None."""
    if field not in extra:
        return False, None
    raw = extra[field]
    return True, None if raw in (None, "") else float(raw)


def _check_decision_risk(report: "_Report", name: str, row, used: list | None,
                         by_id: dict, risk_by_id: dict[str, float], meta) -> None:
    """Optional columns of strategy_comparison.csv (D-049, #142): addressed_decision_risk_rub =
    Σ R of ok assets covered by the strategy's zones usable by the deadline (same zones as
    covered_expected_loss_rub), decision_risk_share = that / Σ R of the portfolio, empty at 0."""
    extra = row.model_extra or {}
    try:
        has_r, addressed = _optional_number(extra, "addressed_decision_risk_rub")
        has_share, share = _optional_number(extra, "decision_risk_share")
    except ValueError:
        report.add(name, f"{row.strategy}: addressed_decision_risk_rub/decision_risk_share — не число")
        return
    s = row.strategy
    if has_r and addressed is None:
        report.add(name, f"{s}: addressed_decision_risk_rub пуст")
    if used is None or not (has_r or has_share):
        return
    a_declared = s == "A" and meta and (meta.strategies.model_extra or {}).get("coverage_A")
    if a_declared:  # A's coverage is declared separately: check only share ↔ addressed below
        expected = addressed
    else:
        expected = covered_expected_loss(used, {a: by_id[a][1] for a in risk_by_id}, risk_by_id)
    if has_r and addressed is not None and not a_declared and abs(expected - addressed) > EL_TOL:
        report.add(name, f"{s}: addressed_decision_risk_rub={addressed} ≠ пересчёту {expected:.2f} "
                         "(Σ R = EL·min(p, 1−p)/p ok-объектов, покрытых зонами к сроку)")
    if not has_share or expected is None:
        return
    addressed = expected if addressed is None else addressed
    total = sum(risk_by_id.values())
    if total > 0 and share is None:
        report.add(name, f"{s}: decision_risk_share пуст при ненулевой Σ R портфеля")
    elif total <= 0 and share is not None:
        report.add(name, f"{s}: decision_risk_share={share} при Σ R портфеля = 0 (должно быть пусто)")
    elif share is not None and abs(share - addressed / total) > 1e-4:
        report.add(name, f"{s}: decision_risk_share={share} ≠ addressed_decision_risk_rub / Σ R "
                         f"= {addressed / total:.6f}")


METRICS_FIELDS = ("method", "split", "event", "iou", "f1", "precision", "recall", "tp", "fp",
                  "fn", "tn", "n_pixels", "threshold", "valid_mask_sha256")


def _check_metrics(report: _Report, run_dir: Path, meta: RunMetadata) -> None:
    """Criterion 12 'metrics of both methods': files listed in run_metadata.metrics exist;
    metrics.csv has baseline and main on the same valid-pixel mask (docs/contracts.md §7);
    every main row is computed at the set's mask threshold (§7a; baseline rows are in dB)."""
    name = F.RUN_METADATA
    for rel in meta.metrics.files:
        path = next((c for c in (run_dir / rel, REPO_ROOT / rel) if c.is_file()), None)
        if path is None:
            report.add(name, f"metrics.files: {rel} не найден")
            continue
        if path.name != "metrics.csv":
            continue
        header, rows, errors = read_csv(path)
        for e in errors:
            report.add(name, f"{rel}: {e}")
        missing = [f for f in METRICS_FIELDS if f not in header]
        if missing:
            report.add(name, f"{rel}: нет столбцов {missing}")
            continue
        if {r["method"] for r in rows} < {"baseline", "main"}:
            report.add(name, f"{rel}: нужны строки обоих методов baseline и main")
        by_key: dict[tuple[str, str], set[str]] = defaultdict(set)
        for r in rows:
            by_key[(r["split"], r["event"])].add(r["valid_mask_sha256"])
        for key, hashes in by_key.items():
            if len(hashes) > 1:
                report.add(name, f"{rel}: методы в {key} посчитаны на разных масках валидных пикселей")
        for r in rows:
            if r["method"] != "main":
                continue
            where = f"{rel}: main {r['split']}/{r['event']}"
            try:
                t = float(r["threshold"])
            except (TypeError, ValueError):  # None: the row is shorter than the header
                report.add(name, f"{where}: threshold пуст или не число — нельзя сверить с "
                                 f"threshold комплекта {meta.threshold}")
                continue
            if not abs(t - meta.threshold) <= THRESHOLD_TOL:
                report.add(name, f"{where}: threshold={r['threshold']} ≠ threshold комплекта "
                                 f"{meta.threshold} (run_metadata.json) — метрики не описывают маску")


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
        except Exception as exc:  # noqa: BLE001 - malformed file: report, keep checking
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
    base_rate = pricing["base_rate"]
    if meta is not None:
        for field, value in (("legal_edition", pricing["legal_edition"]),
                             ("base_rate_rub_km2", float(base_rate["rub_per_km2"])),
                             ("base_rate_status", base_rate["status"])):
            if getattr(meta.pricing, field) != value:
                report.add(F.RUN_METADATA, f"pricing.{field}={getattr(meta.pricing, field)!r} ≠ "
                                           f"configs/pricing.yaml {value!r}")

    if meta is not None:
        _check_metrics(report, run_dir, meta)

    # rasters -----------------------------------------------------------------------------------
    raster = None
    if F.FLOOD_PROBABILITY in present and F.FLOOD_MASK in present:
        try:
            raster = _check_rasters(report, run_dir, meta)
        except Exception as exc:  # noqa: BLE001
            report.add(F.FLOOD_PROBABILITY, f"растры не читаются: {type(exc).__name__}: {exc}")
    if F.FLOOD_PROBABILITY in present and F.FLOOD_MASK in present:
        _check_baseline_mask(report, run_dir)

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
    risk_by_id: dict[str, float] = {}  # R = V·q·min(p, 1−p) of ok assets (D-049)
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
                risk_by_id[r.asset_id] = decision_risk_from_el(r.expected_loss_rub, r.p_flood)
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
        chip = _chip_lonlat(raster) if raster is not None else None
        for c, geom in cands:
            cid = c.candidate_id
            catalog.setdefault(cid, c)
            zones.setdefault(cid, geom)
            for field, value in (("base_rate_rub_km2", float(base_rate["rub_per_km2"])),
                                 ("base_rate_status", base_rate["status"])):
                if getattr(c, field) != value:
                    report.add(name, f"{cid}: {field}={getattr(c, field)!r} ≠ configs/pricing.yaml "
                                     f"{value!r}")
            if not _lonlat_ok(geom):
                report.add(name, f"{cid}: координаты вне диапазона [lon, lat]")
                continue
            if chip is not None and not geom.intersects(chip):
                report.add(name, f"{cid}: зона не пересекает чип (порядок [долгота, широта]?)")
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
            if s == "A" and r.covered_expected_loss_rub != 0 and meta and \
                    not (meta.strategies.model_extra or {}).get("coverage_A"):
                report.add(name, "A: covered_expected_loss_rub ≠ 0 без strategies.coverage_A "
                                 "в run_metadata (у A нет платных зон)")
            used = None  # zones of the strategy that observe the event by the deadline
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
            elif s == "A" and losses is not None:
                used = []  # A buys no zones
            _check_decision_risk(report, name, r, used, by_id, risk_by_id, meta)
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
    marker = "  [ЭТАЛОН tests/fixtures/good, не результат запуска]" \
        if run_dir.resolve() == FIXTURE_RUN.resolve() else ""
    print(f"Комплект: {run_dir}{marker}")
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
