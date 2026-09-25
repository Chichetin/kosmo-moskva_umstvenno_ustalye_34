"""Каталог кандидатных зон заказа и их охват объектов (чистые функции, D-023).

Зоны v0 строятся из точек портфеля и границ чипа, без p, маски и EL (каталог не зависит
от бюджета и метода — один на B и C):

* ``Z-<asset_id>`` — квадрат ``BUFFER_SIDE_KM`` вокруг каждого объекта в EPSG:6933, новая
  гарантированная SAR-съёмка 1 м (наблюдение текущей воды, успевает к сроку);
* ``W-CHIP`` — сплошное покрытие чипа той же съёмкой (широкая зона, правило B);
* ``N-OPT-LATE`` — новая оптическая съёмка по охвату объектов (обрезан по чипу); ближайший безоблачный
  пролёт — после срока решения, поэтому в каталоге есть, но охвата не даёт;
* ``K-ARCH`` / ``K-OPER`` — гипотетические имеющиеся оптические снимки до паводка (``data_role=context``):
  состояние объектов до события, текущую воду не наблюдают.

Все сроки и характеристики съёмки сценарные (``TIMING_STATUS``), не подтверждены поставщиком.
EPSG:6933 — цилиндрическая равновеликая проекция: прямоугольник в ней — прямоугольник в
lon/lat, поэтому площадь, пересчитанная по вершинам WGS 84, совпадает с построенной.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime, timedelta
from typing import Any

from pyproj import Transformer
from shapely.geometry import Point, Polygon, box, mapping, shape

from floodvalue.procurement.geometry import AREA_CRS, order_area_km2

BUFFER_SIDE_KM = 1.2           # 1,44 км² — запас над минимумом 1 км² до округления
EVENT_SENSOR = {"sensor_type": "sar", "resolution_m": 1, "processing_level": "L2"}
OPTICAL = {"sensor_type": "optical", "resolution_m": 1, "processing_level": "L2"}
# Сценарные сдвиги от decision_at (D-023): следующий пролёт SAR через 3 ч, продукт через 9 ч;
# ближайший безоблачный оптический пролёт — через 20 ч, продукт через 30 ч (после срока).
NEW_SAR_OBS_H, NEW_SAR_AVAIL_H = 3, 9
NEW_OPT_OBS_H, NEW_OPT_AVAIL_H = 20, 30
ARCHIVE_AGE_D, OPERATIONAL_AGE_D = 180, 30
TIMING_STATUS = "scenario"
TIMING_BASIS = ("сценарные сроки пролёта и выпуска продукта (D-023), поставщиком не "
                "подтверждены; облачность оптики — допущение, не наблюдение")


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def parse_dt(value: str | datetime) -> datetime:
    dt = value if isinstance(value, datetime) else datetime.fromisoformat(value)
    if dt.tzinfo is None:
        raise ValueError(f"дата без часового пояса: {value!r}")
    return dt


def square_around(lon: float, lat: float, side_km: float = BUFFER_SIDE_KM) -> Polygon:
    """Квадрат со стороной ``side_km`` вокруг точки, построенный в EPSG:6933, в WGS 84."""
    fwd = Transformer.from_crs("EPSG:4326", AREA_CRS, always_xy=True)
    inv = Transformer.from_crs(AREA_CRS, "EPSG:4326", always_xy=True)
    x, y = fwd.transform(lon, lat)
    h = side_km * 1000 / 2
    (x0, x1), (y0, y1) = (x - h, x + h), (y - h, y + h)
    lon0, lat0 = inv.transform(x0, y0)
    lon1, lat1 = inv.transform(x1, y1)
    return box(lon0, lat0, lon1, lat1)


def _feature(cid: str, poly: Polygon, props: Mapping[str, Any], pricing_cfg: Mapping[str, Any]
             ) -> dict[str, Any]:
    area = order_area_km2(poly, float(pricing_cfg["min_area_km2"]),
                          str(pricing_cfg["rounding"]["area_km2"]), pricing_cfg.get("area_crs", AREA_CRS))
    br = pricing_cfg["base_rate"]
    return {"type": "Feature", "geometry": mapping(poly), "properties": {
        "candidate_id": cid, **props, "usage_type": "internal",
        "area_km2": float(area), "base_rate_rub_km2": float(br["rub_per_km2"]),
        "base_rate_status": br["status"], "base_rate_source": br["source"],
        "timing_status": TIMING_STATUS}}


def build_catalog(assets: Iterable[Mapping[str, Any]], chip_bounds: tuple[float, float, float, float],
                  decision_at: datetime, pricing_cfg: Mapping[str, Any]) -> dict[str, Any]:
    """FeatureCollection каталога. ``assets`` — asset_id, longitude, latitude (все 10 объектов)."""
    t0 = parse_dt(decision_at)
    h = timedelta(hours=1)
    new_sar = {**EVENT_SENSOR, "acquisition_type": "new", "guaranteed_purchase": True,
               "data_role": "event_observation",
               "observation_at": _iso(t0 + NEW_SAR_OBS_H * h),
               "available_at": _iso(t0 + NEW_SAR_AVAIL_H * h)}
    feats = []
    assets = sorted(assets, key=lambda a: a["asset_id"])
    for a in assets:
        poly = square_around(float(a["longitude"]), float(a["latitude"]))
        feats.append(_feature(f"Z-{a['asset_id']}", poly,
                              {**new_sar, "zone_kind": "asset_buffer"}, pricing_cfg))
    chip = box(*chip_bounds)
    feats.append(_feature("W-CHIP", chip, {**new_sar, "zone_kind": "wide"}, pricing_cfg))
    lons = [float(a["longitude"]) for a in assets]
    lats = [float(a["latitude"]) for a in assets]
    pad = 0.005
    hull = box(min(lons) - pad, min(lats) - pad, max(lons) + pad, max(lats) + pad)
    clipped = hull.intersection(chip)
    hull = clipped if clipped.geom_type == "Polygon" and not clipped.is_empty else hull
    feats.append(_feature("N-OPT-LATE", hull, {
        **OPTICAL, "acquisition_type": "new", "guaranteed_purchase": True,
        "data_role": "event_observation", "zone_kind": "late_optical",
        "observation_at": _iso(t0 + NEW_OPT_OBS_H * h),
        "available_at": _iso(t0 + NEW_OPT_AVAIL_H * h)}, pricing_cfg))
    for cid, age, acq in (("K-ARCH", ARCHIVE_AGE_D, "archive"),
                          ("K-OPER", OPERATIONAL_AGE_D, "operational")):
        obs = t0 - timedelta(days=age)
        feats.append(_feature(cid, chip, {
            **OPTICAL, "acquisition_type": acq, "guaranteed_purchase": False,
            "data_role": "context", "zone_kind": "context",
            "source_note": ("гипотетический имеющийся снимок (сценарий), не из реального "
                            "каталога поставщика; дата = decision_at − " f"{age} сут"),
            "observation_at": _iso(obs), "available_at": _iso(obs + timedelta(days=2))},
            pricing_cfg))
    return {"type": "FeatureCollection", "features": feats}


def exclusion_reason(props: Mapping[str, Any], decision_deadline: datetime) -> str | None:
    """Почему зона не даёт оперативного охвата (None — даёт)."""
    if props["data_role"] != "event_observation":
        return ("контекстный снимок до паводка: состояние объектов до события, "
                "текущую воду не наблюдает (не считается охватом)")
    avail = props.get("available_at")
    if not avail:
        return "срок доступности не известен — к decision_deadline не учитывается"
    if parse_dt(avail) > parse_dt(decision_deadline):
        return f"доступна {avail} — позже decision_deadline {_iso(parse_dt(decision_deadline))}"
    return None


def coverage(catalog: Mapping[str, Any], assets: Iterable[Mapping[str, Any]],
             decision_deadline: datetime) -> dict[str, list[str]]:
    """candidate_id → отсортированные asset_id, которые зона покрывает (``covers``, граница
    включительно). Только зоны без :func:`exclusion_reason`; у прочих — пустой список.
    ``assets`` — объекты, охват которых учитывается (вызывающий передаёт только status=ok)."""
    pts = [(a["asset_id"], Point(float(a["longitude"]), float(a["latitude"]))) for a in assets]
    out: dict[str, list[str]] = {}
    for f in catalog["features"]:
        p = f["properties"]
        if exclusion_reason(p, decision_deadline) is not None:
            out[p["candidate_id"]] = []
            continue
        poly = shape(f["geometry"])
        out[p["candidate_id"]] = sorted({aid for aid, pt in pts if poly.covers(pt)})
    return out


def covered_assets(ids: Iterable[str], cov: Mapping[str, Iterable[str]]) -> set[str]:
    """Уникальные asset_id, покрытые набором зон (повторное покрытие не умножает)."""
    return {aid for cid in ids for aid in cov.get(cid, ())}
