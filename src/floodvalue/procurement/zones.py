"""Каталог кандидатных зон заказа и их охват объектов (чистые функции, D-023, D-032).

Каталог не зависит от бюджета: один на B и C, строится заново при каждом ``procure``/``plan``.
Геометрии (``geometry_id``, ``zone_kind``):

* ``Z-<asset_id>`` (``asset_buffer``) — квадрат ``BUFFER_SIDE_KM`` вокруг каждого объекта в
  EPSG:6933;
* ``E-<k>`` / ``E-ALL`` (``el_cluster``) — кластеры объектов высокого EL: объекты ``status=ok``
  по убыванию EL, пока их сумма не достигнет ``EL_CLUSTER_SHARE`` суммарного EL; одиночная
  связь при расстоянии ≤ ``EL_CLUSTER_LINK_KM``; выпуклая оболочка точек + буфер
  ``EL_CLUSTER_BUFFER_M`` в EPSG:6933, обрезка по чипу; ``E-ALL`` — одна оболочка всех
  объектов высокого EL (если кластеров больше одного);
* ``U-<k>`` (``uncertainty_cluster``) — кластеры высокой неопределённости: растр
  неопределённости усредняется блоками ``UNC_BLOCK_PX`` пикселей, блоки выше квантиля
  ``UNC_BLOCK_QUANTILE`` связываются по 8 соседям, ``UNC_TOP_CLUSTERS`` связных групп с
  наибольшей суммой — выпуклая оболочка блоков + буфер ``UNC_BUFFER_M``, обрезка по чипу;
* ``G2-<r><c>`` / ``G3-<r><c>`` (``grid``) — регулярная сетка 2×2 и 3×3 по границам чипа;
* ``W-CHIP`` (``wide``) — весь чип (правило B).

Буфер растёт в ``GROW_FACTOR`` раз, пока площадь после обрезки меньше ``1 км² × AREA_MARGIN``
(порог проверяется до округления). Пересечение выпуклого контура с прямоугольником чипа
выпукло, поэтому каждая зона — один Polygon без holes.

На каждую геометрию — одно или несколько сценарных предложений съёмки (:data:`OFFERS`,
``candidate_id = geometry_id + suffix``): новая SAR 1 м (основное), новая SAR 3 м (позже),
новая оптика (облачность не подтверждена / после срока), оперативный снимок события
(срок поставки неизвестен), архив и контекст до паводка. Сроки, облачность и наличие снимков
сценарные (``TIMING_STATUS``), поставщиком не подтверждены. Непригодные позиции остаются в
каталоге с причиной (``exclusion_reason``) и охвата не дают.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime, timedelta
from typing import Any

import numpy as np
from pyproj import Transformer
from shapely.geometry import MultiPoint, Point, Polygon, box, mapping, shape
from shapely.ops import transform as shp_transform

from floodvalue.procurement.geometry import AREA_CRS, order_area_km2, polygon_area_km2
from floodvalue.procurement.pricing import price_basket

BUFFER_SIDE_KM = 1.2           # 1,44 км² — запас над минимумом 1 км² до округления
EL_CLUSTER_SHARE = 0.9         # объекты высокого EL: первые по EL, дающие ≥ 90 % суммарного EL
EL_CLUSTER_LINK_KM = 2.0       # одиночная связь кластера объектов
EL_CLUSTER_BUFFER_M = 600.0    # тот же запас ±600 м, что у квадрата Z (круг вместо квадрата)
UNC_BLOCK_PX = 16              # блок 16×16 пикселей (~160 м) — сглаживает пиксельный шум
UNC_BLOCK_QUANTILE = 0.9       # верхние 10 % блоков по средней неопределённости
UNC_MIN_VALID_SHARE = 0.5      # блок учитывается, если валидна хотя бы половина пикселей
UNC_TOP_CLUSTERS = 3
UNC_BUFFER_M = 150.0
GRID_SIZES = (2, 3)
AREA_MARGIN = 1.001            # строим с запасом 0,1 % над 1 км², проверка — до округления
GROW_FACTOR = 1.15
QUAD_SEGS = 16

TIMING_STATUS = "scenario"
TIMING_BASIS = ("сценарные сроки пролёта и выпуска продукта, наличие имеющихся снимков и "
                "облачность оптики (D-023, D-032) поставщиком не подтверждены")
CLOUD_NOTE = ("оптика: безоблачность района паводка к сроку не подтверждена (сценарий); без "
              "подтверждения пригодности снимок не считается наблюдением текущей воды")

# Сценарные предложения съёмки (D-032). Сдвиги — от decision_at: *_h — часы, *_d — сутки.
OFFERS: dict[str, dict[str, Any]] = {
    "S1N": {"suffix": "", "sensor_type": "sar", "resolution_m": 1, "acquisition_type": "new",
            "data_role": "event_observation", "obs_h": 3, "avail_h": 9,
            "offer_note": "новая гарантированная SAR 1 м: следующий пролёт через 3 ч, продукт через 9 ч"},
    "S3N": {"suffix": "-S3", "sensor_type": "sar", "resolution_m": 3, "acquisition_type": "new",
            "data_role": "event_observation", "obs_h": 6, "avail_h": 11,
            "offer_note": ("новая гарантированная SAR 3 м (широкополосный режим): пролёт через "
                           "6 ч, продукт через 11 ч; грубее и позже S1N, скидка Р по своей группе")},
    "O05N": {"suffix": "-O05", "sensor_type": "optical", "resolution_m": 0.5,
             "acquisition_type": "new", "data_role": "event_observation", "obs_h": 4,
             "avail_h": 10, "cloud_status": "not_confirmed",
             "offer_note": "новая оптика 0,5 м: пролёт через 4 ч, продукт через 10 ч; облачность не подтверждена"},
    "O1L": {"suffix": "-OL", "sensor_type": "optical", "resolution_m": 1,
            "acquisition_type": "new", "data_role": "event_observation", "obs_h": 20,
            "avail_h": 30, "cloud_status": "not_confirmed",
            "offer_note": "новая оптика 1 м: ближайший вероятно безоблачный пролёт через 20 ч, продукт через 30 ч"},
    "S3OP": {"suffix": "-OP", "sensor_type": "sar", "resolution_m": 3,
             "acquisition_type": "operational", "data_role": "event_observation", "obs_h": -6,
             "avail_h": None,
             "offer_note": ("оперативный (≤ 90 сут) снимок SAR 3 м текущего события за 6 ч до "
                            "решения — гипотетический, наличие в каталоге поставщика и срок "
                            "поставки не подтверждены")},
    "O1AR": {"suffix": "-AR", "sensor_type": "optical", "resolution_m": 1,
             "acquisition_type": "archive", "data_role": "context", "obs_d": -180,
             "avail_d": -178,
             "offer_note": "архивная оптика 1 м (180 сут до решения): состояние до паводка"},
    "O1OC": {"suffix": "-OC", "sensor_type": "optical", "resolution_m": 1,
             "acquisition_type": "operational", "data_role": "context", "obs_d": -30,
             "avail_d": -28,
             "offer_note": "оперативная (30 сут) оптика 1 м до паводка: контекст, не текущая вода"},
}
OFFERS_BY_KIND: dict[str, tuple[str, ...]] = {
    "asset_buffer": ("S1N", "S3N", "O05N"),
    "el_cluster": ("S1N", "S3N", "O05N"),
    "uncertainty_cluster": ("S1N", "S3N"),
    "grid2": ("S1N", "S3N", "O1AR"),
    "grid3": ("S1N",),
    "wide": ("S1N", "S3N", "O05N", "O1L", "S3OP", "O1AR", "O1OC"),
}
PRIMARY_OFFER = "S1N"


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def parse_dt(value: str | datetime) -> datetime:
    dt = value if isinstance(value, datetime) else datetime.fromisoformat(value)
    if dt.tzinfo is None:
        raise ValueError(f"дата без часового пояса: {value!r}")
    return dt


def _fwd() -> Transformer:
    return Transformer.from_crs("EPSG:4326", AREA_CRS, always_xy=True)


def _inv() -> Transformer:
    return Transformer.from_crs(AREA_CRS, "EPSG:4326", always_xy=True)


def square_around(lon: float, lat: float, side_km: float = BUFFER_SIDE_KM) -> Polygon:
    """Квадрат со стороной ``side_km`` вокруг точки, построенный в EPSG:6933, в WGS 84."""
    x, y = _fwd().transform(lon, lat)
    h = side_km * 1000 / 2
    lon0, lat0 = _inv().transform(x - h, y - h)
    lon1, lat1 = _inv().transform(x + h, y + h)
    return box(lon0, lat0, lon1, lat1)


def buffered_hull(points: Iterable[tuple[float, float]], radius_m: float,
                  chip: Polygon, min_area_km2: float = 1.0) -> tuple[Polygon, float]:
    """Выпуклая оболочка точек [lon, lat] + буфер в EPSG:6933, обрезка по чипу.

    Буфер растёт в ``GROW_FACTOR`` раз, пока площадь < ``min_area_km2 × AREA_MARGIN``.
    Возвращает (Polygon WGS 84 без holes, итоговый радиус буфера в м).
    """
    fwd, inv = _fwd(), _inv()
    hull_m = shp_transform(fwd.transform, MultiPoint(list(points))).convex_hull
    r = float(radius_m)
    for _ in range(60):
        buf = shp_transform(inv.transform, hull_m.buffer(r, quad_segs=QUAD_SEGS))
        poly = buf.intersection(chip).convex_hull
        if poly.geom_type == "Polygon" and polygon_area_km2(poly) >= min_area_km2 * AREA_MARGIN:
            return poly, r
        r *= GROW_FACTOR
    raise ValueError("не удалось построить зону ≥ 1 км² внутри чипа")


def el_clusters(assets: Iterable[Mapping[str, Any]], loss: Iterable[Mapping[str, Any]],
                share: float = EL_CLUSTER_SHARE, link_km: float = EL_CLUSTER_LINK_KM
                ) -> tuple[list[str], list[list[str]]]:
    """Объекты высокого EL и их кластеры (одиночная связь по расстоянию в EPSG:6933).

    Высокий EL — объекты ``status=ok`` по (−EL, asset_id), пока накопленный EL < ``share`` ×
    суммарного (минимум один объект с EL > 0). Возвращает (high_ids, кластеры по asset_id).
    """
    pos = {a["asset_id"]: (float(a["longitude"]), float(a["latitude"])) for a in assets}
    ok = sorted(((r["asset_id"], float(r["expected_loss_rub"])) for r in loss
                 if r["status"] == "ok" and r["asset_id"] in pos),
                key=lambda t: (-t[1], t[0]))
    total = sum(el for _, el in ok)
    if total <= 0:
        return [], []
    high, acc = [], 0.0
    for aid, el in ok:
        if acc >= share * total or el <= 0:
            break
        high.append(aid)
        acc += el
    fwd = _fwd()
    xy = {a: fwd.transform(*pos[a]) for a in high}
    parent = {a: a for a in high}

    def find(a):
        while parent[a] != a:
            a = parent[a]
        return a

    for i, a in enumerate(high):
        for b in high[i + 1:]:
            d = ((xy[a][0] - xy[b][0]) ** 2 + (xy[a][1] - xy[b][1]) ** 2) ** 0.5
            if d <= link_km * 1000:
                parent[find(b)] = find(a)
    groups: dict[str, list[str]] = {}
    for a in high:
        groups.setdefault(find(a), []).append(a)
    clusters = sorted((sorted(g) for g in groups.values()),
                      key=lambda g: (-sum(dict(ok)[a] for a in g), g))
    return high, clusters


def uncertainty_clusters(unc: np.ndarray, transform: Any, block: int = UNC_BLOCK_PX,
                         quantile: float = UNC_BLOCK_QUANTILE, top: int = UNC_TOP_CLUSTERS
                         ) -> list[dict[str, Any]]:
    """Связные группы блоков высокой неопределённости.

    ``unc`` — (H, W) float, NaN — нет данных; ``transform`` — affine растра (EPSG:4326).
    Возвращает до ``top`` групп по убыванию суммы средних блоков: ``corners`` (углы блоков,
    [lon, lat]), ``n_blocks``, ``score`` (сумма средних), ``threshold`` (порог блока).
    """
    from scipy import ndimage

    a = np.asarray(unc, dtype="float64")
    h, w = a.shape
    nb_r, nb_c = h // block, w // block
    if nb_r == 0 or nb_c == 0:
        return []
    a = a[:nb_r * block, :nb_c * block].reshape(nb_r, block, nb_c, block)
    valid = np.isfinite(a)
    cnt = valid.sum(axis=(1, 3))
    sums = np.where(valid, a, 0.0).sum(axis=(1, 3))
    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.where(cnt >= UNC_MIN_VALID_SHARE * block * block, sums / np.maximum(cnt, 1),
                         np.nan)
    finite = means[np.isfinite(means)]
    if finite.size == 0:
        return []
    thr = float(np.quantile(finite, quantile))
    hot = np.isfinite(means) & (means >= thr) & (means > 0)
    labels, n = ndimage.label(hot, structure=np.ones((3, 3), dtype=int))
    out = []
    for k in range(1, n + 1):
        rr, cc = np.nonzero(labels == k)
        corners = []
        for r, c in zip(rr, cc, strict=True):
            for dr, dc in ((0, 0), (0, 1), (1, 0), (1, 1)):
                x, y = transform @ ((c + dc) * block, (r + dr) * block)
                corners.append((float(x), float(y)))
        out.append({"corners": corners, "n_blocks": int(rr.size),
                    "score": float(means[rr, cc].sum()), "threshold": thr})
    out.sort(key=lambda g: (-g["score"], g["corners"][0]))
    return out[:top]


def grid_cells(bounds: tuple[float, float, float, float], n: int) -> list[tuple[str, Polygon]]:
    """Сетка n×n по границам чипа в lon/lat; id ``G<n>-<строка><столбец>`` с северо-запада."""
    x0, y0, x1, y1 = bounds
    dx, dy = (x1 - x0) / n, (y1 - y0) / n
    cells = []
    for r in range(n):
        for c in range(n):
            top = y1 - r * dy
            cells.append((f"G{n}-{r + 1}{c + 1}",
                          box(x0 + c * dx, top - dy, x0 + (c + 1) * dx, top)))
    return cells


def _offer_props(offer_id: str, t0: datetime) -> dict[str, Any]:
    o = OFFERS[offer_id]
    if "obs_d" in o:
        obs = t0 + timedelta(days=o["obs_d"])
        avail = t0 + timedelta(days=o["avail_d"])
    else:
        obs = t0 + timedelta(hours=o["obs_h"])
        avail = None if o["avail_h"] is None else t0 + timedelta(hours=o["avail_h"])
    props = {"offer_id": offer_id, "sensor_type": o["sensor_type"],
             "resolution_m": o["resolution_m"], "processing_level": "L2",
             "acquisition_type": o["acquisition_type"],
             "guaranteed_purchase": o["acquisition_type"] == "new",
             "data_role": o["data_role"], "observation_at": _iso(obs),
             "available_at": None if avail is None else _iso(avail),
             "offer_note": o["offer_note"]}
    if "cloud_status" in o:
        props["cloud_status"] = o["cloud_status"]
        props["cloud_note"] = CLOUD_NOTE
    return props


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
    if props.get("cloud_status") == "not_confirmed":
        return CLOUD_NOTE
    return None


def _single_price(props: Mapping[str, Any], pricing_cfg: Mapping[str, Any],
                  as_of: Any) -> dict[str, Any]:
    row = price_basket([{**props}], pricing_cfg, as_of=as_of)[0]
    return {"single_unit_price_rub_km2": float(row["unit_price_rub_km2"]),
            "single_discount_coef": float(row["discount_coef"]),
            "single_order_cost_rub": float(row["cost_rub"])}


def build_catalog(assets: Iterable[Mapping[str, Any]],
                  chip_bounds: tuple[float, float, float, float],
                  decision_at: datetime, pricing_cfg: Mapping[str, Any], *,
                  loss: Iterable[Mapping[str, Any]] | None = None,
                  uncertainty: np.ndarray | None = None, transform: Any = None,
                  decision_deadline: datetime | None = None) -> dict[str, Any]:
    """FeatureCollection каталога.

    ``assets`` — asset_id, longitude, latitude (все 10 объектов); ``loss`` — строки
    ``asset_loss.csv`` (для кластеров EL; нет — без ``E-*``); ``uncertainty`` + ``transform`` —
    растр неопределённости (нет — без ``U-*``); ``decision_deadline`` — записать причину
    исключения в свойство ``exclusion_reason``. Метаданные построения — ``catalog["build"]``
    (не часть GeoJSON-объектов; ``run`` переносит в ``procurement_metadata.json``).
    """
    t0 = parse_dt(decision_at)
    as_of = t0.date()
    min_area = float(pricing_cfg["min_area_km2"])
    step = str(pricing_cfg["rounding"]["area_km2"])
    crs = pricing_cfg.get("area_crs", AREA_CRS)
    br = pricing_cfg["base_rate"]
    assets = sorted(assets, key=lambda a: a["asset_id"])
    pts = [(a["asset_id"], Point(float(a["longitude"]), float(a["latitude"]))) for a in assets]
    chip = box(*chip_bounds)
    geoms: list[tuple[str, str, Polygon, str]] = []  # (geometry_id, kind, polygon, construction)
    build: dict[str, Any] = {}

    for a in assets:
        geoms.append((f"Z-{a['asset_id']}", "asset_buffer",
                      square_around(float(a["longitude"]), float(a["latitude"])),
                      f"квадрат {BUFFER_SIDE_KM} × {BUFFER_SIDE_KM} км в EPSG:6933 вокруг {a['asset_id']}"))

    if loss is not None:
        loss = list(loss)
        high, clusters = el_clusters(assets, loss)
        pos = {a["asset_id"]: (float(a["longitude"]), float(a["latitude"])) for a in assets}
        groups = [(f"E-{k}", g) for k, g in enumerate(clusters, 1)]
        if len(clusters) > 1:
            groups.append(("E-ALL", sorted(high)))
        for gid, g in groups:
            poly, r = buffered_hull([pos[a] for a in g], EL_CLUSTER_BUFFER_M, chip, min_area)
            geoms.append((gid, "el_cluster", poly,
                          (f"выпуклая оболочка объектов высокого EL {', '.join(g)} + буфер "
                           f"{r:.0f} м в EPSG:6933, обрезка по чипу")))
        build["el_clusters"] = {"high_el_asset_ids": high, "share": EL_CLUSTER_SHARE,
                                "link_km": EL_CLUSTER_LINK_KM,
                                "buffer_m": EL_CLUSTER_BUFFER_M,
                                "clusters": {gid: g for gid, g in groups}}
    else:
        build["el_clusters"] = "не строились: нет asset_loss.csv"

    if uncertainty is not None and transform is not None:
        ucl = uncertainty_clusters(uncertainty, transform)
        info = {}
        for k, g in enumerate(ucl, 1):
            poly, r = buffered_hull(g["corners"], UNC_BUFFER_M, chip, min_area)
            gid = f"U-{k}"
            geoms.append((gid, "uncertainty_cluster", poly,
                          (f"выпуклая оболочка {g['n_blocks']} блоков {UNC_BLOCK_PX}×{UNC_BLOCK_PX} "
                           f"пикс. с неопределённостью ≥ квантиля {UNC_BLOCK_QUANTILE} "
                           f"({g['threshold']:.4f}) + буфер {r:.0f} м, обрезка по чипу")))
            info[gid] = {"n_blocks": g["n_blocks"], "score": g["score"],
                         "block_threshold": g["threshold"], "buffer_m": r}
        build["uncertainty_clusters"] = {"block_px": UNC_BLOCK_PX,
                                         "quantile": UNC_BLOCK_QUANTILE,
                                         "top": UNC_TOP_CLUSTERS, "clusters": info}
    else:
        build["uncertainty_clusters"] = "не строились: нет растра неопределённости"

    skipped = {}
    for n in GRID_SIZES:
        cells = grid_cells(chip_bounds, n)
        if min(polygon_area_km2(c) for _, c in cells) < min_area * AREA_MARGIN:
            skipped[f"G{n}"] = "ячейка < 1 км² — сетка не строится"
            continue
        for gid, cell in cells:
            geoms.append((gid, f"grid{n}", cell, f"ячейка регулярной сетки {n}×{n} по границам чипа"))
    build["grid"] = {"sizes": list(GRID_SIZES), "skipped": skipped}
    geoms.append(("W-CHIP", "wide", chip, "границы чипа целиком (правило B)"))

    feats = []
    for gid, kind, poly, how in geoms:
        area = order_area_km2(poly, min_area, step, crs)
        covers = sorted(aid for aid, pt in pts if poly.covers(pt))
        for offer_id in OFFERS_BY_KIND[kind]:
            props = {"candidate_id": gid + OFFERS[offer_id]["suffix"], "geometry_id": gid,
                     "zone_kind": "grid" if kind.startswith("grid") else kind,
                     "construction": how, **_offer_props(offer_id, t0),
                     "usage_type": "internal", "area_km2": float(area),
                     "base_rate_rub_km2": float(br["rub_per_km2"]),
                     "base_rate_status": br["status"], "base_rate_source": br["source"],
                     "timing_status": TIMING_STATUS, "covers_asset_ids": covers}
            props.update(_single_price(props, pricing_cfg, as_of))
            if decision_deadline is not None:
                props["exclusion_reason"] = exclusion_reason(props, decision_deadline)
            feats.append({"type": "Feature", "geometry": mapping(poly), "properties": props})
    return {"type": "FeatureCollection", "features": feats, "build": build}


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
