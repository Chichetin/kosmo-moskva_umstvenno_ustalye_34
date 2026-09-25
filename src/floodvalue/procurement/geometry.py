"""Контур заказа: валидность Polygon и оплачиваемая площадь в EPSG:6933.

task.md, «Площадь и геометрия заказа»: один валидный Polygon без вырезов, координаты
GeoJSON [долгота, широта] (always_xy=True), площадь после перевода в равновеликую
EPSG:6933 (область применения ±86°), порог 1 км² — до округления, area_km2 = ROUND_HALF_UP(0,001).
"""
from __future__ import annotations

from collections.abc import Mapping
from decimal import ROUND_HALF_UP, Decimal
from functools import lru_cache
from typing import Any

from pyproj import Transformer
from shapely.geometry import Polygon, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform
from shapely.validation import explain_validity

AREA_CRS = "EPSG:6933"
LAT_LIMIT = 86.0


@lru_cache(maxsize=4)
def _to_equal_area(crs: str) -> Transformer:
    return Transformer.from_crs("EPSG:4326", crs, always_xy=True)


def _as_geometry(geom: BaseGeometry | Mapping[str, Any]) -> BaseGeometry:
    return geom if isinstance(geom, BaseGeometry) else shape(geom)


def validate_order_polygon(geom: BaseGeometry | Mapping[str, Any]) -> Polygon:
    """Вернуть Polygon или ValueError: не Polygon (в т. ч. MultiPolygon), пустой, невалидный, с holes."""
    g = _as_geometry(geom)
    if g.geom_type != "Polygon":
        raise ValueError(f"контур заказа должен быть Polygon, получен {g.geom_type}")
    if g.is_empty:
        raise ValueError("контур заказа пуст")
    if not g.is_valid:
        raise ValueError(f"контур заказа не valid: {explain_validity(g)}")
    if len(g.interiors):
        raise ValueError(f"контур заказа содержит holes: {len(g.interiors)}")
    return g


def polygon_area_km2(geom: BaseGeometry | Mapping[str, Any], crs: str = AREA_CRS) -> float:
    """Неокруглённая площадь контура в км² (м² в EPSG:6933 / 1e6)."""
    poly = validate_order_polygon(geom)
    minx, miny, maxx, maxy = poly.bounds
    if miny < -LAT_LIMIT or maxy > LAT_LIMIT or minx < -180 or maxx > 180:
        raise ValueError(f"контур вне области применения {crs} (широта ±{LAT_LIMIT:g}°, "
                         f"долгота ±180°): bounds={poly.bounds}")
    projected = transform(_to_equal_area(crs).transform, poly)
    if not projected.is_valid:
        raise ValueError(f"контур не valid после перевода в {crs}: {explain_validity(projected)}")
    return projected.area / 1e6


def order_area_km2(geom: BaseGeometry | Mapping[str, Any], min_area_km2: float = 1.0,
                   step: str = "0.001", crs: str = AREA_CRS) -> Decimal:
    """Оплачиваемая площадь: проверка ≥ min_area_km2 до округления, затем ROUND_HALF_UP(step)."""
    raw = polygon_area_km2(geom, crs)
    if raw < min_area_km2:
        raise ValueError(f"площадь {raw:.6f} км² < {min_area_km2} км² (проверка до округления)")
    return Decimal(repr(raw)).quantize(Decimal(step), rounding=ROUND_HALF_UP)
