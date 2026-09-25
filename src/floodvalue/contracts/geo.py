"""Shared rounding and area arithmetic (task.md: EPSG:6933, ROUND_HALF_UP, 1 km² before rounding)."""

from decimal import ROUND_HALF_UP, Decimal

from pyproj import Transformer
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform

AREA_CRS = "EPSG:6933"
MIN_AREA_KM2 = 1.0
_TO_EQUAL_AREA = Transformer.from_crs("EPSG:4326", AREA_CRS, always_xy=True)


def round_half_up(value: float, quantum: str) -> float:
    """ROUND_HALF_UP on the decimal representation of `value` (not banker's `round`)."""
    return float(Decimal(repr(value)).quantize(Decimal(quantum), rounding=ROUND_HALF_UP))


def area_km2(geom_lonlat: BaseGeometry) -> float:
    """Unrounded area in km² of a WGS 84 [lon, lat] geometry, computed in EPSG:6933."""
    return transform(_TO_EQUAL_AREA.transform, geom_lonlat).area / 1e6


def product_half_up(factors: list[float], quantum: str) -> float:
    """Exact decimal product of the written factors, then ROUND_HALF_UP to `quantum`.

    Used for unit_price_rub_km2 = Б × О × П × Т × Р (0.01) and cost_rub = unit × area (0.01),
    so producer and validator agree to the kopeck regardless of float noise.
    """
    total = Decimal(1)
    for factor in factors:
        total *= Decimal(repr(float(factor)))
    return float(total.quantize(Decimal(quantum), rounding=ROUND_HALF_UP))
