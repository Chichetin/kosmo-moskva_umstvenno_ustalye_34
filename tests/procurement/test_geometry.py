"""Площадь и валидность контура заказа (task.md «Площадь и геометрия заказа»)."""
from decimal import Decimal

import pytest
from pyproj import Transformer
from shapely.geometry import MultiPolygon, Polygon, box

from floodvalue.procurement import geometry

TO_WGS = Transformer.from_crs("EPSG:6933", "EPSG:4326", always_xy=True)


def square_km2(area_km2, x0=3_000_000.0, y0=2_000_000.0):
    """Квадрат заданной площади в EPSG:6933, вершины в lon/lat."""
    side = (area_km2 * 1e6) ** 0.5
    pts = [(x0, y0), (x0 + side, y0), (x0 + side, y0 + side), (x0, y0 + side)]
    return Polygon([TO_WGS.transform(x, y) for x, y in pts])


def test_area_in_equal_area_projection_lon_lat_order():
    poly = square_km2(2.5)
    assert geometry.polygon_area_km2(poly) == pytest.approx(2.5, abs=1e-6)
    # перепутанный порядок осей дал бы другое число
    swapped = Polygon([(y, x) for x, y in poly.exterior.coords])
    assert abs(geometry.polygon_area_km2(swapped) - 2.5) > 1e-3


def test_order_area_rounded_half_up_to_0_001():
    assert geometry.order_area_km2(square_km2(1.2346)) == Decimal("1.235")
    assert geometry.order_area_km2(square_km2(1.2344)) == Decimal("1.234")
    assert geometry.order_area_km2(square_km2(1.0)) == Decimal("1.000")


def test_order_area_half_up_on_exact_tie(monkeypatch):
    monkeypatch.setattr(geometry, "polygon_area_km2", lambda *a, **k: 1.0005)
    assert geometry.order_area_km2(box(0, 0, 1, 1)) == Decimal("1.001")  # HALF_EVEN дал бы 1.000


def test_09996_rejected_before_rounding():
    with pytest.raises(ValueError, match="1"):
        geometry.order_area_km2(square_km2(0.9996))


def test_invalid_self_intersection_rejected():
    bowtie = Polygon([(10, 10), (10.1, 10.1), (10.1, 10), (10, 10.1)])
    with pytest.raises(ValueError, match="valid"):
        geometry.validate_order_polygon(bowtie)


def test_holes_rejected():
    outer = box(10, 10, 10.1, 10.1)
    with_hole = Polygon(outer.exterior.coords, [box(10.04, 10.04, 10.06, 10.06).exterior.coords])
    with pytest.raises(ValueError, match="hole"):
        geometry.validate_order_polygon(with_hole)


def test_multipolygon_rejected():
    mp = MultiPolygon([box(10, 10, 10.1, 10.1), box(11, 11, 11.1, 11.1)])
    with pytest.raises(ValueError, match="Polygon"):
        geometry.validate_order_polygon(mp)


def test_empty_rejected():
    with pytest.raises(ValueError):
        geometry.validate_order_polygon(Polygon())


def test_outside_6933_domain_rejected():
    with pytest.raises(ValueError, match="86"):
        geometry.polygon_area_km2(box(10, 86.1, 10.5, 86.5))


def test_accepts_geojson_mapping():
    poly = square_km2(1.5)
    gj = {"type": "Polygon", "coordinates": [list(poly.exterior.coords)]}
    assert geometry.order_area_km2(gj) == Decimal("1.500")
