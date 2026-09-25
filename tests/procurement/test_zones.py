"""Каталог зон и охват (критерий 15): геометрия, поля контракта, срок, covers."""
from datetime import UTC, datetime

from shapely.geometry import box, shape

from floodvalue.procurement import geometry, zones
from floodvalue.procurement.pricing import load_pricing

CFG = load_pricing()
T0 = datetime(2017, 5, 30, 12, tzinfo=UTC)
DEADLINE = datetime(2017, 5, 31, 0, tzinfo=UTC)
ASSETS = [{"asset_id": f"A{i:02d}", "longitude": 80.0 + 0.004 * i, "latitude": 7.0 + 0.003 * i}
          for i in range(1, 11)]
BOUNDS = (79.99, 6.99, 80.05, 7.04)
FIELDS = ["candidate_id", "sensor_type", "resolution_m", "acquisition_type", "data_role",
          "observation_at", "available_at", "processing_level", "usage_type",
          "guaranteed_purchase", "area_km2", "base_rate_rub_km2", "base_rate_status",
          "base_rate_source"]


def catalog():
    return zones.build_catalog(ASSETS, BOUNDS, T0, CFG)


def test_square_area_equal_area_projection():
    poly = zones.square_around(80.0, 7.0, 1.2)
    assert abs(geometry.polygon_area_km2(poly) - 1.44) < 1e-6


def test_catalog_fields_and_geometry():
    cat = catalog()
    ids = [f["properties"]["candidate_id"] for f in cat["features"]]
    assert len(ids) == len(set(ids)) == 14
    for f in cat["features"]:
        p = f["properties"]
        assert all(k in p for k in FIELDS)
        assert p["usage_type"] == "internal" and p["base_rate_status"] == "scenario"
        poly = geometry.validate_order_polygon(f["geometry"])
        raw = geometry.polygon_area_km2(poly)
        assert raw >= 1.0
        assert float(geometry.order_area_km2(poly)) == p["area_km2"]
        if p["acquisition_type"] == "new":
            assert p["guaranteed_purchase"] is True
        else:
            assert p["guaranteed_purchase"] is False
        lon, lat = shape(f["geometry"]).centroid.coords[0]
        assert 79 < lon < 81 and 6 < lat < 8  # [lon, lat]
    roles = {f["properties"]["data_role"] for f in cat["features"]}
    assert roles == {"event_observation", "context"}
    assert {f["properties"]["acquisition_type"] for f in cat["features"]} == {
        "new", "archive", "operational"}


def test_late_and_context_zones_do_not_cover():
    cat = catalog()
    cov = zones.coverage(cat, ASSETS, DEADLINE)
    assert cov["N-OPT-LATE"] == [] and cov["K-ARCH"] == [] and cov["K-OPER"] == []
    assert cov["W-CHIP"] == sorted(a["asset_id"] for a in ASSETS)
    assert "A03" in cov["Z-A03"]
    late = next(f for f in cat["features"] if f["properties"]["candidate_id"] == "N-OPT-LATE")
    assert "позже decision_deadline" in zones.exclusion_reason(late["properties"], DEADLINE)


def test_available_exactly_at_deadline_counts():
    props = {"data_role": "event_observation", "available_at": DEADLINE.isoformat()}
    assert zones.exclusion_reason(props, DEADLINE) is None
    props["available_at"] = "2017-05-31T00:00:01+00:00"
    assert zones.exclusion_reason(props, DEADLINE) is not None


def test_covers_includes_boundary_and_unique():
    poly = box(80.0, 7.0, 80.02, 7.02)
    cat = {"features": [
        {"geometry": poly.__geo_interface__, "properties": {
            "candidate_id": c, "data_role": "event_observation",
            "available_at": T0.isoformat()}} for c in ("x", "y")]}
    edge = [{"asset_id": "E", "longitude": 80.0, "latitude": 7.01}]
    cov = zones.coverage(cat, edge, DEADLINE)
    assert cov == {"x": ["E"], "y": ["E"]}
    assert zones.covered_assets(["x", "y"], cov) == {"E"}
