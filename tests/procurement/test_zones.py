"""Каталог зон и охват (критерий 15): геометрия, поля контракта, срок, covers."""
from datetime import UTC, datetime
from decimal import Decimal

import numpy as np
import pytest
from rasterio.transform import from_origin
from shapely.geometry import Point, box, shape

from floodvalue.procurement import geometry, zones
from floodvalue.procurement.pricing import load_pricing, price_basket

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


LOSS = [{"asset_id": a["asset_id"], "status": "ok", "expected_loss_rub": str(el),
         "uncertainty": "0.3"}
        for a, el in zip(ASSETS, [5e6, 4e6, 1e5, 1e5, 1e5, 1e5, 1e5, 1e5, 1e5, 3e6], strict=True)]
TR = from_origin(BOUNDS[0], BOUNDS[3], 8.983e-05, 8.983e-05)


def unc_raster():
    """Два «горячих» пятна неопределённости: большое (северо-запад) и малое (юго-восток)."""
    u = np.zeros((512, 512))
    u[40:120, 40:140] = 1.0
    u[400:432, 400:432] = 0.9
    u[0:16, 480:512] = np.nan
    return u


def catalog(**kw):
    return zones.build_catalog(ASSETS, BOUNDS, T0, CFG, **kw)


def full_catalog():
    return catalog(loss=LOSS, uncertainty=unc_raster(), transform=TR, decision_deadline=DEADLINE)


def by_id(cat):
    return {f["properties"]["candidate_id"]: f for f in cat["features"]}


def test_square_area_equal_area_projection():
    poly = zones.square_around(80.0, 7.0, 1.2)
    assert abs(geometry.polygon_area_km2(poly) - 1.44) < 1e-6


@pytest.mark.parametrize("make", [catalog, full_catalog])
def test_catalog_fields_and_geometry(make):
    cat = make()
    ids = [f["properties"]["candidate_id"] for f in cat["features"]]
    assert len(ids) == len(set(ids))
    for f in cat["features"]:
        p = f["properties"]
        assert all(k in p for k in FIELDS)
        assert p["usage_type"] == "internal" and p["base_rate_status"] == "scenario"
        poly = geometry.validate_order_polygon(f["geometry"])  # Polygon, valid, без holes
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


def test_full_catalog_has_all_zone_kinds():
    cat = full_catalog()
    kinds = {f["properties"]["zone_kind"] for f in cat["features"]}
    assert kinds == {"asset_buffer", "el_cluster", "uncertainty_cluster", "grid", "wide"}
    geoms = {f["properties"]["geometry_id"] for f in cat["features"]}
    assert {f"G2-{r}{c}" for r in (1, 2) for c in (1, 2)} <= geoms
    assert {f"G3-{r}{c}" for r in (1, 2, 3) for c in (1, 2, 3)} <= geoms
    assert {"U-1", "U-2", "W-CHIP", "Z-A01", "E-1"} <= geoms
    assert "E-ALL" not in geoms or len(cat["build"]["el_clusters"]["clusters"]) > 2


def test_zones_without_loss_or_raster_are_skipped_with_note():
    cat = catalog()
    kinds = {f["properties"]["zone_kind"] for f in cat["features"]}
    assert "el_cluster" not in kinds and "uncertainty_cluster" not in kinds
    assert "не строились" in cat["build"]["el_clusters"]
    assert "не строились" in cat["build"]["uncertainty_clusters"]


def test_variants_share_geometry_and_area():
    """Варианты сенсора/разрешения/типа съёмки одной геометрии: тот же контур и площадь."""
    cat = by_id(full_catalog())
    w = [f for c, f in cat.items() if f["properties"]["geometry_id"] == "W-CHIP"]
    assert {f["properties"]["offer_id"] for f in w} == set(zones.OFFERS)
    assert len({f["properties"]["area_km2"] for f in w}) == 1
    assert len({str(f["geometry"]) for f in w}) == 1
    assert {(f["properties"]["sensor_type"], f["properties"]["resolution_m"]) for f in w} >= {
        ("sar", 1), ("sar", 3), ("optical", 0.5), ("optical", 1)}


def test_exclusion_reasons_recorded_in_catalog():
    cat = by_id(full_catalog())
    reason = {c: f["properties"]["exclusion_reason"] for c, f in cat.items()}
    assert reason["W-CHIP"] is None and reason["Z-A01"] is None and reason["Z-A01-S3"] is None
    assert "позже decision_deadline" in reason["W-CHIP-OL"]
    assert "облачность" in reason["Z-A01-O05"]            # оптика к сроку, но облачность
    assert "срок доступности не известен" in reason["W-CHIP-OP"]
    assert "контекстный" in reason["W-CHIP-AR"] and "контекстный" in reason["W-CHIP-OC"]
    assert cat["W-CHIP-AR"]["properties"]["acquisition_type"] == "archive"
    assert cat["W-CHIP-OC"]["properties"]["acquisition_type"] == "operational"
    assert cat["W-CHIP-OP"]["properties"]["acquisition_type"] == "operational"
    assert cat["W-CHIP-OP"]["properties"]["data_role"] == "event_observation"


def test_single_price_by_contour_area():
    """Цена позиции = price_basket одной позиции по её площади; площадь — по контуру."""
    cat = by_id(full_catalog())
    for cid in ("Z-A01", "E-1", "G3-11", "W-CHIP", "W-CHIP-AR", "W-CHIP-OP"):
        p = cat[cid]["properties"]
        row = price_basket([{**p, "area_km2": Decimal(str(p["area_km2"]))}], CFG,
                           as_of=T0.date())[0]
        assert p["single_order_cost_rub"] == float(row["cost_rub"])
    z = cat["Z-A01"]["properties"]
    assert z["area_km2"] == 1.44 and z["single_order_cost_rub"] == 2452.98
    assert cat["W-CHIP-AR"]["properties"]["single_order_cost_rub"] < cat["W-CHIP"][
        "properties"]["single_order_cost_rub"]  # Т архива 0,6 < 1,8


def test_late_and_context_zones_do_not_cover():
    cat = full_catalog()
    cov = zones.coverage(cat, ASSETS, DEADLINE)
    for cid in ("W-CHIP-OL", "W-CHIP-AR", "W-CHIP-OC", "W-CHIP-OP", "Z-A03-O05"):
        assert cov[cid] == []
    assert cov["W-CHIP"] == sorted(a["asset_id"] for a in ASSETS)
    assert "A03" in cov["Z-A03"] and "A03" in cov["Z-A03-S3"]


def test_excluded_positions_have_empty_covers():
    """У исключённых позиций охват пуст; объекты внутри контура — отдельным полем (#105)."""
    cat = by_id(full_catalog())
    for cid in ("W-CHIP-OL", "W-CHIP-AR", "W-CHIP-OC", "W-CHIP-OP", "Z-A03-O05"):
        p = cat[cid]["properties"]
        assert p["exclusion_reason"] is not None
        assert p["covers_asset_ids"] == []
        assert p["inside_geometry_asset_ids"] == cat[p["geometry_id"]]["properties"][
            "covers_asset_ids"]
    assert cat["W-CHIP"]["properties"]["covers_asset_ids"] == sorted(a["asset_id"] for a in ASSETS)
    assert "inside_geometry_asset_ids" not in cat["W-CHIP"]["properties"]


def test_el_clusters_share_and_link():
    high, clusters = zones.el_clusters(ASSETS, LOSS)
    # 5e6 + 4e6 + 3e6 = 12e6 из 12,7e6 (≥ 90 %) — три объекта высокого EL
    assert high == ["A01", "A02", "A10"]
    # A01–A02 ≈ 0,55 км, A10 — ≈ 4,5 км от них: два кластера, больший EL первым
    assert clusters == [["A01", "A02"], ["A10"]]
    assert zones.el_clusters(ASSETS, [{**r, "expected_loss_rub": "0"} for r in LOSS]) == ([], [])


def test_el_cluster_zone_is_hull_buffer_covering_members():
    cat = by_id(full_catalog())
    e1 = shape(cat["E-1"]["geometry"])
    for aid in ("A01", "A02"):
        a = next(x for x in ASSETS if x["asset_id"] == aid)
        assert e1.covers(Point(a["longitude"], a["latitude"]))
    assert "E-ALL" in cat  # два кластера → общая оболочка
    assert cat["E-2"]["properties"]["area_km2"] == pytest.approx(1.124, abs=0.01)  # круг 600 м


def test_uncertainty_clusters_ranked_by_score():
    ucl = zones.uncertainty_clusters(unc_raster(), TR)
    assert len(ucl) == 2
    assert ucl[0]["score"] > ucl[1]["score"]
    assert ucl[0]["n_blocks"] > ucl[1]["n_blocks"]
    cat = by_id(full_catalog())
    u1 = shape(cat["U-1"]["geometry"])
    x, y = TR @ (90, 80)  # центр большого пятна (col, row)
    assert u1.covers(Point(x, y))


def test_uncertainty_clusters_empty_on_flat_raster():
    assert zones.uncertainty_clusters(np.zeros((64, 64)), TR) == []
    assert zones.uncertainty_clusters(np.full((64, 64), np.nan), TR) == []


def test_buffered_hull_grows_to_min_area_inside_chip():
    chip = box(*BOUNDS)
    poly, r = zones.buffered_hull([(BOUNDS[0], BOUNDS[1])], 100.0, chip)  # угол чипа
    assert r > 100.0
    assert geometry.polygon_area_km2(poly) >= 1.0
    assert chip.buffer(1e-12).covers(poly) and len(poly.interiors) == 0


def test_grid_cells_cover_chip_and_skip_small_chip():
    cells = zones.grid_cells(BOUNDS, 3)
    assert len(cells) == 9 and cells[0][0] == "G3-11"
    union = cells[0][1]
    for _, c in cells[1:]:
        union = union.union(c)
    assert union.equals(box(*BOUNDS))
    small = zones.build_catalog(ASSETS[:1], (80.0, 7.0, 80.015, 7.015), T0, CFG)
    assert "G3" in small["build"]["grid"]["skipped"]


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


def test_el_share_boundary_inclusive():
    """Накопленный EL ровно 90 % — набор высокого EL закрыт (A02 не добавляется)."""
    loss = [{"asset_id": "A01", "status": "ok", "expected_loss_rub": "9"},
            {"asset_id": "A02", "status": "ok", "expected_loss_rub": "1"}]
    high, _ = zones.el_clusters(ASSETS, loss, share=0.9)
    assert high == ["A01"]
