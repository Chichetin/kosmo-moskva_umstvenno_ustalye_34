import math

import numpy as np
import pytest
from rasterio.transform import from_origin

from floodvalue.assets.portfolio import (
    PROB_NODATA,
    Asset,
    assess_assets,
    binary_entropy,
    expected_loss,
    extract_p,
    load_asset_types,
    place_assets,
    rank_assets,
)

TRANSFORM = from_origin(81.0, 7.0, 0.001, 0.001)  # lon0, lat0 (top-left), dx, dy
TYPES = load_asset_types()


def valid_mask(h=20, w=20):
    m = np.ones((h, w), dtype=bool)
    m[:, :10] = False  # left half is S1 nodata
    return m


# --- EL: control arithmetic from task.md -------------------------------------------


def test_expected_loss_control_example():
    assert expected_loss(0.4, 10_000_000, 0.25) == pytest.approx(1_000_000, abs=1e-6)


def test_expected_loss_not_zeroed_below_mask_threshold():
    # p = 0.1 is below any sensible mask threshold, EL still = p*V*q
    assert expected_loss(0.1, 20_000_000, 0.40) == pytest.approx(800_000)


def test_expected_loss_is_rounded_half_up_to_the_kopeck():
    # #127: float product 28478.587199999998 was written as is; now ROUND_HALF_UP to 0.01
    assert 0.017799117 * 8_000_000 * 0.20 == 28478.587199999998
    assert expected_loss(0.017799117, 8_000_000, 0.20) == 28478.59
    # exact half kopeck in the written factors: p*V*q = 28478.585 -> 28478.59
    assert expected_loss(0.017799115625, 8_000_000, 0.20) == 28478.59
    # ... and where the float product is 28481.58499.. in binary, so round(x, 2) goes down
    assert round(0.017800990625 * 8_000_000 * 0.20, 2) == 28481.58
    assert expected_loss(0.017800990625, 8_000_000, 0.20) == 28481.59
    assert expected_loss(0.4, 10_000_000, 0.25) == 1_000_000.0  # task.md control, exact


def test_rank_follows_the_written_kopeck_el_and_ties_break_by_asset_id():
    # A02 is 0.004 rub above A01 before rounding; written EL are equal, so the rank that the
    # validator recomputes from asset_loss.csv (max EL, then asset_id) puts A01 first.
    a01 = Asset("A01", "warehouse", 10_000_000, 0.25, 0, 0, 81.0, 7.0)
    a02 = Asset("A02", "substation", 20_000_000, 0.40, 0, 1, 81.0, 7.0)
    prob = np.array([[0.0004, 1000.004 / 8_000_000]], dtype="float64")
    rows = {r["asset_id"]: r for r in assess_assets([a01, a02], prob)}
    assert rows["A01"]["expected_loss_rub"] == rows["A02"]["expected_loss_rub"] == 1000.0
    assert (rows["A01"]["rank"], rows["A02"]["rank"]) == (1, 2)


def test_el_range_is_rounded_half_up_to_the_kopeck():
    a = Asset("A01", "facility", 8_000_000, 0.20, 0, 0, 81.0, 7.0)
    prob = np.array([[0.02]], dtype="float64")
    lo = np.array([[0.017800990625]], dtype="float64")  # half kopeck, binary below it
    hi = np.array([[0.017799117]], dtype="float64")
    row = assess_assets([a], prob, p_range=(lo, hi))[0]
    assert (row["el_low_rub"], row["el_high_rub"]) == (28481.59, 28478.59)


# --- config ------------------------------------------------------------------------


def test_asset_types_are_the_ten_from_task():
    assert len(TYPES) == 10
    assert len({t.asset_class for t in TYPES}) == 10
    wh = next(t for t in TYPES if t.asset_class == "warehouse")
    assert (wh.asset_value_rub, wh.vulnerability_coef) == (10_000_000, 0.25)


# --- placement ---------------------------------------------------------------------


def test_place_exactly_ten_one_per_type_in_valid_pixels():
    mask = valid_mask()
    assets = place_assets(mask, TRANSFORM, seed=7, asset_types=TYPES)
    assert len(assets) == 10
    assert sorted(a.asset_class for a in assets) == sorted(t.asset_class for t in TYPES)
    assert len({a.asset_id for a in assets}) == 10
    assert len({(a.row, a.col) for a in assets}) == 10  # one asset per pixel
    for a in assets:
        assert mask[a.row, a.col]
        t = next(t for t in TYPES if t.asset_class == a.asset_class)
        assert (a.asset_value_rub, a.vulnerability_coef) == (t.asset_value_rub, t.vulnerability_coef)


def test_place_pixel_centre_lon_lat_order():
    assets = place_assets(valid_mask(), TRANSFORM, seed=7, asset_types=TYPES)
    for a in assets:
        assert a.longitude == pytest.approx(81.0 + (a.col + 0.5) * 0.001)
        assert a.latitude == pytest.approx(7.0 - (a.row + 0.5) * 0.001)


def test_place_is_deterministic_for_seed_and_changes_with_seed():
    a1 = place_assets(valid_mask(), TRANSFORM, seed=7, asset_types=TYPES)
    a2 = place_assets(valid_mask(), TRANSFORM, seed=7, asset_types=TYPES)
    a3 = place_assets(valid_mask(), TRANSFORM, seed=8, asset_types=TYPES)
    assert [(a.row, a.col) for a in a1] == [(a.row, a.col) for a in a2]
    assert [(a.row, a.col) for a in a1] != [(a.row, a.col) for a in a3]


def test_placement_does_not_depend_on_p_or_labels():
    # place_assets has no p/label/EL input at all; the same mask gives the same points,
    # and assessing them with two very different p-rasters leaves the points unchanged.
    mask = valid_mask()
    rng = np.random.default_rng(0)
    p_a = rng.random(mask.shape).astype("float32")
    p_b = (1 - p_a).astype("float32")
    pts_a = place_assets(mask, TRANSFORM, seed=11, asset_types=TYPES)
    assess_assets(pts_a, p_a)
    pts_b = place_assets(mask, TRANSFORM, seed=11, asset_types=TYPES)
    assess_assets(pts_b, p_b)
    assert [(a.asset_id, a.row, a.col) for a in pts_a] == [(a.asset_id, a.row, a.col) for a in pts_b]
    import inspect

    params = set(inspect.signature(place_assets).parameters)
    assert params == {"s1_valid_mask", "transform", "seed", "asset_types"}


def test_place_fails_when_not_enough_valid_pixels():
    mask = np.zeros((5, 5), dtype=bool)
    mask[0, :3] = True
    with pytest.raises(ValueError):
        place_assets(mask, TRANSFORM, seed=1, asset_types=TYPES)


def test_place_uses_single_valid_region_exactly():
    mask = np.zeros((30, 30), dtype=bool)
    mask[3:5, 20:25] = True  # exactly 10 valid pixels
    assets = place_assets(mask, TRANSFORM, seed=3, asset_types=TYPES)
    assert {(a.row, a.col) for a in assets} == set(zip(*np.nonzero(mask)))


# --- p extraction ------------------------------------------------------------------


def test_extract_p_reads_pixel_value_and_nodata():
    prob = np.full((4, 4), 0.3, dtype="float32")
    prob[1, 2] = 0.9
    prob[2, 2] = PROB_NODATA
    prob[3, 3] = np.nan
    assert extract_p(prob, [(1, 2), (0, 0), (2, 2), (3, 3), (9, 9)]) == [
        (pytest.approx(0.9), "ok"),
        (pytest.approx(0.3), "ok"),
        (None, "no_data"),
        (None, "no_data"),
        (None, "no_data"),
    ]


def test_extract_p_boundaries_zero_and_one_are_ok():
    prob = np.array([[0.0, 1.0]], dtype="float32")
    assert extract_p(prob, [(0, 0), (0, 1)]) == [(0.0, "ok"), (1.0, "ok")]


def test_extract_p_rejects_values_outside_unit_interval():
    prob = np.array([[1.5]], dtype="float32")
    with pytest.raises(ValueError):
        extract_p(prob, [(0, 0)])


# --- uncertainty -------------------------------------------------------------------


def test_binary_entropy():
    assert binary_entropy(0.5) == pytest.approx(1.0)
    assert binary_entropy(0.0) == 0.0
    assert binary_entropy(1.0) == 0.0
    assert binary_entropy(0.1) == pytest.approx(-(0.1 * math.log2(0.1) + 0.9 * math.log2(0.9)))
    assert binary_entropy(0.1) == pytest.approx(binary_entropy(0.9))


# --- ranking -----------------------------------------------------------------------


def test_rank_max_el_first_ties_by_asset_id_missing_empty():
    ranks = rank_assets({"A03": 5.0, "A01": 7.0, "A02": 5.0, "A04": None, "A05": 0.0})
    assert ranks == {"A01": 1, "A02": 2, "A03": 3, "A05": 4, "A04": None}


# --- end-to-end on arrays ----------------------------------------------------------


def test_assess_assets_rows():
    mask = np.ones((10, 10), dtype=bool)
    assets = place_assets(mask, TRANSFORM, seed=5, asset_types=TYPES)
    prob = np.full((10, 10), 0.4, dtype="float32")
    nd = assets[0]
    prob[nd.row, nd.col] = PROB_NODATA
    rows = {r["asset_id"]: r for r in assess_assets(assets, prob)}
    assert len(rows) == 10
    assert rows[nd.asset_id]["status"] == "no_data"
    assert rows[nd.asset_id]["p_flood"] is None
    assert rows[nd.asset_id]["expected_loss_rub"] is None
    assert rows[nd.asset_id]["uncertainty"] is None
    assert rows[nd.asset_id]["rank"] is None
    for a in assets[1:]:
        r = rows[a.asset_id]
        assert r["status"] == "ok"
        assert r["expected_loss_rub"] == pytest.approx(0.4 * a.asset_value_rub * a.vulnerability_coef)
        assert r["uncertainty"] == pytest.approx(binary_entropy(0.4))
    assert sorted(r["rank"] for r in rows.values() if r["rank"] is not None) == list(range(1, 10))
    top = max(assets[1:], key=lambda a: a.asset_value_rub * a.vulnerability_coef)
    assert rows[top.asset_id]["rank"] == 1
