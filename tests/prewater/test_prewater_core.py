"""Pre-event water (#149): window, water frequency, new water, diagnostics — pure functions."""

from datetime import date

import numpy as np
import pytest

from floodvalue.prewater import core

NAN = np.nan


def test_window_ends_guard_days_before_s1_and_spans_window_days():
    first, last = core.pre_event_window(date(2017, 5, 30))
    assert last == date(2017, 3, 30)  # 60 full days (Mar 31 … May 29) between window and S1
    assert (last - first).days + 1 == core.WINDOW_DAYS
    assert (date(2017, 5, 30) - last).days == core.GUARD_DAYS + 1


def test_parameters_are_the_fixed_ones():
    # Fixed before the table (#149 p.5): changing them is a new decision, not tuning.
    assert (core.VH_WATER_DB, core.MIN_WATER_SHARE, core.N_MIN_SCENES) == (-23.0, 0.5, 5)
    assert (core.WINDOW_DAYS, core.GUARD_DAYS) == (100, 60)


def test_linear_to_db():
    out = core.linear_to_db(np.array([1.0, 0.01, 0.0, -1.0, NAN, 0.005]))
    assert out[0] == pytest.approx(0.0) and out[1] == pytest.approx(-20.0)
    assert np.isnan(out[2:5]).all()
    assert out[5] == pytest.approx(-23.0103, abs=1e-3)
    assert out.dtype == np.float32


def test_permanent_water_share_threshold_and_nodata():
    # 6 scenes, 4 pixels: always water / water in 3 of 6 / water in 2 of 6 / 4 valid obs only
    px = np.array([
        [-30, -25, -10, -30],
        [-30, -25, -10, -30],
        [-30, -25, -30, -30],
        [-30, -10, -30, -30],
        [-30, -10, -10, NAN],
        [-30, -10, -10, NAN],
    ], dtype=np.float32)
    perm = core.permanent_water(px[:, None, :])[0]
    assert perm.tolist() == [1, 1, 0, 255]  # 3/6 = 0.5 counts; 4 < N_min -> nodata
    assert perm.dtype == np.uint8


def test_threshold_is_inclusive_like_baseline():
    stack = np.full((5, 1, 1), -23.0, dtype=np.float32)
    assert core.permanent_water(stack)[0, 0] == 1
    assert core.permanent_water(stack + 0.01)[0, 0] == 0


def test_nan_never_counts_as_water():
    stack = np.full((10, 1, 1), NAN, dtype=np.float32)
    stack[:5] = -10.0
    n_valid, n_water = core.water_frequency(stack)
    assert (int(n_valid[0, 0]), int(n_water[0, 0])) == (5, 0)


def test_layer_status():
    assert core.layer_status(5) == core.STATUS_OK
    assert core.layer_status(4) == core.STATUS_INSUFFICIENT


def test_new_water_probability_removes_only_pre_event_water():
    p = np.array([0.9, 0.9, 0.2, -9999.0, NAN, 0.7], dtype=np.float32)
    perm = np.array([1, 0, 0, 0, 0, 255], dtype=np.uint8)
    out = core.new_water_probability(p, perm)
    assert out.tolist() == pytest.approx([0.0, 0.9, 0.2, -9999.0, -9999.0, -9999.0])
    assert out.dtype == np.float32
    # never adds probability, keeps p below any threshold as is (p from the raster, not the mask)
    ok = out != -9999.0
    assert (out[ok] <= p[ok]).all()


def test_diagnostics_and_ratios():
    label = np.array([1, 1, 0, 0, -1, 1, 0])
    perm = np.array([1, 0, 1, 0, 1, 255, 0], dtype=np.uint8)
    jrc = np.array([1, 0, 0, 0, 1, 1, 1])
    row = core.diagnostics(label, perm, jrc)
    assert row["valid_px"] == 5  # label -1 and perm 255 dropped
    assert (row["perm_px"], row["perm_and_water_px"], row["perm_and_dry_px"]) == (2, 1, 1)
    assert (row["label_water_px"], row["label_dry_px"]) == (2, 3)
    assert (row["jrc_px"], row["perm_and_jrc_px"], row["perm_or_jrc_px"]) == (2, 1, 3)
    r = core.ratios(row)
    assert r["p_water_given_perm"] == pytest.approx(0.5)
    assert r["perm_on_dry_share"] == pytest.approx(1 / 3)
    assert r["removed_water_share"] == pytest.approx(0.5)
    assert r["perm_share"] == pytest.approx(0.4)
    assert r["iou_jrc"] == pytest.approx(1 / 3)


def test_ratios_empty_denominators_are_none():
    row = core.diagnostics(np.array([0, 0]), np.array([0, 0], dtype=np.uint8))
    r = core.ratios(row)
    assert r["p_water_given_perm"] is None and r["removed_water_share"] is None
    assert r["iou_jrc"] is None and r["perm_share"] == 0.0
