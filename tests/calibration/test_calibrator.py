import json

import numpy as np
import pytest

from floodvalue.calibration import calibrator as C


def synthetic(a=2.0, b=-1.0, n=400_000, seed=0):
    """p_raw = sigmoid(z); true P(water | z) = sigmoid(a z + b)."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 2, n)
    y = rng.random(n) < C.sigmoid(a * z + b)
    return C.sigmoid(z).astype(np.float32), y


def test_platt_recovers_known_slope_and_bias():
    p, y = synthetic(a=2.0, b=-1.0)
    cal = C.fit_platt(p, y)
    assert cal["a"] == pytest.approx(2.0, abs=0.05)
    assert cal["b"] == pytest.approx(-1.0, abs=0.05)


def test_temperature_recovers_known_temperature():
    p, y = synthetic(a=0.5, b=0.0)
    cal = C.fit_temperature(p, y)
    assert cal["t"] == pytest.approx(2.0, rel=0.03)


def test_isotonic_is_monotone_and_reduces_ece():
    p, y = synthetic(a=2.0, b=-1.0)
    cal = C.fit_isotonic(p, y)
    assert np.all(np.diff(cal["y"]) >= 0)
    q = C.apply(cal, p)
    assert C.ece(q, y) < 0.01 < C.ece(p, y)


@pytest.mark.parametrize("kind", ["temperature", "platt", "isotonic"])
def test_every_fitter_improves_logloss_of_miscalibrated_p(kind):
    p, y = synthetic(a=2.0, b=-1.0)
    q = C.apply(C.FITTERS[kind](p, y), p)
    assert C.log_loss(q, y) < C.log_loss(p, y)


def test_already_calibrated_p_stays_put():
    p, y = synthetic(a=1.0, b=0.0)
    cal = C.fit_platt(p, y)
    assert cal["a"] == pytest.approx(1.0, abs=0.03) and cal["b"] == pytest.approx(0.0, abs=0.03)


def test_apply_keeps_nan_shape_dtype_and_range():
    p = np.array([[np.nan, 0.0, 0.2], [0.5, 1.0, np.nan]], dtype=np.float32)
    for cal in ({"kind": "identity"}, {"kind": "temperature", "t": 0.5},
                {"kind": "platt", "a": 3.0, "b": 2.0},
                {"kind": "isotonic", "x": [0.0, 0.5, 1.0], "y": [0.1, 0.3, 0.9]}):
        q = C.apply(cal, p)
        assert q.shape == p.shape and q.dtype == np.float32
        assert np.array_equal(np.isnan(q), np.isnan(p))
        assert np.nanmin(q) >= 0 and np.nanmax(q) <= 1


def test_apply_values():
    p = np.array([0.5, 0.25], dtype=np.float32)
    assert C.apply({"kind": "temperature", "t": 2.0}, p)[0] == pytest.approx(0.5)
    # logit(0.25) = -ln 3; /2 -> sigmoid = 1 / (1 + sqrt 3)
    assert C.apply({"kind": "temperature", "t": 2.0}, p)[1] == pytest.approx(1 / (1 + 3 ** 0.5), abs=1e-6)
    assert C.apply({"kind": "platt", "a": 1.0, "b": np.log(3)}, p)[1] == pytest.approx(0.5, abs=1e-6)
    assert C.apply({"kind": "isotonic", "x": [0.0, 0.5], "y": [0.0, 1.0]}, p)[1] == pytest.approx(0.5)


def test_apply_rejects_unknown_kind():
    with pytest.raises(ValueError):
        C.apply({"kind": "magic"}, np.zeros(3, np.float32))


def test_fit_ignores_nan_pixels():
    p, y = synthetic(a=2.0, b=-1.0, n=100_000)
    p2 = p.copy()
    p2[::3] = np.nan
    a = C.fit_platt(p2, y)
    b = C.fit_platt(p[np.isfinite(p2)], y[np.isfinite(p2)])
    assert a == pytest.approx(b)


def test_ece_brier_logloss_hand_values():
    p = np.array([0.1, 0.1, 0.9, 0.9])
    y = np.array([0, 1, 1, 1], dtype=bool)
    # 10 bins: bin 1 mean 0.1 share 0.5, bin 9 mean 0.9 share 1.0 -> 0.5*0.4 + 0.5*0.1
    assert C.ece(p, y, n_bins=10) == pytest.approx(0.25)
    assert C.brier(p, y) == pytest.approx((0.01 + 0.81 + 0.01 + 0.01) / 4)
    assert C.log_loss(p, y) == pytest.approx(-(np.log(0.9) + np.log(0.1) + 2 * np.log(0.9)) / 4)


def test_reliability_bin_edges_include_one():
    rows = C.reliability(np.array([0.0, 1.0, 0.999]), np.array([0, 1, 1], bool), n_bins=10)
    assert rows[0]["n"] == 1 and rows[-1]["n"] == 2
    assert sum(r["n"] for r in rows) == 3


def test_threshold_curve_counts_mask_p_ge_t():
    p = np.array([0.1, 0.3, 0.3, 0.7, 0.9, np.nan])
    y = np.array([0, 1, 0, 1, 1, 1], dtype=bool)
    rows = {r["threshold"]: r for r in C.threshold_curve(p, y, [0.3, 0.5, 0.95])}
    r = rows[0.3]  # p >= 0.3 -> 0.3, 0.3, 0.7, 0.9 (equality counts as water)
    assert (r["tp"], r["fp"], r["fn"], r["tn"]) == (3, 1, 0, 1)
    assert r["iou"] == pytest.approx(3 / 4)
    r = rows[0.5]
    assert (r["tp"], r["fp"], r["fn"], r["tn"]) == (2, 0, 1, 2)
    assert r["precision"] == 1 and r["recall"] == pytest.approx(2 / 3)
    assert np.isnan(rows[0.95]["precision"]) and rows[0.95]["iou"] == 0


def test_choose_threshold_max_and_lowest_on_tie():
    curve = [{"threshold": 0.2, "iou": 0.5}, {"threshold": 0.3, "iou": 0.7},
             {"threshold": 0.4, "iou": 0.7}, {"threshold": 0.5, "iou": float("nan")}]
    assert C.choose_threshold(curve) == 0.3


def test_load_checks_model_hash_and_threshold(tmp_path):
    path = tmp_path / "calibration.json"
    good = {"calibrator": {"kind": "platt", "a": 1.0, "b": 0.0}, "threshold": 0.4,
            "input_model_sha256": "abc"}
    path.write_text(json.dumps(good))
    assert C.load(path, "abc")["threshold"] == 0.4
    with pytest.raises(RuntimeError, match="make calibrate"):
        C.load(path, "other")
    path.write_text(json.dumps({**good, "threshold": 1.5}))
    with pytest.raises(ValueError):
        C.load(path)
    path.write_text(json.dumps({**good, "calibrator": {"kind": "magic"}}))
    with pytest.raises(ValueError):
        C.load(path)
