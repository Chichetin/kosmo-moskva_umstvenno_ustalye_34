import numpy as np
import pytest
from scipy.special import expit

from floodvalue.baseline import core
from floodvalue.model import metrics


def test_counts_le_is_inclusive_and_handles_out_of_range():
    edges = np.array([-20.0, -10.0, 0.0])
    v = np.array([-30, -20, -15, -10, -10, 5], np.float32)
    assert core.counts_le(v, edges).tolist() == [2, 5, 5]


def test_curve_counts_equal_direct_pixel_confusion_at_every_edge():
    rng = np.random.default_rng(0)
    vh = rng.normal(-18, 5, 5000).astype(np.float32)
    vh[::7] = np.round(vh[::7], 1)  # values exactly on edges
    label = (rng.random(5000) < expit((-18 - vh) / 2)).astype(np.int16)
    edges = core.EDGES
    w = core.counts_le(vh[label == 1], edges)
    d = core.counts_le(vh[label == 0], edges)
    c = core.curve_counts(w, d, int((label == 1).sum()), int((label == 0).sum()))
    valid = np.ones(5000, bool)
    for k in (0, 300, 420, 450, 600, len(edges) - 1):
        direct = metrics.confusion(vh <= edges[k], label, valid)
        assert {key: int(c[key][k]) for key in direct} == direct


def test_best_threshold_maximises_iou():
    edges = np.array([-3.0, -2.0, -1.0, 0.0])
    # water all at -2.5, dry at -1.5 and -0.5
    w = core.counts_le(np.array([-2.5] * 10), edges)
    d = core.counts_le(np.array([-1.5] * 3 + [-0.5] * 5), edges)
    t, iou = core.best_threshold(edges, w, d, 10, 8)
    assert t == -2.0 and iou == pytest.approx(1.0)
    # lower threshold misses water, higher adds dry
    t, _ = core.best_threshold(edges, w, core.counts_le(np.array([-2.2] * 30), edges), 10, 30)
    assert t == -3.0 or t == -2.0


def test_sigmoid_proba_direction_center_and_nan():
    s1 = np.array([[[0.0, 0.0, np.nan, 0.0]], [[-25.0, -17.0, -17.0, np.nan]]], np.float32)
    p = core.sigmoid_proba(s1, channel=1, t=-17.0, s=2.0)
    assert p.dtype == np.float32 and p.shape == (1, 4)
    assert p[0, 0] == pytest.approx(expit(8 / 2))  # darker than threshold -> water
    assert p[0, 1] == pytest.approx(0.5)
    assert np.isnan(p[0, 2]) and np.isnan(p[0, 3])  # NaN in any channel -> no data
    assert p[0, 0] > 0.5


def test_mask_consistent_with_threshold_rule():
    rng = np.random.default_rng(1)
    s1 = rng.normal(-17, 6, (2, 64, 64)).astype(np.float32)
    p = core.sigmoid_proba(s1, 1, -17.3, 1.7)
    assert np.array_equal(p >= 0.5, s1[1] <= -17.3)


def test_fit_scale_recovers_true_scale():
    edges = core.EDGES
    centers = core.bin_centers(edges)
    t, s_true = -18.0, 1.5
    q = expit((t - centers) / s_true)
    n = 1e6 * np.exp(-0.5 * ((centers + 17) / 6) ** 2)
    s = core.fit_scale(t, centers, n * q, n * (1 - q))
    assert s == pytest.approx(s_true, rel=0.02)


def test_hist_from_counts_le_roundtrip():
    v = np.array([-70, -20.05, -20.0, 3.3, 100], np.float64)
    h = core.hist_from_le(core.counts_le(v, core.EDGES))
    assert h.sum() == v.size - 1  # above the last edge is outside
    assert h[0] == 1  # below the first edge lands in the first bin


def test_otsu_splits_bimodal():
    rng = np.random.default_rng(2)
    v = np.concatenate([rng.normal(-25, 1, 3000), rng.normal(-12, 1, 7000)]).astype(np.float32)
    t = core.otsu_threshold(v)
    assert -22 < t < -15
    assert core.otsu_threshold(np.array([np.nan, -5.0, -5.0], np.float32)) == -5.0
