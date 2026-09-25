import numpy as np
import pytest

from floodvalue.model.pixel import features

CLIP = {"VV": [-40.0, 10.0], "VH": [-50.0, 0.0]}


def _brute(x, w, fn):
    r = w // 2
    pad = np.pad(x.astype(np.float64), r, constant_values=np.nan)
    out = np.full(x.shape, np.nan)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            win = pad[i:i + w, j:j + w]
            if np.isfinite(win).any():
                out[i, j] = fn(win)
    return out


@pytest.mark.parametrize("w", [3, 7])
def test_local_stats_equal_nan_aware_window(w):
    rng = np.random.default_rng(0)
    x = rng.normal(-15, 4, (13, 11))
    x[rng.random(x.shape) < 0.2] = np.nan
    x[5:9, 4:8] = np.nan  # a hole larger than the 3x3 window
    mean, std = features.local_stats(x, w)
    np.testing.assert_allclose(mean, _brute(x, w, np.nanmean), atol=1e-4, equal_nan=True)
    np.testing.assert_allclose(std, _brute(x, w, np.nanstd), atol=1e-3, equal_nan=True)
    assert mean.dtype == np.float32


def test_edges_are_not_zero_padded():
    x = np.full((6, 6), -20.0)
    mean, std = features.local_stats(x, 5)
    assert np.allclose(mean, -20.0) and np.allclose(std, 0.0, atol=1e-3)


def test_clip_keeps_nan_and_does_not_touch_input():
    s1 = np.array([[[40.0, -10.0, np.nan]], [[-104.0, -20.0, np.nan]]], np.float32)
    before = s1.copy()
    out = features.clip_db(s1, CLIP)
    assert out[0, 0, 0] == 10.0 and out[1, 0, 0] == -50.0
    assert out[0, 0, 1] == -10.0 and out[1, 0, 1] == -20.0  # dB kept, never re-logged
    assert np.isnan(out[:, 0, 2]).all()
    np.testing.assert_array_equal(s1, before)


def test_compute_order_ratio_and_invalid_pixels():
    s1 = np.full((2, 5, 5), -10.0, np.float32)
    s1[1] = -25.0
    s1[0, 2, 2] = 30.0            # clipped to 10 before VV-VH
    s1[0, 0, 0] = np.nan          # VV missing, VH present -> pixel invalid
    s1[1, 0, 1] = -80.0           # clipped to -50
    names = ["VH", "VV", "VV-VH", "VH_mean3", "VV_std3"]
    f = features.compute(s1, names, CLIP)
    assert f.shape == (5, 5, 5) and f.dtype == np.float32
    assert f[1, 2, 2] == 10.0 and f[2, 2, 2] == 35.0 and f[2, 3, 3] == 15.0
    assert np.isnan(f[:, 0, 0]).all()
    # VH of the invalid pixel (-25) must not enter its neighbour's window: (0,1) sees
    # (0,1)=-50, (0,2), (1,0), (1,1), (1,2) = -25 -> mean of 5 finite pixels
    assert f[3, 0, 1] == pytest.approx((-50 - 25 * 4) / 5, abs=1e-4)
    assert np.isfinite(f[:, 1:, :]).all()


def test_full_names_parse_and_unknown_fails():
    s1 = np.random.default_rng(1).normal(-15, 3, (2, 20, 20)).astype(np.float32)
    f = features.compute(s1, features.FULL, CLIP)
    assert f.shape == (15, 20, 20) and np.isfinite(f).all()
    with pytest.raises(ValueError):
        features.compute(s1, ["VV_median3"], CLIP)
    with pytest.raises(ValueError):
        features.compute(s1[0], ["VV"], CLIP)
