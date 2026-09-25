import math

import numpy as np

from floodvalue.model import metrics as M


def test_confusion_counts_only_masked_pixels():
    pred = np.array([1, 1, 0, 0, 1])
    target = np.array([1, 0, 1, 0, 1])
    mask = np.array([True, True, True, True, False])
    assert M.confusion(pred, target, mask) == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}


def test_scores_arithmetic():
    s = M.scores({"tp": 6, "fp": 2, "fn": 4, "tn": 88})
    assert s["iou"] == 6 / 12
    assert s["f1"] == 12 / 18
    assert s["precision"] == 6 / 8
    assert s["recall"] == 6 / 10
    assert s["n_pixels"] == 100


def test_scores_empty_denominator_is_nan_not_zero():
    s = M.scores({"tp": 0, "fp": 0, "fn": 0, "tn": 5})
    assert math.isnan(s["iou"]) and math.isnan(s["precision"]) and math.isnan(s["recall"])


def test_add_and_bootstrap_bounds():
    chips = [{"tp": 5, "fp": 5, "fn": 0, "tn": 0}, {"tp": 10, "fp": 0, "fn": 0, "tn": 0}]
    assert M.add(chips[0], chips[1]) == {"tp": 15, "fp": 5, "fn": 0, "tn": 0}
    lo, hi = M.bootstrap_iou(chips, n=500, seed=1)
    assert 0.5 <= lo <= 15 / 20 <= hi <= 1.0


def test_mask_sha256_changes_with_mask():
    a = np.zeros((4, 4), bool)
    b = a.copy()
    b[0, 0] = True
    assert M.mask_sha256([a]) != M.mask_sha256([b])
    assert M.mask_sha256([a]) == M.mask_sha256([a.copy()])
