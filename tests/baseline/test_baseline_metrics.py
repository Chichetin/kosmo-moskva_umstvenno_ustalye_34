"""Baseline side of the shared metrics (floodvalue.model.metrics, algorithm shared with #17)."""

import math

import numpy as np
import pytest

from floodvalue.baseline import run
from floodvalue.data import io
from floodvalue.model import metrics as m


def test_valid_mask_needs_finite_s1_in_both_channels_and_label_not_minus_one():
    s1 = np.zeros((2, 2, 2), np.float32)
    s1[0, 0, 0] = np.nan  # VV missing
    s1[1, 0, 1] = np.nan  # VH missing
    label = np.array([[1, 1], [-1, 0]], np.int16)
    assert io.valid_mask(s1, label).tolist() == [[False, False], [False, True]]


def test_confusion_counts_only_valid_pixels_and_minus_one_is_not_water():
    pred = np.array([1, 1, 0, 0, 1, 0], bool)
    label = np.array([1, 0, 1, 0, 1, -1], np.int16)
    valid = np.array([1, 1, 1, 1, 0, 0], bool)
    assert m.confusion(pred, label, valid) == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}


def test_scores_control_arithmetic():
    s = m.scores({"tp": 6, "fp": 2, "fn": 4, "tn": 88})
    assert s["iou"] == pytest.approx(6 / 12)
    assert s["precision"] == pytest.approx(6 / 8)
    assert s["recall"] == pytest.approx(6 / 10)
    assert s["f1"] == pytest.approx(12 / 18)
    assert s["n_pixels"] == 100


def test_row_has_contract_columns_and_blank_for_undefined():
    row = run.metrics_row("baseline", "dev", "USA", {"tp": 0, "fp": 0, "fn": 0, "tn": 5},
                          threshold=-17.0, valid_mask_sha256="ab")
    assert list(row) == m.METRIC_COLUMNS
    assert row["iou"] == "" and row["recall"] == "" and row["n_pixels"] == 5
    assert row["threshold"] == -17.0
    row = run.metrics_row("baseline", "dev", "USA", {"tp": 0, "fp": 3, "fn": 0, "tn": 5},
                          threshold=None, valid_mask_sha256="ab")
    assert row["precision"] == 0 and row["iou"] == 0 and row["threshold"] == ""
    assert not any(isinstance(v, float) and math.isnan(v) for v in row.values())


def _stats(chip_id, valid):
    return run.ChipStats(chip_id, "dev", chip_id.split("_")[0], 0, 0, {}, {}, None,
                         {"tp": 0, "fp": 0, "fn": 0, "tn": 0}, valid)


def test_mask_hash_is_shared_algorithm_in_chip_id_order():
    a, b = np.array([[1, 0]], bool), np.array([[0, 1]], bool)
    h = run.mask_hash([_stats("Y_2", b), _stats("X_1", a)])
    assert h == m.mask_sha256([a, b])  # sorted by chip_id, same function the main method uses
    assert h != run.mask_hash([_stats("X_1", b), _stats("Y_2", a)])
