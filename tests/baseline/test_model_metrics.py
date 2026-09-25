import math

import numpy as np
import pytest

from floodvalue.model import metrics as m


def test_valid_mask_needs_finite_s1_in_both_channels_and_label_not_minus_one():
    s1 = np.zeros((2, 2, 2), np.float32)
    s1[0, 0, 0] = np.nan  # VV missing
    s1[1, 0, 1] = np.nan  # VH missing
    label = np.array([[1, 1], [-1, 0]], np.int16)
    assert m.valid_mask(s1, label).tolist() == [[False, False], [False, True]]


def test_confusion_counts_only_valid_pixels():
    pred = np.array([1, 1, 0, 0, 1, 0], bool)
    label = np.array([1, 0, 1, 0, 1, 1], np.int16)
    valid = np.array([1, 1, 1, 1, 0, 0], bool)
    assert m.confusion(pred, label, valid) == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}


def test_scores_control_arithmetic():
    s = m.scores(tp=6, fp=2, fn=4, tn=88)
    assert s["iou"] == pytest.approx(6 / 12)
    assert s["precision"] == pytest.approx(6 / 8)
    assert s["recall"] == pytest.approx(6 / 10)
    assert s["f1"] == pytest.approx(2 * 6 / (2 * 6 + 2 + 4))
    assert s["n_pixels"] == 100


def test_scores_empty_denominator_is_none_not_zero():
    s = m.scores(tp=0, fp=0, fn=0, tn=10)
    assert s["iou"] is None and s["precision"] is None and s["recall"] is None
    assert s["f1"] is None
    s = m.scores(tp=0, fp=3, fn=0, tn=10)
    assert s["precision"] == 0 and s["recall"] is None and s["iou"] == 0


def test_mask_hash_depends_on_mask_and_chip_but_not_input_order():
    a, b = np.array([[1, 0]], bool), np.array([[0, 1]], bool)
    h = m.valid_mask_sha256([("X_1", a), ("Y_2", b)])
    assert h == m.valid_mask_sha256([("Y_2", b), ("X_1", a)])
    assert h != m.valid_mask_sha256([("X_1", b), ("Y_2", a)])
    assert h != m.valid_mask_sha256([("X_1", a), ("Z_2", b)])
    assert len(h) == 64


def test_row_has_contract_columns_and_blank_for_undefined():
    row = m.metrics_row("baseline", "dev", "USA", {"tp": 0, "fp": 0, "fn": 0, "tn": 5},
                        threshold=-17.0, valid_mask_sha256="ab")
    assert list(row) == m.METRICS_COLUMNS
    assert row["iou"] == "" and row["n_pixels"] == 5 and row["threshold"] == -17.0


def test_add_counts():
    assert m.add_counts({"tp": 1, "fp": 2, "fn": 3, "tn": 4},
                        {"tp": 1, "fp": 1, "fn": 1, "tn": 1}) == {"tp": 2, "fp": 3, "fn": 4, "tn": 5}
    assert not math.isnan(m.scores(**m.add_counts({"tp": 1, "fp": 0, "fn": 0, "tn": 0}, None))["iou"])
