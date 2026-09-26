"""Pixel QC of the chips of an eval-test part (#64): pure arithmetic on synthetic arrays."""

import numpy as np
import pytest

from floodvalue.model import eval_test as E


def synthetic():
    """4×5 chip, 20 pixels, every cell of "NaN S1 × label" known by hand."""
    vv = np.full((4, 5), -10.0, np.float32)
    vh = np.full((4, 5), -20.0, np.float32)
    label = np.zeros((4, 5), np.int16)
    label[0, :] = 1                        # 5 water
    label[3, :] = -1                       # 5 unlabelled
    vv[0, 0] = np.nan                      # VV-only NaN under 1
    vh[1, 0] = np.nan                      # VH-only NaN under 0
    vv[3, 0] = vh[3, 0] = np.nan           # both NaN under -1
    vv[2, 1] = 3.0                         # VV > 0 dB
    vv[2, 2] = 0.0                         # exactly 0 dB is not > 0
    vh[2, 3] = 1.5                         # VH > 0 dB
    vh[2, 4] = 2.5                         # VH > 0 dB
    return np.stack([vv, vh]), label


def test_pixel_qc_counts_every_cell():
    s1, label = synthetic()
    q = E.pixel_qc(s1, label)
    assert q["n_pixels_total"] == 20
    assert q["vv_nan_share"] == pytest.approx(2 / 20)
    assert q["vh_nan_share"] == pytest.approx(2 / 20)
    assert q["s1_nan_share"] == pytest.approx(3 / 20)          # union, not sum
    assert q["label_invalid_share"] == pytest.approx(5 / 20)
    assert q["vv_pos_share"] == pytest.approx(1 / 20)
    assert q["vh_pos_share"] == pytest.approx(2 / 20)
    assert q["s1_nan_label_0_share"] == pytest.approx(1 / 20)
    assert q["s1_nan_label_1_share"] == pytest.approx(1 / 20)
    assert q["s1_nan_label_invalid_share"] == pytest.approx(1 / 20)
    assert q["label_other_share"] == 0.0


def test_pixel_qc_nan_label_cells_add_up_to_s1_nan():
    s1, label = synthetic()
    q = E.pixel_qc(s1, label)
    cells = q["s1_nan_label_0_share"] + q["s1_nan_label_1_share"] + \
        q["s1_nan_label_invalid_share"]
    assert cells == pytest.approx(q["s1_nan_share"])


def test_pixel_qc_flags_label_values_outside_the_scheme():
    s1, label = synthetic()
    label[1, 1] = 2
    label[1, 2] = -9999
    assert E.pixel_qc(s1, label)["label_other_share"] == pytest.approx(2 / 20)


def test_pixel_qc_all_nan_chip():
    s1 = np.full((2, 3, 3), np.nan, np.float32)
    label = np.array([[0, 1, -1]] * 3, np.int16)
    q = E.pixel_qc(s1, label)
    assert q["s1_nan_share"] == 1.0 and q["vv_pos_share"] == 0.0
    assert q["s1_nan_label_0_share"] == pytest.approx(1 / 3)
    assert q["s1_nan_label_1_share"] == pytest.approx(1 / 3)
    assert q["s1_nan_label_invalid_share"] == pytest.approx(1 / 3)


def test_pixel_qc_rejects_mismatched_shapes():
    s1, label = synthetic()
    with pytest.raises(ValueError):
        E.pixel_qc(s1, label[:3])


def qc_row(chip, event, **shares):
    base = {k: 0.0 for k in E.QC_SHARES}
    base.update(shares)
    return {"chip_id": chip, "event": event, "n_pixels_total": 100, **base}


def test_qc_summary_pools_pixels_and_counts_chips():
    rows = [qc_row("A_1", "A", s1_nan_share=0.6, vv_nan_share=0.6, label_invalid_share=0.1,
                   s1_nan_label_0_share=0.2, vv_pos_share=0.01),
            qc_row("A_2", "A", label_invalid_share=0.7),
            qc_row("B_1", "B", s1_nan_share=0.2, vv_nan_share=0.2,
                   s1_nan_label_invalid_share=0.2)]
    s = {g["scope"]: g for g in E.qc_summary(rows)}
    assert list(s) == ["A", "B", "ALL"]
    a, all_ = s["A"], s["ALL"]
    assert a["n_chips"] == 2 and a["s1_nan_share"] == pytest.approx(0.3)
    assert a["label_invalid_share"] == pytest.approx(0.4)
    assert a["chips_with_nan"] == 1 and a["chips_nan_gt_half"] == 1
    assert a["chips_nan_labelled"] == 1 and a["chips_invalid_gt_half"] == 1
    assert a["chips_vv_pos"] == 1
    assert all_["n_chips"] == 3 and all_["s1_nan_share"] == pytest.approx(0.8 / 3)
    assert all_["chips_with_nan"] == 2 and all_["chips_nan_labelled"] == 1   # B: NaN under −1


def test_qc_summary_weights_by_pixels_not_chips():
    rows = [qc_row("A_1", "A", s1_nan_share=1.0), qc_row("A_2", "A")]
    rows[1]["n_pixels_total"] = 300
    assert E.qc_summary(rows)[0]["s1_nan_share"] == pytest.approx(100 / 400)


def test_pixel_qc_vh_only_nan_under_invalid_label_counts():
    s1, label = synthetic()
    s1[1, 3, 1] = np.nan                   # VH-only NaN under -1
    q = E.pixel_qc(s1, label)
    assert q["s1_nan_label_invalid_share"] == pytest.approx(2 / 20)
    assert q["vv_nan_share"] == pytest.approx(2 / 20)


def test_qc_summary_half_is_not_more_than_half():
    """Same rule as data-qc.md: "NaN > 50 %" and "−1 > 50 %" are strict."""
    rows = [qc_row("A_1", "A", s1_nan_share=0.5, label_invalid_share=0.5),
            qc_row("A_2", "A", s1_nan_share=0.51, label_invalid_share=0.51)]
    g = E.qc_summary(rows)[0]
    assert g["chips_nan_gt_half"] == 1 and g["chips_invalid_gt_half"] == 1
