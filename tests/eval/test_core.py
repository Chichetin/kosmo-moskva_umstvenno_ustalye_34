"""Control arithmetic of the final-evaluation functions (#29)."""

import math

import numpy as np
import pytest
from rasterio.transform import from_origin

from floodvalue.model.eval import core as C


def conf(tp, fp, fn, tn=0):
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def test_same_valid_pixels_passes_and_mismatch_raises():
    assert C.require_same_valid_pixels({"main": "ab", "baseline": "ab"}) == "ab"
    with pytest.raises(C.ValidMaskMismatch):
        C.require_same_valid_pixels({"main": "ab", "baseline": "cd"})


def test_paired_bootstrap_identical_methods_give_zero():
    chips = [conf(5, 1, 2), conf(0, 3, 0), conf(10, 0, 5)]
    b = C.paired_bootstrap_delta_iou(chips, chips, n=200, seed=1)
    assert b["delta"] == 0 and b["ci95"] == [0.0, 0.0] and b["share_gt_0"] == 0.0


def test_paired_bootstrap_sign_and_point_estimate():
    main = [conf(9, 1, 0), conf(8, 0, 2)]      # pooled IoU 17/20 = 0.85
    base = [conf(5, 3, 2), conf(4, 2, 4)]      # pooled IoU 9/20 = 0.45
    b = C.paired_bootstrap_delta_iou(main, base, n=500, seed=0)
    assert b["delta"] == pytest.approx(0.40)
    assert b["share_gt_0"] == 1.0 and b["ci95"][0] > 0


def test_paired_bootstrap_needs_same_chips():
    with pytest.raises(ValueError):
        C.paired_bootstrap_delta_iou([conf(1, 0, 0)], [], n=10)


def test_reliability_ece_brier_by_hand():
    p = np.array([0.05, 0.15, 0.95, 1.0])
    y = np.array([0, 1, 1, 1])
    c = C.reliability_counts(p, y, n_bins=10)
    assert c["n"].tolist() == [1, 1, 0, 0, 0, 0, 0, 0, 0, 2]   # p = 1 in the last bin
    # (|0.05 − 0| + |0.15 − 1| + |1.95 − 2|) / 4
    assert C.ece(c) == pytest.approx(0.2375)
    assert C.brier(c) == pytest.approx((0.0025 + 0.7225 + 0.0025 + 0) / 4)
    rows = C.reliability_table(c)
    assert rows[9]["mean_p"] == pytest.approx(0.975) and rows[9]["observed"] == 1.0
    assert math.isnan(rows[5]["mean_p"])
    assert C.mce(c) == pytest.approx(0.85)                     # bin [0.1; 0.2): |0.15 − 1|


def test_ece_uses_the_calibration_bins_and_matches_calibrator_ece():
    """#150: eval-test ECE is the calibration's ECE (15 bins, `calibration.run.N_BINS`)."""
    from floodvalue.calibration import calibrator as CAL
    from floodvalue.calibration import run as CAL_RUN

    assert C.N_BINS == CAL_RUN.N_BINS == 15
    rng = np.random.default_rng(1)
    p = rng.random(5000)
    p[:7] = [0.0, 1.0, 1 / 15, 2 / 15, 14 / 15, 0.5, 0.9999999]          # edges and ends
    y = rng.random(5000) < 0.5 + 0.45 * np.sin(20 * p)       # non-monotone miscalibration
    c = C.reliability_counts(p, y)
    assert len(c["n"]) == 15
    assert C.ece(c) == pytest.approx(CAL.ece(p, y, 15), rel=1e-12, abs=1e-15)
    assert C.log_loss(c) == pytest.approx(CAL.log_loss(p, y), rel=1e-12)
    assert C.brier(c) == pytest.approx(CAL.brier(p, y), rel=1e-12)
    assert C.ece(c) != pytest.approx(C.ece(C.reliability_counts(p, y, n_bins=10)), rel=1e-6)


def test_log_loss_by_hand_clipped_and_accumulated():
    a = C.reliability_counts(np.array([0.8, 0.3]), np.array([1, 0]))
    assert C.log_loss(a) == pytest.approx(-(math.log(0.8) + math.log(0.7)) / 2)
    b = C.reliability_counts(np.array([0.0]), np.array([1]))            # clipped at 1e-6
    assert C.log_loss(b) == pytest.approx(-math.log(1e-6))
    both = C.add_reliability(C.add_reliability(None, a), b)
    assert C.log_loss(both) == pytest.approx(
        -(math.log(0.8) + math.log(0.7) + math.log(1e-6)) / 3)
    s = C.prob_summary(both)
    assert s["n_pixels"] == 3 and s["mean_p"] == pytest.approx(1.1 / 3)
    assert s["water_share"] == pytest.approx(2 / 3)
    assert math.isnan(C.log_loss(C.reliability_counts(np.array([]), np.array([]))))
    assert math.isnan(C.mce(C.reliability_counts(np.array([]), np.array([]))))


def test_bootstrap_calibration_by_chips():
    one = C.reliability_counts(np.array([0.2, 0.9]), np.array([0, 1]))
    b = C.bootstrap_calibration([one], n=50, seed=3)
    assert b["ece_ci95"] == pytest.approx([C.ece(one)] * 2)          # one chip: no spread
    assert b["log_loss_ci95"] == pytest.approx([C.log_loss(one)] * 2)
    assert b["n_used"] == 50 and b["n_chips"] == 1
    good = C.reliability_counts(np.array([0.1, 0.9]), np.array([0, 1]))
    bad = C.reliability_counts(np.array([0.9, 0.1]), np.array([0, 1]))
    b2 = C.bootstrap_calibration([good, bad], n=400, seed=3)
    both = C.add_reliability(C.add_reliability(None, good), bad)
    lo, hi = b2["ece_ci95"]
    assert lo == pytest.approx(C.ece(good)) and hi == pytest.approx(C.ece(bad))
    assert lo <= C.ece(both) <= hi
    lo, hi = b2["log_loss_ci95"]
    assert lo == pytest.approx(C.log_loss(good)) and hi == pytest.approx(C.log_loss(bad))
    assert C.bootstrap_calibration([good, bad], n=400, seed=3) == b2    # fixed seed
    empty = C.reliability_counts(np.array([]), np.array([]))
    e = C.bootstrap_calibration([empty], n=10, seed=3)
    assert e["n_used"] == 0 and math.isnan(e["ece_ci95"][0])
    assert C.bootstrap_calibration([], n=10)["n_used"] == 0


def test_reliability_accumulates_over_chips():
    a = C.reliability_counts(np.array([0.2, 0.7]), np.array([0, 1]))
    b = C.reliability_counts(np.array([0.25]), np.array([1]))
    both = C.add_reliability(C.add_reliability(None, a), b)
    whole = C.reliability_counts(np.array([0.2, 0.7, 0.25]), np.array([0, 1, 1]))
    assert both["n"].tolist() == whole["n"].tolist()
    assert C.ece(both) == pytest.approx(C.ece(whole))
    assert C.brier(both) == pytest.approx(C.brier(whole))


def test_reliability_rejects_nan_or_out_of_range():
    with pytest.raises(ValueError):
        C.reliability_counts(np.array([np.nan]), np.array([0]))
    with pytest.raises(ValueError):
        C.reliability_counts(np.array([1.2]), np.array([1]))


def test_binary_entropy_values():
    h = C.binary_entropy(np.array([0.0, 0.5, 1.0, 0.1]))
    assert h[:3].tolist() == [0.0, 1.0, 0.0]
    assert h[3] == pytest.approx(0.468996, abs=1e-6)


def test_auroc_exact_perfect_reversed_ties():
    y = np.array([0, 0, 1, 1], bool)
    assert C.auroc(np.array([0.1, 0.2, 0.8, 0.9]), y) == 1.0
    assert C.auroc(np.array([0.9, 0.8, 0.2, 0.1]), y) == 0.0
    assert C.auroc(np.ones(4), y) == 0.5
    assert C.auroc(np.array([0.1, 0.5, 0.5, 0.9]), y) == pytest.approx(0.875)
    assert math.isnan(C.auroc(np.ones(3), np.zeros(3, bool)))


def test_auroc_from_hist_matches_exact_on_binned_scores():
    rng = np.random.default_rng(0)
    score = rng.integers(0, 50, 2000) / 50
    err = rng.random(2000) < 0.3 + 0.4 * score
    edges = (np.arange(52) - 0.5) / 50      # one bin per distinct score
    h_pos = np.histogram(score[err], bins=edges)[0]
    h_neg = np.histogram(score[~err], bins=edges)[0]
    assert C.auroc_from_hist(h_pos, h_neg) == pytest.approx(C.auroc(score, err), abs=1e-12)


def test_uncertainty_on_errors_means():
    u = np.array([0.9, 0.8, 0.1, 0.2])
    s = C.uncertainty_on_errors(u, np.array([1, 1, 0, 0], bool))
    assert s["mean_u_error"] == pytest.approx(0.85) and s["mean_u_correct"] == pytest.approx(0.15)
    assert s["auroc"] == 1.0 and s["n_error"] == 2


def test_jrc_confusions_split_valid_pixels():
    pred = np.array([[1, 1], [0, 0]])
    target = np.array([[1, 0], [1, 0]])
    valid = np.array([[True, True], [True, False]])
    jrc = np.array([[1, 0], [1, 1]], np.uint8)
    c = C.jrc_confusions(pred, target, valid, jrc)
    assert c["jrc_1"] == {"tp": 1, "fp": 0, "fn": 1, "tn": 0}
    assert c["jrc_0"] == {"tp": 0, "fp": 1, "fn": 0, "tn": 0}


def test_proxy_control_example_and_exclusions():
    # task.md control example: p = 0.4, V = 10 mln, q = 0.25 -> EL = 1 mln RUB
    pts = [
        {"asset_id": "A01", "V": 10_000_000, "q": 0.25, "p": 0.4, "el": 1_000_000.0, "y": 1},
        {"asset_id": "A02", "V": 10_000_000, "q": 0.25, "p": 0.4, "el": 1_000_000.0, "y": 0},
        {"asset_id": "A03", "V": 10_000_000, "q": 0.25, "p": 0.4, "el": 1_000_000.0, "y": -1},
        {"asset_id": "A04", "V": 10_000_000, "q": 0.25, "p": None, "el": None, "y": 1},
    ]
    rows = C.proxy_rows(pts)
    assert rows[0]["el_ref"] == 2_500_000 and rows[0]["abs_error"] == 1_500_000
    assert rows[1]["el_ref"] == 0 and rows[1]["abs_error"] == 1_000_000
    assert [r["reason"] for r in rows] == ["", "", "label_invalid", "no_data"]
    s = C.proxy_summary(rows)
    assert s["n_placed"] == 4 and s["n_points"] == 2
    assert s["n_excluded_label_invalid"] == 1 and s["n_excluded_no_data"] == 1
    assert s["sum_abs_error_rub"] == 2_500_000 and s["mae_rub"] == 1_250_000
    assert s["sum_el_rub"] == 2_000_000 and s["sum_el_ref_rub"] == 2_500_000


def test_proxy_summary_without_points_is_nan():
    assert math.isnan(C.proxy_summary([])["mae_rub"])


def test_lonlat_to_rowcol_inverts_pixel_centre():
    t = from_origin(81.6388930207821, 7.772942490429395, 8.983152841195857e-05,
                    8.983152841195857e-05)
    for row, col in [(0, 0), (511, 0), (17, 403), (511, 511)]:
        lon, lat = t @ (col + 0.5, row + 0.5)
        assert C.lonlat_to_rowcol(t, lon, lat) == (row, col)


def test_config_sha256_tracks_file_bytes_and_method(tmp_path):
    (tmp_path / "a.json").write_text("1", encoding="utf-8")
    h1 = C.config_sha256(tmp_path, ["a.json", "b.json"], "unet")
    assert h1 == C.config_sha256(tmp_path, ["b.json", "a.json"], "unet")
    assert h1 != C.config_sha256(tmp_path, ["a.json", "b.json"], "baseline")
    (tmp_path / "b.json").write_text("2", encoding="utf-8")          # a missing file that appears changes it
    h2 = C.config_sha256(tmp_path, ["a.json", "b.json"], "unet")
    assert h2 != h1
    (tmp_path / "a.json").write_text("3", encoding="utf-8")
    assert C.config_sha256(tmp_path, ["a.json", "b.json"], "unet") != h2


def test_test_confirmation_gate():
    good = "ab" * 32
    with pytest.raises(C.EvalGateError, match=good):   # tells the expected hash
        C.check_test_confirmation(None, good)
    with pytest.raises(C.EvalGateError):
        C.check_test_confirmation("", good)
    with pytest.raises(C.EvalGateError, match="does not match"):
        C.check_test_confirmation("cd" * 32, good)
    C.check_test_confirmation(" " + good.upper() + "\n", good)


def test_log_row_is_one_markdown_row():
    row = C.log_row("2026-09-26T17:40:00+00:00", "abc", "f" * 64, "baseline, unet", "m|x",
                    "IoU 0,6\nok")
    assert row.startswith("| 2026-09-26") and row.endswith(" |") and "\n" not in row
    assert row.count("|") == 7 and "`abc`" in row


def test_pixel_qc_counts_nan_invalid_positive_db_and_nan_under_label():
    """#64: per-chip pixel QC of eval-test; shares over all pixels, dB > 0 over finite ones."""
    nan = np.nan
    vv = np.array([[nan, 1.0, -5.0, 2.0],
                   [-3.0, -4.0, 0.0, -1.0]], np.float32)       # 0 dB is not > 0
    vh = np.array([[nan, -20.0, nan, 3.0],
                   [-25.0, -22.0, -21.0, -19.0]], np.float32)  # NaN only in VH at [0, 2]
    label = np.array([[1, 0, 0, -1],
                      [-1, 0, 1, 0]], np.int16)
    q = C.pixel_qc(np.stack([vv, vh]), label)
    assert q["n_pixels_total"] == 8
    assert q["s1_nan_share"] == pytest.approx(2 / 8)            # any band non-finite
    assert q["label_invalid_share"] == pytest.approx(2 / 8)
    assert q["vv_gt0_share"] == pytest.approx(2 / 7)            # 1.0, 2.0 of 7 finite VV
    assert q["vh_gt0_share"] == pytest.approx(1 / 6)            # 3.0 of 6 finite VH
    assert q["n_s1_nan_label_01"] == 2                          # [0,0] label 1, [0,2] label 0
    assert q["n_s1_nan"] == 2 and q["n_label_invalid"] == 2
    assert "water_share" not in q


def test_pixel_qc_all_nan_band_gives_nan_share_not_zero():
    s1 = np.full((2, 3, 3), np.nan, np.float32)
    label = np.full((3, 3), -1, np.int16)
    q = C.pixel_qc(s1, label)
    assert q["s1_nan_share"] == 1.0 and q["label_invalid_share"] == 1.0
    assert math.isnan(q["vv_gt0_share"]) and math.isnan(q["vh_gt0_share"])
    assert q["n_s1_nan_label_01"] == 0


def test_qc_summary_pools_pixels_per_event_and_all():
    rows = [{"event": "A", "n_pixels_total": 10, "n_s1_nan": 1, "n_label_invalid": 5,
             "n_vv_finite": 9, "n_vv_gt0": 3, "n_vh_finite": 9, "n_vh_gt0": 0,
             "n_s1_nan_label_01": 1},
            {"event": "A", "n_pixels_total": 30, "n_s1_nan": 0, "n_label_invalid": 0,
             "n_vv_finite": 30, "n_vv_gt0": 0, "n_vh_finite": 30, "n_vh_gt0": 0,
             "n_s1_nan_label_01": 0},
            {"event": "B", "n_pixels_total": 10, "n_s1_nan": 10, "n_label_invalid": 10,
             "n_vv_finite": 0, "n_vv_gt0": 0, "n_vh_finite": 0, "n_vh_gt0": 0,
             "n_s1_nan_label_01": 0}]
    s = C.qc_summary(rows)
    assert list(s) == ["A", "B", "ALL"]
    a = s["A"]
    assert a["n_chips"] == 2 and a["s1_nan_share"] == pytest.approx(1 / 40)
    assert a["label_invalid_share"] == pytest.approx(5 / 40)
    assert a["vv_gt0_share"] == pytest.approx(3 / 39) and a["vh_gt0_share"] == 0
    assert a["n_s1_nan_label_01"] == 1 and a["chips_s1_nan_label_01"] == 1
    assert a["chips_with_nan"] == 1
    assert math.isnan(s["B"]["vv_gt0_share"])
    assert s["ALL"]["n_chips"] == 3 and s["ALL"]["s1_nan_share"] == pytest.approx(11 / 50)
