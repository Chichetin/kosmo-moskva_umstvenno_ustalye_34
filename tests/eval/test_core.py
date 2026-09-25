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
    c = C.reliability_counts(p, y)
    assert c["n"].tolist() == [1, 1, 0, 0, 0, 0, 0, 0, 0, 2]   # p = 1 in the last bin
    # (|0.05 − 0| + |0.15 − 1| + |1.95 − 2|) / 4
    assert C.ece(c) == pytest.approx(0.2375)
    assert C.brier(c) == pytest.approx((0.0025 + 0.7225 + 0.0025 + 0) / 4)
    rows = C.reliability_table(c)
    assert rows[9]["mean_p"] == pytest.approx(0.975) and rows[9]["observed"] == 1.0
    assert math.isnan(rows[5]["mean_p"])


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
