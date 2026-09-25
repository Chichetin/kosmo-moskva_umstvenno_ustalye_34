"""LOEO wiring (#86): folds by event, no held-out chip in training/normalization/threshold,
only train+dev split parts are requested, baseline on the same valid pixels."""

import zlib

import numpy as np
import pytest
import torch

from floodvalue.model import metrics as M
from floodvalue.model.unet import loeo

CFG = {"channels": ["vv", "vh", "vv_minus_vh"], "base": 4, "dropout": 0.0, "batch": 2,
       "lr": 1e-3, "weight_decay": 0.0, "dice_weight": 0.5, "db_jitter": 1.0, "epochs": 2,
       "crop": 16, "seed": 0}
IDS = ["Ghana_1", "Ghana_2", "India_1", "India_2", "USA_1", "USA_2"]


def synthetic_chip(chip_id: str, h: int = 32, w: int = 32):
    rng = np.random.default_rng(zlib.crc32(chip_id.encode()))
    lab = (rng.random((h, w)) < 0.3).astype(np.int16)
    vh = np.where(lab == 1, -26.0, -16.0) + rng.normal(0, 1, (h, w))
    vv = vh + 7.0
    s1 = np.stack([vv, vh]).astype(np.float32)
    s1[:, 0, :4] = np.nan          # no S1 (label under it stays 0/1)
    lab[1, :3] = -1                # unlabeled
    return s1, lab


def test_folds_hold_out_each_event_once_and_do_not_overlap():
    fs = loeo.folds(IDS)
    assert [f["event"] for f in fs] == ["Ghana", "India", "USA"]
    for f in fs:
        assert set(f["train"]).isdisjoint(f["held_out"])
        assert sorted(f["train"] + f["held_out"]) == sorted(IDS)
        assert all(c.startswith(f["event"] + "_") for c in f["held_out"])
        assert not any(c.startswith(f["event"] + "_") for c in f["train"])


def test_pool_requests_only_train_and_dev(monkeypatch):
    asked = []

    def fake(part):
        asked.append(part)
        if part not in ("train", "dev"):
            raise AssertionError(f"LOEO must not request {part}")
        return {"train": ["Ghana_1", "India_1"], "dev": ["USA_1"]}[part]

    monkeypatch.setattr(loeo.D, "split_chip_ids", fake)
    assert loeo.pool_chip_ids() == ["Ghana_1", "India_1", "USA_1"]
    assert asked == ["train", "dev"]


def test_real_pool_is_six_train_dev_events_without_validation_or_test():
    ids = loeo.pool_chip_ids()
    events = {f["event"] for f in loeo.folds(ids)}
    assert events == {"Ghana", "India", "Pakistan", "Paraguay", "Somalia", "USA"}


def test_fit_vh_threshold_finds_separating_threshold_on_valid_pixels():
    s1 = np.full((2, 4, 4), -10.0, np.float32)
    lab = np.zeros((4, 4), np.int16)
    s1[1, :2] = -25.0
    lab[:2] = 1
    s1[1, 3, 3] = np.nan           # invalid S1: ignored
    lab[3, 2] = -1                 # unlabeled: ignored
    s1[1, 3, 2] = -30.0            # would be "water" if the -1 pixel counted
    t, iou = loeo.fit_vh_threshold([(s1, lab)])
    assert t == -25.0 and iou == 1.0
    # lowest threshold on ties, a bit above changes nothing, -10 dB swallows the dry pixels
    pred = loeo.baseline_pred(s1, t)
    assert pred[:2].all() and not pred[2:3].any()


def test_run_fold_never_reads_held_out_before_training_and_uses_same_pixels():
    fold = loeo.folds(IDS)[0]  # Ghana held out
    reads = []

    def read(cid):
        reads.append(cid)
        return synthetic_chip(cid)

    trained_on = {}
    real_train = loeo.train_fold

    def spy_train(x, y, m, cfg, eval_epochs, device, log=print):
        trained_on["n"] = len(x)
        trained_on["reads_before"] = list(reads)
        return real_train(x, y, m, cfg, eval_epochs, device, log=lambda *_: None)

    loeo.train_fold = spy_train
    try:
        res = loeo.run_fold(fold, CFG, read=read, baseline_t=-20.0, eval_epochs=(1, 2),
                            log=lambda *_: None)
    finally:
        loeo.train_fold = real_train
    assert trained_on["n"] == len(fold["train"])
    assert set(trained_on["reads_before"]) == set(fold["train"])
    assert [r["chip_id"] for r in res["rows"]] == fold["held_out"]
    assert res["methods"] == ["unet_e1", "unet_e2", "baseline_fold", "baseline_shipped"]
    rows = loeo.fold_metrics(res, res["methods"])
    # one valid-pixel set for all methods: equal n_pixels and hash
    assert len({r["n_pixels"] for r in rows}) == 1
    assert len({r["valid_mask_sha256"] for r in rows}) == 1
    n_valid = sum(int((np.isfinite(synthetic_chip(c)[0]).all(0)
                       & (synthetic_chip(c)[1] != -1)).sum()) for c in fold["held_out"])
    assert rows[0]["n_pixels"] == n_valid
    # well separated synthetic data: fold-refitted baseline is perfect
    base = next(r for r in rows if r["method"] == "baseline_fold")
    assert base["iou"] == pytest.approx(1.0)
    assert res["baseline_fold_t"] < -16.0
    assert res["ranking"]["baseline"]["ap"] == pytest.approx(1.0)
    assert -26.0 < -res["ranking"]["baseline"]["oracle_score_gt"] < -16.0
    assert 0.0 <= res["ranking"]["unet_e2"]["ap"] <= 1.0
    # paired delta against the fold baseline for every other method
    assert all(r["delta_iou_vs"] == "baseline_fold" for r in rows if r["method"] != "baseline_fold")
    assert rows[0]["delta_iou"] == pytest.approx(rows[0]["iou"] - base["iou"])
    assert base["delta_iou_vs"] == "" and np.isnan(base["delta_iou"])


def test_normalization_uses_fold_training_chips_only():
    fold = loeo.folds(IDS)[2]  # USA held out
    mean, _ = loeo.fold_stats(fold["train"], synthetic_chip, CFG["channels"])
    mean_all, _ = loeo.fold_stats(IDS, synthetic_chip, CFG["channels"])
    s1s = [synthetic_chip(c)[0] for c in fold["train"]]
    vh = np.concatenate([s[1][np.isfinite(s).all(0)] for s in s1s])
    assert mean[1] == pytest.approx(float(np.clip(vh, -45, 0).mean()), abs=1e-4)
    assert mean != mean_all


def test_fold_metrics_pool_counts_over_chips():
    res = {"event": "X", "valid_mask_sha256": "h", "rows": [
        {"unet_tp": 1, "unet_fp": 1, "unet_fn": 0, "unet_tn": 2},
        {"unet_tp": 1, "unet_fp": 0, "unet_fn": 2, "unet_tn": 5}]}
    (r,) = loeo.fold_metrics(res, ["unet"])
    assert (r["tp"], r["fp"], r["fn"], r["tn"]) == (2, 1, 2, 7)
    assert r["iou"] == pytest.approx(2 / 5)
    assert r["f1"] == pytest.approx(M.scores({"tp": 2, "fp": 1, "fn": 2, "tn": 7})["f1"])


def test_shipped_baseline_rule_is_vh():
    rule = loeo.shipped_baseline_rule()
    assert rule["threshold_db"] < 0


def test_ranking_metrics_perfect_and_uninformative_scores():
    edges = loeo.P_EDGES
    target = np.array([1, 1, 0, 0, 0, 0], bool)
    valid = np.ones(6, bool)
    perfect = np.array([0.9, 0.8, 0.1, 0.2, 0.3, 0.05])
    r = loeo.ranking_metrics(*loeo.score_hist(perfect, target, valid, edges), edges)
    assert r["ap"] == pytest.approx(1.0) and r["oracle_iou"] == pytest.approx(1.0)
    assert 0.3 <= r["oracle_score_gt"] < 0.8
    flat = np.full(6, 0.5)
    r = loeo.ranking_metrics(*loeo.score_hist(flat, target, valid, edges), edges)
    assert r["ap"] == pytest.approx(2 / 6) and r["oracle_iou"] == pytest.approx(2 / 6)
    # invalid pixels are not counted
    valid[0] = False
    w, d = loeo.score_hist(perfect, target, valid, edges)
    assert w.sum() == 1 and d.sum() == 4


def test_ranking_metrics_inverted_scores_have_low_ap():
    edges = loeo.P_EDGES
    target = np.array([1, 0, 0, 0], bool)
    r = loeo.ranking_metrics(*loeo.score_hist(np.array([0.1, 0.9, 0.8, 0.7]), target,
                                              np.ones(4, bool), edges), edges)
    assert r["ap"] == pytest.approx(0.25)


def test_train_fold_keeps_states_only_at_fixed_epochs_and_rejects_out_of_range():
    x = torch.zeros(2, 4, 16, 16)
    x[:, 3] = 1
    y = torch.zeros(2, 16, 16, dtype=torch.uint8)
    m = torch.ones(2, 16, 16, dtype=torch.uint8)
    cfg = CFG | {"std": [1.0, 1.0, 1.0]}
    _, states, steps = loeo.train_fold(x, y, m, cfg, (1,), "cpu", log=lambda *_: None)
    assert list(states) == [1] and steps == 2
    with pytest.raises(ValueError, match="exceed"):
        loeo.train_fold(x, y, m, cfg, (3,), "cpu", log=lambda *_: None)


def test_eval_epochs_default_is_fixed_in_advance_from_d019_histories():
    import csv

    from floodvalue.paths import MODELS_DIR

    best = []
    for f in sorted((MODELS_DIR / "unet").glob("history_member_*.csv")):
        with open(f, encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        best.append(int(max(rows, key=lambda r: float(r["dev_iou"]))["epoch"]))
    assert len(best) == 5
    assert loeo.EVAL_EPOCHS == (int(np.median(best)), loeo.DEFAULTS["epochs"])


def test_limit_per_event_and_markdown_table():
    assert loeo.limit_per_event(IDS, 1) == ["Ghana_1", "India_1", "USA_1"]
    assert loeo.limit_per_event(IDS, 0) == IDS
    row = {"event": "USA", "method": "unet_e19", "iou": "0.5", "iou_ci95_lo": "0.4",
           "iou_ci95_hi": "0.6", "f1": "0.66", "precision": "0.7", "recall": "0.6",
           "delta_iou_vs": "baseline_fold", "delta_iou": "0.1", "delta_ci95_lo": "",
           "delta_ci95_hi": "0.2", "n_pixels": "10"}
    table = loeo.markdown_table([row])
    assert "| USA | unet_e19 | 0,500 [0,400; 0,600] |" in table
    assert "0,100 [—; 0,200]" in table


def test_cli_parses_colab_command():
    a = loeo.parse_args(["--out-dir", "o", "--events", "USA", "--max-chips", "4",
                         "--epochs", "2", "--eval-epochs", "1,2"])
    assert (a.epochs, a.eval_epochs, a.max_chips) == (2, "1,2", 4)
    d = loeo.parse_args([])
    assert d.epochs == 60 and d.eval_epochs == "19,60"


def test_held_out_features_use_the_fold_training_normalization(monkeypatch):
    fold = loeo.folds(IDS)[1]  # India held out
    calls = []
    real_build = loeo.build_tensors
    real_stats = loeo.fold_stats
    stats_ids = []

    def spy_build(ids, read, channels, mean, std):
        calls.append((list(ids), list(mean), list(std)))
        return real_build(ids, read, channels, mean, std)

    def spy_stats(ids, read, channels):
        stats_ids.append(list(ids))
        return real_stats(ids, read, channels)

    monkeypatch.setattr(loeo, "build_tensors", spy_build)
    monkeypatch.setattr(loeo, "fold_stats", spy_stats)
    loeo.run_fold(fold, CFG, read=synthetic_chip, eval_epochs=(1,), log=lambda *_: None)
    assert stats_ids == [fold["train"]]
    (train_ids, m_tr, s_tr), (held_ids, m_h, s_h) = calls
    assert train_ids == fold["train"] and held_ids == fold["held_out"]
    assert (m_h, s_h) == (m_tr, s_tr)
