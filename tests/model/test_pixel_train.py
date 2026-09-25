import csv

import numpy as np
import pytest

from floodvalue.model import metrics
from floodvalue.model.pixel import train

BASE = {"channel": "VH", "threshold_db": -20.0}


def _chip(rng, shape=(24, 24), water_db=-26.0, share=0.3):
    label = (rng.random(shape) < share).astype(np.int16)
    s1 = rng.normal(-12, 1.5, (2,) + shape).astype(np.float32)
    s1[1][label == 1] = rng.normal(water_db, 1.5, int(label.sum()))
    label[0, :3] = -1
    label[rng.random(shape) < 0.05] = -1  # masks differ between chips (hash order matters)
    s1[:, 5, 5] = np.nan
    return s1, label


def _data():
    rng = np.random.default_rng(0)
    data = {f"Tr_{i}": _chip(rng) for i in range(4)}
    empty = np.full((2, 24, 24), np.nan, np.float32)
    data["Tr_9"] = (empty, np.zeros((24, 24), np.int16))  # like Paraguay_34417
    data.update({f"Dv_{i}": _chip(rng) for i in range(3)})
    data.update({f"Va_{i}": _chip(rng, water_db=-24.0) for i in range(2)})
    parts = {"train": [f"Tr_{i}" for i in (0, 1, 2, 3, 9)], "dev": ["Dv_1", "Dv_0", "Dv_2"],
             "validation": ["Va_0", "Va_1"]}
    return parts, data


CLIP = {"VV": [-40.0, 10.0], "VH": [-50.0, 0.0]}


def test_sample_skips_empty_chips_never_takes_invalid_and_is_deterministic():
    parts, data = _data()
    names = ["VV", "VH", "VH_mean3"]
    a = train.train_sample(parts["train"], data.__getitem__, names, CLIP, n_per_chip=100, seed=1)
    b = train.train_sample(parts["train"], data.__getitem__, names, CLIP, n_per_chip=100, seed=1)
    assert a["skipped"] == ["Tr_9"] and a["n_chips"] == 4
    assert len(a["y"]) == 400 and np.isfinite(a["x"]).all()
    np.testing.assert_array_equal(a["x"], b["x"])
    assert set(np.unique(a["y"])) <= {0, 1}
    full = train.train_sample(parts["train"], data.__getitem__, names, CLIP, n_per_chip=10**6)
    n_valid = sum(int((np.isfinite(data[c][0]).all(0) & (data[c][1] != -1)).sum())
                  for c in parts["train"])
    assert len(full["y"]) == n_valid  # capped by the valid pixels, label -1 excluded


def test_sample_pixels_uniform_without_replacement():
    valid = np.zeros((10, 10), bool)
    valid[2:5] = True
    idx = train.sample_pixels(valid, 7, np.random.default_rng(0))
    assert len(idx) == 7 and len(set(idx)) == 7 and valid.ravel()[idx].all()


def test_fit_clip_uses_percentiles_of_valid_train_pixels():
    parts, data = _data()
    clip = train.fit_clip(parts["train"], data.__getitem__, step=1)
    vh = np.concatenate([data[c][0][1][np.isfinite(data[c][0]).all(0) & (data[c][1] != -1)]
                         for c in parts["train"]])
    lo, hi = np.percentile(vh, train.CLIP_Q)
    assert clip["VH"] == [round(float(lo), 2), round(float(hi), 2)]


def test_eval_refuses_test():
    with pytest.raises(ValueError, match="test"):
        train.eval_chips({"test": ["X_1"]}, lambda c: None, ["VV"], CLIP, BASE)


@pytest.fixture(scope="module")
def fitted():
    parts, data = _data()
    names = train.SUPERSET
    sample = train.train_sample(parts["train"], data.__getitem__, names, CLIP, n_per_chip=300)
    chips = train.eval_chips({"dev": parts["dev"], "validation": parts["validation"]},
                             data.__getitem__, names, CLIP, BASE)
    return parts, data, sample, chips


def test_select_on_dev_returns_the_best_grid_point(fitted, monkeypatch):
    _, _, sample, chips = fitted
    dev = [c for c in chips if c.part == "dev"]
    grid = [{"feature_set": "tex3", "max_leaf_nodes": 7, "water_weight": 1.0, "max_iter": 1},
            {"feature_set": "tex3", "max_leaf_nodes": 15, "water_weight": 1.0, "max_iter": 30}]
    best, model, res = train.select_on_dev(sample, dev, grid, log=lambda *_: None)
    assert best["dev_iou"] == max(r["dev_iou"] for r in res)
    assert [r["max_iter"] for r in res] == [1, 30]
    assert res[1]["dev_iou"] > 0.9  # separable synthetic data


def test_rows_equal_direct_confusion_and_methods_share_mask_hash(fitted):
    parts, data, sample, chips = fitted
    cols = train.columns(train.FEATURE_SETS["tex3"])
    model = train.fit(sample["x"][:, cols], sample["y"],
                      {"max_leaf_nodes": 15, "water_weight": 1.0, "max_iter": 30})
    pix = train.per_chip_counts(model, chips, cols)
    base = train.baseline_counts(chips)
    rows = train.metric_rows(chips, ("dev", "validation"), pix, base, -20.0)
    assert [(r["split"], r["event"], r["method"]) for r in rows[:4]] == [
        ("dev", "Dv", "baseline"), ("dev", "Dv", "pixel"),
        ("dev", "ALL", "baseline"), ("dev", "ALL", "pixel")]
    for part in ("dev", "validation"):
        ids = sorted(parts[part])  # hash is over chips in chip_id order
        valid = [np.isfinite(data[c][0]).all(0) & (data[c][1] != -1) for c in ids]
        h = metrics.mask_sha256(valid)
        base_tot = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        for c, v in zip(ids, valid):
            s1, label = data[c]
            base_tot = metrics.add(base_tot, metrics.confusion(s1[1] <= -20.0, label == 1, v))
        got = [r for r in rows if r["split"] == part and r["event"] == "ALL"]
        assert {r["valid_mask_sha256"] for r in got} == {h}
        b = next(r for r in got if r["method"] == "baseline")
        assert {k: b[k] for k in base_tot} == base_tot
        assert b["n_pixels"] == int(sum(v.sum() for v in valid))
        p = next(r for r in got if r["method"] == "pixel")
        assert p["n_pixels"] == b["n_pixels"] and p["threshold"] == 0.5


def test_chip_pred_nodata_is_not_water(fitted):
    _, _, sample, chips = fitted
    cols = train.columns(train.FEATURE_SETS["tex3"])
    model = train.fit(sample["x"][:, cols], sample["y"], {"max_leaf_nodes": 7, "water_weight": 1.0})
    c = chips[0]
    pred = train.chip_pred(model, c, cols)
    assert not pred[5, 5] and pred.shape == c.label.shape


def test_water_weight_raises_recall():
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, (4000, 1))
    y = (x[:, 0] + rng.normal(0, 1.5, 4000) < -1.5).astype(int)
    lo = train.fit(x, y, {"max_leaf_nodes": 7, "water_weight": 1.0, "max_iter": 50})
    hi = train.fit(x, y, {"max_leaf_nodes": 7, "water_weight": 8.0, "max_iter": 50})
    assert (hi.predict_proba(x)[:, 1] >= 0.5).sum() > (lo.predict_proba(x)[:, 1] >= 0.5).sum()


def test_bootstrap_and_paired_delta():
    a = {f"c{i}": {"tp": 5 + i, "fp": 2, "fn": 1, "tn": 50} for i in range(6)}
    b = {k: {**v, "fp": v["fp"] + 3} for k, v in a.items()}
    ids = sorted(a)
    d0 = train.paired_delta(a, a, ids)
    assert d0["delta"] == 0 and d0["lo"] == 0 and d0["hi"] == 0
    d = train.paired_delta(a, b, ids)
    assert d["delta"] > 0 and d["lo"] > 0 and d["share_positive"] == 1.0
    lo, hi = train.bootstrap_ci(a, ids)
    assert lo <= train.iou(train.pooled(a, ids)) <= hi


def test_check_against_baseline_detects_mismatch(tmp_path):
    ref = tmp_path / "b.csv"
    row = {"method": "baseline", "split": "dev", "event": "ALL", "iou": 0.5, "f1": "",
           "precision": "", "recall": "", "tp": 1, "fp": 2, "fn": 3, "tn": 4, "n_pixels": 10,
           "threshold": -23.0, "valid_mask_sha256": "abc"}
    train.write_csv(ref, [row])
    assert train.check_against_baseline([row], ref) == []
    bad = {**row, "fp": 3}
    assert train.check_against_baseline([bad], ref) == ["dev/ALL fp: 3 != 2"]
    assert train.check_against_baseline([{**row, "valid_mask_sha256": "x"}], ref)
    with open(ref, newline="") as fh:
        assert next(csv.DictReader(fh))["tp"] == "1"


def test_threshold_curve_matches_direct_counts(fitted):
    _, data, sample, chips = fitted
    cols = train.columns(train.FEATURE_SETS["tex3"])
    model = train.fit(sample["x"][:, cols], sample["y"],
                      {"max_leaf_nodes": 7, "water_weight": 1.0, "max_iter": 5})
    curve = train.threshold_curve(model, chips, ("dev",), cols)
    r = next(x for x in curve if x["event"] == "ALL" and x["threshold"] == 0.5)
    direct = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for c in (c for c in chips if c.part == "dev"):
        direct = metrics.add(direct, metrics.confusion(train.chip_pred(model, c, cols),
                                                       c.label == 1, c.valid))
    assert {k: r[k] for k in direct} == direct


def test_water_backscatter_medians_use_only_water_pixels():
    parts, data = _data()
    got = train.water_backscatter(parts, data.__getitem__)
    assert set(got) == {"train", "dev", "validation:Va"}
    vh = np.concatenate([data[c][0][1][np.isfinite(data[c][0]).all(0) & (data[c][1] == 1)]
                         for c in parts["validation"]])
    assert got["validation:Va"][1] == pytest.approx(float(np.median(vh)))
    assert got["validation:Va"][1] < -20 < got["validation:Va"][0]


def test_reported_mask_equals_published_predict_proba(fitted):
    """Metrics use exactly the mask of the contract p-raster (float32 p >= 0.5)."""
    from floodvalue.model.pixel import predict

    parts, data, sample, chips = fitted
    names = train.FEATURE_SETS["tex3"]
    cols = train.columns(names)
    model = train.fit(sample["x"][:, cols], sample["y"],
                      {"max_leaf_nodes": 15, "water_weight": 1.0, "max_iter": 20})
    cfg = {"features": names, "clip_db": CLIP}
    for c in chips:
        p = predict.predict_proba(data[c.chip_id][0], model, cfg)
        np.testing.assert_array_equal(train.chip_pred(model, c, cols),
                                      np.nan_to_num(p, nan=-1.0) >= 0.5)
