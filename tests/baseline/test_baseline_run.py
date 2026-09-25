import numpy as np
import pytest

from floodvalue.baseline import run
from floodvalue.model import metrics


def _chip(rng, t_water, channel, shape=(32, 32)):
    """Synthetic chip: water is dark in `channel` below t_water, the other channel is noise."""
    label = (rng.random(shape) < 0.3).astype(np.int16)
    s1 = rng.normal(-10, 1.0, (2,) + shape).astype(np.float32)
    s1[channel][label == 1] = rng.normal(t_water - 3, 0.5, int(label.sum()))
    s1[1 - channel][label == 1] = rng.normal(-10.5, 1.0, int(label.sum()))
    label[0, 0] = -1
    s1[:, 1, 1] = np.nan
    return s1, label


def _fixture(val_t=-25.0):
    rng = np.random.default_rng(0)
    data = {}
    for i in range(3):
        data[f"Tr_{i}"] = _chip(rng, -20.0, 1)
    data["Dv_0"] = _chip(rng, -20.0, 1)
    for i in range(2):
        data[f"Va_{i}"] = _chip(rng, val_t, 1)
    parts = {"train": ["Tr_0", "Tr_1", "Tr_2"], "dev": ["Dv_0"], "validation": ["Va_0", "Va_1"]}
    return parts, data


def test_threshold_is_fitted_on_train_only():
    parts, data = _fixture(val_t=-25.0)
    p1, _ = run.fit_and_evaluate(parts, data.__getitem__)
    parts2, data2 = _fixture(val_t=-5.0)  # very different validation
    p2, _ = run.fit_and_evaluate(parts2, data2.__getitem__)
    assert p1["threshold_db"] == p2["threshold_db"]
    assert p1["scale_db"] == p2["scale_db"]
    assert p1["channel"] == "VH" and p1["channel_chosen_on"] == "dev"
    assert -23.5 < p1["threshold_db"] < -10.5


def test_rows_match_direct_confusion_and_share_mask_hash():
    parts, data = _fixture()
    params, res = run.fit_and_evaluate(parts, data.__getitem__)
    ch = run.core.CHANNELS[params["channel"]]
    for part, ids in parts.items():
        tot = None
        for cid in ids:
            s1, label = data[cid]
            valid = run.core.valid_mask(s1, label)
            pred = s1[ch] <= params["threshold_db"]
            c = metrics.confusion(pred, label, valid)
            tot = c if tot is None else metrics.add(tot, c)
        row = next(r for r in res["rows"] if r["split"] == part and r["event"] == "ALL")
        assert {k: row[k] for k in tot} == tot
        h = metrics.mask_sha256([run.core.valid_mask(*data[c]) for c in sorted(ids)])
        assert row["valid_mask_sha256"] == h
    assert {r["method"] for r in res["rows"]} == {"baseline"}
    assert {r["event"] for r in res["rows"]} == {"Tr", "Dv", "Va", "ALL"}


def test_test_part_is_refused():
    with pytest.raises(ValueError):
        run.fit_and_evaluate({"test": ["X_1"]}, lambda c: None)


def test_channel_is_chosen_on_dev_not_train():
    rng = np.random.default_rng(3)
    data = {}
    for i in range(2):  # train: both channels identical -> exact tie on train
        s1, label = _chip(rng, -20.0, 1)
        s1[0] = s1[1]
        data[f"Tr_{i}"] = (s1, label)
    data["Dv_0"] = _chip(rng, -20.0, 1)  # dev: only VH separates water
    with_dev = {"train": ["Tr_0", "Tr_1"], "dev": ["Dv_0"]}
    p, _ = run.fit_and_evaluate(with_dev, data.__getitem__)
    assert p["channel"] == "VH" and p["per_channel"]["VH"]["dev_iou"] > 0.9
    p, _ = run.fit_and_evaluate({"train": ["Tr_0", "Tr_1"]}, data.__getitem__)
    assert p["channel"] == "VV" and p["channel_chosen_on"] == "train"
