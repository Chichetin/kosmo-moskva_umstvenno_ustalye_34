import numpy as np
import pytest

from floodvalue.baseline import run
from floodvalue.data import io
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
            valid = io.valid_mask(s1, label)
            pred = s1[ch] <= params["threshold_db"]
            c = metrics.confusion(pred, label, valid)
            tot = c if tot is None else metrics.add(tot, c)
        row = next(r for r in res["rows"] if r["split"] == part and r["event"] == "ALL")
        assert {k: row[k] for k in tot} == tot
        h = metrics.mask_sha256([io.valid_mask(*data[c]) for c in sorted(ids)])
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


def test_read_chip_refuses_test_labels():
    from floodvalue.data.split import SealedTestError

    with pytest.raises(SealedTestError):
        run.read_chip("Spain_7370579")


def test_chip_without_finite_s1_has_no_otsu_and_zero_counts():
    s1 = np.full((2, 8, 8), np.nan, np.float32)
    label = np.ones((8, 8), np.int16)  # labels exist under S1 NaN, still not valid
    st = run.chip_stats("Va_9", "validation", s1, label)
    assert st.otsu_t is None and not st.valid.any()
    assert st.otsu_counts == {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    assert st.n_water == 0 and st.n_dry == 0


def test_otsu_summary_skips_chips_without_threshold():
    parts, data = _fixture()
    data["Va_9"] = (np.full((2, 32, 32), np.nan, np.float32), np.zeros((32, 32), np.int16))
    parts["validation"] = parts["validation"] + ["Va_9"]
    _, res = run.fit_and_evaluate(parts, data.__getitem__)
    _, res0 = run.fit_and_evaluate(_fixture()[0], _fixture()[1].__getitem__)
    assert res["otsu_t"]["validation"] == res0["otsu_t"]["validation"]
    row = next(r for r in res["variants"]
               if r["split"] == "validation" and r["event"] == "ALL"
               and r["method"] == "otsu_VH_per_chip")
    row0 = next(r for r in res0["variants"]
                if r["split"] == "validation" and r["event"] == "ALL"
                and r["method"] == "otsu_VH_per_chip")
    assert {k: row[k] for k in ("tp", "fp", "fn", "tn")} == \
        {k: row0[k] for k in ("tp", "fp", "fn", "tn")}
    assert row["threshold"] == ""


def test_run_writes_config_csvs_report_and_copies(tmp_path, monkeypatch):
    import csv
    import json

    from floodvalue.data import split

    parts, data = _fixture()
    monkeypatch.setattr(split, "chip_ids", lambda part: parts[part])
    monkeypatch.setattr(run, "read_chip", data.__getitem__)
    monkeypatch.setattr(run, "_split_sha", lambda: "s" * 64)
    monkeypatch.setattr(run, "CONFIG_PATH", tmp_path / "models" / "config.json")
    monkeypatch.setattr(run, "VERIFY_DIR", tmp_path / "verification")

    class Args:
        out = str(tmp_path / "out")

    assert run.run(Args()) == 0
    cfg = json.loads((tmp_path / "models" / "config.json").read_text(encoding="utf-8"))
    params, res = run.fit_and_evaluate(parts, data.__getitem__)
    for k in ("channel", "threshold_db", "scale_db"):
        assert cfg[k] == params[k]
    assert cfg["threshold"] == 0.5 and cfg["fitted_on"] == "train"
    assert cfg["split_sha256"] == "s" * 64 and cfg["commit"] and cfg["created_at"]
    assert len(cfg["weights_sha256"]) == 64

    def rows(p):
        with open(p, newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))

    ver, out = tmp_path / "verification", tmp_path / "out"
    m = rows(ver / "baseline-metrics.csv")
    assert list(m[0]) == metrics.METRIC_COLUMNS
    assert {r["split"] for r in m} == {"train", "dev", "validation"}  # never test
    assert all(r["method"] == "baseline" for r in m)
    assert all(r["threshold"] == str(params["threshold_db"]) for r in m)
    va = next(r for r in m if r["split"] == "validation" and r["event"] == "ALL")
    exp = next(r for r in res["rows"] if r["split"] == "validation" and r["event"] == "ALL")
    assert int(va["tp"]) == exp["tp"] and float(va["iou"]) == exp["iou"]
    for name, copy in (("baseline-metrics.csv", "metrics.csv"),
                       ("baseline-variants.csv", "variants.csv"),
                       ("baseline-threshold-curve.csv", "threshold_curve.csv")):
        assert (ver / name).read_bytes() == (out / copy).read_bytes()
    assert (ver / "baseline-threshold-curve.png").stat().st_size > 0
    report = (ver / "baseline-val.md").read_text(encoding="utf-8")
    assert "Test не читался" in report and "floodvalue.data.io.valid_mask" in report
