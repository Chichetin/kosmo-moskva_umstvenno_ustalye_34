import json
import sys
import types

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from floodvalue.calibration import calibrator as C
from floodvalue.calibration import run as R
from floodvalue.model import predict as P

H = W = 32


def fake_proba(s1):
    """p_raw = sigmoid(-(VH + 20)); NaN where S1 is NaN."""
    p = C.sigmoid(-(s1[1] + 20.0)).astype(np.float32)
    p[~np.isfinite(s1).all(0)] = np.nan
    return p


@pytest.fixture
def fake_method(tmp_path, monkeypatch):
    mod = types.ModuleType("fake_method_mod")
    mod.predict_proba = fake_proba
    monkeypatch.setitem(sys.modules, "fake_method_mod", mod)
    monkeypatch.setitem(P.METHODS, "fake", "fake_method_mod")
    monkeypatch.setitem(R.METHOD_MODULES, "fake", "fake_method_mod")
    models = tmp_path / "models"
    (models / "fake").mkdir(parents=True)
    (models / "fake" / "config.json").write_text(json.dumps(
        {"threshold": 0.5, "threshold_status": "raw", "weights_sha256": "w1"}), encoding="utf-8")
    monkeypatch.setattr(P, "MODELS_DIR", models)
    monkeypatch.setenv("FLOODVALUE_METHOD", "fake")
    # S1 chip on disk
    rng = np.random.default_rng(1)
    s1 = rng.normal(-20, 4, (2, H, W)).astype(np.float32)
    s1[:, :3, :5] = np.nan
    d = tmp_path / "data" / "raw" / "HandLabeled" / "S1Hand"
    d.mkdir(parents=True)
    with rasterio.open(d / "X_1_S1Hand.tif", "w", driver="GTiff", width=W, height=H, count=2,
                       dtype="float32", crs="EPSG:4326", nodata=np.nan,
                       transform=from_origin(80, 7, 9e-5, 9e-5)) as dst:
        dst.write(s1)
    monkeypatch.setenv("FLOODVALUE_DATA_DIR", str(tmp_path / "data"))
    return models / "fake", s1


def run_predict(tmp_path):
    out = tmp_path / "out"
    assert P.run(types.SimpleNamespace(chip="X_1", out=str(out))) == 0
    with rasterio.open(out / "flood_probability.tif") as a, rasterio.open(out / "flood_mask.tif") as b:
        return a.read(1), a.nodata, b.read(1), b.nodata, json.loads((out / "prediction.json").read_text(encoding="utf-8"))


def test_predict_without_calibration_is_raw(fake_method, tmp_path):
    _, s1 = fake_method
    p, _, m, _, meta = run_predict(tmp_path)
    ok = np.isfinite(s1).all(0)
    assert np.allclose(p[ok], fake_proba(s1)[ok])
    assert np.array_equal(m[ok] == 1, p[ok].astype(np.float64) >= 0.5)
    assert meta["threshold"] == 0.5 and meta["calibration"] is None


def test_predict_applies_calibrator_and_its_threshold(fake_method, tmp_path):
    model_dir, s1 = fake_method
    cal = {"kind": "platt", "a": 1.5, "b": 1.2}
    (model_dir / "calibration.json").write_text(json.dumps(
        {"calibrator": cal, "threshold": 0.37, "input_model_sha256": "w1",
         "threshold_status": "calibrated"}), encoding="utf-8")
    p, p_nd, m, m_nd, meta = run_predict(tmp_path)
    ok = np.isfinite(s1).all(0)
    assert np.allclose(p[ok], C.apply(cal, fake_proba(s1))[ok], atol=1e-6)
    assert np.all(p[~ok] == -9999) and p_nd == -9999
    assert np.all(m[~ok] == 255) and m_nd == 255
    assert np.array_equal(m[ok] == 1, p[ok].astype(np.float64) >= 0.37)
    assert meta["threshold"] == 0.37 and meta["calibration"]["kind"] == "platt"
    assert meta["threshold_status"] == "calibrated"


def test_predict_refuses_calibrator_of_other_weights(fake_method, tmp_path):
    model_dir, _ = fake_method
    (model_dir / "calibration.json").write_text(json.dumps(
        {"calibrator": {"kind": "identity"}, "threshold": 0.4, "input_model_sha256": "old"}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="make calibrate"):
        run_predict(tmp_path)


def synthetic_chips(n_per_event=6, seed=0):
    """Two events; raw p under-estimates water (true logit = raw logit + 1.5)."""
    rng = np.random.default_rng(seed)
    chips = []
    for ev in ("Nigeria", "Sri-Lanka"):
        for i in range(n_per_event):
            z = rng.normal(-2.5, 2.5, (H, W))
            y = rng.random((H, W)) < C.sigmoid(1.3 * z + 1.5)
            label = y.astype(np.int16)
            label[0, :4] = -1
            p = C.sigmoid(z).astype(np.float32)
            p[1, :4] = np.nan
            s1 = np.full((2, H, W), -20.0, dtype=np.float32)
            s1[1, 1, :4] = np.nan
            chips.append(R.Chip(f"{ev}_{i}", ev, p, label, np.zeros((H, W), bool),
                               R.io.valid_mask(s1, label)))
    return chips


def test_calibrate_picks_a_calibrator_and_threshold_on_synthetic_validation():
    chips = synthetic_chips()
    res = R.calibrate(chips, "fake", {"weights_sha256": "w1"})
    cal = res["calibration"]
    assert res["kind"] in ("platt", "isotonic")
    assert cal["input_model_sha256"] == "w1" and cal["fitted_on"]["part"] == "validation"
    assert 0 < cal["threshold"] < 1
    after = cal["metrics_validation"]["calibrated"]
    before = cal["metrics_validation"]["raw_0.5"]
    assert after["log_loss"] < before["log_loss"] and after["ece"] < before["ece"]
    assert after["iou"] >= before["iou"]
    # invalid pixels (label -1, NaN p) are out of every count
    n_valid = sum(int(c.valid.sum()) for c in chips)
    assert cal["fitted_on"]["n_pixels"] == n_valid
    assert {r["fit_event"] for r in res["cross"]} == {"Nigeria", "Sri-Lanka"}


def test_select_kind_minimises_mean_out_of_event_logloss():
    cross = [{"kind": "platt", "log_loss": 0.2}, {"kind": "platt", "log_loss": 0.1},
             {"kind": "isotonic", "log_loss": 0.12}, {"kind": "isotonic", "log_loss": 0.13},
             {"kind": "identity", "log_loss": 0.3}, {"kind": "identity", "log_loss": 0.1}]
    assert R.select_kind(cross) == "isotonic"


def test_load_chips_reads_only_validation(monkeypatch):
    asked = []

    def fake_chip_ids(part, **kw):
        asked.append((part, kw))
        return ["Nigeria_1"]

    monkeypatch.setattr(R, "chip_ids", fake_chip_ids)
    monkeypatch.setattr(R.io, "load_s1", lambda cid: (np.full((2, 4, 4), -20, np.float32), {}))
    monkeypatch.setattr(R.io, "load_label", lambda cid: (np.zeros((4, 4), np.int16), {}))
    mod = types.ModuleType("fake_method_mod2")
    mod.predict_proba = fake_proba
    monkeypatch.setitem(sys.modules, "fake_method_mod2", mod)
    monkeypatch.setitem(R.METHOD_MODULES, "fake2", "fake_method_mod2")
    chips = R.load_chips("fake2")
    assert asked == [("validation", {})]
    assert chips[0].event == "Nigeria"


def test_load_chips_uses_shared_s1_and_label_valid_mask(monkeypatch, tmp_path):
    s1 = np.full((2, 2, 2), -20.0, dtype=np.float32)
    s1[1, 0, 1] = np.nan  # one missing S1 band, while the label remains usable
    label = np.array([[0, 1], [-1, 1]], dtype=np.int16)
    p_raw = np.full((2, 2), 0.5, dtype=np.float32)

    mod = types.ModuleType("finite_probability_method")
    mod.predict_proba = lambda _s1: p_raw
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    monkeypatch.setitem(R.METHOD_MODULES, "finite-probability", mod.__name__)
    monkeypatch.setattr(R, "chip_ids", lambda part: ["Nigeria_1"])
    monkeypatch.setattr(R.io, "load_s1", lambda cid: (s1, {}))
    monkeypatch.setattr(R.io, "load_label", lambda cid: (label, {}))
    monkeypatch.setattr(R.io, "layer_path", lambda cid, layer: tmp_path / "absent.tif")

    chip = R.load_chips("finite-probability")[0]

    expected = np.array([[True, False], [False, True]])
    np.testing.assert_array_equal(chip.valid, expected)


def test_load_chips_rejects_nan_p_on_valid_pixels(monkeypatch, tmp_path):
    mod = types.ModuleType("nan_probability_method")
    mod.predict_proba = lambda _s1: np.full((2, 2), np.nan, np.float32)
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    monkeypatch.setitem(R.METHOD_MODULES, "nan-probability", mod.__name__)
    monkeypatch.setattr(R, "chip_ids", lambda part: ["Nigeria_1"])
    monkeypatch.setattr(R.io, "load_s1", lambda cid: (np.full((2, 2, 2), -20, np.float32), {}))
    monkeypatch.setattr(R.io, "load_label", lambda cid: (np.zeros((2, 2), np.int16), {}))
    monkeypatch.setattr(R.io, "layer_path", lambda cid, layer: tmp_path / "absent.tif")
    with pytest.raises(ValueError, match="non-finite"):
        R.load_chips("nan-probability")


def test_mean3x3_keeps_nan_and_ignores_it():
    p = np.array([[0.0, 1.0, np.nan], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], np.float32)
    q = R.mean3x3(p)
    assert np.isnan(q[0, 2])
    assert q[1, 1] == pytest.approx(7 / 8)


def test_shipped_calibration_matches_shipped_model():
    path = R.MODELS_DIR / "unet" / "calibration.json"
    if not path.exists():
        pytest.skip("no calibration shipped")
    config = json.loads((R.MODELS_DIR / "unet" / "config.json").read_text(encoding="utf-8"))
    cal = C.load(path, config["weights_sha256"])
    assert cal["fitted_on"]["part"] == "validation"
    assert set(cal["fitted_on"]["events"]) == {"Sri-Lanka", "Nigeria"}


def test_load_chips_refuses_test_and_train():
    with pytest.raises(ValueError):
        R.load_chips("unet", "test")
    with pytest.raises(ValueError):
        R.load_chips("unet", "train")


def test_metric_rows_follow_contract_and_hash_valid_masks():
    from floodvalue.model import metrics as M
    chips = synthetic_chips(n_per_event=2)
    rows = R.metric_rows({"validation": chips}, "fake", {"kind": "identity"}, 0.4)
    assert [r["event"] for r in rows] == ["ALL", "Nigeria", "Sri-Lanka"]
    assert list(rows[0]) == M.METRIC_COLUMNS
    all_row = rows[0]
    assert all_row["tp"] + all_row["fn"] == sum(int((c.label[c.valid] == 1).sum()) for c in chips)
    assert all_row["n_pixels"] == sum(int(c.valid.sum()) for c in chips)
    assert all_row["valid_mask_sha256"] == M.mask_sha256(
        [c.valid for c in sorted(chips, key=lambda c: c.chip_id)])
    assert sum(r["tp"] for r in rows[1:]) == all_row["tp"]


def test_shipped_calibrated_metrics_share_valid_pixels_with_baseline():
    import csv
    path = R.DOC_DIR / "unet-metrics.csv"
    base = R.REPO_ROOT / "docs" / "verification" / "baseline-metrics.csv"
    if not path.exists() or not base.exists():
        pytest.skip("no shipped metrics")
    cal = json.loads((R.MODELS_DIR / "unet" / "calibration.json").read_text(encoding="utf-8"))
    b = {(r["split"], r["event"]): r["valid_mask_sha256"]
         for r in csv.DictReader(base.read_text(encoding="utf-8").splitlines())}
    rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
    assert rows and {r["split"] for r in rows} == {"dev", "validation"}
    for r in rows:
        assert r["valid_mask_sha256"] == b[(r["split"], r["event"])]
        assert float(r["threshold"]) == cal["threshold"]
