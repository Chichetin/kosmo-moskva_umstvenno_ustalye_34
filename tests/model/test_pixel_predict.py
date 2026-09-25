import json

import joblib
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from sklearn.ensemble import HistGradientBoostingClassifier

from floodvalue.model.pixel import features, predict

NAMES = ["VV", "VH", "VV-VH", "VH_mean3"]
CLIP = {"VV": [-40.0, 10.0], "VH": [-50.0, 0.0]}


def _toy_model():
    """Water is dark in VH: trained on synthetic pixels."""
    rng = np.random.default_rng(0)
    s1 = rng.normal(-12, 1.5, (2, 64, 64)).astype(np.float32)
    water = np.zeros((64, 64), bool)
    water[:, :24] = True
    s1[1][water] = rng.normal(-28, 1.5, int(water.sum()))
    f = features.compute(s1, NAMES, CLIP).reshape(len(NAMES), -1).T
    return HistGradientBoostingClassifier(max_iter=20, random_state=0).fit(f, water.ravel())


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("pixel")
    joblib.dump(_toy_model(), d / "model.joblib")
    cfg = {"method": "pixel_hgb", "features": NAMES, "clip_db": CLIP, "threshold": 0.5,
           "model_file": "model.joblib", "model_sha256": predict.sha256_file(d / "model.joblib")}
    (d / "config.json").write_text(json.dumps(cfg))
    return d


def test_predict_proba_contract(model_dir):
    cfg = predict.load_config(model_dir / "config.json")
    model = predict.load_model(cfg, model_dir)
    s1 = np.full((2, 16, 16), -12.0, np.float32)
    s1[1, :, :5] = -28.0
    s1[:, 3, 3] = np.nan
    s1[0, 10, 10] = np.nan  # one channel missing is no data too
    p = predict.predict_proba(s1, model, cfg)
    assert p.dtype == np.float32 and p.shape == (16, 16)
    assert np.isnan(p[3, 3]) and np.isnan(p[10, 10])
    assert np.isfinite(p).sum() == 16 * 16 - 2
    assert np.nanmin(p) >= 0 and np.nanmax(p) <= 1
    assert (p[:, 0][np.isfinite(p[:, 0])] > 0.5).all() and (p[:, 12:] < 0.5).all()


def test_all_nan_chip_gives_all_nan(model_dir):
    cfg = predict.load_config(model_dir / "config.json")
    p = predict.predict_proba(np.full((2, 4, 4), np.nan, np.float32),
                              predict.load_model(cfg, model_dir), cfg)
    assert np.isnan(p).all()


def test_rasters_mask_equals_p_ge_threshold():
    p = np.array([[0.1, 0.5, np.nan], [0.49999, 0.9, 1.0]], np.float32)
    r = predict.to_rasters(p, 0.5)
    assert r["flood_probability"][0, 2] == predict.P_NODATA and r["flood_mask"][0, 2] == 255
    np.testing.assert_array_equal(r["flood_mask"], [[0, 1, 255], [0, 1, 1]])


def test_load_model_refuses_wrong_sha(model_dir):
    cfg = {**predict.load_config(model_dir / "config.json"), "model_sha256": "0" * 64}
    with pytest.raises(ValueError, match="sha256"):
        predict.load_model(cfg, model_dir)


def test_predict_chip_writes_on_s1_grid(tmp_path, monkeypatch, model_dir):
    raw = tmp_path / "data" / "raw" / "HandLabeled" / "S1Hand"
    raw.mkdir(parents=True)
    tr = from_origin(81.6, 7.7, 8.983152841195215e-05, 8.983152841195215e-05)
    s1 = np.random.default_rng(1).normal(-18, 5, (2, 8, 12)).astype(np.float32)
    s1[:, 2, 2] = np.nan
    with rasterio.open(raw / "X_1_S1Hand.tif", "w", driver="GTiff", count=2, dtype="float32",
                       width=12, height=8, crs="EPSG:4326", transform=tr, nodata=np.nan) as d:
        d.write(s1)
    monkeypatch.setenv("FLOODVALUE_DATA_DIR", str(tmp_path / "data"))
    meta = predict.predict_chip("X_1", tmp_path / "out", model_dir / "config.json")
    for name, dtype, nodata in (("flood_probability", "float32", -9999),
                                ("flood_mask", "uint8", 255)):
        with rasterio.open(tmp_path / "out" / f"{name}.tif") as r:
            assert r.crs.to_epsg() == 4326 and r.transform == tr
            assert (r.width, r.height, r.count) == (12, 8, 1)
            assert r.dtypes[0] == dtype and r.nodata == nodata
            assert r.read(1)[2, 2] == nodata
    with rasterio.open(tmp_path / "out" / "flood_probability.tif") as a, \
            rasterio.open(tmp_path / "out" / "flood_mask.tif") as b:
        p, m = a.read(1), b.read(1)
        ok = p != -9999
        assert np.array_equal(m[ok] == 1, p[ok] >= 0.5)
    assert meta["threshold"] == 0.5 and meta["inputs"].startswith("S1 only")


def test_published_model_matches_its_config():
    """models/pixel/: weights present (not an LFS pointer), sha256 = config, features known."""
    if not predict.CONFIG_PATH.exists():
        pytest.skip("models/pixel/config.json not built yet")
    cfg = predict.load_config()
    path = predict.MODEL_DIR / cfg["model_file"]
    assert not path.read_bytes()[:40].startswith(b"version https://git-lfs"), "run git lfs pull"
    assert predict.sha256_file(path) == cfg["model_sha256"]
    assert path.stat().st_size < 100 * 2**20
    s1 = np.random.default_rng(2).normal(-15, 4, (2, 32, 32)).astype(np.float32)
    s1[:, 0, 0] = np.nan
    p = predict.predict_proba(s1)
    assert p.shape == (32, 32) and np.isnan(p[0, 0]) and np.isfinite(p).sum() == 32 * 32 - 1
    assert set(cfg["features"]) <= set(features.FULL)
