import json

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from floodvalue.baseline import predict

CFG = {"channel": "VH", "threshold_db": -20.0, "scale_db": 2.0, "prob_threshold": 0.5}


def test_predict_proba_contract():
    s1 = np.full((2, 4, 4), -10, np.float32)
    s1[1, 0, :] = -30
    s1[0, 3, 3] = np.nan
    p = predict.predict_proba(s1, CFG)
    assert p.dtype == np.float32 and p.shape == (4, 4)
    assert np.isnan(p[3, 3]) and np.nanmin(p) >= 0 and np.nanmax(p) <= 1
    assert (p[0] > 0.5).all() and (p[1:3] < 0.5).all()


def test_rasters_mask_equals_p_ge_threshold_and_nodata():
    rng = np.random.default_rng(0)
    s1 = rng.normal(-20, 4, (2, 16, 16)).astype(np.float32)
    s1[1, 0, 0] = np.nan
    s1[1, 0, 1] = CFG["threshold_db"]  # exactly on the threshold: p = 0.5 -> water
    r = predict.to_rasters(s1, CFG)
    assert r["flood_probability"][0, 1] == 0.5 and r["flood_mask"][0, 1] == 1
    p, m = r["flood_probability"], r["flood_mask"]
    ok = p != predict.P_NODATA
    assert p[0, 0] == predict.P_NODATA and m[0, 0] == 255 and r["baseline_mask"][0, 0] == 255
    assert np.array_equal(m[ok] == 1, p[ok] >= CFG["prob_threshold"])
    assert set(np.unique(m)) <= {0, 1, 255}
    assert np.array_equal(r["baseline_mask"][ok] == 1, s1[1][ok] <= CFG["threshold_db"])


def test_predict_chip_writes_on_s1_grid(tmp_path, monkeypatch):
    raw = tmp_path / "data" / "raw" / "HandLabeled" / "S1Hand"
    raw.mkdir(parents=True)
    tr = from_origin(81.6, 7.7, 8.983152841195215e-05, 8.983152841195215e-05)
    s1 = np.random.default_rng(1).normal(-18, 5, (2, 8, 12)).astype(np.float32)
    s1[:, 2, 2] = np.nan
    with rasterio.open(raw / "X_1_S1Hand.tif", "w", driver="GTiff", count=2, dtype="float32",
                       width=12, height=8, crs="EPSG:4326", transform=tr, nodata=np.nan) as d:
        d.write(s1)
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps(CFG))
    monkeypatch.setenv("FLOODVALUE_DATA_DIR", str(tmp_path / "data"))
    out = predict.predict_chip("X_1", tmp_path / "out", cfg)
    for name, dtype, nodata in (("flood_probability", "float32", -9999),
                                ("flood_mask", "uint8", 255), ("baseline_mask", "uint8", 255)):
        with rasterio.open(out / f"{name}.tif") as r:
            assert r.crs.to_epsg() == 4326 and r.transform == tr
            assert (r.width, r.height, r.count) == (12, 8, 1)
            assert r.dtypes[0] == dtype and r.nodata == nodata
            assert r.read(1)[2, 2] == nodata
    assert json.loads((out / "predict.json").read_text())["threshold"] == 0.5


def test_run_without_chip_is_error():
    class A:
        chip = None
    assert predict.run(A()) == 2


def test_committed_config_is_complete():
    cfg = predict.load_config()
    for k in ("channel", "threshold_db", "scale_db", "prob_threshold", "commit", "created_at"):
        assert k in cfg
    assert cfg["channel"] in ("VV", "VH") and cfg["scale_db"] > 0
    assert cfg["fitted_on"] == "train"


if __name__ == "__main__":
    pytest.main([__file__])
