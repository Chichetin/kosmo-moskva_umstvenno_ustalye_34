import json

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from floodvalue.baseline import predict

CFG = {"channel": "VH", "threshold_db": -20.0, "scale_db": 2.0, "threshold": 0.5}


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
    assert np.array_equal(m[ok] == 1, p[ok] >= CFG["threshold"])
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
    cfg.write_text(json.dumps(CFG), encoding="utf-8")
    monkeypatch.setenv("FLOODVALUE_DATA_DIR", str(tmp_path / "data"))
    out = predict.predict_chip("X_1", tmp_path / "out", cfg)
    for name, dtype, nodata in (("flood_probability", "float32", -9999),
                                ("flood_mask", "uint8", 255), ("baseline_mask", "uint8", 255)):
        with rasterio.open(out / f"{name}.tif") as r:
            assert r.crs.to_epsg() == 4326 and r.transform == tr
            assert (r.width, r.height, r.count) == (12, 8, 1)
            assert r.dtypes[0] == dtype and r.nodata == nodata
            assert r.read(1)[2, 2] == nodata
    assert json.loads((out / "prediction.json").read_text(encoding="utf-8"))["threshold"] == 0.5


def test_run_without_chip_is_error():
    class A:
        chip = None
    assert predict.run(A()) == 2


def test_committed_config_is_complete():
    cfg = predict.load_config()
    for k in ("channel", "threshold_db", "scale_db", "threshold", "threshold_status",
              "weights_sha256", "commit", "created_at"):
        assert k in cfg
    assert cfg["channel"] in ("VV", "VH") and cfg["scale_db"] > 0
    assert cfg["fitted_on"] == "train"



def test_predict_chip_reads_s1_through_shared_loader(tmp_path, monkeypatch):
    """S1 comes from data.io.load_s1 (D-015), not from a private reader."""
    from floodvalue.data import io

    tr = from_origin(81.6, 7.7, 8.983152841195215e-05, 8.983152841195215e-05)
    s1 = np.full((2, 4, 5), -30.0, np.float32)
    s1[0] = -10.0  # VV bright, VH dark: CFG thresholds VH, so all water only if bands kept
    seen = []

    def fake_load_s1(chip_id):
        seen.append(chip_id)
        return s1, {"crs": rasterio.crs.CRS.from_epsg(4326), "transform": tr,
                    "width": 5, "height": 4}

    monkeypatch.setattr(io, "load_s1", fake_load_s1)
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps(CFG), encoding="utf-8")
    out = predict.predict_chip("Sri-Lanka_1", tmp_path / "out", cfg)
    assert seen == ["Sri-Lanka_1"]
    with rasterio.open(out / "flood_mask.tif") as r:
        assert r.transform == tr and (r.read(1) == 1).all()
    assert json.loads((out / "prediction.json").read_text(encoding="utf-8"))["s1_file"] == \
        "Sri-Lanka_1_S1Hand.tif"


if __name__ == "__main__":
    pytest.main([__file__])


def test_mask_only_writes_baseline_mask_and_keeps_main_rasters(tmp_path, monkeypatch):
    """#115: `--mask-only` adds baseline_mask.tif next to the main method's files, touching nothing else."""
    raw = tmp_path / "data" / "raw" / "HandLabeled" / "S1Hand"
    raw.mkdir(parents=True)
    tr = from_origin(81.6, 7.7, 8.983152841195215e-05, 8.983152841195215e-05)
    s1 = np.random.default_rng(2).normal(-20, 5, (2, 6, 7)).astype(np.float32)
    s1[0, 1, 1] = np.nan
    with rasterio.open(raw / "X_2_S1Hand.tif", "w", driver="GTiff", count=2, dtype="float32",
                       width=7, height=6, crs="EPSG:4326", transform=tr, nodata=np.nan) as d:
        d.write(s1)
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps(CFG), encoding="utf-8")
    monkeypatch.setenv("FLOODVALUE_DATA_DIR", str(tmp_path / "data"))
    out = tmp_path / "out"
    out.mkdir()
    for name in ("flood_probability.tif", "flood_mask.tif", "prediction.json"):
        (out / name).write_bytes(b"main method")
    predict.predict_chip("X_2", out, cfg, mask_only=True)
    for name in ("flood_probability.tif", "flood_mask.tif", "prediction.json"):
        assert (out / name).read_bytes() == b"main method"
    with rasterio.open(out / "baseline_mask.tif") as r:
        assert r.transform == tr and (r.width, r.height) == (7, 6)
        assert r.dtypes[0] == "uint8" and r.nodata == 255
        m = r.read(1)
    assert m[1, 1] == 255
    ok = np.isfinite(s1).all(axis=0)
    assert np.array_equal(m[ok] == 1, s1[1][ok] <= CFG["threshold_db"])
    assert (m[ok] != 255).all()
