from floodvalue.model import predict, train


def test_unknown_method_exits_2(monkeypatch, capsys):
    monkeypatch.setenv("FLOODVALUE_METHOD", "nope")
    args = type("Args", (), {"chip": "Sri-Lanka_713926", "out": None})
    assert train.run(args) == 2
    assert predict.run(args) == 2
    assert "unknown method" in capsys.readouterr().out


def test_default_method_is_unet():
    assert train.DEFAULT_METHOD == predict.DEFAULT_METHOD == "unet"
    assert "unet" in train.METHODS and "unet" in predict.METHODS


def test_predict_requires_chip(monkeypatch, capsys):
    monkeypatch.delenv("FLOODVALUE_METHOD", raising=False)
    args = type("Args", (), {"chip": None, "out": None})
    assert predict.run(args) == 2
    assert "--chip" in capsys.readouterr().out


def test_write_rasters_mask_is_p_ge_threshold_and_nodata(tmp_path):
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    p = np.array([[np.nan, 0.0, 0.49], [0.5, 0.55, 1.0]], dtype=np.float32)
    profile = {"crs": "EPSG:4326", "transform": from_origin(80, 7, 1e-4, 1e-4),
               "width": 3, "height": 2}
    predict.write_rasters(p, 0.5, profile, tmp_path)
    with rasterio.open(tmp_path / "flood_mask.tif") as src:
        assert src.read(1).tolist() == [[255, 0, 0], [1, 1, 1]]
    with rasterio.open(tmp_path / "flood_probability.tif") as src:
        prob = src.read(1)
    assert prob[0, 0] == -9999 and prob[1, 0] == np.float32(0.5)
