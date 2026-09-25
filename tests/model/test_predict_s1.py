"""`predict --s1 <path>`: any S1 file (e.g. S1Weak), no label, rasters on the input grid (#87)."""

import json

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from floodvalue.cli import build_parser
from floodvalue.model import predict

H, W = 6, 5
THRESHOLD = 0.5


class HalfMethod:
    """p = 1 where VH < -20 dB else 0.2; NaN where any band is NaN."""

    @staticmethod
    def predict_proba(s1):
        p = np.where(s1[1] < -20, 1.0, 0.2).astype(np.float32)
        p[~np.isfinite(s1).all(axis=0)] = np.nan
        return p


@pytest.fixture
def s1_file(tmp_path):
    s1 = np.full((2, H, W), -12.0, np.float32)
    s1[1, :, :2] = -25.0            # water on the two left columns
    s1[0, 0, 0] = np.nan            # nodata pixel (only VV missing)
    s1[:, 5, 4] = np.nan            # nodata pixel (both bands)
    path = tmp_path / "in" / "India_1029501_S1Weak.tif"
    path.parent.mkdir()
    transform = from_origin(93.689, 26.538, 8.983e-05, 8.983e-05)
    with rasterio.open(path, "w", driver="GTiff", width=W, height=H, count=2, dtype="float32",
                       crs="EPSG:4326", nodata=float("nan"), transform=transform) as dst:
        dst.write(s1)
    return path


@pytest.fixture
def fake_method(tmp_path, monkeypatch):
    models = tmp_path / "models"
    (models / "half").mkdir(parents=True)
    (models / "half" / "config.json").write_text(json.dumps(
        {"threshold": THRESHOLD, "threshold_status": "fixed", "weights_sha256": "abc"}))
    monkeypatch.setattr(predict, "MODELS_DIR", models)
    monkeypatch.setitem(predict.METHODS, "half", __name__)
    monkeypatch.setenv("FLOODVALUE_METHOD", "half")
    # no labels anywhere: an empty data dir proves that --s1 reads nothing but the file
    monkeypatch.setenv("FLOODVALUE_DATA_DIR", str(tmp_path / "empty_data"))
    monkeypatch.setenv("FLOODVALUE_OUTPUT_DIR", str(tmp_path / "outputs"))


predict_proba = HalfMethod.predict_proba


def args(**kw):
    base = {"chip": None, "s1": None, "out": None}
    base.update(kw)
    return type("Args", (), base)


def test_cli_accepts_s1_flag():
    ns = build_parser().parse_args(["predict", "--s1", "x_S1Weak.tif"])
    assert ns.s1 == "x_S1Weak.tif" and ns.chip is None


def test_chip_id_from_s1_file_name():
    assert predict.chip_id_from_s1_path("a/India_1029501_S1Weak.tif") == "India_1029501"
    assert predict.chip_id_from_s1_path("Sri-Lanka_713926_S1Hand.tif") == "Sri-Lanka_713926"
    assert predict.chip_id_from_s1_path("/x/my_scene.tif") == "my_scene"


def test_s1_writes_rasters_on_input_grid_without_label(s1_file, fake_method, tmp_path):
    assert predict.run(args(s1=str(s1_file))) == 0
    out = tmp_path / "outputs" / "India_1029501"
    with rasterio.open(s1_file) as src:
        grid = (src.crs, src.transform, src.width, src.height)
    with rasterio.open(out / "flood_probability.tif") as src:
        assert (src.crs, src.transform, src.width, src.height) == grid
        assert src.dtypes[0] == "float32" and src.nodata == -9999
        p = src.read(1)
    with rasterio.open(out / "flood_mask.tif") as src:
        assert (src.crs, src.transform, src.width, src.height) == grid
        assert src.dtypes[0] == "uint8" and src.nodata == 255
        m = src.read(1)
    assert p[0, 0] == -9999 and p[5, 4] == -9999
    assert m[0, 0] == 255 and m[5, 4] == 255
    assert m[1, 0] == 1 and m[1, 3] == 0
    assert p[1, 0] == np.float32(1.0) and p[1, 3] == np.float32(0.2)
    valid = p != -9999
    assert np.array_equal(m[valid], (p[valid] >= THRESHOLD).astype(np.uint8))
    for extra in ("uncertainty.tif", "flood_probability_range.tif"):
        with rasterio.open(out / extra) as src:
            assert (src.crs, src.transform, src.width, src.height) == grid
            assert src.read(1)[0, 0] == -9999
    meta = json.loads((out / "prediction.json").read_text(encoding="utf-8"))
    assert meta["label_status"] == "no_label"
    assert meta["chip_id"] == "India_1029501"
    assert meta["s1_file"] == "India_1029501_S1Weak.tif"
    assert meta["threshold"] == THRESHOLD
    assert meta["valid_pixels"] == H * W - 2
    assert "iou" not in json.dumps(meta).lower()


def test_s1_respects_out(s1_file, fake_method, tmp_path):
    assert predict.run(args(s1=str(s1_file), out=str(tmp_path / "o"))) == 0
    assert (tmp_path / "o" / "flood_probability.tif").exists()


def test_s1_missing_file_exits_2(fake_method, tmp_path, capsys):
    assert predict.run(args(s1=str(tmp_path / "nope_S1Weak.tif"))) == 2
    assert "not found" in capsys.readouterr().out


def test_s1_and_chip_together_exit_2(s1_file, fake_method, capsys):
    assert predict.run(args(s1=str(s1_file), chip="Sri-Lanka_713926")) == 2
    assert "either" in capsys.readouterr().out


def test_s1_wrong_band_count_exits_2(fake_method, tmp_path, capsys):
    path = tmp_path / "one_band_S1Weak.tif"
    with rasterio.open(path, "w", driver="GTiff", width=W, height=H, count=1, dtype="float32",
                       crs="EPSG:4326", transform=from_origin(0, 0, 1e-4, 1e-4)) as dst:
        dst.write(np.zeros((1, H, W), np.float32))
    assert predict.run(args(s1=str(path))) == 2
    assert "2 bands" in capsys.readouterr().out


def test_neither_chip_nor_s1_exits_2(fake_method, capsys):
    assert predict.run(args()) == 2
    out = capsys.readouterr().out
    assert "--chip" in out and "--s1" in out
