"""Synthetic Sen1Floods11-like raw/ tree (tiny 8x8 chips), generated at test time: no
Sen1Floods11 rasters are committed (data guard hook)."""

import base64
import hashlib
import json

import numpy as np
import pytest
import rasterio
from affine import Affine

H = W = 8
STEP_S1 = 8.983152841195215e-05
STEP_LABEL = 8.983152841195857e-05  # real ~1e-17 deg mismatch between S1 and labels
ORIGIN = (80.0, 7.8)

CHIPS = ["Sri-Lanka_713926", "Mekong_922373"]


def write_tif(path, arr, dtype, *, step=STEP_S1, origin=ORIGIN, crs="EPSG:4326",
              nodata=None, descriptions=None):
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = arr[None]
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", width=arr.shape[2], height=arr.shape[1],
                       count=arr.shape[0], dtype=dtype, crs=crs,
                       transform=Affine(step, 0.0, origin[0], 0.0, -step, origin[1]),
                       nodata=nodata) as dst:
        dst.write(arr.astype(dtype))
        if descriptions and len(descriptions) == arr.shape[0]:
            dst.descriptions = descriptions


def s1_array(seed=0):
    rng = np.random.default_rng(seed)
    a = np.stack([rng.uniform(-25, 5, (H, W)), rng.uniform(-30, -5, (H, W))]).astype(np.float32)
    a[:, 0, 0] = np.nan
    return a


def label_array():
    lab = np.zeros((H, W), np.int16)
    lab[:4] = 1
    lab[0, 1] = -1
    return lab


def md5_b64(path):
    return base64.b64encode(hashlib.md5(path.read_bytes()).digest()).decode()


def write_chip(raw, chip, s1=None, label=None, **label_kw):
    write_tif(raw / f"HandLabeled/S1Hand/{chip}_S1Hand.tif",
              s1_array() if s1 is None else s1, "float32", nodata=float("nan"),
              descriptions=("VV", "VH"))
    lab = label_array() if label is None else label
    write_tif(raw / f"HandLabeled/LabelHand/{chip}_LabelHand.tif", lab,
              label_kw.pop("dtype", "int16"), step=label_kw.pop("step", STEP_LABEL), **label_kw)
    write_tif(raw / f"HandLabeled/JRCWaterHand/{chip}_JRCWaterHand.tif",
              np.zeros((H, W)), "uint8", step=STEP_LABEL)


def write_bucket_manifest(raw):
    manifest = {}
    for p in sorted(raw.rglob("*.tif")):
        rel = p.relative_to(raw).as_posix()
        manifest[f"v1.1/data/flood_events/{rel}"] = {
            "size": p.stat().st_size, "md5Hash": md5_b64(p), "crc32c": None,
            "updated": "2020-10-01T17:17:33.484Z", "generation": "1601572653484826"}
    (raw / "bucket_manifest.json").write_text(json.dumps(manifest))


@pytest.fixture
def raw(tmp_path, monkeypatch):
    """$FLOODVALUE_DATA_DIR/raw with two chips, author lists and event metadata."""
    monkeypatch.setenv("FLOODVALUE_DATA_DIR", str(tmp_path))
    raw = tmp_path / "raw"
    for chip in CHIPS:
        write_chip(raw, chip)
    split = raw / "splits/flood_handlabeled"
    split.mkdir(parents=True)
    (split / "flood_train_data.csv").write_text(
        "Sri-Lanka_713926_S1Hand.tif,Sri-Lanka_713926_LabelHand.tif\n")
    (split / "flood_valid_data.csv").write_text(
        "Mekong_922373_S1Hand.tif,Mekong_922373_LabelHand.tif\n\n")
    (split / "flood_test_data.csv").write_text("")
    (split / "flood_bolivia_data.csv").write_text("")
    meta = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": None,
         "properties": {"location": "Sri-Lanka", "s1_date": "2017/05/30"}},
        {"type": "Feature", "geometry": None,
         "properties": {"location": "Cambodia", "s1_date": "2018/08/05"}}]}
    (raw / "Sen1Floods11_Metadata.geojson").write_text(json.dumps(meta))
    write_bucket_manifest(raw)
    return raw


@pytest.fixture
def offline(monkeypatch):
    """Any network access fails the test."""
    from floodvalue.data import fetch

    def boom(*a, **k):
        raise AssertionError("network access in offline test")

    monkeypatch.setattr(fetch, "list_objects", boom)
    monkeypatch.setattr(fetch, "download", boom)
