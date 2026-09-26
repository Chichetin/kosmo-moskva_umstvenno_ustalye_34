"""Offline S2 context layer (D-042): rendering, caption, grid check, provenance."""

import json

import numpy as np
import pytest
import rasterio
from affine import Affine
from PIL import Image

from floodvalue.context import s2 as ctx

H = W = 16
TRANSFORM = Affine(0.0001, 0, 81.6, 0, -0.0001, 7.8)


def _s2(value=1500):
    a = np.full((13, H, W), value, dtype=np.int16)
    a[1] = 1200  # B2 blue
    a[2] = 1400  # B3 green
    a[3] = 1800  # B4 red
    a[10] = 5  # B10 cirrus: clear sky (0.0005 reflectance)
    return a


def _write(path, arr, transform=TRANSFORM, dtype=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[1], width=arr.shape[2],
                       count=arr.shape[0], dtype=dtype or arr.dtype, crs="EPSG:4326",
                       transform=transform) as f:
        f.write(arr)


def test_true_color_uses_b4_b3_b2_and_masks_nodata():
    s2 = _s2()
    s2[:, :2, :] = 0  # nodata rows
    s2[3, 5, 5] = 4000  # strong red in one pixel
    rgba = ctx.true_color(s2)
    assert rgba.shape == (H, W, 4) and rgba.dtype == np.uint8
    assert (rgba[:2, :, 3] == 0).all() and (rgba[2:, :, 3] == 255).all()
    r, g, b = (int(v) for v in rgba[5, 5, :3])
    assert r > g and r > b  # channel order is R=B4, G=B3, B=B2, not the file order


def test_true_color_all_nodata_is_transparent():
    assert (ctx.true_color(np.zeros((13, H, W), np.int16))[..., 3] == 0).all()


def test_rough_cloud_fraction():
    s2 = _s2()
    assert ctx.rough_cloud_fraction(s2) == 0.0
    s2[1, :4, :] = 3500  # bright blue -> cloud-like, 4 of 16 rows
    assert ctx.rough_cloud_fraction(s2) == pytest.approx(0.25)
    s2 = _s2()
    s2[10, :, :8] = 200  # cirrus B10 > 0.01 in half of the chip
    assert ctx.rough_cloud_fraction(s2) == pytest.approx(0.5)
    assert ctx.rough_cloud_fraction(np.zeros((13, H, W), np.int16)) is None


def test_event_dates_and_mekong_alias():
    meta = {"features": [{"properties": {"location": "Sri-Lanka", "s1_date": "2017/05/30", "s2_date": "2017/05/28"}},
                         {"properties": {"location": "Cambodia", "s1_date": "2018/08/05", "s2_date": "2018/08/04"}}]}
    assert ctx.event_dates("Sri-Lanka", meta) == {"s1_date": "2017-05-30", "s2_date": "2017-05-28"}
    assert ctx.event_dates("Mekong", meta)["s1_date"] == "2018-08-05"
    assert ctx.event_dates("Nowhere", meta) == {"s1_date": None, "s2_date": None}


@pytest.mark.parametrize("s2_date,when,available,after,phrase", [
    ("2017-05-28", "за 2 дн. до", True, False, "другое состояние"),
    ("2017-05-30", "в день", None, False, "не сверена"),
    ("2017-05-31", "через 1 дн. после", False, True, "недоступен, показан ретроспективно"),
])
def test_caption_and_availability_follow_the_date_gap(s2_date, when, available, after, phrase):
    m = ctx.context_metadata("Sri-Lanka_1", (81.6, 7.7, 81.7, 7.8), "EPSG:4326",
                             {"s1_date": "2017-05-30", "s2_date": s2_date}, 0.0)
    o = m["offline"]
    assert when in o["caption"] and phrase in o["caption"] and "не вход модели" in o["caption"]
    assert o["available_at_decision"] is available and o["after_event"] is after
    assert m["role"] == "context_only" and m["used_by_model"] is False and m["used_by_economics"] is False
    assert o["bounds_lonlat"] == [81.6, 7.7, 81.7, 7.8]  # [west, south, east, north]


def test_unknown_date_is_not_claimed_available():
    o = ctx.context_metadata("X_1", (0, 0, 1, 1), "EPSG:4326", {"s1_date": None, "s2_date": None}, None)["offline"]
    assert o["available_at_decision"] is None and o["after_event"] is None


def test_online_basemaps_carry_attribution_and_after_event_flag():
    m = ctx.context_metadata("X_1", (0, 0, 1, 1), "EPSG:4326", {"s1_date": None, "s2_date": None}, None)
    by_id = {b["id"]: b for b in m["online"]}
    assert all(b["attribution"] and b["offline"] is False and "after_event" in b for b in m["online"])
    assert by_id["esri_world_imagery"]["after_event"] is True and by_id["osm"]["after_event"] is True


def test_test_chips_are_sealed_before_eval_test(tmp_path, monkeypatch):
    monkeypatch.setenv("FLOODVALUE_DATA_DIR", str(tmp_path / "data"))
    with pytest.raises(ctx.SealedTestError):
        ctx.build("Spain_1", tmp_path / "run", download=False)
    with pytest.raises(FileNotFoundError):  # allowed explicitly -> goes on to read the file
        ctx.build("Spain_1", tmp_path / "run", download=False, allow_test=True)


def test_build_writes_png_and_json(tmp_path, monkeypatch):
    raw = tmp_path / "data" / "raw"
    _write(raw / "HandLabeled/S2Hand/Sri-Lanka_1_S2Hand.tif", _s2())
    _write(raw / "HandLabeled/S1Hand/Sri-Lanka_1_S1Hand.tif", np.zeros((2, H, W), np.float32))
    (raw / ctx.METADATA).write_text(json.dumps({"features": [{"properties": {
        "location": "Sri-Lanka", "s1_date": "2017/05/30", "s2_date": "2017/05/28"}}]}),
            encoding="utf-8")
    monkeypatch.setenv("FLOODVALUE_DATA_DIR", str(tmp_path / "data"))
    out = tmp_path / "run"
    meta = ctx.build("Sri-Lanka_1", out, download=False)
    img = Image.open(out / "context_s2_rgb.png")
    assert img.size == (W, H) and img.mode == "RGBA"
    saved = json.loads((out / "context_s2.json").read_text(encoding="utf-8"))
    assert saved == meta and saved["offline"]["days_from_s1"] == -2
    w, s, e, n = saved["offline"]["bounds_lonlat"]
    assert (w, n) == pytest.approx((81.6, 7.8)) and e > w and n > s


def test_build_refuses_misaligned_grid(tmp_path, monkeypatch):
    raw = tmp_path / "data" / "raw"
    _write(raw / "HandLabeled/S2Hand/Sri-Lanka_1_S2Hand.tif", _s2())
    _write(raw / "HandLabeled/S1Hand/Sri-Lanka_1_S1Hand.tif", np.zeros((2, H, W), np.float32),
           transform=Affine(0.0001, 0, 81.6 + 1e-7, 0, -0.0001, 7.8))  # 0.1 µdeg: tiny but wrong
    monkeypatch.setenv("FLOODVALUE_DATA_DIR", str(tmp_path / "data"))
    with pytest.raises(ValueError, match="grid differs"):
        ctx.build("Sri-Lanka_1", tmp_path / "run", download=False)


def test_build_without_download_reports_missing_file(tmp_path, monkeypatch):
    monkeypatch.setenv("FLOODVALUE_DATA_DIR", str(tmp_path / "data"))
    with pytest.raises(FileNotFoundError):
        ctx.build("Sri-Lanka_1", tmp_path / "run", download=False)
