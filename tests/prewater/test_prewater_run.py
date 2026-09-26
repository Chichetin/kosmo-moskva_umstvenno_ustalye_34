"""prewater build/CLI with a mocked archive (#149 p.1, p.3, p.4): files, statuses, no network."""

import hashlib
import json
from datetime import date
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
from affine import Affine

from floodvalue import cli
from floodvalue.prewater import core, stac
from floodvalue.prewater import run as pw

CHIP = "Sri-Lanka_713926"
H = W = 8
TRANSFORM = Affine(8.983e-05, 0, 80.0, 0, -8.983e-05, 7.0)


def _tif(path, arr, dtype, nodata):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = arr if arr.ndim == 3 else arr[None]
    with rasterio.open(path, "w", driver="GTiff", height=H, width=W, count=arr.shape[0],
                       dtype=dtype, crs="EPSG:4326", transform=TRANSFORM, nodata=nodata) as f:
        f.write(arr.astype(dtype))


@pytest.fixture
def data(tmp_path, monkeypatch):
    monkeypatch.setenv("FLOODVALUE_DATA_DIR", str(tmp_path / "data"))
    _tif(tmp_path / "data" / "raw" / "HandLabeled" / "S1Hand" / f"{CHIP}_S1Hand.tif",
         np.full((2, H, W), -15.0), "float32", np.nan)
    return tmp_path


class FakeArchive:
    """Stands in for floodvalue.prewater.stac: n passes, the left half always water."""
    ArchiveUnavailable = stac.ArchiveUnavailable
    acquisitions = staticmethod(stac.acquisitions)

    def __init__(self, n_passes=6, fail_search=False, fail_reads=0):
        self.n, self.fail_search, self.fail_reads = n_passes, fail_search, fail_reads
        self.searched = None

    def collection_license(self):
        return "CC-BY-4.0"

    def sas_token(self):
        return "sig=x"

    def search(self, bbox, first, last):
        if self.fail_search:
            raise stac.ArchiveUnavailable("HTTP 429")
        self.searched = (bbox, first, last)
        return [{"id": f"S1A_{i}", "properties": {"datetime": f"2017-03-{10 + i:02d}T00:25:00Z",
                                                  "platform": "SENTINEL-1A",
                                                  "sat:orbit_state": "descending",
                                                  "sat:relative_orbit": 19},
                 "assets": {"vh": {"href": f"https://x/{i}.tif"}}} for i in range(self.n)] + [
            {"id": "S1B_VV_only", "properties": {"datetime": "2017-03-29T00:25:00Z",
                                                 "platform": "SENTINEL-1B",
                                                 "sat:relative_orbit": 19},
             "assets": {"vv": {"href": "https://x/vv.tif"}}}]  # no VH: must be skipped

    def read_vh(self, href, token, transform, shape):
        if self.fail_reads:
            self.fail_reads -= 1
            raise stac.ArchiveUnavailable("broken read")
        g = np.full(shape, 0.1, dtype=np.float32)  # -10 dB: dry
        g[:, : shape[1] // 2] = 0.001  # -30 dB: water
        return g


def test_build_writes_layer_on_s1_grid(data):
    out = data / "out"
    p = np.full((H, W), 0.8, dtype=np.float32)
    p[0, 0] = -9999.0
    _tif(out / "flood_probability.tif", p, "float32", -9999.0)
    before = hashlib.sha256((out / "flood_probability.tif").read_bytes()).hexdigest()
    archive = FakeArchive()
    meta = pw.build(CHIP, out, archive=archive, today=date(2026, 9, 26))

    assert meta["status"] == "ok" and len(meta["scenes"]) == 6
    assert archive.searched[1:] == (date(2016, 12, 21), date(2017, 3, 30))  # S1 2017-05-30
    with rasterio.open(out / pw.PERM_FILE) as r:
        perm = r.read(1)
        assert (r.dtypes[0], r.nodata, r.shape) == ("uint8", 255, (H, W))
        assert r.transform.almost_equals(TRANSFORM)
    assert (perm[:, :4] == 1).all() and (perm[:, 4:] == 0).all()
    with rasterio.open(out / pw.NEW_WATER_FILE) as r:
        nw = r.read(1)
        assert (r.dtypes[0], r.nodata) == ("float32", -9999.0)
    assert nw[0, 0] == -9999.0 and (nw[1:, :4] == 0).all() and np.allclose(nw[:, 4:], 0.8)
    # p itself is never rewritten (#149 p.4)
    assert hashlib.sha256((out / "flood_probability.tif").read_bytes()).hexdigest() == before

    saved = json.loads((out / pw.META_FILE).read_text(encoding="utf-8"))
    assert saved["used_by_model"] is False and saved["used_by_economics"] is False
    src = saved["sources"][0]
    assert src["observation_date"] <= saved["s1_date"] and src["after_event_note"]
    assert src["license"] == "CC-BY-4.0" and src["available_date"] == "2026-09-26"


def test_too_few_scenes_is_insufficient_without_rasters(data):
    out = data / "out"
    meta = pw.build(CHIP, out, archive=FakeArchive(n_passes=4))
    assert meta["status"] == "insufficient"
    assert not (out / pw.PERM_FILE).exists() and not (out / pw.NEW_WATER_FILE).exists()
    assert json.loads((out / pw.META_FILE).read_text(encoding="utf-8"))["sources"] == []


def test_unreadable_scene_is_skipped_not_fatal(data):
    meta = pw.build(CHIP, data / "out", archive=FakeArchive(n_passes=7, fail_reads=1))
    assert meta["status"] == "ok" and len(meta["scenes"]) == 6 and len(meta["failed_scenes"]) == 1


def test_no_network_is_a_warning_and_exit_0(data, monkeypatch, capsys):
    out = data / "out"
    out.mkdir()
    (out / pw.PERM_FILE).write_bytes(b"stale")  # a layer from an earlier run must not survive
    monkeypatch.setattr(pw, "stac", FakeArchive(fail_search=True))
    rc = cli.main(["prewater", "--chip", CHIP, "--out", str(out)])
    assert rc == 0
    assert "ПРЕДУПРЕЖДЕНИЕ" in capsys.readouterr().err
    meta = json.loads((out / pw.META_FILE).read_text(encoding="utf-8"))
    assert meta["status"] == "unavailable" and "429" in meta["warning"]
    assert not (out / pw.PERM_FILE).exists()


def test_acquisitions_merge_slices_of_one_pass():
    def item(i, dt, orbit):
        return {"id": i, "properties": {"datetime": dt, "platform": "SENTINEL-1A",
                                        "sat:relative_orbit": orbit}}
    groups = stac.acquisitions([item("b", "2017-03-10T00:25:30Z", 19),
                                item("a", "2017-03-10T00:25:05Z", 19),
                                item("c", "2017-03-10T12:40:00Z", 26),
                                item("d", "2017-03-22T00:25:05Z", 19)])
    assert [[i["id"] for i in g] for g in groups] == [["a", "b"], ["c"], ["d"]]


def test_s1_date_from_metadata_uses_the_mekong_alias(data):
    assert pw.s1_date_of("Mekong") == date(2018, 8, 5)  # "Cambodia" in the metadata
    assert pw.s1_date_of("Sri-Lanka") == date(2017, 5, 30)


def test_cli_requires_chip(capsys):
    assert pw.run(SimpleNamespace(chip=None, out=None)) == 2


def test_parameters_are_recorded(data):
    meta = pw.build(CHIP, data / "out", archive=FakeArchive())
    assert meta["params"]["vh_water_db"] == core.VH_WATER_DB
    assert meta["window"] == {"first": "2016-12-21", "last": "2017-03-30",
                              "days": core.WINDOW_DAYS, "guard_days": core.GUARD_DAYS}


def test_sas_token_and_license_are_fetched_once_per_process(monkeypatch):
    calls = []

    def fake_request(method, url, **kw):
        calls.append(url)
        return {"token": "sig=1", "license": "CC-BY-4.0"}
    monkeypatch.setattr(stac, "_request", fake_request)
    monkeypatch.setattr(stac, "_cache", {})
    assert [stac.sas_token() for _ in range(5)] == ["sig=1"] * 5
    assert stac.collection_license() == stac.collection_license() == "CC-BY-4.0"
    assert calls == [stac.SAS_URL, stac.COLLECTION_URL]
