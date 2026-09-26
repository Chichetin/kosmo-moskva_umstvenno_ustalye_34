"""Observation matching needs place AND time; the test split stays sealed; catalogs are not checks."""

import json

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from floodvalue.media import observations as obs
from floodvalue.media.netfetch import Budget, Fetcher

from .conftest import FIXTURES

BOX = {"type": "Polygon", "coordinates": [[[81.60, 7.70], [81.70, 7.70], [81.70, 7.80],
                                           [81.60, 7.80], [81.60, 7.70]]]}
INSIDE = {"type": "Point", "coordinates": [81.65, 7.75]}
FAR = {"type": "Point", "coordinates": [10.0, 50.0]}


@pytest.mark.parametrize("geom, start, end, seen, relation, suit", [
    (INSIDE, "2017-05-26T00:00:00Z", "2017-05-30T00:00:00Z", "2017-05-30", "covers", "suitable"),
    (INSIDE, "2026-09-25T00:00:00Z", None, "2017-05-30", "covers", "too_early"),  # today's news, old chip
    (INSIDE, "2017-05-01T00:00:00Z", "2017-05-02T00:00:00Z", "2017-06-30", "covers", "too_late"),
    (INSIDE, None, None, "2017-05-30", "covers", "unknown_time"),
    (FAR, "2017-05-26T00:00:00Z", None, "2017-05-30", "none", "unusable"),
    (None, "2017-05-26T00:00:00Z", None, "2017-05-30", "unknown", "unusable"),
])
def test_assess(geom, start, end, seen, relation, suit):
    rel, _, suitability, reason = obs.assess(geom, start, end, BOX, seen)
    assert (rel, suitability) == (relation, suit) and reason


def test_region_centroid_is_never_full_cover():
    rel, _, suit, reason = obs.assess(INSIDE, "2017-05-29T00:00:00Z", None, BOX, "2017-05-30", "region")
    assert (rel, suit) == ("intersects", "partial_cover") and "региона" in reason


def test_partial_cover():
    area = {"type": "Polygon", "coordinates": [[[81.65, 7.75], [81.9, 7.75], [81.9, 7.9],
                                                [81.65, 7.9], [81.65, 7.75]]]}
    assert obs.assess(area, "2017-05-29T00:00:00Z", None, BOX, "2017-05-30")[2] == "partial_cover"


def _tif(path, west, north):
    with rasterio.open(path, "w", driver="GTiff", width=4, height=4, count=1, dtype="float32",
                       crs="EPSG:4326", transform=from_origin(west, north, 0.01, 0.01)) as ds:
        ds.write(np.zeros((1, 4, 4), dtype="float32"))


def test_local_catalog_skips_sealed_test_events(conn, tmp_path):
    data = tmp_path / "data"
    s1 = data / "raw" / "HandLabeled" / "S1Hand"
    s1.mkdir(parents=True)
    (data / "interim").mkdir()
    sources = []
    for chip, event in (("Sri-Lanka_1", "Sri-Lanka"), ("Spain_1", "Spain")):  # Spain is test
        _tif(s1 / f"{chip}_S1Hand.tif", 81.6, 7.8)
        sources.append({"chip_id": chip, "event_id": event, "observation_date": "2017-05-30"})
    (data / "interim" / "source_manifest_s1.json").write_text(json.dumps({"sources": sources}))
    refs = [c["ref"] for c in obs.local_catalog(conn, data)]
    assert refs == ["Sri-Lanka_1"]


def test_cdse_saved_answer_is_found_not_analysed(monkeypatch):
    payload = (FIXTURES / "cdse_s1_grd_2026-09-26.json").read_bytes()

    class R:
        def json(self):
            return json.loads(payload)

    f = Fetcher(allowed_hosts=frozenset({obs.CDSE_HOST}), budget=Budget(1), user_agent="t")
    captured = {}
    monkeypatch.setattr(f, "get", lambda url, params=None: captured.update(params) or R())
    event = {"geometry_json": json.dumps({"type": "Point", "coordinates": [76.6455, 9.6287]}),
             "geo_precision": "locality",
             "event_time_start": "2026-09-14T00:00:00Z", "event_time_end": "2026-09-16T00:00:00Z",
             "first_detected_at": "2026-09-26T08:00:00Z"}
    found = obs.search_cdse(f, event)
    assert found and all(m["stage"] == "found" for m in found)
    assert all("не реализована" in m["reason"] for m in found)
    assert "SENTINEL-1" in captured["$filter"] and "POINT(76.6455 9.6287)" in captured["$filter"]


def test_cdse_without_place_does_not_search():
    f = Fetcher(allowed_hosts=frozenset(), budget=Budget(0), user_agent="t")
    assert obs.search_cdse(f, {"geometry_json": None, "geo_precision": "unknown"}) == []
