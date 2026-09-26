"""Observation matching respects place AND time (spec §З, example 9)."""

from datetime import UTC, datetime

from floodvalue.media import observations

SCENE = {"chip_id": "Sri-Lanka_713926", "observation_date": "2017-05-30",
         "bounds": (80.10, 6.90, 80.15, 6.95)}
INSIDE = (6.92, 80.12)


def _at(day: str) -> datetime:
    return datetime.fromisoformat(day).replace(tzinfo=UTC)


def test_covering_scene_in_window_is_suitable_but_not_a_verdict():
    item = observations.assess(*INSIDE, _at("2017-05-28"), None, SCENE, {})
    assert item["suitability"] == "suitable" and item["spatial_relation"] == "covers"
    assert item["stage"] == "prepared" and item["analysis_run_id"] is None
    assert "не проверка" in item["reason"]


def test_scene_two_weeks_before_event_is_too_early():
    item = observations.assess(*INSIDE, _at("2017-06-13"), None, SCENE, {})
    assert item["suitability"] == "too_early"
    assert item["time_offset_days"] == -14.0


def test_todays_report_never_matches_historical_chip():
    item = observations.assess(*INSIDE, _at("2026-09-26"), None, SCENE, {})
    assert item["suitability"] == "too_early"


def test_late_scene_and_unknown_time():
    assert observations.assess(*INSIDE, _at("2017-04-01"), None, SCENE, {})["suitability"] \
        == "too_late"
    assert observations.assess(*INSIDE, None, None, SCENE, {})["suitability"] == "unusable"


def test_neighbouring_scene_is_partial_far_scene_is_skipped():
    near = observations.assess(6.97, 80.12, _at("2017-05-29"), None, SCENE, {})
    assert near["spatial_relation"] == "nearby" and near["suitability"] == "partial_cover"
    assert observations.assess(7.9, 81.5, _at("2017-05-29"), None, SCENE, {}) is None


def test_existing_bundle_marks_scene_analyzed():
    item = observations.assess(*INSIDE, _at("2017-05-29"), None, SCENE,
                               {"Sri-Lanka_713926": "Sri-Lanka_713926_5000_da94206"})
    assert item["stage"] == "analyzed"
    assert item["analysis_run_id"] == "Sri-Lanka_713926_5000_da94206"


def test_sealed_test_chips_are_not_in_catalog(tmp_path, monkeypatch):
    import json

    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    sealed = sorted(observations._sealed_chips())
    assert sealed, "split_v1.json должен содержать test"
    open_chip, sealed_chip = "Synthetic_000001", sealed[0]
    folder = tmp_path / "raw" / "HandLabeled" / "S1Hand"
    folder.mkdir(parents=True)
    for chip in (open_chip, sealed_chip):
        with rasterio.open(folder / f"{chip}_S1Hand.tif", "w", driver="GTiff", width=4,
                           height=4, count=1, dtype="float32", crs="EPSG:4326",
                           transform=from_origin(80.1, 6.95, 0.01, 0.01)) as dst:
            dst.write(np.zeros((1, 4, 4), dtype="float32"))
    interim = tmp_path / "interim"
    interim.mkdir()
    (interim / "source_manifest_s1.json").write_text(json.dumps({"sources": [
        {"chip_id": chip, "file": f"{chip}_S1Hand.tif", "observation_date": "2019-09-17"}
        for chip in (open_chip, sealed_chip)]}))
    monkeypatch.setattr(observations, "data_dir", lambda: tmp_path)
    observations._catalog.cache_clear()
    assert [scene["chip_id"] for scene in observations.local_catalog()] == [open_chip]
    observations._catalog.cache_clear()
