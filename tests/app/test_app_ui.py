"""The operator screen reads one result and only recalculates the procurement plan."""

import hashlib
import io
import json
import shutil
import zipfile

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import transform

from floodvalue.app import run
from floodvalue.paths import CONFIGS_DIR, REPO_ROOT


def _synthetic_s1_manifest(tmp_path, monkeypatch):
    interim = tmp_path / "interim"
    interim.mkdir(exist_ok=True)
    (interim / "source_manifest_s1.json").write_text(json.dumps({"sources": [{
        "chip_id": "Synthetic_000001", "observation_date": "2017-05-30",
    }]}), encoding="utf-8")
    monkeypatch.setenv("FLOODVALUE_DATA_DIR", str(tmp_path))


def test_session_layers_and_export(monkeypatch):
    fixture = REPO_ROOT / "tests" / "fixtures" / "good"
    monkeypatch.setenv("FLOODVALUE_APP_RUN_DIR", str(fixture))
    data = run.session()
    assert data["fixture"] is True
    assert len(data["assets"]["features"]) == 10
    assert data["metadata"]["threshold"] == 0.5
    assert data["layer_rasters"]["probability"]["corners"] == data["raster"]["corners"]
    assert run.raster_layer("probability").body.startswith(b"\x89PNG\r\n\x1a\n")
    assert run.raster_layer("mask").body.startswith(b"\x89PNG\r\n\x1a\n")
    with zipfile.ZipFile(io.BytesIO(run.download().body)) as archive:
        assert {"run_metadata.json", "flood_probability.tif", "asset_loss.csv"} <= set(archive.namelist())


def test_budget_changes_plan_without_writing_rasters(tmp_path, monkeypatch):
    fixture = REPO_ROOT / "tests" / "fixtures" / "good"
    monkeypatch.setenv("FLOODVALUE_APP_RUN_DIR", str(fixture))
    _synthetic_s1_manifest(tmp_path, monkeypatch)
    protected = [fixture / "flood_probability.tif", fixture / "flood_mask.tif",
                 fixture / "run_metadata.json"]
    before = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in protected}
    low = json.loads(run.recalculated_plan(0).body)
    high = json.loads(run.recalculated_plan(5000).body)
    assert low["plans"]["C"] == []
    assert high["comparison"][2]["data_cost_rub"] <= 5000
    assert high["plans"]["C"] != low["plans"]["C"]
    assert before == {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in protected}


def test_plan_warns_when_pricing_changed_after_bundle_creation(tmp_path, monkeypatch):
    from floodvalue import contracts
    from floodvalue.procurement import pricing

    bundle = tmp_path / "bundle"
    shutil.copytree(REPO_ROOT / "tests" / "fixtures" / "good", bundle)
    monkeypatch.setenv("FLOODVALUE_APP_RUN_DIR", str(bundle))
    _synthetic_s1_manifest(tmp_path, monkeypatch)
    configs = tmp_path / "configs"
    configs.mkdir()
    for name in ("run.yaml", "pricing.yaml"):
        shutil.copy(CONFIGS_DIR / name, configs / name)
    monkeypatch.setattr(run, "CONFIGS_DIR", configs, raising=False)
    monkeypatch.setattr(contracts, "CONFIGS_DIR", configs)
    monkeypatch.setattr(pricing, "DEFAULT_CONFIG", configs / "pricing.yaml")
    metadata_path = bundle / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["config_sha256"] = {
        name: hashlib.sha256((configs / name).read_bytes()).hexdigest()
        for name in ("run.yaml", "pricing.yaml")
    }
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    same = json.loads(run.recalculated_plan(5000).body)
    assert same["configuration_warning"] is None

    pricing_path = configs / "pricing.yaml"
    pricing_path.write_text(pricing_path.read_text(encoding="utf-8").replace(
        "rub_per_km2: 788.64", "rub_per_km2: 900.00"), encoding="utf-8")
    changed = json.loads(run.recalculated_plan(5000).body)
    assert "pricing.yaml" in changed["configuration_warning"]


def test_scene_catalog_and_selected_run_stay_consistent(tmp_path, monkeypatch):
    monkeypatch.delenv("FLOODVALUE_APP_RUN_DIR", raising=False)
    monkeypatch.setattr(run, "output_dir", lambda: tmp_path)
    monkeypatch.setattr(run, "data_dir", lambda: tmp_path)
    _synthetic_s1_manifest(tmp_path, monkeypatch)
    fixture = REPO_ROOT / "tests" / "fixtures" / "good"
    for index in (1, 2):
        target = tmp_path / f"scene-{index}"
        shutil.copytree(fixture, target)
        metadata_path = target / "run_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata.update(run_id=f"scene-{index}", chip_id=f"chip-{index}", event_id=f"Area-{index}")
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    later = tmp_path / "scene-3"
    shutil.copytree(fixture, later)
    metadata = json.loads((later / "run_metadata.json").read_text(encoding="utf-8"))
    metadata.update(run_id="scene-3", chip_id="chip-1", event_id="Area-1")
    (later / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (later / "source_manifest.json").write_text(json.dumps({"sources": [{
        "source_id": "test:S1:chip-1", "chip_id": "chip-1", "observation_date": "2018-04-09",
    }]}), encoding="utf-8")
    catalog = run.scenes()
    assert {item["run_id"] for item in catalog} == {"scene-1", "scene-2", "scene-3"}
    assert all(item["position"] is not None for item in catalog)
    assert next(item for item in catalog if item["run_id"] == "scene-3")["observation_date"] == "2018-04-09"
    selected = run.session("scene-2")
    assert selected["run_id"] == "scene-2"
    assert selected["metadata"]["chip_id"] == "chip-2"
    assert run.raster_layer("probability", "scene-2").body.startswith(b"\x89PNG")
    assert run.download("scene-2").headers["content-disposition"].endswith('scene-2.zip"')
    assert json.loads(run.recalculated_plan(5000, "scene-2").body)["plans"]["C"]
    with pytest.raises(run.HTTPException) as missing:
        run.session("../../outside")
    assert missing.value.status_code == 404


def test_scene_coordinates_transform_and_missing_georeference(tmp_path):
    east, north = transform("EPSG:4326", "EPSG:3857", [81.66], [7.75])
    projected = tmp_path / "projected.tif"
    with rasterio.open(projected, "w", driver="GTiff", width=10, height=10, count=1,
                       dtype="float32", crs="EPSG:3857",
                       transform=from_origin(east[0] - 500, north[0] + 500, 100, 100)) as dest:
        dest.write(np.zeros((1, 10, 10), dtype=np.float32))
    position = run._scene_position(projected)
    assert position is not None
    assert position["lat"] == pytest.approx(7.75, abs=0.0001)
    assert position["lon"] == pytest.approx(81.66, abs=0.0001)
    assert run._raster_info(projected)["crs"] == "EPSG:3857"

    unknown = tmp_path / "unknown.tif"
    with rasterio.open(unknown, "w", driver="GTiff", width=10, height=10, count=1,
                       dtype="float32", transform=from_origin(10, 10, 1, 1)) as dest:
        dest.write(np.zeros((1, 10, 10), dtype=np.float32))
    assert run._scene_position(unknown) is None
    assert run._raster_info(unknown)["bounds"] is None


def test_raw_observation_catalog_never_borrows_another_scenes_analysis(tmp_path, monkeypatch):
    monkeypatch.delenv("FLOODVALUE_APP_RUN_DIR", raising=False)
    monkeypatch.setattr(run, "output_dir", lambda: tmp_path / "empty-outputs")
    monkeypatch.setattr(run, "data_dir", lambda: tmp_path)
    raw = tmp_path / "raw" / "HandLabeled" / "S1Hand"
    raw.mkdir(parents=True)
    path = raw / "Test_1_S1Hand.tif"
    with rasterio.open(path, "w", driver="GTiff", width=8, height=8, count=2,
                       dtype="float32", crs="EPSG:4326",
                       transform=from_origin(81.6, 7.8, 0.01, 0.01)) as dest:
        dest.write(np.full((2, 8, 8), -15, dtype=np.float32))
    manifest = tmp_path / "interim" / "source_manifest_s1.json"
    manifest.parent.mkdir()
    manifest.write_text(json.dumps({"sources": [{"chip_id": "Test_1", "file": path.name,
                                                  "event_id": "Test", "observation_date": "2017-05-30"}]}),
                                                      encoding="utf-8")
    catalog = run.scenes()
    assert len(catalog) == 1
    assert catalog[0]["status"] == "observation_only"
    assert catalog[0]["fixture"] is False
    assert catalog[0]["run_id"] is None
    assert catalog[0]["bounds"] == pytest.approx([81.6, 7.72, 81.68, 7.8])
    assert run.observation_preview("Test_1").body.startswith(b"\x89PNG")
    assert run.observation_s1("Test_1").body.startswith(b"\x89PNG")
    with pytest.raises(run.HTTPException) as missing:
        run.session()
    assert missing.value.status_code == 503


def test_empty_catalog_does_not_publish_test_fixture(tmp_path, monkeypatch):
    monkeypatch.delenv("FLOODVALUE_APP_RUN_DIR", raising=False)
    monkeypatch.setattr(run, "output_dir", lambda: tmp_path / "outputs")
    monkeypatch.setattr(run, "data_dir", lambda: tmp_path / "data")
    assert run.scenes() == []
    with pytest.raises(run.HTTPException) as missing:
        run.session()
    assert missing.value.status_code == 503


def test_projected_visual_layer_is_warped_to_wgs84(tmp_path):
    east, north = transform("EPSG:4326", "EPSG:3857", [81.66], [7.75])
    path = tmp_path / "projected.tif"
    with rasterio.open(path, "w", driver="GTiff", width=10, height=10, count=1,
                       dtype="float32", crs="EPSG:3857",
                       transform=from_origin(east[0] - 500, north[0] + 500, 100, 100)) as dest:
        dest.write(np.arange(100, dtype=np.float32).reshape(10, 10), 1)
    values, valid = run._visual_values(path, 1)
    assert values.shape == (10, 10)
    assert valid.any()
    assert values[0, 0] < values[-1, -1]
    assert run._raster_info(path)["bounds"][0] == pytest.approx(81.655, abs=.002)


def test_mask_nodata_and_zero_remain_transparent(tmp_path):
    path = tmp_path / "mask.tif"
    with rasterio.open(path, "w", driver="GTiff", width=2, height=2, count=1,
                       dtype="uint8", nodata=255, crs="EPSG:4326",
                       transform=from_origin(80, 8, .01, .01)) as dest:
        dest.write(np.array([[0, 1], [255, 1]], dtype=np.uint8), 1)
    from PIL import Image
    rgba = np.array(Image.open(io.BytesIO(run._png("mask", path, 1))))
    assert rgba[0, 0, 3] == 0
    assert rgba[1, 0, 3] == 0
    assert rgba[0, 1, 3] == rgba[1, 1, 3] == 157


def test_test_chips_stay_sealed_in_the_raw_catalog(tmp_path, monkeypatch):
    """Until eval-test the operator catalog neither lists nor renders test chips (D-010, D-033)."""
    monkeypatch.delenv("FLOODVALUE_APP_RUN_DIR", raising=False)
    monkeypatch.delenv("FLOODVALUE_APP_ALLOW_TEST", raising=False)
    monkeypatch.setattr(run, "output_dir", lambda: tmp_path / "empty-outputs")
    monkeypatch.setattr(run, "data_dir", lambda: tmp_path)
    raw = tmp_path / "raw" / "HandLabeled" / "S1Hand"
    raw.mkdir(parents=True)
    sources = []
    for chip_id in ("Spain_1", "Nigeria_1"):
        path = raw / f"{chip_id}_S1Hand.tif"
        with rasterio.open(path, "w", driver="GTiff", width=8, height=8, count=2,
                           dtype="float32", crs="EPSG:4326",
                           transform=from_origin(81.6, 7.8, 0.01, 0.01)) as dest:
            dest.write(np.full((2, 8, 8), -15, dtype=np.float32))
        sources.append({"chip_id": chip_id, "file": path.name,
                        "event_id": chip_id.split("_")[0], "observation_date": "2017-05-30"})
    manifest = tmp_path / "interim" / "source_manifest_s1.json"
    manifest.parent.mkdir()
    manifest.write_text(json.dumps({"sources": sources}), encoding="utf-8")

    assert [item["chip_id"] for item in run.scenes()] == ["Nigeria_1"]
    for endpoint in (run.observation_preview, run.observation_s1):
        with pytest.raises(run.HTTPException) as sealed:
            endpoint("Spain_1")
        assert sealed.value.status_code == 404
    assert run.observation_preview("Nigeria_1").body.startswith(b"\x89PNG")

    monkeypatch.setenv("FLOODVALUE_APP_ALLOW_TEST", "1")
    assert sorted(item["chip_id"] for item in run.scenes()) == ["Nigeria_1", "Spain_1"]
    assert run.observation_s1("Spain_1").body.startswith(b"\x89PNG")
