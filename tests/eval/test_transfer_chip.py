"""Сквозной запуск замороженного основного метода на ДРУГОМ чипе (issue #27, критерии 7/11).

Жюри «может самостоятельно... повторить inference на другом официальном S1-чипе»
(task.md, «Проверка на защите»). Здесь тот же сценарий воспроизводится тестом: `predict`
(реальные веса U-Net + калибратор Platt из `models/unet/`, без подгонки) -> `assets` ->
`procure` -> `export` -> `validate` на чипе, который не совпадает с чипом по умолчанию из
`configs/run.yaml`. Чип синтетический (для скорости CI), но метод и веса — ровно те, что
идут в комплект (ту же схему уже использует `tests/eval/test_eval_run.py`).

Проверяется: (1) 12/12 файлов проходят валидатор на новом chip_id; (2) используется
основной метод (unet) с зафиксированным на validation порогом/калибратором, а не baseline
и не другой порог; (3) nodata S1 (NaN) корректно доходит до растров (-9999/255,
docs/contracts.md §2) и портфель не спотыкается о него (правило размещения — только по
валидности S1, D-005; no_data-статус отдельно проверен в `tests/assets/test_run.py`);
(4) никакие файлы Sentinel-2/JRC не читаются и не нужны — режим S1-only (README).

Test-чипы (Spain/Mekong/Bolivia) не участвуют — это не про test (D-010), а про
воспроизводимость на чипе, отличном от демо-чипа validation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from floodvalue.assets import run as assets_run
from floodvalue.contracts import load_run_config
from floodvalue.export import run as export_run
from floodvalue.export.validate import validate_run
from floodvalue.model import predict as main_predict
from floodvalue.paths import MODELS_DIR
from floodvalue.procurement import run as procure_run

# Явно ДРУГОЙ чип: не тот, что по умолчанию в configs/run.yaml (проверяется тестом ниже).
CHIP = "Transfer_000777"
H = W = 150          # ~2,25 км² при шаге 8,983e-05° — выше порога зоны 1 км² (procurement)
BUDGET = 5000.0


def _s1_manifest(chip: str) -> dict:
    return {"dataset": "Sen1Floods11 v1.1", "repository": "https://example.org/repo",
            "sources": [
                {"source_id": f"sen1floods11-v1.1:S1Hand:{chip}", "chip_id": chip,
                 "event_id": chip.rsplit("_", 1)[0], "file": f"{chip}_S1Hand.tif",
                 "url": f"https://example.org/{chip}_S1Hand.tif", "version": "v1.1",
                 "observation_date": "2019-08-01", "published_at": "2020-10-01T00:00:00Z",
                 "published_after_event": True, "license": "см. репозиторий",
                 "purpose": "S1 VV/VH", "limitations": "опубликован в 2020 г.",
                 "label": {"file": f"{chip}_LabelHand.tif", "purpose": "метка"}}]}


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Только S1Hand + манифест S1 — ни S2Hand, ни JRCWaterHand, ни LabelHand на диске нет:
    пайплайн не должен их требовать (режим S1-only)."""
    data, out = tmp_path / "data", tmp_path / "outputs"
    s1_dir = data / "raw" / "HandLabeled" / "S1Hand"
    s1_dir.mkdir(parents=True)
    rng = np.random.default_rng(20260927)
    vv = rng.normal(-13.0, 3.0, (H, W)).astype("float32")
    vh = rng.normal(-19.0, 3.0, (H, W)).astype("float32")
    water = np.zeros((H, W), bool)
    water[15:95, 8:60] = True                               # river-like body, lower VH
    vh[water] = rng.normal(-26.0, 1.5, water.sum())
    vv[water] = rng.normal(-18.0, 1.5, water.sum())
    vv[5:30, 110:140] = rng.normal(24.0, 1.0, (25, 30))     # extreme high VV (built-up, §11)
    s1 = np.stack([vv, vh])
    s1[:, :45, -45:] = np.nan                               # a real nodata corner (~9 %)
    nan_frac = float(np.isnan(s1[0]).mean())
    assert 0.0 < nan_frac < 0.5, "фикстуре нужна area NaN, но не весь чип"
    transform = from_origin(31.0, 2.0, 8.983152841195215e-05, 8.983152841195215e-05)
    with rasterio.open(s1_dir / f"{CHIP}_S1Hand.tif", "w", driver="GTiff", height=H, width=W,
                       count=2, dtype="float32", crs="EPSG:4326", transform=transform,
                       nodata=np.nan) as dst:
        dst.write(s1)
    (data / "interim").mkdir()
    (data / "interim" / "source_manifest_s1.json").write_text(
        json.dumps(_s1_manifest(CHIP)), encoding="utf-8")
    monkeypatch.setenv("FLOODVALUE_DATA_DIR", str(data))
    monkeypatch.setenv("FLOODVALUE_OUTPUT_DIR", str(out))
    monkeypatch.delenv("FLOODVALUE_METHOD", raising=False)   # unet — метод по умолчанию
    return {"data": data, "out": out, "s1": s1, "nan_frac": nan_frac}


def test_chip_differs_from_configs_run_yaml_default():
    """Гарантия, что тест действительно проверяет «другой» чип, а не демо-чип по умолчанию."""
    assert CHIP != load_run_config().chip_id


def test_transfer_run_uses_frozen_main_method_and_passes_validator(env):
    ns = argparse.Namespace(chip=CHIP, budget=BUDGET, out=None)

    assert main_predict.run(ns) == 0
    pred_path = env["out"] / CHIP / "prediction.json"
    prediction = json.loads(pred_path.read_text(encoding="utf-8"))
    assert prediction["method"] == "unet"                        # основной метод, не baseline
    cal = json.loads((MODELS_DIR / "unet" / "calibration.json").read_text(encoding="utf-8"))
    assert prediction["threshold"] == cal["threshold"]            # зафиксирован на validation
    assert prediction["calibration"]["kind"] == cal["calibrator"]["kind"] == "platt"
    assert prediction["threshold_status"] == cal["threshold_status"]  # без подгонки под чип

    # nodata S1 -> nodata растров, docs/contracts.md §2 (-9999 / 255)
    with rasterio.open(env["out"] / CHIP / "flood_probability.tif") as d:
        p = d.read(1)
        assert d.nodata == -9999.0
        nodata_px = int((p == -9999.0).sum())
    with rasterio.open(env["out"] / CHIP / "flood_mask.tif") as d:
        m = d.read(1)
        assert d.nodata == 255
        assert int((m == 255).sum()) == nodata_px
    s1_nan_px = int(np.isnan(env["s1"][0]).sum())
    assert nodata_px == s1_nan_px == H * W - prediction["valid_pixels"]

    assert assets_run.run(ns) == 0
    assert procure_run.run(ns) == 0
    assert export_run.run(ns) == 0

    _, short, _ = export_run.git_commit()
    dest = env["out"] / f"{CHIP}_{export_run.assemble.budget_label(BUDGET)}_{short}"
    failing = {name: errs for name, errs in validate_run(dest).items() if errs}
    assert failing == {}, failing                                 # 12/12

    meta = json.loads((dest / "run_metadata.json").read_text(encoding="utf-8"))
    assert meta["chip_id"] == CHIP
    assert meta["methods"]["main"]["name"] == "unet"
    assert meta["threshold"] == cal["threshold"]
    assert "S1-only" in meta["source"]["s1_only"]
    assert meta["source"]["s1_files"] == [f"{CHIP}_S1Hand.tif"]   # ни один S2/JRC файл не назван

    # портфель: статус каждого объекта — валидный код контракта, ok-объекты оценены,
    # no_data (если бы объект попал в невалидный пиксель) отдельно проверен в
    # tests/assets/test_run.py::test_run_no_data_asset_has_empty_fields
    with open(dest / "asset_loss.csv", encoding="utf-8", newline="") as f:
        import csv
        rows = list(csv.DictReader(f))
    assert len(rows) == 10
    for row in rows:
        assert row["status"] in ("ok", "partial", "no_data")
        if row["status"] == "ok":
            assert row["p_flood"] not in ("", None)
            assert float(row["p_flood"]) == float(row["p_flood"])  # не NaN (не бросает)
        else:
            assert row["p_flood"] == "" and row["expected_loss_rub"] == ""


def test_s1_only_no_s2_or_jrc_files_anywhere_in_data_dir(env):
    """Ни один слой, кроме S1Hand, не лежит на диске — и пайплайн всё равно проходит целиком:
    прямое подтверждение режима S1-only (task.md, критерий 7)."""
    for bad in ("S2Hand", "JRCWaterHand", "S1OtsuLabelHand", "LabelHand"):
        assert not list(env["data"].rglob(f"*{bad}*"))
    ns = argparse.Namespace(chip=CHIP, budget=BUDGET, out=None)
    assert main_predict.run(ns) == 0
    assert assets_run.run(ns) == 0
    assert procure_run.run(ns) == 0
    assert export_run.run(ns) == 0
