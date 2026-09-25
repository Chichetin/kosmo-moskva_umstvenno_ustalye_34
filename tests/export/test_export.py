"""export: one result set from the stage files (D-027), checked by the validator (criterion 12).

The end-to-end test runs every stage on a small synthetic S1 chip: baseline predict ->
assets -> procure -> export -> validate.
"""

import argparse
import csv
import hashlib
import json

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from floodvalue.assets import run as assets_run
from floodvalue.baseline import predict as baseline_predict
from floodvalue.contracts import files as F
from floodvalue.export import assemble
from floodvalue.export import run as export_run
from floodvalue.export.validate import validate_run
from floodvalue.paths import MODELS_DIR, SPLITS_DIR
from floodvalue.procurement import run as procure_run

CHIP = "Fixture_000001"
H = W = 128
TRANSFORM = from_origin(80.0, 7.05, 8.983152841195215e-05, 8.983152841195215e-05)


# --- pure helpers --------------------------------------------------------------------------------

def test_run_id_follows_contract():
    assert assemble.make_run_id("Sri-Lanka_713926", 5000.0, "abc1234") == \
        "Sri-Lanka_713926_5000_abc1234"
    assert assemble.budget_label(5000) == "5000"
    assert assemble.budget_label(2500.5) == "2500.5"
    assert assemble.budget_label(0.0) == "0"
    assert assemble.budget_label(1_000_000.0) == "1000000"


def test_split_hash_matches_frozen_split():
    split = json.loads((SPLITS_DIR / "split_v1.json").read_text(encoding="utf-8"))
    assert assemble.split_sha256(split) == split["sha256"]
    assert assemble.chip_split(split, "Sri-Lanka_713926") == "validation"
    assert assemble.chip_split(split, "Nowhere_1") == "none"


BASE_ROWS = [
    {"method": "baseline", "split": s, "event": "ALL", "iou": "0.5", "f1": "0.6",
     "precision": "0.7", "recall": "0.5", "tp": "5", "fp": "2", "fn": "5", "tn": "88",
     "n_pixels": "100", "threshold": "-23.0", "valid_mask_sha256": f"h-{s}"}
    for s in ("train", "dev", "validation")
]
BASELINE_PRED = {"method": "baseline_sigmoid (temporary main method, D-009)", "threshold": 0.5,
                 "channel": "VH"}


def test_metrics_main_is_baseline_rule_on_same_mask_while_d009():
    rows = assemble.metrics_rows(BASE_ROWS, None, BASELINE_PRED, "b.csv", None)
    assert {r["split"] for r in rows} == {"dev", "validation"}  # train is not a §7 split
    by = {(r["method"], r["split"]): r for r in rows}
    for s in ("dev", "validation"):
        b, m = by[("baseline", s)], by[("main", s)]
        assert b["valid_mask_sha256"] == m["valid_mask_sha256"] == f"h-{s}"
        assert (b["threshold"], b["threshold_units"]) == ("-23.0", "dB VH")
        assert (m["threshold"], m["threshold_units"]) == ("0.5", "p")
        assert m["iou"] == b["iou"]  # p >= 0.5 <=> VH <= -23: the same confusion counts


def test_metrics_real_main_uses_its_own_rows_and_never_copies_baseline():
    main = [{**BASE_ROWS[1], "method": "main", "iou": "0.8", "threshold": "0.42"}]
    pred = {"method": "unet", "threshold": 0.42}
    rows = assemble.metrics_rows(BASE_ROWS, main, pred, "b.csv", "m.csv")
    mains = [r for r in rows if r["method"] == "main"]
    assert [(r["iou"], r["threshold"], r["source_file"]) for r in mains] == \
        [("0.8", "0.42", "m.csv")]
    assert not [r for r in assemble.metrics_rows(BASE_ROWS, None, pred, "b.csv", None)
                if r["method"] == "main"]
    # baseline raster + stray main-metrics: main rows follow the raster, not the file
    stray = assemble.metrics_rows(BASE_ROWS, main, BASELINE_PRED, "b.csv", "m.csv")
    assert {r["iou"] for r in stray if r["method"] == "main"} == {"0.5"}


def test_metrics_rows_named_by_method_become_main():
    unet = [{**BASE_ROWS[2], "method": "unet", "iou": "0.6", "threshold": "0.5"},
            {**BASE_ROWS[0], "method": "unet"}]  # train row is dropped (§7)
    rows = assemble.metrics_rows(BASE_ROWS, unet, {"method": "unet", "threshold": 0.5},
                                 "b.csv", "docs/verification/unet-metrics.csv")
    mains = [r for r in rows if r["method"] == "main"]
    assert [(r["split"], r["iou"], r["method_name"]) for r in mains] == \
        [("validation", "0.6", "unet")]
    assert not [r for r in rows if r["method"] == "unet"]


def test_metrics_prefers_calibrated_file_and_matches_mask_threshold(tmp_path):
    # #62: with a calibrator applied, main rows must come from the calibrated metrics
    # (same threshold as the mask, contracts §7), not the raw docs/verification/unet-metrics.csv.
    prediction = {"method": "unet", "threshold": 0.37,
                 "calibration": {"file": "models/unet/calibration.json",
                                 "sha256": "deadbeef", "kind": "platt"}}
    warnings = []
    files = export_run._metrics(prediction, tmp_path, warnings)
    assert "docs/verification/calibration/unet-metrics.csv" in files
    assert "docs/verification/unet-metrics.csv" not in files
    with open(tmp_path / "metrics.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    main_val_all = next(r for r in rows if r["method"] == "main" and r["split"] == "validation"
                        and r["event"] == "ALL")
    assert float(main_val_all["threshold"]) == pytest.approx(0.37)
    assert float(main_val_all["threshold"]) == prediction["threshold"]  # = mask threshold
    assert not [w for w in warnings if "порог" in w]  # no stale-threshold warning


def test_metrics_without_calibration_uses_raw_file_and_warns_if_calibrated_missing(tmp_path):
    warnings = []
    files = export_run._metrics({"method": "unet", "threshold": 0.5, "calibration": None},
                                tmp_path, warnings)
    assert "docs/verification/unet-metrics.csv" in files
    assert "docs/verification/calibration/unet-metrics.csv" not in files
    assert not warnings  # no calibrator applied: unchanged behaviour, no extra warning

    warnings2 = []
    files2 = export_run._metrics(
        {"method": "no-such-method", "threshold": 0.4,
         "calibration": {"file": "models/no-such-method/calibration.json", "kind": "platt"}},
        tmp_path, warnings2)
    assert "docs/verification/calibration/no-such-method-metrics.csv" not in files2
    assert any("калибратор применён" in w for w in warnings2)  # calibrator flagged but file missing


def _sens(scenario, changed):
    return {"scenario_id": scenario, "strategy": "C", "changed_inputs_json": json.dumps(changed),
            "decision_cost_rub": "1", "covered_expected_loss_rub": "1",
            "result_status": "ok", "interpretation": "x"}


def test_sensitivity_merge_and_missing_keys():
    proc = [_sens("budget_0", {"budget_rub": 0}),
            _sens("rate", {"base_rate_factor": 0.5, "base_rate_rub_km2": 394.32})]
    assert assemble.missing_sensitivity_keys(proc) == \
        ["asset_value_rub", "p_flood", "vulnerability_coef"]
    assets = [_sens("p", {"p_flood": 0.1}), _sens("v", {"asset_value_rub": 1}),
              _sens("q", {"vulnerability_coef": 1}),
              _sens("pv", {"p_flood": 0.1, "asset_value_rub": 1})]  # joint: does not count
    merged = assemble.merge_sensitivity(proc, assets)
    assert [r["scenario_id"] for r in merged][:2] == ["budget_0", "rate"]
    assert assemble.missing_sensitivity_keys(merged) == []
    assert assemble.missing_sensitivity_keys(proc + assets[3:]) == \
        ["asset_value_rub", "p_flood", "vulnerability_coef"]


def test_fixed_plan_c_rows_recompute_covered_el():
    assets = [{"asset_id": a, "asset_value_rub": "1000000", "vulnerability_coef": "0.5"}
              for a in ("A01", "A02", "A03")]
    loss = [{"asset_id": "A01", "p_flood": "0.4", "status": "ok"},
            {"asset_id": "A02", "p_flood": "0.9", "status": "ok"},
            {"asset_id": "A03", "p_flood": "", "status": "no_data"}]
    comp = [{"strategy": "C", "decision_cost_rub": "2500.00",
             "covered_expected_loss_rub": "450000.0", "covered_asset_ids": "A02"}]
    single = {"p_flood": {"factor": 2.0}, "scope": "single_asset", "asset_id": "A02",
              "portfolio_expected_loss_rub": 200000 + 500000}  # p clipped to 1
    portfolio = {"vulnerability_coef": {"factor": 0.5}, "scope": "portfolio",
                 "portfolio_expected_loss_rub": 325000}
    wrong = {"asset_value_rub": {"factor": 2.0}, "scope": "portfolio",
             "portfolio_expected_loss_rub": 1}
    rows, warns = assemble.fixed_plan_rows(
        [_sens("s", single), _sens("p", portfolio), _sens("w", wrong)], assets, loss, comp,
        ["Z-A02"])
    assert [(r["scenario_id"], r["strategy"], r["decision_cost_rub"]) for r in rows] == \
        [("s__C_fixed", "C", "2500.00"), ("p__C_fixed", "C", "2500.00")]
    assert float(rows[0]["covered_expected_loss_rub"]) == pytest.approx(500000)
    assert float(rows[1]["covered_expected_loss_rub"]) == pytest.approx(225000)
    assert json.loads(rows[0]["changed_inputs_json"])["plan_C_fixed"] == ["Z-A02"]
    assert len(warns) == 1 and "w" in warns[0]
    unchanged = {"p_flood": {"factor": 1.0}, "scope": "portfolio",
                 "portfolio_expected_loss_rub": 650000}
    same, _ = assemble.fixed_plan_rows([_sens("k1", unchanged)], assets, loss, comp, ["Z-A02"])
    assert same[0]["covered_expected_loss_rub"] == "450000.00"  # k = 1 -> comparison value
    assert assemble.fixed_plan_rows([_sens("s", single)], assets, loss, [], [])[0] == []
    noted = assemble.with_model_note([_sens("s", single)])
    assert assemble.MODEL_NOTE in noted[0]["interpretation"]


def _s1_manifest(chip):
    return {"dataset": "Sen1Floods11 v1.1", "repository": "https://example.org/repo",
            "sources": [
                {"source_id": f"sen1floods11-v1.1:S1Hand:{chip}", "chip_id": chip,
                 "event_id": chip.rsplit("_", 1)[0], "file": f"{chip}_S1Hand.tif",
                 "url": f"https://example.org/{chip}_S1Hand.tif", "version": "v1.1",
                 "observation_date": "2017-05-30", "published_at": "2020-10-01T17:25:14Z",
                 "published_after_event": True, "license": "см. репозиторий",
                 "purpose": "S1 VV/VH", "limitations": "опубликован в 2020 г.",
                 "label": {"file": f"{chip}_LabelHand.tif", "purpose": "метка"}},
                {"source_id": "sen1floods11-v1.1:S1Hand:Other_1", "chip_id": "Other_1",
                 "file": "Other_1_S1Hand.tif", "url": "u", "version": "v",
                 "observation_date": "2017-05-30", "published_at": "2020-10-01T00:00:00Z",
                 "license": "l", "purpose": "p", "limitations": "l"}]}


def test_source_manifest_keeps_only_this_chip_and_marks_label():
    m = assemble.source_manifest(_s1_manifest(CHIP), CHIP, label_read=False, extra_sources=[])
    assert [s["chip_id"] for s in m["sources"]] == [CHIP]
    s = m["sources"][0]
    assert s["label"]["used_by_method"] is False and s["applies_to"] == CHIP
    assert s["after_event_note"]  # published after the event: the note is explicit
    with pytest.raises(ValueError):
        assemble.source_manifest(_s1_manifest(CHIP), "Missing_1", False, [])


# --- end to end on a synthetic chip ---------------------------------------------------------------

@pytest.fixture
def env(tmp_path, monkeypatch):
    data, out = tmp_path / "data", tmp_path / "outputs"
    s1_dir = data / "raw" / "HandLabeled" / "S1Hand"
    s1_dir.mkdir(parents=True)
    rng = np.random.default_rng(7)
    s1 = rng.normal(-15, 2, (2, H, W)).astype("float32")
    s1[1, 40:90, 30:100] = rng.normal(-27, 1.5, (50, 70))  # a water body in VH
    s1[:, :, :6] = np.nan  # S1 nodata strip
    with rasterio.open(s1_dir / f"{CHIP}_S1Hand.tif", "w", driver="GTiff", height=H, width=W,
                       count=2, dtype="float32", crs="EPSG:4326", transform=TRANSFORM,
                       nodata=np.nan) as dst:
        dst.write(s1)
    (data / "interim").mkdir()
    (data / "interim" / "source_manifest_s1.json").write_text(
        json.dumps(_s1_manifest(CHIP)), encoding="utf-8")
    monkeypatch.setenv("FLOODVALUE_DATA_DIR", str(data))
    monkeypatch.setenv("FLOODVALUE_OUTPUT_DIR", str(out))
    return out


def _stages(out, budget=5000.0):
    work = out / CHIP
    baseline_predict.predict_chip(CHIP, work)
    args = argparse.Namespace(chip=CHIP, budget=budget, out=None)
    assert assets_run.run(args) == 0
    assert procure_run.run(args) == 0
    return work, args


def _failing(errors):
    return {name for name, errs in errors.items() if errs}


def test_all_stages_to_valid_result_set(env):
    work, args = _stages(env)
    assert (work / "assets_sensitivity.csv").is_file()  # written by assets (#9, D-028)
    assert export_run.run(args) == 0
    _, short, _ = export_run.git_commit()
    dest = env / f"{CHIP}_5000_{short}"
    assert _failing(validate_run(dest)) == set()  # 12/12
    for name in (F.FLOOD_PROBABILITY, F.FLOOD_MASK, F.ASSET_LOSS, F.STRATEGY_PLANS):
        assert hashlib.sha256((dest / name).read_bytes()).digest() == \
            hashlib.sha256((work / name).read_bytes()).digest()  # copied, not recomputed
    meta = json.loads((dest / F.RUN_METADATA).read_text(encoding="utf-8"))
    pred = json.loads((work / "prediction.json").read_text(encoding="utf-8"))
    assert meta["run_id"] == dest.name and meta["chip_id"] == CHIP
    assert meta["threshold"] == pred["threshold"] == meta["methods"]["main"]["threshold"]
    assert meta["methods"]["baseline"]["threshold_units"] == "dB VH"
    assert meta["budget"]["budget_rub"] == 5000.0
    assert meta["event_date"] == "2017-05-30" and meta["event_id"] == "Fixture"
    assert meta["split"]["chip_split"] == "none"
    assert meta["proxy_check"]["status"] == "planned"
    assets_meta = json.loads((work / "assets_metadata.json").read_text(encoding="utf-8"))
    assert meta["expected_loss"]["inspection_priority"] == assets_meta["inspection_priority"]
    assert any("D-009" in w for w in meta["warnings"])
    manifest = json.loads((dest / F.SOURCE_MANIFEST).read_text(encoding="utf-8"))
    assert f"{CHIP}_S1Hand.tif" in [s.get("file") for s in manifest["sources"]]
    assert "Other_1_S1Hand.tif" not in [s.get("file") for s in manifest["sources"]]
    with open(dest / F.SENSITIVITY, encoding="utf-8") as f:
        sens = list(csv.DictReader(f))
    ids = {r["scenario_id"] for r in sens}
    assert {"p_portfolio_x0.5", "V_portfolio_x0.5", "q_portfolio_x0.5"} <= ids
    assert "p_portfolio_x0.5__C_fixed" in ids
    assert not [w for w in meta["warnings"] if "план C" in w]
    with open(dest / "metrics.csv", encoding="utf-8") as f:
        metrics = list(csv.DictReader(f))
    assert {r["method"] for r in metrics} == {"baseline", "main"}


def test_export_of_an_older_assets_run_without_inspection_priority(env):
    # assets_metadata.json from before D-035 has no inspection_priority: export still works.
    work, args = _stages(env)
    meta_path = work / "assets_metadata.json"
    assets_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    del assets_meta["inspection_priority"]
    meta_path.write_text(json.dumps(assets_meta, ensure_ascii=False), encoding="utf-8")
    assert export_run.run(args) == 0
    _, short, _ = export_run.git_commit()
    dest = env / f"{CHIP}_5000_{short}"
    assert _failing(validate_run(dest)) == set()
    meta = json.loads((dest / F.RUN_METADATA).read_text(encoding="utf-8"))
    assert meta["expected_loss"]["inspection_priority"] is None


def test_export_carries_calibrator_into_run_metadata_and_drops_uncalibrated_warning(env):
    # #62: once predict applies models/unet/calibration.json, methods.main must expose the
    # calibrator (kind, a, b, threshold, sha256, fit_on) and the metrics.csv threshold must
    # match the mask threshold; the "не калиброваны" warning must not fire.
    work, args = _stages(env)
    cal_path = MODELS_DIR / "unet" / "calibration.json"
    cal = json.loads(cal_path.read_text(encoding="utf-8"))
    pred = json.loads((work / "prediction.json").read_text(encoding="utf-8"))
    pred.update(method="unet", threshold=cal["threshold"], threshold_status=cal["threshold_status"],
               calibration={"file": "models/unet/calibration.json",
                            "sha256": export_run._sha256(cal_path), "kind": "platt"})
    (work / "prediction.json").write_text(json.dumps(pred, ensure_ascii=False), encoding="utf-8")
    assert export_run.run(args) == 0
    _, short, _ = export_run.git_commit()
    dest = env / f"{CHIP}_5000_{short}"
    meta = json.loads((dest / F.RUN_METADATA).read_text(encoding="utf-8"))
    main = meta["methods"]["main"]
    assert main["threshold"] == meta["threshold"] == pytest.approx(0.37)
    calibration = main["calibration"]
    assert calibration["kind"] == "platt"
    assert calibration["a"] == pytest.approx(cal["calibrator"]["a"])
    assert calibration["b"] == pytest.approx(cal["calibrator"]["b"])
    assert calibration["threshold"] == pytest.approx(0.37)
    assert calibration["sha256"] == export_run._sha256(cal_path)
    assert calibration["fit_on"] == "validation"
    assert not [w for w in meta["warnings"] if "не калиброваны" in w]
    with open(dest / "metrics.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    main_val_all = next(r for r in rows if r["method"] == "main" and r["split"] == "validation"
                        and r["event"] == "ALL")
    assert float(main_val_all["threshold"]) == meta["threshold"]  # metrics threshold = mask


def test_without_assets_sensitivity_only_sensitivity_fails(env, capsys):
    work, args = _stages(env)
    (work / "assets_sensitivity.csv").unlink()
    assert export_run.run(args) == 0
    assert "assets_sensitivity.csv" in capsys.readouterr().out
    _, short, _ = export_run.git_commit()
    dest = env / f"{CHIP}_5000_{short}"
    assert _failing(validate_run(dest)) == {F.SENSITIVITY}
    assert not (dest / "assets_sensitivity.csv").exists()


def test_new_budget_is_a_new_set_with_the_same_rasters(env):
    _, args = _stages(env, 5000.0)
    assert export_run.run(args) == 0
    _, args2 = _stages(env, 20000.0)
    assert export_run.run(args2) == 0
    _, short, _ = export_run.git_commit()
    a, b = env / f"{CHIP}_5000_{short}", env / f"{CHIP}_20000_{short}"
    for name in (F.FLOOD_PROBABILITY, F.FLOOD_MASK, F.ASSET_LOSS):
        assert (a / name).read_bytes() == (b / name).read_bytes()
    ma = json.loads((a / F.RUN_METADATA).read_text(encoding="utf-8"))
    mb = json.loads((b / F.RUN_METADATA).read_text(encoding="utf-8"))
    assert ma["threshold"] == mb["threshold"]
    assert (ma["budget"]["budget_rub"], mb["budget"]["budget_rub"]) == (5000.0, 20000.0)


def test_budget_mismatch_with_procure_is_refused(env, capsys):
    _, _ = _stages(env, 5000.0)
    assert export_run.run(argparse.Namespace(chip=CHIP, budget=7000.0, out=None)) == 1
    assert "make procure BUDGET=7000" in capsys.readouterr().out


def test_missing_stage_files_are_named(env, capsys):
    assert export_run.run(argparse.Namespace(chip=CHIP, budget=None, out=None)) == 1
    assert "flood_probability.tif" in capsys.readouterr().out


def test_in_place_export_when_stages_used_out(env):
    work = env / "custom"
    baseline_predict.predict_chip(CHIP, work)
    args = argparse.Namespace(chip=CHIP, budget=5000.0, out=str(work))
    assert assets_run.run(args) == 0 and procure_run.run(args) == 0
    assert export_run.run(args) == 0
    assert _failing(validate_run(work)) == set()
