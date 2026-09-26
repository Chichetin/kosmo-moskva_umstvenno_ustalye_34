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
from floodvalue.contracts import load_run_config
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


def test_reproduce_follows_makefile_and_has_no_stale_fallback():
    out = "${FLOODVALUE_OUTPUT_DIR:-outputs}/Sri-Lanka_713926_5000_abc123"
    cmds = assemble.reproduce_commands("abc123", "Sri-Lanka_713926", "5000", out, "unet")
    text = " ".join(cmds)
    assert cmds[0] == "git checkout abc123"
    assert "FLOODVALUE_METHOD=unet make all CHIP=Sri-Lanka_713926 BUDGET=5000" in cmds
    assert "FLOODVALUE_METHOD=unet make predict CHIP=Sri-Lanka_713926" in text
    assert "--mask-only" in text  # baseline_mask.tif, as in make all (#115)
    assert "пока нет" not in text and "floodvalue.model.predict" not in text
    # the legacy D-009 method name is not an env value: no FLOODVALUE_METHOD then
    legacy = assemble.reproduce_commands("abc", "X_1", "5000", "o",
                                         "baseline_sigmoid (temporary main method, D-009)")
    assert "make all CHIP=X_1 BUDGET=5000" in legacy
    assert "FLOODVALUE_METHOD" not in " ".join(legacy)


def test_perturbed_el_is_rounded_half_up_to_the_kopeck_like_assets():
    # #127: the scenario EL must equal what assets writes (exact decimal p*V*q, ROUND_HALF_UP);
    # the float product 28481.58499.. would give 28481.58 and break the ΣEL match of C_fixed rows
    assets = {"A01": {"asset_value_rub": "8000000", "vulnerability_coef": "0.2"}}
    loss = {"A01": {"p_flood": "0.017800990625"}}
    el = assemble.perturbed_el(assets, loss, {"p_flood": {"factor": 1.0}, "scope": "portfolio"})
    assert el == {"A01": 28481.59}


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


S1_TOP = {"citation": "Bonafilia D. et al. Sen1Floods11. CVPRW 2020",
          "license": "не указана явно", "license_note": "Q-01: цитировать Bonafilia et al. 2020",
          "events_source": "https://example.org/Sen1Floods11_Metadata.geojson",
          "events": [{"event_id": "Fixture", "s1_date": "2017-05-30", "s2_date": "2017-05-31"}]}


def _s1_manifest(chip, top=True):
    return {"dataset": "Sen1Floods11 v1.1", "repository": "https://example.org/repo",
            **(S1_TOP if top else {}),
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
    m = assemble.source_manifest(_s1_manifest(CHIP), CHIP, label_read=False, extra_sources=[],
                                 chip_part="none")
    assert [s["chip_id"] for s in m["sources"]] == [CHIP]
    s = m["sources"][0]
    assert s["label"]["used_by_method"] is False and s["applies_to"] == CHIP
    assert s["after_event_note"]  # published after the event: the note is explicit
    with pytest.raises(ValueError):
        assemble.source_manifest(_s1_manifest(CHIP), "Missing_1", False, [], chip_part="none")


def test_source_manifest_carries_citation_license_note_and_events():
    # #67 / Q-01: the dataset citation and the events table of the data stage reach the set
    m = assemble.source_manifest(_s1_manifest(CHIP), CHIP, label_read=False, extra_sources=[],
                                 chip_part="validation")
    for key, value in S1_TOP.items():
        assert m[key] == value
    assert assemble.missing_s1_manifest_keys(_s1_manifest(CHIP)) == []


def test_source_manifest_without_top_keys_does_not_fail_and_names_them():
    old = _s1_manifest(CHIP, top=False)
    m = assemble.source_manifest(old, CHIP, label_read=False, extra_sources=[],
                                 chip_part="validation")
    assert "citation" not in m and m["sources"][0]["chip_id"] == CHIP
    assert assemble.missing_s1_manifest_keys(old) == sorted(S1_TOP)


def _read_note(part):
    m = assemble.source_manifest(_s1_manifest(CHIP), CHIP, label_read=False, extra_sources=[],
                                 chip_part=part)
    return m["sources"][0]["label"]["read_note"]


def test_read_note_of_a_test_chip_says_the_label_is_sealed():
    # D-031: QC of a test chip reads only the header and the hash, never the label pixels
    note = _read_note("test")
    assert "формат и значения" not in note
    assert "заголовок и хеш" in note and "eval-test" in note and "D-031" in note
    assert "калибр" in note  # says explicitly: not in calibration / threshold either
    assert "разведке до разбиения" in note  # D-031: no claim that test was never looked at


def test_read_note_of_a_validation_chip_names_calibration_and_threshold():
    # D-029: calibrator and threshold of the main method are fitted on validation labels
    note = _read_note("validation")
    assert "формат и значения" in note and "D-029" in note and "порог" in note
    assert "eval-test" not in note
    assert "только по S1" in note


def test_read_note_differs_by_split_part():
    notes = {p: _read_note(p) for p in ("train", "dev", "validation", "test", "none")}
    assert len(set(notes.values())) == 5
    assert "D-021" in notes["train"] and "D-021" in notes["dev"]
    for part, note in notes.items():
        assert "predict, assets, procure" in note, part


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
    for name in (F.FLOOD_PROBABILITY, F.FLOOD_MASK, F.ASSET_LOSS, F.STRATEGY_PLANS,
                 "baseline_mask.tif"):
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
    # #127: reproducible commands without the local path, seed from configs/run.yaml
    assert f"make validate OUT=${{FLOODVALUE_OUTPUT_DIR:-outputs}}/{dest.name}" in meta["reproduce"]
    assert not any(str(env) in c for c in meta["reproduce"])  # no machine-specific path
    assert meta["portfolio"]["seed_source"] == "configs/run.yaml: portfolio_seed"
    assert meta["portfolio"]["seed"] == load_run_config().portfolio_seed
    assert "not yet available" not in json.dumps(meta, ensure_ascii=False)
    assets_meta = json.loads((work / "assets_metadata.json").read_text(encoding="utf-8"))
    assert meta["expected_loss"]["inspection_priority"] == assets_meta["inspection_priority"]
    assert any("D-009" in w for w in meta["warnings"])
    manifest = json.loads((dest / F.SOURCE_MANIFEST).read_text(encoding="utf-8"))
    assert f"{CHIP}_S1Hand.tif" in [s.get("file") for s in manifest["sources"]]
    assert "Other_1_S1Hand.tif" not in [s.get("file") for s in manifest["sources"]]
    assert manifest["citation"] == S1_TOP["citation"] and manifest["events"] == S1_TOP["events"]
    s1 = next(s for s in manifest["sources"] if s.get("chip_id") == CHIP)
    assert s1["label"]["read_note"] == _read_note("none")  # fixture chip is outside the split
    # D-043: the decision is taken on the S1 date of the chip, never before the observation
    assert meta["budget"]["decision_at"][:10] == s1["observation_date"] == meta["event_date"]
    assert meta["budget"]["decision_deadline"] > meta["budget"]["decision_at"]
    assert not [w for w in meta["warnings"] if "source_manifest_s1.json" in w]
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


def test_old_s1_manifest_without_citation_exports_with_a_warning(env, capsys):
    (env.parent / "data" / "interim" / "source_manifest_s1.json").write_text(
        json.dumps(_s1_manifest(CHIP, top=False)), encoding="utf-8")
    _, args = _stages(env)
    assert export_run.run(args) == 0
    assert "citation" in capsys.readouterr().out
    _, short, _ = export_run.git_commit()
    dest = env / f"{CHIP}_5000_{short}"
    meta = json.loads((dest / F.RUN_METADATA).read_text(encoding="utf-8"))
    warn = [w for w in meta["warnings"] if "source_manifest_s1.json" in w]
    assert len(warn) == 1 and "citation" in warn[0] and "make data" in warn[0]
    assert F.SOURCE_MANIFEST not in _failing(validate_run(dest))


def test_export_takes_read_note_from_the_split_part_of_the_chip(env, monkeypatch):
    # reviewer #88: run.py must pass the chip's own split part, not a constant
    _, args = _stages(env)
    real = assemble.chip_split
    monkeypatch.setattr(assemble, "chip_split",
                        lambda split, chip: "test" if chip == CHIP else real(split, chip))
    assert export_run.run(args) == 0
    _, short, _ = export_run.git_commit()
    manifest = json.loads((env / f"{CHIP}_5000_{short}" / F.SOURCE_MANIFEST)
                          .read_text(encoding="utf-8"))
    s1 = next(s for s in manifest["sources"] if s.get("chip_id") == CHIP)
    assert s1["label"]["read_note"] == _read_note("test")


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


def test_set_without_baseline_mask_is_valid_but_warns(env):
    """#115: baseline_mask.tif is optional — absent in the work dir -> absent in the set, 12/12,
    a warning names the command that writes it; a stale copy from an earlier export is removed."""
    work, args = _stages(env)
    assert export_run.run(args) == 0
    _, short, _ = export_run.git_commit()
    dest = env / f"{CHIP}_5000_{short}"
    assert (dest / "baseline_mask.tif").is_file()
    meta = json.loads((dest / F.RUN_METADATA).read_text(encoding="utf-8"))
    assert not [w for w in meta["warnings"] if "baseline_mask" in w]
    (work / "baseline_mask.tif").unlink()
    assert export_run.run(args) == 0
    assert not (dest / "baseline_mask.tif").exists()
    assert _failing(validate_run(dest)) == set()
    meta = json.loads((dest / F.RUN_METADATA).read_text(encoding="utf-8"))
    assert [w for w in meta["warnings"] if "baseline_mask.tif" in w and "--mask-only" in w]


def _prewater_files(work, status="ok"):
    """What `python -m floodvalue prewater` leaves in the work dir (#149), written by hand:
    export must handle it as files only, without importing floodvalue.prewater."""
    source = {"id": f"pc:sentinel-1-rtc:{CHIP}", "file": "pre_event_water.tif",
              "url": "https://planetarycomputer.microsoft.com/api/stac/v1/collections/sentinel-1-rtc",
              "version": "Planetary Computer STAC sentinel-1-rtc, 6 сцен(ы)",
              "observation_date": "2017-03-28", "available_date": "2026-09-26",
              "license": "CC-BY-4.0", "purpose": "вода до события", "limitations": "окно до события",
              "after_event_note": "RTC обработан позже по наблюдениям до события"}
    meta = {"chip_id": CHIP, "status": status, "warning": None if status == "ok" else "HTTP 429",
            "sources": [source] if status == "ok" else []}
    (work / "pre_event_water.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    if status == "ok":
        with rasterio.open(work / "flood_mask.tif") as r:
            profile = r.profile
        with rasterio.open(work / "pre_event_water.tif", "w", **profile) as dst:
            dst.write(np.zeros((1, H, W), dtype="uint8"))
    return source


def test_prewater_layer_is_optional_and_changes_nothing_of_the_12_files(env):
    """#149 p.4: the pre-event layer only adds optional files and its sources; p, mask,
    threshold, assets, EL and the plan stay byte-for-byte the same."""
    work, args = _stages(env)
    assert export_run.run(args) == 0
    _, short, _ = export_run.git_commit()
    dest = env / f"{CHIP}_5000_{short}"
    unchanged = (F.FLOOD_PROBABILITY, F.FLOOD_MASK, F.ASSETS_GEOJSON, F.ASSETS_CSV, F.ASSET_LOSS,
                 F.CANDIDATE_ORDERS, F.PROCUREMENT_PLAN, F.STRATEGY_PLANS,
                 F.STRATEGY_COMPARISON, F.SENSITIVITY, "metrics.csv")
    before = {n: (dest / n).read_bytes() for n in unchanged}
    meta_before = json.loads((dest / F.RUN_METADATA).read_text(encoding="utf-8"))

    source = _prewater_files(work)
    assert export_run.run(args) == 0
    assert {n: (dest / n).read_bytes() for n in unchanged} == before
    assert (dest / "pre_event_water.tif").is_file() and (dest / "pre_event_water.json").is_file()
    meta = json.loads((dest / F.RUN_METADATA).read_text(encoding="utf-8"))
    assert meta["threshold"] == meta_before["threshold"]
    assert meta["expected_loss"] == meta_before["expected_loss"]
    manifest = json.loads((dest / F.SOURCE_MANIFEST).read_text(encoding="utf-8"))
    assert source in manifest["sources"]
    assert _failing(validate_run(dest)) == set()  # still 12/12


def test_prewater_not_built_is_a_warning_without_sources(env):
    work, args = _stages(env)
    _prewater_files(work, status="unavailable")
    assert export_run.run(args) == 0
    _, short, _ = export_run.git_commit()
    dest = env / f"{CHIP}_5000_{short}"
    meta = json.loads((dest / F.RUN_METADATA).read_text(encoding="utf-8"))
    assert [w for w in meta["warnings"] if "pre_event_water.json" in w and "unavailable" in w]
    manifest = json.loads((dest / F.SOURCE_MANIFEST).read_text(encoding="utf-8"))
    assert not [s for s in manifest["sources"] if "sentinel-1-rtc" in str(s.get("id"))]
    assert not (dest / "pre_event_water.tif").exists()
    assert _failing(validate_run(dest)) == set()


def test_decision_risk_block_reaches_run_metadata_and_columns_validate(env):
    """#142 (D-049): the decision-risk block of procurement_metadata.json is carried into
    run_metadata.strategies; the real procurement columns pass the validator's recomputation."""
    work, args = _stages(env)
    assert export_run.run(args) == 0
    _, short, _ = export_run.git_commit()
    dest = env / f"{CHIP}_5000_{short}"
    meta = json.loads((dest / F.RUN_METADATA).read_text(encoding="utf-8"))
    proc = json.loads((work / "procurement_metadata.json").read_text(encoding="utf-8"))
    risk = meta["strategies"]["decision_risk"]
    assert risk == proc["decision_risk"]
    assert "min(p, 1−p)" in risk["formula"] and "не фактическое снижение потерь" in risk["meaning"]
    assert risk["status"] == "scenario"
    with open(dest / F.STRATEGY_COMPARISON, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert all(r["addressed_decision_risk_rub"] != "" for r in rows)
    assert _failing(validate_run(dest)) == set()


# --- #260: complete and single-chip result set ---------------------------------------------------

def _raster_like(work, name, count=1, shift=0.0):
    """A float32 raster on the grid of flood_probability.tif, optionally shifted in longitude
    (a stale file of another chip)."""
    with rasterio.open(work / F.FLOOD_PROBABILITY) as r:
        profile = r.profile
    t = profile["transform"]
    profile.update(count=count, transform=rasterio.Affine(t.a, t.b, t.c + shift, t.d, t.e, t.f))
    with rasterio.open(work / name, "w", **profile) as dst:
        dst.write(np.full((count, H, W), 0.5, dtype="float32"))


def _meta(dest):
    return json.loads((dest / F.RUN_METADATA).read_text(encoding="utf-8"))


def test_probability_range_raster_is_part_of_the_set(env):
    # #260 p.1: prediction.json.range_file points at it — it must travel with the set
    work, args = _stages(env)
    _raster_like(work, "flood_probability_range.tif", count=2)
    assert export_run.run(args) == 0
    _, short, _ = export_run.git_commit()
    dest = env / f"{CHIP}_5000_{short}"
    assert (dest / "flood_probability_range.tif").read_bytes() == \
        (work / "flood_probability_range.tif").read_bytes()
    (work / "flood_probability_range.tif").unlink()
    assert export_run.run(args) == 0
    assert not (dest / "flood_probability_range.tif").exists()  # stale copy removed


def test_procurement_of_another_chip_is_refused(env, capsys):
    # #260 p.2: predict/assets of one chip + procure of another -> no mixed set
    work, args = _stages(env)
    path = work / "procurement_metadata.json"
    proc = json.loads(path.read_text(encoding="utf-8"))
    proc["chip_id"] = "Other_000002"
    path.write_text(json.dumps(proc, ensure_ascii=False), encoding="utf-8")
    assert export_run.run(args) == 1
    out = capsys.readouterr().out
    assert "procurement_metadata.json" in out and "Other_000002" in out


def test_off_grid_extra_raster_is_dropped_from_the_set_with_a_warning(env):
    # #260 p.3: an extra raster of another grid (stale, another chip) never enters the set
    work, args = _stages(env)
    _raster_like(work, "uncertainty.tif", shift=0.5)
    assert export_run.run(args) == 0
    _, short, _ = export_run.git_commit()
    dest = env / f"{CHIP}_5000_{short}"
    assert not (dest / "uncertainty.tif").exists()
    assert (work / "uncertainty.tif").exists()  # the work dir of the stages is not touched
    meta = _meta(dest)
    assert [w for w in meta["warnings"] if "uncertainty.tif" in w and "сетк" in w
            and "make predict" in w]
    assert meta["uncertainty"]["raster"] is None
    assert _failing(validate_run(dest)) == set()


def test_in_place_export_removes_stale_raster_of_another_chip(env):
    # #260 p.3, dest == work (--out): a baseline_mask.tif of another chip's grid is removed
    work = env / "custom"
    baseline_predict.predict_chip(CHIP, work)
    args = argparse.Namespace(chip=CHIP, budget=5000.0, out=str(work))
    assert assets_run.run(args) == 0 and procure_run.run(args) == 0
    _raster_like(work, "baseline_mask.tif", shift=-1.0)
    _raster_like(work, "flood_probability_range.tif", count=2)  # same grid: stays
    assert export_run.run(args) == 0
    assert not (work / "baseline_mask.tif").exists()
    assert (work / "flood_probability_range.tif").exists()
    assert [w for w in _meta(work)["warnings"] if "baseline_mask.tif" in w and "сетк" in w]
    assert _failing(validate_run(work)) == set()


@pytest.mark.parametrize("in_place", [False, True])
def test_prewater_rasters_not_in_the_set_when_layer_not_built(env, in_place):
    # #260 p.3: status != ok -> pre-event rasters (leftovers of an earlier run) stay out
    if in_place:
        work = env / "custom"
        baseline_predict.predict_chip(CHIP, work)
        args = argparse.Namespace(chip=CHIP, budget=5000.0, out=str(work))
        assert assets_run.run(args) == 0 and procure_run.run(args) == 0
    else:
        work, args = _stages(env)
    _prewater_files(work, status="unavailable")
    _raster_like(work, "pre_event_water.tif")
    _raster_like(work, "new_water_probability.tif")
    assert export_run.run(args) == 0
    _, short, _ = export_run.git_commit()
    dest = work if in_place else env / f"{CHIP}_5000_{short}"
    assert not (dest / "pre_event_water.tif").exists()
    assert not (dest / "new_water_probability.tif").exists()
    assert (dest / "pre_event_water.json").exists()  # documents that the layer is unavailable
    assert _failing(validate_run(dest)) == set()


def test_prewater_of_another_chip_adds_no_sources_and_no_files(env):
    work, args = _stages(env)
    _prewater_files(work)
    meta_path = work / "pre_event_water.json"
    pw = json.loads(meta_path.read_text(encoding="utf-8"))
    pw["chip_id"] = "Other_000002"
    meta_path.write_text(json.dumps(pw, ensure_ascii=False), encoding="utf-8")
    assert export_run.run(args) == 0
    _, short, _ = export_run.git_commit()
    dest = env / f"{CHIP}_5000_{short}"
    manifest = json.loads((dest / F.SOURCE_MANIFEST).read_text(encoding="utf-8"))
    assert not [s for s in manifest["sources"] if "sentinel-1-rtc" in str(s.get("id"))]
    assert not (dest / "pre_event_water.json").exists()
    assert not (dest / "pre_event_water.tif").exists()
    assert [w for w in _meta(dest)["warnings"] if "pre_event_water.json" in w
            and "Other_000002" in w]


def test_commit_dirty_sees_untracked_but_not_ignored_files(tmp_path):
    # #260 p.4: an untracked src/ file changes the run; ignored data/outputs do not
    import subprocess

    def git(*a):
        subprocess.run(["git", "-C", str(tmp_path), *a], check=True, capture_output=True)
    git("init", "-q")
    git("config", "user.email", "t@t")
    git("config", "user.name", "t")
    (tmp_path / ".gitignore").write_text("/outputs/\n*.tif\n", encoding="utf-8")
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    git("add", ".gitignore", "a.py")
    git("commit", "-q", "-m", "init")
    (tmp_path / "outputs").mkdir()
    (tmp_path / "outputs" / "run.json").write_text("{}", encoding="utf-8")
    (tmp_path / "p.tif").write_bytes(b"x")
    assert export_run.git_commit(tmp_path)[2] is False
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "new.py").write_text("y = 2\n", encoding="utf-8")
    assert export_run.git_commit(tmp_path)[2] is True


def test_prewater_rasters_without_their_metadata_stay_out(env):
    # #260 p.3: without pre_event_water.json the layer status is unknown -> not in the set
    work, args = _stages(env)
    _raster_like(work, "pre_event_water.tif")
    assert export_run.run(args) == 0
    _, short, _ = export_run.git_commit()
    dest = env / f"{CHIP}_5000_{short}"
    assert not (dest / "pre_event_water.tif").exists()
    assert [w for w in _meta(dest)["warnings"] if "pre_event_water.tif" in w
            and "pre_event_water.json" in w]
