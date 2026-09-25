"""Pure helpers of the export stage: run_id, split hash, metrics.csv, sensitivity.csv,
source_manifest.json and run_metadata.json from the files of the earlier stages (D-027).

No file I/O here: `floodvalue.export.run` reads the stage outputs and passes plain dicts.
"""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from typing import Any

from floodvalue.contracts import files as F

SPLIT_PARTS = ("train", "dev", "validation", "test")
METRICS_FIELDS = ["method", "split", "event", "iou", "f1", "precision", "recall", "tp", "fp",
                  "fn", "tn", "n_pixels", "threshold", "valid_mask_sha256"]
METRICS_EXTRA = ["threshold_units", "method_name", "source_file"]
METRICS_SPLITS = ("dev", "validation", "test")  # contracts §7; train rows stay in verification
DATASET = "Sen1Floods11 v1.1 HandLabeled"
DATASET_URL = "https://github.com/cloudtostreet/Sen1Floods11"


def budget_label(budget_rub: float) -> str:
    """5000 -> '5000', 2500.5 -> '2500.5' (no exponent, no trailing zeros)."""
    d = Decimal(str(budget_rub)).normalize()
    return format(d.quantize(Decimal(1)) if d == d.to_integral_value() else d, "f")


def make_run_id(chip_id: str, budget_rub: float, short_commit: str) -> str:
    """contracts §1: <chip_id>_<budget>_<short commit>."""
    return f"{chip_id}_{budget_label(budget_rub)}_{short_commit}"


def split_sha256(split: dict) -> str:
    """contracts §4 / D-016: sha256 of the canonical JSON of the four part lists."""
    canon = json.dumps({k: split[k] for k in SPLIT_PARTS}, sort_keys=True,
                       separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canon.encode("ascii")).hexdigest()


def chip_split(split: dict, chip_id: str) -> str:
    """Part of the split that holds the chip ('none' if the chip is not in the split)."""
    return next((p for p in SPLIT_PARTS if chip_id in split.get(p, [])), "none")


def is_baseline_prediction(prediction: dict) -> bool:
    """True when the p-raster came from the baseline sigmoid (temporary main method, D-009)."""
    return "baseline" in str(prediction.get("method", "")).lower()


def metrics_rows(baseline_rows: list[dict], main_rows: list[dict] | None,
                 prediction: dict, baseline_file: str, main_file: str | None) -> list[dict]:
    """metrics.csv (contracts §7): baseline and main on the same valid-pixel masks.

    `threshold` keeps each method's own units (baseline: dB of its channel, main: p) and the
    extra column `threshold_units` names them. While the main method is the baseline sigmoid
    (D-009), p >= 0.5 <=> VH <= t, so the main rows are the baseline confusion counts with the
    main threshold in p; a real main method must bring its own rows (main_rows).
    """
    base = [r for r in baseline_rows if r.get("method") == "baseline"
            and r.get("split") in METRICS_SPLITS]
    channel = prediction.get("channel", "VH")
    out = [{**{k: r[k] for k in METRICS_FIELDS}, "threshold_units": f"dB {channel}",
            "method_name": "threshold_baseline", "source_file": baseline_file} for r in base]
    if main_rows and not is_baseline_prediction(prediction):
        for r in main_rows:
            if r.get("method") == "main" and r.get("split") in METRICS_SPLITS:
                out.append({**{k: r[k] for k in METRICS_FIELDS},
                            "threshold_units": r.get("threshold_units") or "p",
                            "method_name": str(prediction.get("method", "main")),
                            "source_file": main_file})
    elif is_baseline_prediction(prediction):
        for r in base:
            out.append({**{k: r[k] for k in METRICS_FIELDS}, "method": "main",
                        "threshold": str(prediction["threshold"]), "threshold_units": "p",
                        "method_name": "baseline_sigmoid (temporary main method, D-009)",
                        "source_file": baseline_file})
    return out


def merge_sensitivity(procurement_rows: list[dict], assets_rows: list[dict] | None) -> list[dict]:
    """sensitivity.csv = procurement (budget, rate) ∪ assets (p, V, q) rows, in that order."""
    return [*procurement_rows, *(assets_rows or [])]


def sensitivity_fields(rows: list[dict]) -> list[str]:
    fields = list(F.SENSITIVITY_FIELDS)
    for r in rows:
        fields += [k for k in r if k not in fields]
    return fields


def missing_sensitivity_keys(rows: list[dict]) -> list[str]:
    """Required one-at-a-time variations (contracts §7a) absent from the merged rows."""
    required = set(F.SENSITIVITY_REQUIRED_KEYS)
    seen: set[str] = set()
    for r in rows:
        try:
            changed = set(json.loads(r["changed_inputs_json"])) & required
        except (ValueError, TypeError, KeyError):
            continue
        if len(changed) == 1:
            seen |= changed
    return sorted(required - seen)


def source_manifest(s1_manifest: dict, chip_id: str, label_read: bool,
                    extra_sources: list[dict]) -> dict:
    """source_manifest.json (contracts §9): the S1 entry of this chip as written by `data`
    (D-015), marked whether its label was read by this run, plus the other inputs used."""
    s1 = [e for e in s1_manifest.get("sources", []) if e.get("chip_id") == chip_id
          or f"{chip_id}_S1Hand" in str(e.get("source_id", "")) + str(e.get("file", ""))]
    if not s1:
        raise ValueError(f"в source_manifest_s1.json нет записи S1 для {chip_id}")
    entries = []
    for e in s1:
        e = dict(e)
        e.setdefault("chip_id", chip_id)
        e.setdefault("applies_to", chip_id)
        if "label" in e:
            e["label"] = {**e["label"], "used_by_method": label_read,
                          "read_note": ("метку читает только QC этапа data (формат и значения) "
                                        "и расчёт метрик по split в docs/verification; в predict, "
                                        "assets, procure и в выбор порога она не входит — метод "
                                        "работает только по S1")
                          if not label_read else "метка использована методом в этом запуске"}
        if e.get("published_after_event") and not e.get("after_event_note"):
            e["after_event_note"] = e.get("limitations", "")
        entries.append(e)
    return {"dataset": s1_manifest.get("dataset", DATASET),
            "repository": s1_manifest.get("repository", DATASET_URL),
            "sources": entries + list(extra_sources)}


def config_sources(pricing: dict, event_date: str) -> list[dict]:
    """Non-imagery inputs of the run: the case tables for V/q and the price formula."""
    return [
        {"id": "case:assets-table", "file": "configs/assets.yaml",
         "url": "task.md, раздел I «Типы и характеристики 10 объектов»",
         "version": "материалы кейса КосмоХакатон 2026", "observation_date": None,
         "available_date": "2026-09-25", "license": "материалы организатора кейса",
         "purpose": "стоимость V и уязвимость q 10 типов объектов",
         "limitations": "учебные значения кейса, не оценка реальных активов",
         "applies_to": "assets.geojson, assets.csv, asset_loss.csv",
         "after_event_note": f"справочник, не наблюдение; событие {event_date}"},
        {"id": "pp840:pricing", "file": "configs/pricing.yaml",
         "url": "http://publication.pravo.gov.ru/ (ПП РФ № 840 от 29.06.2019)",
         "version": str(pricing.get("legal_edition", "")),
         "observation_date": None,
         "available_date": str(pricing.get("case_reference_date") or "2026-09-20"),
         "license": "нормативный акт РФ (общедоступен)",
         "purpose": "формула цены дополнительных данных РП = Б × К × О × П × Т × Р",
         "limitations": ("Б = сценарная учебная ставка кейса (base_rate_status=scenario), "
                         "не подтверждённый тариф"),
         "applies_to": "candidate_orders.geojson, procurement_plan.csv, strategy_comparison.csv",
         "after_event_note": ("норма в редакции 2025 г. применяется к имитации решения; это "
                              "правило цены, а не сведения о событии")},
    ]


def _method_info(name: str, version: str, weights: list[dict], threshold: float, units: str,
                 source: str, **extra: Any) -> dict:
    return {"name": name, "version": version, "weights": weights, "threshold": threshold,
            "threshold_units": units, "threshold_source": source, **extra}


def run_metadata(*, run_id: str, chip_id: str, event_id: str, event_date: str, created_at: str,
                 commit: str, commit_dirty: bool, prediction: dict, baseline_config: dict,
                 baseline_config_path: str, baseline_config_sha256: str,
                 assets_meta: dict, procurement_meta: dict, pricing: dict, split: dict,
                 split_file: str, run_config: dict, uncertainty_raster: str | None,
                 metrics_files: list[str], reproduce: list[str], versions: dict,
                 warnings: list[str]) -> dict:
    """run_metadata.json (contracts §8) from the stage metadata files; no recomputation."""
    baseline_weights = [{"path": baseline_config_path, "sha256": baseline_config_sha256}]
    baseline = _method_info(
        "threshold_baseline", f"{baseline_config.get('method', 'threshold_baseline')} "
        f"{baseline_config.get('channel')} (D-021)", baseline_weights,
        float(baseline_config["threshold_db"]), f"dB {baseline_config.get('channel')}",
        f"max IoU on {baseline_config.get('fitted_on', 'train')}, channel chosen on "
        f"{baseline_config.get('channel_chosen_on', 'dev')} (D-021)",
        rule=baseline_config.get("rule"))
    if is_baseline_prediction(prediction):
        main = _method_info(
            "baseline_sigmoid (temporary main method, D-009)",
            f"sigmoid((threshold_db - {prediction.get('channel', 'VH')}) / scale_db), "
            f"scale_db={prediction.get('scale_db')}", baseline_weights,
            float(prediction["threshold"]), "p",
            str(prediction.get("threshold_status") or "models/baseline/config.json"),
            status="temporary: основной метод ещё не влит (#16/#17), D-009",
            proba=baseline_config.get("proba"))
    else:
        main = _method_info(
            str(prediction.get("method", "main")), str(prediction.get("version", "")) or "n/a",
            list(prediction.get("weights", [])), float(prediction["threshold"]), "p",
            str(prediction.get("threshold_source") or prediction.get("threshold_status")
                or "prediction.json"))
    unc = assets_meta.get("uncertainty", {})
    zones = procurement_meta.get("zones", {})
    split_hash = split_sha256(split)
    return {
        "contract_version": F.CONTRACT_VERSION,
        "run_id": run_id,
        "chip_id": chip_id,
        "event_id": event_id,
        "event_date": event_date,
        "created_at": created_at,
        "commit": commit,
        "commit_dirty": commit_dirty,
        "threshold": float(prediction["threshold"]),
        "source": {"dataset": DATASET, "dataset_url": DATASET_URL,
                   "s1_files": [prediction.get("s1_file") or f"{chip_id}_S1Hand.tif"],
                   "source_manifest": F.SOURCE_MANIFEST,
                   "s1_only": "основной метод и baseline читают только S1 VV/VH (режим S1-only)"},
        "split": {"file": split_file, "sha256": split_hash,
                  "sha256_matches_file": split_hash == split.get("sha256"),
                  "chip_split": chip_split(split, chip_id),
                  "chip_split_run_yaml": run_config.get("chip_split"),
                  "decision": "D-016"},
        "portfolio": {"seed": assets_meta["seed"], "placement_rule": assets_meta["placement_rule"],
                      "assets_config": "configs/assets.yaml",
                      "seed_source": assets_meta.get("seed_source"),
                      "run_yaml_seed": run_config.get("portfolio_seed")},
        "methods": {"baseline": baseline, "main": main,
                    "evaluation_order": (
                        "baseline: порог по train, канал по dev (D-021); основной метод: выбор "
                        "на dev, калибровка и порог на validation (D-012/D-016); оба сравниваются "
                        "на одной маске валидных пикселей (valid_mask_sha256 в metrics.csv); test "
                        "— только make eval-test замороженной конфигурации (D-010)"),
                    "prediction_file": "prediction.json"},
        "expected_loss": {"formula": "EL = p_flood × V × q",
                          "el_model": assets_meta.get("el_model"),
                          "p_source": assets_meta["p_extraction_rule"],
                          "status_rules": assets_meta["status_rule"],
                          "unassessed_asset_ids": assets_meta.get("unassessed_asset_ids", [])},
        "budget": {"budget_rub": float(procurement_meta["budget_rub"]),
                   "decision_at": procurement_meta["decision_at"],
                   "decision_deadline": procurement_meta["decision_deadline"],
                   "source": procurement_meta.get("run_config_source", "configs/run.yaml")},
        "pricing": {"config": "configs/pricing.yaml",
                    "legal_edition": pricing["legal_edition"],
                    "base_rate_rub_km2": float(pricing["base_rate"]["rub_per_km2"]),
                    "base_rate_status": pricing["base_rate"]["status"],
                    "base_rate_source": pricing["base_rate"].get("source"),
                    "formula": pricing.get("formula"),
                    "discount_rule": ("Р = min(1; max(floor; slope × ln(S / r²) + intercept)), "
                                      "S — км² корзины группы + previous_area_km2, r — м; "
                                      "коэффициенты — configs/pricing.yaml discount; Р до 9 "
                                      "знаков, цена ROUND_HALF_UP до копеек (D-017)"),
                    "grouping": ("группа скидки = семейство сенсора × resolution_m × "
                                 f"календарный год ({pricing.get('discount', {}).get('group_by')}); "
                                 "discount_group_id = <семейство>_<r>m_<год> (D-017)"),
                    "as_of": procurement_meta.get("pricing_as_of"),
                    "as_of_rule": procurement_meta.get("pricing_as_of_rule")},
        "zones": {"construction": zones.get("construction", ""),
                  "availability_rule": (
                      f"{zones.get('coverage_rule', '')}; сроки {zones.get('timing_status', '')}: "
                      f"{zones.get('timing_basis', '')}"),
                  "area_method": zones.get("area_method", ""),
                  "unknown_dates_rule": (
                      "observation_at/available_at = null — дата неизвестна: такая зона не "
                      "считается доступной к decision_deadline и не даёт охвата; все даты "
                      "каталога — сценарные (timing_status=scenario, D-023)"),
                  "chip_bounds_source": zones.get("chip_bounds_source"),
                  "excluded_from_coverage": zones.get("excluded_from_coverage", {}),
                  "coverage": zones.get("coverage", {})},
        "strategies": {"rule_B": procurement_meta["rule_B"], "rule_C": procurement_meta["rule_C"],
                       "other_costs": json.dumps(procurement_meta.get("other_cost", {}),
                                                 ensure_ascii=False),
                       "coverage_aggregation": (
                           "covered_expected_loss_rub = Σ EL уникальных asset_id status=ok, "
                           "которые covers хотя бы одна зона стратегии с "
                           "data_role=event_observation и available_at ≤ decision_deadline; "
                           "coverage_share = covered / Σ EL(ok), руб./руб."),
                       "coverage_share_note": procurement_meta.get("coverage_share_empty_reason"),
                       "coverage_share_denominator_rub":
                           procurement_meta.get("coverage_share_denominator_rub"),
                       "uncertainty_basis": procurement_meta.get("uncertainty"),
                       "reasons": procurement_meta.get("reasons")},
        "uncertainty": {"meaning": unc.get("rule", "не задано"),
                        "units": ("биты, 0…1" if unc.get("name") == "binary_entropy_bits"
                                  else str(unc.get("units", "не задано"))),
                        "method": f"{unc.get('name', 'n/a')} ({unc.get('status', 'n/a')})",
                        "raster": uncertainty_raster},
        "proxy_check": {"status": "planned", "test_chip": None, "control_points": None,
                        "fixed_v_q": True, "n_points": None, "error_metric": None,
                        "result": {"estimated": False,
                                   "reason": ("прокси-проверка ущерба не выполнялась: по D-010 "
                                              "метки test читает только eval-test замороженной "
                                              "конфигурации; протокол и расчёт — P24a")}},
        "metrics": {"files": metrics_files},
        "reproduce": reproduce,
        "seeds": {"portfolio_seed": assets_meta["seed"], "split_seed": split.get("seed")},
        "versions": versions,
        "warnings": warnings,
    }
