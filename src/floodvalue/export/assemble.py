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
from floodvalue.contracts.geo import round_half_up

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
            if r.get("method") in ("main", prediction.get("method")) \
                    and r.get("split") in METRICS_SPLITS:
                out.append({**{k: r[k] for k in METRICS_FIELDS}, "method": "main",
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


MODEL_NOTE = ("строка — про модель ущерба (p, V, q), не про корзину: план закупки не "
              "пересчитывается")
PARAM_CLIP = {"p_flood": True, "asset_value_rub": False, "vulnerability_coef": True}


def perturbed_el(assets: dict[str, dict], loss: dict[str, dict], changed: dict) -> dict[str, float]:
    """EL of every `ok` asset under one assets_sensitivity scenario (D-028 rules):
    x' = x * k for the scenario's parameter, clipped to [0, 1] for p and q; `scope` is
    `portfolio` (all ok assets) or `single_asset` (`asset_id`)."""
    param = next(k for k in PARAM_CLIP if k in changed)
    k = float(changed[param]["factor"])
    targets = ({changed["asset_id"]} if changed.get("scope") == "single_asset"
               else set(loss))
    out = {}
    for aid, r in loss.items():
        x = {"p_flood": float(r["p_flood"]), "asset_value_rub": float(assets[aid]["asset_value_rub"]),
             "vulnerability_coef": float(assets[aid]["vulnerability_coef"])}
        if aid in targets:
            x[param] *= k
            if PARAM_CLIP[param]:
                x[param] = min(1.0, max(0.0, x[param]))
        out[aid] = x["p_flood"] * x["asset_value_rub"] * x["vulnerability_coef"]
    return out


def fixed_plan_rows(assets_rows: list[dict], assets: list[dict], loss_rows: list[dict],
                    comparison: list[dict], plan_c: list[str]) -> tuple[list[dict], list[str]]:
    """For each p/V/q scenario of `assets` (strategy A, no plan) add a row for the fixed plan
    C of this set: decision_cost_rub of C and the EL it covers under the changed inputs
    (reviewer #55). Returns (rows, warnings); a scenario whose recomputed portfolio EL does
    not match `portfolio_expected_loss_rub` of assets is skipped with a warning."""
    by_asset = {a["asset_id"]: a for a in assets}
    ok = {r["asset_id"]: r for r in loss_rows if r["status"] == "ok"}
    c = next((r for r in comparison if r["strategy"] == "C"), None)
    if c is None or not c.get("covered_expected_loss_rub"):
        return [], ["strategy_comparison.csv без строки C с охватом: строки плана C не добавлены"]
    covered_ids = set((c.get("covered_asset_ids") or "").split()) & set(ok)
    out, warnings = [], []
    for r in assets_rows:
        try:
            changed = json.loads(r["changed_inputs_json"])
            el = perturbed_el(by_asset, ok, changed)
        except (ValueError, KeyError, StopIteration, TypeError) as exc:
            warnings.append(f"{r.get('scenario_id')}: план C не пересчитан ({exc})")
            continue
        total = changed.get("portfolio_expected_loss_rub")
        if total is not None and abs(sum(el.values()) - float(total)) > 0.01:
            warnings.append(f"{r['scenario_id']}: ΣEL {sum(el.values()):.2f} ≠ assets "
                            f"{float(total):.2f} — строка для плана C не добавлена")
            continue
        covered = round_half_up(sum(el[a] for a in covered_ids), "0.01")
        out.append({
            "scenario_id": f"{r['scenario_id']}__C_fixed", "strategy": "C",
            "changed_inputs_json": json.dumps({**changed, "plan_C_fixed": sorted(plan_c)},
                                              ensure_ascii=False, sort_keys=True),
            "decision_cost_rub": c["decision_cost_rub"],
            "covered_expected_loss_rub": f"{covered:.2f}",
            "result_status": "fixed_plan",
            "interpretation": (
                f"план C зафиксирован ({', '.join(sorted(plan_c)) or 'пусто'}), стоимость решения "
                f"{c['decision_cost_rub']} руб. не меняется; охват EL "
                f"{float(c['covered_expected_loss_rub']):.2f} -> {covered:.2f} руб.; "
                f"{MODEL_NOTE}")})
    return out, warnings


def with_model_note(rows: list[dict]) -> list[dict]:
    return [{**r, "interpretation": f"{r['interpretation']}; {MODEL_NOTE}"}
            if MODEL_NOTE not in r.get("interpretation", "") else r for r in rows]


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


S1_MANIFEST_TOP = ("citation", "license", "license_note", "events_source", "events")
_NOT_INPUT = "во входы predict, assets, procure не подаётся — метод работает только по S1"
_QC_OPEN = "QC этапа data читает её формат и значения (D-031)"
LABEL_READ_NOTES = {
    "test": ("метка чипа test запечатана (D-010, D-016): QC этапа data читает только заголовок "
             "и хеш файла (D-031), пиксели метки не читает ни один этап до заморозки; в "
             "обучение, калибровку и выбор порога не входит; пиксели метки — только make "
             "eval-test для замороженной конфигурации (docs/verification/test-log.md); "
             "оговорка: сводные доли меток всех 446 чипов смотрели на разведке до разбиения "
             "(docs/data.md §5, D-016); " + _NOT_INPUT),
    "validation": (f"{_QC_OPEN}; метки validation — калибровка вероятностей и выбор порога "
                   "основного метода (D-029) и метрики в docs/verification; " + _NOT_INPUT),
    "dev": (f"{_QC_OPEN}; метки dev — выбор канала baseline (D-021), отбор моделей основного "
            "метода и метрики в docs/verification; " + _NOT_INPUT),
    "train": (f"{_QC_OPEN}; метки train — обучение основного метода и порог baseline (D-021); "
              + _NOT_INPUT),
    "none": (f"чип вне train/dev/validation/test: метка не входит ни в обучение, ни в "
             f"калибровку и выбор порога; {_QC_OPEN}, если чип в excluded split; " + _NOT_INPUT),
}


def missing_s1_manifest_keys(s1_manifest: dict) -> list[str]:
    """Top-level keys of interim/source_manifest_s1.json (D-031) absent in an older file."""
    return sorted(k for k in S1_MANIFEST_TOP if k not in s1_manifest)


def source_manifest(s1_manifest: dict, chip_id: str, label_read: bool,
                    extra_sources: list[dict], *, chip_part: str) -> dict:
    """source_manifest.json (contracts §9): the S1 entry of this chip as written by `data`
    (D-015), marked how its label was read given the chip's split part (D-031), the
    citation / licence / events of the data stage (Q-01), plus the other inputs used."""
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
                          "read_note": LABEL_READ_NOTES[chip_part]
                          if not label_read else "метка использована методом в этом запуске"}
        if e.get("published_after_event") and not e.get("after_event_note"):
            e["after_event_note"] = e.get("limitations", "")
        entries.append(e)
    return {"dataset": s1_manifest.get("dataset", DATASET),
            "repository": s1_manifest.get("repository", DATASET_URL),
            **{k: s1_manifest[k] for k in S1_MANIFEST_TOP if k in s1_manifest},
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


def eval_summary_file(part: str) -> str:
    """Repo-relative path of the eval-test summary of a split part (written by eval-test).

    Same prefix as model.eval_test.prefix_of: test -> test-eval, others -> eval-<part>.
    """
    prefix = "test-eval" if part == "test" else f"eval-{part}"
    return f"docs/verification/{prefix}-summary.json"


def proxy_check(summary: dict | None, source: str, chip_id: str, part: str,
                prediction: dict) -> dict:
    """run_metadata.proxy_check (contracts §8) from the eval-<part> summary, #107.

    The numbers are taken only when the summary belongs to the chip's split part, its demo chip
    is this chip and its main method and threshold match prediction.json of the set; otherwise
    the check stays `planned` with the reason. Only test is independent (status `done`); on
    validation/dev the numbers are preliminary (the part tuned the method, D-012/D-016).
    """
    def planned(reason: str) -> dict:
        return {"status": "planned", "test_chip": None, "control_points": None,
                "fixed_v_q": True, "n_points": None, "error_metric": None,
                "result": {"estimated": False, "reason": reason}}

    if summary is None:
        where = f" (нет {source})" if source else ""
        return planned("прокси-проверка ущерба для этого чипа не выполнялась" + where + ": её "
                       "считает eval-test той части split, где лежит чип; по D-010 метки test "
                       "читает только eval-test замороженной конфигурации")
    pc = summary.get("proxy_check") or {}
    s_chip = pc.get("test_chip")
    if s_chip != chip_id:
        return planned(f"сводка {source} посчитана на другой чип ({s_chip}), не на {chip_id}")
    s_method, s_thr = summary.get("main_method"), summary.get("threshold")
    p_method, p_thr = prediction.get("method"), prediction.get("threshold")
    same_thr = (s_thr is not None and p_thr is not None
                and abs(float(s_thr) - float(p_thr)) <= 1e-9)
    if summary.get("part") != part or s_method != p_method or not same_thr:
        return planned(f"сводка {source} устарела: часть {summary.get('part')}, метод "
                       f"{s_method}, порог {s_thr}; у комплекта часть {part}, метод {p_method}, "
                       f"порог {p_thr} — перезапустить eval-test --part {part}")
    res = pc.get("result") or {}
    independent = part == "test"
    return {"status": "done" if independent else "planned", "test_chip": chip_id,
            "control_points": pc.get("control_points"), "fixed_v_q": True,
            "n_points": pc.get("n_points"), "error_metric": pc.get("error_metric"),
            "result": {"estimated": True, "part": part, "independent": independent,
                       "demo_chip": res.get("demo_chip"),
                       "all_chips_of_part": res.get("all_chips_of_part"),
                       "note": res.get("note"),
                       "fixed_v_q_source": pc.get("fixed_v_q"),
                       "preliminary": (None if independent else
                                       f"часть {part} использовалась для настройки метода "
                                       "(D-012/D-016) — не независимая оценка"),
                       "source": source, "commit": summary.get("commit"),
                       "config_sha256": summary.get("config_sha256")}}


def _method_info(name: str, version: str, weights: list[dict], threshold: float, units: str,
                 source: str, **extra: Any) -> dict:
    return {"name": name, "version": version, "weights": weights, "threshold": threshold,
            "threshold_units": units, "threshold_source": source, **extra}


def run_metadata(*, run_id: str, chip_id: str, event_id: str, event_date: str, created_at: str,
                 commit: str, commit_dirty: bool, prediction: dict, baseline_config: dict,
                 baseline_config_path: str, baseline_config_sha256: str,
                 main_config: dict | None = None, main_calibration: dict | None = None,
                 assets_meta: dict, procurement_meta: dict, pricing: dict, split: dict,
                 split_file: str, run_config: dict, uncertainty_raster: str | None,
                 metrics_files: list[str], reproduce: list[str], versions: dict,
                 warnings: list[str], proxy: dict | None = None) -> dict:
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
        cfg = main_config or {}
        name = str(prediction.get("method", "main"))
        weights = [{"path": f"models/{name}/{m['file']}", "sha256": m["sha256"]}
                   for m in cfg.get("members", []) if m.get("file") and m.get("sha256")]
        arch = cfg.get("arch")
        cal_info = prediction.get("calibration")
        calibration = None
        if cal_info and main_calibration:
            calibration = {**main_calibration.get("calibrator", {}),
                           "threshold": main_calibration.get("threshold"),
                           "file": cal_info.get("file"), "sha256": cal_info.get("sha256"),
                           "fit_on": (main_calibration.get("fitted_on") or {}).get(
                               "part", "validation")}
        main = _method_info(
            name, f"{name}@{str(prediction.get("weights_sha256") or cfg.get("weights_sha256") or "")[:12]}", weights,
            float(prediction["threshold"]), "p",
            str(prediction.get("threshold_source") or prediction.get("threshold_status")
                or "prediction.json"),
            weights_sha256=prediction.get("weights_sha256") or cfg.get("weights_sha256"),
            config=f"models/{name}/config.json" if cfg else None,
            ensemble=cfg.get("ensemble"), selection=cfg.get("selection"), arch=arch,
            calibration=calibration)
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
                          "unassessed_asset_ids": assets_meta.get("unassessed_asset_ids", []),
                          "inspection_priority": assets_meta.get("inspection_priority")},
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
        "proxy_check": proxy or proxy_check(None, "", chip_id, "none", prediction),
        "metrics": {"files": metrics_files},
        "reproduce": reproduce,
        "seeds": {"portfolio_seed": assets_meta["seed"], "split_seed": split.get("seed")},
        "versions": versions,
        "warnings": warnings,
    }
