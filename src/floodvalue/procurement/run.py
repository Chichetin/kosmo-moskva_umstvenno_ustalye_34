"""`python -m floodvalue procure [--chip <id>] [--budget <руб>] [--out <dir>]`: зоны и A/B/C.

Читает из каталога комплекта файлы assets (``assets.csv``, ``asset_loss.csv``) и только
заголовок ``flood_probability.tif`` (границы чипа для широкой зоны). Пишет
``candidate_orders.geojson``, ``procurement_plan.csv``, ``strategy_plans.json``,
``strategy_comparison.csv``, ``procurement_sensitivity.csv`` (export сливает с
``assets_sensitivity.csv`` в ``sensitivity.csv``) и ``procurement_metadata.json``.
:func:`floodvalue.procurement.plan` — тот же пересчёт без записи, для интерфейса (D-013).
"""
from __future__ import annotations

import csv
import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml

from floodvalue.paths import CONFIGS_DIR, output_dir
from floodvalue.procurement import strategies, zones
from floodvalue.procurement.pricing import load_pricing

# До влития configs/run.yaml (issue contracts) — те же значения, что в его черновике.
# TODO: удалить запасные значения, когда configs/run.yaml будет в main.
FALLBACK_RUN = {"chip_id": "Sri-Lanka_713926", "budget_rub": 5000,
                "decision_at": "2017-05-30T12:00:00+00:00",
                "decision_deadline": "2017-05-31T00:00:00+00:00"}
SENS_BUDGETS = (0, 1000, 2500, 5000, 10000, 20000, 50000)
SENS_RATE_FACTORS = (Decimal("0.5"), Decimal("1.5"))

PLAN_FIELDS = ["strategy", "candidate_id", "area_km2", "base_rate_rub_km2", "base_rate_status",
               "processing_level", "usage_type", "guaranteed_purchase", "processing_coef",
               "usage_coef", "freshness_coef", "discount_coef", "discount_group_id",
               "group_area_km2", "unit_price_rub_km2", "cost_rub", "formula", "legal_edition",
               "sensor_type", "resolution_m", "acquisition_type", "reason"]
COMPARISON_FIELDS = ["strategy", "data_cost_rub", "other_cost_rub", "decision_cost_rub",
                     "budget_rub", "budget_feasible", "covered_expected_loss_rub",
                     "coverage_share", "residual_uncertainty", "uncertainty_status",
                     "uncertainty_basis", "covered_asset_ids"]
SENS_FIELDS = ["scenario_id", "strategy", "changed_inputs_json", "decision_cost_rub",
               "covered_expected_loss_rub", "result_status", "interpretation"]


def run_config() -> dict[str, Any]:
    path = CONFIGS_DIR / "run.yaml"
    cfg = dict(FALLBACK_RUN)
    source = "floodvalue.procurement.run.FALLBACK_RUN (configs/run.yaml нет в main)"
    if path.exists():
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        cfg.update({k: loaded[k] for k in FALLBACK_RUN if k in loaded})
        source = "configs/run.yaml"
    cfg["source"] = source
    return cfg


def _read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _chip_bounds(run_dir: Path, assets: list[dict[str, str]]) -> tuple[tuple, str]:
    prob = run_dir / "flood_probability.tif"
    if prob.exists():
        import rasterio
        with rasterio.open(prob) as src:
            if src.crs is None or src.crs.to_epsg() != 4326:
                raise ValueError(f"{prob}: ожидается EPSG:4326, получено {src.crs}")
            return tuple(src.bounds), "flood_probability.tif bounds"
    lons = [float(a["longitude"]) for a in assets]
    lats = [float(a["latitude"]) for a in assets]
    return ((min(lons) - 0.01, min(lats) - 0.01, max(lons) + 0.01, max(lats) + 0.01),
            "bbox объектов ± 0,01° (flood_probability.tif нет)")


def _context(run_dir: Path) -> dict[str, Any]:
    """Всё, что не зависит от бюджета: каталог, охват, EL, конфиги."""
    run_dir = Path(run_dir)
    rc = run_config()
    pricing_cfg = load_pricing()
    assets = _read_csv(run_dir / "assets.csv")
    loss = _read_csv(run_dir / "asset_loss.csv")
    decision_at = zones.parse_dt(rc["decision_at"])
    deadline = zones.parse_dt(rc["decision_deadline"])
    # The catalog is always rebuilt from current assets, chip bounds and decision_at
    # (deterministic, cheap): a stale candidate_orders.geojson never drives coverage.
    bounds, bounds_src = _chip_bounds(run_dir, assets)
    catalog = zones.build_catalog(assets, bounds, decision_at, pricing_cfg)
    ok_ids = {r["asset_id"] for r in loss if r["status"] == "ok"}
    ok_assets = [a for a in assets if a["asset_id"] in ok_ids]
    cov = zones.coverage(catalog, ok_assets, deadline)
    excluded = {f["properties"]["candidate_id"]: r for f in catalog["features"]
                if (r := zones.exclusion_reason(f["properties"], deadline))}
    eligible = {c for c in cov if c not in excluded}
    return {"rc": rc, "pricing": pricing_cfg, "assets": assets, "loss": loss,
            "catalog": catalog, "cov": cov, "excluded": excluded, "eligible": eligible,
            "decision_at": decision_at, "deadline": deadline, "as_of": decision_at.date(),
            "bounds_source": bounds_src}


def _plan(ctx: dict[str, Any], budget_rub: float) -> dict[str, Any]:
    res = strategies.compute_plan(ctx["catalog"], ctx["cov"], ctx["eligible"], ctx["loss"],
                                  Decimal(str(budget_rub)), ctx["pricing"], ctx["as_of"])
    chosen = set(res["plans"]["B"]) | set(res["plans"]["C"])
    not_chosen = {}
    for f in ctx["catalog"]["features"]:
        cid = f["properties"]["candidate_id"]
        if cid in ctx["excluded"]:
            not_chosen[cid] = ctx["excluded"][cid]
        elif cid not in res["plans"]["C"]:
            not_chosen[cid] = ("не выбрана в C: не добавляет нового оценённого EL или не "
                               "помещается в остаток бюджета при пересчёте всей корзины")
    res["reasons"]["not_selected_C"] = not_chosen
    res["reasons"]["chosen_any"] = sorted(chosen)
    return res


def plan(run_dir: Path, budget_rub: float) -> dict[str, Any]:
    """Пересчёт A/B/C для бюджета без записи файлов (для интерфейса, D-013).

    Читает assets.csv, asset_loss.csv, заголовок flood_probability.tif и конфиги; каталог зон
    каждый раз строится заново из текущих входов (детерминированно, как в файле); p-растр, маску, порог и портфель не меняет. Ключи: plans, plan_rows,
    comparison, reasons.
    """
    return _plan(_context(Path(run_dir)), budget_rub)


def _cell(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, Decimal):
        return f"{v:f}"
    if isinstance(v, float):
        return repr(v)
    return str(v)


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(fields)
        for r in rows:
            w.writerow([_cell(r.get(k)) for k in fields])


def write_outputs(run_dir: Path, budget_rub: float) -> dict[str, Any]:
    ctx = _context(run_dir)
    res = _plan(ctx, budget_rub)
    run_dir = Path(run_dir)
    (run_dir / "candidate_orders.geojson").write_text(
        json.dumps(ctx["catalog"], ensure_ascii=False, indent=1), encoding="utf-8")
    rows = [{**r, "reason": (res["reasons"]["C"] if r["strategy"] == "C"
                             else res["reasons"]["B"]).get(r["candidate_id"], "")}
            for r in res["plan_rows"]]
    _write_csv(run_dir / "procurement_plan.csv", PLAN_FIELDS, rows)
    (run_dir / "strategy_plans.json").write_text(json.dumps(res["plans"], indent=1),
                                                 encoding="utf-8")
    _write_csv(run_dir / "strategy_comparison.csv", COMPARISON_FIELDS, res["comparison"])
    sens = strategies.sensitivity(ctx["catalog"], ctx["cov"], ctx["eligible"], ctx["loss"],
                                  Decimal(str(budget_rub)), ctx["pricing"], ctx["as_of"],
                                  sorted({*map(Decimal, map(str, SENS_BUDGETS)),
                                          Decimal(str(budget_rub))}),
                                  SENS_RATE_FACTORS)
    _write_csv(run_dir / "procurement_sensitivity.csv", SENS_FIELDS, sens)
    denom = sum(float(r["expected_loss_rub"]) for r in ctx["loss"] if r["status"] == "ok")
    meta = {
        "budget_rub": float(budget_rub),
        "run_config_source": ctx["rc"]["source"],
        "decision_at": ctx["decision_at"].isoformat(),
        "decision_deadline": ctx["deadline"].isoformat(),
        "pricing_as_of": ctx["as_of"].isoformat(),
        "pricing_as_of_rule": "дата расчёта цены = календарная дата decision_at (UTC), D-023",
        "zones": {
            "construction": ("Z-<asset_id>: квадрат 1,2 × 1,2 км вокруг каждого объекта в "
                             "EPSG:6933; W-CHIP: границы чипа; N-OPT-LATE: bbox объектов "
                             "± 0,005°; K-ARCH/K-OPER: границы чипа (контекст)"),
            "chip_bounds_source": ctx["bounds_source"],
            "area_method": "вершины WGS 84 → EPSG:6933 (always_xy), м²/1e6, ≥ 1 км² до округления, ROUND_HALF_UP 0,001",
            "timing_status": zones.TIMING_STATUS,
            "timing_basis": zones.TIMING_BASIS,
            "excluded_from_coverage": ctx["excluded"],
            "coverage": ctx["cov"],
            "coverage_rule": ("polygon.covers(point) по объектам status=ok, уникально по "
                              "asset_id; только data_role=event_observation и "
                              "available_at ≤ decision_deadline"),
        },
        "rule_B": strategies.RULE_B,
        "rule_C": strategies.RULE_C,
        "reasons": res["reasons"],
        "other_cost": {k: (f"{v:f}" if isinstance(v, Decimal) else v)
                       for k, v in strategies.OTHER_COST.items()},
        "coverage_share_denominator_rub": denom,
        "coverage_share_empty_reason": (None if denom > 0 else
                                        "суммарный оценённый EL портфеля = 0 (нет объектов "
                                        "status=ok или все p = 0) — доля не определена"),
        "uncertainty": {"A": strategies.UNCERTAINTY_A_BASIS,
                        "B_C": strategies.UNCERTAINTY_BC_BASIS},
        "sensitivity_file": "procurement_sensitivity.csv (export сливает в sensitivity.csv)",
    }
    (run_dir / "procurement_metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return res


def run(args) -> int:
    rc = run_config()
    chip_id = args.chip or rc["chip_id"]
    out = Path(args.out) if args.out else output_dir() / chip_id
    budget = args.budget if args.budget is not None else float(rc["budget_rub"])
    if budget < 0:
        print("procure: бюджет не может быть отрицательным")
        return 1
    for name in ("assets.csv", "asset_loss.csv"):
        if not (out / name).exists():
            print(f"procure: нет {out / name} — сначала `make assets CHIP={chip_id}`")
            return 1
    res = write_outputs(out, budget)
    for r in res["comparison"]:
        print(f"procure: {r['strategy']}: данные {r['data_cost_rub']} руб., "
              f"бюджет {r['budget_rub']}, в бюджете {r['budget_feasible']}, "
              f"охват EL {r['covered_expected_loss_rub']:.2f} руб., "
              f"зоны {res['plans'][r['strategy']]}")
    print(f"procure: файлы в {out}")
    return 0

