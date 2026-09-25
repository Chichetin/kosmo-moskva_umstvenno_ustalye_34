"""`python -m floodvalue procure [--chip <id>] [--budget <руб>] [--out <dir>]`: зоны и A/B/C.

Читает из каталога комплекта файлы assets (``assets.csv``, ``asset_loss.csv``),
``flood_probability.tif`` (границы чипа; растр — только для запасного показателя
неопределённости) и ``uncertainty.tif``, если он есть (кластеры высокой неопределённости,
контракт §2). p, маску, порог и портфель не меняет. Пишет
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

from floodvalue.contracts import load_run_config
from floodvalue.paths import output_dir
from floodvalue.procurement import strategies, zones
from floodvalue.procurement.pricing import load_pricing

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
    """configs/run.yaml через контракт (floodvalue.contracts.load_run_config).

    Кроме полей схемы ``RunConfig`` берёт доп. ключи ``other_cost`` (сценарные прочие затраты)
    и ``social_constraint`` (опция C) — схема допускает доп. поля (``extra=allow``).
    """
    rc = load_run_config()
    extra = rc.model_extra or {}
    if "other_cost" not in extra:
        raise ValueError("configs/run.yaml: нет блока other_cost (прочие затраты решения, D-033)")
    return {"chip_id": rc.chip_id, "budget_rub": rc.budget_rub,
            "decision_at": rc.decision_at.isoformat(),
            "decision_deadline": rc.decision_deadline.isoformat(),
            "other_cost": dict(extra["other_cost"]),
            "social_constraint": dict(extra.get("social_constraint") or {"enabled": False,
                                                                        "asset_classes": []}),
            "source": "configs/run.yaml"}


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


def binary_entropy_bits(p: Any) -> Any:
    """H(p) в битах; NaN сохраняется (запасной показатель, тот же, что D-018 в asset_loss.csv)."""
    import numpy as np
    pc = np.clip(np.asarray(p, dtype="float64"), 1e-12, 1 - 1e-12)
    h = -(pc * np.log2(pc) + (1 - pc) * np.log2(1 - pc))
    return np.where(np.isfinite(p), h, np.nan)


def _uncertainty_raster(run_dir: Path) -> tuple[Any, Any, str]:
    """(массив, transform, источник): uncertainty.tif, иначе бинарная энтропия p, иначе нет."""
    import numpy as np
    import rasterio

    def read(path):
        with rasterio.open(path) as src:
            a = src.read(1).astype("float64")
            if src.nodata is not None:
                a[a == src.nodata] = np.nan
            return a, src.transform

    unc = run_dir / "uncertainty.tif"
    if unc.exists():
        a, tr = read(unc)
        return a, tr, "uncertainty.tif комплекта (смысл — run_metadata.uncertainty)"
    prob = run_dir / "flood_probability.tif"
    if prob.exists():
        a, tr = read(prob)
        return (binary_entropy_bits(a), tr,
                ("запасной вариант: бинарная энтропия p из flood_probability.tif (uncertainty.tif "
                 "нет; тот же показатель, что uncertainty в asset_loss.csv, D-018)"))
    return None, None, "нет растров — кластеры неопределённости не строятся"


def _context(run_dir: Path) -> dict[str, Any]:
    """Всё, что не зависит от бюджета: каталог, охват, EL, конфиги."""
    run_dir = Path(run_dir)
    rc = run_config()
    pricing_cfg = load_pricing()
    assets = _read_csv(run_dir / "assets.csv")
    loss = _read_csv(run_dir / "asset_loss.csv")
    decision_at = zones.parse_dt(rc["decision_at"])
    deadline = zones.parse_dt(rc["decision_deadline"])
    # The catalog is always rebuilt from current inputs (deterministic, budget-independent):
    # a stale candidate_orders.geojson never drives coverage.
    bounds, bounds_src = _chip_bounds(run_dir, assets)
    unc, transform, unc_src = _uncertainty_raster(run_dir)
    catalog = zones.build_catalog(assets, bounds, decision_at, pricing_cfg, loss=loss,
                                  uncertainty=unc, transform=transform,
                                  decision_deadline=deadline)
    ok_ids = {r["asset_id"] for r in loss if r["status"] == "ok"}
    ok_assets = [a for a in assets if a["asset_id"] in ok_ids]
    cov = zones.coverage(catalog, ok_assets, deadline)
    excluded = {f["properties"]["candidate_id"]: r for f in catalog["features"]
                if (r := zones.exclusion_reason(f["properties"], deadline))}
    eligible = {c for c in cov if c not in excluded}
    social_cfg = rc["social_constraint"]
    social_ids = sorted(a["asset_id"] for a in assets
                        if a.get("asset_class") in set(social_cfg.get("asset_classes") or ()))
    return {"rc": rc, "pricing": pricing_cfg, "assets": assets, "loss": loss,
            "catalog": catalog, "cov": cov, "excluded": excluded, "eligible": eligible,
            "decision_at": decision_at, "deadline": deadline, "as_of": decision_at.date(),
            "bounds_source": bounds_src, "uncertainty_source": unc_src,
            "social_ids": social_ids,
            "required": social_ids if social_cfg.get("enabled") else []}


def _not_selected_reason(cid: str, ctx: dict[str, Any], res: dict[str, Any]) -> str:
    if cid in ctx["excluded"]:
        return ctx["excluded"][cid]
    covered = {a for c in res["plans"]["C"] for a in ctx["cov"].get(c, ())}
    own = [a for a in ctx["cov"].get(cid, ()) if a in {r["asset_id"] for r in ctx["loss"]
                                                        if r["status"] == "ok"}]
    if not own:
        return "не покрывает ни одного оценённого объекта (status=ok)"
    if set(own) <= covered:
        return f"объекты {', '.join(own)} уже покрыты выбранными позициями C — EL не добавляет"
    return ("не выбрана в C: хуже по EL на рубль прироста цены корзины или не помещается в "
            "остаток бюджета при пересчёте всей корзины")


def _plan(ctx: dict[str, Any], budget_rub: float, social: bool | None = None) -> dict[str, Any]:
    required = ctx["required"] if social is None else (ctx["social_ids"] if social else [])
    res = strategies.compute_plan(ctx["catalog"], ctx["cov"], ctx["eligible"], ctx["loss"],
                                  Decimal(str(budget_rub)), ctx["pricing"], ctx["as_of"],
                                  other_cfg=ctx["rc"]["other_cost"], required=required)
    chosen = set(res["plans"]["B"]) | set(res["plans"]["C"])
    res["reasons"]["not_selected_C"] = {
        f["properties"]["candidate_id"]: _not_selected_reason(f["properties"]["candidate_id"],
                                                              ctx, res)
        for f in ctx["catalog"]["features"]
        if f["properties"]["candidate_id"] not in res["plans"]["C"]}
    res["reasons"]["chosen_any"] = sorted(chosen)
    return res


def plan(run_dir: Path, budget_rub: float, social: bool | None = None) -> dict[str, Any]:
    """Пересчёт A/B/C для бюджета без записи файлов (для интерфейса, D-013).

    Читает assets.csv, asset_loss.csv, растры комплекта (только чтение) и конфиги; каталог зон
    каждый раз строится заново из текущих входов (детерминированно, как в файле); p-растр,
    маску, порог и портфель не меняет. ``social`` — включить/выключить социальное ограничение
    C (None — как в configs/run.yaml). Ключи: plans, plan_rows, comparison, reasons,
    social_constraint.
    """
    return _plan(_context(Path(run_dir)), budget_rub, social)


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


ZONES_CONSTRUCTION = (
    "Z-<asset_id>: квадрат 1,2 × 1,2 км в EPSG:6933 вокруг каждого объекта (по чипу не "
    "обрезается: у объектов у края оплачиваемая площадь частично вне сцены S1); E-<k>/E-ALL: "
    "выпуклая оболочка кластера объектов высокого EL (первые по EL до 90 % суммы, одиночная "
    "связь ≤ 2 км) + буфер 600 м, обрезка по чипу; U-<k>: выпуклая оболочка связной группы "
    "блоков 16×16 пикс. с неопределённостью ≥ 0,9-квантиля блоков (3 группы с наибольшей "
    "суммой) + буфер 150 м, обрезка по чипу; G2-*/G3-*: регулярная сетка 2×2 и 3×3 по чипу; "
    "W-CHIP: весь чип. Буфер растёт ×1,15, пока площадь < 1,001 км². На геометрию — "
    "сценарные предложения (offer_id): S1N новая SAR 1 м (+3/+9 ч), S3N новая SAR 3 м "
    "(+6/+11 ч), O05N новая оптика 0,5 м (+4/+10 ч, облачность не подтверждена), O1L новая "
    "оптика 1 м (+20/+30 ч, после срока), S3OP оперативный SAR 3 м события (−6 ч, срок "
    "поставки неизвестен), O1AR архив оптики 180 сут, O1OC оптика 30 сут до паводка "
    "(контекст); candidate_id = geometry_id + суффикс предложения (у S1N суффикса нет)")


def write_outputs(run_dir: Path, budget_rub: float) -> dict[str, Any]:
    ctx = _context(run_dir)
    res = _plan(ctx, budget_rub)
    run_dir = Path(run_dir)
    catalog = {"type": "FeatureCollection", "features": ctx["catalog"]["features"]}
    (run_dir / "candidate_orders.geojson").write_text(
        json.dumps(catalog, ensure_ascii=False, indent=1), encoding="utf-8")
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
                                  SENS_RATE_FACTORS, other_cfg=ctx["rc"]["other_cost"],
                                  required=ctx["required"], social_ids=ctx["social_ids"])
    _write_csv(run_dir / "procurement_sensitivity.csv", SENS_FIELDS, sens)
    denom = sum(float(r["expected_loss_rub"]) for r in ctx["loss"] if r["status"] == "ok")
    kinds: dict[str, int] = {}
    for f in ctx["catalog"]["features"]:
        k = f["properties"]["zone_kind"]
        kinds[k] = kinds.get(k, 0) + 1
    meta = {
        "budget_rub": float(budget_rub),
        "run_config_source": ctx["rc"]["source"],
        "decision_at": ctx["decision_at"].isoformat(),
        "decision_deadline": ctx["deadline"].isoformat(),
        "pricing_as_of": ctx["as_of"].isoformat(),
        "pricing_as_of_rule": "дата расчёта цены = календарная дата decision_at (UTC), D-023",
        "zones": {
            "construction": ZONES_CONSTRUCTION,
            "decision": "D-033 (каталог), D-023 (v0, правила A/B/C)",
            "n_candidates": len(ctx["catalog"]["features"]),
            "n_by_zone_kind": kinds,
            "build": ctx["catalog"]["build"],
            "offers": {k: {kk: vv for kk, vv in v.items() if kk != "suffix"}
                       for k, v in zones.OFFERS.items()},
            "offers_by_zone_kind": zones.OFFERS_BY_KIND,
            "uncertainty_source": ctx["uncertainty_source"],
            "chip_bounds_source": ctx["bounds_source"],
            "area_method": "вершины WGS 84 → EPSG:6933 (always_xy), м²/1e6, ≥ 1 км² до округления, ROUND_HALF_UP 0,001",
            "single_price_note": ("single_order_cost_rub в каталоге — цена позиции, заказанной "
                                  "отдельно; в корзине B/C скидка Р пересчитывается по группе "
                                  "целиком (procurement_plan.csv)"),
            "timing_status": zones.TIMING_STATUS,
            "timing_basis": zones.TIMING_BASIS,
            "unknown_dates_rule": ("available_at = null → срок неизвестен, охвата не даёт"),
            "cloud_rule": zones.CLOUD_NOTE,
            "excluded_from_coverage": ctx["excluded"],
            "coverage": ctx["cov"],
            "coverage_rule": ("polygon.covers(point) по объектам status=ok, уникально по "
                              "asset_id; только data_role=event_observation, available_at ≤ "
                              "decision_deadline и без неподтверждённой облачности"),
        },
        "rule_B": strategies.RULE_B,
        "rule_C": strategies.RULE_C,
        "social_constraint": {**ctx["rc"]["social_constraint"],
                              "asset_ids": ctx["social_ids"], "result": res["social_constraint"],
                              "note": ("отдельное ограничение покрытия C, не множитель рублей EL "
                                       "(task.md); эффект — строка social_constraint_on в "
                                       "procurement_sensitivity.csv")},
        "c_selection": res["c_selection"],
        "reasons": res["reasons"],
        "other_cost": {**ctx["rc"]["other_cost"],
                       "comparison_note": ("C_review начисляется за каждую купленную позицию: B "
                                           "(одна позиция на весь чип) по этой статье дешевле C "
                                           "из нескольких мелких позиций — сравнивать вместе с "
                                           "data_cost_rub и охватом")},
        "coverage_share_denominator_rub": denom,
        "coverage_share_empty_reason": (None if denom > 0 else
                                        "суммарный оценённый EL портфеля = 0 (нет объектов "
                                        "status=ok или все p = 0) — доля не определена"),
        "uncertainty": {"A": strategies.UNCERTAINTY_A_BASIS,
                        "B_C": strategies.UNCERTAINTY_BC_BASIS,
                        "B_C_status": "scenario",
                        "units": ("единицы поля uncertainty asset_loss.csv (смысл — "
                                  "run_metadata.uncertainty; сейчас бинарная энтропия p, биты, "
                                  "D-018, временно до issue #21)"),
                        "aggregation": "сумма по объектам status=ok (A — все, B/C — непокрытые)"},
        "sensitivity_file": "procurement_sensitivity.csv (export сливает в sensitivity.csv)",
    }
    (run_dir / "procurement_metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
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

