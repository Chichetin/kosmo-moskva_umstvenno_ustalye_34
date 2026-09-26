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
import math
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from floodvalue.contracts import load_run_config
from floodvalue.contracts.kopeck import kopeck_sum, ok_rows
from floodvalue.paths import data_dir, output_dir
from floodvalue.procurement import strategies, zones
from floodvalue.procurement.pricing import load_pricing

SENS_BUDGETS = (0, 1000, 2500, 5000, 10000, 20000, 50000)
SENS_RATE_FACTORS = tuple(map(Decimal, ("0.5", "0.75", "1.25", "1.5", "2.0")))
# decision_deadline = decision_at + h (D-001: 6–12 h window; 24/48 h — later decision).
SENS_DEADLINE_HOURS = (6, 12, 24, 48)

PLAN_FIELDS = ["strategy", "candidate_id", "area_km2", "base_rate_rub_km2", "base_rate_status",
               "processing_level", "usage_type", "guaranteed_purchase", "processing_coef",
               "usage_coef", "freshness_coef", "discount_coef", "discount_group_id",
               "group_area_km2", "unit_price_rub_km2", "cost_rub", "formula", "legal_edition",
               "sensor_type", "resolution_m", "acquisition_type", "reason"]
COMPARISON_FIELDS = ["strategy", "data_cost_rub", "other_cost_rub", "decision_cost_rub",
                     "budget_rub", "budget_feasible", "covered_expected_loss_rub",
                     "coverage_share", "residual_uncertainty", "uncertainty_status",
                     "uncertainty_basis", "covered_asset_ids", "addressed_decision_risk_rub",
                     "decision_risk_share"]
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
        raise ValueError("configs/run.yaml: нет блока other_cost (прочие затраты решения, D-038)")
    return {"chip_id": rc.chip_id, "budget_rub": rc.budget_rub,
            "decision_at": rc.decision_at.isoformat(),
            "decision_deadline": rc.decision_deadline.isoformat(),
            "other_cost": dict(extra["other_cost"]),
            "social_constraint": dict(extra.get("social_constraint") or {"enabled": False,
                                                                        "asset_classes": []}),
            "source": "configs/run.yaml"}


def chip_s1_date(chip_id: str) -> date:
    """Дата S1 чипа: ``observation_date`` в ``$FLOODVALUE_DATA_DIR/interim/source_manifest_s1.json``.

    Манифест пишет ``make data`` из ``s1_date`` события в Sen1Floods11_Metadata.geojson; то же
    поле export кладёт в ``run_metadata.event_date`` — порядок времени сверяется с одним источником.
    """
    path = data_dir() / "interim" / "source_manifest_s1.json"
    if not path.exists():
        raise FileNotFoundError(f"нет {path} — дата S1 чипа неизвестна, сначала `make data`")
    sources = json.loads(path.read_text(encoding="utf-8")).get("sources", [])
    obs = next((s.get("observation_date") for s in sources if s.get("chip_id") == chip_id), None)
    if not obs:
        raise ValueError(f"в {path} нет observation_date чипа {chip_id} — сначала "
                         f"`make data` (дата S1 из метаданных события)")
    return date.fromisoformat(str(obs)[:10])


def decision_times(template_at: datetime, template_deadline: datetime,
                   s1_date: date) -> tuple[datetime, datetime]:
    """(decision_at, decision_deadline) для чипа с датой S1 ``s1_date`` (D-043).

    Из run.yaml берутся только сценарное время суток с поясом (``decision_at``) и окно
    ``decision_deadline − decision_at`` (D-001: 12 ч); дата — дата S1 чипа.
    """
    at = datetime.combine(s1_date, template_at.timetz())
    return at, at + (template_deadline - template_at)


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


_CONTEXT_CACHE: dict[Path, tuple[tuple, dict[str, Any]]] = {}


def _inputs_key(run_dir: Path) -> tuple:
    """(путь, mtime_ns, размер) всех входов контекста; нет файла — (путь, None, None)."""
    from floodvalue.paths import CONFIGS_DIR
    paths = [run_dir / n for n in ("assets.csv", "asset_loss.csv", "flood_probability.tif",
                                   "uncertainty.tif")]
    paths += [CONFIGS_DIR / "run.yaml", CONFIGS_DIR / "pricing.yaml",
              data_dir() / "interim" / "source_manifest_s1.json"]
    key = []
    for p in paths:
        try:
            st = p.stat()
            key.append((str(p), st.st_mtime_ns, st.st_size))
        except FileNotFoundError:
            key.append((str(p), None, None))
    return tuple(key)


def _cached_context(run_dir: Path) -> dict[str, Any]:
    """:func:`_context` с кешем по :func:`_inputs_key`: смена бюджета не пересобирает каталог (#105).

    Любая правка входа (mtime или размер) даёт новый ключ — устаревший контекст не читается.
    """
    run_dir = Path(run_dir).resolve()
    key = _inputs_key(run_dir)
    hit = _CONTEXT_CACHE.get(run_dir)
    if hit is not None and hit[0] == key:
        return hit[1]
    ctx = _context(run_dir)
    _CONTEXT_CACHE[run_dir] = (key, ctx)
    return ctx


def _context(run_dir: Path) -> dict[str, Any]:
    """Всё, что не зависит от бюджета: каталог, охват, EL, конфиги."""
    run_dir = Path(run_dir)
    rc = run_config()
    pricing_cfg = load_pricing()
    assets = _read_csv(run_dir / "assets.csv")
    loss = _read_csv(run_dir / "asset_loss.csv")
    chips = {a["chip_id"] for a in assets}
    if len(chips) != 1:
        raise ValueError(f"{run_dir / 'assets.csv'}: ожидается один chip_id, получено {sorted(chips)}")
    (chip_id,) = chips
    s1_date = chip_s1_date(chip_id)
    template_at = zones.parse_dt(rc["decision_at"])
    if chip_id == rc["chip_id"] and template_at.date() != s1_date:
        raise ValueError(f"configs/run.yaml: дата decision_at {template_at.date()} ≠ дате S1 "
                         f"{s1_date} чипа {chip_id} из run.yaml — исправить конфиг")
    decision_at, deadline = decision_times(template_at, zones.parse_dt(rc["decision_deadline"]),
                                           s1_date)
    # The catalog is always rebuilt from current inputs (deterministic, budget-independent):
    # a stale candidate_orders.geojson never drives coverage.
    bounds, bounds_src = _chip_bounds(run_dir, assets)
    unc, transform, unc_src = _uncertainty_raster(run_dir)
    catalog = zones.build_catalog(assets, bounds, decision_at, pricing_cfg, loss=loss,
                                  uncertainty=unc, transform=transform,
                                  decision_deadline=deadline)
    ok_ids = {r["asset_id"] for r in ok_rows(loss)}
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
            "chip_id": chip_id, "s1_date": s1_date,
            "bounds_source": bounds_src, "uncertainty_source": unc_src,
            "social_ids": social_ids,
            "required": social_ids if social_cfg.get("enabled") else []}


def _not_selected_reason(cid: str, ctx: dict[str, Any], res: dict[str, Any]) -> str:
    if cid in ctx["excluded"]:
        return ctx["excluded"][cid]
    covered = {a for c in res["plans"]["C"] for a in ctx["cov"].get(c, ())}
    el = strategies.assessed(ctx["loss"])
    own = [a for a in ctx["cov"].get(cid, ()) if a in el]
    if not own:
        return "не покрывает ни одного оценённого объекта (status=ok)"
    if set(own) <= covered:
        return f"объекты {', '.join(own)} уже покрыты выбранными позициями C — R и EL не добавляет"
    new = [a for a in own if a not in covered]
    if sum(el[a]["risk"] for a in new) <= 0:
        return (f"низкая ценность информации: исход по объекту почти известен ({', '.join(new)}: "
                f"p = 0 или 1, цена ошибки решения R = V·q·min(p, 1−p) = 0) — снимок "
                f"решения не меняет (D-049)")
    return ("не выбрана в C: хуже по цене ошибки решения на рубль прироста цены корзины или не "
            "помещается в остаток бюджета при пересчёте всей корзины")


def _plan(ctx: dict[str, Any], budget_rub: float, social: bool | None = None,
          priority_asset_id: str | None = None) -> dict[str, Any]:
    required = ctx["required"] if social is None else (ctx["social_ids"] if social else [])
    if priority_asset_id:
        required = sorted(set(required) | {priority_asset_id})
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
    строится из текущих входов (детерминированно, как в файле); p-растр,
    маску, порог и портфель не меняет. ``social`` — включить/выключить социальное ограничение
    C (None — как в configs/run.yaml). Ключи: plans, plan_rows, comparison, reasons,
    social_constraint. Контекст (каталог, охват, EL) кешируется по mtime и размеру входов —
    повторная смена бюджета не пересобирает каталог, результат тот же (#105).
    """
    return _plan(_cached_context(Path(run_dir)), budget_rub, social)


def priority_plan(run_dir: Path, budget_rub: float, asset_id: str) -> dict[str, Any]:
    """Preview plan C with one assessed asset covered first; leave bundle files untouched."""
    ctx = _cached_context(Path(run_dir))
    assessed_ids = strategies.assessed(ctx["loss"])
    if asset_id not in assessed_ids:
        raise ValueError(f"Объект {asset_id!r} не найден среди оценённых объектов")
    baseline = _plan(ctx, budget_rub)
    result = _plan(ctx, budget_rub, priority_asset_id=asset_id)
    covered = any(asset_id in ctx["cov"].get(cid, ()) for cid in result["plans"]["C"])
    available = any(asset_id in ctx["cov"].get(cid, ()) for cid in ctx["eligible"])
    baseline_c = next(row for row in baseline["comparison"] if row["strategy"] == "C")
    result["priority"] = {
        "asset_id": asset_id,
        "covered": covered,
        "reason": ("Объект включён в зону плана C" if covered else
                   "Нет доступной к сроку зоны для объекта" if not available else
                   "При этом бюджете зона для объекта не помещается в план C"),
        "baseline_candidate_ids": baseline["plans"]["C"],
        "baseline_data_cost_rub": baseline_c["data_cost_rub"],
        "baseline_coverage_share": baseline_c["coverage_share"],
    }
    return result


def _cell(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, Decimal):
        return f"{v:f}"
    if isinstance(v, float):
        return repr(v) if math.isfinite(v) else ""  # contract: empty, not "nan" (#235)
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
                                  required=ctx["required"], social_ids=ctx["social_ids"],
                                  decision_at=ctx["decision_at"],
                                  deadline_hours=SENS_DEADLINE_HOURS)
    _write_csv(run_dir / "procurement_sensitivity.csv", SENS_FIELDS, sens)
    # exact sum of the kopeck EL of asset_loss.csv (#127), same as coverage_share in the CSV
    denom = kopeck_sum(r["expected_loss_rub"] for r in ok_rows(ctx["loss"]))
    kinds: dict[str, int] = {}
    for f in ctx["catalog"]["features"]:
        k = f["properties"]["zone_kind"]
        kinds[k] = kinds.get(k, 0) + 1
    meta = {
        "budget_rub": float(budget_rub),
        "run_config_source": ctx["rc"]["source"],
        "decision_at": ctx["decision_at"].isoformat(),
        "decision_deadline": ctx["deadline"].isoformat(),
        "decision_at_rule": ("decision_at = дата S1 чипа + сценарное время суток decision_at из "
                             "configs/run.yaml; decision_deadline = decision_at + окно run.yaml "
                             "(12 ч, D-001); D-043"),
        "chip_id": ctx["chip_id"],
        "s1_observation_date": ctx["s1_date"].isoformat(),
        "s1_observation_date_source": ("observation_date в interim/source_manifest_s1.json "
                                       "(s1_date события, Sen1Floods11_Metadata.geojson)"),
        "pricing_as_of": ctx["as_of"].isoformat(),
        "pricing_as_of_rule": ("дата расчёта цены = календарная дата decision_at (UTC), D-023, "
                               "D-043: давность снимков и год группы скидки — от даты события; "
                               "формула, коэффициенты и сценарная ставка Б — по проверенной "
                               "редакции ПП 840 (configs/pricing.yaml), которой на дату события "
                               "могло не быть: гибридный учебный сценарий, не историческая цена"),
        "zones": {
            "construction": ZONES_CONSTRUCTION,
            "decision": "D-038 (каталог), D-023 (v0, правила A/B/C)",
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
            "excluded_covers_rule": ("у позиций с exclusion_reason covers_asset_ids = [] (к сроку "
                                     "решения ничего не наблюдают); объекты внутри их контура — "
                                     "inside_geometry_asset_ids, это не охват (D-048)"),
        },
        "rule_B": strategies.RULE_B,
        "rule_C": strategies.RULE_C,
        "rule_C_by_EL": strategies.RULE_C_BY_EL,
        "decision_risk": {
            "formula": "R = V·q·min(p, 1−p) = EL·min(p, 1−p)/p, руб. (сценарий), D-049",
            "status": "scenario",
            "meaning": ("адресуемая съёмкой ожидаемая цена ошибки решения (EVPI при бинарном "
                        "решении, симметричной цене ошибки V·q, идеальном снимке и "
                        "откалиброванной p); не фактическое снижение потерь"),
            "columns": "strategy_comparison.csv: addressed_decision_risk_rub, decision_risk_share",
            "excluded": "объекты partial/no_data в R не входят (p нет)"},
        "social_constraint": {**ctx["rc"]["social_constraint"],
                              "asset_ids": ctx["social_ids"], "result": res["social_constraint"],
                              "note": ("отдельное ограничение покрытия C, не множитель рублей EL "
                                       "(task.md); эффект — строка "
                                       + ("social_constraint_off (опция включена)"
                                          if ctx["required"] else
                                          "social_constraint_on (опция выключена)")
                                       + " в procurement_sensitivity.csv")},
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
    # whole kopecks only: a rounded budget would differ from run_metadata.budget.budget_rub
    if (not math.isfinite(budget) or budget < 0
            or Decimal(str(budget)) != Decimal(str(budget)).quantize(Decimal("0.01"))):
        print("procure: бюджет — конечное неотрицательное число рублей с копейками")
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
              f"цена ошибки решения {r['addressed_decision_risk_rub']:.2f} руб., "
              f"зоны {res['plans'][r['strategy']]}")
    print(f"procure: файлы в {out}")
    return 0
