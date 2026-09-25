"""Стратегии A/B/C над одним каталогом и одной функцией цены (чистые функции, D-023, D-033).

* A — открытые данные: платных зон нет.
* B — широкая закупка по заранее объявленному правилу :data:`RULE_B`; бюджет не
  ограничивает состав, при превышении ``budget_feasible=false`` (контрфактический вариант).
* C — жадный отбор :data:`RULE_C`: на каждом шаге корзина переоценивается целиком
  (скидка Р по группе), берётся зона с наибольшим приростом уникально покрытого EL на рубль
  прироста цены корзины, пока корзина ≤ бюджета; опция — социальное ограничение покрытия.

Остаточная неопределённость B/C — ``scenario`` (сумма неопределённости непокрытых оценённых
объектов, «если бы снимок снял неопределённость покрытых»), не наблюдение (D-033).

EL, неопределённость и охват берутся из файлов assets (``asset_loss.csv``) и каталога;
p-растр, маска и порог здесь не читаются — смена бюджета их не меняет.
"""
from __future__ import annotations

import copy
import json
from collections.abc import Iterable, Mapping
from datetime import date
from decimal import Decimal
from typing import Any

from floodvalue.procurement.pricing import basket_total, price_basket

RULE_B = ("B: одна позиция W-CHIP (zone_kind=wide, основное предложение S1N — новая SAR 1 м) — "
          "сплошная новая съёмка всей области чипа, если она доступна к decision_deadline; "
          "состав не зависит от бюджета, EL и цены; сверх бюджета — контрфактическая")
RULE_C = ("C: только позиции event_observation, пригодные и доступные к decision_deadline и "
          "покрывающие хотя бы один оценённый объект без охвата; проходы — чисто жадный и «старт "
          "с каждой пригодной позиции, далее жадно»; берётся проход с наибольшим уникальным EL "
          "(при равенстве — дешевле, затем чисто жадный, затем id старта); жадный шаг — max(прирост "
          "уникального EL / прирост цены всей корзины), при равенстве — больше суммарная "
          "неопределённость новых объектов, меньше прирост цены, раньше available_at, меньше "
          "candidate_id; корзина переоценивается целиком (скидка Р по группе) при каждой пробе; "
          "стоп — когда ни одна позиция не помещается в остаток бюджета или не добавляет EL. "
          "Где входят факторы: EL — числитель критерия; цена — знаменатель (прирост цены всей "
          "корзины) и лимит бюджета; срок — фильтр available_at ≤ decision_deadline и тай-брейк; "
          "неопределённость — тай-брейк и зоны U-* в каталоге (с рублями EL не смешивается). "
          "Опция social_constraint (configs/run.yaml): сначала покрыть объекты социальных "
          "классов (min прироста цены корзины на один новый обязательный объект), затем общий "
          "отбор; рубли EL не умножаются")

UNCERTAINTY_A_BASIS = ("сумма поля uncertainty asset_loss.csv по объектам status=ok "
                       "(исходная неопределённость карты в точках объектов, без новых наблюдений)")
UNCERTAINTY_BC_BASIS = ("сценарий, не наблюдение: сумма uncertainty asset_loss.csv по объектам "
                        "status=ok, которые стратегия НЕ покрывает (оптимистичная граница: снимок "
                        "полностью снял бы неопределённость покрытых объектов); новые снимки не "
                        "получены и не обработаны — фактическое снижение неопределённости не "
                        "заявляется")


def catalog_items(catalog: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """candidate_id → позиция для :func:`price_basket` (свойства каталога)."""
    return {f["properties"]["candidate_id"]: {**f["properties"],
                                              "area_km2": Decimal(str(f["properties"]["area_km2"]))}
            for f in catalog["features"]}


def price(ids: Iterable[str], items: Mapping[str, Mapping[str, Any]],
          cfg: Mapping[str, Any], as_of: date) -> list[dict[str, Any]]:
    ids = list(ids)
    return price_basket([items[c] for c in ids], cfg, as_of=as_of) if ids else []


def other_cost(n_zones: int, other_cfg: Mapping[str, Any]) -> Decimal:
    """C_other = C_compute + C_review × число купленных позиций (ставки — configs/run.yaml)."""
    return (Decimal(str(other_cfg["compute_rub"]))
            + Decimal(str(other_cfg["review_per_zone_rub"])) * n_zones).quantize(Decimal("0.01"))


def assessed(loss: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    """asset_id → {el, unc} только для status=ok (прочие не входят ни в охват, ни в знаменатель)."""
    out = {}
    for r in loss:
        if r["status"] != "ok":
            continue
        unc = r.get("uncertainty")
        out[r["asset_id"]] = {"el": float(r["expected_loss_rub"]),
                              "unc": float(unc) if unc not in (None, "") else 0.0}
    return out


def select_b(catalog: Mapping[str, Any], eligible: set[str]) -> list[str]:
    return sorted(f["properties"]["candidate_id"] for f in catalog["features"]
                  if f["properties"].get("zone_kind") == "wide"
                  and f["properties"].get("offer_id", "S1N") == "S1N"
                  and f["properties"]["candidate_id"] in eligible)


def _avail_key(item: Mapping[str, Any]) -> str:
    return str(item.get("available_at") or "")


def _run_c(items: Mapping[str, Mapping[str, Any]], cov: Mapping[str, list[str]],
           el: Mapping[str, Mapping[str, float]], budget: Decimal, cfg: Mapping[str, Any],
           as_of: date, required: list[str], start: str | None) -> dict[str, Any]:
    """Один проход отбора C: социальная фаза → (опц.) стартовая позиция ``start`` →
    жадное добавление по EL на рубль прироста цены всей корзины."""
    chosen: list[str] = []
    reasons: dict[str, str] = {}
    covered: set[str] = set()
    total = Decimal("0.00")
    pool = sorted(c for c, a in cov.items() if any(x in el for x in a))
    step = 0

    def best_candidate(mode: str, targets: set[str] | None = None, only: str | None = None):
        best = None
        for cid in ([only] if only else pool):
            if cid in chosen:
                continue
            new = sorted(a for a in cov[cid] if a in el and a not in covered)
            if not new or (targets is not None and not set(new) & targets):
                continue
            new_total = basket_total(price(chosen + [cid], items, cfg, as_of))
            if new_total > budget:
                continue
            d_cost = new_total - total
            d_el = sum(el[a]["el"] for a in new)
            d_unc = sum(el[a]["unc"] for a in new)
            ratio = d_el / float(d_cost) if d_cost > 0 else float("inf")
            avail = _avail_key(items[cid])
            if mode == "social":  # cheapest basket increment per required asset covered
                key = (d_cost / len(set(new) & targets), -ratio, avail, cid)
            else:
                key = (-ratio, -d_unc, d_cost, avail, cid)
            if best is None or key < best[0]:
                best = (key, cid, new, new_total, d_cost, d_el, d_unc)
        return best

    def take(best, why):
        nonlocal step, total
        _, cid, new, new_total, d_cost, d_el, d_unc = best
        step += 1
        chosen.append(cid)
        covered.update(new)
        reasons[cid] = (f"шаг {step}{why}: новый охват {', '.join(new)} — EL +{d_el:.2f} руб., "
                        f"неопределённость +{d_unc:.3f}; цена корзины +{d_cost} руб. "
                        f"(итого {new_total} из бюджета {budget}); "
                        f"EL на рубль {d_el / float(d_cost) if d_cost > 0 else float('inf'):.3f}; "
                        f"доступна {items[cid].get('available_at')} — к сроку решения")
        total = new_total

    social: dict[str, Any] = {"required_asset_ids": required}
    if required:
        while (targets := set(required) - covered) and (b := best_candidate("social", targets)):
            take(b, " (социальное ограничение)")
        missing = sorted(set(required) - covered)
        social.update(satisfied=not missing, uncovered_required=missing,
                      note=("все обязательные объекты покрыты" if not missing else
                            "бюджета или пригодных позиций не хватило для обязательных объектов"))
    if start is not None:
        if start not in pool or not (b := best_candidate("ratio", only=start)):
            return {"chosen": [], "total": None, "covered_el": -1.0}  # start does not fit
        take(b, " (стартовая позиция прохода)")
    while (b := best_candidate("ratio")):
        take(b, "")
    return {"chosen": chosen, "reasons": reasons, "social": social, "total": total,
            "covered_el": sum(el[a]["el"] for a in covered)}


def select_c(items: Mapping[str, Mapping[str, Any]], cov: Mapping[str, list[str]],
             el: Mapping[str, Mapping[str, float]], budget: Decimal,
             cfg: Mapping[str, Any], as_of: date,
             required: Iterable[str] = ()) -> tuple[list[str], dict[str, str], dict[str, Any]]:
    """Отбор C: чисто жадный проход и проходы «старт с каждой пригодной позиции, далее
    жадно» (частичный перебор первой позиции — защищает жадное правило от мелких дешёвых
    шагов, закрывающих бюджет перед крупной позицией). Берётся проход с наибольшим уникальным
    EL; при равенстве — дешевле, затем чисто жадный, затем меньший id стартовой позиции.

    ``required`` — asset_id, которые опция социального ограничения требует покрыть первыми.
    Возвращает (состав в порядке выбора, причины по позициям, сведения об отборе:
    ``social`` — итог ограничения, ``variant`` — выбранный проход, ``alternatives`` — EL и
    цена чисто жадного и выбранного проходов, ``n_passes`` — число допустимых проходов).
    """
    req = sorted(a for a in set(required) if a in el)
    pool = sorted(c for c, a in cov.items() if any(x in el for x in a))
    runs = {"greedy": _run_c(items, cov, el, budget, cfg, as_of, req, None)}
    for cid in pool:
        r = _run_c(items, cov, el, budget, cfg, as_of, req, cid)
        if r["total"] is not None:
            runs[f"start:{cid}"] = r
    variant = min(runs, key=lambda k: (-round(runs[k]["covered_el"], 6), runs[k]["total"],
                                       k != "greedy", k))
    r = runs[variant]
    info = {"social": r["social"], "variant": variant, "n_passes": len(runs),
            "alternatives": {k: {"chosen": runs[k]["chosen"],
                                 "data_cost_rub": f"{runs[k]['total']:f}",
                                 "covered_expected_loss_rub": runs[k]["covered_el"]}
                             for k in dict.fromkeys(["greedy", variant])}}
    return r["chosen"], r["reasons"], info


def comparison_row(strategy: str, ids: list[str], rows: list[Mapping[str, Any]],
                   budget: Decimal, cov: Mapping[str, list[str]],
                   el: Mapping[str, Mapping[str, float]],
                   other_cfg: Mapping[str, Any]) -> dict[str, Any]:
    data = basket_total(rows)
    other = other_cost(len(ids), other_cfg)
    covered = sorted(a for a in {x for c in ids for x in cov.get(c, ())} if a in el)
    covered_el = sum(el[a]["el"] for a in covered)
    denom = sum(v["el"] for v in el.values())
    uncs = {a: v["unc"] for a, v in el.items()}
    if strategy == "A":
        res_u = sum(uncs.values()) if uncs else None
        status, basis = ("baseline", UNCERTAINTY_A_BASIS) if uncs else (
            "not_estimated", "нет оценённых объектов")
    elif uncs:
        res_u = sum(u for a, u in uncs.items() if a not in covered)
        status, basis = "scenario", UNCERTAINTY_BC_BASIS
    else:
        res_u, status, basis = None, "not_estimated", "нет оценённых объектов"
    return {
        "strategy": strategy, "data_cost_rub": data, "other_cost_rub": other,
        "decision_cost_rub": data + other, "budget_rub": budget,
        "budget_feasible": data <= budget,
        "covered_expected_loss_rub": covered_el,
        "coverage_share": covered_el / denom if denom > 0 else None,
        "residual_uncertainty": res_u, "uncertainty_status": status,
        "uncertainty_basis": basis, "covered_asset_ids": " ".join(covered),
    }


def compute_plan(catalog: Mapping[str, Any], cov: Mapping[str, list[str]], eligible: set[str],
                 loss: Iterable[Mapping[str, Any]], budget: Decimal, cfg: Mapping[str, Any],
                 as_of: date, *, other_cfg: Mapping[str, Any],
                 required: Iterable[str] = ()) -> dict[str, Any]:
    """Полный пересчёт A/B/C для бюджета: plans, plan_rows, comparison, reasons, social."""
    items = catalog_items(catalog)
    el = assessed(loss)
    budget = Decimal(str(budget)).quantize(Decimal("0.01"))
    cov_ok = {c: a for c, a in cov.items() if c in eligible}
    plans = {"A": [], "B": select_b(catalog, eligible)}
    plans["C"], c_reasons, info = select_c(items, cov_ok, el, budget, cfg, as_of, required)
    priced = {s: price(plans[s], items, cfg, as_of) for s in "ABC"}
    plan_rows = [{"strategy": s, **r} for s in "BC" for r in priced[s]]
    comparison = [comparison_row(s, plans[s], priced[s], budget, cov, el, other_cfg)
                  for s in "ABC"]
    reasons = {"B": {c: RULE_B for c in plans["B"]}, "C": c_reasons}
    if not plans["B"]:
        reasons["B_empty"] = ("B пуста: позиция W-CHIP (S1N) непригодна или недоступна к "
                              "decision_deadline — широкого покрытия в каталоге нет")
    return {"plans": plans, "plan_rows": plan_rows, "comparison": comparison,
            "reasons": reasons, "social_constraint": info["social"],
            "c_selection": {"variant": info["variant"], "alternatives": info["alternatives"]}}


def with_base_rate(cfg: Mapping[str, Any], factor: Decimal) -> dict[str, Any]:
    new = copy.deepcopy(dict(cfg))
    rate = Decimal(str(cfg["base_rate"]["rub_per_km2"])) * factor
    new["base_rate"]["rub_per_km2"] = str(rate.quantize(Decimal("0.01")))
    return new


def sensitivity(catalog: Mapping[str, Any], cov: Mapping[str, list[str]], eligible: set[str],
                loss: list[Mapping[str, Any]], budget: Decimal, cfg: Mapping[str, Any],
                as_of: date, budgets: Iterable[Decimal],
                rate_factors: Iterable[Decimal], *, other_cfg: Mapping[str, Any],
                required: Iterable[str] = (),
                social_ids: Iterable[str] = ()) -> list[dict[str, Any]]:
    """Строки чувствительности плана: C при разных бюджетах; B и C при разной ставке Б;
    C с включённым социальным ограничением (если заданы ``social_ids``)."""
    kw = {"other_cfg": other_cfg, "required": list(required)}
    base = compute_plan(catalog, cov, eligible, loss, budget, cfg, as_of, **kw)
    base_c = base["plans"]["C"]
    rows = []

    def unc_note(c):
        return (f"остаточная неопределённость {c['uncertainty_status']} "
                f"{'' if c['residual_uncertainty'] is None else round(c['residual_uncertainty'], 3)}"
                f" (сценарий: без новых наблюдений фактическое снижение не заявляется)")

    def row(sid, strategy, changed, res, note):
        cmp_ = next(r for r in res["comparison"] if r["strategy"] == strategy)
        ids = res["plans"][strategy]
        feasible = cmp_["budget_feasible"]
        status = ("counterfactual_over_budget" if not feasible
                  else "empty_plan" if not ids else "within_budget")
        rows.append({"scenario_id": sid, "strategy": strategy,
                     "changed_inputs_json": json.dumps(changed, ensure_ascii=False, sort_keys=True),
                     "decision_cost_rub": cmp_["decision_cost_rub"],
                     "covered_expected_loss_rub": cmp_["covered_expected_loss_rub"],
                     "result_status": status,
                     "interpretation": note(ids, cmp_)})

    def diff(ids):
        add = [c for c in ids if c not in base_c]
        rem = [c for c in base_c if c not in ids]
        return (f"состав C [{', '.join(ids) or 'пусто'}]; относительно базы +[{', '.join(add)}] "
                f"−[{', '.join(rem)}]")

    for b in budgets:
        b = Decimal(str(b)).quantize(Decimal("0.01"))
        res = compute_plan(catalog, cov, eligible, loss, b, cfg, as_of, **kw)
        row(f"budget_{b}", "C", {"budget_rub": float(b)}, res,
            lambda ids, c, b=b: (f"{diff(ids)}; данные {c['data_cost_rub']} руб. ≤ бюджета {b}; "
                            f"уникальный охват EL {c['covered_expected_loss_rub']:.2f} руб.; "
                            f"{unc_note(c)}; p, маска, порог и портфель не менялись"))
    for f in rate_factors:
        f = Decimal(str(f))
        cfg_f = with_base_rate(cfg, f)
        res = compute_plan(catalog, cov, eligible, loss, budget, cfg_f, as_of, **kw)
        changed = {"base_rate_rub_km2": float(cfg_f["base_rate"]["rub_per_km2"]),
                   "base_rate_factor": float(f), "base_rate_status": "scenario"}
        row(f"rate_x{f}", "C", changed, res,
            lambda ids, c, f=f: (f"ставка Б × {f}: {diff(ids)}; данные {c['data_cost_rub']} руб. при "
                            f"бюджете {budget}; охват EL {c['covered_expected_loss_rub']:.2f} руб.; "
                            f"{unc_note(c)}"))
        row(f"rate_x{f}", "B", changed, res,
            lambda ids, c, f=f: (f"ставка Б × {f}: B [{', '.join(ids) or 'пусто'}] стоит "
                            f"{c['data_cost_rub']} руб. при бюджете {budget} — "
                            + ("в бюджете" if c["budget_feasible"] else "контрфактическая")
                            + f"; {unc_note(c)}"))
    social_ids = sorted(social_ids)
    if social_ids and not list(required):
        res = compute_plan(catalog, cov, eligible, loss, budget, cfg, as_of,
                           other_cfg=other_cfg, required=social_ids)
        sc = res["social_constraint"]
        row("social_constraint_on", "C",
            {"social_constraint": True, "social_asset_ids": social_ids}, res,
            lambda ids, c: (f"социальное ограничение включено (сначала покрыть "
                            f"{', '.join(social_ids)}): {diff(ids)}; выполнено: "
                            f"{'да' if sc.get('satisfied') else 'нет'}; данные "
                            f"{c['data_cost_rub']} руб. при бюджете {budget}; охват EL "
                            f"{c['covered_expected_loss_rub']:.2f} руб.; {unc_note(c)}"))
    elif list(required):  # constraint enabled in config: show the plan without it
        res = compute_plan(catalog, cov, eligible, loss, budget, cfg, as_of,
                           other_cfg=other_cfg, required=[])
        row("social_constraint_off", "C", {"social_constraint": False}, res,
            lambda ids, c: (f"социальное ограничение выключено (в конфиге включено): "
                            f"{diff(ids)}; данные {c['data_cost_rub']} руб. при бюджете "
                            f"{budget}; охват EL {c['covered_expected_loss_rub']:.2f} руб.; "
                            f"{unc_note(c)}"))
    return rows
