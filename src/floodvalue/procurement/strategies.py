"""Стратегии A/B/C над одним каталогом и одной функцией цены (чистые функции, D-023).

* A — открытые данные: платных зон нет.
* B — широкая закупка по заранее объявленному правилу :data:`RULE_B`; бюджет не
  ограничивает состав, при превышении ``budget_feasible=false`` (контрфактический вариант).
* C — жадный отбор :data:`RULE_C`: на каждом шаге корзина переоценивается целиком
  (скидка Р по группе), берётся зона с наибольшим приростом уникально покрытого EL на рубль
  прироста цены корзины, пока корзина ≤ бюджета.

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

RULE_B = ("B: все зоны сплошного покрытия (zone_kind=wide) с data_role=event_observation и "
          "available_at ≤ decision_deadline — новая съёмка всей валидной области чипа, "
          "независимо от бюджета и EL")
RULE_C = ("C: только зоны event_observation, доступные к decision_deadline и покрывающие "
          "хотя бы один оценённый объект без охвата; жадно по max(прирост уникального EL / "
          "прирост цены всей корзины), при равенстве — больше суммарная неопределённость "
          "новых объектов, затем меньше прирост цены, затем candidate_id; корзина "
          "переоценивается целиком (скидка по группе) при каждом добавлении; стоп — когда "
          "ни одна зона не помещается в остаток бюджета или не добавляет EL")

# Прочие затраты решения (task.md: C_decision = C_data + C_compute + C_review + …).
# Сценарные статьи команды, не измерены; C_field и C_action_errors не оцениваются (нет
# модели действия). TODO: перенести в configs/run.yaml, когда он будет в main.
OTHER_COST = {
    "compute_rub": Decimal("1000.00"),      # обработка S1 и выпуск карты — у всех стратегий
    "review_per_zone_rub": Decimal("500.00"),  # 0,5 ч аналитика × 1000 руб/ч на купленную зону
    "status": "scenario",
    "basis": ("C_other = C_compute (1000 руб: обработка S1 и выпуск карты, одинаково для "
              "A/B/C) + C_review (500 руб × число купленных зон: 0,5 ч аналитика × 1000 руб/ч "
              "на дешифрирование и сверку с объектами); сценарные ставки команды; выезды и "
              "ошибки действий не оцениваются"),
}

UNCERTAINTY_A_BASIS = ("сумма поля uncertainty asset_loss.csv по объектам status=ok "
                       "(исходная неопределённость карты в точках объектов, без новых наблюдений)")
UNCERTAINTY_BC_BASIS = ("новые наблюдения не получены и не обработаны: снижение "
                        "неопределённости от закупки не оценивается и не заявляется")


def catalog_items(catalog: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """candidate_id → позиция для :func:`price_basket` (свойства каталога)."""
    return {f["properties"]["candidate_id"]: {**f["properties"],
                                              "area_km2": Decimal(str(f["properties"]["area_km2"]))}
            for f in catalog["features"]}


def price(ids: Iterable[str], items: Mapping[str, Mapping[str, Any]],
          cfg: Mapping[str, Any], as_of: date) -> list[dict[str, Any]]:
    ids = list(ids)
    return price_basket([items[c] for c in ids], cfg, as_of=as_of) if ids else []


def other_cost(n_zones: int) -> Decimal:
    return OTHER_COST["compute_rub"] + OTHER_COST["review_per_zone_rub"] * n_zones


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
                  and f["properties"]["candidate_id"] in eligible)


def select_c(items: Mapping[str, Mapping[str, Any]], cov: Mapping[str, list[str]],
             el: Mapping[str, Mapping[str, float]], budget: Decimal,
             cfg: Mapping[str, Any], as_of: date) -> tuple[list[str], dict[str, str]]:
    """Жадный отбор C. Возвращает (состав в порядке выбора, причины по выбранным зонам)."""
    chosen: list[str] = []
    reasons: dict[str, str] = {}
    covered: set[str] = set()
    total = Decimal("0.00")
    pool = sorted(c for c, a in cov.items() if any(x in el for x in a))
    step = 0
    while True:
        best = None
        for cid in pool:
            if cid in chosen:
                continue
            new = sorted(a for a in cov[cid] if a in el and a not in covered)
            if not new:
                continue
            new_total = basket_total(price(chosen + [cid], items, cfg, as_of))
            if new_total > budget:
                continue
            d_cost = new_total - total
            d_el = sum(el[a]["el"] for a in new)
            d_unc = sum(el[a]["unc"] for a in new)
            ratio = d_el / float(d_cost) if d_cost > 0 else float("inf")
            key = (ratio, d_unc, -d_cost, [-ord(ch) for ch in cid])
            if best is None or key > best[0]:
                best = (key, cid, new, new_total, d_cost, d_el, d_unc)
        if best is None:
            break
        _, cid, new, new_total, d_cost, d_el, d_unc = best
        step += 1
        chosen.append(cid)
        covered.update(new)
        reasons[cid] = (f"шаг {step}: новый охват {', '.join(new)} — EL +{d_el:.2f} руб., "
                        f"неопределённость +{d_unc:.3f}; цена корзины +{d_cost} руб. "
                        f"(итого {new_total} из бюджета {budget}); "
                        f"EL на рубль {d_el / float(d_cost) if d_cost > 0 else float('inf'):.3f}; "
                        f"доступна к сроку решения")
        total = new_total
    return chosen, reasons


def comparison_row(strategy: str, ids: list[str], rows: list[Mapping[str, Any]],
                   budget: Decimal, cov: Mapping[str, list[str]],
                   el: Mapping[str, Mapping[str, float]]) -> dict[str, Any]:
    data = basket_total(rows)
    other = other_cost(len(ids))
    covered = sorted(a for a in {x for c in ids for x in cov.get(c, ())} if a in el)
    covered_el = sum(el[a]["el"] for a in covered)
    denom = sum(v["el"] for v in el.values())
    if strategy == "A":
        uncs = [v["unc"] for v in el.values()]
        res_u = sum(uncs) if uncs else None
        status, basis = "baseline", UNCERTAINTY_A_BASIS
    else:
        res_u, status, basis = None, "not_estimated", UNCERTAINTY_BC_BASIS
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
                 as_of: date) -> dict[str, Any]:
    """Полный пересчёт A/B/C для бюджета: plans, plan_rows, comparison, reasons."""
    items = catalog_items(catalog)
    el = assessed(loss)
    budget = Decimal(str(budget)).quantize(Decimal("0.01"))
    plans = {"A": [], "B": select_b(catalog, eligible)}
    plans["C"], c_reasons = select_c(items, cov, el, budget, cfg, as_of)
    priced = {s: price(plans[s], items, cfg, as_of) for s in "ABC"}
    plan_rows = [{"strategy": s, **r} for s in "BC" for r in priced[s]]
    comparison = [comparison_row(s, plans[s], priced[s], budget, cov, el) for s in "ABC"]
    reasons = {"B": {c: RULE_B for c in plans["B"]}, "C": c_reasons}
    return {"plans": plans, "plan_rows": plan_rows, "comparison": comparison,
            "reasons": reasons}


def with_base_rate(cfg: Mapping[str, Any], factor: Decimal) -> dict[str, Any]:
    new = copy.deepcopy(dict(cfg))
    rate = Decimal(str(cfg["base_rate"]["rub_per_km2"])) * factor
    new["base_rate"]["rub_per_km2"] = str(rate.quantize(Decimal("0.01")))
    return new


def sensitivity(catalog: Mapping[str, Any], cov: Mapping[str, list[str]], eligible: set[str],
                loss: list[Mapping[str, Any]], budget: Decimal, cfg: Mapping[str, Any],
                as_of: date, budgets: Iterable[Decimal],
                rate_factors: Iterable[Decimal]) -> list[dict[str, Any]]:
    """Строки чувствительности плана: C при разных бюджетах; B и C при разной ставке Б."""
    base = compute_plan(catalog, cov, eligible, loss, budget, cfg, as_of)
    base_c = base["plans"]["C"]
    rows = []

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
        res = compute_plan(catalog, cov, eligible, loss, b, cfg, as_of)
        row(f"budget_{b}", "C", {"budget_rub": float(b)}, res,
            lambda ids, c, b=b: (f"{diff(ids)}; данные {c['data_cost_rub']} руб. ≤ бюджета {b}; "
                            f"уникальный охват EL {c['covered_expected_loss_rub']:.2f} руб.; "
                            f"неопределённость not_estimated (новых наблюдений нет); p, маска, "
                            f"порог и портфель не менялись"))
    for f in rate_factors:
        f = Decimal(str(f))
        cfg_f = with_base_rate(cfg, f)
        res = compute_plan(catalog, cov, eligible, loss, budget, cfg_f, as_of)
        changed = {"base_rate_rub_km2": float(cfg_f["base_rate"]["rub_per_km2"]),
                   "base_rate_factor": float(f), "base_rate_status": "scenario"}
        row(f"rate_x{f}", "C", changed, res,
            lambda ids, c, f=f: (f"ставка Б × {f}: {diff(ids)}; данные {c['data_cost_rub']} руб. при "
                            f"бюджете {budget}; охват EL {c['covered_expected_loss_rub']:.2f} руб.; "
                            f"неопределённость not_estimated (новых наблюдений нет)"))
        row(f"rate_x{f}", "B", changed, res,
            lambda ids, c, f=f: (f"ставка Б × {f}: B [{', '.join(ids) or 'пусто'}] стоит "
                            f"{c['data_cost_rub']} руб. при бюджете {budget} — "
                            + ("в бюджете" if c["budget_feasible"] else "контрфактическая")
                            + "; неопределённость not_estimated (новых наблюдений нет)"))
    return rows
