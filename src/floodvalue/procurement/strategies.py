"""Стратегии A/B/C над одним каталогом и одной функцией цены (чистые функции, D-023, D-038).

* A — открытые данные: платных зон нет.
* B — широкая закупка по заранее объявленному правилу :data:`RULE_B`; бюджет не
  ограничивает состав, при превышении ``budget_feasible=false`` (контрфактический вариант).
* C — жадный отбор :data:`RULE_C`: на каждом шаге корзина переоценивается целиком
  (скидка Р по группе), берётся зона с наибольшим приростом уникально покрытой цены ошибки
  решения R = V·q·min(p, 1−p) (:func:`decision_risk_rub`, D-048) на рубль прироста цены
  корзины, пока корзина ≤ бюджета; опция — социальное ограничение покрытия. Прежнее правило
  (EL на рубль, D-038) — ``rule_c="by_EL"``, строка ``C_by_EL`` чувствительности.

Остаточная неопределённость B/C — ``scenario`` (сумма неопределённости непокрытых оценённых
объектов, «если бы снимок снял неопределённость покрытых»), не наблюдение (D-038).

EL, неопределённость и охват берутся из файлов assets (``asset_loss.csv``) и каталога;
p-растр, маска и порог здесь не читаются — смена бюджета их не меняет.
"""
from __future__ import annotations

import copy
import json
from collections.abc import Iterable, Mapping
from datetime import date, datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from floodvalue.procurement import zones
from floodvalue.procurement.pricing import basket_total, price_basket

RULE_B = ("B: одна позиция W-CHIP (zone_kind=wide, основное предложение S1N — новая SAR 1 м) — "
          "сплошная новая съёмка всей области чипа, если она доступна к decision_deadline; "
          "состав не зависит от бюджета, EL и цены; сверх бюджета — контрфактическая")
RULE_C = ("C (D-048): цель — ожидаемая цена ошибки решения R = V·q·min(p, 1−p) (сценарий, "
          "руб.) по объектам status=ok; p — калиброванная вероятность из flood_probability.tif "
          "(EL = p·V·q, V·q = EL / p). R — ценность полной информации (EVPI) при допущениях: "
          "решение по объекту бинарное (выезд/нет); цена ошибки симметрична и равна V·q; снимок "
          "идеальный (снимает неопределённость покрытого объекта); p откалибрована. Решение по "
          "объекту переключается при p = 0,5, а маска — при пороге 0,37 (порог маски выбран по "
          "IoU карты, не по цене ошибки). R — адресуемая, а не снятая цена ошибки: не фактическое "
          "снижение потерь, новые снимки не получены. min(p, 1−p) — монотонная функция энтропии "
          "калиброванной p (показатель неопределённости D-039), переведённая в рубли. "
          "Позиции: только event_observation, пригодные и доступные к decision_deadline и "
          "покрывающие хотя бы один оценённый объект без охвата; проходы — чисто жадный и «старт "
          "с каждой пригодной позиции, далее жадно»; берётся проход с наибольшей уникально "
          "покрытой R (при равенстве — больше уникальный EL, затем дешевле, затем чисто жадный, "
          "затем id старта); жадный шаг — max(прирост уникальной R / прирост цены всей корзины), "
          "при равенстве — больше прирост EL, больше объектов mask_unstable, шире el_high−el_low "
          "(D-039), больше неопределённость, меньше прирост цены, раньше available_at, меньше "
          "candidate_id; корзина переоценивается целиком (скидка Р по группе) при каждой пробе; "
          "позиция с приростом R = 0 не покупается (низкая ценность информации: исход по объекту "
          "почти известен, p ∈ {0; 1}); стоп — когда ни одна позиция не помещается в остаток "
          "бюджета или не добавляет R. Где входят факторы: EL и неопределённость — числитель "
          "(R); цена — знаменатель (прирост цены всей корзины) и лимит бюджета; срок — фильтр "
          "available_at ≤ decision_deadline и тай-брейк. Ограничение: объекты partial/no_data "
          "в R не входят (p нет — их проверяет осмотр, D-035; в CSV пусто, не 0). "
          "covered_expected_loss_rub и coverage_share считаются как прежде (уникальный EL). "
          "Прежнее правило (max EL на рубль, D-038) — строка C_by_EL в sensitivity.csv. "
          "Опция social_constraint (configs/run.yaml): сначала покрыть объекты социальных "
          "классов (min прироста цены корзины на один новый обязательный объект, R = 0 не "
          "исключает), затем общий отбор; рубли не умножаются")
RULE_C_BY_EL = ("C_by_EL (прежнее правило D-038, для сравнения): max(прирост уникального EL / "
                "прирост цены всей корзины), проход — с наибольшим уникальным EL")
RULES_C = ("risk", "by_EL")

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


def decision_risk_rub(p: float, value_rub: float, vulnerability: float) -> float:
    """Ожидаемая цена ошибки решения R = V·q·min(p, 1−p), руб. (сценарий, D-048).

    EVPI бинарного решения при симметричной цене ошибки V·q, идеальном снимке и откалиброванной
    p: 0,5·V·q при p = 0,5; 0 при p ∈ {0; 1}; симметрична по p ↔ 1−p."""
    return float(value_rub) * float(vulnerability) * min(float(p), 1.0 - float(p))


def decision_risk_from_el(el: float, p: float) -> float:
    """R из полей asset_loss.csv: V·q = EL / p ⇒ R = EL при p ≤ 0,5, EL·(1−p)/p при p > 0,5
    (при p = 0 EL = 0 и R = 0)."""
    el, p = float(el), float(p)
    if p <= 0.5:
        return el
    return el * (1.0 - p) / p


def _num(v: Any) -> float | None:
    return None if v in (None, "") else float(v)


def assessed(loss: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    """asset_id → {el, unc, risk, unstable, el_range} только для status=ok (прочие не входят
    ни в охват, ни в знаменатель, ни в R). ``risk`` — :func:`decision_risk_from_el`;
    ``unstable`` — 1 при mask_unstable=true; ``el_range`` — el_high_rub − el_low_rub (D-039,
    пусто → 0) — только тай-брейк."""
    out = {}
    for r in loss:
        if r["status"] != "ok":
            continue
        el = float(r["expected_loss_rub"])
        lo, hi = _num(r.get("el_low_rub")), _num(r.get("el_high_rub"))
        out[r["asset_id"]] = {
            "el": el, "unc": _num(r.get("uncertainty")) or 0.0,
            "risk": decision_risk_from_el(el, float(r["p_flood"])),
            "unstable": 1 if str(r.get("mask_unstable", "")).lower() == "true" else 0,
            "el_range": hi - lo if lo is not None and hi is not None else 0.0}
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
           as_of: date, required: list[str], start: str | None,
           rule_c: str = "risk") -> dict[str, Any]:
    """Один проход отбора C: социальная фаза → (опц.) стартовая позиция ``start`` →
    жадное добавление по R (``rule_c="risk"``) или EL (``"by_EL"``) на рубль прироста цены
    всей корзины."""
    by_el = rule_c == "by_EL"
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
            d_r = sum(el[a]["risk"] for a in new)
            gain = d_el if by_el else d_r
            if mode != "social" and not by_el and d_r <= 0:
                continue  # no value of information: outcome for the assets is (almost) known
            ratio = gain / float(d_cost) if d_cost > 0 else float("inf")
            avail = _avail_key(items[cid])
            if mode == "social":  # cheapest basket increment per required asset covered
                key = (d_cost / len(set(new) & targets), -ratio, avail, cid)
            elif by_el:
                key = (-ratio, -d_unc, d_cost, avail, cid)
            else:
                key = (-ratio, -d_el, -sum(el[a].get("unstable", 0) for a in new),
                       -sum(el[a].get("el_range", 0.0) for a in new), -d_unc, d_cost, avail, cid)
            if best is None or key < best[0]:
                best = (key, cid, new, new_total, d_cost, d_el, d_unc, d_r)
        return best

    def take(best, why):
        nonlocal step, total
        _, cid, new, new_total, d_cost, d_el, d_unc, d_r = best
        step += 1
        chosen.append(cid)
        covered.update(new)
        per_rub = ((d_el if by_el else d_r) / float(d_cost) if d_cost > 0 else float("inf"))
        reasons[cid] = (f"шаг {step}{why}: новый охват {', '.join(new)} — цена ошибки решения "
                        f"+{d_r:.2f} руб. (сценарий, R = V·q·min(p, 1−p)), EL +{d_el:.2f} руб., "
                        f"неопределённость +{d_unc:.3f}; цена корзины +{d_cost} руб. "
                        f"(итого {new_total} из бюджета {budget}); "
                        f"{'EL' if by_el else 'цена ошибки решения'} на рубль {per_rub:.3f}; "
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
            "covered_el": sum(el[a]["el"] for a in covered),
            "covered_risk": sum(el[a]["risk"] for a in covered)}


def select_c(items: Mapping[str, Mapping[str, Any]], cov: Mapping[str, list[str]],
             el: Mapping[str, Mapping[str, float]], budget: Decimal,
             cfg: Mapping[str, Any], as_of: date,
             required: Iterable[str] = (),
             rule_c: str = "risk") -> tuple[list[str], dict[str, str], dict[str, Any]]:
    """Отбор C: чисто жадный проход и проходы «старт с каждой пригодной позиции, далее
    жадно» (частичный перебор первой позиции — защищает жадное правило от мелких дешёвых
    шагов, закрывающих бюджет перед крупной позицией). ``rule_c="risk"`` (D-048): берётся
    проход с наибольшей уникально покрытой ценой ошибки решения R, при равенстве — больше
    уникальный EL; ``"by_EL"`` (прежнее, D-038) — наибольший уникальный EL. Далее — дешевле,
    затем чисто жадный, затем меньший id стартовой позиции.

    ``required`` — asset_id, которые опция социального ограничения требует покрыть первыми.
    Возвращает (состав в порядке выбора, причины по позициям, сведения об отборе:
    ``social`` — итог ограничения, ``variant`` — выбранный проход, ``alternatives`` — EL и
    цена чисто жадного и выбранного проходов, ``n_passes`` — число допустимых проходов).
    """
    if rule_c not in RULES_C:
        raise ValueError(f"rule_c={rule_c!r}: ожидается одно из {RULES_C}")
    req = sorted(a for a in set(required) if a in el)
    pool = sorted(c for c, a in cov.items() if any(x in el for x in a))
    runs = {"greedy": _run_c(items, cov, el, budget, cfg, as_of, req, None, rule_c)}
    for cid in pool:
        r = _run_c(items, cov, el, budget, cfg, as_of, req, cid, rule_c)
        if r["total"] is not None:
            runs[f"start:{cid}"] = r

    def pass_key(k):
        r = runs[k]
        head = ((-round(r["covered_el"], 6),) if rule_c == "by_EL" else
                (-round(r["covered_risk"], 6), -round(r["covered_el"], 6)))
        return (*head, r["total"], k != "greedy", k)

    variant = min(runs, key=pass_key)
    r = runs[variant]
    info = {"social": r["social"], "variant": variant, "n_passes": len(runs),
            "alternatives": {k: {"chosen": runs[k]["chosen"],
                                 "data_cost_rub": f"{runs[k]['total']:f}",
                                 "covered_expected_loss_rub": runs[k]["covered_el"],
                                 "addressed_decision_risk_rub": runs[k]["covered_risk"]}
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
    covered_r = sum(el[a]["risk"] for a in covered)
    denom_r = sum(v["risk"] for v in el.values())
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
        "addressed_decision_risk_rub": covered_r,
        "decision_risk_share": covered_r / denom_r if denom_r > 0 else None,
        "residual_uncertainty": res_u, "uncertainty_status": status,
        "uncertainty_basis": basis, "covered_asset_ids": " ".join(covered),
    }


def compute_plan(catalog: Mapping[str, Any], cov: Mapping[str, list[str]], eligible: set[str],
                 loss: Iterable[Mapping[str, Any]], budget: Decimal, cfg: Mapping[str, Any],
                 as_of: date, *, other_cfg: Mapping[str, Any],
                 required: Iterable[str] = (), rule_c: str = "risk") -> dict[str, Any]:
    """Полный пересчёт A/B/C для бюджета: plans, plan_rows, comparison, reasons, social.
    ``rule_c`` — правило C (D-048 ``"risk"``; ``"by_EL"`` — прежнее, для сравнения)."""
    items = catalog_items(catalog)
    el = assessed(loss)
    budget = Decimal(str(budget)).quantize(Decimal("0.01"))
    cov_ok = {c: a for c, a in cov.items() if c in eligible}
    plans = {"A": [], "B": select_b(catalog, eligible)}
    plans["C"], c_reasons, info = select_c(items, cov_ok, el, budget, cfg, as_of, required,
                                                 rule_c)
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
    new["base_rate"]["rub_per_km2"] = str(rate.quantize(Decimal("0.01"), ROUND_HALF_UP))
    return new


def coverage_at(catalog: Mapping[str, Any], ok_ids: Iterable[str],
                deadline: datetime) -> tuple[dict[str, list[str]], set[str]]:
    """Охват и пригодные позиции при другом сроке решения — без перестройки каталога:
    ``covers_asset_ids`` позиции (``covers`` при построении) ∩ оценённые объекты, если
    :func:`zones.exclusion_reason` при этом сроке пуста (event_observation, available_at ≤
    срока, облачность подтверждена)."""
    ok = set(ok_ids)
    cov: dict[str, list[str]] = {}
    eligible: set[str] = set()
    for f in catalog["features"]:
        p = f["properties"]
        if zones.exclusion_reason(p, deadline) is None:
            eligible.add(p["candidate_id"])
            cov[p["candidate_id"]] = sorted(a for a in p["covers_asset_ids"] if a in ok)
        else:
            cov[p["candidate_id"]] = []
    return cov, eligible


def sensitivity(catalog: Mapping[str, Any], cov: Mapping[str, list[str]], eligible: set[str],
                loss: list[Mapping[str, Any]], budget: Decimal, cfg: Mapping[str, Any],
                as_of: date, budgets: Iterable[Decimal],
                rate_factors: Iterable[Decimal], *, other_cfg: Mapping[str, Any],
                required: Iterable[str] = (),
                social_ids: Iterable[str] = (),
                decision_at: datetime | None = None,
                deadline_hours: Iterable[int] = ()) -> list[dict[str, Any]]:
    """Строки чувствительности плана (B и C на каждом сценарии): бюджеты ``budgets``;
    ставка Б × ``rate_factors`` (сценарная); срок решения ``decision_at`` + ``deadline_hours``
    (пересчёт пригодности позиций, :func:`coverage_at`); социальное ограничение C вкл/выкл.
    Интерпретация строки — изменение корзины относительно базы (текущие бюджет, ставка, срок),
    стоимость данных и решения, уникально покрытый EL, статус остаточной неопределённости."""
    kw = {"other_cfg": other_cfg, "required": list(required)}
    budget = Decimal(str(budget)).quantize(Decimal("0.01"))
    base = compute_plan(catalog, cov, eligible, loss, budget, cfg, as_of, **kw)
    rows = []

    def unc_note(c, ids):
        val = '' if c['residual_uncertainty'] is None else round(c['residual_uncertainty'], 3)
        if not ids:  # nothing bought: the residual is the initial one, as for A
            return (f"остаточная неопределённость {val}: закупки нет — неопределённость исходная "
                    f"(как A, baseline) (без новых наблюдений фактическое снижение не заявляется)")
        return (f"остаточная неопределённость {c['uncertainty_status']} {val}"
                f" (без новых наблюдений фактическое снижение не заявляется)")

    def row(sid, strategy, changed, res, head, tail=""):
        cmp_ = next(r for r in res["comparison"] if r["strategy"] == strategy)
        base_cmp = next(r for r in base["comparison"] if r["strategy"] == strategy)
        ids, base_ids = res["plans"][strategy], base["plans"][strategy]
        feasible = cmp_["budget_feasible"]
        status = ("counterfactual_over_budget" if not feasible
                  else "empty_plan" if not ids else "within_budget")
        add = [c for c in ids if c not in base_ids]
        rem = [c for c in base_ids if c not in ids]
        d_el = cmp_["covered_expected_loss_rub"] - base_cmp["covered_expected_loss_rub"]
        fit = (f"данные {cmp_['data_cost_rub']} руб. > бюджета {cmp_['budget_rub']} — "
               f"контрфактическая (сверх бюджета не закупается)" if not feasible else
               f"данные {cmp_['data_cost_rub']} руб. ≤ бюджета {cmp_['budget_rub']}")
        text = (f"{head}: {strategy} [{', '.join(ids) or 'пусто'}], относительно базы "
                f"+[{', '.join(add)}] −[{', '.join(rem)}]; {fit}; decision_cost "
                f"{cmp_['decision_cost_rub']} руб.; уникально покрытый EL "
                f"{cmp_['covered_expected_loss_rub']:.2f} руб. ({d_el:+.2f} к базе); "
                f"адресуемая цена ошибки решения {cmp_['addressed_decision_risk_rub']:.2f} руб. "
                f"(сценарий, не снижение потерь); "
                f"{unc_note(cmp_, ids)}{tail}")
        rows.append({"scenario_id": sid, "strategy": strategy,
                     "changed_inputs_json": json.dumps(changed, ensure_ascii=False, sort_keys=True),
                     "decision_cost_rub": cmp_["decision_cost_rub"],
                     "covered_expected_loss_rub": cmp_["covered_expected_loss_rub"],
                     "result_status": status, "interpretation": text})

    same_map = "; p, маска, порог и портфель не менялись"
    res = compute_plan(catalog, cov, eligible, loss, budget, cfg, as_of, **kw, rule_c="by_EL")
    row("C_by_EL", "C", {"rule_C": "by_EL"}, res,
        f"прежнее правило C — max EL на рубль (D-038) вместо цены ошибки решения (D-048), "
        f"бюджет {budget} руб.", same_map)
    for b in budgets:
        b = Decimal(str(b)).quantize(Decimal("0.01"))
        res = compute_plan(catalog, cov, eligible, loss, b, cfg, as_of, **kw)
        for s in "CB":
            row(f"budget_{b}", s, {"budget_rub": float(b)}, res,
                f"бюджет {b} руб." + (" (состав B от бюджета не зависит)" if s == "B" else ""),
                same_map)
    for f in rate_factors:
        f = Decimal(str(f))
        cfg_f = with_base_rate(cfg, f)
        res = compute_plan(catalog, cov, eligible, loss, budget, cfg_f, as_of, **kw)
        rate = cfg_f["base_rate"]["rub_per_km2"]
        changed = {"base_rate_rub_km2": float(rate), "base_rate_factor": float(f),
                   "base_rate_status": "scenario"}
        for s in "CB":
            row(f"rate_x{f}", s, changed, res,
                f"ставка Б × {f} = {rate} руб./км² (сценарная, base_rate_status=scenario; "
                f"бюджет {budget} руб.)")
    if decision_at is not None:
        ok_ids = assessed(loss)
        for h in deadline_hours:
            deadline = decision_at + timedelta(hours=int(h))
            cov_d, elig_d = coverage_at(catalog, ok_ids, deadline)
            res = compute_plan(catalog, cov_d, elig_d, loss, budget, cfg, as_of, **kw)
            changed = {"decision_deadline": deadline.isoformat(),
                       "deadline_hours_after_decision": int(h)}
            cloudy = sum(zones.exclusion_reason(f["properties"], deadline) == zones.CLOUD_NOTE
                         for f in catalog["features"])
            head = (f"срок решения decision_at + {int(h)} ч = {deadline.isoformat()} (сроки "
                    f"доступности предложений сценарные, D-038 — вывод о сроке следует из этого "
                    f"допущения): пригодных позиций {len(elig_d)} (available_at ≤ срока"
                    + (", как в базе" if elig_d == set(eligible) else "")
                    + (f"; ещё {cloudy} доступны к сроку, но охвата не дают (облачность не "
                       f"подтверждена)" if cloudy else "") + ")")
            for s in "CB":
                extra = ""
                if s == "B" and not res["plans"]["B"]:
                    wide = next((f["properties"].get("available_at") for f in catalog["features"]
                                 if f["properties"]["candidate_id"] == "W-CHIP"), None)
                    extra = (f"; правило B (W-CHIP) к сроку не исполнимо — W-CHIP доступна "
                             f"{wide or 'срок не известен'}")
                row(f"deadline_+{int(h)}h", s, changed, res, head + extra, same_map)
    social_ids = sorted(social_ids)
    if social_ids and not list(required):
        res = compute_plan(catalog, cov, eligible, loss, budget, cfg, as_of,
                           other_cfg=other_cfg, required=social_ids)
        sc = res["social_constraint"]
        row("social_constraint_on", "C",
            {"social_constraint": True, "social_asset_ids": social_ids}, res,
            f"социальное ограничение включено (сначала покрыть {', '.join(social_ids)}; "
            f"выполнено: {'да' if sc.get('satisfied') else 'нет'})")
    elif list(required):  # constraint enabled in config: show the plan without it
        res = compute_plan(catalog, cov, eligible, loss, budget, cfg, as_of,
                           other_cfg=other_cfg, required=[])
        row("social_constraint_off", "C", {"social_constraint": False}, res,
            "социальное ограничение выключено (в конфиге включено)")
    return rows
