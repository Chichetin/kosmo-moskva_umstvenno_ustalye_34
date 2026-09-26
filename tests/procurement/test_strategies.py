"""A/B/C (критерий 16): бюджет, контрфактическая B, пересчёт корзины, охват, неопределённость."""
from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

from floodvalue.procurement import strategies, zones
from floodvalue.procurement.pricing import basket_total, load_pricing

CFG = load_pricing()
AS_OF = date(2017, 5, 30)
T0 = datetime(2017, 5, 30, 12, tzinfo=UTC)
DEADLINE = datetime(2017, 5, 31, 0, tzinfo=UTC)
ASSETS = [{"asset_id": f"A{i:02d}", "longitude": 80.0 + 0.02 * i, "latitude": 7.0}
          for i in range(1, 11)]  # 2 км между объектами — буферы не перекрываются
# p = 0,5: цена ошибки решения R = V·q·min(p, 1−p) = EL — порядок C как по EL (D-049)
LOSS = [{"asset_id": a["asset_id"], "status": "ok", "p_flood": "0.5",
         "expected_loss_rub": str(1_000_000 * (11 - i)), "uncertainty": "0.5"}
        for i, a in enumerate(ASSETS, 1)]
OTHER = {"compute_rub": 1000, "review_per_zone_rub": 500}


def ctx(loss=LOSS):
    cat = zones.build_catalog(ASSETS, (79.99, 6.99, 80.21, 7.01), T0, CFG)
    ok = {r["asset_id"] for r in loss if r["status"] == "ok"}
    cov = zones.coverage(cat, [a for a in ASSETS if a["asset_id"] in ok], DEADLINE)
    elig = {f["properties"]["candidate_id"] for f in cat["features"]
            if zones.exclusion_reason(f["properties"], DEADLINE) is None}
    return cat, cov, elig


def plan(budget, loss=LOSS, cfg=CFG, required=()):
    cat, cov, elig = ctx(loss)
    return strategies.compute_plan(cat, cov, elig, loss, Decimal(budget), cfg, AS_OF,
                                   other_cfg=OTHER, required=required)


def cmp_(res, s):
    return next(r for r in res["comparison"] if r["strategy"] == s)


@pytest.mark.parametrize("budget", ["0", "2452.97", "2452.98", "5000", "12000", "100000"])
def test_c_within_budget_and_sum_of_rows(budget):
    res = plan(budget)
    c = cmp_(res, "C")
    rows = [r for r in res["plan_rows"] if r["strategy"] == "C"]
    assert c["data_cost_rub"] == basket_total(rows) <= Decimal(budget)
    assert c["budget_feasible"] is True
    assert [r["candidate_id"] for r in rows] == res["plans"]["C"]
    assert c["decision_cost_rub"] == c["data_cost_rub"] + c["other_cost_rub"]


def test_budget_boundary_inclusive():
    assert plan("2452.98")["plans"]["C"] == ["Z-A01"]  # 1,44 км² × 1703,46 = 2452,98
    assert plan("2452.97")["plans"]["C"] == []


def test_sub_kopeck_budget_not_rounded_up():
    """Бюджет 2452,975 не превращается в 2452,98: C не покупает Z-A01 за 2452,98 (#248)."""
    res = plan("2452.975")
    assert res["plans"]["C"] == [] and cmp_(res, "C")["budget_rub"] <= Decimal("2452.975")


def test_c_orders_by_el_per_rub():
    assert plan("5000")["plans"]["C"] == ["Z-A01", "Z-A02"]


def test_b_counterfactual_when_over_budget():
    res = plan("5000")
    b = cmp_(res, "B")
    assert res["plans"]["B"] == ["W-CHIP"]
    assert b["data_cost_rub"] > Decimal(5000) and b["budget_feasible"] is False
    assert cmp_(plan("1000000"), "B")["budget_feasible"] is True


def test_a_baseline_uncertainty_bc_scenario_residual():
    """A — baseline (сумма uncertainty ok-объектов); B/C — scenario: сумма по НЕпокрытым."""
    res = plan("5000")
    a = cmp_(res, "A")
    assert res["plans"]["A"] == [] and a["data_cost_rub"] == 0
    assert a["uncertainty_status"] == "baseline" and a["residual_uncertainty"] == pytest.approx(5.0)
    assert a["covered_expected_loss_rub"] == 0
    assert not [r for r in res["plan_rows"] if r["strategy"] == "A"]
    b, c = cmp_(res, "B"), cmp_(res, "C")
    assert b["uncertainty_status"] == c["uncertainty_status"] == "scenario"
    assert b["residual_uncertainty"] == pytest.approx(0.0)       # B покрывает все 10
    n_cov = len(c["covered_asset_ids"].split())
    assert n_cov == 2 and c["residual_uncertainty"] == pytest.approx(0.5 * (10 - n_cov))
    assert "не заявляется" in c["uncertainty_basis"] and "сценарий" in c["uncertainty_basis"]


def test_residual_uncertainty_not_estimated_without_assessed_assets():
    loss = [{**r, "status": "no_data", "expected_loss_rub": "", "uncertainty": ""} for r in LOSS]
    res = plan("5000", loss)
    for s in "ABC":
        r = cmp_(res, s)
        assert r["uncertainty_status"] == "not_estimated" and r["residual_uncertainty"] is None


def test_full_repricing_discount_on_whole_basket():
    """Скидка Р по группе целиком: при 8 зонах (11,52 км²) Р < 1 у всех строк, а цена
    корзины меньше суммы одиночных цен."""
    res = plan("25000")
    rows = [r for r in res["plan_rows"] if r["strategy"] == "C"]
    assert len(rows) >= 4
    s = sum(r["area_km2"] for r in rows)
    assert all(r["group_area_km2"] == s for r in rows)
    assert all(r["discount_coef"] < 1 for r in rows)
    assert basket_total(rows) < len(rows) * Decimal("2452.98")


def test_coverage_unique_and_share():
    res = plan("1000000")
    b = cmp_(res, "B")
    assert b["covered_expected_loss_rub"] == 55_000_000
    assert b["coverage_share"] == 1.0
    c = cmp_(plan("2452.98"), "C")
    assert c["covered_expected_loss_rub"] == 10_000_000
    assert c["coverage_share"] == pytest.approx(10 / 55)


def test_coverage_share_empty_when_denominator_zero():
    loss = [{**r, "expected_loss_rub": "0"} for r in LOSS]
    res = plan("100000", loss)
    assert all(r["coverage_share"] is None for r in res["comparison"])


def test_unassessed_assets_not_covered():
    loss = [dict(r) for r in LOSS]
    loss[0] = {"asset_id": "A01", "status": "no_data", "expected_loss_rub": "",
               "uncertainty": ""}
    res = plan("5000", loss)
    assert "Z-A01" not in res["plans"]["C"]
    assert "A01" not in cmp_(plan("1000000", loss), "B")["covered_asset_ids"]


def test_reasons_for_every_c_zone():
    res = plan("12000")
    assert set(res["reasons"]["C"]) == set(res["plans"]["C"])
    assert all("EL" in v for v in res["reasons"]["C"].values())


BUDGETS = [Decimal(b) for b in ("0", "2500", "5000", "10000", "50000")]
RATES = [Decimal(f) for f in ("0.5", "0.75", "1.25", "1.5", "2.0")]


def sens_rows(**kw):
    cat, cov, elig = ctx()
    return strategies.sensitivity(cat, cov, elig, LOSS, Decimal(5000), CFG, AS_OF,
                                  kw.pop("budgets", BUDGETS), kw.pop("rates", RATES),
                                  other_cfg=OTHER, **kw)


def _of(rows, prefix, strategy):
    return [r for r in rows if r["scenario_id"].startswith(prefix) and r["strategy"] == strategy]


def test_sensitivity_rows():
    import json
    rows = sens_rows()
    budgets = _of(rows, "budget_", "C")
    assert len(budgets) == 5 and len(_of(rows, "budget_", "B")) == 5
    assert len(_of(rows, "rate_", "C")) == 5 and len(_of(rows, "rate_", "B")) == 5
    assert all(json.loads(r["changed_inputs_json"]) for r in rows)
    costs = [r["covered_expected_loss_rub"] for r in budgets]
    assert costs == sorted(costs)
    assert budgets[0]["result_status"] == "empty_plan"
    b_rates = _of(rows, "rate_", "B")
    assert all(r["result_status"] == "counterfactual_over_budget" for r in b_rates)
    cheap = next(r for r in _of(rows, "rate_x0.5", "C"))
    base = next(r for r in budgets if r["scenario_id"] == "budget_5000.00")
    assert cheap["covered_expected_loss_rub"] > base["covered_expected_loss_rub"]


def test_sensitivity_changed_inputs_one_required_key_each():
    """Каждая строка бюджета меняет только budget_rub, каждая строка ставки — только
    base_rate_rub_km2 из обязательных ключей контракта §7a; ставка — сценарная."""
    import json

    from floodvalue.contracts import files as F
    required = set(F.SENSITIVITY_REQUIRED_KEYS)
    rows = sens_rows()
    for r in rows:
        changed = json.loads(r["changed_inputs_json"])
        if r["scenario_id"].startswith("budget_"):
            assert set(changed) & required == {"budget_rub"}
        if r["scenario_id"].startswith("rate_"):
            assert set(changed) & required == {"base_rate_rub_km2"}
            assert changed["base_rate_status"] == "scenario"
            factor = Decimal(str(changed["base_rate_factor"]))
            assert Decimal(str(changed["base_rate_rub_km2"])) == (
                Decimal(str(CFG["base_rate"]["rub_per_km2"])) * factor).quantize(Decimal("0.01"))
    assert not any("тариф" in r["interpretation"].lower() for r in rows)


def test_sensitivity_b_on_every_budget_counterfactual_when_over():
    """B не зависит от бюджета по составу; сверх бюджета — counterfactual_over_budget и
    «контрфактическая» в интерпретации, в бюджете — within_budget."""
    rows = sens_rows(budgets=[Decimal("0"), Decimal("5000"), Decimal("10000000")])
    b = {r["scenario_id"]: r for r in _of(rows, "budget_", "B")}
    assert {r["decision_cost_rub"] for r in b.values()} == {b["budget_0.00"]["decision_cost_rub"]}
    assert b["budget_0.00"]["result_status"] == "counterfactual_over_budget"
    assert "контрфактическ" in b["budget_0.00"]["interpretation"]
    assert b["budget_10000000.00"]["result_status"] == "within_budget"
    assert "контрфактическ" not in b["budget_10000000.00"]["interpretation"]


def test_sensitivity_interpretation_content():
    rows = sens_rows()
    for r in rows:
        text = r["interpretation"]
        assert "относительно базы" in text, text
        assert "decision_cost" in text and "EL" in text
        assert "без новых наблюдений фактическое снижение не заявляется" in text
        if r["scenario_id"].startswith("budget_"):
            assert "p, маска, порог и портфель не менялись" in text
        if r["scenario_id"].startswith("rate_"):
            assert "сценарн" in text
            assert "бюджет 5000.00 руб." in text


def test_rate_factor_scales_unit_price_at_fixed_basket():
    """Корзина фиксирована: Р (по всей корзине) не зависит от ставки, unit_price =
    ROUND_HALF_UP(k·Б × О × П × Т × Р) и отличается от k × unit_price(Б) не более чем на
    округление; при k = 2 и 0,5 — ровно пропорционально (1703,46 → 3406,92 / 851,73)."""
    from decimal import ROUND_HALF_UP
    res = plan("100000")
    items = strategies.catalog_items(ctx()[0])
    basket = res["plans"]["C"]
    assert len(basket) > 1
    base = strategies.price(basket, items, CFG, AS_OF)
    for k in RATES:
        rows = strategies.price(basket, items, strategies.with_base_rate(CFG, k), AS_OF)
        for r0, rk in zip(base, rows, strict=True):
            assert rk["discount_coef"] == r0["discount_coef"]
            exact = (rk["base_rate_rub_km2"] * rk["processing_coef"] * rk["usage_coef"]
                     * rk["freshness_coef"] * rk["discount_coef"])
            assert rk["unit_price_rub_km2"] == exact.quantize(Decimal("0.01"), ROUND_HALF_UP)
            assert abs(rk["unit_price_rub_km2"] - k * r0["unit_price_rub_km2"]) <= Decimal("0.01")
    one = [c for c in items if c == "Z-A01"]
    u = {k: strategies.price(one, items, strategies.with_base_rate(CFG, k), AS_OF)[0]
         ["unit_price_rub_km2"] for k in (Decimal(1), Decimal(2), Decimal("0.5"))}
    assert u == {Decimal(1): Decimal("1703.46"), Decimal(2): Decimal("3406.92"),
                 Decimal("0.5"): Decimal("851.73")}


def test_sensitivity_deadline_rows_recompute_eligibility():
    """Срок решения decision_at + h: позиции с available_at > срока не дают охвата.
    +6 ч — ни одна новая съёмка не готова (S1N +9 ч) → C и B пусты; +12 ч — как база."""
    rows = sens_rows(budgets=[], rates=[], decision_at=T0, deadline_hours=[6, 12, 24])
    c = {r["scenario_id"]: r for r in _of(rows, "deadline_", "C")}
    b = {r["scenario_id"]: r for r in _of(rows, "deadline_", "B")}
    assert set(c) == set(b) == {"deadline_+6h", "deadline_+12h", "deadline_+24h"}
    assert c["deadline_+6h"]["covered_expected_loss_rub"] == 0
    assert c["deadline_+6h"]["result_status"] == "empty_plan"
    assert b["deadline_+6h"]["result_status"] == "empty_plan"
    base_c = cmp_(plan("5000"), "C")
    assert c["deadline_+12h"]["covered_expected_loss_rub"] == base_c["covered_expected_loss_rub"]
    import json
    ch = json.loads(c["deadline_+6h"]["changed_inputs_json"])
    assert ch == {"decision_deadline": "2017-05-30T18:00:00+00:00",
                  "deadline_hours_after_decision": 6}


def test_selection_uses_repriced_basket_not_sum_of_singles():
    """8 зон: сумма одиночных цен 19 623,84 > 19 300, а корзина со скидкой по группе
    (11,52 км², Р < 1) — 19 272,96 ≤ 19 300: C обязана взять все 8."""
    res = plan("19300")
    assert len(res["plans"]["C"]) == 8
    assert cmp_(res, "C")["data_cost_rub"] <= Decimal(19300)


def test_overlapping_zones_do_not_double_count():
    cov = {"x": ["A01", "A02"], "y": ["A02"]}
    el = {"A01": {"el": 10.0, "unc": 0.1, "risk": 10.0}, "A02": {"el": 5.0, "unc": 0.2, "risk": 5.0}}
    row = strategies.comparison_row("B", ["x", "y"], [], Decimal(0), cov, el, OTHER)
    assert row["covered_expected_loss_rub"] == 15.0
    assert row["covered_asset_ids"] == "A01 A02"


def test_covered_el_of_kopeck_el_has_no_float_noise():
    # #127: EL in asset_loss.csv is to the kopeck; their float sum 0.1 + 0.2 = 0.30000000000000004
    # must be written as the exact kopeck sum
    cov = {"x": ["A01", "A02"]}
    el = {"A01": {"el": 0.1, "unc": 0.1, "risk": 0.1}, "A02": {"el": 0.2, "unc": 0.2, "risk": 0.2}}
    row = strategies.comparison_row("B", ["x"], [], Decimal(0), cov, el, OTHER)
    assert repr(row["covered_expected_loss_rub"]) == "0.3"
    assert row["coverage_share"] == 1.0


def test_b_takes_only_primary_wide_offer():
    res = plan("1000000")
    assert res["plans"]["B"] == ["W-CHIP"]  # не W-CHIP-S3 / -O05 / -OP и т. д.


def test_c_never_selects_excluded_positions():
    cat, _, elig = ctx()
    res = plan("1000000")
    assert set(res["plans"]["C"]) <= elig
    for f in cat["features"]:
        p = f["properties"]
        if p["candidate_id"] in res["plans"]["C"]:
            assert p["data_role"] == "event_observation" and p.get("cloud_status") is None


def test_c_prefers_primary_offer_on_tie_by_availability():
    """Z-A01 и Z-A01-S3 — одна геометрия и одна цена (Р = 1): берётся более ранняя SAR 1 м."""
    res = plan("2452.98")
    assert res["plans"]["C"] == ["Z-A01"]


def test_tie_break_by_candidate_id_prefers_smaller_id():
    items = {c: {"candidate_id": c, "sensor_type": "sar", "resolution_m": 1,
                 "acquisition_type": "new", "guaranteed_purchase": True,
                 "observation_at": None, "processing_level": "L2", "usage_type": "internal",
                 "area_km2": Decimal("1.44"), "available_at": "2017-05-30T21:00:00+00:00"}
             for c in ("Z-A01", "Z-A01-X")}
    cov = {"Z-A01": ["A01"], "Z-A01-X": ["A01"]}
    el = {"A01": {"el": 1.0, "unc": 0.1, "risk": 1.0}}
    chosen, _, _ = strategies.select_c(items, cov, el, Decimal(10000), CFG, AS_OF)
    assert chosen == ["Z-A01"]


def test_social_constraint_is_coverage_not_rub_multiplier():
    """Социальное ограничение: сначала покрываются обязательные объекты (здесь — самые
    дешёвые по EL A09, A10), EL в рублях не меняется."""
    base = plan("5000")
    soc = plan("5000", required=["A09", "A10"])
    c = cmp_(soc, "C")
    assert {"A09", "A10"} <= set(c["covered_asset_ids"].split())
    assert soc["social_constraint"]["satisfied"] is True
    assert c["covered_expected_loss_rub"] == 1_000_000 + 2_000_000  # чистый EL, без множителя
    assert cmp_(base, "C")["covered_expected_loss_rub"] > c["covered_expected_loss_rub"]
    assert all("социальное" in soc["reasons"]["C"][z] for z in soc["plans"]["C"][:2])


def test_social_constraint_unsatisfied_reported():
    soc = plan("0", required=["A10"])
    assert soc["plans"]["C"] == []
    assert soc["social_constraint"] == {
        "required_asset_ids": ["A10"], "satisfied": False, "uncovered_required": ["A10"],
        "note": "бюджета или пригодных позиций не хватило для обязательных объектов"}


def test_other_cost_from_config():
    res = strategies.compute_plan(*ctx(), LOSS, Decimal(5000), CFG, AS_OF,
                                  other_cfg={"compute_rub": "10", "review_per_zone_rub": "1.5"})
    c = cmp_(res, "C")
    assert c["other_cost_rub"] == Decimal(10) + Decimal("1.5") * len(res["plans"]["C"])
    assert cmp_(res, "A")["other_cost_rub"] == Decimal("10.00")


def test_sensitivity_social_row():
    cat, cov, elig = ctx()
    rows = strategies.sensitivity(cat, cov, elig, LOSS, Decimal(5000), CFG, AS_OF, [], [],
                                  other_cfg=OTHER, social_ids=["A10"])
    (r,) = [r for r in rows if r["scenario_id"] == "social_constraint_on"]
    assert r["strategy"] == "C" and "A10" in r["interpretation"]


def _item(cid, area):
    return {"candidate_id": cid, "sensor_type": "sar", "resolution_m": 1,
            "acquisition_type": "new", "guaranteed_purchase": True, "observation_at": None,
            "processing_level": "L2", "usage_type": "internal", "area_km2": Decimal(area),
            "available_at": "2017-05-30T21:00:00+00:00"}


def test_c_best_single_start_beats_myopic_greedy():
    """Жадный по EL/руб берёт S1 (10 за 2452,98) и S2, после чего BIG (12 за 5110,38) не
    помещается: 11 < 12. Проход со стартом с наибольшего EL выбирает BIG."""
    items = {"BIG": _item("BIG", "3.000"), "S1": _item("S1", "1.440"), "S2": _item("S2", "1.440")}
    cov = {"BIG": ["A01", "A02", "A03"], "S1": ["A01"], "S2": ["A02"]}
    el = {"A01": {"el": 10.0, "unc": 0, "risk": 10.0}, "A02": {"el": 1.0, "unc": 0, "risk": 1.0},
          "A03": {"el": 1.0, "unc": 0, "risk": 1.0}}
    chosen, reasons, info = strategies.select_c(items, cov, el, Decimal("5200"), CFG, AS_OF)
    assert chosen == ["BIG"] and info["variant"] == "start:BIG"
    assert info["alternatives"]["greedy"]["chosen"] == ["S1", "S2"]
    assert info["alternatives"]["greedy"]["covered_expected_loss_rub"] == 11.0
    assert "стартовая" in reasons["BIG"]
    # бюджет на всё: оба прохода дают EL 12, но старт с BIG дешевле (одна позиция вместо трёх)
    chosen, _, info = strategies.select_c(items, cov, el, Decimal("100000"), CFG, AS_OF)
    alt = info["alternatives"]
    assert alt["greedy"]["covered_expected_loss_rub"] == 12.0 and len(alt["greedy"]["chosen"]) == 3
    assert chosen == ["BIG"] and info["variant"] == "start:BIG"
    # равные EL и цена → жадный проход (детерминированный выбор)
    one = {"S1": items["S1"]}
    chosen, _, info = strategies.select_c(one, {"S1": ["A01"]}, el, Decimal("5000"), CFG, AS_OF)
    assert chosen == ["S1"] and info["variant"] == "greedy"


def test_social_phase_takes_cheapest_cover_of_required():
    items = {"BIG": _item("BIG", "3.000"), "S1": _item("S1", "1.440")}
    cov = {"BIG": ["A01", "A02"], "S1": ["A01"]}
    el = {"A01": {"el": 1.0, "unc": 0, "risk": 1.0}, "A02": {"el": 1.0, "unc": 0, "risk": 1.0}}
    chosen, reasons, info = strategies.select_c(items, cov, el, Decimal("10000"), CFG, AS_OF,
                                                required=["A01"])
    assert chosen[0] == "S1" and "социальное" in reasons["S1"]  # дешевле BIG на объект A01
    assert info["social"]["satisfied"] is True


def test_compute_plan_uses_only_eligible_positions():
    cat, cov, elig = ctx()
    res = strategies.compute_plan(cat, cov, elig - {"Z-A01"}, LOSS, Decimal("2452.98"), CFG,
                                  AS_OF, other_cfg=OTHER)
    assert "Z-A01" not in res["plans"]["C"]
    assert res["plans"]["C"] == ["Z-A01-S3"]  # та же геометрия, другое предложение


def test_b_empty_reason_when_wide_not_eligible():
    cat, cov, elig = ctx()
    res = strategies.compute_plan(cat, cov, elig - {"W-CHIP"}, LOSS, Decimal(5000), CFG, AS_OF,
                                  other_cfg=OTHER)
    assert res["plans"]["B"] == [] and "B пуста" in res["reasons"]["B_empty"]


def test_uncertainty_tie_break_prefers_larger_uncertainty():
    """Равные EL на рубль и цена: берётся позиция с большей неопределённостью новых объектов."""
    items = {"P": _item("P", "1.440"), "Q": _item("Q", "1.440")}
    cov = {"P": ["A01"], "Q": ["A02"]}
    el = {"A01": {"el": 5.0, "unc": 0.1, "risk": 5.0}, "A02": {"el": 5.0, "unc": 0.9, "risk": 5.0}}
    chosen, _, _ = strategies.select_c(items, cov, el, Decimal("2452.98"), CFG, AS_OF)
    assert chosen == ["Q"]


def test_social_phase_cost_per_required_not_el_ratio():
    """BIG лучше по EL на рубль, но S1 дешевле на обязательный объект A01 — социальная фаза
    берёт S1; без ограничения берётся BIG."""
    items = {"BIG": _item("BIG", "3.000"), "S1": _item("S1", "1.440")}
    cov = {"BIG": ["A01", "A02"], "S1": ["A01"]}
    el = {"A01": {"el": 1.0, "unc": 0, "risk": 1.0}, "A02": {"el": 100.0, "unc": 0, "risk": 100.0}}
    chosen, _, _ = strategies.select_c(items, cov, el, Decimal("5200"), CFG, AS_OF,
                                       required=["A01"])
    assert chosen == ["S1"]
    chosen, _, _ = strategies.select_c(items, cov, el, Decimal("5200"), CFG, AS_OF)
    assert chosen == ["BIG"]


def test_sensitivity_social_off_row_when_enabled():
    cat, cov, elig = ctx()
    rows = strategies.sensitivity(cat, cov, elig, LOSS, Decimal(5000), CFG, AS_OF, [], [],
                                  other_cfg=OTHER, required=["A10"], social_ids=["A10"])
    ids = [r["scenario_id"] for r in rows]
    assert "social_constraint_off" in ids and "social_constraint_on" not in ids


def test_sensitivity_interpretation_lists_basket_change():
    rows = sens_rows(budgets=[Decimal("0"), Decimal("12000")], rates=[])
    zero, big = _of(rows, "budget_0.00", "C")[0], _of(rows, "budget_12000.00", "C")[0]
    assert "C [пусто], относительно базы +[] −[Z-A01, Z-A02]" in zero["interpretation"]
    assert "относительно базы +[Z-A03" in big["interpretation"]
    assert "−[]" in big["interpretation"]


def test_coverage_at_only_assessed_assets():
    cat = ctx()[0]
    cov, elig = strategies.coverage_at(cat, {"A02"}, DEADLINE)
    assert cov["Z-A01"] == [] and cov["Z-A02"] == ["A02"]
    assert "Z-A01" in elig and "W-CHIP-OC" not in elig


def test_deadline_rows_name_scenario_timing_and_why():
    """Скептик D-047: сроки доступности — сценарные; пустая B к сроку объясняет, когда
    доступна W-CHIP; более поздний срок называет доступную, но не дающую охвата оптику."""
    rows = sens_rows(budgets=[], rates=[], decision_at=T0, deadline_hours=[6, 48])
    rows = [r for r in rows if r["scenario_id"].startswith("deadline_")]
    assert len(rows) == 4
    for r in rows:
        assert "сроки доступности предложений сценарные" in r["interpretation"]
    b6 = _of(rows, "deadline_+6h", "B")[0]["interpretation"]
    assert "правило B (W-CHIP) к сроку не исполнимо — W-CHIP доступна 2017-05-30T21:00:00" in b6
    c48 = _of(rows, "deadline_+48h", "C")[0]["interpretation"]
    assert "ещё 12 доступны к сроку, но охвата не дают (облачность не подтверждена)" in c48
    assert "не дают" not in _of(rows, "deadline_+6h", "C")[0]["interpretation"]


def test_empty_plan_uncertainty_is_initial_not_scenario():
    """Скептик D-047: у пустого плана остаточная неопределённость = исходная (как A)."""
    rows = sens_rows(budgets=[Decimal("0"), Decimal("5000")], rates=[])
    empty = _of(rows, "budget_0.00", "C")[0]["interpretation"]
    assert "закупки нет — неопределённость исходная (как A, baseline)" in empty
    assert "scenario" not in empty.split("бюджета 0.00")[1].split("(без новых")[0]
    full = _of(rows, "budget_5000.00", "C")[0]["interpretation"]
    assert "остаточная неопределённость scenario" in full


def test_with_base_rate_rounds_half_up():
    """reviewer #109: сценарная ставка округляется ROUND_HALF_UP, не банковски (1,25 × 0,5 = 0,625)."""
    cfg = {"base_rate": {"rub_per_km2": "1.25"}}
    assert strategies.with_base_rate(cfg, Decimal("0.5"))["base_rate"]["rub_per_km2"] == "0.63"
    assert cfg["base_rate"]["rub_per_km2"] == "1.25"
