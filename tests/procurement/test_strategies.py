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
LOSS = [{"asset_id": a["asset_id"], "status": "ok",
         "expected_loss_rub": str(1_000_000 * (11 - i)), "uncertainty": "0.5"}
        for i, a in enumerate(ASSETS, 1)]


def ctx(loss=LOSS):
    cat = zones.build_catalog(ASSETS, (79.99, 6.99, 80.21, 7.01), T0, CFG)
    ok = {r["asset_id"] for r in loss if r["status"] == "ok"}
    cov = zones.coverage(cat, [a for a in ASSETS if a["asset_id"] in ok], DEADLINE)
    elig = {f["properties"]["candidate_id"] for f in cat["features"]
            if zones.exclusion_reason(f["properties"], DEADLINE) is None}
    return cat, cov, elig


def plan(budget, loss=LOSS, cfg=CFG):
    cat, cov, elig = ctx(loss)
    return strategies.compute_plan(cat, cov, elig, loss, Decimal(budget), cfg, AS_OF)


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


def test_c_orders_by_el_per_rub():
    assert plan("5000")["plans"]["C"] == ["Z-A01", "Z-A02"]


def test_b_counterfactual_when_over_budget():
    res = plan("5000")
    b = cmp_(res, "B")
    assert res["plans"]["B"] == ["W-CHIP"]
    assert b["data_cost_rub"] > Decimal(5000) and b["budget_feasible"] is False
    assert cmp_(plan("1000000"), "B")["budget_feasible"] is True


def test_a_empty_baseline_uncertainty_bc_not_estimated():
    res = plan("5000")
    a = cmp_(res, "A")
    assert res["plans"]["A"] == [] and a["data_cost_rub"] == 0
    assert a["uncertainty_status"] == "baseline" and a["residual_uncertainty"] == pytest.approx(5.0)
    assert a["covered_expected_loss_rub"] == 0
    for s in "BC":
        assert cmp_(res, s)["uncertainty_status"] == "not_estimated"
        assert cmp_(res, s)["residual_uncertainty"] is None
    assert not [r for r in res["plan_rows"] if r["strategy"] == "A"]


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


def test_sensitivity_rows():
    cat, cov, elig = ctx()
    rows = strategies.sensitivity(cat, cov, elig, LOSS, Decimal(5000), CFG, AS_OF,
                                  [Decimal(b) for b in ("0", "2500", "5000", "10000", "50000")],
                                  [Decimal("0.5"), Decimal("1.5")])
    import json
    budgets = [r for r in rows if r["scenario_id"].startswith("budget_")]
    rates = [r for r in rows if r["scenario_id"].startswith("rate_")]
    assert len(budgets) == 5 and len(rates) == 4
    assert all(json.loads(r["changed_inputs_json"]) for r in rows)
    costs = [r["covered_expected_loss_rub"] for r in budgets]
    assert costs == sorted(costs)
    assert budgets[0]["result_status"] == "empty_plan"
    b_rates = [r for r in rates if r["strategy"] == "B"]
    assert all(r["result_status"] == "counterfactual_over_budget" for r in b_rates)
    cheap = next(r for r in rates if r["scenario_id"] == "rate_x0.5" and r["strategy"] == "C")
    base = next(r for r in budgets if r["scenario_id"] == "budget_5000.00")
    assert cheap["covered_expected_loss_rub"] > base["covered_expected_loss_rub"]


def test_selection_uses_repriced_basket_not_sum_of_singles():
    """8 зон: сумма одиночных цен 19 623,84 > 19 300, а корзина со скидкой по группе
    (11,52 км², Р < 1) — 19 272,96 ≤ 19 300: C обязана взять все 8."""
    res = plan("19300")
    assert len(res["plans"]["C"]) == 8
    assert cmp_(res, "C")["data_cost_rub"] <= Decimal(19300)


def test_overlapping_zones_do_not_double_count():
    cov = {"x": ["A01", "A02"], "y": ["A02"]}
    el = {"A01": {"el": 10.0, "unc": 0.1}, "A02": {"el": 5.0, "unc": 0.2}}
    row = strategies.comparison_row("B", ["x", "y"], [], Decimal(0), cov, el)
    assert row["covered_expected_loss_rub"] == 15.0
    assert row["covered_asset_ids"] == "A01 A02"
