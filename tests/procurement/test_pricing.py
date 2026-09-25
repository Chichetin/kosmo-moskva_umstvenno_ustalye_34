"""Цена ПП 840 по configs/pricing.yaml — контрольная арифметика task.md (критерий 17)."""
from datetime import date
from decimal import Decimal

import pytest

from floodvalue.procurement import pricing

CFG = pricing.load_pricing()
AS_OF = date(2026, 9, 20)


def item(cid="z1", **kw):
    base = {"candidate_id": cid, "sensor_type": "optical", "resolution_m": 1,
            "acquisition_type": "new", "guaranteed_purchase": True, "observation_at": None,
            "processing_level": "L2", "usage_type": "internal", "area_km2": Decimal("1.000")}
    base.update(kw)
    return base


FRESHNESS_ITEM = {
    "new_guaranteed": {"acquisition_type": "new", "guaranteed_purchase": True},
    "operational": {"acquisition_type": "operational", "guaranteed_purchase": False,
                    "observation_at": date(2026, 9, 10)},
    "archive": {"acquisition_type": "archive", "guaranteed_purchase": False,
                "observation_at": date(2025, 1, 1)},
}


@pytest.mark.parametrize("ex", CFG["control_examples"], ids=lambda e: e["freshness"])
def test_control_examples_from_config(ex):
    rows = pricing.price_basket(
        [item(sensor_type=ex["sensor"], resolution_m=ex["resolution_m"],
              processing_level=ex["processing_level"], usage_type=ex["usage_type"],
              area_km2=ex["area_km2"], **FRESHNESS_ITEM[ex["freshness"]])],
        CFG, as_of=AS_OF)
    assert rows[0]["cost_rub"] == Decimal(str(ex["cost_rub"]))


def test_control_values_literal():
    """1 703,46 / 946,37 / 567,82 — числа постановки, не только из конфига."""
    got = [pricing.price_basket([item(**FRESHNESS_ITEM[k])], CFG, as_of=AS_OF)[0]["cost_rub"]
           for k in ("new_guaranteed", "operational", "archive")]
    assert got == [Decimal("1703.46"), Decimal("946.37"), Decimal("567.82")]


def test_row_carries_intermediate_multipliers():
    row = pricing.price_basket([item()], CFG, as_of=AS_OF)[0]
    assert row["processing_coef"] == Decimal("1.2")
    assert row["usage_coef"] == Decimal(1)
    assert row["freshness_coef"] == Decimal("1.8")
    assert row["discount_coef"] == Decimal("1.000000000")
    assert row["unit_price_rub_km2"] == Decimal("1703.46")
    assert row["base_rate_rub_km2"] == Decimal("788.64")
    assert row["base_rate_status"] == "scenario"
    assert row["formula"] == "РП = Б × К × О × П × Т × Р"
    assert "1297" in row["legal_edition"]
    assert row["discount_group_id"] == "optical_1m_2026"
    assert row["group_area_km2"] == Decimal("1.000")
    for key in ("candidate_id", "area_km2", "processing_level", "usage_type",
                "guaranteed_purchase", "sensor_type", "resolution_m", "acquisition_type"):
        assert key in row


@pytest.mark.parametrize("level,coef", [("L0", "1"), ("L1", "1"), ("L2", "1.2")])
def test_processing_coef(level, coef):
    assert pricing.processing_coef(level, CFG) == Decimal(coef)


@pytest.mark.parametrize("usage,coef", [("internal", "1"), ("limited", "1.2"),
                                        ("unrestricted", "1.5")])
def test_usage_coef(usage, coef):
    assert pricing.usage_coef(usage, CFG) == Decimal(coef)


def test_unknown_category_rejected():
    with pytest.raises(ValueError):
        pricing.processing_coef("L3", CFG)
    with pytest.raises(ValueError):
        pricing.usage_coef("public", CFG)


@pytest.mark.parametrize("days,coef,cat", [(0, "1", "operational"), (89, "1", "operational"),
                                           (90, "1", "operational"), (91, "0.6", "archive")])
def test_freshness_boundary_90_days_inclusive(days, coef, cat):
    from datetime import timedelta
    observed = AS_OF - timedelta(days=days)
    assert pricing.existing_data_category(observed, AS_OF, CFG) == cat
    assert pricing.freshness_coef(cat, False, observed, AS_OF, CFG) == Decimal(coef)


def test_freshness_age_uses_calendar_date_of_datetimes():
    from datetime import UTC, datetime
    # 23:59 минус 90 дней и 00:01 — всё ещё 90 календарных дней
    assert pricing.existing_data_category(datetime(2026, 6, 22, 0, 1, tzinfo=UTC),
                                          datetime(2026, 9, 20, 23, 59, tzinfo=UTC),
                                          CFG) == "operational"
    assert pricing.existing_data_category("2026-06-22 10:00", "2026-09-20", CFG) == "operational"
    assert pricing.existing_data_category("2026-06-21T23:59:00Z", "2026-09-20", CFG) == "archive"


def test_freshness_declared_category_must_match_age():
    with pytest.raises(ValueError, match="archive"):
        pricing.freshness_coef("operational", False, date(2026, 6, 1), AS_OF, CFG)
    with pytest.raises(ValueError):
        pricing.freshness_coef("archive", False, date(2026, 9, 1), AS_OF, CFG)


def test_new_without_guarantee_is_error():
    with pytest.raises(ValueError, match="guaranteed_purchase"):
        pricing.freshness_coef("new", False, None, AS_OF, CFG)
    with pytest.raises(ValueError):
        pricing.price_basket([item(guaranteed_purchase=False)], CFG, as_of=AS_OF)


def test_guarantee_only_for_new():
    with pytest.raises(ValueError, match="guaranteed_purchase"):
        pricing.freshness_coef("operational", True, date(2026, 9, 10), AS_OF, CFG)


def test_existing_data_needs_observation_date_not_in_future():
    with pytest.raises(ValueError):
        pricing.freshness_coef("operational", False, None, AS_OF, CFG)
    with pytest.raises(ValueError):
        pricing.freshness_coef("operational", False, date(2026, 9, 21), AS_OF, CFG)


def test_discount_optical_uses_km2_not_m2():
    # S = 10 км², r = 1 м: 1,037 − 0,058·ln 10 = 0,903450065 (в м² было бы 0,2 — пол)
    assert pricing.discount_coef("optical", Decimal(10), 1, CFG) == Decimal("0.903450065")


def test_discount_sar_and_resolution():
    # SAR, S = 1000 км², r = 10 м: ln(1000/100) = ln 10 → 1,031 − 0,02·2,302585093 = 0,984948298
    assert pricing.discount_coef("sar", Decimal(1000), 10, CFG) == Decimal("0.984948298")


@pytest.mark.parametrize("sensor,s,expected", [
    ("optical", Decimal(1), Decimal(1)),              # min(1; 1,037)
    ("optical", Decimal("1e9"), Decimal("0.2")),          # пол оптики
    ("sar", Decimal("1e12"), Decimal("0.7")),             # пол SAR
])
def test_discount_clamps(sensor, s, expected):
    assert pricing.discount_coef(sensor, s, 1, CFG) == expected


def test_discount_rejects_nonpositive_and_unknown_sensor():
    with pytest.raises(ValueError):
        pricing.discount_coef("optical", Decimal(0), 1, CFG)
    with pytest.raises(ValueError):
        pricing.discount_coef("optical", Decimal(1), 0, CFG)
    with pytest.raises(ValueError):
        pricing.discount_coef("lidar", Decimal(1), 1, CFG)


def test_discount_recomputed_over_whole_group_when_zone_added():
    one = pricing.price_basket([item("a", area_km2="1.5")], CFG, as_of=AS_OF)
    two = pricing.price_basket([item("a", area_km2="1.5"), item("b", area_km2="1.5")],
                               CFG, as_of=AS_OF)
    assert one[0]["discount_coef"] == Decimal(1)
    # S = 3 км²: 1,037 − 0,058·ln 3 = 0,973280487 — у ОБЕИХ строк группы
    assert [r["discount_coef"] for r in two] == [Decimal("0.973280487")] * 2
    assert [r["group_area_km2"] for r in two] == [Decimal("3.000")] * 2
    assert two[0]["cost_rub"] < one[0]["cost_rub"]
    # 788,64·1,2·1,8·0,973280487 = 1657,946... → 1657,95; ×1,5 = 2486,925 → 2486,93
    assert two[0]["unit_price_rub_km2"] == Decimal("1657.95")
    assert two[0]["cost_rub"] == Decimal("2486.93")


def test_groups_split_by_sensor_resolution_and_year():
    rows = pricing.price_basket([item("a", area_km2="2"), item("b", area_km2="2",
                                                                sensor_type="sar"),
                                 item("c", area_km2="2", resolution_m=3)], CFG, as_of=AS_OF)
    ids = {r["candidate_id"]: r["discount_group_id"] for r in rows}
    assert len(set(ids.values())) == 3
    assert all(r["group_area_km2"] == Decimal("2.000") for r in rows)


def test_group_key_is_sensor_family():
    cfg = {**CFG, "discount": {**CFG["discount"],
                               "sensor_family": {"optical": ["optical", "optical_b"],
                                                 "sar": ["sar"]}}}
    rows = pricing.price_basket([item("a", area_km2="1.5"),
                                 item("b", area_km2="1.5", sensor_type="optical_b")],
                                cfg, as_of=AS_OF)
    assert {r["discount_group_id"] for r in rows} == {"optical_1m_2026"}
    assert all(r["group_area_km2"] == Decimal("3.000") for r in rows)


def test_previous_area_enters_group_area():
    rows = pricing.price_basket([item(area_km2="1")], CFG, as_of=AS_OF,
                                previous_area_km2={"optical_1m_2026": Decimal(2)})
    assert rows[0]["group_area_km2"] == Decimal("3.000")
    assert rows[0]["discount_coef"] == Decimal("0.973280487")


def test_rounding_half_up_on_cost_and_total():
    # SAR 1 м, S = 2,5 км² → Р = 1; 946,37 × 2,5 = 2365,925 → HALF_UP 2365,93 (HALF_EVEN дал бы ,92)
    rows = pricing.price_basket([item(area_km2="2.5", sensor_type="sar",
                                      **FRESHNESS_ITEM["operational"])], CFG, as_of=AS_OF)
    assert rows[0]["cost_rub"] == Decimal("2365.93")
    assert pricing.round_money(Decimal("0.125")) == Decimal("0.13")
    assert pricing.round_money(Decimal("2.345")) == Decimal("2.35")


def test_basket_total_is_sum_of_rounded_rows():
    rows = pricing.price_basket([item("a", area_km2="1.333"), item("b", area_km2="1.777")],
                                CFG, as_of=AS_OF)
    assert pricing.basket_total(rows) == sum(r["cost_rub"] for r in rows)
    assert all(r["cost_rub"] == r["cost_rub"].quantize(Decimal("0.01")) for r in rows)


def test_duplicate_candidate_rejected():
    with pytest.raises(ValueError):
        pricing.price_basket([item("a"), item("a")], CFG, as_of=AS_OF)


def test_area_below_minimum_rejected_in_basket():
    with pytest.raises(ValueError):
        pricing.price_basket([item(area_km2="0.999")], CFG, as_of=AS_OF)


def test_empty_basket_is_empty():
    assert pricing.price_basket([], CFG, as_of=AS_OF) == []
    assert pricing.basket_total([]) == Decimal("0.00")
