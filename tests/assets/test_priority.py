"""Inspection order (criterion 14, D-035): a separate rule next to `rank`, never replacing it."""

import pytest

from floodvalue.assets.portfolio import binary_entropy, rank_assets
from floodvalue.assets.priority import (
    P_DRY,
    SOCIAL_CLASSES,
    SOCIAL_MAX_POSITION,
    UNCERTAINTY_HIGH,
    inspection_order,
    order_moves,
)


def row(aid, p, el, u="entropy", status="ok"):
    if status != "ok":
        return {"asset_id": aid, "p_flood": None, "expected_loss_rub": None, "uncertainty": None,
                "rank": None, "status": status}
    return {"asset_id": aid, "p_flood": p, "expected_loss_rub": el,
            "uncertainty": binary_entropy(p) if u == "entropy" else u, "rank": None,
            "status": status}


def with_ranks(rows):
    ranks = rank_assets({r["asset_id"]: r["expected_loss_rub"] for r in rows})
    return [{**r, "rank": ranks[r["asset_id"]]} for r in rows]


def ids(order):
    return [o["asset_id"] for o in order]


CLASSES = {f"A{i:02d}": "warehouse" for i in range(1, 11)}


def test_parameters_follow_the_declared_rule():
    assert P_DRY == 0.1
    assert UNCERTAINTY_HIGH == pytest.approx(binary_entropy(P_DRY))  # 0.469 bits: p in [0.1, 0.9]
    assert UNCERTAINTY_HIGH == pytest.approx(0.469, abs=1e-3)
    assert set(SOCIAL_CLASSES) == {"clinic", "school"}
    assert SOCIAL_MAX_POSITION == 3


def test_groups_at_risk_then_not_assessed_then_likely_dry():
    rows = with_ranks([
        row("A01", 0.02, 50_000.0),         # likely dry, but the largest EL of the dry ones
        row("A02", 0.9, 1_800_000.0),       # at risk, EL rank 1
        row("A03", None, None, status="no_data"),
        row("A04", 0.3, 300_000.0),         # at risk
        row("A05", 0.01, 10_000.0),         # likely dry
        row("A06", None, None, status="partial"),
    ])
    order = inspection_order(rows, CLASSES)
    assert ids(order) == ["A02", "A04", "A03", "A06", "A01", "A05"]
    assert [o["inspection_order"] for o in order] == [1, 2, 3, 4, 5, 6]
    assert [o["group"] for o in order] == ["at_risk", "at_risk", "not_assessed", "not_assessed",
                                           "likely_dry", "likely_dry"]
    by_id = {o["asset_id"]: o for o in order}
    assert by_id["A02"]["el_rank"] == 1 and by_id["A03"]["el_rank"] is None
    for o in order:
        assert isinstance(o["reason"], str) and o["reason"]
    assert "нет оценки" in by_id["A03"]["reason"]
    assert "вероятно сухо" in by_id["A01"]["reason"]


def test_not_assessed_is_ordered_by_asset_id_not_by_rubles():
    rows = [row("A07", None, None, status="no_data"), row("A02", None, None, status="no_data"),
            row("A05", None, None, status="partial")]
    order = inspection_order(rows, CLASSES)
    assert ids(order) == ["A02", "A05", "A07"]  # zero ok assets: nothing breaks
    assert {o["group"] for o in order} == {"not_assessed"}


def test_p_dry_boundary_is_inclusive_for_at_risk():
    rows = with_ranks([row("A01", 0.1, 100.0, u=None), row("A02", 0.0999, 200.0, u=None)])
    by_id = {o["asset_id"]: o for o in inspection_order(rows, CLASSES)}
    assert by_id["A01"]["group"] == "at_risk"
    assert by_id["A02"]["group"] == "likely_dry"


def test_high_uncertainty_lifts_a_low_p_asset_out_of_likely_dry():
    # A future uncertainty measure (#21) may be high while p is low: the model is not sure.
    rows = with_ranks([row("A01", 0.05, 100.0, u=0.9), row("A02", 0.05, 200.0, u=0.1)])
    by_id = {o["asset_id"]: o for o in inspection_order(rows, CLASSES, uncertainty_high=0.5)}
    assert by_id["A01"]["group"] == "at_risk" and by_id["A01"]["inspection_order"] == 1
    assert "не уверена" in by_id["A01"]["reason"]
    assert by_id["A02"]["group"] == "likely_dry"


def test_uncertainty_threshold_is_inclusive():
    rows = with_ranks([row("A01", 0.05, 100.0, u=0.5)])
    assert inspection_order(rows, CLASSES, uncertainty_high=0.5)[0]["group"] == "at_risk"


def test_empty_uncertainty_groups_by_p_and_says_so():
    rows = with_ranks([row("A01", 0.05, 100.0, u=None), row("A02", 0.3, 50.0, u=None)])
    by_id = {o["asset_id"]: o for o in inspection_order(rows, CLASSES)}
    assert by_id["A01"]["group"] == "likely_dry"
    assert by_id["A02"]["group"] == "at_risk"
    assert "неопределённость не оценена" in by_id["A01"]["reason"]


def test_no_uncertainty_threshold_uses_p_only():
    rows = with_ranks([row("A01", 0.05, 100.0, u=0.99)])
    assert inspection_order(rows, CLASSES, uncertainty_high=None)[0]["group"] == "likely_dry"


def test_equal_el_higher_uncertainty_first_then_asset_id():
    rows = with_ranks([row("A03", 0.5, 1000.0, u=0.2), row("A01", 0.5, 1000.0, u=0.2),
                       row("A02", 0.5, 1000.0, u=0.9), row("A04", 0.5, 1000.0, u=None)])
    assert ids(inspection_order(rows, CLASSES)) == ["A02", "A01", "A03", "A04"]


def test_el_orders_within_a_group_even_against_higher_uncertainty():
    # Rejected alternative (z): a confident high-EL asset is not pushed back by an uncertain one.
    rows = with_ranks([row("A01", 0.99, 5_000_000.0), row("A02", 0.5, 100_000.0)])
    assert ids(inspection_order(rows, CLASSES)) == ["A01", "A02"]


SOCIAL = {**CLASSES, "A06": "clinic", "A07": "school"}


def test_social_at_risk_asset_is_lifted_to_position_k_without_touching_rubles():
    rows = with_ranks([
        row("A01", 0.9, 5_000_000.0), row("A02", 0.9, 4_000_000.0), row("A03", 0.9, 3_000_000.0),
        row("A04", 0.9, 2_000_000.0), row("A06", 0.3, 1_000_000.0),  # clinic, EL position 5
    ])
    before = {r["asset_id"]: dict(r) for r in rows}
    order = inspection_order(rows, SOCIAL)
    assert ids(order) == ["A01", "A02", "A06", "A03", "A04"]
    clinic = order[2]
    assert clinic["inspection_order"] == SOCIAL_MAX_POSITION and clinic["el_rank"] == 5
    assert "социальное ограничение" in clinic["reason"] and "с места 5" in clinic["reason"]
    assert {r["asset_id"]: r for r in rows} == before  # EL, p, rank untouched


def test_social_constraint_never_pushes_a_social_asset_already_in_top_k_out():
    rows = with_ranks([
        row("A01", 0.9, 5_000_000.0), row("A02", 0.9, 4_000_000.0),
        row("A06", 0.9, 3_000_000.0),                                  # clinic already 3rd
        row("A03", 0.9, 2_000_000.0), row("A07", 0.9, 1_000_000.0),  # school 5th
    ])
    order = ids(inspection_order(rows, SOCIAL))
    assert order.index("A06") < SOCIAL_MAX_POSITION and order.index("A07") < SOCIAL_MAX_POSITION
    assert order == ["A01", "A06", "A07", "A02", "A03"]


def test_social_asset_already_ahead_is_never_delayed_to_position_k():
    rows = with_ranks([row("A06", 0.9, 9_000_000.0), row("A01", 0.9, 5_000_000.0),
                       row("A02", 0.9, 4_000_000.0), row("A03", 0.9, 3_000_000.0)])
    order = inspection_order(rows, SOCIAL)
    assert ids(order) == ["A06", "A01", "A02", "A03"]
    assert "социальное ограничение" not in order[0]["reason"]


def test_social_not_assessed_is_lifted_but_social_likely_dry_is_not():
    rows = with_ranks([
        row("A01", 0.9, 5_000_000.0), row("A02", 0.9, 4_000_000.0), row("A03", 0.9, 3_000_000.0),
        row("A06", None, None, status="no_data"),   # clinic without estimate
        row("A07", 0.01, 10_000.0),                  # school, confidently dry
        row("A04", 0.02, 20_000.0),
    ])
    order = inspection_order(rows, SOCIAL)
    by_id = {o["asset_id"]: o for o in order}
    assert by_id["A06"]["inspection_order"] == SOCIAL_MAX_POSITION
    assert by_id["A06"]["group"] == "not_assessed"
    assert ids(order)[-2:] == ["A04", "A07"]  # dry school stays in EL order at the tail


def test_social_assets_beyond_k_take_the_first_places():
    rows = with_ranks([row("A01", 0.9, 5_000_000.0), row("A06", None, None, status="no_data"),
                       row("A07", None, None, status="no_data")])
    assert ids(inspection_order(rows, SOCIAL, social_max_position=1)) == ["A06", "A07", "A01"]


def test_all_ten_not_assessed_keeps_everyone_and_social_within_k():
    classes = {**SOCIAL}
    rows = [row(f"A{i:02d}", None, None, status="no_data") for i in range(10, 0, -1)]
    order = inspection_order(rows, classes)
    assert ids(order) == ["A01", "A06", "A07", "A02", "A03", "A04", "A05", "A08", "A09", "A10"]
    assert {o["group"] for o in order} == {"not_assessed"}
    assert all(o["el_rank"] is None for o in order)


def test_action_depends_on_uncertainty_not_only_on_group():
    rows = with_ranks([
        row("A01", 0.95, 900.0),                 # sure water -> go now
        row("A02", 0.5, 500.0),                  # model not sure -> confirm (inspection/image)
        row("A03", 0.3, 300.0, u=None),          # at risk, uncertainty unknown
        row("A04", None, None, status="no_data"),
        row("A05", 0.01, 1.0),                   # likely dry
    ])
    act = {o["asset_id"]: o["action"] for o in inspection_order(rows, CLASSES)}
    assert "почти наверняка" in act["A01"]
    assert "не уверена" in act["A02"] and "снимк" in act["A02"]
    assert act["A03"] != act["A01"] and act["A03"] != act["A02"]
    assert "нет оценки" in act["A04"]
    assert "последним" in act["A05"]


def test_reason_explains_an_asset_dry_on_the_mask_but_at_risk():
    rows = with_ranks([row("A01", 0.25, 100.0), row("A02", 0.5, 50.0)])
    by_id = {o["asset_id"]: o for o in inspection_order(rows, CLASSES, mask_threshold=0.37)}
    assert "ниже порога карты 0,37" in by_id["A01"]["reason"]
    assert "ниже порога карты" not in by_id["A02"]["reason"]


def test_order_moves_lists_changed_positions_and_groups():
    def o(aid, pos, group="at_risk"):
        return {"asset_id": aid, "inspection_order": pos, "group": group}

    before = [o("A01", 1), o("A02", 2), o("A03", 3, "likely_dry")]
    after = [o("A02", 1), o("A01", 2), o("A03", 3)]
    assert order_moves(before, after) == ["A02 2→1", "A01 1→2", "A03 3→3 (likely_dry→at_risk)"]
    assert order_moves(before, before) == []
