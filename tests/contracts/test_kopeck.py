"""Kopeck arithmetic of asset_loss.csv shared by procurement, assets and validate (#232)."""

import pytest

from floodvalue.contracts.kopeck import decision_risk_from_el, kopeck_sum, ok_rows


def test_kopeck_sum_is_exact_without_float_tail():
    assert 0.1 + 0.2 != 0.3
    assert kopeck_sum([0.1, 0.2]) == 0.3
    assert kopeck_sum([1703.46, 946.37, 567.82]) == 3217.65


def test_kopeck_sum_reads_csv_strings_and_skips_none():
    assert kopeck_sum(["12.50", "0.05", None, 0.45]) == 13.0
    assert kopeck_sum([]) == 0.0


def test_decision_risk_from_el_is_el_below_half_and_scaled_above():
    # R = V·q·min(p, 1−p); from the file V·q = EL/p (D-049)
    assert decision_risk_from_el(1_000_000, 0.5) == 1_000_000
    assert decision_risk_from_el(400.0, 0.2) == 400.0
    assert decision_risk_from_el(800_000, 0.8) == pytest.approx(200_000)
    assert decision_risk_from_el(1_000_000, 1.0) == 0.0
    assert decision_risk_from_el(0.0, 0.0) == 0.0
    assert decision_risk_from_el("800000", "0.8") == pytest.approx(200_000)


def test_ok_rows_keeps_only_assessed_assets():
    rows = [{"asset_id": "a", "status": "ok"}, {"asset_id": "b", "status": "partial"},
            {"asset_id": "c", "status": "no_data"}, {"asset_id": "d", "status": "ok"}]
    assert [r["asset_id"] for r in ok_rows(rows)] == ["a", "d"]
