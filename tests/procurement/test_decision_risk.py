"""C по цене ошибки решения R = V·q·min(p, 1−p) (D-048, #114; критерии 10, 15, 16)."""
import json
from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

from floodvalue.procurement import strategies
from floodvalue.procurement.pricing import load_pricing

CFG = load_pricing()
AS_OF = date(2017, 5, 30)
OTHER = {"compute_rub": 1000, "review_per_zone_rub": 500}
ONE_ZONE = Decimal("2452.98")  # 1,44 км² × 1703,46 — ровно одна зона Z


def _item(cid, area="1.440"):
    return {"candidate_id": cid, "sensor_type": "sar", "resolution_m": 1,
            "acquisition_type": "new", "guaranteed_purchase": True, "observation_at": None,
            "processing_level": "L2", "usage_type": "internal", "area_km2": Decimal(area),
            "available_at": "2017-05-30T21:00:00+00:00"}


def _loss(aid, p, vq, status="ok", **extra):
    if status != "ok":
        return {"asset_id": aid, "status": status, "p_flood": "", "expected_loss_rub": "",
                "uncertainty": ""}
    return {"asset_id": aid, "status": "ok", "p_flood": str(p),
            "expected_loss_rub": f"{p * vq:.2f}", "uncertainty": "0.5", **extra}


# --- формула -----------------------------------------------------------------------------------

@pytest.mark.parametrize("p, share", [(0.5, 0.5), (0.98, 0.02), (0.1, 0.1), (0.0, 0.0),
                                      (1.0, 0.0), (0.02, 0.02)])
def test_decision_risk_formula(p, share):
    assert strategies.decision_risk_rub(p, 1_000_000, 0.8) == pytest.approx(800_000 * share)


def test_decision_risk_symmetric():
    for p in (0.01, 0.2, 0.37, 0.49):
        assert strategies.decision_risk_rub(p, 5e6, 0.5) == pytest.approx(
            strategies.decision_risk_rub(1 - p, 5e6, 0.5))


@pytest.mark.parametrize("p", [0.0, 0.1, 0.37, 0.45, 0.5, 0.55, 0.63, 0.98, 0.997, 1.0])
def test_risk_from_el_equals_formula(p):
    """R из EL и p (asset_loss.csv) = V·q·min(p, 1−p): V·q = EL / p."""
    v, q = 3_000_000, 0.7
    assert strategies.decision_risk_from_el(p * v * q, p) == pytest.approx(
        strategies.decision_risk_rub(p, v, q))


def test_assessed_only_ok_and_risk():
    loss = [_loss("A01", 0.98, 10e6), _loss("A02", 0.5, 2e6),
            _loss("A03", None, None, status="no_data"), _loss("A04", None, None, status="partial")]
    el = strategies.assessed(loss)
    assert set(el) == {"A01", "A02"}
    assert el["A01"]["risk"] == pytest.approx(0.02 * 10e6)
    assert el["A02"]["risk"] == pytest.approx(1e6)


# --- регрессия случая jury ---------------------------------------------------------------------

JURY_ITEMS = {"HI": _item("HI"), "MID": _item("MID")}
JURY_COV = {"HI": ["A01", "A02"], "MID": ["A03"]}
JURY_LOSS = [_loss("A01", 0.98, 10e6), _loss("A02", 0.997, 5e6), _loss("A03", 0.5, 2e6)]


def test_jury_case_c_takes_uncertain_asset_not_known_high_el():
    """p 0,98 / 0,997 (исход почти известен) против p ≈ 0,5; бюджет на одну зону:
    C покупает снимок там, где он может изменить решение; прежнее правило — высокий EL."""
    el = strategies.assessed(JURY_LOSS)
    chosen, reasons, _ = strategies.select_c(JURY_ITEMS, JURY_COV, el, ONE_ZONE, CFG, AS_OF)
    assert chosen == ["MID"]
    assert "цена ошибки решения" in reasons["MID"]
    by_el, _, _ = strategies.select_c(JURY_ITEMS, JURY_COV, el, ONE_ZONE, CFG, AS_OF,
                                      rule_c="by_EL")
    assert by_el == ["HI"]


def test_comparison_has_decision_risk_columns():
    el = strategies.assessed(JURY_LOSS)
    row = strategies.comparison_row("C", ["MID"], [], ONE_ZONE, JURY_COV, el, OTHER)
    total = 0.02 * 10e6 + 0.003 * 5e6 + 1e6
    assert row["addressed_decision_risk_rub"] == pytest.approx(1e6)
    assert row["decision_risk_share"] == pytest.approx(1e6 / total)
    assert row["covered_expected_loss_rub"] == pytest.approx(1e6)  # EL-охват не меняется
    a = strategies.comparison_row("A", [], [], ONE_ZONE, JURY_COV, el, OTHER)
    assert a["addressed_decision_risk_rub"] == 0 and a["decision_risk_share"] == 0


def test_decision_risk_share_empty_when_no_risk():
    el = strategies.assessed([_loss("A01", 1.0, 1e6), _loss("A02", 0.0, 1e6)])
    row = strategies.comparison_row("C", [], [], ONE_ZONE, {"Z": ["A01"]}, el, OTHER)
    assert row["decision_risk_share"] is None
    assert row["addressed_decision_risk_rub"] == 0


# --- выбор прохода по ΣR -----------------------------------------------------------------------

def test_pass_choice_by_sum_risk_not_covered_el():
    """Жадный: S1 (R 1,0, EL 1,0) и S2 (R 0,9); BIG (EL 30, R 1,5) не помещается после них.
    Старт с BIG даёт больше EL (30 > 1,9), но меньше ΣR (1,5 < 1,9) — берётся жадный."""
    items = {"BIG": _item("BIG", "3.000"), "S1": _item("S1"), "S2": _item("S2")}
    cov = {"BIG": ["A03", "A04"], "S1": ["A01"], "S2": ["A02"]}
    el = {"A01": {"el": 1.0, "unc": 0, "risk": 1.0},
          "A02": {"el": 0.9, "unc": 0, "risk": 0.9},
          "A03": {"el": 20.0, "unc": 0, "risk": 1.0},
          "A04": {"el": 10.0, "unc": 0, "risk": 0.5}}
    chosen, _, info = strategies.select_c(items, cov, el, Decimal(5200), CFG, AS_OF)
    assert chosen == ["S1", "S2"] and info["variant"] == "greedy"
    alt = info["alternatives"]["greedy"]
    assert alt["addressed_decision_risk_rub"] == pytest.approx(1.9)
    by_el, _, _ = strategies.select_c(items, cov, el, Decimal(5200), CFG, AS_OF,
                                      rule_c="by_EL")
    assert by_el == ["BIG"]


def test_step_tie_on_risk_prefers_more_el():
    """Равный прирост R на рубль — больше прирост EL."""
    items = {"P": _item("P"), "Q": _item("Q")}
    cov = {"P": ["A01"], "Q": ["A02"]}
    el = {"A01": {"el": 1.0, "unc": 0, "risk": 1.0}, "A02": {"el": 5.0, "unc": 0, "risk": 1.0}}
    chosen, _, _ = strategies.select_c(items, cov, el, ONE_ZONE, CFG, AS_OF)
    assert chosen == ["Q"]


# --- ΔR = 0: не покупается ----------------------------------------------------------------------

def test_zero_risk_positions_not_bought():
    """Позиция, покрывающая только объекты с p ∈ {0, 1}, не покупается и при большом бюджете."""
    items = {"KNOWN": _item("KNOWN"), "MID": _item("MID")}
    cov = {"KNOWN": ["A01"], "MID": ["A02"]}
    el = strategies.assessed([_loss("A01", 1.0, 10e6), _loss("A02", 0.4, 1e6)])
    chosen, _, _ = strategies.select_c(items, cov, el, Decimal(100000), CFG, AS_OF)
    assert chosen == ["MID"]
    by_el, _, _ = strategies.select_c(items, cov, el, Decimal(100000), CFG, AS_OF, rule_c="by_EL")
    assert set(by_el) == {"KNOWN", "MID"}


# --- перекрытие, partial/no_data ---------------------------------------------------------------

def test_overlap_no_double_count_of_risk():
    cov = {"x": ["A01", "A02"], "y": ["A02"]}
    el = strategies.assessed([_loss("A01", 0.5, 2e6), _loss("A02", 0.9, 1e6)])
    row = strategies.comparison_row("C", ["x", "y"], [], ONE_ZONE, cov, el, OTHER)
    assert row["addressed_decision_risk_rub"] == pytest.approx(1e6 + 0.1 * 1e6)
    items = {"x": _item("x"), "y": _item("y")}
    chosen, _, _ = strategies.select_c(items, cov, el, Decimal(100000), CFG, AS_OF)
    assert chosen == ["x"]  # y ничего не добавляет после x


def test_partial_no_data_outside_risk():
    loss = [_loss("A01", 0.5, 2e6), _loss("A02", None, None, status="partial"),
            _loss("A03", None, None, status="no_data")]
    el = strategies.assessed(loss)
    cov = {"x": ["A01", "A02", "A03"], "y": ["A02", "A03"]}
    row = strategies.comparison_row("C", ["x"], [], ONE_ZONE, cov, el, OTHER)
    assert row["addressed_decision_risk_rub"] == pytest.approx(1e6)
    assert row["decision_risk_share"] == pytest.approx(1.0)
    chosen, _, _ = strategies.select_c({"x": _item("x"), "y": _item("y")}, cov, el,
                                       Decimal(100000), CFG, AS_OF)
    assert chosen == ["x"]


# --- тай-брейк ---------------------------------------------------------------------------------

def test_tie_break_mask_unstable_then_el_range():
    items = {"P": _item("P"), "Q": _item("Q")}
    cov = {"P": ["A01"], "Q": ["A02"]}
    base = {"el": 1.0, "unc": 0.5, "risk": 1.0}
    el = {"A01": {**base, "unstable": 0, "el_range": 5.0},
          "A02": {**base, "unstable": 1, "el_range": 0.0}}
    assert strategies.select_c(items, cov, el, ONE_ZONE, CFG, AS_OF)[0] == ["Q"]
    el = {"A01": {**base, "unstable": 0, "el_range": 5.0},
          "A02": {**base, "unstable": 0, "el_range": 1.0}}
    assert strategies.select_c(items, cov, el, ONE_ZONE, CFG, AS_OF)[0] == ["P"]


def test_assessed_reads_mask_unstable_and_el_range():
    el = strategies.assessed([_loss("A01", 0.4, 1e6, mask_unstable="true", el_low_rub="100",
                                    el_high_rub="350"),
                              _loss("A02", 0.4, 1e6, mask_unstable="", el_low_rub="",
                                    el_high_rub="")])
    assert el["A01"]["unstable"] == 1 and el["A01"]["el_range"] == pytest.approx(250)
    assert el["A02"]["unstable"] == 0 and el["A02"]["el_range"] == 0


# --- sensitivity: строка прежнего правила ------------------------------------------------------

def test_sensitivity_has_c_by_el_row():
    from floodvalue.contracts.schemas import SensitivityRow
    from floodvalue.procurement import zones
    t0 = datetime(2017, 5, 30, 12, tzinfo=UTC)
    deadline = datetime(2017, 5, 31, 0, tzinfo=UTC)
    assets = [{"asset_id": f"A{i:02d}", "longitude": 80.0 + 0.02 * i, "latitude": 7.0}
              for i in range(1, 4)]
    loss = [_loss("A01", 0.98, 10e6), _loss("A02", 0.997, 5e6), _loss("A03", 0.5, 2e6)]
    cat = zones.build_catalog(assets, (79.99, 6.99, 80.07, 7.01), t0, CFG)
    cov = zones.coverage(cat, assets, deadline)
    elig = {f["properties"]["candidate_id"] for f in cat["features"]
            if zones.exclusion_reason(f["properties"], deadline) is None}
    rows = strategies.sensitivity(cat, cov, elig, loss, Decimal(5000), CFG, AS_OF, [], [],
                                  other_cfg=OTHER)
    (r,) = [r for r in rows if r["scenario_id"] == "C_by_EL"]
    assert r["strategy"] == "C"
    assert json.loads(r["changed_inputs_json"]) == {"rule_C": "by_EL"}
    SensitivityRow.model_validate({k: ("" if v is None else str(v)) for k, v in r.items()})
    assert "цена ошибки решения" in r["interpretation"]
    by_el = strategies.compute_plan(cat, cov, elig, loss, Decimal(5000), CFG, AS_OF,
                                    other_cfg=OTHER, rule_c="by_EL")["plans"]["C"]
    base = strategies.compute_plan(cat, cov, elig, loss, Decimal(5000), CFG, AS_OF,
                                   other_cfg=OTHER)["plans"]["C"]
    assert by_el != base  # прежнее правило — другой состав на случае jury
    assert f"C [{', '.join(by_el)}]" in r["interpretation"]


def test_rule_c_text_names_assumptions():
    t = strategies.RULE_C
    for s in ("min(p, 1−p)", "бинарн", "симметрич", "идеальн", "откалибр", "0,5", "0,37",
              "не фактическое снижение потерь", "partial"):
        assert s in t, s


def test_unknown_rule_rejected():
    with pytest.raises(ValueError):
        strategies.select_c(JURY_ITEMS, JURY_COV, strategies.assessed(JURY_LOSS), ONE_ZONE, CFG,
                            AS_OF, rule_c="bogus")

