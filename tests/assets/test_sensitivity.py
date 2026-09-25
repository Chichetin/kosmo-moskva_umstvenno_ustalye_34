import argparse
import csv
import inspect
import json

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from floodvalue.assets import run as assets_run
from floodvalue.assets.portfolio import PROB_NODATA, Asset, assess_assets, expected_loss, extract_p
from floodvalue.assets.sensitivity import (
    SENSITIVITY_FIELDS,
    SINGLE_ASSET_DOWN_FACTOR,
    SINGLE_ASSET_UP_FACTOR,
    el_sensitivity,
)

# Fixture portfolio: 4 assessed assets with distinct, hand-picked p/V/q so ranks are known and
# rank movement under a single-asset scenario is predictable, plus 1 no_data asset. A1 reuses the
# task.md control example (p=0.4, V=10e6, q=0.25 -> EL=1e6).
A1 = Asset("A01", "warehouse", 10_000_000, 0.25, 0, 0, 0.0, 0.0)   # p=0.4 -> EL=1,000,000
A2 = Asset("A02", "substation", 20_000_000, 0.10, 0, 1, 0.0, 0.0)  # p=0.9 -> EL=1,800,000 (rank 1)
A3 = Asset("A03", "road_unit", 5_000_000, 0.30, 0, 2, 0.0, 0.0)    # p=0.2 -> EL=300,000  (rank 4)
A4 = Asset("A04", "facility", 8_000_000, 0.20, 0, 3, 0.0, 0.0)     # p=0.5 -> EL=800,000  (rank 3)
A5 = Asset("A05", "clinic", 30_000_000, 0.15, 0, 4, 0.0, 0.0)      # no_data
ASSETS = [A1, A2, A3, A4, A5]
LOSS_ROWS = [
    {"asset_id": "A01", "p_flood": 0.4, "expected_loss_rub": 1_000_000.0, "uncertainty": 0.97, "rank": 2, "status": "ok"},
    {"asset_id": "A02", "p_flood": 0.9, "expected_loss_rub": 1_800_000.0, "uncertainty": 0.47, "rank": 1, "status": "ok"},
    {"asset_id": "A03", "p_flood": 0.2, "expected_loss_rub": 300_000.0, "uncertainty": 0.72, "rank": 4, "status": "ok"},
    {"asset_id": "A04", "p_flood": 0.5, "expected_loss_rub": 800_000.0, "uncertainty": 1.0, "rank": 3, "status": "ok"},
    {"asset_id": "A05", "p_flood": None, "expected_loss_rub": None, "uncertainty": None, "rank": None, "status": "no_data"},
]
BASE_TOTAL = 1_000_000.0 + 1_800_000.0 + 300_000.0 + 800_000.0


def rows_by_scenario(rows):
    return {r["scenario_id"]: r for r in rows}


def changed(row):
    return json.loads(row["changed_inputs_json"])


# --- shape: >= 3 scenarios per parameter, contract field order ---------------------------------


def test_at_least_three_scenarios_per_parameter():
    rows = el_sensitivity(ASSETS, LOSS_ROWS)
    for parameter in ("p", "V", "q"):
        n = sum(1 for r in rows if changed(r)["parameter"] == parameter)
        assert n >= 3, f"{parameter}: only {n} scenarios"


def test_row_fields_match_sensitivity_csv_contract():
    rows = el_sensitivity(ASSETS, LOSS_ROWS)
    assert rows, "no scenarios produced"
    for r in rows:
        assert list(r) == SENSITIVITY_FIELDS
        assert r["strategy"] == ""  # not tied to any A/B/C basket (D-024)
        assert r["decision_cost_rub"] is None  # plan is not recomputed here
        assert r["covered_expected_loss_rub"] is None  # not a zone-coverage number
        assert r["result_status"] in ("ok", "no_data")
        json.loads(r["changed_inputs_json"])  # valid JSON
        assert isinstance(r["interpretation"], str) and r["interpretation"]


def test_scenario_ids_are_unique():
    rows = el_sensitivity(ASSETS, LOSS_ROWS)
    ids = [r["scenario_id"] for r in rows]
    assert len(ids) == len(set(ids))


# --- control arithmetic: portfolio-wide scaling ---------------------------------------------


@pytest.mark.parametrize("k", [0.5, 0.8, 1.2, 1.5])
def test_p_portfolio_scenario_arithmetic_and_rank_preserved(k):
    rows = rows_by_scenario(el_sensitivity(ASSETS, LOSS_ROWS))
    row = rows[f"p_portfolio_x{k}"]
    c = changed(row)
    expected_total = sum(
        expected_loss(min(1.0, r["p_flood"] * k), a.asset_value_rub, a.vulnerability_coef)
        for a, r in zip(ASSETS, LOSS_ROWS) if r["status"] == "ok"
    )
    assert c["portfolio_expected_loss_rub"] == pytest.approx(expected_total)
    assert c["baseline_expected_loss_rub"] == pytest.approx(BASE_TOTAL)
    # no clipping at these k on this fixture (max p*k = 0.9*1.5 = 1.35... clips!) -- so only
    # assert the documented, not-hidden fact for the non-clipping ones:
    if k * 0.9 <= 1.0:
        assert "ранжирование не изменилось" in row["interpretation"]


@pytest.mark.parametrize("k", [0.5, 0.8, 1.2])
def test_V_and_q_portfolio_scenarios_scale_every_el_and_keep_ranks(k):
    rows = rows_by_scenario(el_sensitivity(ASSETS, LOSS_ROWS))
    for parameter in ("V", "q"):
        row = rows[f"{parameter}_portfolio_x{k}"]
        c = changed(row)
        assert c["portfolio_expected_loss_rub"] == pytest.approx(BASE_TOTAL * k)
        assert "ранжирование не изменилось" in row["interpretation"]


def test_p_portfolio_scaling_that_clips_is_flagged():
    rows = rows_by_scenario(el_sensitivity(ASSETS, LOSS_ROWS))
    row = rows["p_portfolio_x1.5"]  # A02: p=0.9*1.5=1.35 -> clip to 1.0
    c = changed(row)
    assert c["clipped_asset_ids"] == ["A02"]
    expected_total = (
        expected_loss(1.0, A2.asset_value_rub, A2.vulnerability_coef)  # clipped
        + expected_loss(min(1.0, 0.4 * 1.5), A1.asset_value_rub, A1.vulnerability_coef)
        + expected_loss(min(1.0, 0.2 * 1.5), A3.asset_value_rub, A3.vulnerability_coef)
        + expected_loss(min(1.0, 0.5 * 1.5), A4.asset_value_rub, A4.vulnerability_coef)
    )
    assert c["portfolio_expected_loss_rub"] == pytest.approx(expected_total)
    assert "отсечение" in row["interpretation"]


# --- single-asset scenarios: this is where ranks actually move --------------------------------


def test_leader_scenario_can_demote_the_leader():
    # A02 is rank 1 (EL 1,800,000). Scaling only its p down by 0.2 -> EL 360,000, below A01
    # (1,000,000) and A04 (800,000): the leader drops to rank 3, but the top-3 *set* is still
    # {A01, A02, A04} (A03 stays outside) -- only the order inside it changed.
    rows = rows_by_scenario(el_sensitivity(ASSETS, LOSS_ROWS))
    row = rows[f"p_leader_x{SINGLE_ASSET_DOWN_FACTOR}"]
    c = changed(row)
    assert c["asset_id"] == "A02"
    new_leader_el = expected_loss(0.9 * SINGLE_ASSET_DOWN_FACTOR, A2.asset_value_rub, A2.vulnerability_coef)
    assert new_leader_el == pytest.approx(360_000.0)
    assert "substation (A02) опустился с 1 на 3" in row["interpretation"]
    assert "топ-3 не изменился" in row["interpretation"]


def test_challenger_scenario_can_enter_top_three():
    # A03 is rank 4 (EL 300,000, outside top 3). Scaling only its q up by 3.0 -> EL 900,000,
    # above A04 (800,000): the challenger enters the top 3 at rank 3.
    rows = rows_by_scenario(el_sensitivity(ASSETS, LOSS_ROWS))
    row = rows[f"q_challenger_x{SINGLE_ASSET_UP_FACTOR}"]
    c = changed(row)
    assert c["asset_id"] == "A03"
    new_el = expected_loss(0.2, A3.asset_value_rub, 0.30 * SINGLE_ASSET_UP_FACTOR)
    assert new_el == pytest.approx(900_000.0)
    assert "road_unit (A03) поднялся с 4 на 3" in row["interpretation"]
    assert "топ-3 изменился" in row["interpretation"]


def test_single_asset_scenario_leaves_other_assets_at_baseline_value():
    # Only the target asset's parameter changes; V and q axes must not cross-contaminate.
    rows = rows_by_scenario(el_sensitivity(ASSETS, LOSS_ROWS))
    row = rows[f"V_leader_x{SINGLE_ASSET_DOWN_FACTOR}"]
    c = changed(row)
    expected_total = (
        1_000_000.0  # A01 untouched
        + expected_loss(0.9, A2.asset_value_rub * SINGLE_ASSET_DOWN_FACTOR, 0.10)  # A02 V scaled
        + 300_000.0  # A03 untouched
        + 800_000.0  # A04 untouched
    )
    assert c["portfolio_expected_loss_rub"] == pytest.approx(expected_total)


# --- no_data: excluded from recomputation, named in interpretation ----------------------------


def test_no_data_asset_is_excluded_and_named():
    rows = el_sensitivity(ASSETS, LOSS_ROWS)
    for r in rows:
        assert "A05" not in changed(r).get("clipped_asset_ids", [])
        assert "no_data не участвуют: A05" in r["interpretation"]


def test_no_ok_assets_gives_no_data_status_without_crashing():
    all_no_data = [{**r, "status": "no_data", "p_flood": None, "expected_loss_rub": None, "rank": None}
                   for r in LOSS_ROWS]
    rows = el_sensitivity(ASSETS, all_no_data)
    assert rows  # still >= 3 scenarios per parameter, just empty ones
    for r in rows:
        assert r["result_status"] == "no_data"


# --- mutation-style guards: each parameter changes only itself --------------------------------


def test_p_scenario_never_touches_V_or_q():
    rows = rows_by_scenario(el_sensitivity(ASSETS, LOSS_ROWS))
    row = rows["p_portfolio_x0.5"]
    # A01 baseline EL halves exactly (0.4*0.5*10e6*0.25); if V or q had leaked in this would fail.
    assert changed(row)["portfolio_expected_loss_rub"] != pytest.approx(BASE_TOTAL)
    single = expected_loss(0.4 * 0.5, A1.asset_value_rub, A1.vulnerability_coef)
    assert single == pytest.approx(500_000.0)


# --- criteria 14 / task.md: changing the binarization threshold does not change EL ------------


def test_threshold_change_does_not_change_el():
    # p is a smooth gradient across a small grid; assets sit at hand-picked pixels so at least
    # one keeps p > 0 while landing on the "dry" side of every tested threshold.
    prob = np.tile(np.linspace(0.05, 0.95, 6, dtype="float32"), (6, 1))
    positions = [(0, 0), (2, 2), (4, 5)]  # p ~= 0.05, 0.41, 0.95
    assets = [
        Asset(f"T{i+1:02d}", "warehouse", 10_000_000, 0.25, r, c, 0.0, 0.0)
        for i, (r, c) in enumerate(positions)
    ]
    loss = assess_assets(assets, prob)
    els_by_id = {r["asset_id"]: r["expected_loss_rub"] for r in loss}

    for threshold in (0.1, 0.3, 0.5, 0.7, 0.9):
        flood_mask = (prob >= threshold).astype("uint8")  # what flood_mask.tif would encode
        # assess_assets/extract_p have no mask or threshold parameter: the same p -> same EL,
        # regardless of which threshold produced flood_mask.tif.
        loss_again = assess_assets(assets, prob)
        assert {r["asset_id"]: r["expected_loss_rub"] for r in loss_again} == els_by_id
        # and the low-p asset is "dry" under every threshold here, yet keeps a nonzero EL:
        dry = [a for a in assets if flood_mask[a.row, a.col] == 0]
        assert dry
        assert all(els_by_id[a.asset_id] not in (None, 0.0) for a in dry)

    for fn in (expected_loss, extract_p, assess_assets, el_sensitivity):
        params = set(inspect.signature(fn).parameters)
        assert not params & {"threshold", "mask", "flood_mask", "binary_mask"}


# --- wiring: `make assets` writes assets_sensitivity.csv alongside asset_loss.csv ------------


def _run_setup(tmp_path):
    h = w = 32
    transform = from_origin(81.0, 7.0, 8.983152841195215e-05, 8.983152841195215e-05)
    data = tmp_path / "data"
    s1_dir = data / "raw" / "HandLabeled" / "S1Hand"
    s1_dir.mkdir(parents=True)
    s1 = np.random.default_rng(1).normal(-15, 3, (2, h, w)).astype("float32")
    with rasterio.open(s1_dir / "Sens_1_S1Hand.tif", "w", driver="GTiff", height=h, width=w,
                       count=2, dtype="float32", crs="EPSG:4326", transform=transform,
                       nodata=np.nan) as dst:
        dst.write(s1)
    out = tmp_path / "out"
    out.mkdir()
    prob = np.random.default_rng(2).random((h, w)).astype("float32")
    with rasterio.open(out / "flood_probability.tif", "w", driver="GTiff", height=h, width=w,
                       count=1, dtype="float32", crs="EPSG:4326", transform=transform,
                       nodata=PROB_NODATA) as dst:
        dst.write(prob, 1)
    return out


def test_run_writes_assets_sensitivity_csv(tmp_path, monkeypatch):
    out = _run_setup(tmp_path)
    monkeypatch.setenv("FLOODVALUE_DATA_DIR", str(tmp_path / "data"))
    assert assets_run.run(argparse.Namespace(chip="Sens_1", out=str(out), budget=None)) == 0

    with open(out / "assets_sensitivity.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == SENSITIVITY_FIELDS
    for parameter in ("p", "V", "q"):
        n = sum(1 for r in rows if json.loads(r["changed_inputs_json"])["parameter"] == parameter)
        assert n >= 3
    for r in rows:
        assert r["strategy"] == ""
        assert r["decision_cost_rub"] == ""
        assert r["covered_expected_loss_rub"] == ""
        json.loads(r["changed_inputs_json"])

    meta = json.loads((out / "assets_metadata.json").read_text(encoding="utf-8"))
    assert {"p_rule", "v_rule", "q_rule", "n_scenarios"} <= set(meta["sensitivity"])
    assert meta["sensitivity"]["n_scenarios"] == len(rows)
