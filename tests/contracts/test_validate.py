"""Validator of the 12 required files: reference set passes, each broken set fails its file."""

import csv
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
import rasterio
from pyproj import Transformer
from shapely.geometry import Polygon, mapping

from floodvalue import cli
from floodvalue.contracts import files as F
from floodvalue.contracts.geo import area_km2
from floodvalue.export import validate as V

GOOD = Path(__file__).resolve().parents[1] / "fixtures" / "good"


@pytest.fixture
def run_dir(tmp_path) -> Path:
    dst = tmp_path / "run"
    shutil.copytree(GOOD, dst)
    return dst


def failing(run_dir: Path) -> dict[str, list[str]]:
    return {k: v for k, v in V.validate_run(run_dir).items() if v}


def edit_csv(path: Path, fn) -> None:
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        header, rows = reader.fieldnames, list(reader)
    rows = fn(rows) or rows
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)


def edit_json(path: Path, fn) -> None:
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc = fn(doc) or doc
    path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")


def row(rows, **match):
    return next(r for r in rows if all(r[k] == v for k, v in match.items()))


def only(errors: dict, name: str) -> str:
    assert set(errors) == {name}, errors
    return " ".join(errors[name])


# --- reference set -----------------------------------------------------------------------------

def test_good_fixture_passes_12_of_12():
    assert failing(GOOD) == {}


def test_cli_validate_good_exits_0(capsys):
    assert cli.main(["validate", "--out", str(GOOD)]) == 0
    assert "12/12" in capsys.readouterr().out


def test_cli_validate_broken_exits_1(run_dir, capsys):
    (run_dir / F.SENSITIVITY).unlink()
    assert cli.main(["validate", "--out", str(run_dir)]) == 1
    assert "11/12" in capsys.readouterr().out


def test_default_run_dir_prefers_newest_output_else_fixture(tmp_path, monkeypatch):
    monkeypatch.setenv("FLOODVALUE_OUTPUT_DIR", str(tmp_path / "none"))
    assert V.default_run_dir() == V.FIXTURE_RUN
    monkeypatch.setenv("FLOODVALUE_OUTPUT_DIR", str(tmp_path))
    (tmp_path / "half").mkdir()  # no run_metadata.json -> not a set
    assert V.default_run_dir() == V.FIXTURE_RUN
    for name in ("old", "new"):
        (tmp_path / name).mkdir()
        (tmp_path / name / F.RUN_METADATA).write_text("{}", encoding="utf-8")
    os.utime(tmp_path / "old" / F.RUN_METADATA, (1, 1))
    assert V.default_run_dir() == tmp_path / "new"


# --- NaN / Inf ---------------------------------------------------------------------------------

def test_nan_literal_in_csv_fails(run_dir):
    edit_csv(run_dir / F.ASSET_LOSS, lambda rows: row(rows, asset_id="a01").update(
        p_flood="NaN"))
    assert "NaN" in only(failing(run_dir), F.ASSET_LOSS)


def test_inf_in_comparison_fails(run_dir):
    edit_csv(run_dir / F.STRATEGY_COMPARISON, lambda rows: row(rows, strategy="A").update(
        residual_uncertainty="inf"))
    assert "'inf' — вместо NaN/Inf нужна пустая ячейка" in only(failing(run_dir), F.STRATEGY_COMPARISON)


def test_nan_in_json_fails(run_dir):
    text = (run_dir / F.RUN_METADATA).read_text(encoding="utf-8")
    (run_dir / F.RUN_METADATA).write_text(text.replace('"threshold": 0.5', '"threshold": NaN', 1),
                                          encoding="utf-8")
    assert "недопустимое значение NaN" in " ".join(failing(run_dir)[F.RUN_METADATA])


def test_nan_in_probability_raster_fails(run_dir):
    with rasterio.open(run_dir / F.FLOOD_PROBABILITY, "r+") as dst:
        p = dst.read(1)
        p[40, 40] = np.nan
        dst.write(p, 1)
    assert "NaN/Inf в 1 пикселях" in " ".join(failing(run_dir)[F.FLOOD_PROBABILITY])


# --- rasters -----------------------------------------------------------------------------------

def test_mask_not_matching_threshold_fails(run_dir):
    with rasterio.open(run_dir / F.FLOOD_MASK, "r+") as dst:
        m = dst.read(1)
        m[50, 50] = 1 - m[50, 50]
        dst.write(m, 1)
    assert "threshold" in only(failing(run_dir), F.FLOOD_MASK)


def test_mask_boundary_p_equal_threshold_is_water(run_dir):
    # p == threshold must be 1 (p >= threshold): set a pixel to exactly 0.5 with mask 0
    with rasterio.open(run_dir / F.FLOOD_PROBABILITY, "r+") as dst:
        p = dst.read(1)
        p[62, 0] = 0.5
        dst.write(p, 1)
    with rasterio.open(run_dir / F.FLOOD_MASK, "r+") as dst:
        m = dst.read(1)
        m[62, 0] = 0
        dst.write(m, 1)
    assert F.FLOOD_MASK in failing(run_dir)


def test_mask_nodata_inconsistent_fails(run_dir):
    with rasterio.open(run_dir / F.FLOOD_MASK, "r+") as dst:
        m = dst.read(1)
        m[0, 0] = 0  # probability is -9999 there
        dst.write(m, 1)
    assert "nodata" in only(failing(run_dir), F.FLOOD_MASK)


def test_raster_grid_mismatch_fails(run_dir):
    with rasterio.open(run_dir / F.FLOOD_MASK) as src:
        prof, m = src.profile, src.read(1)
    prof["transform"] = rasterio.transform.from_origin(80.001, 7.8, 0.0005, 0.0005)
    with rasterio.open(run_dir / F.FLOOD_MASK, "w", **prof) as dst:
        dst.write(m, 1)
    assert "сетка" in " ".join(failing(run_dir)[F.FLOOD_MASK])


def test_probability_wrong_nodata_fails(run_dir):
    with rasterio.open(run_dir / F.FLOOD_PROBABILITY) as src:
        prof, p = src.profile, src.read(1)
    prof["nodata"] = -1.0
    with rasterio.open(run_dir / F.FLOOD_PROBABILITY, "w", **prof) as dst:
        dst.write(p, 1)
    assert "nodata -1.0, нужен -9999" in " ".join(failing(run_dir)[F.FLOOD_PROBABILITY])


# --- assets ------------------------------------------------------------------------------------

def test_duplicate_asset_id_fails(run_dir):
    def dup(doc):
        doc["features"][1]["properties"]["asset_id"] = "a01"
    edit_json(run_dir / F.ASSETS_GEOJSON, dup)
    errs = failing(run_dir)
    assert "повторяется" in " ".join(errs[F.ASSETS_GEOJSON])


def test_duplicate_asset_id_in_loss_fails(run_dir):
    edit_csv(run_dir / F.ASSET_LOSS, lambda rows: rows + [dict(rows[-1])])
    assert "повторяется" in only(failing(run_dir), F.ASSET_LOSS)


def test_wrong_vq_fails(run_dir):
    def wrong(doc):
        doc["features"][3]["properties"]["vulnerability_coef"] = 0.5
    edit_json(run_dir / F.ASSETS_GEOJSON, wrong)
    assert "V/q" in " ".join(failing(run_dir)[F.ASSETS_GEOJSON])


def test_nine_assets_fails(run_dir):
    edit_json(run_dir / F.ASSETS_GEOJSON, lambda d: d["features"].pop() and None)
    assert "ровно 10" in " ".join(failing(run_dir)[F.ASSETS_GEOJSON])


def test_swapped_lat_lon_fails(run_dir):
    def swap(doc):
        c = doc["features"][4]["geometry"]["coordinates"]
        c.reverse()
    edit_json(run_dir / F.ASSETS_GEOJSON, swap)
    assert "вне чипа" in " ".join(failing(run_dir)[F.ASSETS_GEOJSON])


def test_assets_csv_coordinates_differ_fails(run_dir):
    edit_csv(run_dir / F.ASSETS_CSV, lambda rows: row(rows, asset_id="a05").update(
        longitude=str(float(row(rows, asset_id="a05")["longitude"]) + 0.001)))
    assert "координаты" in only(failing(run_dir), F.ASSETS_CSV)


# --- asset_loss --------------------------------------------------------------------------------

def test_el_not_p_v_q_fails(run_dir):
    edit_csv(run_dir / F.ASSET_LOSS, lambda rows: row(rows, asset_id="a04").update(
        expected_loss_rub="507936.60"))
    assert "p·V·q" in only(failing(run_dir), F.ASSET_LOSS)


def test_p_not_from_raster_fails(run_dir):
    # p taken from the mask (0/1) instead of the probability raster
    def from_mask(rows):
        r = row(rows, asset_id="a04")
        r["p_flood"], r["expected_loss_rub"] = "0.0", "0.00"
        return sorted_ranks(rows)
    edit_csv(run_dir / F.ASSET_LOSS, from_mask)
    assert "растру" in " ".join(failing(run_dir)[F.ASSET_LOSS])


def sorted_ranks(rows):
    ok = sorted((r for r in rows if r["status"] == "ok"),
                key=lambda r: (-float(r["expected_loss_rub"]), r["asset_id"]))
    for i, r in enumerate(ok, start=1):
        r["rank"] = str(i)
    return rows


def test_rank_rule_fails(run_dir):
    def swap(rows):
        a, b = row(rows, rank="1"), row(rows, rank="2")
        a["rank"], b["rank"] = "2", "1"
    edit_csv(run_dir / F.ASSET_LOSS, swap)
    assert "rank" in only(failing(run_dir), F.ASSET_LOSS)


def test_rank_tie_broken_by_asset_id(run_dir):
    # a02 and a10 get equal EL -> a02 must rank before a10
    def tie(rows):
        row(rows, asset_id="a10")["expected_loss_rub"] = row(rows, asset_id="a02")["expected_loss_rub"]
        row(rows, asset_id="a02")["rank"], row(rows, asset_id="a10")["rank"] = "2", "1"
    edit_csv(run_dir / F.ASSET_LOSS, tie)
    assert "rank" in " ".join(failing(run_dir)[F.ASSET_LOSS])


def test_no_data_with_zero_p_fails(run_dir):
    edit_csv(run_dir / F.ASSET_LOSS, lambda rows: row(rows, asset_id="a01").update(
        p_flood="0", expected_loss_rub="0"))
    assert "пустыми" in only(failing(run_dir), F.ASSET_LOSS)


def test_ok_on_nodata_pixel_fails(run_dir):
    def fake_ok(rows):
        r = row(rows, asset_id="a01")
        r.update(p_flood="0.0", expected_loss_rub="0.00", uncertainty="0", status="ok")
        return sorted_ranks(rows)
    edit_csv(run_dir / F.ASSET_LOSS, fake_ok)
    assert "nodata" in only(failing(run_dir), F.ASSET_LOSS)


# --- candidate zones ---------------------------------------------------------------------------

TO_EA = Transformer.from_crs("EPSG:4326", "EPSG:6933", always_xy=True)
TO_LL = Transformer.from_crs("EPSG:6933", "EPSG:4326", always_xy=True)


def square_ll(x, y, side, hole=None):
    def ring(cx, cy, s):
        h = s / 2
        return [TO_LL.transform(px, py) for px, py in
                [(cx - h, cy - h), (cx + h, cy - h), (cx + h, cy + h), (cx - h, cy + h)]]
    return Polygon(ring(x, y, side), [ring(x, y, hole)] if hole else None)


def replace_zone(run_dir, poly, area_km2_value):
    def upd(doc):
        f = next(f for f in doc["features"] if f["properties"]["candidate_id"] == "Z3")
        f["geometry"] = mapping(poly)
        f["geometry"]["coordinates"] = [list(map(list, r)) for r in f["geometry"]["coordinates"]]
        f["properties"]["area_km2"] = area_km2_value
    edit_json(run_dir / F.CANDIDATE_ORDERS, upd)


def test_polygon_with_hole_fails(run_dir):
    x, y = TO_EA.transform(80.01, 7.78)
    replace_zone(run_dir, square_ll(x, y, 1300, hole=100), 1.68)
    assert "holes" in " ".join(failing(run_dir)[F.CANDIDATE_ORDERS])


def test_area_09996_rounded_to_1_fails(run_dir):
    x, y = TO_EA.transform(80.01, 7.78)
    poly = square_ll(x, y, 0.9996 ** 0.5 * 1000)
    assert 0.9995 < area_km2(poly) < 0.99965
    replace_zone(run_dir, poly, 1.0)
    msg = " ".join(failing(run_dir)[F.CANDIDATE_ORDERS])
    assert "< 1 км²" in msg


def test_area_km2_not_rounded_half_up_fails(run_dir):
    def upd(doc):
        doc["features"][0]["properties"]["area_km2"] = 1.2
    edit_json(run_dir / F.CANDIDATE_ORDERS, upd)
    assert "ROUND_HALF_UP" in " ".join(failing(run_dir)[F.CANDIDATE_ORDERS])


def test_multipolygon_fails(run_dir):
    def upd(doc):
        g = doc["features"][0]["geometry"]
        doc["features"][0]["geometry"] = {"type": "MultiPolygon", "coordinates": [g["coordinates"]]}
    edit_json(run_dir / F.CANDIDATE_ORDERS, upd)
    assert "'MultiPolygon', ожидается Polygon" in " ".join(failing(run_dir)[F.CANDIDATE_ORDERS])


def test_self_intersecting_polygon_fails(run_dir):
    def upd(doc):
        ring = doc["features"][2]["geometry"]["coordinates"][0]
        ring[1], ring[2] = ring[2], ring[1]  # bow-tie
    edit_json(run_dir / F.CANDIDATE_ORDERS, upd)
    assert "невалидный" in " ".join(failing(run_dir)[F.CANDIDATE_ORDERS])


def test_duplicate_candidate_id_fails(run_dir):
    def upd(doc):
        doc["features"][3]["properties"]["candidate_id"] = "Z3"
    edit_json(run_dir / F.CANDIDATE_ORDERS, upd)
    assert "повторяется" in " ".join(failing(run_dir)[F.CANDIDATE_ORDERS])


def test_new_without_guarantee_fails(run_dir):
    def upd(doc):
        doc["features"][0]["properties"]["guaranteed_purchase"] = False
    edit_json(run_dir / F.CANDIDATE_ORDERS, upd)
    assert "acquisition_type=new без guaranteed_purchase=true" in \
        " ".join(failing(run_dir)[F.CANDIDATE_ORDERS])


def test_archive_with_guarantee_fails(run_dir):
    def upd(doc):
        f = next(f for f in doc["features"] if f["properties"]["candidate_id"] == "Z3")
        f["properties"]["guaranteed_purchase"] = True
    edit_json(run_dir / F.CANDIDATE_ORDERS, upd)
    assert "guaranteed_purchase=true допустимо только для acquisition_type=new" in \
        " ".join(failing(run_dir)[F.CANDIDATE_ORDERS])


# --- plan, strategies, comparison --------------------------------------------------------------

def test_c_over_budget_fails(run_dir):
    def small(rows):
        for r in rows:
            r["budget_rub"] = "2000.00"
            r["budget_feasible"] = str(float(r["data_cost_rub"]) <= 2000).lower()
    edit_csv(run_dir / F.STRATEGY_COMPARISON, small)
    edit_json(run_dir / F.RUN_METADATA, lambda d: d["budget"].update(budget_rub=2000.0))
    assert "C: data_cost_rub" in only(failing(run_dir), F.STRATEGY_COMPARISON)


def test_b_over_budget_marked_feasible_fails(run_dir):
    edit_csv(run_dir / F.STRATEGY_COMPARISON, lambda rows: row(rows, strategy="B").update(
        budget_feasible="true"))
    assert "budget_feasible" in only(failing(run_dir), F.STRATEGY_COMPARISON)


def test_data_cost_not_sum_of_rows_fails(run_dir):
    def upd(rows):
        r = row(rows, strategy="C")
        r["data_cost_rub"], r["decision_cost_rub"] = "2900.00", "4400.00"
    edit_csv(run_dir / F.STRATEGY_COMPARISON, upd)
    assert "сумме строк" in only(failing(run_dir), F.STRATEGY_COMPARISON)


def test_decision_cost_not_data_plus_other_fails(run_dir):
    edit_csv(run_dir / F.STRATEGY_COMPARISON, lambda rows: row(rows, strategy="A").update(
        decision_cost_rub="0.00"))
    assert "decision_cost_rub" in only(failing(run_dir), F.STRATEGY_COMPARISON)


def test_not_estimated_with_number_fails(run_dir):
    edit_csv(run_dir / F.STRATEGY_COMPARISON, lambda rows: row(rows, strategy="B").update(
        residual_uncertainty="0.1"))
    assert "not_estimated" in only(failing(run_dir), F.STRATEGY_COMPARISON)


def test_unknown_uncertainty_status_fails(run_dir):
    edit_csv(run_dir / F.STRATEGY_COMPARISON, lambda rows: row(rows, strategy="C").update(
        uncertainty_status="reduced"))
    assert "uncertainty_status" in only(failing(run_dir), F.STRATEGY_COMPARISON)


def test_a_must_be_baseline(run_dir):
    edit_csv(run_dir / F.STRATEGY_COMPARISON, lambda rows: row(rows, strategy="A").update(
        uncertainty_status="scenario"))
    assert "baseline" in only(failing(run_dir), F.STRATEGY_COMPARISON)


def test_covered_counts_late_zone_fails(run_dir):
    # claiming that B covers more EL than zones available by the deadline can observe
    edit_csv(run_dir / F.STRATEGY_COMPARISON, lambda rows: row(rows, strategy="B").update(
        covered_expected_loss_rub="9000000.00", coverage_share=""))
    assert "covered_expected_loss_rub" in " ".join(failing(run_dir)[F.STRATEGY_COMPARISON])


def test_strategy_plans_differ_from_plan_rows_fails(run_dir):
    edit_json(run_dir / F.STRATEGY_PLANS, lambda d: d["C"].append("Z4"))
    assert "procurement_plan" in only(failing(run_dir), F.STRATEGY_PLANS)


def test_strategy_a_with_zones_fails(run_dir):
    edit_json(run_dir / F.STRATEGY_PLANS, lambda d: d["A"].append("Z1"))
    assert "список A должен быть пустым" in only(failing(run_dir), F.STRATEGY_PLANS)


def test_plan_row_for_a_fails(run_dir):
    edit_csv(run_dir / F.PROCUREMENT_PLAN, lambda rows: rows + [dict(rows[-1], strategy="A")])
    assert "strategy: Input should be 'B' or 'C'" in only(failing(run_dir), F.PROCUREMENT_PLAN)


def test_plan_duplicate_pair_fails(run_dir):
    edit_csv(run_dir / F.PROCUREMENT_PLAN, lambda rows: rows + [dict(rows[-1])])
    assert "повторяется" in " ".join(failing(run_dir)[F.PROCUREMENT_PLAN])


def test_plan_cost_not_recomputed_fails(run_dir):
    edit_csv(run_dir / F.PROCUREMENT_PLAN, lambda rows: row(rows, strategy="C", candidate_id="Z1")
             .update(cost_rub="2061.20"))
    assert "cost_rub" in " ".join(failing(run_dir)[F.PROCUREMENT_PLAN])


def test_plan_discount_of_single_zone_in_basket_fails(run_dir):
    # B's optical basket (3 zones) must use the basket discount, not the single-zone Р = 1
    def single(rows):
        r = row(rows, strategy="B", candidate_id="Z1")
        r.update(discount_coef="1.000000000", unit_price_rub_km2="1703.46", cost_rub="2061.19")
    edit_csv(run_dir / F.PROCUREMENT_PLAN, single)
    assert "discount_coef" in " ".join(failing(run_dir)[F.PROCUREMENT_PLAN])


def test_plan_category_differs_from_catalog_fails(run_dir):
    edit_csv(run_dir / F.PROCUREMENT_PLAN, lambda rows: row(rows, strategy="C", candidate_id="Z2")
             .update(usage_type="limited", usage_coef="1.2"))
    assert "каталогу" in " ".join(failing(run_dir)[F.PROCUREMENT_PLAN])


def test_plan_wrong_freshness_fails(run_dir):
    edit_csv(run_dir / F.PROCUREMENT_PLAN, lambda rows: row(rows, strategy="B", candidate_id="Z3")
             .update(freshness_coef="1.0"))
    assert "freshness_coef" in " ".join(failing(run_dir)[F.PROCUREMENT_PLAN])


def test_plan_bool_must_be_true_false(run_dir):
    edit_csv(run_dir / F.PROCUREMENT_PLAN, lambda rows: row(rows, strategy="C", candidate_id="Z1")
             .update(guaranteed_purchase="1"))
    assert "ожидается true/false" in only(failing(run_dir), F.PROCUREMENT_PLAN)


# --- sensitivity, metadata, manifest -----------------------------------------------------------

def test_sensitivity_missing_q_fails(run_dir):
    edit_csv(run_dir / F.SENSITIVITY, lambda rows: [r for r in rows
                                                    if "vulnerability_coef" not in r["changed_inputs_json"]])
    assert "vulnerability_coef" in only(failing(run_dir), F.SENSITIVITY)


def test_sensitivity_invalid_json_fails(run_dir):
    edit_csv(run_dir / F.SENSITIVITY, lambda rows: rows[0].update(changed_inputs_json="{p: 1"))
    assert "строка 2: changed_inputs_json" in only(failing(run_dir), F.SENSITIVITY)


def test_metadata_missing_key_fails(run_dir):
    edit_json(run_dir / F.RUN_METADATA, lambda d: d.pop("proxy_check"))
    assert "proxy_check: Field required" in " ".join(failing(run_dir)[F.RUN_METADATA])


def test_metadata_threshold_changed_breaks_mask(run_dir):
    def upd(d):
        d["threshold"] = 0.6
        d["methods"]["main"]["threshold"] = 0.6
    edit_json(run_dir / F.RUN_METADATA, upd)
    assert "threshold" in only(failing(run_dir), F.FLOOD_MASK)


def test_manifest_without_chip_s1_fails(run_dir):
    def upd(d):
        d["sources"][0]["id"] = "other"
        d["sources"][0]["file"] = "other.tif"
    edit_json(run_dir / F.SOURCE_MANIFEST, upd)
    assert "S1" in only(failing(run_dir), F.SOURCE_MANIFEST)


def test_manifest_future_source_without_note_fails(run_dir):
    def upd(d):
        d["sources"].append(dict(d["sources"][0], id="jrc", file="jrc.tif",
                                 observation_date="2018-12-31"))
    edit_json(run_dir / F.SOURCE_MANIFEST, upd)
    assert "after_event_note" in only(failing(run_dir), F.SOURCE_MANIFEST)


def test_missing_file_fails(run_dir):
    (run_dir / F.STRATEGY_PLANS).unlink()
    assert "отсутствует" in only(failing(run_dir), F.STRATEGY_PLANS)


def test_rank_tie_order_independent_of_row_order(run_dir):
    # equal EL for a02 and a10, rows written a10 first: a02 still ranks first
    def tie(rows):
        row(rows, asset_id="a10")["expected_loss_rub"] = row(rows, asset_id="a02")["expected_loss_rub"]
        rows.sort(key=lambda r: r["asset_id"], reverse=True)
        sorted_ranks(rows)
    edit_csv(run_dir / F.ASSET_LOSS, tie)
    errs = " ".join(failing(run_dir).get(F.ASSET_LOSS, []))
    assert "rank" not in errs
    assert row_rank(run_dir, "a02") < row_rank(run_dir, "a10")


def row_rank(run_dir, aid):
    with open(run_dir / F.ASSET_LOSS, encoding="utf-8") as f:
        return int(next(r for r in csv.DictReader(f) if r["asset_id"] == aid)["rank"])


def test_coverage_counts_boundary_point_once():
    from shapely.geometry import Point, box
    zones = [box(0, 0, 1, 1), box(1, 0, 2, 1)]  # a on the shared edge, b on a corner
    pts = {"a": Point(1, 0.5), "b": Point(0, 0), "c": Point(5, 5)}
    assert V.covered_expected_loss(zones, pts, {"a": 10.0, "b": 1.0, "c": 100.0}) == 11.0


def test_metrics_of_both_methods_required(run_dir):
    edit_csv(run_dir / "metrics.csv", lambda rows: [r for r in rows if r["method"] != "baseline"])
    assert "baseline и main" in only(failing(run_dir), F.RUN_METADATA)


def test_metrics_on_different_valid_masks_fails(run_dir):
    edit_csv(run_dir / "metrics.csv", lambda rows: rows[0].update(valid_mask_sha256="1" * 64))
    assert "разных масках" in only(failing(run_dir), F.RUN_METADATA)


def test_metrics_file_missing_fails(run_dir):
    (run_dir / "metrics.csv").unlink()
    assert "не найден" in only(failing(run_dir), F.RUN_METADATA)


def test_strategy_a_nonzero_coverage_without_declaration_fails(run_dir):
    edit_csv(run_dir / F.STRATEGY_COMPARISON, lambda rows: row(rows, strategy="A").update(
        covered_expected_loss_rub="100.00", coverage_share=""))
    assert "coverage_A" in " ".join(failing(run_dir)[F.STRATEGY_COMPARISON])


def test_cli_marks_fixture_fallback(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FLOODVALUE_OUTPUT_DIR", str(tmp_path / "none"))
    assert cli.main(["validate"]) == 0
    assert "ЭТАЛОН" in capsys.readouterr().out


# --- P01b (#14): [lon, lat] of zones, closed rings, catalog ↔ pricing.yaml, plan ↔ plans ↔ catalog

def zone(doc, cid):
    return next(f for f in doc["features"] if f["properties"]["candidate_id"] == cid)


def test_zone_swapped_lat_lon_fails(run_dir):
    # [lat, lon] instead of [lon, lat]: (7.78, 80.01) is still in range, but not on the chip
    def swap(doc):
        for pos in zone(doc, "Z2")["geometry"]["coordinates"][0]:
            pos.reverse()
    edit_json(run_dir / F.CANDIDATE_ORDERS, swap)
    assert "Z2: зона не пересекает чип (порядок [долгота, широта]?)" in \
        " ".join(failing(run_dir)[F.CANDIDATE_ORDERS])


def test_zone_outside_chip_fails(run_dir):
    # a valid ≥ 1 km² zone far from the chip, area written correctly: only the extent check fires
    x, y = TO_EA.transform(81.5, 7.78)
    poly = square_ll(x, y, 1300)
    replace_zone(run_dir, poly, V.round_half_up(area_km2(poly), "0.001"))
    msg = " ".join(failing(run_dir)[F.CANDIDATE_ORDERS])
    assert "Z3: зона не пересекает чип" in msg
    assert "ROUND_HALF_UP" not in msg


def test_zone_ring_not_closed_fails(run_dir):
    # GeoJSON (RFC 7946 §3.1.6): the first and last positions of a ring are identical
    def unclose(doc):
        zone(doc, "Z1")["geometry"]["coordinates"][0].pop()
    edit_json(run_dir / F.CANDIDATE_ORDERS, unclose)
    assert "объект 0: кольцо 0 не замкнуто" in only(failing(run_dir), F.CANDIDATE_ORDERS)


def test_nan_coordinate_in_geojson_fails(run_dir):
    text = (run_dir / F.ASSETS_GEOJSON).read_text(encoding="utf-8")
    doc = json.loads(text)
    lon = doc["features"][0]["geometry"]["coordinates"][0]
    (run_dir / F.ASSETS_GEOJSON).write_text(text.replace(json.dumps(lon), "NaN", 1), encoding="utf-8")
    assert "недопустимое значение NaN" in " ".join(failing(run_dir)[F.ASSETS_GEOJSON])


def test_catalog_base_rate_not_pricing_config_fails(run_dir):
    edit_json(run_dir / F.CANDIDATE_ORDERS,
              lambda d: zone(d, "Z4")["properties"].update(base_rate_rub_km2=800.0))
    assert "Z4: base_rate_rub_km2=800.0 ≠ configs/pricing.yaml 788.64" in \
        " ".join(failing(run_dir)[F.CANDIDATE_ORDERS])


def test_catalog_base_rate_status_official_without_config_fails(run_dir):
    edit_json(run_dir / F.CANDIDATE_ORDERS,
              lambda d: zone(d, "Z4")["properties"].update(base_rate_status="official"))
    assert "Z4: base_rate_status='official' ≠ configs/pricing.yaml 'scenario'" in \
        " ".join(failing(run_dir)[F.CANDIDATE_ORDERS])


def test_metadata_pricing_differs_from_config_fails(run_dir):
    edit_json(run_dir / F.RUN_METADATA, lambda d: d["pricing"].update(base_rate_status="official"))
    assert "pricing.base_rate_status='official' ≠ configs/pricing.yaml 'scenario'" in \
        only(failing(run_dir), F.RUN_METADATA)


def test_metadata_legal_edition_differs_from_config_fails(run_dir):
    edit_json(run_dir / F.RUN_METADATA, lambda d: d["pricing"].update(legal_edition="ред. 2019"))
    assert "pricing.legal_edition" in only(failing(run_dir), F.RUN_METADATA)


def test_plan_candidate_not_in_catalog_fails(run_dir):
    edit_csv(run_dir / F.PROCUREMENT_PLAN, lambda rows: row(rows, strategy="C", candidate_id="Z2")
             .update(candidate_id="Z9"))
    assert "C/Z9: candidate_id нет в candidate_orders.geojson" in \
        " ".join(failing(run_dir)[F.PROCUREMENT_PLAN])


def test_strategy_plans_unknown_candidate_fails(run_dir):
    edit_json(run_dir / F.STRATEGY_PLANS, lambda d: d["B"].append("Z9"))
    msg = only(failing(run_dir), F.STRATEGY_PLANS)
    assert "B: нет в candidate_orders.geojson: ['Z9']" in msg


def test_strategy_plans_missing_plan_row_fails(run_dir):
    edit_json(run_dir / F.STRATEGY_PLANS, lambda d: d["B"].remove("Z3"))
    assert "B: состав ≠ строкам procurement_plan.csv" in only(failing(run_dir), F.STRATEGY_PLANS)


def test_b_over_budget_counterfactual_passes(run_dir):
    # reference B exceeds the budget and is marked budget_feasible=false: that is allowed
    with open(run_dir / F.STRATEGY_COMPARISON, encoding="utf-8") as f:
        b = row(list(csv.DictReader(f)), strategy="B")
    assert float(b["data_cost_rub"]) > float(b["budget_rub"]) and b["budget_feasible"] == "false"
    assert failing(run_dir) == {}


def test_zone_ring_too_short_fails(run_dir):
    # [a, b, a] is "closed" but not a linear ring: at least 4 positions
    def short(doc):
        ring = zone(doc, "Z1")["geometry"]["coordinates"][0]
        ring[:] = [ring[0], ring[1], ring[0]]
    edit_json(run_dir / F.CANDIDATE_ORDERS, short)
    assert "объект 0: кольцо 0 не замкнуто или короче 4 точек" in \
        only(failing(run_dir), F.CANDIDATE_ORDERS)


def test_chip_extent_non_square_and_projected():
    from rasterio.crs import CRS
    from rasterio.transform import from_origin
    p = np.zeros((2, 5), dtype=np.float64)  # 2 rows, 5 columns
    prof = {"crs": CRS.from_epsg(4326), "transform": from_origin(80.0, 7.8, 0.01, 0.01)}
    assert V._chip_lonlat((p, p > 0, prof)).bounds == pytest.approx((80.0, 7.78, 80.05, 7.8))
    utm = {"crs": CRS.from_epsg(32644), "transform": from_origin(500000.0, 862000.0, 10.0, 10.0)}
    west, south, east, north = V._chip_lonlat((p, p > 0, utm)).bounds
    assert 80.9 < west < east < 81.1 and 7.7 < south < north < 7.9  # degrees, not metres


# --- acceptance #14: each broken fixture from the task gives its own, distinguishable error ----

def _break_nan(run_dir):
    with rasterio.open(run_dir / F.FLOOD_PROBABILITY, "r+") as dst:
        p = dst.read(1)
        p[40, 40] = np.nan
        dst.write(p, 1)


def _break_hole(run_dir):
    x, y = TO_EA.transform(80.01, 7.78)
    poly = square_ll(x, y, 1300, hole=100)
    replace_zone(run_dir, poly, V.round_half_up(area_km2(poly), "0.001"))


def _break_09996(run_dir):
    x, y = TO_EA.transform(80.01, 7.78)
    replace_zone(run_dir, square_ll(x, y, 0.9996 ** 0.5 * 1000), 1.0)


def _break_dup_asset(run_dir):
    edit_json(run_dir / F.ASSETS_GEOJSON,
              lambda d: d["features"][1]["properties"].update(asset_id="a01"))


def _break_mask(run_dir):
    with rasterio.open(run_dir / F.FLOOD_MASK, "r+") as dst:
        m = dst.read(1)
        m[50, 50] = 1 - m[50, 50]
        dst.write(m, 1)


def _break_b_unmarked(run_dir):
    edit_csv(run_dir / F.STRATEGY_COMPARISON,
             lambda rows: row(rows, strategy="B").update(budget_feasible="true"))


BROKEN = {
    "nan": (_break_nan, F.FLOOD_PROBABILITY, "NaN/Inf в 1 пикселях"),
    "hole": (_break_hole, F.CANDIDATE_ORDERS, "Z3: есть внутренние вырезы (holes)"),
    "area_09996": (_break_09996, F.CANDIDATE_ORDERS, "< 1 км² (до округления)"),
    "dup_asset_id": (_break_dup_asset, F.ASSETS_GEOJSON, "asset_id a01 повторяется 2 раз"),
    "mask_threshold": (_break_mask, F.FLOOD_MASK, "mask ≠ (p ≥ threshold=0.5) в 1 пикселях"),
    "b_over_budget": (_break_b_unmarked, F.STRATEGY_COMPARISON,
                      "B: budget_feasible=true, а data_cost > бюджета"),
}


@pytest.mark.parametrize("case", sorted(BROKEN))
def test_broken_fixture_gives_its_own_error(run_dir, case):
    breaker, name, signature = BROKEN[case]
    breaker(run_dir)
    errors = failing(run_dir)
    assert signature in " ".join(errors.get(name, [])), errors
    everything = " ".join(" ".join(v) for v in errors.values())
    for other, (_, _, other_sig) in BROKEN.items():
        if other != case:
            assert other_sig not in everything, (case, other)
