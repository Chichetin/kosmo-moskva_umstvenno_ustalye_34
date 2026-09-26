"""plan(run_dir, budget): пересчёт без изменения p-растра, маски и портфеля (D-013)."""
import csv
import hashlib
import json

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point, box, shape

from floodvalue import procurement
from floodvalue.cli import main
from floodvalue.procurement import run as prun

FILES = ["flood_probability.tif", "flood_mask.tif", "assets.csv", "asset_loss.csv",
         "run_metadata.json"]


@pytest.fixture(autouse=True)
def _s1_manifest(tmp_path_factory, monkeypatch):
    """Дата S1 синтетических чипов (decision_at от неё, D-043) — как у Sri-Lanka."""
    data = tmp_path_factory.mktemp("data")
    (data / "interim").mkdir()
    (data / "interim" / "source_manifest_s1.json").write_text(json.dumps({"sources": [
        {"chip_id": c, "observation_date": "2017-05-30"} for c in ("X_1", "Synthetic_000001")]}),
        encoding="utf-8")
    monkeypatch.setenv("FLOODVALUE_DATA_DIR", str(data))


def _write_run_dir(d):
    tr = from_origin(80.0, 7.05, 8.983e-05, 8.983e-05)
    prof = {"driver": "GTiff", "height": 512, "width": 512, "count": 1, "crs": "EPSG:4326",
            "transform": tr}
    p = np.random.default_rng(0).random((512, 512)).astype("float32")
    with rasterio.open(d / "flood_probability.tif", "w", dtype="float32", nodata=-9999, **prof) as f:
        f.write(p, 1)
    with rasterio.open(d / "flood_mask.tif", "w", dtype="uint8", nodata=255, **prof) as f:
        f.write((p >= 0.5).astype("uint8"), 1)
    with open(d / "assets.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["asset_id", "chip_id", "asset_class", "asset_value_rub",
                    "vulnerability_coef", "longitude", "latitude"])
        for i in range(1, 11):
            w.writerow([f"A{i:02d}", "X_1", f"c{i}", 1e6, 0.5, 80.002 + 0.0042 * i,
                        7.048 - 0.0042 * i])
    with open(d / "asset_loss.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["asset_id", "chip_id", "p_flood", "expected_loss_rub", "uncertainty",
                    "rank", "status"])
        for i in range(1, 11):
            w.writerow([f"A{i:02d}", "X_1", i / 10, i * 5e4, 0.5, 11 - i, "ok"])
    (d / "run_metadata.json").write_text(json.dumps({"threshold": 0.5}), encoding="utf-8")


def _hashes(d):
    return {n: hashlib.sha256((d / n).read_bytes()).hexdigest() for n in FILES}


def test_budget_change_keeps_inputs(tmp_path):
    _write_run_dir(tmp_path)
    before = _hashes(tmp_path)
    prun.write_outputs(tmp_path, 5000)
    small = procurement.plan(tmp_path, 1000)
    big = procurement.plan(tmp_path, 100000)
    assert set(small) >= {"plans", "plan_rows", "comparison", "reasons"}
    assert small["plans"]["C"] == [] and len(big["plans"]["C"]) > 1
    b_small = next(r for r in small["comparison"] if r["strategy"] == "B")
    assert b_small["budget_feasible"] is False
    assert _hashes(tmp_path) == before
    cat_hash = hashlib.sha256((tmp_path / "candidate_orders.geojson").read_bytes()).hexdigest()
    prun.write_outputs(tmp_path, 20000)
    assert _hashes(tmp_path) == before
    assert json.loads((tmp_path / "run_metadata.json").read_text())["threshold"] == 0.5
    with open(tmp_path / "procurement_sensitivity.csv", encoding="utf-8") as f:
        sens = list(csv.DictReader(f))
    for prefix in ("budget_", "rate_x"):
        for s in "BC":
            assert sum(r["scenario_id"].startswith(prefix) and r["strategy"] == s
                       for r in sens) >= 5, (prefix, s)
    assert hashlib.sha256((tmp_path / "candidate_orders.geojson").read_bytes()).hexdigest() == cat_hash


def test_plan_cache_same_result_and_invalidation(tmp_path, monkeypatch):
    """Кеш контекста: тот же dict, смена бюджета не пересобирает каталог, правка входа
    сбрасывает кеш (#105). Без порога по времени — считаются сборки контекста."""
    import os
    _write_run_dir(tmp_path)
    cold = {b: prun._plan(prun._context(tmp_path), b) for b in (1000, 20000)}
    builds = []
    real = prun._context
    monkeypatch.setattr(prun, "_context", lambda d: builds.append(d) or real(d))
    prun._CONTEXT_CACHE.clear()
    assert procurement.plan(tmp_path, 20000) == cold[20000]
    assert procurement.plan(tmp_path, 1000) == cold[1000]
    assert procurement.plan(tmp_path, 20000) == cold[20000]
    assert len(builds) == 1
    # same size, newer mtime -> rebuilt
    loss = tmp_path / "asset_loss.csv"
    st = loss.stat()
    os.utime(loss, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    procurement.plan(tmp_path, 1000)
    assert len(builds) == 2
    # changed content -> rebuilt, result follows the new input
    loss.write_text(",no_data".join(loss.read_text(encoding="utf-8").rsplit(",ok", 1)),
                    encoding="utf-8")
    os.utime(loss, ns=(st.st_atime_ns, st.st_mtime_ns + 2 * 10**9))
    assert procurement.plan(tmp_path, 20000) != cold[20000]
    assert len(builds) == 3


def test_cli_procure_writes_contract_files(tmp_path):
    _write_run_dir(tmp_path)
    assert main(["procure", "--budget", "5000", "--out", str(tmp_path)]) == 0
    plans = json.loads((tmp_path / "strategy_plans.json").read_text(encoding="utf-8"))
    with open(tmp_path / "procurement_plan.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for s in "BC":
        assert [r["candidate_id"] for r in rows if r["strategy"] == s] == plans[s]
    assert plans["A"] == []
    with open(tmp_path / "strategy_comparison.csv", encoding="utf-8") as f:
        cmp_ = {r["strategy"]: r for r in csv.DictReader(f)}
    assert float(cmp_["C"]["data_cost_rub"]) <= 5000
    assert cmp_["B"]["budget_feasible"] == "false"
    assert cmp_["B"]["uncertainty_status"] == "scenario"
    assert float(cmp_["B"]["residual_uncertainty"]) == 0.0  # B покрывает все объекты
    assert abs(sum(float(r["cost_rub"]) for r in rows if r["strategy"] == "C")
               - float(cmp_["C"]["data_cost_rub"])) < 1e-9
    with open(tmp_path / "procurement_sensitivity.csv", encoding="utf-8") as f:
        sens = list(csv.DictReader(f))
    assert sum(r["scenario_id"].startswith("budget_") for r in sens) >= 5
    assert {r["scenario_id"] for r in sens if r["scenario_id"].startswith("rate_")} == {
        "rate_x0.5", "rate_x0.75", "rate_x1.25", "rate_x1.5", "rate_x2.0"}
    assert {r["scenario_id"] for r in sens if r["scenario_id"].startswith("deadline_")} == {
        "deadline_+6h", "deadline_+12h", "deadline_+24h", "deadline_+48h"}
    meta = json.loads((tmp_path / "procurement_metadata.json").read_text(encoding="utf-8"))
    assert "W-CHIP-OL" in meta["zones"]["excluded_from_coverage"]
    assert meta["other_cost"]["status"] == "scenario"
    assert "бинарная энтропия" in meta["zones"]["uncertainty_source"]



def test_coverage_denominator_is_exact_kopeck_sum(tmp_path):
    # #127: EL are to the kopeck (Sri-Lanka_713926 values); their float sum is 16842208.290000003
    els = ["2425393.18", "7825102.4", "113413.68", "28478.6", "335218.13", "146091.74",
           "147612.03", "1013717.05", "21718.04", "4785463.44"]
    _write_run_dir(tmp_path)
    with open(tmp_path / "asset_loss.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["asset_id", "chip_id", "p_flood", "expected_loss_rub", "uncertainty",
                    "rank", "status"])
        for i, el in enumerate(els, start=1):
            w.writerow([f"A{i:02d}", "X_1", 0.5, el, 0.5, i, "ok"])
    assert sum(map(float, els)) != 16842208.29  # the float noise this test is about
    assert main(["procure", "--budget", "5000", "--out", str(tmp_path)]) == 0
    meta = json.loads((tmp_path / "procurement_metadata.json").read_text(encoding="utf-8"))
    assert meta["coverage_share_denominator_rub"] == 16842208.29


def test_cli_missing_assets(tmp_path):
    assert main(["procure", "--budget", "5000", "--out", str(tmp_path)]) == 1


def test_stale_catalog_file_is_not_used(tmp_path):
    """Каталог строится из текущих assets: объекты переставлены в обратном порядке —
    зона Z-A10 обязана лежать вокруг нового места A10 (старый файл каталога не читается)."""
    _write_run_dir(tmp_path)
    prun.write_outputs(tmp_path, 5000)
    with open(tmp_path / "assets.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    coords = [(r["longitude"], r["latitude"]) for r in rows][::-1]
    with open(tmp_path / "assets.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r, (lon, lat) in zip(rows, coords, strict=True):
            w.writerow({**r, "longitude": lon, "latitude": lat})
    res = procurement.plan(tmp_path, 2500)
    assert res["plans"]["C"][0] in ("Z-A09", "Z-A10")  # зоны у A10 (max EL) и соседей
    c = next(r for r in res["comparison"] if r["strategy"] == "C")
    assert "A10" in c["covered_asset_ids"].split()


def test_zones_inside_chip_except_asset_buffers(tmp_path):
    _write_run_dir(tmp_path)
    prun.write_outputs(tmp_path, 5000)
    cat = json.loads((tmp_path / "candidate_orders.geojson").read_text(encoding="utf-8"))
    assert "build" not in cat  # служебные метаданные — в procurement_metadata.json
    with rasterio.open(tmp_path / "flood_probability.tif") as src:
        chip = box(*src.bounds).buffer(1e-9)
    kinds = set()
    for f in cat["features"]:
        kinds.add(f["properties"]["zone_kind"])
        if f["properties"]["zone_kind"] != "asset_buffer":
            assert chip.covers(shape(f["geometry"])), f["properties"]["candidate_id"]
    assert kinds == {"asset_buffer", "el_cluster", "uncertainty_cluster", "grid", "wide"}


def test_uncertainty_tif_is_used_when_present(tmp_path):
    """uncertainty.tif (контракт §2) заменяет запасную энтропию без правки кода."""
    _write_run_dir(tmp_path)
    with rasterio.open(tmp_path / "flood_probability.tif") as src:
        prof = src.profile
    u = np.zeros((512, 512), dtype="float32")
    u[300:400, 300:400] = 1.0
    with rasterio.open(tmp_path / "uncertainty.tif", "w", **prof) as f:
        f.write(u, 1)
    prun.write_outputs(tmp_path, 5000)
    meta = json.loads((tmp_path / "procurement_metadata.json").read_text(encoding="utf-8"))
    assert meta["zones"]["uncertainty_source"].startswith("uncertainty.tif")
    cat = json.loads((tmp_path / "candidate_orders.geojson").read_text(encoding="utf-8"))
    (u1,) = {f["properties"]["geometry_id"]: f for f in cat["features"]
             if f["properties"]["zone_kind"] == "uncertainty_cluster"}.values()
    x, y = prof["transform"] @ (350, 350)
    assert shape(u1["geometry"]).covers(Point(x, y))


def test_plan_social_switch(tmp_path):
    """plan(..., social=True) включает ограничение без правки конфигов; в run.yaml — выключено."""
    _write_run_dir(tmp_path)
    with open(tmp_path / "assets.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows[0]["asset_class"] = "clinic"  # A01 — наименьший EL
    with open(tmp_path / "assets.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    off = procurement.plan(tmp_path, 2500)
    on = procurement.plan(tmp_path, 2500, social=True)
    c_on = next(r for r in on["comparison"] if r["strategy"] == "C")
    assert "A01" in c_on["covered_asset_ids"].split()
    assert on["social_constraint"]["satisfied"] is True
    assert off["plans"]["C"] != on["plans"]["C"]
    prun.write_outputs(tmp_path, 2500)
    with open(tmp_path / "procurement_sensitivity.csv", encoding="utf-8") as f:
        sens = [r for r in csv.DictReader(f) if r["scenario_id"] == "social_constraint_on"]
    assert len(sens) == 1 and "A01" in sens[0]["changed_inputs_json"]


def test_procurement_files_pass_validator_on_fixture(tmp_path):
    """Эталонный комплект с перестроенными procure файлами: 4 файла закупки без ошибок
    валидатора (цена, скидка по корзине, охват covers, бюджет, статусы неопределённости)."""
    import shutil

    from floodvalue.export.validate import FIXTURE_RUN, validate_run
    run = tmp_path / "run"
    shutil.copytree(FIXTURE_RUN, run)
    meta = json.loads((run / "run_metadata.json").read_text(encoding="utf-8"))
    budget = meta["budget"]["budget_rub"]
    prun.write_outputs(run, budget)
    # export takes decision_at/decision_deadline from procurement_metadata.json (D-043)
    proc = json.loads((run / "procurement_metadata.json").read_text(encoding="utf-8"))
    meta["budget"].update({k: proc[k] for k in ("decision_at", "decision_deadline")})
    (run / "run_metadata.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    errors = validate_run(run)
    for name in ("candidate_orders.geojson", "procurement_plan.csv", "strategy_plans.json",
                 "strategy_comparison.csv"):
        assert errors[name] == [], (name, errors[name])


def test_procurement_sensitivity_passes_validator(tmp_path):
    """sensitivity.csv = строки procure (бюджет, ставка, срок) ∪ строки p, V, q эталона:
    валидатор принимает ключ ставки и бюджета как «изменён один вход»."""
    import shutil

    from floodvalue.export import assemble
    from floodvalue.export.validate import FIXTURE_RUN, validate_run
    run = tmp_path / "run"
    shutil.copytree(FIXTURE_RUN, run)
    meta = json.loads((run / "run_metadata.json").read_text(encoding="utf-8"))
    prun.write_outputs(run, meta["budget"]["budget_rub"])
    with open(run / "procurement_sensitivity.csv", encoding="utf-8") as f:
        proc = list(csv.DictReader(f))
    with open(run / "sensitivity.csv", encoding="utf-8") as f:
        pvq = [r for r in csv.DictReader(f) if set(json.loads(r["changed_inputs_json"]))
               & {"p_flood", "asset_value_rub", "vulnerability_coef"}]
    rows = assemble.merge_sensitivity(proc, pvq)
    with open(run / "sensitivity.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, assemble.sensitivity_fields(rows), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    assert validate_run(run)["sensitivity.csv"] == []
    assert assemble.missing_sensitivity_keys(proc) == [
        "asset_value_rub", "p_flood", "vulnerability_coef"]


def test_coverage_at_base_deadline_matches_context(tmp_path):
    """Пересчёт пригодности по сроку без перестройки каталога совпадает с _context."""
    from floodvalue.procurement import strategies
    _write_run_dir(tmp_path)
    ctx = prun._context(tmp_path)
    ok = {r["asset_id"] for r in ctx["loss"] if r["status"] == "ok"}
    cov, elig = strategies.coverage_at(ctx["catalog"], ok, ctx["deadline"])
    assert cov == ctx["cov"] and elig == ctx["eligible"]
