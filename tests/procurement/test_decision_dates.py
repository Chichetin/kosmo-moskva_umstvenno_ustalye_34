"""decision_at/decision_deadline от даты S1 чипа (#84, D-001, D-023, D-043).

decision_at = дата S1 чипа (source_manifest_s1.json ← s1_date Sen1Floods11_Metadata.geojson)
+ сценарное время суток из configs/run.yaml; decision_deadline = decision_at + окно run.yaml
(12 ч); as_of = календарная дата decision_at. Метки не читаются — только события и даты.
"""
import csv
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from floodvalue.contracts import load_run_config
from floodvalue.procurement import run as prun
from floodvalue.procurement import zones

from test_plan import _write_run_dir

REPO = Path(__file__).resolve().parents[2]
METADATA = REPO / "docs" / "sen1floods11" / "Sen1Floods11_Metadata.geojson"


def _metadata_s1_date(location: str) -> date:
    """Только location → s1_date (YYYY/MM/DD) из метаданных событий набора в git."""
    d = json.loads(METADATA.read_text(encoding="utf-8"))
    (s1,) = [f["properties"]["s1_date"] for f in d["features"]
             if f["properties"]["location"] == location]
    return date(*map(int, s1.split("/")))


def _write_manifest(data_dir: Path, dates: dict[str, str]) -> None:
    (data_dir / "interim").mkdir(parents=True, exist_ok=True)
    sources = [{"source_id": f"sen1floods11-v1.1:S1Hand:{c}", "chip_id": c,
                "event_id": c.rsplit("_", 1)[0], "observation_date": d, "event_date": d}
               for c, d in dates.items()]
    (data_dir / "interim" / "source_manifest_s1.json").write_text(
        json.dumps({"sources": sources}), encoding="utf-8")


def _set_chip(run_dir: Path, chip: str) -> None:
    for name in ("assets.csv", "asset_loss.csv"):
        with open(run_dir / name, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        with open(run_dir / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows({**r, "chip_id": chip} for r in rows)


UTC = timezone.utc
TEMPLATE_AT = datetime(2017, 5, 30, 23, 59, 59, tzinfo=UTC)
TEMPLATE_DEADLINE = datetime(2017, 5, 31, 11, 59, 59, tzinfo=UTC)


@pytest.mark.parametrize("location,chip", [("Sri-Lanka", "Sri-Lanka_713926"),
                                           ("Spain", "Spain_7370579")])
def test_decision_at_not_before_s1_date_of_chip_event(location, chip):
    s1 = _metadata_s1_date(location)
    at, deadline = prun.decision_times(TEMPLATE_AT, TEMPLATE_DEADLINE, s1)
    assert at.date() == s1 and at >= datetime.combine(s1, datetime.min.time(), UTC)
    assert at.timetz() == TEMPLATE_AT.timetz()  # сценарное время суток из run.yaml
    assert deadline - at == timedelta(hours=12)  # D-001


def test_control_dates_of_two_events():
    assert _metadata_s1_date("Sri-Lanka") == date(2017, 5, 30)
    assert _metadata_s1_date("Spain") == date(2019, 9, 17)
    at, deadline = prun.decision_times(TEMPLATE_AT, TEMPLATE_DEADLINE, date(2019, 9, 17))
    assert at == datetime(2019, 9, 17, 23, 59, 59, tzinfo=UTC)
    assert deadline == datetime(2019, 9, 18, 11, 59, 59, tzinfo=UTC)


def test_run_yaml_template_matches_s1_date_of_its_own_chip():
    rc = load_run_config()
    assert rc.decision_at.date() == _metadata_s1_date(rc.event_id)
    assert rc.decision_deadline - rc.decision_at == timedelta(hours=12)


@pytest.mark.parametrize("location", ["Sri-Lanka", "Spain"])
def test_run_yaml_decision_not_before_any_s1_pass_of_that_day(location):
    """Метаданные знают только дату S1 (пролёт ≈ 06/18 ч местного солнечного времени:
    Sri-Lanka ≈ 12:40 UTC, Spain ≈ 18:25 UTC) — решение не раньше конца суток S1 по UTC."""
    rc = load_run_config()
    s1 = _metadata_s1_date(location)
    at, _ = prun.decision_times(rc.decision_at, rc.decision_deadline, s1)
    assert at.date() == s1  # as_of = дата S1 (D-023)
    assert at >= datetime.combine(s1 + timedelta(days=1), datetime.min.time(), UTC) - \
        timedelta(seconds=1)


def test_chip_s1_date_from_manifest(tmp_path, monkeypatch):
    _write_manifest(tmp_path, {"Spain_7370579": "2019-09-17", "Sri-Lanka_713926": "2017-05-30"})
    monkeypatch.setenv("FLOODVALUE_DATA_DIR", str(tmp_path))
    assert prun.chip_s1_date("Spain_7370579") == date(2019, 9, 17)
    with pytest.raises(ValueError, match="make data"):
        prun.chip_s1_date("Nowhere_1")


def test_chip_s1_date_without_manifest(tmp_path, monkeypatch):
    monkeypatch.setenv("FLOODVALUE_DATA_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="make data"):
        prun.chip_s1_date("Spain_7370579")


def _run(tmp_path, monkeypatch, chip, s1):
    run_dir, data = tmp_path / chip, tmp_path / "data"
    run_dir.mkdir()
    _write_run_dir(run_dir)
    _set_chip(run_dir, chip)
    _write_manifest(data, {chip: s1})
    monkeypatch.setenv("FLOODVALUE_DATA_DIR", str(data))
    res = prun.write_outputs(run_dir, 5000)
    meta = json.loads((run_dir / "procurement_metadata.json").read_text(encoding="utf-8"))
    cat = json.loads((run_dir / "candidate_orders.geojson").read_text(encoding="utf-8"))
    return res, meta, cat


def test_other_event_chip_orders_time_after_its_s1(tmp_path, monkeypatch):
    """Чип другого события (Spain, 2019-09-17) при run.yaml с Sri-Lanka: решение не раньше S1."""
    _, meta, cat = _run(tmp_path, monkeypatch, "Spain_7370579", "2019-09-17")
    at = zones.parse_dt(meta["decision_at"])
    deadline = zones.parse_dt(meta["decision_deadline"])
    assert at.date() >= date(2019, 9, 17)  # run_metadata.budget.decision_at ≥ event_date
    assert deadline - at == timedelta(hours=12)
    assert meta["pricing_as_of"] == "2019-09-17"  # D-023: календарная дата decision_at
    assert meta["s1_observation_date"] == "2019-09-17"
    assert "не историческая цена" in meta["pricing_as_of_rule"]  # крит. 18: редакция 2025
    for f in cat["features"]:
        p = f["properties"]
        if p["acquisition_type"] == "new":
            assert zones.parse_dt(p["observation_at"]) >= at, p["candidate_id"]
        if p["observation_at"]:
            assert zones.parse_dt(p["observation_at"]).year in (2019, 2018), p["candidate_id"]


def test_coverage_by_deadline_is_the_same_for_both_events(tmp_path, monkeypatch):
    """Сдвиг дат целиком: охват по available_at ≤ decision_deadline и планы не меняются."""
    res_sl, meta_sl, _ = _run(tmp_path, monkeypatch, "Sri-Lanka_713926", "2017-05-30")
    res_es, meta_es, _ = _run(tmp_path, monkeypatch, "Spain_7370579", "2019-09-17")
    assert meta_sl["decision_at"] == "2017-05-30T23:59:59+00:00"  # run.yaml, D-043
    assert meta_sl["decision_deadline"] == "2017-05-31T11:59:59+00:00"
    assert res_sl["plans"] == res_es["plans"] and res_es["plans"]["C"]
    assert meta_sl["zones"]["coverage"] == meta_es["zones"]["coverage"]
    assert meta_sl["zones"]["excluded_from_coverage"].keys() == \
        meta_es["zones"]["excluded_from_coverage"].keys()
    assert [r["data_cost_rub"] for r in res_sl["comparison"]] == \
        [r["data_cost_rub"] for r in res_es["comparison"]]


def test_stale_run_yaml_date_for_its_own_chip_fails(tmp_path, monkeypatch):
    """Дата decision_at в run.yaml ≠ дате S1 чипа run.yaml — ошибка, а не тихая подмена."""
    rc = load_run_config()
    with pytest.raises(ValueError, match="run.yaml"):
        _run(tmp_path, monkeypatch, rc.chip_id, "2017-06-02")
