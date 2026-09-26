"""Контракт §10 docs/contracts.md: сигнатура plan() и ключи результата (D-013, D-038)."""
import inspect
import json
import shutil

import pytest

from floodvalue import procurement
from floodvalue.export.validate import FIXTURE_RUN
from floodvalue.procurement import run as prun

TOP_KEYS = {"plans", "plan_rows", "comparison", "reasons", "social_constraint", "c_selection"}


@pytest.fixture
def run_dir(tmp_path, monkeypatch):
    """Эталонный комплект + манифест S1 с датой его чипа (вход plan(), D-043)."""
    d = tmp_path / "run"
    shutil.copytree(FIXTURE_RUN, d)
    data = tmp_path / "data"
    (data / "interim").mkdir(parents=True)
    (data / "interim" / "source_manifest_s1.json").write_text(json.dumps({"sources": [
        {"chip_id": "Synthetic_000001", "observation_date": "2017-05-30"}]}), encoding="utf-8")
    monkeypatch.setenv("FLOODVALUE_DATA_DIR", str(data))
    return d


def test_signature_social_is_optional():
    for fn in (procurement.plan, prun.plan):
        params = inspect.signature(fn).parameters
        assert list(params) == ["run_dir", "budget_rub", "social"]
        assert params["social"].default is None


def test_top_level_keys_and_shapes(run_dir):
    res = procurement.plan(run_dir, 5000)
    assert set(res) == TOP_KEYS
    assert set(res["plans"]) == {"A", "B", "C"} and res["plans"]["A"] == []
    assert [r["strategy"] for r in res["comparison"]] == ["A", "B", "C"]
    assert all("reason" not in r for r in res["plan_rows"])  # reason adds procure
    assert {"B", "C", "not_selected_C", "chosen_any"} <= set(res["reasons"])
    assert set(res["reasons"]) - {"B", "C", "not_selected_C", "chosen_any"} <= {"B_empty"}
    assert set(res["c_selection"]) == {"variant", "alternatives"}
    assert res["c_selection"]["variant"] in res["c_selection"]["alternatives"]


def test_social_default_follows_config_and_switch(run_dir):
    """None = configs/run.yaml (enabled: false) = False; True — ограничение с итогом."""
    default = procurement.plan(run_dir, 5000)
    off = procurement.plan(run_dir, 5000, social=False)
    on = procurement.plan(run_dir, 5000, social=True)
    enabled = prun.run_config()["social_constraint"]["enabled"]
    assert default == (on if enabled else off)
    assert set(off["social_constraint"]) == {"required_asset_ids"}
    assert off["social_constraint"]["required_asset_ids"] == []
    assert set(on["social_constraint"]) == {"required_asset_ids", "satisfied",
                                            "uncovered_required", "note"}
    assert on["social_constraint"]["required_asset_ids"]
    assert set(on) == TOP_KEYS


def test_missing_s1_manifest_raises(run_dir, tmp_path, monkeypatch):
    """Без манифеста S1 или даты чипа plan() бросает исключение, а не строит план (§10)."""
    monkeypatch.setenv("FLOODVALUE_DATA_DIR", str(tmp_path / "empty"))
    with pytest.raises(FileNotFoundError):
        procurement.plan(run_dir, 5000)
    other = tmp_path / "other"
    (other / "interim").mkdir(parents=True)
    (other / "interim" / "source_manifest_s1.json").write_text('{"sources": []}', encoding="utf-8")
    monkeypatch.setenv("FLOODVALUE_DATA_DIR", str(other))
    with pytest.raises(ValueError):
        procurement.plan(run_dir, 5000)


def test_csv_writer_leaves_non_finite_numbers_empty(tmp_path):
    """Contract: empty cell instead of NaN/Inf, not "nan" (validate rejects it) — #235."""
    from floodvalue.contracts.io import read_csv
    from floodvalue.procurement import run

    path = tmp_path / "t.csv"
    run._write_csv(path, ["a", "b", "c", "d"],
                   [{"a": float("nan"), "b": float("inf"), "c": -float("inf"), "d": 1.5}])
    assert path.read_text(encoding="utf-8") == "a,b,c,d\n,,,1.5\n"
    assert read_csv(path)[2] == []  # the strict contract reader finds no errors
