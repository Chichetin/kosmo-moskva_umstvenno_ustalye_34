"""Контракт §10 docs/contracts.md: сигнатура plan() и ключи результата (D-013, D-038)."""
import inspect
import shutil

import pytest

from floodvalue import procurement
from floodvalue.export.validate import FIXTURE_RUN
from floodvalue.procurement import run as prun

TOP_KEYS = {"plans", "plan_rows", "comparison", "reasons", "social_constraint", "c_selection"}


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("plan") / "run"
    shutil.copytree(FIXTURE_RUN, d)
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
