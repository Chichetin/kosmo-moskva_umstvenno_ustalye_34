"""Export under the budget chosen in the interface and the budget slider's saturation (#174):
procure → export → validate rerun on a copy; the source bundle, p and the mask never change."""

import hashlib
import io
import json
import subprocess
import zipfile

import pytest
from fastapi import HTTPException

from floodvalue.app import run
from floodvalue.paths import REPO_ROOT

FIXTURE = REPO_ROOT / "tests" / "fixtures" / "good"


def _hashes(path):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in path.iterdir() if p.is_file()}


@pytest.fixture
def bundle(monkeypatch):
    monkeypatch.setenv("FLOODVALUE_APP_RUN_DIR", str(FIXTURE))
    return json.loads((FIXTURE / "run_metadata.json").read_text())


def _fake_steps(monkeypatch, fail_at=None):
    calls, workdirs = [], []

    def fake(command, **kwargs):
        calls.append(command)
        work = run.Path(command[command.index("--out") + 1])
        workdirs.append(work)
        if command[3] == fail_at:
            return subprocess.CompletedProcess(command, 1, "", f"boom in {work}")
        if command[3] == "export":
            budget = float(command[command.index("--budget") + 1])
            metadata = json.loads((work / "run_metadata.json").read_text())
            metadata["budget"]["budget_rub"] = budget
            metadata["run_id"] = f"X_{budget:.0f}"
            (work / "run_metadata.json").write_text(json.dumps(metadata))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(run.subprocess, "run", fake)
    return calls, workdirs


def test_same_budget_returns_the_bundle_as_is(bundle, monkeypatch):
    calls, _ = _fake_steps(monkeypatch)
    response = run.download(None, bundle["budget"]["budget_rub"])
    assert FIXTURE.name in response.headers["content-disposition"] and calls == []
    assert run.download(None, None).headers == response.headers


def test_other_budget_reruns_procure_export_validate_on_a_copy(bundle, monkeypatch):
    before = _hashes(FIXTURE)
    calls, workdirs = _fake_steps(monkeypatch)
    response = run.download(None, 2500.0)
    assert [c[3] for c in calls] == ["procure", "export", "validate"]
    assert all(c[:3] == [run.sys.executable, "-m", "floodvalue"] for c in calls)
    assert calls[0][calls[0].index("--budget") + 1] == "2500.00"
    assert "--budget" not in calls[2]
    with zipfile.ZipFile(io.BytesIO(response.body)) as archive:
        assert json.loads(archive.read("run_metadata.json"))["budget"]["budget_rub"] == 2500.0
        assert hashlib.sha256(archive.read("flood_probability.tif")).hexdigest() == \
            before["flood_probability.tif"]
    assert 'filename="X_2500.zip"' in response.headers["content-disposition"]
    assert _hashes(FIXTURE) == before  # the source bundle is untouched
    assert not workdirs[0].exists()  # the copy is removed
    assert not run._rebudget_lock.locked()


def test_failed_step_is_422_without_server_paths_and_frees_the_lock(bundle, monkeypatch):
    _, workdirs = _fake_steps(monkeypatch, fail_at="procure")
    with pytest.raises(HTTPException) as err:
        run.download(None, 2500.0)
    assert err.value.status_code == 422 and "procure" in err.value.detail
    assert str(workdirs[0]) not in err.value.detail
    assert not run._rebudget_lock.locked()


def test_one_rebudgeted_export_at_a_time(bundle):
    assert run._rebudget_lock.acquire(blocking=False)
    try:
        with pytest.raises(HTTPException) as err:
            run.download(None, 2500.0)
        assert err.value.status_code == 429
    finally:
        run._rebudget_lock.release()


def test_saturation_is_where_plan_c_stops_changing(tmp_path, monkeypatch, bundle):
    interim = tmp_path / "interim"
    interim.mkdir()
    (interim / "source_manifest_s1.json").write_text(json.dumps({"sources": [{
        "chip_id": bundle["chip_id"], "observation_date": "2017-05-30"}]}))
    monkeypatch.setenv("FLOODVALUE_DATA_DIR", str(tmp_path))
    saturation = run.session()["plan_c_saturation_rub"]
    assert saturation is not None and saturation > 0
    unlimited = run.plan(FIXTURE, run.UNLIMITED_BUDGET_RUB)["plans"]["C"]
    assert run.plan(FIXTURE, saturation)["plans"]["C"] == unlimited
    assert run.plan(FIXTURE, saturation * 2)["plans"]["C"] == unlimited


def test_lock_is_freed_when_the_temp_dir_cannot_be_made(bundle, monkeypatch):
    def no_space(**kwargs):
        raise OSError(28, "No space left on device")
    monkeypatch.setattr(run.tempfile, "mkdtemp", no_space)
    with pytest.raises(HTTPException) as err:
        run.download(None, 2500.0)
    assert err.value.status_code == 503 and not run._rebudget_lock.locked()


def test_budget_argument_is_rounded_half_up(bundle, monkeypatch):
    calls, _ = _fake_steps(monkeypatch)
    run.download(None, 2500.005)
    assert calls[0][calls[0].index("--budget") + 1] == "2500.01"


def test_saturation_is_none_when_the_plan_cannot_be_computed(monkeypatch):
    def broken(run_dir, budget):
        raise ValueError("no manifest")
    monkeypatch.setattr(run, "plan", broken)
    assert run._plan_c_saturation(FIXTURE) is None
