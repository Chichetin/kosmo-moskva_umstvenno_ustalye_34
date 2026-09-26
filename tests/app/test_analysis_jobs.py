"""Analysis from the interface (#174): the same stages as `make all`, one job at a time, test
chips sealed, an existing analysis is not recomputed, a failing stage is reported, not hidden."""

import json
import subprocess

import pytest
from fastapi import HTTPException

from floodvalue.app import run


@pytest.fixture
def catalog(tmp_path, monkeypatch):
    """One raw observation in the catalog, empty outputs, no worker thread."""
    monkeypatch.delenv("FLOODVALUE_APP_RUN_DIR", raising=False)
    monkeypatch.setattr(run, "output_dir", lambda: tmp_path / "outputs")
    monkeypatch.setattr(run, "data_dir", lambda: tmp_path)
    monkeypatch.setenv("FLOODVALUE_DATA_DIR", str(tmp_path))
    raw = tmp_path / "raw" / "HandLabeled" / "S1Hand"
    raw.mkdir(parents=True)
    (tmp_path / "interim").mkdir()
    manifest = tmp_path / "interim" / "source_manifest_s1.json"
    manifest.write_text(json.dumps({"sources": [
        {"chip_id": "Synthetic_000001", "file": "Synthetic_000001_S1Hand.tif",
         "observation_date": "2017-05-30"},
        {"chip_id": "Spain_7370579", "file": "Spain_7370579_S1Hand.tif",
         "observation_date": "2019-09-17"}]}))
    for name in ("Synthetic_000001_S1Hand.tif", "Spain_7370579_S1Hand.tif"):
        (raw / name).write_bytes(b"raw")
    monkeypatch.setattr(run, "_jobs", {})
    monkeypatch.setattr(run, "_job_queue", run.queue.Queue())
    monkeypatch.setattr(run, "_ensure_worker", lambda: None)
    return tmp_path


def _fake_stages(monkeypatch, fail_at: str | None = None):
    calls = []

    def fake(command, **kwargs):
        calls.append(command)
        out = command[command.index("--out") + 1]
        if fail_at and fail_at in command:
            return subprocess.CompletedProcess(command, 1, "", "Traceback: boom")
        if "export" in command:
            (run.Path(out)).mkdir(parents=True, exist_ok=True)
            (run.Path(out) / "run_metadata.json").write_text(json.dumps({"run_id": "R-1"}))
        return subprocess.CompletedProcess(command, 0, "ok", "")

    monkeypatch.setattr(run.subprocess, "run", fake)
    return calls


def test_job_runs_the_make_all_stages_in_order(catalog, monkeypatch):
    calls = _fake_stages(monkeypatch)
    queued = json.loads(run.start_analysis("Synthetic_000001").body)
    assert queued["status"] == "queued" and "out" not in queued
    run._run_job(queued["job_id"])
    status = run.analysis_status(queued["job_id"])
    assert status["status"] == "done" and status["run_id"] == "R-1"
    assert status["step"] == 6 == status["steps"]
    modules = [c[2] + (" " + c[3] if c[2] == "floodvalue" else "") for c in calls]
    assert modules == ["floodvalue predict", "floodvalue.baseline.predict", "floodvalue assets",
                       "floodvalue procure", "floodvalue export", "floodvalue validate"]
    assert all(c[0] == run.sys.executable for c in calls)
    assert len({c[c.index("--out") + 1] for c in calls}) == 1  # one bundle directory


def test_failed_stage_is_reported_and_stops_the_job(catalog, monkeypatch):
    calls = _fake_stages(monkeypatch, fail_at="procure")
    job_id = json.loads(run.start_analysis("Synthetic_000001").body)["job_id"]
    run._run_job(job_id)
    status = run.analysis_status(job_id)
    assert status["status"] == "failed" and status["step"] == 4
    assert "Зоны и план съёмки" in status["error"] and "boom" in status["detail"]
    assert not any("export" in c for c in calls)


def test_test_chips_unknown_chips_and_single_bundle_mode_are_refused(catalog, monkeypatch):
    monkeypatch.delenv("FLOODVALUE_APP_ALLOW_TEST", raising=False)
    for chip, code in (("Spain_7370579", 403), ("Nowhere_1", 404)):
        with pytest.raises(HTTPException) as err:
            run.start_analysis(chip)
        assert err.value.status_code == code
    monkeypatch.setenv("FLOODVALUE_APP_RUN_DIR", str(catalog))
    with pytest.raises(HTTPException) as err:
        run.start_analysis("Synthetic_000001")
    assert err.value.status_code == 409


def test_same_chip_is_not_queued_twice_and_queue_is_bounded(catalog):
    first = json.loads(run.start_analysis("Synthetic_000001").body)
    again = json.loads(run.start_analysis("Synthetic_000001").body)
    assert again["job_id"] == first["job_id"] and len(run._jobs) == 1
    run._jobs.clear()
    for index in range(run.ANALYSIS_QUEUE_MAX):  # other chips fill the queue
        run._jobs[f"busy{index}"] = {**first, "job_id": f"busy{index}", "chip_id": f"X_{index}"}
    with pytest.raises(HTTPException) as err:
        run.start_analysis("Synthetic_000001")
    assert err.value.status_code == 429


def test_existing_analysis_is_returned_without_a_new_job(catalog, monkeypatch):
    fixture = run.REPO_ROOT / "tests" / "fixtures" / "good"
    monkeypatch.setattr(run, "_available_runs", lambda: [fixture])
    chip = json.loads((fixture / "run_metadata.json").read_text())["chip_id"]
    monkeypatch.setattr(run, "_observation_file", lambda chip_id: fixture)
    result = run.start_analysis(chip)
    assert result["status"] == "done" and result["job_id"] is None and run._jobs == {}


def test_unknown_job_is_404():
    with pytest.raises(HTTPException) as err:
        run.analysis_status("nope")
    assert err.value.status_code == 404


def test_test_chip_is_refused_even_with_allow_test(catalog, monkeypatch):
    monkeypatch.setenv("FLOODVALUE_APP_ALLOW_TEST", "1")
    with pytest.raises(HTTPException) as err:
        run.start_analysis("Spain_7370579")
    assert err.value.status_code == 403


def test_unexpected_error_fails_the_job_and_hides_server_paths(catalog, monkeypatch):
    def boom(command, **kwargs):
        raise RuntimeError(f"cannot read {run.output_dir()}/x and {run.REPO_ROOT}/y")
    monkeypatch.setattr(run.subprocess, "run", boom)
    job_id = json.loads(run.start_analysis("Synthetic_000001").body)["job_id"]
    run._run_job(job_id)  # RuntimeError is not OSError: the stage loop does not catch it
    status = run.analysis_status(job_id)
    assert status["status"] == "failed"
    assert str(run.REPO_ROOT) not in status["detail"] and "<outputs>" in status["detail"]


def test_finished_jobs_are_bounded(catalog):
    for index in range(run.ANALYSIS_JOBS_KEPT + 5):
        run._jobs[f"old{index}"] = {"job_id": f"old{index}", "chip_id": f"X_{index}",
                                    "status": "done", "finished_at": index}
    run.start_analysis("Synthetic_000001")
    finished = [job for job in run._jobs.values() if job["status"] == "done"]
    assert len(finished) == run.ANALYSIS_JOBS_KEPT
    assert "old0" not in run._jobs and f"old{run.ANALYSIS_JOBS_KEPT + 4}" in run._jobs
