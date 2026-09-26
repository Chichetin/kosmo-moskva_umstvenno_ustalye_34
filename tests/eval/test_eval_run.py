"""`eval-test` end to end on tiny synthetic chips, and the seal on test (D-010, #29).

Chips are written to a temporary FLOODVALUE_DATA_DIR; the main method is the shipped U-Net
(weights through LFS), the portfolio is the real `assets` stage. Real test chips are never
touched: the "test" chip here is synthetic and its chip list is injected.
"""

import csv
import json
from argparse import Namespace

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from floodvalue import cli
from floodvalue.data.split import SealedTestError
from floodvalue.model import eval_test as E
from floodvalue.model.eval import core as C

H = W = 48


def write_chip(root, chip, seed):
    rng = np.random.default_rng(seed)
    water = np.zeros((H, W), bool)
    water[:, : W // 3] = True                              # a river on the left
    vh = np.where(water, -28.0, -15.0) + rng.normal(0, 1.0, (H, W))
    vv = np.where(water, -20.0, -8.0) + rng.normal(0, 1.0, (H, W))
    s1 = np.stack([vv, vh]).astype(np.float32)
    s1[:, :3, -3:] = np.nan                                # S1 nodata corner
    label = water.astype(np.int16)
    label[-4:, :] = -1                                     # unlabelled strip
    label[:3, -3:] = 1                                     # label under S1 NaN (a trap)
    jrc = np.zeros((H, W), np.uint8)
    jrc[:, : W // 6] = 1
    t = from_origin(80.0, 7.0, 8.983e-05, 8.983e-05)
    for layer, arr, dtype in (("S1Hand", s1, "float32"), ("LabelHand", label[None], "int16"),
                              ("JRCWaterHand", jrc[None], "uint8")):
        d = root / "raw" / "HandLabeled" / layer
        d.mkdir(parents=True, exist_ok=True)
        with rasterio.open(d / f"{chip}_{layer}.tif", "w", driver="GTiff", width=W, height=H,
                           count=arr.shape[0], dtype=dtype, crs="EPSG:4326", transform=t) as dst:
            dst.write(arr.astype(dtype))


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    root = tmp_path / "data"
    for i, chip in enumerate(["Sri-Lanka_1", "Sri-Lanka_2", "Nigeria_3", "Spain_9"]):
        write_chip(root, chip, i)
    monkeypatch.setenv("FLOODVALUE_DATA_DIR", str(root))
    monkeypatch.delenv("FLOODVALUE_METHOD", raising=False)
    monkeypatch.delenv(E.CONFIRM_ENV, raising=False)
    return root


@pytest.fixture(autouse=True)
def test_log(tmp_path, monkeypatch):
    """Every test writes to a temporary log, never to docs/verification/test-log.md."""
    monkeypatch.delenv(E.CONFIRM_ENV, raising=False)
    log = tmp_path / "test-log.md"
    log.write_text("# Журнал\n\n| Дата | Commit | sha256 | Метод(ы) | Кто | Результат |\n"
                   "|---|---|---|---|---|---|\n", encoding="utf-8")
    monkeypatch.setattr(E, "TEST_LOG", log)
    return log


def forbid_test_reads(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("test data read before the gate")
    monkeypatch.setattr(E, "chip_ids", boom)
    monkeypatch.setattr(E, "evaluate", boom)
    monkeypatch.setattr(E.io, "load_label", boom)


def test_cli_part_defaults_to_test():
    assert cli.build_parser().parse_args(["eval-test"]).part == "test"
    assert cli.build_parser().parse_args(["eval-test", "--part", "validation"]).part == \
        "validation"
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["eval-test", "--part", "train"])


def test_part_test_without_confirmation_is_refused(monkeypatch, capsys, tmp_path, test_log):
    forbid_test_reads(monkeypatch)
    before = test_log.read_text(encoding="utf-8")
    rc = cli.main(["eval-test", "--out", str(tmp_path / "o")])       # default part = test
    out = capsys.readouterr().out
    assert rc == 3 and "отказ" in out and E.frozen_config_hash() in out
    assert test_log.read_text(encoding="utf-8") == before            # nothing logged
    assert not (tmp_path / "o").exists()


def test_part_test_with_wrong_hash_is_refused(monkeypatch, capsys, tmp_path, test_log):
    forbid_test_reads(monkeypatch)
    monkeypatch.setenv(E.CONFIRM_ENV, "0" * 64)
    assert cli.main(["eval-test", "--part", "test", "--out", str(tmp_path)]) == 3
    assert "does not match" in capsys.readouterr().out


def test_part_test_with_uncommitted_config_is_refused(monkeypatch, capsys, tmp_path, test_log):
    forbid_test_reads(monkeypatch)
    monkeypatch.setenv(E.CONFIRM_ENV, E.frozen_config_hash())
    monkeypatch.setattr(E, "uncommitted_frozen_paths", lambda: " M models/unet/config.json")
    assert cli.main(["eval-test", "--part", "test", "--out", str(tmp_path)]) == 3
    assert "uncommitted" in capsys.readouterr().out


@pytest.mark.parametrize("confirm", [False, True])
def test_part_test_for_non_final_method_is_refused_before_hash(monkeypatch, capsys, tmp_path,
                                                                test_log, confirm):
    """D-025: test opens only for the final method; a candidate is refused before any test
    read and before the configuration hash is even computed (#111)."""
    if confirm:
        monkeypatch.setenv(E.CONFIRM_ENV, E.frozen_config_hash(method="pixel"))
    forbid_test_reads(monkeypatch)

    def no_hash(*a, **k):
        raise AssertionError("config hash requested for a non-final method")
    monkeypatch.setattr(E, "frozen_config_hash", no_hash)
    monkeypatch.setattr(E, "uncommitted_frozen_paths", lambda *a, **k: "")
    monkeypatch.setenv("FLOODVALUE_METHOD", "pixel")
    before = test_log.read_text(encoding="utf-8")
    rc = cli.main(["eval-test", "--part", "test", "--out", str(tmp_path / "o")])
    out = capsys.readouterr().out
    assert rc == 3 and "отказ" in out and "D-025" in out and "pixel" in out
    assert test_log.read_text(encoding="utf-8") == before
    assert not (tmp_path / "o").exists()


def test_part_test_gate_accepts_final_method_named_explicitly(monkeypatch):
    monkeypatch.setenv("FLOODVALUE_METHOD", E.main_predict.DEFAULT_METHOD)
    monkeypatch.setattr(E, "uncommitted_frozen_paths", lambda *a, **k: "")
    monkeypatch.setenv(E.CONFIRM_ENV, E.frozen_config_hash())
    assert E.open_test_gate()[0] == E.frozen_config_hash()


def test_non_test_part_keeps_candidate_method(data_root, tmp_path, monkeypatch):
    """dev/validation still evaluate a candidate chosen by FLOODVALUE_METHOD."""
    monkeypatch.setenv("FLOODVALUE_METHOD", "pixel")
    seen = {}

    def fake_eval(part, chips, workdir, **k):
        seen["part"] = part
        raise RuntimeError("stop after the gate")
    monkeypatch.setattr(E, "chip_ids", lambda part, **k: ["Sri-Lanka_2"])
    monkeypatch.setattr(E, "evaluate", fake_eval)
    with pytest.raises(RuntimeError, match="stop after the gate"):
        cli.main(["eval-test", "--part", "validation", "--out", str(tmp_path / "o")])
    assert seen["part"] == "validation"


def test_labels_of_test_events_stay_sealed_without_flag(data_root):
    with pytest.raises(SealedTestError):
        E.io.load_label("Spain_9")


def test_validation_run_writes_report_on_identical_pixels(data_root, tmp_path, monkeypatch):
    chips = ["Sri-Lanka_1", "Sri-Lanka_2", "Nigeria_3"]
    monkeypatch.setattr(E, "chip_ids", lambda part, **k: chips if part == "validation" else 1 / 0)
    monkeypatch.setitem(E.PROXY_CHIPS, "validation", "Sri-Lanka_2")
    out = tmp_path / "report"
    assert cli.main(["eval-test", "--part", "validation", "--out", str(out)]) == 0
    names = sorted(p.name for p in out.iterdir())
    assert names == sorted(f"eval-validation{s}" for s in (
        ".md", ".csv", "-chips.csv", "-reliability.csv", "-proxy.csv", "-summary.json",
        "-errors.png", "-qc.csv"))
    with open(out / "eval-validation-qc.csv", newline="", encoding="utf-8") as f:
        qc = list(csv.DictReader(f))
    assert list(qc[0]) == C.QC_COLUMNS and [r["chip_id"] for r in qc] == sorted(chips)
    for r in qc:                                            # write_chip: 3×3 NaN, 4 rows of −1
        assert int(r["n_pixels_total"]) == H * W
        assert float(r["s1_nan_share"]) == pytest.approx(9 / (H * W), abs=1e-6)
        assert float(r["label_invalid_share"]) == pytest.approx(4 * W / (H * W), abs=1e-6)
        assert int(r["n_s1_nan_label_01"]) == 9            # label 1 under the NaN corner
        assert float(r["vv_gt0_share"]) == 0.0 and float(r["vh_gt0_share"]) == 0.0
    with open(out / "eval-validation.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == E.M.METRIC_COLUMNS
    assert {(r["method"], r["event"]) for r in rows} == {
        (m, e) for m in ("baseline", "main") for e in ("Nigeria", "Sri-Lanka", "ALL")}
    for event in ("Nigeria", "Sri-Lanka", "ALL"):
        pair = [r for r in rows if r["event"] == event]
        assert pair[0]["valid_mask_sha256"] == pair[1]["valid_mask_sha256"]
        assert pair[0]["n_pixels"] == pair[1]["n_pixels"]
    all_main = next(r for r in rows if r["method"] == "main" and r["event"] == "ALL")
    # valid = S1 finite and label != -1: 3 chips × (48·48 − 4·48 − 9) pixels
    assert int(all_main["n_pixels"]) == 3 * (H * W - 4 * W - 9)
    assert float(all_main["iou"]) > 0.8                     # an obvious river
    s = json.loads((out / "eval-validation-summary.json").read_text(encoding="utf-8"))
    pc = s["proxy_check"]
    assert pc["status"] == "planned" and pc["test_chip"] == "Sri-Lanka_2"
    assert pc["result"]["demo_chip"]["n_placed"] == 10
    assert pc["result"]["all_chips_of_part"]["n_placed"] == 30
    assert pc["result"]["demo_chip"]["n_points"] + \
        pc["result"]["demo_chip"]["n_excluded_label_invalid"] == 10
    assert set(s["uncertainty"]) == {"Nigeria", "Sri-Lanka", "ALL"}
    assert 0 <= s["calibration"]["ALL"]["ece"] <= 1
    md = (out / "eval-validation.md").read_text(encoding="utf-8")
    assert "не фактические потери" in md and "Test не читался" in md
    assert "Пиксельный QC чипов части" in md and "eval-validation-qc.csv" in md
    calibrated = (E.MODELS_DIR / "unet" / "calibration.json").exists()   # #20, D-029
    assert ("Калибровка: применена в predict" in md) == calibrated
    assert ("сырая p" in md) == (not calibrated)
    with open(out / "eval-validation-proxy.csv", newline="", encoding="utf-8") as f:
        proxy = list(csv.DictReader(f))
    for r in proxy:                                         # EL_ref = y·V·q on scored points
        if r["included"] == "true":
            assert float(r["el_ref"]) == pytest.approx(int(r["y"]) * float(r["V"]) * float(r["q"]))


def test_pixel_qc_uses_the_single_read_of_each_chip(data_root, tmp_path, monkeypatch):
    """#64: QC comes from the arrays of the evaluation loop, no second read of S1 or label."""
    calls = {"label": 0, "s1": 0}
    real_label, real_s1 = E.io.load_label, E.io.load_s1

    def count_label(*a, **k):
        calls["label"] += 1
        return real_label(*a, **k)

    def count_s1(*a, **k):
        calls["s1"] += 1
        return real_s1(*a, **k)

    monkeypatch.setattr(E.io, "load_label", count_label)
    monkeypatch.setattr(E.io, "load_s1", count_s1)
    res = E.evaluate("validation", ["Sri-Lanka_1", "Nigeria_3"], tmp_path)
    assert calls == {"label": 2, "s1": 2}
    assert [q["chip_id"] for q in res["qc"]] == ["Nigeria_3", "Sri-Lanka_1"]
    assert res["qc"][0]["event"] == "Nigeria" and res["qc"][0]["n_s1_nan_label_01"] == 9


def test_different_valid_pixels_fail_the_run(data_root, tmp_path, monkeypatch):
    real = E.predict_baseline

    def shrunk(s1):
        finite, mask, cfg = real(s1)
        finite = finite.copy()
        finite[20, 20] = False                            # baseline loses one pixel
        return finite, mask & finite, cfg

    monkeypatch.setattr(E, "predict_baseline", shrunk)
    with pytest.raises(C.ValidMaskMismatch):
        E.evaluate("validation", ["Sri-Lanka_1"], tmp_path)


def test_confirmed_test_run_reads_test_and_logs_one_row(data_root, tmp_path, monkeypatch,
                                                        test_log):
    monkeypatch.setattr(E, "chip_ids",
                        lambda part, allow_test=False, **k: ["Spain_9"] if allow_test else 1 / 0)
    monkeypatch.setattr(E, "uncommitted_frozen_paths", lambda: "")
    monkeypatch.setenv(E.CONFIRM_ENV, E.frozen_config_hash())
    monkeypatch.setenv("FLOODVALUE_EVAL_WHO", "tester")
    out = tmp_path / "report"
    assert E.run(Namespace(part="test", out=str(out))) == 0
    assert (out / "test-eval.md").exists() and (out / "test-eval.csv").exists()
    rows = [line for line in test_log.read_text(encoding="utf-8").splitlines()
            if line.startswith("| 20")]
    assert len(rows) == 1
    assert E.frozen_config_hash() in rows[0] and "tester" in rows[0]
    assert "test-eval.md" in rows[0] and "начат" not in rows[0]
    s = json.loads((out / "test-eval-summary.json").read_text(encoding="utf-8"))
    assert s["proxy_check"]["status"] == "done" and s["proxy_check"]["result"]["independent"]


def test_failed_test_run_is_logged_as_interrupted(data_root, tmp_path, monkeypatch, test_log):
    monkeypatch.setattr(E, "chip_ids", lambda part, allow_test=False, **k: ["Spain_9"])
    monkeypatch.setattr(E, "uncommitted_frozen_paths", lambda: "")
    monkeypatch.setenv(E.CONFIRM_ENV, E.frozen_config_hash())

    def crash(*a, **k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(E, "write_report", crash)
    with pytest.raises(RuntimeError):
        E.run(Namespace(part="test", out=str(tmp_path)))
    text = test_log.read_text(encoding="utf-8")
    assert "прерван: RuntimeError" in text and "| начат |" not in text


def git_repo(tmp_path):
    import subprocess

    root = tmp_path / "repo"
    (root / "models" / "unet").mkdir(parents=True)
    (root / "models" / "baseline").mkdir(parents=True)
    (root / "splits").mkdir()
    (root / "models" / "unet" / "config.json").write_text("{}", encoding="utf-8")
    (root / "models" / "baseline" / "config.json").write_text("{}", encoding="utf-8")
    (root / "splits" / "split_v1.json").write_text("{}", encoding="utf-8")
    for cmd in (["init", "-q"], ["add", "."],
                ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "x"]):
        subprocess.run(["git", "-C", str(root), *cmd], check=True, capture_output=True)
    return root


def test_calibrator_next_to_method_changes_the_config_hash(tmp_path):
    root = git_repo(tmp_path)
    before = E.frozen_config_hash(root, "unet")
    (root / "models" / "unet" / "calibration.json").write_text('{"a": 1}', encoding="utf-8")   # where #20 puts it
    assert "models/unet/calibration.json" in E.config_files("unet", root)
    after = E.frozen_config_hash(root, "unet")
    assert after != before
    (root / "models" / "unet" / "calibration.json").write_text('{"a": 2}', encoding="utf-8")
    assert E.frozen_config_hash(root, "unet") != after


def test_gate_sees_untracked_calibrator_and_modified_config(tmp_path):
    root = git_repo(tmp_path)
    assert E.uncommitted_frozen_paths(root, "unet") == ""
    (root / "models" / "unet" / "calibration.json").write_text("{}", encoding="utf-8")         # make calibrate
    assert "calibration.json" in E.uncommitted_frozen_paths(root, "unet")
    (root / "models" / "unet" / "calibration.json").unlink()
    (root / "models" / "baseline" / "config.json").write_text('{"t": -22}', encoding="utf-8")
    assert "config.json" in E.uncommitted_frozen_paths(root, "unet")


def test_gate_refuses_when_git_fails(tmp_path):
    assert E.uncommitted_frozen_paths(tmp_path, "unet").startswith("git status failed")
