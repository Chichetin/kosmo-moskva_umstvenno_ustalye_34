"""export: run_metadata.proxy_check from docs/verification/eval-<part>-summary.json (#107).

Export only reads the aggregated summary written by eval-test; it never reads labels or rasters.
The numbers are taken only for the same chip, main method and threshold as the result set.
"""

import argparse
import json

from floodvalue.contracts.schemas import ProxyCheckInfo
from floodvalue.export import assemble
from floodvalue.export import run as export_run

from test_export import CHIP, _stages, env  # noqa: F401  (pytest fixture)

SRC = "docs/verification/eval-validation-summary.json"
PRED = {"method": "unet", "threshold": 0.37}


def _summary(part="validation", chip="Sri-Lanka_713926", method="unet", threshold=0.37):
    demo = {"n_placed": 10, "n_points": 10, "mae_rub": 274285.6582, "sum_abs_error_rub": 2742856.58}
    allp = {"n_placed": 600, "n_points": 536, "mae_rub": 284926.4366}
    return {"part": part, "commit": "af52cc72a2d6", "config_sha256": "506ce620",
            "main_method": method, "threshold": threshold,
            "proxy_check": {
                "status": "done" if part == "test" else "planned", "test_chip": chip,
                "control_points": "10 точек портфеля", "fixed_v_q": "configs/assets.yaml",
                "n_points": 10, "error_metric": "MAE, руб.",
                "result": {"demo_chip": demo, "all_chips_of_part": allp, "note": "прокси",
                           "independent": part == "test"}}}


def _proxy(summary, chip="Sri-Lanka_713926", part="validation", pred=PRED, source=SRC):
    out = assemble.proxy_check(summary, source, chip, part, pred)
    ProxyCheckInfo.model_validate(out)  # contract unchanged
    return out


def test_validation_summary_of_the_same_chip_gives_numbers_but_stays_planned():
    out = _proxy(_summary())
    assert out["status"] == "planned"
    assert out["test_chip"] == "Sri-Lanka_713926" and out["n_points"] == 10
    assert out["fixed_v_q"] is True
    r = out["result"]
    assert r["estimated"] is True and r["independent"] is False
    assert r["demo_chip"]["mae_rub"] == 274285.6582
    assert r["all_chips_of_part"]["n_points"] == 536
    assert r["fixed_v_q_source"] == "configs/assets.yaml"
    assert r["source"] == SRC and r["commit"] == "af52cc72a2d6" and r["config_sha256"] == "506ce620"
    assert r["part"] == "validation"


def test_test_summary_is_done_and_independent():
    out = _proxy(_summary(part="test", chip="Spain_7370579"), chip="Spain_7370579", part="test",
                 source="docs/verification/eval-test-summary.json")
    assert out["status"] == "done" and out["result"]["independent"] is True
    assert out["result"]["demo_chip"]["n_points"] == 10


def _no_numbers(out, *words):
    assert out["status"] == "planned" and out["n_points"] is None
    r = out["result"]
    assert r["estimated"] is False and "demo_chip" not in r
    assert all(w in r["reason"] for w in words), r["reason"]


def test_other_chip_gives_no_numbers():
    _no_numbers(_proxy(_summary(), chip="Sri-Lanka_101973"), "другой чип", "Sri-Lanka_713926")


def test_stale_threshold_gives_no_numbers():
    _no_numbers(_proxy(_summary(threshold=0.5)), "устарела", "0.5", "0.37")


def test_other_main_method_gives_no_numbers():
    pred = {"method": "baseline_sigmoid", "threshold": 0.37}
    _no_numbers(_proxy(_summary(), pred=pred), "устарела", "unet")


def test_summary_of_another_part_gives_no_numbers():
    _no_numbers(_proxy(_summary(part="dev")), "устарела")


def test_test_chip_with_validation_summary_is_never_done():
    out = _proxy(_summary(chip="Spain_7370579"), chip="Spain_7370579", part="test")
    assert out["status"] != "done"


def test_no_summary_keeps_the_previous_status():
    _no_numbers(_proxy(None), "eval-test", SRC)


def test_export_reads_the_summary_of_the_chip_part(env, monkeypatch, tmp_path):  # noqa: F811
    _, args = _stages(env)
    ver = tmp_path / "verification"
    ver.mkdir()
    monkeypatch.setattr(export_run, "VERIFICATION_DIR", ver)
    monkeypatch.setattr(assemble, "chip_split", lambda split, chip: "validation")
    work_pred = json.loads((env / CHIP / "prediction.json").read_text(encoding="utf-8"))

    def meta():
        assert export_run.run(argparse.Namespace(chip=CHIP, budget=5000.0, out=None)) == 0
        _, short, _ = export_run.git_commit()
        return json.loads((env / f"{CHIP}_5000_{short}" / "run_metadata.json")
                          .read_text(encoding="utf-8"))["proxy_check"]

    assert meta()["result"]["estimated"] is False  # no summary file
    (ver / "eval-validation-summary.json").write_text(json.dumps(
        _summary(chip=CHIP, method=work_pred["method"], threshold=work_pred["threshold"])),
        encoding="utf-8")
    got = meta()
    assert got["status"] == "planned" and got["result"]["estimated"] is True
    assert got["result"]["source"] == "docs/verification/eval-validation-summary.json"
    assert got["result"]["demo_chip"]["mae_rub"] == 274285.6582
