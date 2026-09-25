"""`python -m floodvalue calibrate`: calibrator and binary threshold of the main method, on
validation only (Sri-Lanka, Nigeria; D-012, D-016). Test chips are never listed or read.

Method: env `FLOODVALUE_METHOD` (default `unet`), raw p from its `predict_proba` (contracts §3).
Protocol (two things are chosen on validation, both disclosed in the report):
  1. calibrator kind among identity / temperature / Platt / isotonic — minimum mean
     out-of-event log-loss (fit on Nigeria -> score on Sri-Lanka and back);
  2. threshold on calibrated p — maximum pooled IoU over the grid 0.01..0.99 on all
     validation; out-of-event transfer of the rule is reported alongside.
The final calibrator is refitted on all validation pixels.
Writes:
  models/<method>/calibration.json          calibrator, threshold, provenance (read by predict)
  docs/verification/calibration-val.md      report: before -> after, reliability, JRC, nodata
  docs/verification/calibration/*.csv|png   threshold curve, reliability, figures;
                                            <method>-metrics.csv = contracts §7 rows of the
                                            calibrated method (dev + validation) at the threshold
"""

from __future__ import annotations

import csv
import datetime as dt
import importlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter

from floodvalue.calibration import calibrator as C
from floodvalue.data import io
from floodvalue.data.split import chip_ids, load_split
from floodvalue.model import metrics as M
from floodvalue.model.predict import METHODS as METHOD_MODULES
from floodvalue.paths import MODELS_DIR, REPO_ROOT

DEFAULT_METHOD = "unet"
EVENTS = ("Nigeria", "Sri-Lanka")
GRID = np.round(np.arange(0.01, 1.0, 0.01), 2)
RAW_THRESHOLD = 0.5
N_BINS = 15
DOC = REPO_ROOT / "docs" / "verification" / "calibration-val.md"
DOC_DIR = REPO_ROOT / "docs" / "verification" / "calibration"
DEMO_CHIP = "Sri-Lanka_713926"


@dataclass
class Chip:
    chip_id: str
    event: str
    p_raw: np.ndarray   # (H, W) float32, NaN where S1 has no data
    label: np.ndarray   # (H, W) int16, -1 / 0 / 1
    jrc: np.ndarray | None  # (H, W) bool, JRC water (diagnostic only, never a model input)
    valid: np.ndarray   # (H, W), shared S1 and label validity mask


def load_chips(method: str, part: str = "validation") -> list[Chip]:
    if part not in ("validation", "dev"):
        raise ValueError(f"calibrate reads only validation (fit) and dev (report), not {part!r}")
    predict_proba = importlib.import_module(METHOD_MODULES[method]).predict_proba
    chips = []
    for cid in chip_ids(part):
        s1, _ = io.load_s1(cid)
        label, _ = io.load_label(cid)
        jrc_path = io.layer_path(cid, "JRCWaterHand")
        jrc = None
        if jrc_path.exists():
            import rasterio
            with rasterio.open(jrc_path) as src:
                jrc = src.read(1) == 1
        chips.append(Chip(cid, io.parse_chip_id(cid)[0], predict_proba(s1), label, jrc,
                          io.valid_mask(s1, label)))
    return chips


def pooled(chips: list[Chip], event: str | None = None, cal: dict | None = None,
           smooth: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Valid pixels of the chips (optionally one event): p (raw or calibrated), y, jrc."""
    ps, ys, js = [], [], []
    for c in chips:
        if event and c.event != event:
            continue
        p = C.apply(cal, c.p_raw) if cal else c.p_raw
        if smooth:
            p = mean3x3(p)
        v = c.valid
        ps.append(p[v])
        ys.append(c.label[v] == 1)
        js.append(c.jrc[v] if c.jrc is not None else np.zeros(int(v.sum()), bool))
    return np.concatenate(ps), np.concatenate(ys), np.concatenate(js)


def mean3x3(p: np.ndarray) -> np.ndarray:
    """NaN-aware 3x3 mean (post-processing diagnostic); NaN pixels stay NaN."""
    ok = np.isfinite(p)
    s = uniform_filter(np.where(ok, p, 0.0).astype(np.float64), 3, mode="constant")
    n = uniform_filter(ok.astype(np.float64), 3, mode="constant")
    out = np.where(ok, s / np.maximum(n, 1e-9), np.nan)
    return out.astype(np.float32)


def at(p, y, t) -> dict:
    return C.threshold_curve(p, y, [t])[0]


def cross_event(chips: list[Chip]) -> list[dict]:
    """Fit on one event, score on the other: probability scores and threshold transfer."""
    rows = []
    for fit_ev, eval_ev in ((EVENTS[0], EVENTS[1]), (EVENTS[1], EVENTS[0])):
        p_fit, y_fit, _ = pooled(chips, fit_ev)
        p_ev, y_ev, _ = pooled(chips, eval_ev)
        for kind in ("identity", *C.FITTERS):
            cal = {"kind": "identity"} if kind == "identity" else C.FITTERS[kind](p_fit, y_fit)
            q_fit, q_ev = C.apply(cal, p_fit), C.apply(cal, p_ev)
            t = C.choose_threshold(C.threshold_curve(q_fit, y_fit, GRID))
            best_curve = C.threshold_curve(q_ev, y_ev, GRID)
            t_oracle = C.choose_threshold(best_curve)
            rows.append({"fit_event": fit_ev, "eval_event": eval_ev, "kind": kind,
                         "params": params_str(cal), **C.prob_scores(q_ev, y_ev, N_BINS),
                         "threshold_from_fit": t, "iou_transfer": at(q_ev, y_ev, t)["iou"],
                         "iou_raw_0_5": at(p_ev, y_ev, RAW_THRESHOLD)["iou"],
                         "iou_oracle": at(q_ev, y_ev, t_oracle)["iou"], "threshold_oracle": t_oracle})
    return rows


def paired_bootstrap(chips: list[Chip], cal: dict, threshold: float, n: int = 2000,
                     seed: int = 20260925) -> dict:
    """Pooled IoU raw@0.5 vs calibrated@threshold, chips resampled together (paired)."""
    rows = []
    for c in chips:
        v = c.valid
        y = c.label[v] == 1
        a = at(c.p_raw[v], y, RAW_THRESHOLD)
        b = at(C.apply(cal, c.p_raw[v]), y, threshold)
        rows.append([a["tp"], a["fp"], a["fn"], b["tp"], b["fp"], b["fn"]])
    arr = np.array(rows, dtype=np.float64)
    idx = np.random.default_rng(seed).integers(0, len(arr), size=(n, len(arr)))
    s = arr[idx].sum(axis=1)
    iou_a = s[:, 0] / np.maximum(s[:, 0] + s[:, 1] + s[:, 2], 1)
    iou_b = s[:, 3] / np.maximum(s[:, 3] + s[:, 4] + s[:, 5], 1)
    d = iou_b - iou_a
    lo, hi = np.percentile(d, [2.5, 97.5])
    tot = arr.sum(axis=0)
    delta = tot[3] / tot[3:].sum() - tot[0] / tot[:3].sum()
    return {"delta_iou": float(delta), "ci_lo": float(lo), "ci_hi": float(hi), "n_boot": n,
            "n_chips": len(arr), "seed": seed}


def params_str(cal: dict) -> str:
    k = cal["kind"]
    if k == "temperature":
        return f"T={cal['t']:.3f}"
    if k == "platt":
        return f"a={cal['a']:.3f}, b={cal['b']:.3f}"
    if k == "isotonic":
        return f"{len(cal['x'])} узлов"
    return "—"


def select_kind(cross: list[dict]) -> str:
    """Minimum mean out-of-event log-loss (a proper score); ECE is reported, not optimised."""
    means = {}
    for r in cross:
        means.setdefault(r["kind"], []).append(r["log_loss"])
    return min(means, key=lambda k: (float(np.mean(means[k])), k))


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def transfer_check(chips: list[Chip], cal: dict, threshold: float) -> list[dict]:
    """Report-only: raw@0.5 vs calibrated@threshold on another part (dev, USA, low water)."""
    p, y, _ = pooled(chips)
    q = C.apply(cal, p)
    return [{"stage": "raw_0.5", "threshold": RAW_THRESHOLD, **at(p, y, RAW_THRESHOLD),
             **C.prob_scores(p, y, N_BINS)},
            {"stage": "calibrated", "threshold": threshold, **at(q, y, threshold),
             **C.prob_scores(q, y, N_BINS)}]


def metric_rows(parts: dict[str, list[Chip]], method: str, cal: dict, threshold: float) -> list[dict]:
    """docs/contracts.md §7 rows of the calibrated main method: pooled per part and per event,
    valid-mask hash over chips in chip_id order (same algorithm as baseline and U-Net)."""
    rows = []
    for part, chips in parts.items():
        events = sorted({c.event for c in chips})
        for ev in ("ALL", *events):
            mine = sorted((c for c in chips if ev == "ALL" or c.event == ev), key=lambda c: c.chip_id)
            q, y, _ = pooled(mine, None, cal)
            r = at(q, y, threshold)
            rows.append({"method": method, "split": part, "event": ev, "iou": r["iou"], "f1": r["f1"],
                         "precision": r["precision"], "recall": r["recall"], "tp": r["tp"],
                         "fp": r["fp"], "fn": r["fn"], "tn": r["tn"],
                         "n_pixels": r["tp"] + r["fp"] + r["fn"] + r["tn"], "threshold": threshold,
                         "valid_mask_sha256": M.mask_sha256([c.valid for c in mine])})
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if isinstance(v, float) and not np.isfinite(v) else
                            (round(v, 6) if isinstance(v, float) else v)) for k, v in r.items()})


def run(args) -> int:
    method = os.environ.get("FLOODVALUE_METHOD", DEFAULT_METHOD)
    if method not in METHOD_MODULES:
        print(f"unknown method {method!r}; known: {sorted(METHOD_MODULES)}")
        return 2
    config = json.loads((MODELS_DIR / method / "config.json").read_text(encoding="utf-8"))
    chips = load_chips(method)
    result = calibrate(chips, method, config)
    cal = result["calibration"]
    dev = load_chips(method, "dev")
    result["metrics"] = metric_rows({"dev": dev, "validation": chips},
                                    method, cal["calibrator"], cal["threshold"])
    result["dev_check"] = transfer_check(dev, cal["calibrator"], cal["threshold"])
    write_outputs(result, chips, method)
    print(json.dumps({k: result["calibration"][k] for k in
                      ("method", "threshold", "calibrator")}, ensure_ascii=False))
    print(f"report: {DOC.relative_to(REPO_ROOT)}")
    return 0


def calibrate(chips: list[Chip], method: str, config: dict) -> dict:
    """All numbers of the report; pure given the chips (no file writes)."""
    cross = cross_event(chips)
    kind = select_kind(cross)
    p_all, y_all, jrc_all = pooled(chips)
    cal = {"kind": "identity"} if kind == "identity" else C.FITTERS[kind](p_all, y_all)
    q_all = C.apply(cal, p_all)
    curve = C.threshold_curve(q_all, y_all, GRID)
    threshold = C.choose_threshold(curve, "iou")
    t_f1 = C.choose_threshold(curve, "f1")

    in_sample = []
    for k in ("identity", *C.FITTERS):
        c = {"kind": "identity"} if k == "identity" else C.FITTERS[k](p_all, y_all)
        in_sample.append({"kind": k, "params": params_str(c),
                          **C.prob_scores(C.apply(c, p_all), y_all, N_BINS)})

    before_after = []
    for ev in ("ALL", *EVENTS):
        p, y, _ = pooled(chips, None if ev == "ALL" else ev)
        q = C.apply(cal, p)
        before_after.append({"event": ev, "stage": "raw_0.5", "threshold": RAW_THRESHOLD,
                             **at(p, y, RAW_THRESHOLD), **C.prob_scores(p, y, N_BINS)})
        before_after.append({"event": ev, "stage": "calibrated", "threshold": threshold,
                             **at(q, y, threshold), **C.prob_scores(q, y, N_BINS)})

    jrc = []
    for name, sel in (("JRC = 1", jrc_all), ("JRC = 0", ~jrc_all)):
        if not sel.any():
            continue
        p, y = p_all[sel], y_all[sel]
        jrc.append({"zone": name, "n_pixels": int(sel.sum()), "water_share": float(y.mean()),
                    "raw_0.5": at(p, y, RAW_THRESHOLD), "calibrated": at(C.apply(cal, p), y, threshold),
                    "mean_p_cal": float(np.mean(C.apply(cal, p)))})

    post = []
    for ev in ("ALL", *EVENTS):
        q, y, _ = pooled(chips, None if ev == "ALL" else ev, cal)
        qs, ys, _ = pooled(chips, None if ev == "ALL" else ev, cal, smooth=True)
        post.append({"event": ev, "none": at(q, y, threshold)["iou"],
                     "mean3x3": at(qs, ys, threshold)["iou"]})

    boot = paired_bootstrap(chips, cal, threshold)
    split = load_split()
    calibration = {
        "method": method, "calibrator": cal, "threshold": threshold, "threshold_units": "p",
        "threshold_rule": "max pooled IoU of calibrated p on validation (Sri-Lanka + Nigeria), "
                          "grid 0.01..0.99; ties -> lowest threshold",
        "threshold_status": f"calibrated ({kind}) and chosen on validation, D-029",
        "kind_rule": "min mean out-of-event log-loss (fit Nigeria -> Sri-Lanka and back)",
        "fitted_on": {"part": "validation", "events": list(EVENTS), "n_chips": len(chips),
                      "n_pixels": int(p_all.size), "water_share": float(y_all.mean()),
                      "split_file": "splits/split_v1.json", "split_sha256": split["sha256"]},
        "input_model_sha256": config["weights_sha256"],
        "raw_threshold_equivalent": raw_equivalent(cal, threshold),
        "metrics_validation": {r["stage"]: {k: r[k] for k in ("iou", "f1", "precision", "recall",
                                                               "ece", "brier", "log_loss")}
                               for r in before_after if r["event"] == "ALL"},
        "delta_iou_paired_bootstrap": boot,
        "post_processing": "none (3x3 mean of p checked on validation, see report)",
        "nodata": "NaN in raw p (S1 not finite) stays NaN -> -9999 / 255 in rasters; label -1 "
                  "excluded from fit and metrics",
        "created_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "commit": git_commit(), "report": "docs/verification/calibration-val.md",
    }
    return {"calibration": calibration, "cross": cross, "kind": kind, "curve": curve,
            "t_f1": t_f1, "in_sample": in_sample, "before_after": before_after, "jrc": jrc,
            "post": post, "boot": boot, "rel_raw": C.reliability(p_all, y_all, N_BINS),
            "rel_cal": C.reliability(q_all, y_all, N_BINS),
            "curve_raw": C.threshold_curve(p_all, y_all, GRID)}


def raw_equivalent(cal: dict, t: float) -> float | None:
    """Smallest raw p on a fine grid whose calibrated value reaches t (monotone calibrators)."""
    grid = np.linspace(0, 1, 100001, dtype=np.float32)
    hit = np.nonzero(C.apply(cal, grid) >= np.float32(t))[0]
    return float(grid[hit[0]]) if hit.size else None


def write_outputs(result: dict, chips: list[Chip], method: str) -> None:
    from floodvalue.calibration import report
    out = MODELS_DIR / method / "calibration.json"
    out.write_text(json.dumps(result["calibration"], indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(DOC_DIR / "threshold_curve.csv",
              [{"threshold": r["threshold"], **{f"cal_{k}": r[k] for k in ("iou", "f1", "precision", "recall")},
                **{f"raw_{k}": q[k] for k in ("iou", "f1", "precision", "recall")}}
               for r, q in zip(result["curve"], result["curve_raw"], strict=True)])
    write_csv(DOC_DIR / "reliability.csv",
              [{"bin_lo": a["bin_lo"], "bin_hi": a["bin_hi"], "raw_n": a["n"], "raw_mean_p": a["mean_p"],
                "raw_water_share": a["water_share"], "cal_n": b["n"], "cal_mean_p": b["mean_p"],
                "cal_water_share": b["water_share"]}
               for a, b in zip(result["rel_raw"], result["rel_cal"], strict=True)])
    write_csv(DOC_DIR / "cross_event.csv", result["cross"])
    if result.get("metrics"):
        write_csv(DOC_DIR / f"{method}-metrics.csv", result["metrics"])
    report.figures(result, chips, DOC_DIR, DEMO_CHIP)
    DOC.write_text(report.markdown(result, method), encoding="utf-8")
