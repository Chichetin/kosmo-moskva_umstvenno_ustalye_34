"""Probability calibration `p_raw -> p` on top of any `predict_proba` (docs/contracts.md §3).

Pure functions, no raster I/O. A calibrator is a small JSON-serializable dict:

  {"kind": "identity"}
  {"kind": "temperature", "t": T}          p = sigmoid(logit(p_raw) / T)
  {"kind": "platt", "a": a, "b": b}        p = sigmoid(a * logit(p_raw) + b)
  {"kind": "isotonic", "x": [...], "y": [...]}   p = interp(p_raw; x, y), non-decreasing

Fitting works on a histogram of logit(p_raw) (bins of width 0.01 in logit space), so tens of
millions of validation pixels fit in seconds and the fit does not depend on pixel order.
NaN (no data) stays NaN through `apply`; fitting ignores it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression

EPS = 1e-6
LOGIT_MAX = 16.0
BIN_WIDTH = 0.01
KINDS = ("identity", "temperature", "platt", "isotonic")


def logit(p: np.ndarray) -> np.ndarray:
    q = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    return np.log(q) - np.log1p(-q)


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(z, dtype=np.float64)))


def _finite_pairs(p: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(p, dtype=np.float64).ravel()
    y = np.asarray(y).ravel()
    ok = np.isfinite(p)
    return p[ok], y[ok].astype(bool)


def binned(p: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Histogram of logit(p) in bins of BIN_WIDTH: (bin centers in logit, n_pos, n_neg).
    Only non-empty bins are returned."""
    p, y = _finite_pairs(p, y)
    z = np.clip(logit(p), -LOGIT_MAX, LOGIT_MAX)
    n_bins = round(2 * LOGIT_MAX / BIN_WIDTH) + 1
    idx = np.rint((z + LOGIT_MAX) / BIN_WIDTH).astype(np.int64)
    pos = np.bincount(idx[y], minlength=n_bins).astype(np.float64)
    neg = np.bincount(idx[~y], minlength=n_bins).astype(np.float64)
    keep = (pos + neg) > 0
    centers = np.arange(n_bins) * BIN_WIDTH - LOGIT_MAX
    return centers[keep], pos[keep], neg[keep]


def _nll(a: float, b: float, z: np.ndarray, pos: np.ndarray, neg: np.ndarray) -> float:
    s = a * z + b
    # -log sigmoid(s) = softplus(-s); -log(1 - sigmoid(s)) = softplus(s)
    total = (pos * np.logaddexp(0.0, -s) + neg * np.logaddexp(0.0, s)).sum()
    return float(total / (pos.sum() + neg.sum()))


def fit_platt(p: np.ndarray, y: np.ndarray) -> dict:
    """Logistic regression on logit(p_raw): minimum log-loss over (a, b)."""
    z, pos, neg = binned(p, y)
    res = minimize(lambda v: _nll(v[0], v[1], z, pos, neg), x0=[1.0, 0.0], method="Nelder-Mead",
                   options={"xatol": 1e-7, "fatol": 1e-10, "maxiter": 4000})
    return {"kind": "platt", "a": float(res.x[0]), "b": float(res.x[1])}


def fit_temperature(p: np.ndarray, y: np.ndarray) -> dict:
    """One parameter T > 0: p = sigmoid(logit(p_raw) / T); minimum log-loss."""
    z, pos, neg = binned(p, y)
    res = minimize(lambda v: _nll(float(np.exp(-v[0])), 0.0, z, pos, neg), x0=[0.0],
                   method="Nelder-Mead", options={"xatol": 1e-8, "fatol": 1e-10})
    return {"kind": "temperature", "t": float(np.exp(res.x[0]))}


def fit_isotonic(p: np.ndarray, y: np.ndarray) -> dict:
    """Non-decreasing step map p_raw -> p (pool adjacent violators on the histogram).
    Output clipped to [EPS, 1 - EPS] so log-loss stays finite on unseen pixels."""
    z, pos, neg = binned(p, y)
    n = pos + neg
    x = sigmoid(z)
    iso = IsotonicRegression(y_min=EPS, y_max=1 - EPS, increasing=True, out_of_bounds="clip")
    iso.fit(x, pos / n, sample_weight=n)
    return {"kind": "isotonic", "x": [float(v) for v in iso.X_thresholds_],
            "y": [float(v) for v in iso.y_thresholds_]}


FITTERS = {"temperature": fit_temperature, "platt": fit_platt, "isotonic": fit_isotonic}


def apply(cal: dict, p_raw: np.ndarray) -> np.ndarray:
    """Calibrated p (float32, same shape); NaN where p_raw is NaN."""
    p_raw = np.asarray(p_raw, dtype=np.float32)
    kind = cal["kind"]
    nan = ~np.isfinite(p_raw)
    q = np.where(nan, 0.5, p_raw).astype(np.float64)
    if kind == "identity":
        out = q
    elif kind == "temperature":
        out = sigmoid(logit(q) / cal["t"])
    elif kind == "platt":
        out = sigmoid(cal["a"] * logit(q) + cal["b"])
    elif kind == "isotonic":
        out = np.interp(q, cal["x"], cal["y"])
    else:
        raise ValueError(f"unknown calibrator kind {kind!r}; known: {KINDS}")
    out = np.clip(out, 0.0, 1.0).astype(np.float32)
    out[nan] = np.nan
    return out


# --- scores (pure, NaN-aware) -------------------------------------------------------------

def reliability(p: np.ndarray, y: np.ndarray, n_bins: int = 15) -> list[dict]:
    """Equal-width bins on [0, 1]; per bin: count, mean p, observed water share."""
    p, y = _finite_pairs(p, y)
    idx = np.minimum((p * n_bins).astype(np.int64), n_bins - 1)
    n = np.bincount(idx, minlength=n_bins)
    sp = np.bincount(idx, weights=p, minlength=n_bins)
    sy = np.bincount(idx, weights=y.astype(np.float64), minlength=n_bins)
    rows = []
    for i in range(n_bins):
        rows.append({"bin_lo": i / n_bins, "bin_hi": (i + 1) / n_bins, "n": int(n[i]),
                     "mean_p": sp[i] / n[i] if n[i] else float("nan"),
                     "water_share": sy[i] / n[i] if n[i] else float("nan")})
    return rows


def ece(p: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error: sum_b n_b / N * |mean p_b - water share_b|."""
    rows = reliability(p, y, n_bins)
    total = sum(r["n"] for r in rows)
    if total == 0:
        return float("nan")
    return float(sum(r["n"] / total * abs(r["mean_p"] - r["water_share"])
                     for r in rows if r["n"]))


def brier(p: np.ndarray, y: np.ndarray) -> float:
    p, y = _finite_pairs(p, y)
    return float(np.mean((p - y) ** 2)) if p.size else float("nan")


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p, y = _finite_pairs(p, y)
    if not p.size:
        return float("nan")
    q = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(np.where(y, np.log(q), np.log1p(-q))))


def prob_scores(p: np.ndarray, y: np.ndarray, n_bins: int = 15) -> dict:
    p_f, y_f = _finite_pairs(p, y)
    return {"ece": ece(p_f, y_f, n_bins), "brier": brier(p_f, y_f),
            "log_loss": log_loss(p_f, y_f), "mean_p": float(p_f.mean()) if p_f.size else float("nan"),
            "water_share": float(y_f.mean()) if y_f.size else float("nan"), "n_pixels": int(p_f.size)}


def threshold_curve(p: np.ndarray, y: np.ndarray, thresholds) -> list[dict]:
    """mask = p >= t for each t: pooled TP/FP/FN/TN and IoU, F1, precision, recall."""
    p, y = _finite_pairs(p, y)
    pos = np.sort(p[y])
    neg = np.sort(p[~y])
    rows = []
    for t in thresholds:
        t = float(t)
        tp = int(pos.size - np.searchsorted(pos, t, side="left"))
        fp = int(neg.size - np.searchsorted(neg, t, side="left"))
        fn = int(pos.size - tp)
        tn = int(neg.size - fp)
        rows.append({"threshold": t, **counts_scores(tp, fp, fn, tn)})
    return rows


def counts_scores(tp: int, fp: int, fn: int, tn: int) -> dict:
    def r(a, b):
        return a / b if b > 0 else float("nan")
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "iou": r(tp, tp + fp + fn),
            "f1": r(2 * tp, 2 * tp + fp + fn), "precision": r(tp, tp + fp), "recall": r(tp, tp + fn)}


def choose_threshold(curve: list[dict], key: str = "iou") -> float:
    """Threshold with the maximum `key`; ties go to the lowest threshold (favours recall:
    a missed flooded asset costs the operator more than an extra inspection, D-001)."""
    best = None
    for row in curve:
        v = row[key]
        if np.isfinite(v) and (best is None or v > best[0]):
            best = (v, row["threshold"])
    if best is None:
        raise ValueError("no finite score on the curve")
    return float(best[1])


# --- calibration file ---------------------------------------------------------------------

def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path: Path, input_model_sha256: str | None = None) -> dict:
    """Read calibration.json; refuse if it was fitted on another model (stale calibrator)."""
    cal = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("calibrator", "threshold", "input_model_sha256"):
        if key not in cal:
            raise ValueError(f"{path}: missing key {key!r}")
    if cal["calibrator"]["kind"] not in KINDS:
        raise ValueError(f"{path}: unknown calibrator kind {cal['calibrator']['kind']!r}")
    t = float(cal["threshold"])
    if not 0.0 < t < 1.0:
        raise ValueError(f"{path}: threshold {t} outside (0, 1)")
    if input_model_sha256 is not None and cal["input_model_sha256"] != input_model_sha256:
        raise RuntimeError(
            f"{path}: fitted on model sha256 {cal['input_model_sha256']}, current model is "
            f"{input_model_sha256}; rerun `make calibrate`")
    return cal
