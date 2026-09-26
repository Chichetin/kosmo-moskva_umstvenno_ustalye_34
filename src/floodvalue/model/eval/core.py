"""Pure functions of the final evaluation (#29, criteria 4, 7, 9, 10, 14). No file I/O.

Confusion counts and IoU come from `floodvalue.model.metrics` (docs/contracts.md §7); this
module adds what the comparison report needs on top: the identical-pixel check, the paired
chip bootstrap of ΔIoU, reliability/ECE/Brier, uncertainty-vs-error AUROC, the damage proxy
check, the per-chip pixel QC (#64) and the test gate (configuration hash, test-log row).
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from floodvalue.data import io
from floodvalue.model import metrics as M

N_BINS = 10


class ValidMaskMismatch(ValueError):
    """The two methods are not evaluated on the identical set of valid pixels."""


class EvalGateError(PermissionError):
    """Test part requested without the explicit confirmation of the frozen configuration."""


# --- identical pixels ---------------------------------------------------------------------

def require_same_valid_pixels(hashes: dict[str, str]) -> str:
    """All methods must have the same valid_mask_sha256; returns it or raises."""
    values = set(hashes.values())
    if len(values) != 1:
        detail = ", ".join(f"{k}={v[:12]}" for k, v in sorted(hashes.items()))
        raise ValidMaskMismatch(f"valid pixel masks differ between methods: {detail}")
    return values.pop()


# --- comparison ---------------------------------------------------------------------------

def pooled_iou(confusions: Iterable[dict[str, int]]) -> float:
    total = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for c in confusions:
        total = M.add(total, c)
    return M.scores(total)["iou"]


def paired_bootstrap_delta_iou(main: list[dict[str, int]], base: list[dict[str, int]],
                               n: int = 2000, seed: int = 20260925) -> dict:
    """ΔIoU = IoU(main) − IoU(base), pooled over chips; both resampled with the same chips.

    Returns the point Δ, the 95 % percentile CI and the share of resamples with Δ > 0.
    Resamples where a method has an empty denominator are dropped (reported as n_used)."""
    if len(main) != len(base):
        raise ValueError("paired bootstrap needs the same chips for both methods")
    if not main:
        nan = float("nan")
        return {"delta": nan, "ci95": [nan, nan], "share_gt_0": nan, "n": n, "n_used": 0}
    a = np.array([[c["tp"], c["fp"], c["fn"]] for c in main], dtype=np.float64)
    b = np.array([[c["tp"], c["fp"], c["fn"]] for c in base], dtype=np.float64)
    idx = np.random.default_rng(seed).integers(0, len(a), size=(n, len(a)))
    sa, sb = a[idx].sum(axis=1), b[idx].sum(axis=1)
    da, db = sa.sum(axis=1), sb.sum(axis=1)
    ok = (da > 0) & (db > 0)
    delta = sa[ok, 0] / da[ok] - sb[ok, 0] / db[ok]
    point = pooled_iou(main) - pooled_iou(base)
    if not ok.any():
        return {"delta": point, "ci95": [float("nan")] * 2, "share_gt_0": float("nan"),
                "n": n, "n_used": 0}
    lo, hi = np.percentile(delta, [2.5, 97.5])
    return {"delta": float(point), "ci95": [float(lo), float(hi)],
            "share_gt_0": float((delta > 0).mean()), "n": n, "n_used": int(ok.sum())}


# --- reliability --------------------------------------------------------------------------

def reliability_counts(p: np.ndarray, y: np.ndarray, n_bins: int = N_BINS) -> dict:
    """Accumulable per-bin sums over pixels: equal-width bins on [0, 1], p = 1 in the last bin.
    `p` in [0, 1] (valid pixels only), `y` 0/1."""
    p = np.asarray(p, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if p.size and (np.nanmin(p) < 0 or np.nanmax(p) > 1 or not np.isfinite(p).all()):
        raise ValueError("reliability expects finite p in [0, 1]")
    idx = np.minimum((p * n_bins).astype(np.int64), n_bins - 1)
    return {"n": np.bincount(idx, minlength=n_bins).astype(np.int64),
            "sum_p": np.bincount(idx, weights=p, minlength=n_bins),
            "sum_y": np.bincount(idx, weights=y, minlength=n_bins),
            "sum_sq": float(((p - y) ** 2).sum())}


def add_reliability(a: dict | None, b: dict) -> dict:
    if a is None:
        return {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in b.items()}
    return {k: a[k] + b[k] for k in b}


def reliability_table(counts: dict) -> list[dict]:
    """Rows: bin, lo, hi, n, mean_p, observed water share (NaN for empty bins)."""
    n_bins = len(counts["n"])
    rows = []
    for i in range(n_bins):
        n = int(counts["n"][i])
        rows.append({"bin": i, "lo": i / n_bins, "hi": (i + 1) / n_bins, "n": n,
                     "mean_p": counts["sum_p"][i] / n if n else float("nan"),
                     "observed": counts["sum_y"][i] / n if n else float("nan")})
    return rows


def ece(counts: dict) -> float:
    """Expected calibration error: Σ_b n_b / N · |mean_p_b − observed_b|."""
    total = int(counts["n"].sum())
    if total == 0:
        return float("nan")
    return float(np.abs(counts["sum_p"] - counts["sum_y"]).sum() / total)


def brier(counts: dict) -> float:
    total = int(counts["n"].sum())
    return counts["sum_sq"] / total if total else float("nan")


# --- uncertainty --------------------------------------------------------------------------

def binary_entropy(p: np.ndarray) -> np.ndarray:
    """H(p) in bits, 0 at p ∈ {0, 1}, 1 at p = 0.5 (same measure as assets, D-018)."""
    p = np.asarray(p, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    return np.where((p <= 0) | (p >= 1), 0.0, h)


def auroc(score: np.ndarray, positive: np.ndarray) -> float:
    """Exact ROC AUC (Mann–Whitney, ties = ½): P(score of a positive > score of a negative).
    NaN if one class is empty."""
    score = np.asarray(score, dtype=np.float64).ravel()
    positive = np.asarray(positive, dtype=bool).ravel()
    n_pos = int(positive.sum())
    n_neg = positive.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(score)
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def auroc_from_hist(h_pos: np.ndarray, h_neg: np.ndarray) -> float:
    """AUROC from histograms on the same ascending bins; ties inside a bin count ½."""
    h_pos, h_neg = np.asarray(h_pos, np.float64), np.asarray(h_neg, np.float64)
    n_pos, n_neg = h_pos.sum(), h_neg.sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    below = np.concatenate([[0.0], np.cumsum(h_neg)[:-1]])
    return float((h_pos * (below + 0.5 * h_neg)).sum() / (n_pos * n_neg))


def uncertainty_on_errors(u: np.ndarray, error: np.ndarray) -> dict:
    """Mean uncertainty on wrong vs right pixels and AUROC(uncertainty -> error)."""
    u = np.asarray(u, dtype=np.float64).ravel()
    error = np.asarray(error, dtype=bool).ravel()
    return {"n_error": int(error.sum()), "n_correct": int((~error).sum()),
            "mean_u_error": float(u[error].mean()) if error.any() else float("nan"),
            "mean_u_correct": float(u[~error].mean()) if (~error).any() else float("nan"),
            "auroc": auroc(u, error)}


# --- JRC diagnostics ----------------------------------------------------------------------

def jrc_confusions(pred: np.ndarray, target: np.ndarray, valid: np.ndarray,
                   jrc: np.ndarray) -> dict[str, dict[str, int]]:
    """Confusions inside JRCWaterHand = 1 (water seen by JRC) and outside, on valid pixels."""
    inside = np.asarray(jrc) == 1
    return {"jrc_1": M.confusion(pred, target, valid & inside),
            "jrc_0": M.confusion(pred, target, valid & ~inside)}


# --- damage proxy check -------------------------------------------------------------------

def proxy_rows(points: list[dict]) -> list[dict]:
    """Per point: EL = p·V·q (from asset_loss.csv) vs EL_ref = y·V·q (y = hand label).

    Each point dict: asset_id, V, q, p (None if no_data), el (None if no_data), y (-1/0/1).
    A point is scored only if p exists and the label is valid (y ∈ {0, 1})."""
    rows = []
    for pt in points:
        y, p = pt["y"], pt["p"]
        if p is None:
            reason = "no_data"
        elif y not in (0, 1):
            reason = "label_invalid"
        else:
            reason = ""
        row = dict(pt, included=not reason, reason=reason, el_ref=None, abs_error=None)
        if not reason:
            row["el_ref"] = y * pt["V"] * pt["q"]
            row["abs_error"] = abs(pt["el"] - row["el_ref"])
        rows.append(row)
    return rows


def proxy_summary(rows: list[dict]) -> dict:
    """n points, MAE and sum of absolute errors (RUB), and Σ EL vs Σ EL_ref of scored points."""
    used = [r for r in rows if r["included"]]
    n = len(used)
    sum_abs = float(sum(r["abs_error"] for r in used))
    return {"n_placed": len(rows), "n_points": n,
            "n_excluded_label_invalid": sum(r["reason"] == "label_invalid" for r in rows),
            "n_excluded_no_data": sum(r["reason"] == "no_data" for r in rows),
            "mae_rub": sum_abs / n if n else float("nan"), "sum_abs_error_rub": sum_abs,
            "sum_el_rub": float(sum(r["el"] for r in used)),
            "sum_el_ref_rub": float(sum(r["el_ref"] for r in used))}


def lonlat_to_rowcol(transform, lon: float, lat: float) -> tuple[int, int]:
    """Pixel containing (lon, lat) on an affine grid (inverse of `transform @ (col, row)`)."""
    col, row = ~transform @ (lon, lat)
    return int(math.floor(row)), int(math.floor(col))


# --- test gate ----------------------------------------------------------------------------

def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def config_sha256(root: Path, files: list[str], method: str) -> str:
    """Hash of the frozen evaluation configuration: main method name + `relpath:sha256` of each
    file (sorted). A missing file hashes as `relpath:missing` so its later appearance changes
    the hash."""
    lines = [f"method:{method}"]
    for rel in sorted(set(files)):
        p = Path(root) / rel
        lines.append(f"{rel}:{file_sha256(p) if p.is_file() else 'missing'}")
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def check_test_confirmation(confirm: str | None, expected: str) -> None:
    """Test is read only when the caller typed the sha256 of the frozen configuration."""
    if not confirm:
        raise EvalGateError(
            "test is sealed (D-010): set FLOODVALUE_EVAL_TEST_CONFIRM=<sha256 of the frozen "
            f"configuration>; the current configuration is {expected}")
    if confirm.strip().lower() != expected:
        raise EvalGateError(
            f"FLOODVALUE_EVAL_TEST_CONFIRM={confirm.strip()} does not match the current "
            f"configuration {expected}: the configuration changed after it was confirmed")


def log_row(when: str, commit: str, config_hash: str, methods: str, who: str,
                 result: str) -> str:
    cells = [when, f"`{commit}`", f"`{config_hash}`", methods, who, result]
    return "| " + " | ".join(c.replace("|", "/").replace("\n", " ") for c in cells) + " |"


# --- pixel QC of the evaluated chips (#64, criterion 1) ----------------------------------------

QC_COLUMNS = ["chip_id", "event", "n_pixels_total", "s1_nan_share", "label_invalid_share",
              "vv_gt0_share", "vh_gt0_share", "n_s1_nan_label_01", "n_s1_nan",
              "n_label_invalid", "n_vv_finite", "n_vv_gt0",
              "n_vh_finite", "n_vh_gt0"]


def _share(num: int, den: int) -> float:
    return num / den if den else float("nan")


def pixel_qc(s1_db: np.ndarray, label: np.ndarray) -> dict:
    """Pixel QC of one chip from the arrays eval-test already holds: share of pixels with S1
    non-finite in any band, share of label -1 (both over all pixels), share of VV > 0 dB and
    VH > 0 dB among finite pixels of that band, and the count of pixels with non-finite S1
    under label 0/1 (the dataset trap). Water share is not a QC field (it is in the metrics)."""
    n = int(label.size)
    s1_nan = ~io.valid_mask(s1_db)
    n_nan, n_inv = int(s1_nan.sum()), int((label == -1).sum())
    out = {"n_pixels_total": n, "n_s1_nan": n_nan, "n_label_invalid": n_inv,
           "s1_nan_share": _share(n_nan, n), "label_invalid_share": _share(n_inv, n),
           "n_s1_nan_label_01": int((s1_nan & ((label == 0) | (label == 1))).sum())}
    for band, name in ((0, "vv"), (1, "vh")):
        x = s1_db[band]
        finite = np.isfinite(x)
        n_fin, n_pos = int(finite.sum()), int((finite & (np.nan_to_num(x) > 0)).sum())
        out.update({f"n_{name}_finite": n_fin, f"n_{name}_gt0": n_pos,
                    f"{name}_gt0_share": _share(n_pos, n_fin)})
    return out


def qc_summary(rows: list[dict]) -> dict[str, dict]:
    """Pooled pixel QC per event (in order of first appearance) and ALL."""
    scopes = list(dict.fromkeys(r["event"] for r in rows)) + ["ALL"]
    out = {}
    for scope in scopes:
        rs = [r for r in rows if scope in ("ALL", r["event"])]
        n = sum(r["n_pixels_total"] for r in rs)
        out[scope] = {
            "n_chips": len(rs),
            "s1_nan_share": _share(sum(r["n_s1_nan"] for r in rs), n),
            "label_invalid_share": _share(sum(r["n_label_invalid"] for r in rs), n),
            "vv_gt0_share": _share(sum(r["n_vv_gt0"] for r in rs),
                                   sum(r["n_vv_finite"] for r in rs)),
            "vh_gt0_share": _share(sum(r["n_vh_gt0"] for r in rs),
                                   sum(r["n_vh_finite"] for r in rs)),
            "n_s1_nan_label_01": sum(r["n_s1_nan_label_01"] for r in rs),
            "chips_s1_nan_label_01": sum(r["n_s1_nan_label_01"] > 0 for r in rs),
            "chips_with_nan": sum(r["n_s1_nan"] > 0 for r in rs)}
    return out
