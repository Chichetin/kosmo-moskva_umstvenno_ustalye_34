"""Pixel metrics shared by the methods (docs/contracts.md §7): pure functions, no I/O.

Rows follow the `metrics.csv` columns: method, split, event, iou, f1, precision, recall,
tp, fp, fn, tn, n_pixels, threshold, valid_mask_sha256. Undefined ratios are NaN (empty in CSV).
"""

import hashlib

import numpy as np

METRIC_COLUMNS = ["method", "split", "event", "iou", "f1", "precision", "recall",
                  "tp", "fp", "fn", "tn", "n_pixels", "threshold", "valid_mask_sha256"]


def confusion(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict[str, int]:
    """TP/FP/FN/TN over pixels where `mask` is True; pred/target are 0/1 or bool."""
    p = np.asarray(pred).astype(bool)[mask]
    t = np.asarray(target).astype(bool)[mask]
    return {"tp": int((p & t).sum()), "fp": int((p & ~t).sum()),
            "fn": int((~p & t).sum()), "tn": int((~p & ~t).sum())}


def add(a: dict[str, int], b: dict[str, int]) -> dict[str, int]:
    return {k: a.get(k, 0) + b[k] for k in ("tp", "fp", "fn", "tn")}


def _ratio(num: int, den: int) -> float:
    return num / den if den > 0 else float("nan")


def scores(c: dict[str, int]) -> dict[str, float]:
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    return {"iou": _ratio(tp, tp + fp + fn), "f1": _ratio(2 * tp, 2 * tp + fp + fn),
            "precision": _ratio(tp, tp + fp), "recall": _ratio(tp, tp + fn),
            "n_pixels": tp + fp + fn + tn}


def mask_sha256(masks: list[np.ndarray]) -> str:
    """Hash of the valid-pixel masks (in the given chip order) — equal masks, equal pixels."""
    h = hashlib.sha256()
    for m in masks:
        h.update(np.packbits(np.asarray(m, dtype=bool)).tobytes())
    return h.hexdigest()


def bootstrap_iou(per_chip: list[dict[str, int]], n: int = 1000, seed: int = 0,
                  alpha: float = 0.05) -> tuple[float, float]:
    """Percentile CI of pooled IoU, resampling chips with replacement."""
    if not per_chip:
        return float("nan"), float("nan")
    arr = np.array([[c["tp"], c["fp"], c["fn"]] for c in per_chip], dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n, len(arr)))
    s = arr[idx].sum(axis=1)
    den = s.sum(axis=1)
    iou = np.where(den > 0, s[:, 0] / np.where(den > 0, den, 1), np.nan)
    lo, hi = np.nanpercentile(iou, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)
