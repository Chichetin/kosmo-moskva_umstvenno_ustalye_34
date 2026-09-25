"""Segmentation metrics shared by all methods (contracts §7).

Every method is scored on the same valid-pixel mask: S1 finite in both channels AND
label != −1. `valid_mask_sha256` fingerprints that mask so two rows of `metrics.csv`
can be checked to cover identical pixels. Pure functions, no I/O.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

import numpy as np

METRICS_COLUMNS = ["method", "split", "event", "iou", "f1", "precision", "recall",
                   "tp", "fp", "fn", "tn", "n_pixels", "threshold", "valid_mask_sha256"]
_KEYS = ("tp", "fp", "fn", "tn")


def valid_mask(s1_db: np.ndarray, label: np.ndarray) -> np.ndarray:
    """(2, H, W) S1 in dB + (H, W) LabelHand -> bool (H, W) of pixels that count."""
    return np.isfinite(s1_db).all(axis=0) & (label != -1)


def confusion(pred: np.ndarray, label: np.ndarray, valid: np.ndarray) -> dict[str, int]:
    """Pixel counts over `valid`; pred is water (bool), label 1 = water."""
    p = pred.astype(bool)[valid]
    y = label[valid] == 1
    return {
        "tp": int(np.count_nonzero(p & y)),
        "fp": int(np.count_nonzero(p & ~y)),
        "fn": int(np.count_nonzero(~p & y)),
        "tn": int(np.count_nonzero(~p & ~y)),
    }


def add_counts(a: dict[str, int], b: dict[str, int] | None) -> dict[str, int]:
    if b is None:
        return dict(a)
    return {k: a[k] + b[k] for k in _KEYS}


def _ratio(num: int, den: int) -> float | None:
    return num / den if den else None


def scores(tp: int, fp: int, fn: int, tn: int) -> dict:
    """IoU, F1 (= Dice), precision, recall; None where the denominator is zero."""
    return {
        "iou": _ratio(tp, tp + fp + fn),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "n_pixels": tp + fp + fn + tn,
    }


def valid_mask_sha256(masks: Iterable[tuple[str, np.ndarray]]) -> str:
    """sha256 over (chip_id, packed bool mask) in chip_id order."""
    h = hashlib.sha256()
    for chip_id, mask in sorted(masks, key=lambda t: t[0]):
        h.update(chip_id.encode("ascii") + b"\0")
        h.update(np.asarray(mask.shape, np.int64).tobytes())
        h.update(np.packbits(mask.astype(bool), axis=None).tobytes())
    return h.hexdigest()


def metrics_row(method: str, split: str, event: str, counts: dict[str, int],
                threshold: float | None, valid_mask_sha256: str) -> dict:
    """One row of metrics.csv; undefined ratios are written as empty strings."""
    s = scores(**{k: counts[k] for k in _KEYS})
    row = {"method": method, "split": split, "event": event}
    for k in ("iou", "f1", "precision", "recall"):
        row[k] = "" if s[k] is None else round(s[k], 6)
    row.update({k: counts[k] for k in _KEYS})
    row["n_pixels"] = s["n_pixels"]
    row["threshold"] = "" if threshold is None else threshold
    row["valid_mask_sha256"] = valid_mask_sha256
    return {c: row[c] for c in METRICS_COLUMNS}
