"""Threshold baseline on Sentinel-1 backscatter: pure functions, no I/O.

Rule: water <=> backscatter (dB) <= t in one channel (VV or VH), t chosen on train by
pixel-pooled IoU. Probability (temporary main method, D-009):
p = sigmoid((t - x) / s), s fitted on train by log-loss, so p >= 0.5 <=> x <= t.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit

CHANNELS = {"VV": 0, "VH": 1}
# Candidate thresholds, dB, step 0.1. Values below the first edge count as <= every t.
EDGES = np.round(np.arange(-60.0, 30.0 + 1e-9, 0.1), 1)


def valid_mask(s1_db: np.ndarray, label: np.ndarray) -> np.ndarray:
    """Pixels that count in metrics: VV and VH finite AND label != -1 (label 0/1)."""
    return np.isfinite(s1_db).all(axis=0) & (label != -1)


def counts_le(values: np.ndarray, edges: np.ndarray = EDGES) -> np.ndarray:
    """int64 array: number of finite values <= each edge."""
    v = np.sort(values[np.isfinite(values)], kind="stable")
    return np.searchsorted(v, edges, side="right").astype(np.int64)


def curve_counts(water_le: np.ndarray, dry_le: np.ndarray, n_water: int, n_dry: int) -> dict:
    """tp/fp/fn/tn arrays for the rule `x <= edge` at every edge."""
    return {"tp": water_le, "fp": dry_le, "fn": n_water - water_le, "tn": n_dry - dry_le}


def iou_curve(water_le, dry_le, n_water: int) -> np.ndarray:
    den = (n_water + dry_le).astype(np.float64)  # tp + fp + fn
    return np.divide(water_le, den, out=np.zeros(den.shape), where=den > 0)


def best_threshold(edges, water_le, dry_le, n_water: int, n_dry: int) -> tuple[float, float]:
    """Edge with maximal IoU (lowest edge on ties) and that IoU."""
    iou = iou_curve(water_le, dry_le, n_water)
    k = int(np.argmax(iou))
    return float(edges[k]), float(iou[k])


def bin_centers(edges: np.ndarray = EDGES) -> np.ndarray:
    """Representative value of each bin (edge[k-1], edge[k]]; first bin is (-inf, edge[0]]."""
    step = np.diff(edges, prepend=edges[0])
    return edges - step / 2


def hist_from_le(le: np.ndarray) -> np.ndarray:
    """Per-bin counts from cumulative counts_le (values above the last edge are dropped)."""
    return np.diff(le, prepend=0)


def fit_scale(t: float, centers: np.ndarray, water_hist, dry_hist,
              bounds: tuple[float, float] = (0.05, 20.0)) -> float:
    """s > 0 minimising binned log-loss of p = sigmoid((t - x)/s) against labels."""
    w = np.asarray(water_hist, np.float64)
    d = np.asarray(dry_hist, np.float64)
    z0 = t - centers

    def loss(log_s: float) -> float:
        z = z0 / np.exp(log_s)
        # -log sigmoid(z) = logaddexp(0, -z); -log(1 - sigmoid(z)) = logaddexp(0, z)
        return float((w * np.logaddexp(0, -z) + d * np.logaddexp(0, z)).sum() / (w.sum() + d.sum()))

    res = minimize_scalar(loss, bounds=np.log(bounds), method="bounded",
                          options={"xatol": 1e-6})
    return float(np.exp(res.x))


def sigmoid_proba(s1_db: np.ndarray, channel: int, t: float, s: float) -> np.ndarray:
    """(2, H, W) dB -> float32 (H, W) p in [0, 1]; NaN where any channel is not finite."""
    x = s1_db[channel].astype(np.float64)
    p = expit((t - x) / s).astype(np.float32)
    p[~np.isfinite(s1_db).all(axis=0)] = np.nan
    return p


def threshold_mask(s1_db: np.ndarray, channel: int, t: float) -> np.ndarray:
    """bool (H, W): x <= t. Invalid pixels are False; mask them separately."""
    x = s1_db[channel]
    return np.isfinite(s1_db).all(axis=0) & (x <= t)


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """Otsu threshold of the finite values (maximal between-class variance)."""
    v = values[np.isfinite(values)].astype(np.float64)
    lo, hi = float(v.min()), float(v.max())
    if hi <= lo:
        return lo
    hist, edges = np.histogram(v, bins=bins, range=(lo, hi))
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centers)
    mu0 = np.divide(m0, w0, out=np.zeros(bins), where=w0 > 0)
    mu1 = np.divide(m0[-1] - m0, w1, out=np.zeros(bins), where=w1 > 0)
    between = w0 * w1 * (mu0 - mu1) ** 2
    return float(edges[int(np.argmax(between)) + 1])
