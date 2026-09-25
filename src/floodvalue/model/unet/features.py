"""Pure input preparation for the U-Net: S1 dB (2, H, W) with NaN -> normalized features.

Channels (config "channels"): any of "vv", "vh", "vv_minus_vh" (dB difference = log ratio),
followed by a validity channel (1 = S1 finite in both bands). Invalid pixels get 0 in every
data channel *after* normalization, so NaN never becomes a "bright dry" pixel.
"""

import numpy as np

# dB clip bounds per channel; VV reaches +25 dB on strong scatterers (docs/data.md).
CLIP_DB = {"vv": (-40.0, 10.0), "vh": (-45.0, 0.0), "vv_minus_vh": (-5.0, 25.0)}


def valid_mask(s1_db: np.ndarray) -> np.ndarray:
    """True where both VV and VH are finite."""
    return np.isfinite(s1_db).all(axis=0)


def raw_channels(s1_db: np.ndarray, channels: list[str]) -> np.ndarray:
    """Clipped dB channels (C, H, W) float32; NaN stays NaN."""
    vv, vh = s1_db[0].astype(np.float32), s1_db[1].astype(np.float32)
    source = {"vv": vv, "vh": vh, "vv_minus_vh": vv - vh}
    out = []
    for name in channels:
        lo, hi = CLIP_DB[name]
        out.append(np.clip(source[name], lo, hi))
    return np.stack(out).astype(np.float32)


def make_features(s1_db: np.ndarray, channels: list[str], mean: list[float],
                  std: list[float]) -> np.ndarray:
    """Normalized features (len(channels) + 1, H, W) float32, finite everywhere."""
    if s1_db.ndim != 3 or s1_db.shape[0] != 2:
        raise ValueError(f"expected s1_db of shape (2, H, W), got {s1_db.shape}")
    valid = valid_mask(s1_db)
    x = raw_channels(s1_db, channels)
    m = np.asarray(mean, dtype=np.float32)[:, None, None]
    s = np.asarray(std, dtype=np.float32)[:, None, None]
    x = (x - m) / s
    x = np.where(valid[None], x, 0.0).astype(np.float32)
    return np.concatenate([x, valid[None].astype(np.float32)], axis=0)


def channel_stats(chips: list[np.ndarray], channels: list[str]) -> tuple[list[float], list[float]]:
    """Mean/std of clipped channels over valid pixels of the given (train) chips."""
    total = np.zeros(len(channels), dtype=np.float64)
    total_sq = np.zeros(len(channels), dtype=np.float64)
    n = 0
    for s1 in chips:
        v = valid_mask(s1)
        x = raw_channels(s1, channels)[:, v].astype(np.float64)
        total += x.sum(axis=1)
        total_sq += (x ** 2).sum(axis=1)
        n += int(v.sum())
    if n == 0:
        raise ValueError("no valid pixels to compute normalization statistics")
    mean = total / n
    std = np.sqrt(np.maximum(total_sq / n - mean ** 2, 1e-12))
    return mean.tolist(), std.tolist()


def loss_mask(s1_db: np.ndarray, label: np.ndarray) -> np.ndarray:
    """Pixels that count in loss and metrics: S1 finite AND label in {0, 1}."""
    return valid_mask(s1_db) & ((label == 0) | (label == 1))
