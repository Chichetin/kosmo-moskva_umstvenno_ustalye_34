"""Per-pixel features from an S1 chip in dB: pure functions, no I/O.

A feature is named by a string:
    "VV", "VH"            clipped backscatter, dB
    "VV-VH"               cross-pol ratio in dB (difference of clipped dB values)
    "<CH>_mean<w>"        local mean of the clipped channel over a w x w window
    "<CH>_std<w>"         local standard deviation over the same window
Local statistics ignore NaN neighbours (sum and count of finite pixels), so a chip edge
or a no-data hole does not pull the value towards zero. A pixel whose own S1 is not
finite gets NaN in every feature.
"""

from __future__ import annotations

import re

import numpy as np
from scipy.ndimage import uniform_filter

CHANNELS = {"VV": 0, "VH": 1}
WINDOWS = (3, 7, 15)
_LOCAL = re.compile(r"^(VV|VH)_(mean|std)(\d+)$")


def texture_names(windows=WINDOWS, channels=("VV", "VH")) -> list[str]:
    return [f"{c}_{s}{w}" for w in windows for c in channels for s in ("mean", "std")]


FULL = ["VV", "VH", "VV-VH", *texture_names()]


def clip_db(s1_db: np.ndarray, clip: dict[str, list[float]]) -> np.ndarray:
    """Clip each channel to [lo, hi] dB; NaN stays NaN (np.clip keeps NaN)."""
    out = np.array(s1_db, dtype=np.float32, copy=True)
    for ch, (lo, hi) in clip.items():
        out[CHANNELS[ch]] = np.clip(out[CHANNELS[ch]], lo, hi)
    return out


def local_stats(x: np.ndarray, w: int) -> tuple[np.ndarray, np.ndarray]:
    """NaN-aware mean and std of `x` (H, W) over a w x w window centred on each pixel.

    Only finite neighbours count; windows without finite pixels give NaN. Outside the
    array counts as missing (not as zero)."""
    ok = np.isfinite(x)
    v = np.where(ok, x, 0.0).astype(np.float64)
    n = uniform_filter(ok.astype(np.float64), w, mode="constant", cval=0.0)
    s1 = uniform_filter(v, w, mode="constant", cval=0.0)
    s2 = uniform_filter(v * v, w, mode="constant", cval=0.0)
    has = n > 0.5 / (w * w)  # at least one finite neighbour (float noise guard)
    safe = np.where(has, n, 1.0)
    mean = np.where(has, s1 / safe, np.nan)
    var = np.where(has, s2 / safe - (s1 / safe) ** 2, np.nan)
    std = np.sqrt(np.clip(var, 0.0, None))
    return mean.astype(np.float32), std.astype(np.float32)


def compute(s1_db: np.ndarray, names: list[str], clip: dict[str, list[float]]) -> np.ndarray:
    """float32 (F, H, W) features in the order of `names`; NaN where S1 is not finite."""
    if s1_db.ndim != 3 or s1_db.shape[0] != 2:
        raise ValueError(f"expected S1 (2, H, W) [VV, VH] in dB, got {s1_db.shape}")
    x = clip_db(s1_db, clip)
    valid = np.isfinite(x).all(axis=0)
    x[:, ~valid] = np.nan  # a pixel counts only when both channels exist
    cache: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    out = np.empty((len(names),) + x.shape[1:], np.float32)
    for i, name in enumerate(names):
        if name in CHANNELS:
            f = x[CHANNELS[name]]
        elif name == "VV-VH":
            f = x[0] - x[1]
        else:
            m = _LOCAL.match(name)
            if not m:
                raise ValueError(f"unknown feature {name!r}")
            ch, stat, w = m.group(1), m.group(2), int(m.group(3))
            if (ch, w) not in cache:
                cache[(ch, w)] = local_stats(x[CHANNELS[ch]], w)
            f = cache[(ch, w)][0 if stat == "mean" else 1]
        out[i] = f
    out[:, ~valid] = np.nan
    return out
