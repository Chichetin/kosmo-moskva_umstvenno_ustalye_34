"""Pure functions of the pre-event water layer: window, water frequency, new water, diagnostics.

Parameters are fixed before any table is computed and are never tuned on validation (#149 p.5).
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np

# Same rule as the shipped baseline (D-021): water <=> VH <= -23.0 dB.
VH_WATER_DB = -23.0
# Pre-event water: water in at least half of the pixel's valid observations.
MIN_WATER_SHARE = 0.5
# Fewer scenes (or fewer valid observations of a pixel) -> not enough to call water permanent.
N_MIN_SCENES = 5
# Window of archive scenes and the gap between its end and the S1 date of the event. Flood
# onset dates are not in the dataset; a fixed gap keeps the flood itself out of the window for
# floods that started up to GUARD_DAYS before the S1 image (explicit limitation, #149 p.2).
WINDOW_DAYS = 100
GUARD_DAYS = 60

PERM_NODATA = 255
P_NODATA = -9999.0

STATUS_OK = "ok"
STATUS_INSUFFICIENT = "insufficient"
STATUS_UNAVAILABLE = "unavailable"


def pre_event_window(s1_date: date, window_days: int = WINDOW_DAYS,
                     guard_days: int = GUARD_DAYS) -> tuple[date, date]:
    """(first, last) calendar day of archive scenes, both inclusive; last < s1_date - guard."""
    last = s1_date - timedelta(days=guard_days + 1)
    return last - timedelta(days=window_days - 1), last


def linear_to_db(gamma0: np.ndarray) -> np.ndarray:
    """RTC gamma0 power (linear) -> dB; non-positive or non-finite -> NaN."""
    g = np.asarray(gamma0, dtype=np.float64)
    ok = np.isfinite(g) & (g > 0)
    out = np.full(g.shape, np.nan, dtype=np.float32)
    out[ok] = (10.0 * np.log10(g[ok])).astype(np.float32)
    return out


def water_frequency(vh_db: np.ndarray, threshold_db: float = VH_WATER_DB
                    ) -> tuple[np.ndarray, np.ndarray]:
    """(n_valid, n_water) per pixel of a (N, H, W) VH stack in dB with NaN = no observation."""
    stack = np.asarray(vh_db, dtype=np.float32)
    valid = np.isfinite(stack)
    water = valid & (np.where(valid, stack, np.inf) <= threshold_db)
    return valid.sum(axis=0), water.sum(axis=0)


def permanent_water(vh_db: np.ndarray, threshold_db: float = VH_WATER_DB,
                    min_share: float = MIN_WATER_SHARE, n_min: int = N_MIN_SCENES) -> np.ndarray:
    """UInt8 (H, W): 1 pre-event water, 0 not, 255 fewer than `n_min` valid observations."""
    n_valid, n_water = water_frequency(vh_db, threshold_db)
    out = np.full(n_valid.shape, PERM_NODATA, dtype=np.uint8)
    enough = n_valid >= n_min
    share = np.divide(n_water, n_valid, out=np.zeros(n_valid.shape, dtype=np.float64),
                      where=n_valid > 0)
    out[enough] = (share[enough] >= min_share).astype(np.uint8)
    return out


def layer_status(n_scenes: int, n_min: int = N_MIN_SCENES) -> str:
    return STATUS_OK if n_scenes >= n_min else STATUS_INSUFFICIENT


def new_water_probability(p: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """Float32 p x (1 - perm); -9999 where p has no data or perm is 255.

    p comes from flood_probability.tif as is (never the mask): an asset below the threshold
    keeps its p. Pre-event water only removes probability, it never adds any."""
    p = np.asarray(p, dtype=np.float32)
    perm = np.asarray(perm)
    nodata = ~np.isfinite(p) | (p == P_NODATA) | (perm == PERM_NODATA)
    out = np.where(nodata, P_NODATA, p * (1.0 - (perm == 1)))
    return out.astype(np.float32)


def diagnostics(label: np.ndarray, perm: np.ndarray, jrc: np.ndarray | None = None) -> dict:
    """Per-chip table row: how the pre-event layer relates to the hand label (and JRC).

    Valid pixel: label != -1 and perm != 255. Counts are returned so events can be pooled."""
    label = np.asarray(label)
    perm = np.asarray(perm)
    valid = (label != -1) & (perm != PERM_NODATA)
    y1 = valid & (label == 1)
    y0 = valid & (label == 0)
    p1 = valid & (perm == 1)
    row = {
        "valid_px": int(valid.sum()),
        "perm_px": int(p1.sum()),
        "label_water_px": int(y1.sum()),
        "label_dry_px": int(y0.sum()),
        "perm_and_water_px": int((p1 & y1).sum()),
        "perm_and_dry_px": int((p1 & y0).sum()),
        "jrc_px": None,
        "perm_and_jrc_px": None,
        "perm_or_jrc_px": None,
    }
    if jrc is not None:
        j = valid & (np.asarray(jrc) == 1)
        row |= {"jrc_px": int(j.sum()), "perm_and_jrc_px": int((p1 & j).sum()),
                "perm_or_jrc_px": int((p1 | j).sum())}
    return row


def ratios(row: dict) -> dict:
    """Shares of a (pooled) diagnostics row; None where the denominator is 0.

    p_water_given_perm   P(label = 1 | perm = 1) — how often pre-event water is water now
    perm_on_dry_share    share of label = 0 pixels marked pre-event water (perm ∩ dry)
    removed_water_share  share of label = 1 pixels that are pre-event water (removed from "new")
    perm_share           share of valid pixels that are pre-event water
    iou_jrc              IoU of perm with JRCWaterHand (diagnostic only; JRC may be post-event)
    """
    def div(a, b):
        return None if not b or a is None else a / b
    return {
        "p_water_given_perm": div(row["perm_and_water_px"], row["perm_px"]),
        "perm_on_dry_share": div(row["perm_and_dry_px"], row["label_dry_px"]),
        "removed_water_share": div(row["perm_and_water_px"], row["label_water_px"]),
        "perm_share": div(row["perm_px"], row["valid_px"]),
        "iou_jrc": div(row["perm_and_jrc_px"], row["perm_or_jrc_px"]),
    }
