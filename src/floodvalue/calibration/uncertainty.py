"""Per-pixel uncertainty on top of any `predict_proba` (issue #21, criterion 10, D-038).

Pure functions, no raster I/O. Input: raw p of the K members of the method, float32
(K, H, W) with NaN where S1 has no data (a single-model method is K = 1), the calibrator
of `models/<method>/calibration.json` (D-029) and the mask threshold t. With
q_k = calibrate(p_k) and the delivered p = calibrate(mean_k p_k):

  entropy_p         H(p), bits, 0..1: indecision of the delivered p (the D-018 placeholder)
  threshold_margin  exp(-|logit p - logit t|), 0..1, 1 at the mask threshold: closeness of
                    the delivered p to the decision threshold (odds ratio to t)
  member_std        std_k q_k (ddof=0), units of p, 0..0.5: spread of the seed ensemble
  mutual_info       H(mean_k q_k) - mean_k H(q_k), bits, 0..1: epistemic part (BALD); the
                    members are passed through the calibrator fitted on the ensemble mean,
                    an approximation

The first two only restate the model's own p (model confidence); the last two need >= 2
members. Which one is delivered is chosen on validation (D-038) and verified against errors;
none of them is a verified uncertainty by construction.

`p_range` gives [min_k q_k, max_k q_k]; the calibrator is monotone, so the delivered p lies
inside it. It is the min-max spread of the seeds trained on the same data, not a confidence
interval of the flood extent or of actual losses.
"""

from __future__ import annotations

import numpy as np

from floodvalue.calibration import calibrator as C

# order = simplicity, used to break near-ties in the selection rule (D-038)
MEASURES = ("entropy_p", "threshold_margin", "member_std", "mutual_info")
ENSEMBLE_MEASURES = ("member_std", "mutual_info")
UNITS = {"entropy_p": "биты, 0…1", "threshold_margin": "безразмерная, 0…1 (1 — p на пороге)",
         "member_std": "единицы p, 0…0,5", "mutual_info": "биты, 0…1"}
FALLBACK = "entropy_p"
EPS = 1e-6


def entropy_bits(p: np.ndarray) -> np.ndarray:
    """Binary entropy in bits, 0 at p in {0, 1}, 1 at 0.5; NaN stays NaN."""
    p = np.asarray(p, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    h = np.where((p <= 0) | (p >= 1), 0.0, h)
    return np.where(np.isfinite(p), h, np.nan)


def threshold_margin(p: np.ndarray, threshold: float) -> np.ndarray:
    """exp(-|logit p - logit t|) = min(odds/odds_t, odds_t/odds); 1 at p = t; NaN stays NaN."""
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    t = float(np.clip(threshold, EPS, 1 - EPS))
    z = np.log(p) - np.log1p(-p) - (np.log(t) - np.log1p(-t))
    return np.exp(-np.abs(z))


def _members(members_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = np.asarray(members_raw, dtype=np.float32)
    if m.ndim == 2:
        m = m[None]
    return m, ~np.isfinite(m).all(axis=0)


def _calibrate(cal: dict | None, p: np.ndarray) -> np.ndarray:
    return np.asarray(C.apply(cal, p) if cal else p, dtype=np.float64)


def measures(members_raw: np.ndarray, cal: dict | None, threshold: float,
             p: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """All candidate measures, float32 (H, W), NaN where any member has no data.
    `p` — the delivered (calibrated) p raster; default calibrate(mean of members)."""
    m, nodata = _members(members_raw)
    p = _calibrate(cal, m.mean(axis=0)) if p is None else np.asarray(p, dtype=np.float64)
    q = _calibrate(cal, m)
    h_total = entropy_bits(q.mean(axis=0))
    out = {"entropy_p": entropy_bits(p),
           "threshold_margin": threshold_margin(p, threshold),
           "member_std": q.std(axis=0),
           # H is concave, so MI >= 0; the clip only removes rounding below zero
           "mutual_info": np.maximum(h_total - entropy_bits(q).mean(axis=0), 0.0)}
    return {k: np.where(nodata | ~np.isfinite(p), np.nan, v).astype(np.float32)
            for k, v in out.items()}


def p_range(members_raw: np.ndarray, cal: dict | None) -> tuple[np.ndarray, np.ndarray]:
    """(p_low, p_high) = min/max over members of the calibrated member p, float32, NaN = no data."""
    m, nodata = _members(members_raw)
    q = _calibrate(cal, m)
    lo = np.where(nodata, np.nan, q.min(axis=0)).astype(np.float32)
    hi = np.where(nodata, np.nan, q.max(axis=0)).astype(np.float32)
    return lo, hi


def resolve(measure: str, n_members: int) -> str:
    """The measure actually computed: an ensemble measure needs >= 2 members, else entropy_p."""
    if measure not in MEASURES:
        raise ValueError(f"unknown uncertainty measure {measure!r}; known: {MEASURES}")
    return FALLBACK if measure in ENSEMBLE_MEASURES and n_members < 2 else measure


# --- selection on validation (D-038) --------------------------------------------------------

def auroc_hist(h_pos: np.ndarray, h_neg: np.ndarray) -> np.ndarray:
    """AUROC from histograms on the same ascending bins (last axis), ties inside a bin = 1/2:
    P(u of a random error > u of a random correct pixel). NaN if a class is empty."""
    h_pos, h_neg = np.asarray(h_pos, np.float64), np.asarray(h_neg, np.float64)
    below = np.cumsum(h_neg, axis=-1) - h_neg
    n_pos, n_neg = h_pos.sum(-1), h_neg.sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (h_pos * (below + 0.5 * h_neg)).sum(-1) / (n_pos * n_neg)


def worst_event_auroc(hists: dict[str, tuple[np.ndarray, np.ndarray]]) -> float:
    """min over events of AUROC; hists: event -> (errors histogram, correct histogram)."""
    return float(min(auroc_hist(e, c) for e, c in hists.values()))


def select(per_chip: dict[str, list[tuple[str, np.ndarray, np.ndarray]]], n_boot: int = 1000,
           seed: int = 20260925) -> dict:
    """Selection rule of D-038 on per-chip histograms.

    per_chip: measure -> [(event, errors histogram, correct histogram) per chip], same chip
    order for every measure. Score S = worst per-event AUROC (pooled over the chips of the
    event). Best = max S; a simpler measure (earlier in MEASURES) replaces it when the 95 %
    chip-bootstrap interval of S_best - S_simpler (chips resampled within each event, the same
    resamples for all measures) contains 0. Returns scores, the bootstrap intervals and the choice."""
    names = [m for m in MEASURES if m in per_chip]
    events = [e for e, _, _ in per_chip[names[0]]]
    ev_names = sorted(set(events))
    arr = {m: (np.stack([h for _, h, _ in per_chip[m]]), np.stack([h for _, _, h in per_chip[m]]))
           for m in names}

    def score(m: str, idx_by_event: dict[str, np.ndarray]) -> float:
        e, c = arr[m]
        return float(min(auroc_hist(e[i].sum(0), c[i].sum(0)) for i in idx_by_event.values()))

    full = {ev: np.array([i for i, e in enumerate(events) if e == ev]) for ev in ev_names}
    scores = {m: score(m, full) for m in names}
    best = max(names, key=lambda m: (scores[m], -names.index(m)))
    rng = np.random.default_rng(seed)
    boots = {m: [] for m in names}
    for _ in range(n_boot):
        idx = {ev: rng.choice(ix, size=len(ix), replace=True) for ev, ix in full.items()}
        for m in names:
            boots[m].append(score(m, idx))
    diffs = {}
    for m in names:
        d = np.array(boots[best]) - np.array(boots[m])
        lo, hi = np.percentile(d, [2.5, 97.5])
        diffs[m] = {"delta": scores[best] - scores[m], "ci_lo": float(lo), "ci_hi": float(hi)}
    chosen = next(m for m in names if m == best or diffs[m]["ci_lo"] <= 0.0)
    return {"scores": scores, "best": best, "chosen": chosen, "vs_best": diffs,
            "n_boot": n_boot, "seed": seed, "events": ev_names}
