"""Portfolio of 10 point assets and their expected direct loss (task.md, criteria 13-14).

Pure functions, no I/O except reading configs/assets.yaml. Placement never sees p, EL or
labels: it takes only the S1 validity mask, the chip transform and a seed.
"""

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from rasterio import Affine

from floodvalue.paths import CONFIGS_DIR

PROB_NODATA = -9999.0
MAX_DRAWS = 1_000_000

PLACEMENT_RULE = (
    "numpy.random.default_rng(seed); asset types in configs/assets.yaml order get ids "
    "A01..A10; for each type draw row = rng.integers(0, H), col = rng.integers(0, W) and "
    "redraw while the pixel is S1 nodata (VV or VH not finite) or already taken by an "
    "earlier asset; the point is the pixel centre transform @ (col + 0.5, row + 0.5) in "
    "EPSG:4326 as [longitude, latitude]. Inputs: S1 validity mask, transform, seed only; "
    "no probabilities, masks, expected loss or labels are read."
)
P_EXTRACTION_RULE = (
    "p_flood = value of flood_probability.tif at the asset pixel (row, col) of the placement, "
    "no interpolation, independent of flood_mask.tif and threshold; nodata (-9999 or NaN) "
    "or a pixel outside the raster -> status=no_data with empty p_flood/expected_loss_rub/"
    "uncertainty/rank. EL = p_flood * asset_value_rub * vulnerability_coef, never zeroed "
    "below the mask threshold."
)
STATUS_RULE = (
    "ok: finite p at the asset pixel; no_data: p nodata at the asset pixel. partial is not "
    "produced: every asset is a single pixel, so its p is either observed or not; partial is "
    "reserved for multi-pixel assets (part of the footprint without data)."
)
UNCERTAINTY_NAME = "binary_entropy_bits"
UNCERTAINTY_RULE = (
    "uncertainty = value of uncertainty.tif (written by predict next to flood_probability.tif; "
    "measure, units and verification status in prediction.json / run_metadata.uncertainty, "
    "D-038) at the asset pixel, no interpolation; empty if that pixel is nodata. Without "
    "uncertainty.tif: binary entropy of p_flood in bits, H(p) = -p*log2(p) - (1-p)*log2(1-p), "
    "unverified. Either way it describes the flood map at the pixel, not the uncertainty of V/q."
)
UNCERTAINTY_FALLBACK_RULE = (
    "Fallback (no uncertainty.tif): binary entropy of p_flood in bits, 0 at p in {0, 1}, 1 at "
    "p = 0.5; model indecision only, not verified against errors."
)
EL_RANGE_RULE = (
    "p_low/p_high = flood_probability_range.tif (min/max over the ensemble members of the "
    "calibrated member p) at the asset pixel; el_low_rub = p_low*V*q, el_high_rub = p_high*V*q "
    "rounded to 0.01 rub (EL spread of the seed ensemble, not a confidence interval of actual "
    "losses); "
    "mask_unstable = p_low < threshold <= p_high (the mask may flip for another seed). Empty "
    "without the range raster or for status != ok."
)


@dataclass(frozen=True)
class AssetType:
    asset_class: str
    asset_value_rub: int
    vulnerability_coef: float


@dataclass(frozen=True)
class Asset:
    asset_id: str
    asset_class: str
    asset_value_rub: int
    vulnerability_coef: float
    row: int
    col: int
    longitude: float
    latitude: float


def load_asset_types(path: Path = CONFIGS_DIR / "assets.yaml") -> list[AssetType]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))["asset_types"]
    return [
        AssetType(t["asset_class"], int(t["asset_value_rub"]), float(t["vulnerability_coef"]))
        for t in raw
    ]


def place_assets(
    s1_valid_mask: np.ndarray, transform: Affine, seed: int, asset_types: list[AssetType]
) -> list[Asset]:
    """One asset per type in a random S1-valid pixel (PLACEMENT_RULE)."""
    mask = np.asarray(s1_valid_mask, dtype=bool)
    if int(mask.sum()) < len(asset_types):
        raise ValueError(f"only {int(mask.sum())} valid S1 pixels for {len(asset_types)} assets")
    h, w = mask.shape
    rng = np.random.default_rng(seed)
    taken: set[tuple[int, int]] = set()
    assets = []
    for i, t in enumerate(asset_types, start=1):
        for _ in range(MAX_DRAWS):
            row, col = int(rng.integers(0, h)), int(rng.integers(0, w))
            if mask[row, col] and (row, col) not in taken:
                break
        else:
            raise ValueError(f"no valid pixel found for {t.asset_class} in {MAX_DRAWS} draws")
        taken.add((row, col))
        lon, lat = transform @ (col + 0.5, row + 0.5)
        assets.append(Asset(f"A{i:02d}", t.asset_class, t.asset_value_rub, t.vulnerability_coef,
                            row, col, float(lon), float(lat)))
    return assets


def extract_values(raster: np.ndarray, points: list[tuple[int, int]]) -> list[float | None]:
    """Value of a float raster at each (row, col); None for nodata (-9999/NaN) or outside."""
    h, w = raster.shape
    out: list[float | None] = []
    for row, col in points:
        if not (0 <= row < h and 0 <= col < w):
            out.append(None)
            continue
        v = raster[row, col]
        x = float(np.format_float_positional(v)) if v.dtype == np.float32 else float(v)
        out.append(None if not math.isfinite(x) or x == PROB_NODATA else x)
    return out


def extract_p(
    prob_raster: np.ndarray, points: list[tuple[int, int]]
) -> list[tuple[float | None, str]]:
    """(p, status) at each (row, col): ok with the pixel value, or (None, 'no_data')."""
    h, w = prob_raster.shape
    out: list[tuple[float | None, str]] = []
    for row, col in points:
        if not (0 <= row < h and 0 <= col < w):
            out.append((None, "no_data"))
            continue
        v = prob_raster[row, col]
        # float32 raster: keep the shortest decimal that round-trips (0.4, not 0.4000000059604645)
        p = float(np.format_float_positional(v)) if v.dtype == np.float32 else float(v)
        if not math.isfinite(p) or p == PROB_NODATA:
            out.append((None, "no_data"))
            continue
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"probability {p} outside [0, 1] at pixel ({row}, {col})")
        out.append((p, "ok"))
    return out


def expected_loss(p: float, value_rub: float, vulnerability: float) -> float:
    return p * value_rub * vulnerability


def binary_entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))


def rank_assets(losses: dict[str, float | None]) -> dict[str, int | None]:
    """1 = largest EL, ties by asset_id; assets without EL get None."""
    scored = sorted((-el, aid) for aid, el in losses.items() if el is not None)
    ranks: dict[str, int | None] = {aid: None for aid in losses}
    for r, (_, aid) in enumerate(scored, start=1):
        ranks[aid] = r
    return ranks


def assess_assets(assets: list[Asset], prob_raster: np.ndarray,
                  uncertainty_raster: np.ndarray | None = None,
                  p_range: tuple[np.ndarray, np.ndarray] | None = None) -> list[dict]:
    """Rows of asset_loss.csv (without chip_id) in asset order. Optional rasters (same grid):
    uncertainty.tif (else binary entropy of p) and (p_low, p_high) of the member range."""
    points = [(a.row, a.col) for a in assets]
    ps = extract_p(prob_raster, points)
    unc = extract_values(uncertainty_raster, points) if uncertainty_raster is not None else None
    lows = extract_values(p_range[0], points) if p_range is not None else [None] * len(assets)
    highs = extract_values(p_range[1], points) if p_range is not None else [None] * len(assets)
    losses = {
        a.asset_id: None if p is None else expected_loss(p, a.asset_value_rub, a.vulnerability_coef)
        for a, (p, _) in zip(assets, ps)
    }
    ranks = rank_assets(losses)
    rows = []
    for i, (a, (p, status)) in enumerate(zip(assets, ps)):
        lo, hi = (lows[i], highs[i]) if p is not None else (None, None)
        has_range = lo is not None and hi is not None
        rows.append({
            "asset_id": a.asset_id,
            "p_flood": p,
            "expected_loss_rub": losses[a.asset_id],
            "uncertainty": None if p is None else (unc[i] if unc is not None else binary_entropy(p)),
            "rank": ranks[a.asset_id],
            "status": status,
            "p_low": lo if has_range else None,
            "p_high": hi if has_range else None,
            # rubles to the kopeck: a spread for ranking, not an exact amount
            "el_low_rub": round(expected_loss(lo, a.asset_value_rub, a.vulnerability_coef), 2)
            if has_range else None,
            "el_high_rub": round(expected_loss(hi, a.asset_value_rub, a.vulnerability_coef), 2)
            if has_range else None,
        })
    return rows


def mask_unstable(row: dict, threshold: float | None) -> bool | None:
    """p_low < threshold <= p_high: members disagree on the mask at the asset (EL is unaffected:
    EL = p*V*q never uses the threshold). None without a range or a threshold."""
    if threshold is None or row.get("p_low") is None or row.get("p_high") is None:
        return None
    return bool(row["p_low"] < threshold <= row["p_high"])
