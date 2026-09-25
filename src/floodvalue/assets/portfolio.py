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
UNCERTAINTY_RULE = (
    "Temporary: binary entropy of p_flood in bits, H(p) = -p*log2(p) - (1-p)*log2(1-p), "
    "0 at p in {0, 1}, 1 at p = 0.5. Measures only the model's own indecision at the pixel, "
    "not calibration error and not uncertainty of V/q; to be replaced by the calibration/"
    "uncertainty stage."
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
        p = float(prob_raster[row, col])
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


def assess_assets(assets: list[Asset], prob_raster: np.ndarray) -> list[dict]:
    """Rows of asset_loss.csv (without chip_id) in asset order."""
    ps = extract_p(prob_raster, [(a.row, a.col) for a in assets])
    losses = {
        a.asset_id: None if p is None else expected_loss(p, a.asset_value_rub, a.vulnerability_coef)
        for a, (p, _) in zip(assets, ps)
    }
    ranks = rank_assets(losses)
    return [
        {
            "asset_id": a.asset_id,
            "p_flood": p,
            "expected_loss_rub": losses[a.asset_id],
            "uncertainty": None if p is None else binary_entropy(p),
            "rank": ranks[a.asset_id],
            "status": status,
        }
        for a, (p, status) in zip(assets, ps)
    ]
