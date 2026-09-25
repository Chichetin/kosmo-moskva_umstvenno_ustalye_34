"""Chip naming, paths and readers for Sen1Floods11 HandLabeled chips (docs/data.md §3-§4).

Layout under $FLOODVALUE_DATA_DIR/raw mirrors the bucket without the `v1.1/data/flood_events/`
prefix: raw/HandLabeled/<LAYER>/<EVENT>_<ID>_<LAYER>.tif.

`load_label` refuses chips of test events unless `allow_test=True` (only `eval-test`,
D-010). `load_s1` is split-agnostic: inference on any official chip is allowed.
"""

from pathlib import Path

import numpy as np
import rasterio

from floodvalue.data.split import EVENT_PARTS, SealedTestError
from floodvalue.paths import data_dir

S1_LAYER = "S1Hand"
LABEL_LAYER = "LabelHand"
JRC_LAYER = "JRCWaterHand"
LAYERS = (S1_LAYER, LABEL_LAYER, JRC_LAYER)


def raw_dir() -> Path:
    return data_dir() / "raw"


def interim_dir() -> Path:
    return data_dir() / "interim"


def parse_chip_id(chip_id: str) -> tuple[str, str]:
    """`Sri-Lanka_713926` -> ("Sri-Lanka", "713926"). The event may contain '-', never '_'."""
    event, sep, num = chip_id.rpartition("_")
    if not sep or not event or not num.isdigit():
        raise ValueError(f"bad chip_id {chip_id!r}, expected EVENT_NUMBER")
    return event, num


def chip_id_from_filename(name: str) -> str:
    """`Sri-Lanka_713926_S1Hand.tif` -> `Sri-Lanka_713926`."""
    event, num, _layer = Path(name).name.rsplit("_", 2)
    chip_id = f"{event}_{num}"
    parse_chip_id(chip_id)
    return chip_id


def layer_relpath(chip_id: str, layer: str) -> str:
    """Path relative to raw/ (and, prefixed with `v1.1/data/flood_events/`, the bucket name)."""
    parse_chip_id(chip_id)
    return f"HandLabeled/{layer}/{chip_id}_{layer}.tif"


def layer_path(chip_id: str, layer: str) -> Path:
    return raw_dir() / layer_relpath(chip_id, layer)


def load_s1(chip_id: str) -> tuple[np.ndarray, dict]:
    """S1 chip as float32 (2, H, W): [VV, VH] in dB exactly as stored (already dB, never
    re-logged), NaN = no data. Returns (array, rasterio profile)."""
    with rasterio.open(layer_path(chip_id, S1_LAYER)) as src:
        arr = src.read(masked=False).astype(np.float32, copy=False)
        profile = dict(src.profile)
    if arr.shape[0] != 2:
        raise ValueError(f"{chip_id}: S1Hand has {arr.shape[0]} bands, expected 2 (VV, VH)")
    return arr, profile


def load_label(chip_id: str, *, allow_test: bool = False) -> tuple[np.ndarray, dict]:
    """LabelHand as int16 (H, W): -1 no reliable label, 0 not water, 1 water.
    -1 is not declared as nodata in the file; mask it yourself.
    Chips of test events are sealed (D-010, D-016): only eval-test passes allow_test=True."""
    event, _ = parse_chip_id(chip_id)
    if event in EVENT_PARTS["test"] and not allow_test:
        raise SealedTestError(f"{chip_id}: test labels are sealed; only eval-test reads them")
    with rasterio.open(layer_path(chip_id, LABEL_LAYER)) as src:
        arr = src.read(1).astype(np.int16, copy=False)
        profile = dict(src.profile)
    return arr, profile


def valid_mask(s1_db: np.ndarray, label: np.ndarray | None = None) -> np.ndarray:
    """Pixel is valid when both S1 bands are finite (and label != -1 if a label is given).
    Under S1 NaN the label can be 0/1, so both conditions are needed (docs/data.md §8.2)."""
    ok = np.isfinite(s1_db).all(axis=0)
    if label is not None:
        ok &= label != -1
    return ok
