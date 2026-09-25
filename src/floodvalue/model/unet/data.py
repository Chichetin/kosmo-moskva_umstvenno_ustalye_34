"""Chip I/O for training/evaluation of the U-Net. Chip lists come from the frozen split
(`floodvalue.data.split`, D-016); test is sealed there and never requested here (D-010)."""

from pathlib import Path

import numpy as np
import rasterio

from floodvalue.data.split import SealedTestError, chip_ids, load_split, part_of_event
from floodvalue.paths import data_dir


def split_chip_ids(part: str) -> list[str]:
    """Chips of an open part (train/dev/validation); test raises SealedTestError."""
    return chip_ids(part)


def split_sha256() -> str:
    """`sha256` field of the frozen split (D-016)."""
    return load_split()["sha256"]


def event_of(chip_id: str) -> str:
    return chip_id.rsplit("_", 1)[0]


def s1_path(chip_id: str, root: Path | None = None) -> Path:
    root = root or data_dir()
    return Path(root) / "raw" / "HandLabeled" / "S1Hand" / f"{chip_id}_S1Hand.tif"


def label_path(chip_id: str, root: Path | None = None) -> Path:
    root = root or data_dir()
    return Path(root) / "raw" / "HandLabeled" / "LabelHand" / f"{chip_id}_LabelHand.tif"


def read_s1(chip_id: str, root: Path | None = None) -> np.ndarray:
    """(2, H, W) float32 dB, NaN = no data (nodata tag honoured just in case)."""
    with rasterio.open(s1_path(chip_id, root)) as src:
        a = src.read().astype(np.float32)
        if src.nodata is not None and np.isfinite(src.nodata):
            a[a == src.nodata] = np.nan
    return a


def read_label(chip_id: str, root: Path | None = None) -> np.ndarray:
    """LabelHand (H, W) int16. Labels of test events are sealed (D-010)."""
    if part_of_event(event_of(chip_id)) == "test":
        raise SealedTestError(f"label of test chip {chip_id} is sealed (eval-test only)")
    with rasterio.open(label_path(chip_id, root)) as src:
        return src.read(1).astype(np.int16)


def load_chips(ids: list[str], root: Path | None = None) -> list[tuple[str, np.ndarray, np.ndarray]]:
    return [(c, read_s1(c, root), read_label(c, root)) for c in ids]
