"""Chip I/O for training/evaluation of the U-Net. Chip lists come from the frozen split
(`floodvalue.data.split`, D-016), rasters from `floodvalue.data.io`; test labels are sealed
there (SealedTestError) and test chips are never requested here (D-010)."""

import numpy as np

from floodvalue.data import io
from floodvalue.data.split import SealedTestError, chip_ids, load_split

__all__ = ["SealedTestError", "split_chip_ids", "split_sha256", "event_of", "read_s1",
           "read_label", "load_chips"]


def split_chip_ids(part: str) -> list[str]:
    """Chips of an open part (train/dev/validation); test raises SealedTestError."""
    return chip_ids(part)


def split_sha256() -> str:
    """`sha256` field of the frozen split (D-016)."""
    return load_split()["sha256"]


def event_of(chip_id: str) -> str:
    return io.parse_chip_id(chip_id)[0]


def read_s1(chip_id: str) -> np.ndarray:
    """(2, H, W) float32 dB, NaN = no data."""
    return io.load_s1(chip_id)[0].astype(np.float32)


def read_label(chip_id: str) -> np.ndarray:
    """LabelHand (H, W) int16; test labels raise SealedTestError (D-010)."""
    return io.load_label(chip_id)[0].astype(np.int16)


def load_chips(ids: list[str]) -> list[tuple[str, np.ndarray, np.ndarray]]:
    return [(c, read_s1(c), read_label(c)) for c in ids]
