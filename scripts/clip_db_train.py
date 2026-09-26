"""Train-only dB statistics of S1 vs the fixed U-Net CLIP_DB bounds (issue #125).

Reads only `chip_ids("train")` via `load_s1` (dB as stored, never re-logged). A pixel
counts when both VV and VH are finite; labels are not read. Test chips are never opened.

    FLOODVALUE_DATA_DIR=... uv run --frozen python scripts/clip_db_train.py
"""

import numpy as np

from floodvalue.data.io import load_s1
from floodvalue.data.split import chip_ids
from floodvalue.model.unet.features import CLIP_DB

Q = (0.01, 0.1, 1.0, 50.0, 99.0, 99.9, 99.99)


def main() -> None:
    ids = chip_ids("train")
    per_chip = []
    for cid in ids:
        s1, _ = load_s1(cid)
        ok = np.isfinite(s1).all(axis=0)
        per_chip.append(s1[:, ok])  # (2, n_valid)
    n = sum(a.shape[1] for a in per_chip)
    print(f"train chips: {len(ids)}; chips with valid S1: "
          f"{sum(a.shape[1] > 0 for a in per_chip)}; valid pixels (VV and VH finite): {n}")
    header = "| channel | min | " + " | ".join(f"p{q:g}" for q in Q) + \
        " | max | CLIP_DB | share < lo | share > hi |"
    print(header)
    print("|" + "---|" * (header.count("|") - 1))
    for name in ("vv", "vh", "vv_minus_vh"):
        if name == "vv":
            x = np.concatenate([a[0] for a in per_chip])
        elif name == "vh":
            x = np.concatenate([a[1] for a in per_chip])
        else:
            x = np.concatenate([a[0] - a[1] for a in per_chip])
        lo, hi = CLIP_DB[name]
        qs = np.percentile(x.astype(np.float64), Q)
        below, above = float((x < lo).mean()), float((x > hi).mean())
        print(f"| {name} | {x.min():.2f} | " + " | ".join(f"{v:.2f}" for v in qs)
              + f" | {x.max():.2f} | [{lo:g}; {hi:g}] | {below:.5%} | {above:.5%} |")
        del x


if __name__ == "__main__":
    main()
