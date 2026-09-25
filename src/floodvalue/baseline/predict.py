"""Baseline as the temporary main method (D-009): S1 chip -> p-raster and masks.

    uv run --frozen python -m floodvalue.baseline.predict --chip Sri-Lanka_713926

Writes to --out (default $FLOODVALUE_OUTPUT_DIR/predict/<chip>/), on the S1 grid:
  flood_probability.tif  Float32, p in [0, 1], nodata -9999
  flood_mask.tif         UInt8 1/0, nodata 255, 1 <=> p >= prob_threshold
  baseline_mask.tif      UInt8 1/0, nodata 255, the threshold rule itself
  predict.json           method, thresholds, config sha256
Reads only S1 (S1-only mode); labels are never opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from floodvalue.baseline import core
from floodvalue.paths import MODELS_DIR, data_dir, output_dir

CONFIG_PATH = MODELS_DIR / "baseline" / "config.json"
P_NODATA = -9999.0
MASK_NODATA = 255


def load_config(path: Path = CONFIG_PATH) -> dict:
    return json.loads(path.read_text())


def predict_proba(s1_db: np.ndarray, config: dict | None = None) -> np.ndarray:
    """Contract §3: float32 (2, H, W) [VV, VH] dB with NaN -> float32 (H, W) p, NaN = no data."""
    cfg = config or load_config()
    return core.sigmoid_proba(s1_db, core.CHANNELS[cfg["channel"]], cfg["threshold_db"],
                              cfg["scale_db"])


def to_rasters(s1_db: np.ndarray, config: dict) -> dict[str, np.ndarray]:
    """Arrays ready to write: p with -9999, masks with 255 where S1 is not finite."""
    p = predict_proba(s1_db, config)
    nodata = ~np.isfinite(p)
    mask = (p >= config["prob_threshold"]).astype(np.uint8)
    mask[nodata] = MASK_NODATA
    base = core.threshold_mask(s1_db, core.CHANNELS[config["channel"]],
                               config["threshold_db"]).astype(np.uint8)
    base[nodata] = MASK_NODATA
    return {"flood_probability": np.where(nodata, P_NODATA, p).astype(np.float32),
            "flood_mask": mask, "baseline_mask": base}


def s1_path(chip_id: str) -> Path:
    return data_dir() / "raw" / "HandLabeled" / "S1Hand" / f"{chip_id}_S1Hand.tif"


def predict_chip(chip_id: str, out_dir: Path, config_path: Path = CONFIG_PATH) -> Path:
    import rasterio

    config = load_config(config_path)
    with rasterio.open(s1_path(chip_id)) as src:
        s1 = src.read().astype(np.float32)
        grid = {"crs": src.crs, "transform": src.transform, "width": src.width,
                "height": src.height}
    arrays = to_rasters(s1, config)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, arr in arrays.items():
        is_p = arr.dtype == np.float32
        profile = {"driver": "GTiff", "count": 1, "dtype": "float32" if is_p else "uint8",
                   "nodata": P_NODATA if is_p else MASK_NODATA, "compress": "deflate", **grid}
        with rasterio.open(out_dir / f"{name}.tif", "w", **profile) as dst:
            dst.write(arr, 1)
    meta = {"chip_id": chip_id, "method": "baseline_sigmoid (temporary main method, D-009)",
            "threshold": config["prob_threshold"],
            "baseline_threshold_db": config["threshold_db"], "channel": config["channel"],
            "scale_db": config["scale_db"],
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "s1_file": s1_path(chip_id).name}
    (out_dir / "predict.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n")
    return out_dir


def run(args) -> int:
    if not getattr(args, "chip", None):
        print("predict: нужен --chip <chip_id> (make predict CHIP=...)")
        return 2
    out = Path(args.out) if getattr(args, "out", None) else output_dir() / "predict" / args.chip
    predict_chip(args.chip, out)
    print(f"{args.chip}: flood_probability.tif, flood_mask.tif, baseline_mask.tif -> {out}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(prog="floodvalue.baseline.predict")
    ap.add_argument("--chip", required=True)
    ap.add_argument("--out")
    raise SystemExit(run(ap.parse_args()))
