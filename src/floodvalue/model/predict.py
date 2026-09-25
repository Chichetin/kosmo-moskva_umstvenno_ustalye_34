"""`python -m floodvalue predict --chip <id> [--out <dir>]`: p-raster and mask of the main method.

Reads only the S1 chip (no labels), runs the method's `predict_proba` (docs/contracts.md §3)
and writes on the S1 grid (§2):
  flood_probability.tif  Float32, p in [0, 1], nodata -9999
  flood_mask.tif         UInt8, 1 <=> p >= threshold, nodata 255
  prediction.json        method, weights sha256, threshold and its status, timing
Method name: env `FLOODVALUE_METHOD` (default `unet`). If `models/<method>/calibration.json`
exists (written by `calibrate`, #20, D-029), the raw p is passed through its calibrator and the
mask uses its validation-chosen threshold; the file must name the same weights sha256.
Otherwise the method's stored threshold is used.
"""

import importlib
import json
import os
import time
from pathlib import Path

import numpy as np
import rasterio

from floodvalue.paths import MODELS_DIR, data_dir, output_dir

METHODS = {"unet": "floodvalue.model.unet.predict",
           "baseline": "floodvalue.baseline.predict",
           "pixel": "floodvalue.model.pixel.predict"}
DEFAULT_METHOD = "unet"
P_NODATA = -9999.0
MASK_NODATA = 255


def s1_chip_path(chip_id: str) -> Path:
    return data_dir() / "raw" / "HandLabeled" / "S1Hand" / f"{chip_id}_S1Hand.tif"


def write_rasters(p: np.ndarray, threshold: float, profile: dict, out: Path) -> None:
    """p (H, W) float32 with NaN for no data -> the two contract rasters in `out`."""
    out.mkdir(parents=True, exist_ok=True)
    nodata = ~np.isfinite(p)
    base = {k: profile[k] for k in ("crs", "transform", "width", "height")}
    base.update(driver="GTiff", count=1, compress="deflate")
    prob = np.where(nodata, P_NODATA, p).astype(np.float32)
    with rasterio.open(out / "flood_probability.tif", "w", dtype="float32", nodata=P_NODATA,
                       **base) as dst:
        dst.write(prob, 1)
    mask = np.where(nodata, MASK_NODATA, (np.nan_to_num(p, nan=0.0) >= threshold)).astype(np.uint8)
    with rasterio.open(out / "flood_mask.tif", "w", dtype="uint8", nodata=MASK_NODATA,
                       **base) as dst:
        dst.write(mask, 1)


def run(args) -> int:
    method = os.environ.get("FLOODVALUE_METHOD", DEFAULT_METHOD)
    if method not in METHODS:
        print(f"unknown method {method!r}; known: {sorted(METHODS)}")
        return 2
    if not args.chip:
        print("predict needs --chip <chip_id>, e.g. --chip Sri-Lanka_713926")
        return 2
    src_path = s1_chip_path(args.chip)
    if not src_path.exists():
        print(f"S1 chip not found: {src_path} (FLOODVALUE_DATA_DIR={data_dir()})")
        return 2
    with rasterio.open(src_path) as src:
        s1 = src.read().astype(np.float32)
        profile = src.profile
    module = importlib.import_module(METHODS[method])
    t0 = time.time()
    p = module.predict_proba(s1)
    seconds = time.time() - t0
    config = json.loads((MODELS_DIR / method / "config.json").read_text(encoding="utf-8"))
    threshold = float(config["threshold"])
    threshold_status = config.get("threshold_status", "")
    calibration = None
    cal_path = MODELS_DIR / method / "calibration.json"
    if cal_path.exists():
        from floodvalue.calibration import calibrator  # p_raw -> p, contracts §3
        cal = calibrator.load(cal_path, config["weights_sha256"])
        p = calibrator.apply(cal["calibrator"], p)
        threshold = float(cal["threshold"])
        threshold_status = cal.get("threshold_status", "calibrated on validation")
        calibration = {"file": f"models/{method}/calibration.json",
                       "sha256": calibrator.file_sha256(cal_path),
                       "kind": cal["calibrator"]["kind"]}
    out = Path(args.out) if args.out else output_dir() / args.chip
    write_rasters(p, threshold, profile, out)
    meta = {"chip_id": args.chip, "method": method, "s1_file": src_path.name,
            "weights_sha256": config["weights_sha256"], "threshold": threshold,
            "threshold_status": threshold_status, "calibration": calibration,
            "valid_pixels": int(np.isfinite(p).sum()), "seconds": round(seconds, 2)}
    (out / "prediction.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
                                         encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False))
    return 0
