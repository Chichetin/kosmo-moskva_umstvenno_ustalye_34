"""`python -m floodvalue predict --chip <id> [--out <dir>]`: p-raster and mask of the main method.

Reads only the S1 chip (no labels), runs the method's `predict_proba` (docs/contracts.md §3)
and writes on the S1 grid (§2):
  flood_probability.tif  Float32, p in [0, 1], nodata -9999
  flood_mask.tif         UInt8, 1 <=> p >= threshold, nodata 255
  prediction.json        method, weights sha256, threshold and its status, timing
  uncertainty.tif        Float32, per-pixel uncertainty, nodata -9999 (measure: prediction.json
                         "uncertainty"; chosen by `models/<method>/uncertainty.json`, D-038,
                         else entropy of p, unverified)
  flood_probability_range.tif  Float32 x 2 (p_low, p_high): min/max of the calibrated member p
                         (ensemble methods; a single model gives p_low = p_high = p)
Consumers of the two extra rasters: `assets` (uncertainty, p_low/p_high in asset_loss.csv),
`export` (copies uncertainty.tif), `eval-test` (entropy of p = the same measure), `app`.
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
           "baseline": "floodvalue.baseline.predict"}
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


def write_uncertainty(u: np.ndarray, lo: np.ndarray, hi: np.ndarray, profile: dict,
                      out: Path) -> None:
    """uncertainty.tif (1 band) and flood_probability_range.tif (p_low, p_high), NaN -> -9999."""
    base = {k: profile[k] for k in ("crs", "transform", "width", "height")}
    base.update(driver="GTiff", compress="deflate", dtype="float32", nodata=P_NODATA)
    with rasterio.open(out / "uncertainty.tif", "w", count=1, **base) as dst:
        dst.write(np.where(np.isfinite(u), u, P_NODATA).astype(np.float32), 1)
    with rasterio.open(out / "flood_probability_range.tif", "w", count=2, **base) as dst:
        for band, (arr, name) in enumerate(((lo, "p_low"), (hi, "p_high")), start=1):
            dst.write(np.where(np.isfinite(arr), arr, P_NODATA).astype(np.float32), band)
            dst.set_band_description(band, name)


def uncertainty_rasters(module, s1: np.ndarray, p_raw: np.ndarray, p: np.ndarray,
                        cal: dict | None, threshold: float, method: str,
                        weights_sha256: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Measure chosen in models/<method>/uncertainty.json (D-038) on the method's members."""
    from floodvalue.calibration import uncertainty as U  # contracts §2 uncertainty.tif
    members = module.predict_members(s1) if hasattr(module, "predict_members") else p_raw[None]
    info_path = MODELS_DIR / method / "uncertainty.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else None
    if info and info.get("input_model_sha256") != weights_sha256:
        raise RuntimeError(f"{info_path}: input_model_sha256 != weights of models/{method}")
    wanted = info["measure"] if info else U.FALLBACK
    measure = U.resolve(wanted, members.shape[0])
    u = U.measures(members, cal, threshold, p)[measure]
    lo, hi = U.p_range(members, cal)
    verified = bool(info) and measure == wanted
    meta = {"file": "uncertainty.tif", "measure": measure, "units": U.UNITS[measure],
            "status": info["status"] if verified else
            "не проверено: нет models/<method>/uncertainty.json или показатель недоступен для "
            "метода — энтропия p без проверки на ошибках",
            "choice": f"models/{method}/uncertainty.json" if info else None,
            "validation": info.get("validation") if verified else None,
            "n_members": int(members.shape[0]),
            "range_file": "flood_probability_range.tif",
            "range_meaning": "p_low/p_high = min/max over members of calibrated member p; "
                             "seed spread, not a confidence interval"}
    return u, lo, hi, meta


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
    p_raw = p = module.predict_proba(s1)
    seconds = time.time() - t0
    config = json.loads((MODELS_DIR / method / "config.json").read_text(encoding="utf-8"))
    threshold = float(config["threshold"])
    threshold_status = config.get("threshold_status", "")
    calibration = cal_fn = None
    cal_path = MODELS_DIR / method / "calibration.json"
    if cal_path.exists():
        from floodvalue.calibration import calibrator  # p_raw -> p, contracts §3
        cal = calibrator.load(cal_path, config["weights_sha256"])
        cal_fn = cal["calibrator"]
        p = calibrator.apply(cal_fn, p)
        threshold = float(cal["threshold"])
        threshold_status = cal.get("threshold_status", "calibrated on validation")
        calibration = {"file": f"models/{method}/calibration.json",
                       "sha256": calibrator.file_sha256(cal_path),
                       "kind": cal["calibrator"]["kind"]}
    out = Path(args.out) if args.out else output_dir() / args.chip
    write_rasters(p, threshold, profile, out)
    u, lo, hi, unc_meta = uncertainty_rasters(module, s1, p_raw, p, cal_fn, threshold, method,
                                              config["weights_sha256"])
    write_uncertainty(u, lo, hi, profile, out)
    meta = {"chip_id": args.chip, "method": method, "s1_file": src_path.name,
            "weights_sha256": config["weights_sha256"], "threshold": threshold,
            "threshold_status": threshold_status, "calibration": calibration,
            "uncertainty": unc_meta,
            "valid_pixels": int(np.isfinite(p).sum()), "seconds": round(seconds, 2)}
    (out / "prediction.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
                                         encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False))
    return 0
