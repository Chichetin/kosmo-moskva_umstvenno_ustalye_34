"""Main method candidate A: S1 chip -> p-raster (contract §3) and mask. S1-only, CPU.

    uv run --frozen python -m floodvalue.model.pixel.predict --chip Sri-Lanka_713926

Writes to --out (default $FLOODVALUE_OUTPUT_DIR/pixel/<chip>/), on the S1 grid:
  flood_probability.tif  Float32, p in [0, 1], nodata -9999
  flood_mask.tif         UInt8 1/0, nodata 255, 1 <=> p >= threshold (config.json)
  prediction.json        method, threshold, config and model sha256, timing
Labels are never opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from floodvalue.model.pixel import features
from floodvalue.paths import MODELS_DIR, output_dir

MODEL_DIR = MODELS_DIR / "pixel"
CONFIG_PATH = MODEL_DIR / "config.json"
P_NODATA = -9999.0
MASK_NODATA = 255


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_config(path: Path = CONFIG_PATH) -> dict:
    return json.loads(Path(path).read_text())


def load_model(config: dict, model_dir: Path = MODEL_DIR):
    """Classifier from models/pixel/, refusing a file whose sha256 differs from config
    (e.g. a Git LFS pointer instead of the weights)."""
    import joblib

    path = Path(model_dir) / config["model_file"]
    digest = sha256_file(path)
    if digest != config["model_sha256"]:
        raise ValueError(f"{path}: sha256 {digest} != config {config['model_sha256']} "
                         "(weights missing? run `git lfs pull`)")
    return joblib.load(path)


def proba_from_features(model, feats: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """p (H, W) float32 from features (F, H, W); NaN outside `valid`."""
    p = np.full(valid.shape, np.nan, np.float32)
    if valid.any():
        x = feats[:, valid].T
        p[valid] = model.predict_proba(x)[:, 1].astype(np.float32)
    return p


def predict_proba(s1_db: np.ndarray, model=None, config: dict | None = None) -> np.ndarray:
    """Contract §3: float32 (2, H, W) [VV, VH] dB with NaN -> float32 (H, W) p, NaN = no data."""
    cfg = config or load_config()
    mdl = model if model is not None else load_model(cfg)
    s1 = np.asarray(s1_db, dtype=np.float32)
    feats = features.compute(s1, cfg["features"], cfg["clip_db"])
    valid = np.isfinite(s1).all(axis=0)
    return proba_from_features(mdl, feats, valid)


def to_rasters(p: np.ndarray, threshold: float) -> dict[str, np.ndarray]:
    """Arrays ready to write: p with -9999 and mask with 255 where p is NaN."""
    nodata = ~np.isfinite(p)
    mask = (np.nan_to_num(p, nan=-1.0) >= threshold).astype(np.uint8)
    mask[nodata] = MASK_NODATA
    return {"flood_probability": np.where(nodata, P_NODATA, p).astype(np.float32),
            "flood_mask": mask}


def predict_chip(chip_id: str, out_dir: Path, config_path: Path = CONFIG_PATH) -> dict:
    import rasterio

    from floodvalue.data.io import layer_path, load_s1

    t0 = time.perf_counter()
    config = load_config(config_path)
    model = load_model(config, Path(config_path).parent)
    s1, profile = load_s1(chip_id)
    p = predict_proba(s1, model, config)
    arrays = to_rasters(p, config["threshold"])
    out_dir.mkdir(parents=True, exist_ok=True)
    grid = {k: profile[k] for k in ("crs", "transform", "width", "height")}
    for name, arr in arrays.items():
        is_p = arr.dtype == np.float32
        prof = {"driver": "GTiff", "count": 1, "dtype": "float32" if is_p else "uint8",
                "nodata": P_NODATA if is_p else MASK_NODATA, "compress": "deflate", **grid}
        with rasterio.open(out_dir / f"{name}.tif", "w", **prof) as dst:
            dst.write(arr, 1)
    meta = {"chip_id": chip_id, "method": "pixel", "model": config["method"],
            "threshold": config["threshold"],
            "threshold_status": config.get("threshold_status", ""),
            "features": config["features"],
            "config_sha256": sha256_file(Path(config_path)),
            "weights_sha256": config["model_sha256"],
            "s1_file": layer_path(chip_id, "S1Hand").name, "inputs": "S1 only (VV, VH dB)",
            "seconds": round(time.perf_counter() - t0, 2)}
    (out_dir / "prediction.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n")
    return meta


def run(args) -> int:
    if not getattr(args, "chip", None):
        print("predict: нужен --chip <chip_id>")
        return 2
    out = Path(args.out) if getattr(args, "out", None) else output_dir() / "pixel" / args.chip
    meta = predict_chip(args.chip, out)
    print(f"{args.chip}: flood_probability.tif, flood_mask.tif -> {out} ({meta['seconds']} s)")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(prog="floodvalue.model.pixel.predict")
    ap.add_argument("--chip", required=True)
    ap.add_argument("--out")
    raise SystemExit(run(ap.parse_args()))
