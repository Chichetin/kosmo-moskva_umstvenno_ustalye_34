"""CPU-capable inference of the U-Net: `predict_proba(s1_db) -> p` (docs/contracts.md §3).

Weights and normalization come from `models/unet/` (config.json + model.pt); the sha256 of
the weights is checked on load so an un-pulled LFS pointer fails loudly, not silently.
"""

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from floodvalue.model.unet.features import make_features, valid_mask
from floodvalue.model.unet.net import UNet
from floodvalue.paths import MODELS_DIR

DEFAULT_MODEL_DIR = MODELS_DIR / "unet"
WEIGHTS_FILE = "model.pt"
CONFIG_FILE = "config.json"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build_model(config: dict) -> UNet:
    arch = config["arch"]
    return UNet(in_channels=len(config["features"]["channels"]) + 1, base=arch["base"],
                dropout=arch.get("dropout", 0.0))


@lru_cache(maxsize=4)
def load_model(model_dir: str) -> tuple[UNet, dict]:
    d = Path(model_dir)
    config = json.loads((d / CONFIG_FILE).read_text(encoding="utf-8"))
    weights = d / WEIGHTS_FILE
    actual = file_sha256(weights)
    if actual != config["weights_sha256"]:
        raise RuntimeError(
            f"{weights}: sha256 {actual} != config {config['weights_sha256']} "
            "(LFS pointer not pulled? run `git lfs pull`)")
    model = build_model(config)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    model.eval()
    return model, config


def logits_padded(model: torch.nn.Module, x: torch.Tensor, multiple: int = 16) -> torch.Tensor:
    """Run model on (N, C, H, W), zero-padding (= invalid pixels) to a multiple of 16."""
    h, w = x.shape[-2:]
    ph, pw = (-h) % multiple, (-w) % multiple
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph))
    return model(x)[..., :h, :w]


def predict_proba(s1_db: np.ndarray, model_dir: str | Path | None = None) -> np.ndarray:
    """s1_db: float32 (2, H, W), [VV, VH] in dB, NaN = no data.
    Returns float32 (H, W): p in [0, 1], NaN where S1 has no data."""
    model, config = load_model(str(model_dir or DEFAULT_MODEL_DIR))
    f = config["features"]
    x = make_features(np.asarray(s1_db, dtype=np.float32), f["channels"], f["mean"], f["std"])
    with torch.inference_mode():
        logits = logits_padded(model, torch.from_numpy(x)[None])[0]
    p = torch.sigmoid(logits.float()).numpy().astype(np.float32)
    p[~valid_mask(s1_db)] = np.nan
    return p
