"""CPU-capable inference of the U-Net: `predict_proba(s1_db) -> p` (docs/contracts.md §3).

Weights and normalization come from `models/unet/` (config.json + member_*.pt). The model may
be an ensemble of seeds (config "members"); p is the mean of member probabilities. Every
member's sha256 is checked on load so an un-pulled LFS pointer fails loudly, not silently.
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


def members_sha256(members: list[dict]) -> str:
    """One hash for the whole ensemble: sha256 of "file:sha256" lines in member order."""
    text = "\n".join(f"{m['file']}:{m['sha256']}" for m in members)
    return hashlib.sha256(text.encode()).hexdigest()


def config_members(config: dict) -> list[dict]:
    """Ensemble members; a single-model config is an ensemble of one."""
    if "members" in config:
        return config["members"]
    return [{"file": config.get("weights_file", WEIGHTS_FILE), "sha256": config["weights_sha256"]}]


@lru_cache(maxsize=4)
def load_model(model_dir: str) -> tuple[list[UNet], dict]:
    d = Path(model_dir)
    config = json.loads((d / CONFIG_FILE).read_text(encoding="utf-8"))
    models = []
    for member in config_members(config):
        weights = d / member["file"]
        actual = file_sha256(weights)
        if actual != member["sha256"]:
            raise RuntimeError(
                f"{weights}: sha256 {actual} != config {member['sha256']} "
                "(LFS pointer not pulled? run `git lfs pull`)")
        model = build_model(config)
        model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        model.eval()
        models.append(model)
    return models, config


def logits_padded(model: torch.nn.Module, x: torch.Tensor, multiple: int = 16) -> torch.Tensor:
    """Run model on (N, C, H, W), zero-padding (= invalid pixels) to a multiple of 16."""
    h, w = x.shape[-2:]
    ph, pw = (-h) % multiple, (-w) % multiple
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph))
    return model(x)[..., :h, :w]


def _member_probs(s1_db: np.ndarray, model_dir: str | Path | None) -> torch.Tensor:
    """(K, H, W) sigmoid probabilities of the K ensemble members (NaN not applied yet)."""
    models, config = load_model(str(model_dir or DEFAULT_MODEL_DIR))
    f = config["features"]
    x = torch.from_numpy(make_features(np.asarray(s1_db, dtype=np.float32), f["channels"],
                                       f["mean"], f["std"]))[None]
    with torch.inference_mode():
        return torch.stack([torch.sigmoid(logits_padded(m, x)[0].float()) for m in models])


def predict_proba(s1_db: np.ndarray, model_dir: str | Path | None = None) -> np.ndarray:
    """s1_db: float32 (2, H, W), [VV, VH] in dB, NaN = no data.
    Returns float32 (H, W): p in [0, 1], NaN where S1 has no data."""
    # ensemble = mean of member probabilities
    p = _member_probs(s1_db, model_dir).mean(0).numpy().astype(np.float32)
    p[~valid_mask(s1_db)] = np.nan
    return p


def predict_members(s1_db: np.ndarray, model_dir: str | Path | None = None) -> np.ndarray:
    """Raw p of every ensemble member: float32 (K, H, W), NaN where S1 has no data.
    Their mean equals `predict_proba` (used for the uncertainty raster, #21)."""
    p = _member_probs(s1_db, model_dir).numpy().astype(np.float32)
    p[:, ~valid_mask(s1_db)] = np.nan
    return p
