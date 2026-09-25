"""Training of the compact U-Net on S1 only (D-009, D-012).

    python -m floodvalue train                       # default configuration -> models/unet/
    python -m floodvalue.model.unet.train --help     # variants for the controlled experiment

Protocol: normalization statistics from train; checkpoint = best pooled IoU at p >= 0.5 on
dev (USA, D-012); validation metrics are only reported (calibration/threshold is #20);
test is never read. Loss counts only pixels with finite S1 and label 0/1.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from floodvalue.model import metrics as M
from floodvalue.model.unet import data as D
from floodvalue.model.unet.features import CLIP_DB, channel_stats, loss_mask, make_features
from floodvalue.model.unet.predict import CONFIG_FILE, WEIGHTS_FILE, build_model, file_sha256
from floodvalue.paths import MODELS_DIR

DEFAULTS = {
    "name": "unet",
    "channels": ["vv", "vh", "vv_minus_vh"],
    "base": 16,
    "dropout": 0.1,
    "epochs": 60,
    "batch": 16,
    "crop": 256,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "dice_weight": 0.5,
    "db_jitter": 1.0,
    "seed": 20260925,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="floodvalue.model.unet.train")
    p.add_argument("--name", default=DEFAULTS["name"])
    p.add_argument("--channels", default=",".join(DEFAULTS["channels"]))
    for key in ("base", "epochs", "batch", "crop", "seed"):
        p.add_argument(f"--{key.replace('_', '-')}", type=int, default=DEFAULTS[key])
    for key in ("dropout", "lr", "weight_decay", "dice_weight", "db_jitter"):
        p.add_argument(f"--{key.replace('_', '-')}", type=float, default=DEFAULTS[key])
    p.add_argument("--out-dir", default=str(MODELS_DIR / "unet"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-chips", type=int, default=0, help="smoke test: limit chips per part")
    return p.parse_args(argv)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def masked_loss(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                dice_weight: float) -> torch.Tensor:
    """BCE + soft Dice over pixels where mask is True (label -1 and NaN S1 excluded)."""
    logits, target, m = logits.float(), target.float(), mask.float()
    n = m.sum().clamp_min(1.0)
    bce = (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * m).sum() / n
    if dice_weight == 0:
        return bce
    p = torch.sigmoid(logits) * m
    t = target * m
    dice = 1 - (2 * (p * t).sum() + 1) / (p.sum() + t.sum() + 1)
    return bce + dice_weight * dice


def prepare(part: str, cfg: dict, max_chips: int = 0) -> dict:
    ids = D.split_chip_ids(part)
    if max_chips:
        ids = ids[:max_chips]
    chips = D.load_chips(ids)
    return {"ids": ids, "raw": chips}


def tensors(chips: list, feats: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = np.stack([make_features(s1, feats["channels"], feats["mean"], feats["std"])
                  for _, s1, _ in chips])
    y = np.stack([(lab == 1) for _, _, lab in chips]).astype(np.uint8)
    m = np.stack([loss_mask(s1, lab) for _, s1, lab in chips]).astype(np.uint8)
    return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(m)


def augment(x: torch.Tensor, y: torch.Tensor, m: torch.Tensor, crop: int, jitter: float,
            std: torch.Tensor, n_data: int, gen: torch.Generator):
    """Random crop, flips, rot90 and a global dB offset (same for VV and VH) per sample."""
    b, _, h, w = x.shape
    xs, ys, ms = [], [], []
    for i in range(b):
        top = int(torch.randint(0, h - crop + 1, (1,), generator=gen))
        left = int(torch.randint(0, w - crop + 1, (1,), generator=gen))
        xi = x[i, :, top:top + crop, left:left + crop]
        yi = y[i, top:top + crop, left:left + crop]
        mi = m[i, top:top + crop, left:left + crop]
        k = int(torch.randint(0, 4, (1,), generator=gen))
        flip = bool(torch.randint(0, 2, (1,), generator=gen))
        xi, yi, mi = (torch.rot90(t, k, dims=(-2, -1)) for t in (xi, yi, mi))
        if flip:
            xi, yi, mi = (torch.flip(t, dims=(-1,)) for t in (xi, yi, mi))
        if jitter > 0:
            shift_db = (torch.rand(1, generator=gen).item() * 2 - 1) * jitter
            valid = xi[n_data:n_data + 1]
            xi = xi.clone()
            xi[:n_data] = xi[:n_data] + valid * shift_db / std[:, None, None]
        xs.append(xi)
        ys.append(yi)
        ms.append(mi)
    return torch.stack(xs), torch.stack(ys), torch.stack(ms)


def jitter_std(channels: list[str], std: list[float]) -> torch.Tensor:
    """A common dB offset shifts VV and VH; VV−VH is unchanged (std set to inf)."""
    return torch.tensor([s if c in ("vv", "vh") else float("inf")
                         for c, s in zip(channels, std)], dtype=torch.float32)


@torch.no_grad()
def predict_probs(model, x: torch.Tensor, device: str, batch: int = 8) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(x), batch):
        xb = x[i:i + batch].to(device)
        with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16,
                            enabled=device.startswith("cuda")):
            out.append(torch.sigmoid(model(xb).float()).cpu().numpy())
    return np.concatenate(out)


def evaluate(probs: np.ndarray, ids: list[str], y: torch.Tensor, m: torch.Tensor,
             threshold: float = 0.5) -> dict:
    """Pooled and per-event metrics at `threshold` plus per-chip counts for bootstrap."""
    y, m = y.numpy().astype(bool), m.numpy().astype(bool)
    per_chip = [M.confusion(probs[i] >= threshold, y[i], m[i]) for i in range(len(ids))]
    events: dict[str, dict] = {}
    for cid, c in zip(ids, per_chip):
        events[D.event_of(cid)] = M.add(events.get(D.event_of(cid), {}), c)
    total: dict[str, int] = {}
    for c in per_chip:
        total = M.add(total, c)
    return {"all": total, "events": events, "per_chip": per_chip,
            "valid_mask_sha256": M.mask_sha256(list(m))}


def best_threshold_iou(probs: np.ndarray, y: torch.Tensor, m: torch.Tensor) -> tuple[float, float]:
    """Diagnostic only (not used for selection): max pooled IoU over a threshold grid."""
    mb = m.numpy().astype(bool)
    p = probs[mb]
    t = y.numpy().astype(bool)[mb]
    best = (0.5, -1.0)
    for thr in np.round(np.arange(0.05, 0.96, 0.05), 2):
        iou = M.scores(M.confusion(p >= thr, t, np.ones_like(t)))["iou"]
        if iou > best[1]:
            best = (float(thr), float(iou))
    return best


def metric_rows(name: str, split: str, ev: dict, threshold: float) -> list[dict]:
    rows = []
    for event, c in [("ALL", ev["all"])] + sorted(ev["events"].items()):
        rows.append({"method": name, "split": split, "event": event, **M.scores(c), **c,
                     "threshold": threshold, "valid_mask_sha256": ev["valid_mask_sha256"]})
    return rows


def train(args: argparse.Namespace) -> dict:
    seed_everything(args.seed)
    channels = [c.strip() for c in args.channels.split(",") if c.strip()]
    device = args.device
    t0 = time.time()
    tr = prepare("train", vars(args), args.max_chips)
    dv = prepare("dev", vars(args), args.max_chips)
    mean, std = channel_stats([s1 for _, s1, _ in tr["raw"]], channels)
    feats = {"channels": channels, "mean": mean, "std": std}
    xtr, ytr, mtr = tensors(tr["raw"], feats)
    xdv, ydv, mdv = tensors(dv["raw"], feats)
    del tr["raw"], dv["raw"]
    print(f"loaded train {len(xtr)} dev {len(xdv)} chips in {time.time() - t0:.0f}s; "
          f"mean {np.round(mean, 2)} std {np.round(std, 2)}", flush=True)

    config = {"method": "unet", "arch": {"name": "UNet", "base": args.base,
                                         "dropout": args.dropout, "levels": 4},
              "features": {**feats, "validity_channel": True,
                           "clip_db": {c: list(CLIP_DB[c]) for c in channels}}}
    model = build_model(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, (2 * len(xtr)) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, pct_start=0.1,
                                                total_steps=args.epochs * steps_per_epoch)
    gen = torch.Generator().manual_seed(args.seed)
    jstd = jitter_std(channels, std)
    xtr_d, ytr_d, mtr_d = xtr.to(device), ytr.to(device), mtr.to(device)
    jstd_d = jstd.to(device)
    amp = device.startswith("cuda")

    history, best = [], {"iou": -1.0, "epoch": -1, "state": None}
    for epoch in range(1, args.epochs + 1):
        model.train()
        te, losses = time.time(), []
        for _ in range(steps_per_epoch):
            idx = torch.randint(0, len(xtr), (args.batch,), generator=gen)
            xb, yb, mb = augment(xtr_d[idx.to(device)], ytr_d[idx.to(device)],
                                 mtr_d[idx.to(device)], args.crop, args.db_jitter, jstd_d,
                                 len(channels), gen)
            with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16,
                                enabled=amp):
                logits = model(xb)
            loss = masked_loss(logits, yb, mb.bool(), args.dice_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
        probs = predict_probs(model, xdv, device)
        ev = evaluate(probs, dv["ids"], ydv, mdv)
        s = M.scores(ev["all"])
        with torch.no_grad():
            dloss = masked_loss(torch.logit(torch.from_numpy(probs).clamp(1e-6, 1 - 1e-6)),
                                ydv, mdv.bool(), 0.0).item()
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "dev_bce": dloss,
                        "dev_iou": s["iou"], "dev_f1": s["f1"], "dev_precision": s["precision"],
                        "dev_recall": s["recall"], "lr": sched.get_last_lr()[0],
                        "seconds": time.time() - te})
        if s["iou"] > best["iou"]:
            best = {"iou": s["iou"], "epoch": epoch,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
        print(f"epoch {epoch:3d} loss {np.mean(losses):.4f} dev_bce {dloss:.4f} "
              f"dev IoU {s['iou']:.4f} P {s['precision']:.3f} R {s['recall']:.3f} "
              f"best {best['iou']:.4f}@{best['epoch']}", flush=True)

    model.load_state_dict(best["state"])
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    weights = out / WEIGHTS_FILE
    torch.save(best["state"], weights)

    # Final report of the selected checkpoint: dev (selection) and validation (report only).
    probs_dv = predict_probs(model, xdv, device)
    ev_dv = evaluate(probs_dv, dv["ids"], ydv, mdv)
    del xtr_d, ytr_d, mtr_d, xtr
    va = prepare("validation", vars(args), args.max_chips)
    xva, yva, mva = tensors(va["raw"], feats)
    del va["raw"]
    probs_va = predict_probs(model, xva, device)
    ev_va = evaluate(probs_va, va["ids"], yva, mva)
    rows = metric_rows(args.name, "dev", ev_dv, 0.5) + metric_rows(args.name, "validation",
                                                                   ev_va, 0.5)
    ci = {"dev": M.bootstrap_iou(ev_dv["per_chip"], seed=args.seed),
          "validation": M.bootstrap_iou(ev_va["per_chip"], seed=args.seed)}
    diag = {"dev_best_threshold_iou": best_threshold_iou(probs_dv, ydv, mdv),
            "validation_best_threshold_iou_diagnostic": best_threshold_iou(probs_va, yva, mva)}

    config.update({
        "name": args.name,
        "weights_file": WEIGHTS_FILE,
        "weights_sha256": file_sha256(weights),
        "n_parameters": n_params,
        "threshold": 0.5,
        "threshold_status": "uncalibrated: raw p >= 0.5; calibration and final threshold "
                            "are chosen on validation by calibration (#20, D-012)",
        "selection": {"part": "dev", "metric": "pooled IoU at p >= 0.5",
                      "epoch": best["epoch"], "dev_iou": best["iou"]},
        "split": {"file": "splits/split_v1.json", "sha256": D.split_sha256(),
                  "train_chips": len(tr["ids"]),
                  "dev_chips": len(dv["ids"])},
        "train": {k: getattr(args, k) for k in ("epochs", "batch", "crop", "lr", "weight_decay",
                                                  "dice_weight", "db_jitter", "seed")}
                 | {"loss": "masked BCE + dice_weight * soft Dice", "optimizer": "AdamW",
                    "schedule": "OneCycle", "steps_per_epoch": steps_per_epoch,
                    "amp": "bfloat16" if amp else "off", "augment": "crop, rot90, hflip, "
                    "global dB offset U(-db_jitter, db_jitter) on VV and VH",
                    "device": torch.cuda.get_device_name(0) if amp else "cpu",
                    "seconds": round(time.time() - t0)},
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "numpy": np.__version__},
        "command": "python -m floodvalue.model.unet.train " + " ".join(sys.argv[1:]),
        "gpu_tolerance_note": "GPU training is not bit-exact across hardware; CPU inference of "
                              "the stored weights is deterministic",
    })
    (out / CONFIG_FILE).write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n",
                                   encoding="utf-8")
    with open(out / "history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0]))
        w.writeheader()
        w.writerows(history)
    with open(out / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=M.METRIC_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if isinstance(r[k], float) and not np.isfinite(r[k]) else r[k])
                        for k in M.METRIC_COLUMNS})
    summary = {"name": args.name, "best_epoch": best["epoch"], "iou_ci95": ci, **diag,
               "weights_sha256": config["weights_sha256"]}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def run(args) -> int:
    """`python -m floodvalue train` entry (method dispatcher passes the CLI namespace)."""
    train(parse_args([]))
    return 0


if __name__ == "__main__":
    train(parse_args())
