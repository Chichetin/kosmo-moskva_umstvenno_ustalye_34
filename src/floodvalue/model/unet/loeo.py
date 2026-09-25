"""Leave-one-event-out (LOEO) transfer check of the U-Net recipe on train+dev events (#86).

    python -m floodvalue.model.unet.loeo --out-dir outputs/loeo          # GPU (Colab), D-036
    python -m floodvalue.model.unet.loeo --table-from outputs/loeo       # markdown table

Each fold holds out one of the 6 train+dev events (Ghana, India, Pakistan, Paraguay, Somalia,
USA), trains one U-Net on the other 5 and scores pooled IoU/F1 at p >= 0.5 on the held-out
event. The baseline is scored on the same valid pixels (`data.io.valid_mask`), both with the
shipped rule from `models/baseline/config.json` and refitted on the fold's training events
(VH <= t, 0.1 dB grid, max pooled IoU — the rule of D-008 with the channel fixed to VH).

No leakage into the fold: normalization statistics come from the fold's training events; there
is no checkpoint selection on any event — the D-019 schedule (60 epochs, OneCycle) is run and
the held-out event is scored at fixed epochs given in advance (EVAL_EPOCHS: 19 = median best
epoch of the 5 D-019 members, chosen on USA, so the USA fold is marked; 60 = last epoch);
validation and test chips are never requested (only split parts "train" and "dev" are read).
models/ is only read (baseline rule), never written. This estimates the transfer of the
recipe, not of the shipped weights (those were trained on all 5 train events, selected on USA).
Training runs in Colab only (D-036): `notebooks/loeo_colab.ipynb`.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from floodvalue.data import io
from floodvalue.model import metrics as M
from floodvalue.model.unet import data as D
from floodvalue.model.unet.features import channel_stats, make_features
from floodvalue.model.unet.predict import build_model
from floodvalue.model.unet.report import paired_bootstrap
from floodvalue.model.unet.train import (
    DEFAULTS,
    augment,
    jitter_std,
    masked_loss,
    predict_probs,
    seed_everything,
)
from floodvalue.paths import MODELS_DIR, output_dir

# Only these split parts are ever read here: validation and test stay untouched.
OPEN_PARTS = ("train", "dev")
# Threshold grid of the baseline rule (same as D-008): dB, step 0.1.
EDGES = np.round(np.arange(-60.0, 30.0 + 1e-9, 0.1), 1)
# Fixed scoring epochs of the 60-epoch D-019 schedule: best epochs of the shipped members were
# 18, 28, 19, 21, 14 (models/unet/history_member_*.csv) -> median 19; and the last epoch.
EVAL_EPOCHS = (19, 60)
VH = 1
# Score bins for threshold-free metrics: U-Net p in 0.01 steps; baseline -VH on the dB grid.
P_EDGES = np.round(np.arange(0.0, 1.0 + 1e-9, 0.01), 2)

Reader = Callable[[str], tuple[np.ndarray, np.ndarray]]


# D-016: weak data (S1Weak + S2IndexLabelWeak) of dev, validation, test events and Colombia are
# never used; only these train events may contribute weakly-labeled chips.
WEAK_EVENTS = ("Ghana", "India", "Pakistan", "Paraguay", "Somalia")
WEAK_LABEL = "S2IndexLabelWeak"


def weak_chip_ids(events: set[str]) -> list[str]:
    """Weakly-labeled chips (label from the Sentinel-2 index, an independent sensor) of the
    given fold-training events, restricted to WEAK_EVENTS; sorted chip ids."""
    root = io.raw_dir() / "WeaklyLabeled" / WEAK_LABEL
    ids = sorted(p.name[: -len(f"_{WEAK_LABEL}.tif")] for p in root.glob(f"*_{WEAK_LABEL}.tif"))
    return [c for c in ids if D.event_of(c) in events and D.event_of(c) in WEAK_EVENTS]


def read_weak_chip(chip_id: str) -> tuple[np.ndarray, np.ndarray]:
    """S1Weak (2, H, W) float32 dB with NaN and the weak label mapped to the LabelHand codes
    (1 water, 0 not water, -1 no label)."""
    import rasterio

    root = io.raw_dir() / "WeaklyLabeled"
    with rasterio.open(root / "S1Weak" / f"{chip_id}_S1Weak.tif") as f:
        s1 = f.read().astype(np.float32)
    if s1.shape[0] != 2:
        raise ValueError(f"{chip_id}: S1Weak must have 2 bands (VV, VH), got {s1.shape[0]}")
    with rasterio.open(root / WEAK_LABEL / f"{chip_id}_{WEAK_LABEL}.tif") as f:
        raw = f.read(1)
    lab = np.full(raw.shape, -1, np.int16)
    lab[raw == 0] = 0
    lab[raw == 1] = 1
    return s1, lab


def sample_batch(gen: torch.Generator, batch: int, hand: torch.Tensor, weak: torch.Tensor,
                 hand_frac: float | None) -> torch.Tensor:
    """Row indices of one batch. hand_frac None: uniform over all rows (D-019 / plain weak
    mixing). Else round(batch * hand_frac) rows from `hand`, the rest from `weak`."""
    if hand_frac is None:
        return torch.randint(0, len(hand) + len(weak), (batch,), generator=gen)
    n_h = int(round(batch * hand_frac))
    parts = [hand[torch.randint(0, len(hand), (n_h,), generator=gen)],
             weak[torch.randint(0, len(weak), (batch - n_h,), generator=gen)]]
    return torch.cat(parts)


def pool_chip_ids() -> list[str]:
    """Chips of train+dev (the LOEO pool); never validation or test."""
    ids: list[str] = []
    for part in OPEN_PARTS:
        ids += D.split_chip_ids(part)
    return ids


def folds(chip_ids: list[str]) -> list[dict]:
    """One fold per event: held-out event's chips vs chips of all other events."""
    events = sorted({D.event_of(c) for c in chip_ids})
    out = []
    for ev in events:
        held = [c for c in chip_ids if D.event_of(c) == ev]
        train = [c for c in chip_ids if D.event_of(c) != ev]
        out.append({"event": ev, "train": train, "held_out": held})
    return out


def read_chip(chip_id: str) -> tuple[np.ndarray, np.ndarray]:
    return D.read_s1(chip_id), D.read_label(chip_id)


def predict_probs_tta(model, x: torch.Tensor, device: str) -> np.ndarray:
    """Mean p over the 8 dihedral transforms (rot90 x flip), each mapped back to the input grid.
    Test-time augmentation only; the model and its weights are the same."""
    acc = None
    for k in range(4):
        for flip in (False, True):
            xt = torch.rot90(x, k, dims=(-2, -1))
            if flip:
                xt = torch.flip(xt, dims=(-1,))
            p = torch.from_numpy(predict_probs(model, xt, device))
            if flip:
                p = torch.flip(p, dims=(-1,))
            p = torch.rot90(p, -k, dims=(-2, -1)).numpy()
            acc = p if acc is None else acc + p
    return acc / 8.0


def fold_stats(ids: list[str], read: Reader, channels: list[str]) -> tuple[list, list]:
    """Normalization mean/std over the given (fold training) chips only."""
    return channel_stats([read(c)[0] for c in ids], channels)


def build_tensors(ids: list[str], read: Reader, channels: list[str], mean, std):
    """Features, water target and valid mask (`data.io.valid_mask`), chip by chip."""
    s1, _ = read(ids[0])
    h, w = s1.shape[1:]
    x = torch.empty((len(ids), len(channels) + 1, h, w), dtype=torch.float32)
    y = torch.empty((len(ids), h, w), dtype=torch.uint8)
    m = torch.empty((len(ids), h, w), dtype=torch.uint8)
    for i, c in enumerate(ids):
        s1, lab = read(c)
        x[i] = torch.from_numpy(make_features(s1, channels, mean, std))
        y[i] = torch.from_numpy((lab == 1).astype(np.uint8))
        m[i] = torch.from_numpy(io.valid_mask(s1, lab).astype(np.uint8))
    return x, y, m


def counts_le(values: np.ndarray) -> np.ndarray:
    v = np.sort(values[np.isfinite(values)])
    return np.searchsorted(v, EDGES, side="right").astype(np.int64)


def fit_vh_threshold(chips: list[tuple[np.ndarray, np.ndarray]]) -> tuple[float, float]:
    """Pooled-IoU-optimal t for `VH <= t` over valid pixels (lowest t on ties) and its IoU."""
    water = np.zeros(len(EDGES), np.int64)
    dry = np.zeros(len(EDGES), np.int64)
    n_water = 0
    for s1, lab in chips:
        ok = io.valid_mask(s1, lab)
        vh = s1[VH][ok]
        is_w = lab[ok] == 1
        water += counts_le(vh[is_w])
        dry += counts_le(vh[~is_w])
        n_water += int(is_w.sum())
    den = (n_water + dry).astype(np.float64)
    iou = np.divide(water, den, out=np.zeros(len(EDGES)), where=den > 0)
    k = int(np.argmax(iou))
    return float(EDGES[k]), float(iou[k])


def score_hist(score: np.ndarray, target: np.ndarray, valid: np.ndarray,
               edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Counts of water / dry valid pixels per score bin (bins by `edges`, right-closed)."""
    k = np.searchsorted(edges, score[valid], side="left")
    t = target[valid].astype(bool)
    n = len(edges) + 1
    return np.bincount(k[t], minlength=n), np.bincount(k[~t], minlength=n)


def ranking_metrics(water: np.ndarray, dry: np.ndarray, edges: np.ndarray) -> dict:
    """Threshold-free summary from score histograms (higher bin = more water-like):
    average precision, and the best pooled IoU over bin thresholds (an *oracle* diagnostic
    that looks at held-out labels; never used for any choice). `oracle_score_gt`: the best
    rule is score > this value (bins from `score_hist` with the same edges)."""
    w = np.cumsum(np.asarray(water, np.int64)[::-1])
    d = np.cumsum(np.asarray(dry, np.int64)[::-1])
    n_w = int(w[-1]) if len(w) else 0
    if n_w == 0:
        return {"ap": float("nan"), "oracle_iou": float("nan"), "oracle_score_gt": float("nan")}
    prec = np.divide(w, w + d, out=np.zeros(len(w)), where=(w + d) > 0)
    rec_step = np.diff(w, prepend=0) / n_w
    iou = w / (n_w + d)
    k = int(np.argmax(iou))
    b = len(w) - 1 - k  # lowest bin included in the best rule
    return {"ap": float((rec_step * prec).sum()), "oracle_iou": float(iou[k]),
            "oracle_score_gt": float(edges[b - 1]) if b > 0 else float("-inf")}


def baseline_pred(s1: np.ndarray, t: float) -> np.ndarray:
    """p >= 0.5 of the baseline <=> VH <= t (invalid pixels are masked separately)."""
    with np.errstate(invalid="ignore"):
        return s1[VH] <= t


def shipped_baseline_rule(path: Path | None = None) -> dict:
    cfg = json.loads((path or MODELS_DIR / "baseline" / "config.json").read_text("utf-8"))
    if cfg["channel"] != "VH":
        raise ValueError(f"LOEO baseline assumes the VH rule, got {cfg['channel']}")
    return {"threshold_db": float(cfg["threshold_db"]), "fitted_on": cfg.get("fitted_on")}


def train_fold(x, y, m, cfg: dict, eval_epochs=EVAL_EPOCHS, device: str = "cpu", log=print):
    """One U-Net on the fold's training tensors (D-019 loop). Returns {epoch: state_dict} at
    the fixed `eval_epochs` (no selection) and steps per epoch."""
    seed_everything(cfg["seed"])
    channels = cfg["channels"]
    config = {"arch": {"base": cfg["base"], "dropout": cfg["dropout"]},
              "features": {"channels": channels}}
    model = build_model(config).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    steps = max(1, (2 * len(x) * cfg.get("steps_mult", 1)) // cfg["batch"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], pct_start=0.1,
                                                total_steps=cfg["epochs"] * steps)
    gen = torch.Generator().manual_seed(cfg["seed"])
    jstd = jitter_std(channels, cfg["std"]).to(device)
    weak_w = cfg.get("weak_weight")
    hand_frac = cfg.get("hand_frac") if weak_w is not None else None
    if weak_w is None:  # D-019 path: everything on the device, boolean mask
        x, y, m = x.to(device), y.to(device), m.to(device)
        rows_hand, rows_weak = torch.arange(len(x)), torch.arange(0)
    else:  # mask codes: 2 hand label, 1 weak label (rows are hand chips first, then weak)
        top = m.flatten(1).amax(1)
        rows_hand, rows_weak = (top == 2).nonzero().flatten(), (top == 1).nonzero().flatten()
        if len(rows_hand) == 0:
            raise ValueError("no hand-labeled chip with valid pixels in this fold")
    amp = device.startswith("cuda")
    states = {}
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        te, losses = time.time(), []
        for _ in range(steps):
            if hand_frac is None:
                idx = torch.randint(0, len(x), (cfg["batch"],), generator=gen)
            else:
                idx = sample_batch(gen, cfg["batch"], rows_hand, rows_weak, hand_frac)
            if weak_w is None:
                idx = idx.to(device)
                xs, ys, ms = x[idx], y[idx], m[idx]
            else:  # data stay in host memory; only the batch goes to the device
                xs, ys, ms = (t[idx].to(device) for t in (x, y, m))
            xb, yb, mb = augment(xs, ys, ms, cfg["crop"], cfg["db_jitter"], jstd,
                                 len(channels), gen)
            with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16,
                                enabled=amp):
                logits = model(xb)
            if weak_w is None:
                weight = mb.bool()
            else:  # mask codes: 2 hand label, 1 weak label, 0 excluded
                weight = torch.where(mb == 2, 1.0, torch.where(mb == 1, float(weak_w), 0.0))
            loss = masked_loss(logits, yb, weight, cfg["dice_weight"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
        if epoch in eval_epochs:
            states[epoch] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        log(f"  epoch {epoch:3d}/{cfg['epochs']} loss {np.mean(losses):.4f} "
            f"{time.time() - te:.1f}s")
    missing = set(eval_epochs) - set(states)
    if missing:
        raise ValueError(f"eval epochs {sorted(missing)} exceed --epochs {cfg['epochs']}")
    ft = cfg.get("finetune_epochs") or 0
    if ft:
        # Stage 2: fine-tune on the hand-labeled rows only (mask code 2 when weak data are mixed
        # in, else every row), fresh OneCycle at lr * finetune_lr_mult. Scored as "ft" — fixed
        # in advance, no selection.
        hand = rows_hand
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"] * cfg["finetune_lr_mult"],
                                weight_decay=cfg["weight_decay"])
        steps_ft = max(1, (2 * len(hand)) // cfg["batch"])
        if ft * steps_ft < 10:
            log(f"  WARNING: fine-tune has only {ft * steps_ft} steps — OneCycle barely trains")
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=cfg["lr"] * cfg["finetune_lr_mult"], pct_start=0.1,
            total_steps=ft * steps_ft)
        for epoch in range(1, ft + 1):
            model.train()
            te, losses = time.time(), []
            for _ in range(steps_ft):
                idx = hand[torch.randint(0, len(hand), (cfg["batch"],), generator=gen)]
                if weak_w is None:
                    idx = idx.to(device)
                    xs, ys, ms = x[idx], y[idx], m[idx]
                else:
                    xs, ys, ms = (t[idx].to(device) for t in (x, y, m))
                xb, yb, mb = augment(xs, ys, ms, cfg["crop"], cfg["db_jitter"], jstd,
                                     len(channels), gen)
                with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16,
                                    enabled=amp):
                    logits = model(xb)
                loss = masked_loss(logits, yb, mb.bool(), cfg["dice_weight"])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                losses.append(loss.item())
            log(f"  finetune {epoch:3d}/{ft} loss {np.mean(losses):.4f} {time.time() - te:.1f}s")
        states["ft"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return model, states, steps


def run_fold(fold: dict, cfg: dict, read: Reader = read_chip, baseline_t: float | None = None,
             eval_epochs=EVAL_EPOCHS, device: str = "cpu", log=print) -> dict:
    """Train on fold["train"], score U-Net (at each fixed epoch) and baselines on
    fold["held_out"] over the same valid pixels. Held-out chips are read only after training."""
    if set(fold["train"]) & set(fold["held_out"]):
        raise ValueError("fold train and held-out chips overlap")
    channels = cfg["channels"]
    t0 = time.time()
    mean, std = fold_stats(fold["train"], read, channels)
    t_fold, t_fold_iou = fit_vh_threshold([read(c) for c in fold["train"]])
    x, y, m = build_tensors(fold["train"], read, channels, mean, std)
    n_weak = 0
    if cfg.get("weak_weight") is not None:
        wids = weak_chip_ids({D.event_of(c) for c in fold["train"]})
        weak_events = {D.event_of(c) for c in wids}
        held_events = {D.event_of(c) for c in fold["held_out"]}  # independent of fold["event"]
        if weak_events & held_events or not weak_events <= set(WEAK_EVENTS):
            raise ValueError(f"weak events {sorted(weak_events)} overlap held-out "
                             f"{sorted(held_events)} or leave {WEAK_EVENTS}")
        xw, yw, mw = build_tensors(wids, read_weak_chip, channels, mean, std)
        x, y, m = torch.cat([x, xw]), torch.cat([y, yw]), torch.cat([m * 2, mw])
        n_weak = len(wids)
        del xw, yw, mw
    log(f"fold {fold['event']}: train {len(fold['train'])} chips + weak {n_weak}, held-out "
        f"{len(fold['held_out'])}; mean {np.round(mean, 2)} std {np.round(std, 2)}; "
        f"VH t {t_fold} dB; loaded {time.time() - t0:.0f}s")
    model, states, steps = train_fold(x, y, m, cfg | {"std": std}, eval_epochs, device, log)
    del x, y, m
    xh, yh, mh = build_tensors(fold["held_out"], read, channels, mean, std)
    probs = {}
    for epoch, state in states.items():
        model.load_state_dict(state)
        probs[f"unet_e{epoch}"] = predict_probs(model, xh, device)
        if cfg.get("tta"):
            probs[f"unet_e{epoch}_tta"] = predict_probs_tta(model, xh, device)
    b_edges = -EDGES[::-1]  # baseline score = -VH (more water-like = higher)
    edges = {k: P_EDGES for k in probs} | {"baseline": b_edges}
    hist = {k: [np.zeros(len(e) + 1, np.int64) for _ in range(2)] for k, e in edges.items()}
    rows = []
    for i, cid in enumerate(fold["held_out"]):
        s1, _ = read(cid)
        valid = mh[i].numpy().astype(bool)
        target = yh[i].numpy().astype(bool)
        row = {"event": fold["event"], "chip_id": cid,
               "water_pixels": int(target[valid].sum()), "valid_pixels": int(valid.sum())}
        preds = {k: p[i] >= 0.5 for k, p in probs.items()}
        preds["baseline_fold"] = baseline_pred(s1, t_fold)
        if baseline_t is not None:
            preds["baseline_shipped"] = baseline_pred(s1, baseline_t)
        for name, pred in preds.items():
            row |= {f"{name}_{k}": v for k, v in M.confusion(pred, target, valid).items()}
        rows.append(row)
        scores = {k: p[i] for k, p in probs.items()} | {"baseline": -s1[VH].astype(np.float64)}
        for name, score in scores.items():
            wh, dh = score_hist(score, target, valid, edges[name])
            hist[name][0] += wh
            hist[name][1] += dh
    mask_hash = M.mask_sha256([mh[i].numpy().astype(bool)
                               for i in np.argsort(fold["held_out"], kind="stable")])
    return {"event": fold["event"], "rows": rows, "valid_mask_sha256": mask_hash,
            "methods": list(preds), "mean": mean, "std": std, "baseline_fold_t": t_fold,
            "baseline_fold_train_iou": t_fold_iou, "steps_per_epoch": steps,
            "seconds": round(time.time() - t0),
            "ranking": {k: ranking_metrics(*v, edges[k]) for k, v in hist.items()}}


def per_chip(rows: list[dict], name: str) -> list[dict]:
    return [{k: r[f"{name}_{k}"] for k in ("tp", "fp", "fn", "tn")} for r in rows]


def fold_metrics(res: dict, methods: list[str], reference: str = "baseline_fold") -> list[dict]:
    """Pooled metrics of one held-out event per method (one valid-pixel hash for all), chip
    bootstrap CI of IoU and the paired chip bootstrap of IoU(method) − IoU(reference)."""
    out = []
    for name in methods:
        chips = per_chip(res["rows"], name)
        c = {k: sum(ch[k] for ch in chips) for k in ("tp", "fp", "fn", "tn")}
        lo, hi = M.bootstrap_iou(chips, seed=0)
        row = {"method": name, "split": "loeo", "event": res["event"], **M.scores(c), **c,
               "threshold": 0.5, "valid_mask_sha256": res["valid_mask_sha256"],
               "iou_ci95_lo": lo, "iou_ci95_hi": hi, "delta_iou_vs": "",
               "delta_iou": float("nan"), "delta_ci95_lo": float("nan"),
               "delta_ci95_hi": float("nan")}
        if name != reference and reference in methods:
            pb = paired_bootstrap(chips, per_chip(res["rows"], reference))
            row |= {"delta_iou_vs": reference, "delta_iou": pb["diff"],
                    "delta_ci95_lo": pb["ci95"][0], "delta_ci95_hi": pb["ci95"][1]}
        out.append(row)
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if isinstance(v, float) and not np.isfinite(v) else v)
                        for k, v in r.items()})


def _num(v: str, nd: int = 3) -> str:
    return f"{float(v):.{nd}f}".replace(".", ",") if v not in ("", None) else "—"


def markdown_table(rows: list[dict]) -> str:
    """Report table from loeo-metrics.csv rows: IoU [CI], F1, P, R per event and method."""
    lines = [("| Отложенное событие | Метод | IoU [95% CI] | F1 | Precision | Recall | "
              "ΔIoU к baseline фолда [95% CI] | Валидных пикселей |"),
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        delta = "—" if not r.get("delta_iou_vs") else (
            f"{_num(r['delta_iou'])} [{_num(r['delta_ci95_lo'])}; {_num(r['delta_ci95_hi'])}]")
        # USA*: D-019 hyperparameters, the VH channel and epoch 19 were chosen on USA (dev).
        event = r["event"] + ("*" if r["event"] == "USA" else "")
        lines.append(f"| {event} | {r['method']} | {_num(r['iou'])} "
                     f"[{_num(r['iou_ci95_lo'])}; {_num(r['iou_ci95_hi'])}] | {_num(r['f1'])} | "
                     f"{_num(r['precision'])} | {_num(r['recall'])} | {delta} | "
                     f"{r['n_pixels']} |")
    lines.append("\n\\* USA — dev D-019: гиперпараметры, канал VH и эпоха 19 выбраны на нём, "
                 "фолд не полностью независим.")
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="floodvalue.model.unet.loeo")
    p.add_argument("--out-dir", default=None, help="default $FLOODVALUE_OUTPUT_DIR/loeo")
    p.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    p.add_argument("--eval-epochs", default=",".join(map(str, EVAL_EPOCHS)))
    p.add_argument("--crop", type=int, default=DEFAULTS["crop"])
    p.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    for key in ("base", "batch"):
        p.add_argument(f"--{key}", type=int, default=DEFAULTS[key])
    for key in ("dropout", "lr", "weight_decay", "dice_weight", "db_jitter"):
        p.add_argument(f"--{key.replace('_', '-')}", type=float, default=DEFAULTS[key])
    p.add_argument("--steps-mult", type=int, default=1,
                   help="multiply optimizer steps per epoch (D-019: 2 x chips / batch)")
    p.add_argument("--weak-weight", type=float, default=None,
                   help=f"add weakly-labeled chips ({WEAK_LABEL}) of the fold's training events "
                        f"among {', '.join(WEAK_EVENTS)} with this loss weight (hand = 1)")
    p.add_argument("--hand-frac", type=float, default=None,
                   help="with --weak-weight: fraction of each batch drawn from hand-labeled chips")
    p.add_argument("--finetune-epochs", type=int, default=0,
                   help="stage 2: fine-tune on hand-labeled chips only for N epochs (scored as unet_eft)")
    p.add_argument("--finetune-lr-mult", type=float, default=0.3)
    p.add_argument("--tta", action="store_true",
                   help="also score the mean of 8 rot90/flip predictions at each eval epoch")
    p.add_argument("--events", default="", help="comma list: run only these folds (smoke)")
    p.add_argument("--max-chips", type=int, default=0,
                   help="smoke: limit chips per event (numbers are then not a result)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--table-from", default=None,
                   help="print the markdown table of <dir>/loeo-metrics.csv and exit")
    return p.parse_args(argv)


def replay_command(args: argparse.Namespace) -> str:
    """Command line that re-runs exactly this configuration (flags without values for booleans;
    0 and 0.0 kept — `--weak-weight 0.0` is not the same run as no weak data)."""
    parts = []
    for k, v in vars(args).items():
        if k in ("out_dir", "device", "table_from") or v is None or v is False or v == "":
            continue
        flag = f"--{k.replace('_', '-')}"
        parts.append(flag if v is True else f"{flag} {v}")
    return "python -m floodvalue.model.unet.loeo " + " ".join(parts)


def git_commit() -> str:
    """Commit of the code that produced the numbers (FLOODVALUE_COMMIT or `git rev-parse`)."""
    import os
    import subprocess

    if os.environ.get("FLOODVALUE_COMMIT"):
        return os.environ["FLOODVALUE_COMMIT"]
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       cwd=Path(__file__).parent).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def limit_per_event(ids: list[str], n: int) -> list[str]:
    if not n:
        return ids
    seen: dict[str, int] = {}
    out = []
    for c in ids:
        e = D.event_of(c)
        if seen.get(e, 0) < n:
            out.append(c)
            seen[e] = seen.get(e, 0) + 1
    return out


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.table_from:
        with open(Path(args.table_from) / "loeo-metrics.csv", encoding="utf-8") as f:
            print(markdown_table(list(csv.DictReader(f))), end="")
        return 0
    out = Path(args.out_dir) if args.out_dir else output_dir() / "loeo"
    out.mkdir(parents=True, exist_ok=True)
    eval_epochs = tuple(int(e) for e in args.eval_epochs.split(",") if e.strip())
    cfg = {"channels": DEFAULTS["channels"]}
    cfg |= {k: getattr(args, k) for k in ("base", "dropout", "batch", "lr", "weight_decay",
                                          "dice_weight", "db_jitter")}
    cfg |= {"epochs": args.epochs, "crop": args.crop, "seed": args.seed,
            "steps_mult": args.steps_mult, "tta": args.tta, "weak_weight": args.weak_weight,
            "finetune_epochs": args.finetune_epochs, "finetune_lr_mult": args.finetune_lr_mult,
            "hand_frac": args.hand_frac}
    shipped = shipped_baseline_rule()
    only = {e.strip() for e in args.events.split(",") if e.strip()}
    command = replay_command(args)
    t_all = time.time()
    all_rows, chip_rows, info = [], [], []
    for fold in folds(limit_per_event(pool_chip_ids(), args.max_chips)):
        if only and fold["event"] not in only:
            continue
        res = run_fold(fold, cfg, baseline_t=shipped["threshold_db"], eval_epochs=eval_epochs,
                       device=args.device)
        rows = fold_metrics(res, res["methods"])
        all_rows += rows
        chip_rows += res["rows"]
        info.append({k: res[k] for k in ("event", "mean", "std", "baseline_fold_t",
                                         "baseline_fold_train_iou", "steps_per_epoch",
                                         "seconds", "valid_mask_sha256", "ranking")}
                    | {"train_chips": len(fold["train"]), "held_out_chips": len(fold["held_out"])})
        for r in rows:
            print(f"  {r['event']:10s} {r['method']:17s} IoU {r['iou']:.4f} F1 {r['f1']:.4f} "
                  f"P {r['precision']:.3f} R {r['recall']:.3f}", flush=True)
        write_csv(out / "loeo-metrics.csv", all_rows)
        write_csv(out / "loeo-chips.csv", chip_rows)
        summary = {"config": cfg, "eval_epochs": list(eval_epochs), "max_chips": args.max_chips,
                   "baseline_shipped": shipped, "folds": info,
                   "seconds": round(time.time() - t_all), "split_sha256": D.split_sha256(),
                   "device": (torch.cuda.get_device_name(0) if args.device.startswith("cuda")
                              else "cpu"),
                   "versions": {"python": platform.python_version(), "torch": torch.__version__,
                                "numpy": np.__version__},
                   "command": command, "commit": git_commit()}
        (out / "loeo-summary.json").write_text(json.dumps(summary, indent=2) + "\n", "utf-8")
    print(f"done in {round(time.time() - t_all)}s -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
