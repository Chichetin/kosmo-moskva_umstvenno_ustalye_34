"""Assemble a seed ensemble of trained U-Net runs into `models/unet/` and evaluate it.

    python -m floodvalue.model.unet.ensemble --members runs/a,runs/b,runs/c

Each member is the output dir of `floodvalue.model.unet.train` (same features and
architecture, different seed). p of the ensemble = mean of member probabilities. Metrics of
the ensemble are computed on dev (selection, D-012) and validation (report only); test is
never read.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

from floodvalue.model import metrics as M
from floodvalue.model.unet import train as T
from floodvalue.model.unet.predict import (CONFIG_FILE, WEIGHTS_FILE, build_model, file_sha256,
                                           members_sha256)
from floodvalue.paths import MODELS_DIR


def ensemble_probs(models: list, x: torch.Tensor, device: str) -> np.ndarray:
    return np.mean([T.predict_probs(m, x, device) for m in models], axis=0)


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(prog="floodvalue.model.unet.ensemble")
    ap.add_argument("--members", required=True, help="comma-separated train output dirs")
    ap.add_argument("--name", default="unet")
    ap.add_argument("--out-dir", default=str(MODELS_DIR / "unet"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    dirs = [Path(p) for p in args.members.split(",") if p]
    configs = [json.loads((d / CONFIG_FILE).read_text(encoding="utf-8")) for d in dirs]
    ref = configs[0]
    for c in configs[1:]:
        if c["features"] != ref["features"] or c["arch"] != ref["arch"]:
            raise ValueError("ensemble members must share features and architecture")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    members, models = [], []
    for i, (d, c) in enumerate(zip(dirs, configs)):
        name = f"member_{i}.pt"
        shutil.copyfile(d / WEIGHTS_FILE, out / name)
        shutil.copyfile(d / "history.csv", out / f"history_member_{i}.csv")
        members.append({"file": name, "sha256": file_sha256(out / name),
                        "seed": c["train"]["seed"], "epoch": c["selection"]["epoch"],
                        "dev_iou_single": c["selection"]["dev_iou"], "run": c["name"]})
        m = build_model(ref)
        m.load_state_dict(torch.load(out / name, map_location="cpu", weights_only=True))
        models.append(m.to(args.device).eval())

    feats = ref["features"]
    ev, probs, data = {}, {}, {}
    for part in ("dev", "validation"):
        prep = T.prepare(part, {})
        x, y, m = T.tensors(prep["raw"], feats)
        probs[part] = ensemble_probs(models, x, args.device)
        ev[part] = T.evaluate(probs[part], prep["ids"], y, m)
        data[part] = (y, m)
    rows = T.metric_rows(args.name, "dev", ev["dev"], 0.5) + \
        T.metric_rows(args.name, "validation", ev["validation"], 0.5)

    config = {k: v for k, v in ref.items()
              if k not in ("weights_file", "weights_sha256", "selection", "name", "command")}
    config.update({
        "name": args.name,
        "members": members,
        "weights_sha256": members_sha256(members),
        "weights_sha256_note": "sha256 of 'file:sha256' lines of members (each member's "
                               "sha256 is checked on load)",
        "ensemble": "mean of member probabilities (seed ensemble)",
        "selection": {"part": "dev", "metric": "pooled IoU at p >= 0.5; per member best epoch",
                      "dev_iou": M.scores(ev["dev"]["all"])["iou"]},
        "command": "python -m floodvalue.model.unet.ensemble " + " ".join(sys.argv[1:]),
        "member_commands": [c["command"] for c in configs],
    })
    (out / CONFIG_FILE).write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n",
                                   encoding="utf-8")
    with open(out / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=M.METRIC_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if isinstance(r[k], float) and not np.isfinite(r[k]) else r[k])
                        for k in M.METRIC_COLUMNS})
    summary = {
        "name": args.name, "members": len(members),
        "iou_ci95": {p: M.bootstrap_iou(ev[p]["per_chip"], seed=0) for p in ev},
        "dev_best_threshold_iou": T.best_threshold_iou(probs["dev"], *data["dev"]),
        "validation_best_threshold_iou_diagnostic":
            T.best_threshold_iou(probs["validation"], *data["validation"]),
        "weights_sha256": config["weights_sha256"],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
