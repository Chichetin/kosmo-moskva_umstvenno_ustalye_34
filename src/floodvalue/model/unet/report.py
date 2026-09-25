"""Numbers of docs/verification/unet-metrics.md that are not written by train/ensemble.

    python -m floodvalue.model.unet.report [--runs <dir with variant runs>]

1. Paired chip bootstrap of IoU(U-Net) − IoU(baseline) on dev and validation: both methods
   through their `predict_proba`, p >= 0.5, one valid mask per chip (S1 finite, label 0/1).
2. With --runs: docs/verification/unet-variants.csv (one row per training run).
3. Learning curves of the shipped members: docs/verification/unet-learning-curves.png.
Only train/dev/validation parts are read; test is sealed (D-010).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from floodvalue.model import metrics as M
from floodvalue.model.unet import data as D
from floodvalue.model.unet.features import loss_mask
from floodvalue.paths import MODELS_DIR, REPO_ROOT

VERIFICATION = REPO_ROOT / "docs" / "verification"


def paired_bootstrap(a: list[dict], b: list[dict], n: int = 2000, seed: int = 0) -> dict:
    """Pooled IoU of a and b and the percentile CI of (a − b) over the same chip resamples."""
    arr = [np.array([[c["tp"], c["fp"], c["fn"]] for c in x], float) for x in (a, b)]
    idx = np.random.default_rng(seed).integers(0, len(arr[0]), (n, len(arr[0])))

    def iou(s):
        return s[..., 0] / np.maximum(s.sum(-1), 1)

    diff = iou(arr[0][idx].sum(1)) - iou(arr[1][idx].sum(1))
    lo, hi = np.percentile(diff, [2.5, 97.5])
    return {"iou_a": float(iou(arr[0].sum(0))), "iou_b": float(iou(arr[1].sum(0))),
            "diff": float(iou(arr[0].sum(0)) - iou(arr[1].sum(0))),
            "ci95": [float(lo), float(hi)], "p_diff_le_0": float((diff <= 0).mean())}


def compare_with_baseline() -> dict:
    from floodvalue.baseline.predict import predict_proba as baseline_proba
    from floodvalue.model.unet.predict import predict_proba as unet_proba

    out = {}
    for part in ("dev", "validation"):
        u, b = [], []
        for chip in sorted(D.split_chip_ids(part)):
            s1, label = D.read_s1(chip), D.read_label(chip)
            m = loss_mask(s1, label)
            for probs, acc in ((unet_proba(s1), u), (baseline_proba(s1), b)):
                acc.append(M.confusion(np.nan_to_num(probs) >= 0.5, label == 1, m))
        out[part] = paired_bootstrap(u, b)
    return out


def variants_table(runs: Path, path: Path) -> None:
    cols = ["run", "channels", "base", "dice_weight", "db_jitter", "seed", "best_epoch",
            "dev_iou", "dev_iou_ci_lo", "dev_iou_ci_hi", "validation_iou", "nigeria_iou",
            "sri_lanka_iou", "validation_precision", "validation_recall", "train_seconds"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for d in sorted(p for p in runs.iterdir() if (p / "summary.json").exists()):
            c = json.loads((d / "config.json").read_text(encoding="utf-8"))
            s = json.loads((d / "summary.json").read_text(encoding="utf-8"))
            with open(d / "metrics.csv", newline="", encoding="utf-8") as fm:
                rows = {(r["split"], r["event"]): r for r in csv.DictReader(fm)}
            t = c["train"]

            def g(key, field="iou", rows=rows):
                return round(float(rows[key][field]), 4)

            w.writerow([d.name, "+".join(c["features"]["channels"]), c["arch"]["base"],
                        t["dice_weight"], t["db_jitter"], t["seed"], s["best_epoch"],
                        g(("dev", "ALL")), round(s["iou_ci95"]["dev"][0], 4),
                        round(s["iou_ci95"]["dev"][1], 4), g(("validation", "ALL")),
                        g(("validation", "Nigeria")), g(("validation", "Sri-Lanka")),
                        g(("validation", "ALL"), "precision"),
                        g(("validation", "ALL"), "recall"), t["seconds"]])


def learning_curves(model_dir: Path, path: Path, baseline_dev_iou: float = 0.390798) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for f in sorted(model_dir.glob("history_member_*.csv")):
        with open(f, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        e = [int(r["epoch"]) for r in rows]
        ax[0].plot(e, [float(r["train_loss"]) for r in rows], lw=1, label=f.stem[8:])
        ax[1].plot(e, [float(r["dev_iou"]) for r in rows], lw=1)
        k = max(range(len(rows)), key=lambda i: float(rows[i]["dev_iou"]))
        ax[1].plot(e[k], float(rows[k]["dev_iou"]), "k.", ms=8)
    ax[1].axhline(baseline_dev_iou, color="gray", ls="--", lw=1, label="baseline VH ≤ −23 dB")
    ax[0].set(title="train loss (BCE + 0,5·Dice)", xlabel="эпоха")
    ax[1].set(title="dev (USA) IoU при p ≥ 0,5; точка — выбранный чекпойнт", xlabel="эпоха")
    ax[0].legend(fontsize=7)
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(prog="floodvalue.model.unet.report")
    ap.add_argument("--runs", help="dir with variant runs -> unet-variants.csv")
    args = ap.parse_args(argv)
    if args.runs:
        variants_table(Path(args.runs), VERIFICATION / "unet-variants.csv")
    learning_curves(MODELS_DIR / "unet", VERIFICATION / "unet-learning-curves.png")
    result = compare_with_baseline()
    (VERIFICATION / "unet-vs-baseline-bootstrap.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
