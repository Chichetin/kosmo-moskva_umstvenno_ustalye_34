"""Error analysis of the LOEO run from its committed files (#126).

python scripts/loeo_error_analysis.py docs/verification/loeo

Reads only loeo-summary.json and loeo-chips.csv (train/dev events; validation and test
are not part of LOEO). Prints markdown tables: ranking quality per fold (AP, oracle IoU)
and how concentrated the errors of U-Net e19 are over chips, plus IoU by water share.
"""
import csv
import json
import sys
from pathlib import Path


def fmt(x: float) -> str:
    return f"{x:.3f}".replace(".", ",")


def main(d: Path) -> None:
    summary = json.loads((d / "loeo-summary.json").read_text(encoding="utf-8"))
    with open(d / "loeo-chips.csv", encoding="utf-8", newline="") as f:
        chips = list(csv.DictReader(f))

    print("| Событие | AP e19 | AP e60 | AP baseline | IoU e19 при p ≥ 0,5 | oracle-IoU e19 (порог) |")
    print("|---|---|---|---|---|---|")
    for fold in summary["folds"]:
        ev, r = fold["event"], fold["ranking"]
        rows = [c for c in chips if c["event"] == ev]
        tp, fp, fn = (sum(int(c[f"unet_e19_{k}"]) for c in rows) for k in ("tp", "fp", "fn"))
        iou = tp / (tp + fp + fn) if tp + fp + fn else float("nan")
        print(f"| {ev} | {fmt(r['unet_e19']['ap'])} | {fmt(r['unet_e60']['ap'])} | "
              f"{fmt(r['baseline']['ap'])} | {fmt(iou)} | {fmt(r['unet_e19']['oracle_iou'])} "
              f"({fmt(r['unet_e19']['oracle_score_gt'])}) |")

    print()
    print("| Событие | чипов | ошибок e19 (FP+FN) | чипов на ≥ 50 % ошибок | два худших чипа (доля ошибок) |")
    print("|---|---|---|---|---|")
    for ev in [f["event"] for f in summary["folds"]]:
        rows = [c for c in chips if c["event"] == ev]
        err = sorted(((int(c["unet_e19_fp"]) + int(c["unet_e19_fn"]), c["chip_id"]) for c in rows),
                     reverse=True)
        total = sum(e for e, _ in err)
        acc, k = 0, 0
        for e, _ in err:
            acc += e
            k += 1
            if acc >= total / 2:
                break
        worst = ", ".join(f"{cid} ({fmt(e / total)})" for e, cid in err[:2]) if total else "—"
        print(f"| {ev} | {len(rows)} | {total} | {k} | {worst} |")

    print()
    bins = [(0, 0.01), (0.01, 0.05), (0.05, 0.15), (0.15, 0.30), (0.30, 1.01)]
    print("| Доля воды в чипе | чипов | pooled IoU e19 | pooled IoU baseline фолда |")
    print("|---|---|---|---|")
    for lo, hi in bins:
        rows = [c for c in chips if int(c["valid_pixels"]) > 0
                and lo <= int(c["water_pixels"]) / int(c["valid_pixels"]) < hi]
        out = []
        for m in ("unet_e19", "baseline_fold"):
            tp, fp, fn = (sum(int(c[f"{m}_{k}"]) for c in rows) for k in ("tp", "fp", "fn"))
            out.append(fmt(tp / (tp + fp + fn)) if tp + fp + fn else "—")
        label = f"[{fmt(lo)}; {fmt(min(hi, 1.0))}{']' if hi > 1 else ')'}"
        print(f"| {label} | {len(rows)} | {out[0]} | {out[1]} |")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "docs/verification/loeo"))
