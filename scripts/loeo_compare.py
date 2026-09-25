"""Compare LOEO recipe runs: mean held-out IoU over events and per-event deltas vs reference.

    python scripts/loeo_compare.py [run_dir ...]   (reference: docs/verification/loeo-sweep/r0_ref)

Pooled IoU per held-out event from loeo-metrics.csv; `mean6` = mean over the 6 events,
`mean5` = without USA (dev: D-019 choices were made on it). Seed noise = |r1_seed1 - ref|.
"""

import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1] / "docs/verification/loeo-sweep"
REF = Path(__file__).resolve().parents[1] / "docs/verification/loeo-sweep/r0_ref"


def load(d: Path) -> dict[tuple[str, str], dict]:
    f = d / "loeo-metrics.csv"
    if not f.exists():
        return {}
    return {(r["event"], r["method"]): r for r in csv.DictReader(open(f, encoding="utf-8"))}


def main(argv: list[str]) -> None:
    runs = [Path(a) for a in argv] or sorted(p for p in HERE.iterdir() if (p / "loeo-metrics.csv").exists())
    ref = load(REF)
    events = sorted({e for e, _ in ref})
    methods = ["unet_e60", "unet_e60_tta", "unet_e19"]
    base = {e: float(ref[(e, "baseline_fold")]["iou"]) for e in events}
    r0 = {e: float(ref[(e, "unet_e60")]["iou"]) for e in events}
    print(f"{'run':22s} {'method':13s} " + " ".join(f"{e[:8]:>8s}" for e in events)
          + f" {'mean6':>7s} {'mean5':>7s} {'Δmean6':>7s} {'n':>2s}")
    print(f"{'baseline_fold':22s} {'':13s} " + " ".join(f"{base[e]:8.3f}" for e in events)
          + f" {sum(base.values()) / 6:7.3f} {sum(v for e, v in base.items() if e != 'USA') / 5:7.3f}")
    print(f"{'R0 ref seed20260925':22s} {'unet_e60':13s} " + " ".join(f"{r0[e]:8.3f}" for e in events)
          + f" {sum(r0.values()) / 6:7.3f} {sum(v for e, v in r0.items() if e != 'USA') / 5:7.3f}")
    for d in runs:
        m = load(d)
        for meth in methods:
            vals = {e: float(m[(e, meth)]["iou"]) for e in events if (e, meth) in m}
            if not vals:
                continue
            done = len(vals)
            row = " ".join(f"{vals[e]:8.3f}" if e in vals else f"{'…':>8s}" for e in events)
            if done == 6:
                mean6 = sum(vals.values()) / 6
                mean5 = sum(v for e, v in vals.items() if e != "USA") / 5
                delta = mean6 - sum(r0.values()) / 6
                tail = f" {mean6:7.3f} {mean5:7.3f} {delta:+7.3f}"
            else:
                common = [e for e in vals]
                delta = sum(vals[e] - r0[e] for e in common) / len(common)
                tail = f" {'':7s} {'':7s} {delta:+7.3f}"
            print(f"{d.name[:22]:22s} {meth:13s} {row}{tail} {done:2d}")


if __name__ == "__main__":
    main(sys.argv[1:])
