"""`make baseline`: choose the S1 threshold on train, check it on dev and validation.

Reads only train/dev/validation chips (`chip_ids`, test stays sealed, D-010). Writes:
  models/baseline/config.json                      channel, threshold, sigmoid scale s
  docs/verification/baseline-metrics.csv           contracts §7 rows, method=baseline
  docs/verification/baseline-threshold-curve.csv   threshold -> metrics (train/dev/validation)
  docs/verification/baseline-variants.csv          VV vs VH vs per-chip Otsu (VH)
  docs/verification/baseline-val.md                report
and copies of the CSVs into $FLOODVALUE_OUTPUT_DIR/baseline/.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import subprocess
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from floodvalue.baseline import core
from floodvalue.model import metrics
from floodvalue.paths import MODELS_DIR, REPO_ROOT, output_dir

PARTS = ("train", "dev", "validation")
METHOD = "baseline"
CONFIG_PATH = MODELS_DIR / "baseline" / "config.json"
VERIFY_DIR = REPO_ROOT / "docs" / "verification"
CURVE_RANGE = (-35.0, 0.0)  # dB range written to threshold_curve.csv
SEED = 20260925
N_BOOT = 1000


@dataclass
class ChipStats:
    chip_id: str
    part: str
    event: str
    n_water: int
    n_dry: int
    water_le: dict[str, np.ndarray]
    dry_le: dict[str, np.ndarray]
    otsu_t: float | None
    otsu_counts: dict[str, int]
    valid: np.ndarray = field(repr=False)


def chip_stats(chip_id: str, part: str, s1_db: np.ndarray, label: np.ndarray) -> ChipStats:
    valid = core.valid_mask(s1_db, label)
    water = valid & (label == 1)
    dry = valid & (label != 1)
    wl = {c: core.counts_le(s1_db[i][water]) for c, i in core.CHANNELS.items()}
    dl = {c: core.counts_le(s1_db[i][dry]) for c, i in core.CHANNELS.items()}
    vh = s1_db[core.CHANNELS["VH"]]
    finite = np.isfinite(s1_db).all(axis=0)
    otsu_t = core.otsu_threshold(vh[finite]) if finite.any() else None
    pred = core.threshold_mask(s1_db, core.CHANNELS["VH"], otsu_t) if otsu_t is not None \
        else np.zeros(label.shape, bool)
    return ChipStats(chip_id, part, chip_id.rsplit("_", 1)[0], int(water.sum()), int(dry.sum()),
                     wl, dl, otsu_t, metrics.confusion(pred, label, valid), valid)


def _sum_le(chips: list[ChipStats], ch: str):
    w = np.sum([c.water_le[ch] for c in chips], axis=0)
    d = np.sum([c.dry_le[ch] for c in chips], axis=0)
    return w, d, sum(c.n_water for c in chips), sum(c.n_dry for c in chips)


def counts_at(chips: list[ChipStats], ch: str, t: float) -> dict[str, int]:
    """Confusion of the rule `x <= t` pooled over chips; t must be one of core.EDGES."""
    k = int(np.flatnonzero(np.isclose(core.EDGES, t))[0])
    w, d, nw, nd = _sum_le(chips, ch)
    c = core.curve_counts(w, d, nw, nd)
    return {key: int(c[key][k]) for key in ("tp", "fp", "fn", "tn")}


def otsu_counts(chips: list[ChipStats]) -> dict[str, int]:
    out = None
    for c in chips:
        out = dict(c.otsu_counts) if out is None else metrics.add(out, c.otsu_counts)
    return out


def groups(chips: list[ChipStats], part: str) -> list[tuple[str, list[ChipStats]]]:
    """(event, chips) for each event of the part, then ('ALL', all chips of the part)."""
    mine = [c for c in chips if c.part == part]
    events = sorted({c.event for c in mine})
    return [(e, [c for c in mine if c.event == e]) for e in events] + [("ALL", mine)]


def mask_hash(chips: Iterable[ChipStats]) -> str:
    """Shared algorithm (model.metrics.mask_sha256): packed masks in chip_id order."""
    return metrics.mask_sha256([c.valid for c in sorted(chips, key=lambda c: c.chip_id)])


def metrics_row(method: str, split: str, event: str, counts: dict[str, int],
                threshold: float | None, valid_mask_sha256: str) -> dict:
    """One row of metrics.csv (contracts §7); undefined ratios are empty strings."""
    s = metrics.scores(counts)
    row = {"method": method, "split": split, "event": event}
    for k in ("iou", "f1", "precision", "recall"):
        row[k] = "" if np.isnan(s[k]) else round(s[k], 6)
    row.update({k: counts[k] for k in ("tp", "fp", "fn", "tn")})
    row["n_pixels"] = s["n_pixels"]
    row["threshold"] = "" if threshold is None else threshold
    row["valid_mask_sha256"] = valid_mask_sha256
    return {c: row[c] for c in metrics.METRIC_COLUMNS}


def _boot_idx(n_chips: int, n: int = N_BOOT, seed: int = SEED) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n_chips, (n, n_chips))


def _boot_ious(chips: list[ChipStats], ch: str, t: float, idx: np.ndarray) -> np.ndarray:
    """Pooled IoU for each resample of chips (rows of idx)."""
    per = np.array([[counts_at([c], ch, t)[k] for k in ("tp", "fp", "fn")] for c in chips],
                   np.int64)
    tp, fp, fn = (per[idx, j].sum(axis=1) for j in range(3))
    den = tp + fp + fn
    return np.divide(tp, den, out=np.zeros(len(idx)), where=den > 0)


def bootstrap_iou(chips: list[ChipStats], ch: str, t: float) -> tuple[float, float]:
    """95 % percentile interval of pooled IoU, resampling chips with replacement."""
    iou = _boot_ious(chips, ch, t, _boot_idx(len(chips)))
    lo, hi = np.percentile(iou, [2.5, 97.5])
    return float(lo), float(hi)


def paired_delta(chips: list[ChipStats], a: tuple[str, float], b: tuple[str, float]) -> dict:
    """Paired chip bootstrap of IoU(a) - IoU(b): same resamples for both variants."""
    idx = _boot_idx(len(chips))
    d = _boot_ious(chips, *a, idx) - _boot_ious(chips, *b, idx)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"mean": float(d.mean()), "lo": float(lo), "hi": float(hi),
            "share_positive": float((d > 0).mean())}


def fit(chips: list[ChipStats]) -> dict:
    """Threshold and scale are fitted on train; the channel (VV/VH) is chosen on dev.

    Protocol D-012/D-016: dev (USA) is the part for choosing between variants, validation
    only reports. Without dev chips the channel falls back to the train IoU.
    """
    train = [c for c in chips if c.part == "train"]
    dev = [c for c in chips if c.part == "dev"]
    per_channel = {}
    for ch in core.CHANNELS:
        w, d, nw, nd = _sum_le(train, ch)
        t, iou = core.best_threshold(core.EDGES, w, d, nw, nd)
        per_channel[ch] = {"threshold_db": t, "train_iou": round(iou, 6)}
        if dev:
            s = metrics.scores(counts_at(dev, ch, t))
            per_channel[ch]["dev_iou"] = round(0.0 if np.isnan(s["iou"]) else s["iou"], 6)
    key = "dev_iou" if dev else "train_iou"
    channel = max(per_channel, key=lambda ch: per_channel[ch][key])
    t = per_channel[channel]["threshold_db"]
    w, d, _, _ = _sum_le(train, channel)
    centers = core.bin_centers(core.EDGES)
    s = core.fit_scale(t, centers, core.hist_from_le(w), core.hist_from_le(d))
    return {"channel": channel, "channel_chosen_on": "dev" if dev else "train",
            "threshold_db": t, "scale_db": round(s, 4), "per_channel": per_channel}


def evaluate(chips: list[ChipStats], params: dict) -> dict:
    ch, t = params["channel"], params["threshold_db"]
    rows, variants, curve, oracle = [], [], [], []
    lo, hi = CURVE_RANGE
    keep = (core.EDGES >= lo - 1e-9) & (core.EDGES <= hi + 1e-9)
    for part in PARTS:
        for event, grp in groups(chips, part):
            if not grp:
                continue
            h = mask_hash(grp)
            rows.append(metrics_row(METHOD, part, event, counts_at(grp, ch, t), t, h))
            for name, cnt, thr in (
                *[(f"threshold_{c}", counts_at(grp, c, params["per_channel"][c]["threshold_db"]),
                   params["per_channel"][c]["threshold_db"]) for c in core.CHANNELS],
                ("otsu_VH_per_chip", otsu_counts(grp), None),
            ):
                r = metrics_row(name, part, event, cnt, thr, h)
                variants.append(r)
            w, d, nw, nd = _sum_le(grp, ch)
            best_t, best_iou = core.best_threshold(core.EDGES, w, d, nw, nd)
            oracle.append({"split": part, "event": event, "best_threshold_db": best_t,
                           "best_iou": round(best_iou, 6)})
        grp = [c for c in chips if c.part == part]
        if not grp:
            continue
        for c_name in core.CHANNELS:
            w, d, nw, nd = _sum_le(grp, c_name)
            cc = core.curve_counts(w, d, nw, nd)
            for k in np.flatnonzero(keep):
                cnt = {key: int(cc[key][k]) for key in ("tp", "fp", "fn", "tn")}
                s = metrics.scores(cnt)
                curve.append({"split": part, "channel": c_name,
                              "threshold_db": float(core.EDGES[k]),
                              **{m: ("" if np.isnan(s[m]) else round(s[m], 6))
                                 for m in ("iou", "f1", "precision", "recall")},
                              **cnt, "n_pixels": s["n_pixels"]})
    boot = {}
    for part in ("dev", "validation"):
        grp = [c for c in chips if c.part == part]
        if grp:
            boot[part] = bootstrap_iou(grp, ch, t)
    dev = [c for c in chips if c.part == "dev"]
    other = [c for c in core.CHANNELS if c != ch]
    delta = (paired_delta(dev, (ch, t), (other[0], params["per_channel"][other[0]]["threshold_db"]))
             if dev and other else None)
    otsu_t = {c.part: [] for c in chips}
    for c in chips:
        if c.otsu_t is not None:
            otsu_t[c.part].append(c.otsu_t)
    return {"rows": rows, "variants": variants, "curve": curve, "oracle": oracle,
            "bootstrap": boot, "delta_dev": delta,
            "water_share": {p: _water_share([c for c in chips if c.part == p]) for p in PARTS},
            "otsu_t": {p: (float(np.median(v)), float(np.min(v)), float(np.max(v)))
                       for p, v in otsu_t.items() if v},
            "n_chips": {p: sum(c.part == p for c in chips) for p in PARTS}}


def _water_share(chips: list[ChipStats]) -> float | None:
    nw, nd = sum(c.n_water for c in chips), sum(c.n_dry for c in chips)
    return nw / (nw + nd) if nw + nd else None


def fit_and_evaluate(parts: dict[str, list[str]],
                     read: Callable[[str], tuple[np.ndarray, np.ndarray]]) -> tuple[dict, dict]:
    if "test" in parts:
        raise ValueError("baseline never reads test (D-010)")
    chips = [chip_stats(cid, part, *read(cid)) for part, ids in parts.items() for cid in ids]
    params = fit(chips)
    return params, evaluate(chips, params)


# ---------------------------------------------------------------- I/O


def read_chip(chip_id: str) -> tuple[np.ndarray, np.ndarray]:
    """S1 (2, H, W) dB and LabelHand via the sealed loader: test labels raise (D-010)."""
    from floodvalue.data.io import load_label, load_s1

    label = load_label(chip_id)[0]  # seal check first, before any file is opened
    return load_s1(chip_id)[0], label


def git_commit() -> str:
    try:
        head = subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], cwd=REPO_ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "src/floodvalue/baseline",
                                "src/floodvalue/model/metrics.py"], cwd=REPO_ROOT,
                               capture_output=True, text=True, check=True).stdout.strip()
        return head + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)


def _fmt(v, nd=3) -> str:
    if v == "" or v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def render_report(params: dict, result: dict, config: dict) -> str:
    ch, t, s = params["channel"], params["threshold_db"], params["scale_db"]
    out = [
        "# Baseline: порог по S1 — проверка на dev и validation",
        "",
        (f"Сгенерировано `make baseline` (commit `{config['commit']}`, {config['created_at']}). "
         "Числа ниже — из `docs/verification/baseline-metrics.csv`, "
         "`baseline-variants.csv`, `baseline-threshold-curve.csv`. Test не читался (D-010)."),
        "",
        "## Правило",
        "",
        (f"- Вода ⇔ **{ch} ≤ {t:.1f} dB** (S1 уже в dB). Порог — максимум IoU по всем валидным "
         "пикселям train (пулинг пикселей), перебор от −60 до +30 dB с шагом 0,1."),
        "- Для каждого канала порог подобран на train; канал выбран по IoU на "
        f"**{params['channel_chosen_on']}** (D-012: dev — выбор варианта, validation — только "
        "отчёт): " + "; ".join(
            f"{c}: порог {v['threshold_db']:.1f} dB, IoU train {v['train_iou']:.3f}"
            + (f", IoU dev {v['dev_iou']:.3f}" if "dev_iou" in v else "")
            for c, v in params["per_channel"].items()) + ".",
        (f"- Вероятность (временный основной метод, D-009): p = sigmoid(({t:.1f} − {ch}) / s), "
         f"**s = {s:.3f} dB** — минимум log-loss на train при фиксированном пороге. "
         "p ≥ 0,5 ⇔ маска baseline. Порог 0,5 и сигмоида не выбраны на validation "
         "(калибровка — отдельный этап): доля воды среди валидных пикселей "
         + ", ".join(f"{p} {v * 100:.1f} %" for p, v in result["water_share"].items()
                     if v is not None) + "."),
        ("- Валидный пиксель: VV и VH конечны **и** LabelHand ≠ −1 (`floodvalue.model.metrics`). "
         "`valid_mask_sha256` — отпечаток этой маски; основной метод обязан получить тот же хеш."),
        (f"- Чипов: train {result['n_chips']['train']}, dev {result['n_chips']['dev']}, "
         f"validation {result['n_chips']['validation']}."),
        "",
        "## Метрики baseline по событиям",
        "",
        "| split | событие | IoU | F1 | Precision | Recall | TP | FP | FN | TN | пикселей |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in result["rows"]:
        ev = f"**{r['event']}**" if r["event"] == "ALL" else r["event"]
        out.append(f"| {r['split']} | {ev} | {_fmt(r['iou'])} | {_fmt(r['f1'])} | "
                   f"{_fmt(r['precision'])} | {_fmt(r['recall'])} | {r['tp']} | {r['fp']} | "
                   f"{r['fn']} | {r['tn']} | {r['n_pixels']} |")
    out += ["", "95 % бутстрэп-интервал IoU (ALL, чипы с возвращением, "
            f"{N_BOOT} повторов, seed {SEED}): " + "; ".join(
                f"{p} [{lo:.3f}; {hi:.3f}]" for p, (lo, hi) in result["bootstrap"].items()) + ".",
            *([] if not result["delta_dev"] else [
                "", ("Выбор канала на dev, парный бутстрэп по чипам (те же выборки для обоих "
                     f"каналов): ΔIoU({ch} − другой канал) = {result['delta_dev']['mean']:+.3f}, "
                     f"95 % [{result['delta_dev']['lo']:+.3f}; {result['delta_dev']['hi']:+.3f}], "
                     f"доля выборок с Δ > 0 — {result['delta_dev']['share_positive']:.3f}. "
                     "Dev — одно событие (USA), поэтому выбор канала опирается и на D-009.")]),
            "", "## Варианты (ALL): VV, VH, Otsu по чипу", "",
            ("Пороги VV и VH — каждый свой, выбран на train. Otsu считается по VH каждого чипа "
             "отдельно, без обучения (порог чипа — по всем конечным пикселям S1)."), "",
            "| split | вариант | порог, dB | IoU | F1 | Precision | Recall |",
            "|---|---|---|---|---|---|---|"]
    for r in result["variants"]:
        if r["event"] != "ALL":
            continue
        out.append(f"| {r['split']} | {r['method']} | {_fmt(r['threshold'], 1)} | "
                   f"{_fmt(r['iou'])} | {_fmt(r['f1'])} | {_fmt(r['precision'])} | "
                   f"{_fmt(r['recall'])} |")
    ot = result["otsu_t"]
    out += ["", "Порог Otsu по чипам, медиана [мин; макс], dB: " + "; ".join(
        f"{p} {m:.1f} [{a:.1f}; {b:.1f}]" for p, (m, a, b) in ot.items()) + ".",
        "", "## Устойчивость порога между событиями (диагностика, не выбор)", "",
        (f"Порог {ch}, максимизирующий IoU отдельно на каждом событии. Для выбора не "
         "используется — показывает, насколько один глобальный порог далёк от "
         "оптимального на новом событии."), "",
        "| split | событие | лучший порог, dB | IoU при нём |", "|---|---|---|---|"]
    for o in result["oracle"]:
        out.append(f"| {o['split']} | {o['event']} | {o['best_threshold_db']:.1f} | "
                   f"{o['best_iou']:.3f} |")
    out += ["", "## Кривая «порог → метрики»", "",
            (f"`docs/verification/baseline-threshold-curve.csv`: split × канал × порог от "
             f"{CURVE_RANGE[0]:.0f} до {CURVE_RANGE[1]:.0f} dB с шагом 0,1 (IoU, F1, P, R, "
             "TP/FP/FN/TN). График — `docs/verification/baseline-threshold-curve.png`."), "",
            "## Воспроизведение", "", "```bash",
            "set -a; . ./.env; set +a      # FLOODVALUE_DATA_DIR",
            "make baseline                 # порог, s, метрики, этот отчёт",
            "uv run --frozen python -m floodvalue.baseline.predict --chip Sri-Lanka_713926",
            "uv run --frozen pytest tests/baseline", "```", ""]
    return "\n".join(out)


def plot_curve(curve: list[dict], params: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    for part in PARTS:
        for ch, ls in (("VH", "-"), ("VV", ":")):
            pts = [(r["threshold_db"], r["iou"]) for r in curve
                   if r["split"] == part and r["channel"] == ch and r["iou"] != ""]
            if pts:
                x, y = zip(*pts)
                ax.plot(x, y, ls, label=f"{part} {ch}")
    ax.axvline(params["threshold_db"], color="k", lw=0.8)
    ax.set_xlabel("порог, dB (вода ⇔ x ≤ порог)")
    ax.set_ylabel("IoU")
    ax.legend(fontsize=7, ncol=3)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def run(args) -> int:
    from floodvalue.data.split import chip_ids

    parts = {p: chip_ids(p) for p in PARTS}
    params, result = fit_and_evaluate(parts, read_chip)
    config = {
        "method": "threshold_baseline",
        "rule": f"water <=> {params['channel']} <= threshold_db (dB)",
        "proba": "p = sigmoid((threshold_db - x) / scale_db), x = channel in dB; "
                 "p >= threshold <=> x <= threshold_db",
        **params,
        "threshold": 0.5,
        "threshold_status": "fixed 0.5 (= baseline rule), not chosen on validation; "
                            "temporary main method D-009, calibration replaces it",
        "weights_sha256": hashlib.sha256(json.dumps(
            {k: params[k] for k in ("channel", "threshold_db", "scale_db")},
            sort_keys=True).encode()).hexdigest(),
        "weights_note": "no weight file: sha256 of canonical JSON {channel, threshold_db, scale_db}",
        "fitted_on": "train",
        "split_sha256": _split_sha(),
        "created_at": dt.datetime.now(dt.UTC).date().isoformat(),
        "commit": git_commit(),
    }
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(config, ensure_ascii=False, indent=1) + "\n")
    out_dir = Path(args.out) if getattr(args, "out", None) else output_dir() / "baseline"
    files = {"baseline-metrics.csv": result["rows"],
             "baseline-variants.csv": result["variants"],
             "baseline-threshold-curve.csv": result["curve"]}
    for name, rows in files.items():
        write_csv(VERIFY_DIR / name, rows)
        write_csv(out_dir / name.removeprefix("baseline-").replace("-", "_"), rows)
    plot_curve(result["curve"], params, VERIFY_DIR / "baseline-threshold-curve.png")
    (VERIFY_DIR / "baseline-val.md").write_text(render_report(params, result, config))
    for r in result["rows"]:
        if r["event"] == "ALL":
            print(f"{r['split']:<10} IoU {_fmt(r['iou'])} F1 {_fmt(r['f1'])} "
                  f"P {_fmt(r['precision'])} R {_fmt(r['recall'])} n {r['n_pixels']}")
    print(f"baseline: {params['channel']} <= {params['threshold_db']} dB, "
          f"s = {params['scale_db']} -> {CONFIG_PATH.relative_to(REPO_ROOT)}")
    return 0


def _split_sha() -> str:
    from floodvalue.data.split import load_split

    return load_split()["sha256"]
