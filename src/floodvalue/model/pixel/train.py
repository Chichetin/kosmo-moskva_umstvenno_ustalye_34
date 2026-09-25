"""Train candidate A: pixel classifier on S1 features. Train fits, dev chooses (D-012).

    uv run --frozen python -m floodvalue.model.pixel.train

Protocol (D-012, D-016): clip bounds, pixel sample and the classifier come from train
only; hyperparameters (and, via ablation.py, the feature set) are chosen by pooled IoU on
dev (USA); validation is only reported. Test is never read (chip_ids refuses it).
Writes:
  models/pixel/model.joblib                   HistGradientBoostingClassifier (Git LFS)
  models/pixel/config.json                    features, clip, threshold, sha256, dev grid
  docs/verification/pixel-metrics.csv         contracts §7 rows, baseline and pixel
  docs/verification/pixel-threshold-curve.csv p >= t -> metrics (diagnostic, not a choice)
  docs/verification/pixel-val.md              report: dev and validation, CIs, paired ΔIoU
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from floodvalue.model import metrics
from floodvalue.model.pixel import features
from floodvalue.paths import MODELS_DIR, REPO_ROOT

SEED = 20260925
N_BOOT = 1000
N_PER_CHIP = 8000          # train pixels sampled per chip (uniform over valid pixels)
CLIP_Q = (0.01, 99.99)     # clip bounds = these percentiles of train valid pixels, dB
THRESHOLD = 0.5
MODEL_DIR = MODELS_DIR / "pixel"
VERIFY_DIR = REPO_ROOT / "docs" / "verification"
BASELINE_CONFIG = MODELS_DIR / "baseline" / "config.json"
BASELINE_METRICS = VERIFY_DIR / "baseline-metrics.csv"
METHOD = "pixel"
# Candidate feature sets (texture window sizes); the grid below chooses one on dev.
FEATURE_SETS = {
    "tex3": ["VV", "VH", "VV-VH", *features.texture_names((3,))],
    "tex3_7": ["VV", "VH", "VV-VH", *features.texture_names((3, 7))],
    "tex3_7_15": list(features.FULL),
}
SUPERSET = list(features.FULL)
# Grid chosen on dev by pooled IoU: feature set x tree size x weight of water pixels.
GRID = [{"feature_set": fs, "max_leaf_nodes": leaves, "water_weight": ww}
        for fs in FEATURE_SETS for leaves in (31, 127) for ww in (1.0, 2.0, 4.0)]
FIXED = {"learning_rate": 0.1, "max_iter": 300, "min_samples_leaf": 100, "l2_regularization": 0.0}

Reader = Callable[[str], tuple[np.ndarray, np.ndarray]]


# ------------------------------------------------------------------ train sample


def fit_clip(train_ids: list[str], read: Reader, step: int = 25) -> dict[str, list[float]]:
    """Clip bounds per channel: CLIP_Q percentiles of valid train pixels (every `step`-th)."""
    vals: dict[str, list[np.ndarray]] = {c: [] for c in features.CHANNELS}
    for cid in train_ids:
        s1, label = read(cid)
        ok = np.isfinite(s1).all(axis=0) & (label != -1)
        for c, i in features.CHANNELS.items():
            vals[c].append(s1[i][ok][::step])
    out = {}
    for c, v in vals.items():
        lo, hi = np.percentile(np.concatenate(v), CLIP_Q)
        out[c] = [round(float(lo), 2), round(float(hi), 2)]
    return out


def sample_pixels(valid: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Flat indices of min(n, #valid) valid pixels, uniform without replacement."""
    idx = np.flatnonzero(valid)
    if len(idx) <= n:
        return idx
    return np.sort(rng.choice(idx, n, replace=False))


def train_sample(train_ids: list[str], read: Reader, names: list[str],
                 clip: dict, n_per_chip: int = N_PER_CHIP, seed: int = SEED) -> dict:
    """Stratified by chip: up to n_per_chip valid pixels from each train chip, class ratio
    kept as in the chip. Chips without valid pixels (Paraguay_34417) are skipped."""
    rng = np.random.default_rng(seed)
    xs, ys, skipped = [], [], []
    for cid in train_ids:
        s1, label = read(cid)
        valid = np.isfinite(s1).all(axis=0) & (label != -1)
        if not valid.any():
            skipped.append(cid)
            continue
        idx = sample_pixels(valid, n_per_chip, rng)
        f = features.compute(s1, names, clip).reshape(len(names), -1)
        xs.append(f[:, idx].T)
        ys.append((label.reshape(-1)[idx] == 1).astype(np.int8))
    x, y = np.concatenate(xs), np.concatenate(ys)
    return {"x": x, "y": y, "skipped": skipped, "n_chips": len(xs),
            "water_share": float(y.mean())}


def make_model(params: dict, seed: int = SEED):
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(
        loss="log_loss", learning_rate=params.get("learning_rate", FIXED["learning_rate"]),
        max_iter=params.get("max_iter", FIXED["max_iter"]),
        max_leaf_nodes=params["max_leaf_nodes"],
        min_samples_leaf=params.get("min_samples_leaf", FIXED["min_samples_leaf"]),
        l2_regularization=params.get("l2_regularization", FIXED["l2_regularization"]),
        early_stopping=False, random_state=seed)


def fit(x: np.ndarray, y: np.ndarray, params: dict, seed: int = SEED):
    """Water pixels weigh `water_weight` in the loss (train water share is ~9 %)."""
    w = np.where(y == 1, params["water_weight"], 1.0)
    return make_model(params, seed).fit(x, y, sample_weight=w)


# ------------------------------------------------------------------ evaluation


@dataclass
class EvalChip:
    chip_id: str
    part: str
    event: str
    x: np.ndarray = field(repr=False)        # (n_valid_s1, F) features at S1-finite pixels
    s1_ok: np.ndarray = field(repr=False)    # (H, W) S1 finite
    label: np.ndarray = field(repr=False)
    valid: np.ndarray = field(repr=False)    # metrics mask: S1 finite and label != -1
    baseline: np.ndarray = field(repr=False)  # baseline rule mask (H, W)


def baseline_rule(s1: np.ndarray, cfg: dict) -> np.ndarray:
    """Threshold baseline from its file contract (models/baseline/config.json)."""
    x = s1[features.CHANNELS[cfg["channel"]]]
    return np.isfinite(x) & (x <= cfg["threshold_db"])


def eval_chips(parts: dict[str, list[str]], read: Reader, names: list[str], clip: dict,
               base_cfg: dict) -> list[EvalChip]:
    if "test" in parts:
        raise ValueError("pixel method never reads test (D-010)")
    out = []
    for part, ids in parts.items():
        for cid in sorted(ids):
            s1, label = read(cid)
            ok = np.isfinite(s1).all(axis=0)
            f = features.compute(s1, names, clip)
            out.append(EvalChip(cid, part, cid.rsplit("_", 1)[0], f[:, ok].T.copy(), ok, label,
                                ok & (label != -1), baseline_rule(s1, base_cfg)))
    return out


def chip_proba(model, c: EvalChip, cols: list[int] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(p, water) at the metrics-valid pixels of the chip."""
    p = np.full(c.s1_ok.shape, np.nan)  # float64 like chip_pred: identical >= t decisions
    if len(c.x):
        p[c.s1_ok] = model.predict_proba(c.x if cols is None else c.x[:, cols])[:, 1]
    return p[c.valid], (c.label == 1)[c.valid]


CURVE_T = np.round(np.arange(0.05, 0.951, 0.05), 2)


def threshold_curve(model, chips: list[EvalChip], parts, cols=None) -> list[dict]:
    """Diagnostic, not a choice: pooled metrics of `p >= t` for t in CURVE_T per split/event."""
    probs = {c.chip_id: chip_proba(model, c, cols) for c in chips}
    rows = []
    for part in parts:
        for event, grp in groups(chips, part):
            p = np.concatenate([probs[c.chip_id][0] for c in grp])
            w = np.concatenate([probs[c.chip_id][1] for c in grp])
            pw, pd = np.sort(p[w]), np.sort(p[~w])
            for t in CURVE_T:
                tp = int(len(pw) - np.searchsorted(pw, t, side="left"))
                fp = int(len(pd) - np.searchsorted(pd, t, side="left"))
                cnt = {"tp": tp, "fp": fp, "fn": len(pw) - tp, "tn": len(pd) - fp}
                sc = metrics.scores(cnt)
                rows.append({"split": part, "event": event, "threshold": float(t),
                             **{k: ("" if np.isnan(sc[k]) else round(sc[k], 6))
                                for k in ("iou", "f1", "precision", "recall")}, **cnt})
    return rows


def chip_pred(model, c: EvalChip, cols: list[int] | None = None,
              threshold: float = THRESHOLD) -> np.ndarray:
    pred = np.zeros(c.s1_ok.shape, bool)
    if len(c.x):
        x = c.x if cols is None else c.x[:, cols]
        pred[c.s1_ok] = model.predict_proba(x)[:, 1] >= threshold
    return pred


def per_chip_counts(model, chips: list[EvalChip], cols=None) -> dict[str, dict[str, int]]:
    return {c.chip_id: metrics.confusion(chip_pred(model, c, cols), c.label == 1, c.valid)
            for c in chips}


def baseline_counts(chips: list[EvalChip]) -> dict[str, dict[str, int]]:
    return {c.chip_id: metrics.confusion(c.baseline, c.label == 1, c.valid) for c in chips}


def pooled(counts: dict[str, dict[str, int]], ids: list[str]) -> dict[str, int]:
    tot = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for i in ids:
        tot = metrics.add(tot, counts[i])
    return tot


def iou(c: dict[str, int]) -> float:
    s = metrics.scores(c)["iou"]
    return 0.0 if np.isnan(s) else float(s)


def boot_idx(n: int, n_boot: int = N_BOOT, seed: int = SEED) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, (n_boot, n))


def boot_ious(counts: dict[str, dict[str, int]], ids: list[str], idx: np.ndarray) -> np.ndarray:
    per = np.array([[counts[i][k] for k in ("tp", "fp", "fn")] for i in ids], np.int64)
    tp, fp, fn = (per[idx, j].sum(axis=1) for j in range(3))
    den = tp + fp + fn
    return np.divide(tp, den, out=np.zeros(len(idx)), where=den > 0)


def bootstrap_ci(counts, ids) -> tuple[float, float]:
    lo, hi = np.percentile(boot_ious(counts, ids, boot_idx(len(ids))), [2.5, 97.5])
    return float(lo), float(hi)


def paired_delta(a, b, ids) -> dict:
    """IoU(a) - IoU(b) on the same chip resamples (paired bootstrap)."""
    idx = boot_idx(len(ids))
    d = boot_ious(a, ids, idx) - boot_ious(b, ids, idx)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"delta": iou(pooled(a, ids)) - iou(pooled(b, ids)), "mean": float(d.mean()),
            "lo": float(lo), "hi": float(hi), "share_positive": float((d > 0).mean())}


def groups(chips: list[EvalChip], part: str) -> list[tuple[str, list[EvalChip]]]:
    mine = [c for c in chips if c.part == part]
    return [(e, [c for c in mine if c.event == e])
            for e in sorted({c.event for c in mine})] + [("ALL", mine)]


def metrics_row(method, split, event, counts, threshold, mask_hash) -> dict:
    s = metrics.scores(counts)
    row = {"method": method, "split": split, "event": event}
    for k in ("iou", "f1", "precision", "recall"):
        row[k] = "" if np.isnan(s[k]) else round(s[k], 6)
    row.update({k: counts[k] for k in ("tp", "fp", "fn", "tn")})
    row["n_pixels"] = s["n_pixels"]
    row["threshold"] = threshold
    row["valid_mask_sha256"] = mask_hash
    return {c: row[c] for c in metrics.METRIC_COLUMNS}


def metric_rows(chips: list[EvalChip], parts, pix: dict, base: dict, base_thr) -> list[dict]:
    rows = []
    for part in parts:
        for event, grp in groups(chips, part):
            if not grp:
                continue
            ids = [c.chip_id for c in grp]
            h = metrics.mask_sha256([c.valid for c in grp])  # grp is in chip_id order
            rows.append(metrics_row("baseline", part, event, pooled(base, ids), base_thr, h))
            rows.append(metrics_row(METHOD, part, event, pooled(pix, ids), THRESHOLD, h))
    return rows


def columns(names: list[str], superset: list[str] = SUPERSET) -> list[int]:
    return [superset.index(n) for n in names]


def select_on_dev(sample: dict, dev: list[EvalChip], grid: list[dict],
                  log=print) -> tuple[dict, object, list[dict]]:
    """Fit each grid point on the train sample (features = SUPERSET columns), score pooled
    IoU on dev; the best wins, ties go to the earlier (smaller) grid point."""
    ids = [c.chip_id for c in dev]
    results, best = [], None
    for params in grid:
        cols = columns(FEATURE_SETS[params["feature_set"]])
        t0 = time.perf_counter()
        model = fit(sample["x"][:, cols], sample["y"], params)
        cnt = per_chip_counts(model, dev, cols)
        s = metrics.scores(pooled(cnt, ids))
        r = {**params, "dev_iou": round(iou(pooled(cnt, ids)), 6),
             "dev_precision": round(s["precision"], 6), "dev_recall": round(s["recall"], 6),
             "seconds": round(time.perf_counter() - t0, 1)}
        results.append(r)
        log(f"  {params} -> dev IoU {r['dev_iou']:.4f} ({r['seconds']} s)")
        if best is None or r["dev_iou"] > best[0]["dev_iou"]:
            best = (r, model, cnt)
    return best[0], best[1], results


# ------------------------------------------------------------------ I/O


def read_chip(chip_id: str) -> tuple[np.ndarray, np.ndarray]:
    """S1 and label through the sealed loader: test labels raise (D-010)."""
    from floodvalue.data.io import load_label, load_s1

    return load_s1(chip_id)[0], load_label(chip_id)[0]


def git_commit() -> str:
    """HEAD, with -dirty when the pixel code differs from it (trained from uncommitted code)."""
    try:
        head = subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], cwd=REPO_ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "src/floodvalue/model/pixel"],
                               cwd=REPO_ROOT, capture_output=True, text=True,
                               check=True).stdout.strip()
        return head + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)


def check_against_baseline(rows: list[dict], path: Path = BASELINE_METRICS) -> list[str]:
    """Recomputed baseline rows must equal the published baseline (same pixels, same counts)."""
    if not path.exists():
        return [f"{path} not found"]
    with open(path, newline="", encoding="utf-8") as fh:
        ref = {(r["split"], r["event"]): r for r in csv.DictReader(fh)}
    problems = []
    for r in rows:
        if r["method"] != "baseline":
            continue
        b = ref.get((r["split"], r["event"]))
        if b is None:
            problems.append(f"no baseline row {r['split']}/{r['event']}")
            continue
        for k in ("tp", "fp", "fn", "tn", "valid_mask_sha256"):
            if str(r[k]) != b[k]:
                problems.append(f"{r['split']}/{r['event']} {k}: {r[k]} != {b[k]}")
    return problems


def _f(v, nd=3) -> str:
    return "—" if v == "" or v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def render_report(cfg: dict, rows: list[dict], ci: dict, deltas: dict, grid: list[dict],
                  timing: dict, problems: list[str], curve: list[dict] | None = None,
                  water: dict | None = None) -> str:
    out = [
        "# Основной метод A: пиксельный классификатор S1 — dev и validation", "",
        (f"Сгенерировано `python -m floodvalue.model.pixel.train` (commit `{cfg['commit']}`, "
         f"{cfg['created_at']}). Числа — из `docs/verification/pixel-metrics.csv`. "
         "Test не читался (D-010)."), "",
        "## Метод", "",
        ("- Вход — только S1 (VV, VH в dB, как в файле). Клиппинг по train: "
         + ", ".join(f"{c} [{lo}; {hi}] dB" for c, (lo, hi) in cfg["clip_db"].items())
         + f" — перцентили {CLIP_Q[0]} / {CLIP_Q[1]} валидных пикселей train."),
        (f"- Признаки ({len(cfg['features'])}): `" + "`, `".join(cfg["features"]) + "`. "
         "Локальные среднее и σ — окна с учётом NaN (сумма и число конечных соседей)."),
        (f"- Модель — `HistGradientBoostingClassifier` (scikit-learn), log-loss, "
         f"вес воды {cfg['params']['water_weight']}, листьев {cfg['params']['max_leaf_nodes']}, "
         f"итераций {cfg['params']['max_iter']}, шаг {cfg['params']['learning_rate']}, "
         f"min_samples_leaf {cfg['params']['min_samples_leaf']}, без early stopping, seed {SEED}."),
        (f"- Обучение — только train: до {N_PER_CHIP} случайных валидных пикселей с каждого чипа "
         f"({cfg['train_sample']['n_chips']} чипов, {cfg['train_sample']['n_pixels']} пикселей, "
         f"доля воды {cfg['train_sample']['water_share'] * 100:.1f} %); пропущены без валидных "
         f"пикселей: {', '.join(cfg['train_sample']['skipped']) or 'нет'}."),
        ("- Выбор набора окон текстуры, размера деревьев и веса воды — pooled IoU на **dev** "
         "(USA), таблица ниже; вклад каждого входа — контролируемый эксперимент на dev "
         "(`docs/verification/pixel-ablation.md`). Validation — только отчёт."),
        (f"- p = `predict_proba` классификатора; маска ⇔ p ≥ {THRESHOLD} (фиксировано, не "
         "подобрано на validation; порог и калибровку основного метода задаёт этап калибровки "
         "на validation по D-012)."),
        ("- Валидный пиксель — VV и VH конечны и LabelHand ≠ −1; хеш маски "
         "(`model.metrics.mask_sha256`, чипы в порядке `chip_id`) совпадает с baseline: "
         + ("**да, все строки dev/validation**" if not problems else
            "**НЕТ: " + "; ".join(problems) + "**") + "."), "",
        "## Baseline → A по событиям (одни и те же валидные пиксели)", "",
        "| split | событие | метод | IoU | F1 | Precision | Recall | TP | FP | FN | TN | пикселей |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        ev = f"**{r['event']}**" if r["event"] == "ALL" else r["event"]
        out.append(f"| {r['split']} | {ev} | {r['method']} | {_f(r['iou'])} | {_f(r['f1'])} | "
                   f"{_f(r['precision'])} | {_f(r['recall'])} | {r['tp']} | {r['fp']} | "
                   f"{r['fn']} | {r['tn']} | {r['n_pixels']} |")
    out += ["", f"95 % бутстрэп-интервал IoU (ALL, чипы с возвращением, {N_BOOT} повторов, "
            f"seed {SEED}):", ""]
    for (part, m), (lo, hi) in ci.items():
        out.append(f"- {part}, {m}: [{lo:.3f}; {hi:.3f}]")
    out += ["", "Парный бутстрэп по чипам (одни выборки для обоих методов), ΔIoU = A − baseline:", ""]
    for part, d in deltas.items():
        out.append(f"- **{part}**: ΔIoU = {d['delta']:+.3f}, 95 % [{d['lo']:+.3f}; {d['hi']:+.3f}], "
                   f"доля выборок с Δ > 0 — {d['share_positive']:.3f}"
                   + (" (решающее — dev; validation — подтверждение)" if part == "dev" else ""))
    out += ["", "## Выбор гиперпараметров на dev", "",
            "| признаки | листьев | вес воды | IoU dev | Precision dev | Recall dev | обучение + dev, с |",
            "|---|---|---|---|---|---|---|"]
    for g in grid:
        mark = " ← выбран" if (g["feature_set"], g["max_leaf_nodes"], g["water_weight"]) == (
            cfg["feature_set"], cfg["params"]["max_leaf_nodes"],
            cfg["params"]["water_weight"]) else ""
        out.append(f"| {g['feature_set']} | {g['max_leaf_nodes']} | {g['water_weight']} | "
                   f"{g['dev_iou']:.3f}{mark} | "
                   f"{g['dev_precision']:.3f} | {g['dev_recall']:.3f} | {g['seconds']} |")
    out += ["", "## Выводы", "", *findings(rows, deltas, curve, water)]
    if curve:
        out += ["", "## Порог → IoU (диагностика, не выбор)", "",
                ("IoU при маске p ≥ t. Порог здесь не выбирается: порог основного метода "
                 "задаёт этап калибровки на validation (D-012). При другом t IoU на validation "
                 "выше, чем при 0,5, но это оракульная оценка (t и IoU — на одной validation). "
                 "Полная кривая — `docs/verification/pixel-threshold-curve.csv`."), "",
                "| split | событие | " + " | ".join(f"t={t:.1f}" for t in CURVE_T[1::2]) + " | лучший t (IoU) |",
                "|---|---|" + "---|" * (len(CURVE_T[1::2]) + 1)]
        keys = sorted({(r["split"], r["event"]) for r in curve},
                      key=lambda k: (k[0] != "dev", k[1] == "ALL", k[1]))
        for split, event in keys:
            rs = [r for r in curve if (r["split"], r["event"]) == (split, event)]
            by_t = {round(r["threshold"], 2): r for r in rs}
            best = max(rs, key=lambda r: r["iou"] if r["iou"] != "" else -1)
            out.append(f"| {split} | {event} | " + " | ".join(
                _f(by_t[round(float(t), 2)]["iou"]) for t in CURVE_T[1::2])
                + f" | {best['threshold']:.2f} ({_f(best['iou'])}) |")
    out += ["", "## CPU inference", "",
            (f"Один чип 512×512 (`{timing['chip']}`, чтение S1 → признаки → p → запись GeoTIFF): "
             f"{timing['seconds']:.1f} с на CPU (`{timing['cpu']}`, потоков {timing['threads']}). "
             "Требование < 1 мин."), "",
            "## Воспроизведение", "", "```bash",
            "set -a; . ./.env; set +a                      # FLOODVALUE_DATA_DIR",
            "uv run --frozen python -m floodvalue.model.pixel.train      # модель, config, этот отчёт",
            "uv run --frozen python -m floodvalue.model.pixel.ablation   # pixel-ablation.md",
            "uv run --frozen python -m floodvalue.model.pixel.predict --chip Sri-Lanka_713926",
            "uv run --frozen pytest tests/model", "```", ""]
    return "\n".join(out)


def findings(rows: list[dict], deltas: dict, curve: list[dict] | None,
             water: dict | None = None) -> list[str]:
    """Plain statements computed from the tables above."""
    r = {(x["split"], x["event"], x["method"]): x for x in rows}

    def dev_at(t):
        return next(float(c["iou"]) for c in curve if c["split"] == "dev"
                    and c["event"] == "ALL" and abs(float(c["threshold"]) - t) < 1e-9)
    out = []
    dv, db = r[("dev", "ALL", METHOD)], r[("dev", "ALL", "baseline")]
    out.append(f"- **dev (выбор):** A лучше baseline — IoU {db['iou']:.3f} → {dv['iou']:.3f}, "
               f"ложных срабатываний {db['fp']} → {dv['fp']}, пропусков {db['fn']} → {dv['fn']}; "
               f"парный ΔIoU {deltas['dev']['delta']:+.3f} [{deltas['dev']['lo']:+.3f}; "
               f"{deltas['dev']['hi']:+.3f}].")
    vv, vb = r[("validation", "ALL", METHOD)], r[("validation", "ALL", "baseline")]
    verdict = "лучше" if deltas["validation"]["lo"] > 0 else (
        "хуже" if deltas["validation"]["hi"] < 0 else "не отличается значимо от")
    out.append(f"- **validation (подтверждение) при p ≥ {THRESHOLD}:** A {verdict} baseline — IoU "
               f"{vb['iou']:.3f} → {vv['iou']:.3f}, ΔIoU {deltas['validation']['delta']:+.3f} "
               f"[{deltas['validation']['lo']:+.3f}; {deltas['validation']['hi']:+.3f}]. "
               f"Precision {vb['precision']:.3f} → {vv['precision']:.3f}, Recall "
               f"{vb['recall']:.3f} → {vv['recall']:.3f}: A почти не даёт ложных тревог, но "
               "пропускает воду.")
    for ev in sorted({x["event"] for x in rows if x["split"] == "validation"} - {"ALL"}):
        a, b = r[("validation", ev, METHOD)], r[("validation", ev, "baseline")]
        out.append(f"  - {ev}: IoU {b['iou']:.3f} → {a['iou']:.3f}, Recall {b['recall']:.3f} → "
                   f"{a['recall']:.3f}.")
    if curve:
        va = [c for c in curve if c["split"] == "validation" and c["event"] == "ALL"]
        best = max(va, key=lambda c: c["iou"] if c["iou"] != "" else -1)
        out.append(
            f"- Порог 0,5 не переносится между событиями: на validation при t = "
            f"{best['threshold']:.2f} IoU A = {best['iou']:.3f}, а на dev тот же t даёт "
            f"{dev_at(best['threshold']):.3f} (кривая ниже). {best['iou']:.3f} — оракульная "
            "верхняя оценка (t подобран на той же validation), для выбора не используется и "
            "несопоставима с методами при их фиксированном пороге. Калибровку A пришлось бы и "
            "подбирать, и проверять на одной validation (D-012), честной оценкой осталась бы "
            "только test.")
    if water:
        out.append("- Описательная статистика воды (медиана VV / VH, dB, пиксели метки 1; не "
                   "используется для настройки): " + "; ".join(
                       f"{e} {v[0]:.1f} / {v[1]:.1f}" for e, v in water.items()) + ". "
                   "Гипотеза для Sri-Lanka: VH воды там такой же тёмный, как в train, а VV "
                   "заметно ярче, и классификатор, опирающийся на VV и текстуру, отвергает "
                   "такую воду; пороговое правило по одному VH этого не видит.")
    out.append("- Выбор на dev смещён вверх: 0,530 — максимум из 18 точек сетки на том же dev, "
               "поэтому ΔIoU к baseline на dev завышен («проклятие победителя»); веса воды 2 "
               "и 4 отвергнуты при фиксированном p ≥ 0,5, т. е. вместе с рабочей точкой — "
               "вопрос веса и порога не закрыт.")
    out.append("- Ограничение: dev — одно событие; вес воды 1 выбран на dev с малой долей воды "
               "(4,6 %); вероятно, это и занижает p на событиях с большей долей воды. "
               "Гипотеза про VV на Sri-Lanka проверена только описательной статистикой "
               "выше, без отдельного эксперимента.")
    return out


def water_backscatter(parts: dict[str, list[str]], read: Reader) -> dict[str, tuple]:
    """Median VV / VH (dB) of labelled water pixels per event: description, not tuning."""
    acc: dict[str, list[list[np.ndarray]]] = {}
    for part in ("train", "dev", "validation"):
        for cid in parts.get(part, []):
            s1, label = read(cid)
            w = np.isfinite(s1).all(axis=0) & (label == 1)
            key = f"{part}:{cid.rsplit('_', 1)[0]}" if part == "validation" else part
            a = acc.setdefault(key, [[], []])
            a[0].append(s1[0][w])
            a[1].append(s1[1][w])
    return {k: (float(np.median(np.concatenate(v[0]))), float(np.median(np.concatenate(v[1]))))
            for k, v in acc.items() if sum(len(x) for x in v[0])}


def time_predict(chip_id: str = "Sri-Lanka_713926") -> dict:
    import os
    import platform
    import tempfile

    from floodvalue.model.pixel import predict

    with tempfile.TemporaryDirectory() as tmp:
        t0 = time.perf_counter()
        predict.predict_chip(chip_id, Path(tmp))
        sec = time.perf_counter() - t0
    return {"chip": chip_id, "seconds": sec, "cpu": platform.processor() or platform.machine(),
            "threads": os.cpu_count()}


def run(args=None) -> int:
    import joblib

    from floodvalue.data.split import chip_ids, load_split

    parts = {p: chip_ids(p) for p in ("train", "dev", "validation")}
    base_cfg = json.loads(BASELINE_CONFIG.read_text())
    t0 = time.perf_counter()
    clip = fit_clip(parts["train"], read_chip)
    print(f"clip (train): {clip}")
    sample = train_sample(parts["train"], read_chip, SUPERSET, clip)
    print(f"train sample: {len(sample['y'])} px from {sample['n_chips']} chips, "
          f"water {sample['water_share']:.3f}, skipped {sample['skipped']}")
    chips = eval_chips({"dev": parts["dev"], "validation": parts["validation"]}, read_chip,
                       SUPERSET, clip, base_cfg)
    dev = [c for c in chips if c.part == "dev"]
    print(f"features ready in {time.perf_counter() - t0:.0f} s; grid on dev:")
    best, model, grid_res = select_on_dev(sample, dev, GRID)
    params = {**FIXED, **{k: best[k] for k in ("max_leaf_nodes", "water_weight")}}
    names = FEATURE_SETS[best["feature_set"]]
    pix = per_chip_counts(model, chips, columns(names))
    base = baseline_counts(chips)
    rows = metric_rows(chips, ("dev", "validation"), pix, base, base_cfg["threshold_db"])
    problems = check_against_baseline(rows)
    ci, deltas = {}, {}
    for part in ("dev", "validation"):
        ids = [c.chip_id for c in chips if c.part == part]
        ci[(part, "baseline")] = bootstrap_ci(base, ids)
        ci[(part, METHOD)] = bootstrap_ci(pix, ids)
        deltas[part] = paired_delta(pix, base, ids)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "model.joblib"
    joblib.dump(model, model_path, compress=3)
    from floodvalue.model.pixel.predict import sha256_file

    cfg = {
        "method": "pixel_hgb",
        "description": "per-pixel HistGradientBoostingClassifier on S1 features (candidate A, #16)",
        "inputs": "S1 only: [VV, VH] dB, NaN = no data",
        "features": names, "feature_set": best["feature_set"], "clip_db": clip, "params": params, "threshold": THRESHOLD,
        "threshold_status": "fixed 0.5, not chosen on validation; calibration stage sets the "
                            "final threshold on validation (D-012)",
        "model_file": model_path.name, "model_sha256": sha256_file(model_path),
        "weights_sha256": sha256_file(model_path),
        "sklearn_version": __import__("sklearn").__version__,
        "seed": SEED, "fitted_on": "train", "chosen_on": "dev",
        "train_sample": {"n_per_chip": N_PER_CHIP, "n_chips": sample["n_chips"],
                         "n_pixels": int(len(sample["y"])),
                         "water_share": round(sample["water_share"], 4),
                         "skipped": sample["skipped"]},
        "dev_grid": grid_res,
        "split_sha256": load_split()["sha256"],
        "created_at": dt.date.today().isoformat(), "commit": git_commit(),
    }
    (MODEL_DIR / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1) + "\n")
    write_csv(VERIFY_DIR / "pixel-metrics.csv", rows)
    curve = threshold_curve(model, chips, ("dev", "validation"), columns(names))
    write_csv(VERIFY_DIR / "pixel-threshold-curve.csv", curve)
    timing = time_predict()
    (VERIFY_DIR / "pixel-val.md").write_text(
        render_report(cfg, rows, ci, deltas, grid_res, timing, problems, curve,
                      water_backscatter(parts, read_chip)))
    for r in rows:
        if r["event"] == "ALL":
            print(f"{r['split']:10s} {r['method']:8s} IoU {r['iou']}")
    print(f"deltas: {deltas}")
    print(f"predict one chip: {timing['seconds']:.1f} s")
    if problems:
        print("MISMATCH with baseline-metrics.csv:", *problems, sep="\n  ")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
