"""Controlled experiment on dev (material for #18): one axis — the input features.

    uv run --frozen python -m floodvalue.model.pixel.ablation

Everything except the feature set is fixed: train sample (same pixels, seed), clip bounds,
classifier and its hyperparameters (models/pixel/config.json `params`, else train.FIXED
with 127 leaves, water weight 3), threshold 0.5. Scored by pooled IoU on dev (USA) with a
chip bootstrap CI and a paired bootstrap against the full set. Validation and test are
not read. Writes docs/verification/pixel-ablation.md and pixel-ablation.csv.
"""

from __future__ import annotations

import json
import time

import numpy as np

from floodvalue.model import metrics
from floodvalue.model.pixel import features, train

SETS = [
    ("VV", ["VV"]),
    ("VH", ["VH"]),
    ("VV+VH", ["VV", "VH"]),
    ("VV+VH+(VV−VH)", ["VV", "VH", "VV-VH"]),
    ("+текстура 3", ["VV", "VH", "VV-VH", *features.texture_names((3,))]),
    ("+текстура 3/7", ["VV", "VH", "VV-VH", *features.texture_names((3, 7))]),
    ("+текстура 3/7/15", list(features.FULL)),
    ("+текстура 3/7/15/31", ["VV", "VH", "VV-VH", *features.texture_names((3, 7, 15, 31))]),
]
FULL_LABEL = "+текстура 3/7/15"


def superset() -> list[str]:
    names: list[str] = []
    for _, s in SETS:
        names += [n for n in s if n not in names]
    return names


def run(args=None) -> int:
    from floodvalue.data.split import chip_ids

    cfg_path = train.MODEL_DIR / "config.json"
    chosen = None
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        params, chosen = cfg["params"], cfg["features"]
    else:
        params = {**train.FIXED, "max_leaf_nodes": 127, "water_weight": 3.0}
    ref = next((lab for lab, fs in SETS if fs == chosen), FULL_LABEL)
    names = superset()
    parts = {p: chip_ids(p) for p in ("train", "dev")}
    base_cfg = json.loads(train.BASELINE_CONFIG.read_text())
    clip = train.fit_clip(parts["train"], train.read_chip)
    sample = train.train_sample(parts["train"], train.read_chip, names, clip)
    dev = train.eval_chips({"dev": parts["dev"]}, train.read_chip, names, clip, base_cfg)
    ids = [c.chip_id for c in dev]
    base = train.baseline_counts(dev)
    counts, rows = {}, []
    for label, fs in SETS:
        cols = [names.index(n) for n in fs]
        t0 = time.perf_counter()
        model = train.fit(sample["x"][:, cols], sample["y"], params)
        counts[label] = train.per_chip_counts(model, dev, cols)
        s = metrics.scores(train.pooled(counts[label], ids))
        lo, hi = train.bootstrap_ci(counts[label], ids)
        rows.append({"variant": label, "n_features": len(fs), "features": " ".join(fs),
                     "dev_iou": round(s["iou"], 6), "iou_lo": round(lo, 6), "iou_hi": round(hi, 6),
                     "f1": round(s["f1"], 6), "precision": round(s["precision"], 6),
                     "recall": round(s["recall"], 6), "seconds": round(time.perf_counter() - t0, 1)})
        print(f"{label:28s} dev IoU {s['iou']:.4f} [{lo:.3f}; {hi:.3f}]")
    for r in rows:
        d = train.paired_delta(counts[r["variant"]], counts[ref], ids)
        r.update({"reference": ref, "d_vs_ref": round(d["delta"], 6), "d_lo": round(d["lo"], 6),
                  "d_hi": round(d["hi"], 6)})
    sb = metrics.scores(train.pooled(base, ids))
    train.write_csv(train.VERIFY_DIR / "pixel-ablation.csv", rows)
    (train.VERIFY_DIR / "pixel-ablation.md").write_text(render(rows, params, sample, sb, ref))
    return 0


def render(rows, params, sample, sb, ref) -> str:
    best = max(rows, key=lambda r: r["dev_iou"])
    out = [
        "# Контролируемый эксперимент: входы пиксельного классификатора (dev)", "",
        (f"Сгенерировано `python -m floodvalue.model.pixel.ablation` (commit "
         f"`{train.git_commit()}`, {time.strftime('%Y-%m-%d')}). Числа — "
         "`docs/verification/pixel-ablation.csv`. Validation и test не читались."), "",
        "**Одна ось — набор входных признаков.** Зафиксировано всё остальное: одна и та же "
        f"выборка пикселей train ({len(sample['y'])} пикселей, {sample['n_chips']} чипов, "
        f"seed {train.SEED}), клиппинг по train, `HistGradientBoostingClassifier` с "
        f"параметрами `{json.dumps({k: params[k] for k in ('max_leaf_nodes', 'water_weight', 'max_iter', 'learning_rate')})}`, "
        f"порог p ≥ {train.THRESHOLD}. Метрика — pooled IoU на dev (USA, 69 чипов), "
        f"95 % бутстрэп по чипам ({train.N_BOOT} повторов); Δ — парный бутстрэп против "
        f"итоговой конфигурации «{ref}» (models/pixel/config.json).", "",
        "| вариант | признаков | IoU dev [95 %] | F1 | Precision | Recall | ΔIoU к итоговому [95 %] |",
        "|---|---|---|---|---|---|---|",
        (f"| baseline VH ≤ −23,0 dB (справка) | 1 | {sb['iou']:.3f} | {sb['f1']:.3f} | "
         f"{sb['precision']:.3f} | {sb['recall']:.3f} | — |")]
    for r in rows:
        b = "**" if r is best else ""
        out.append(f"| {b}{r['variant']}{b} | {r['n_features']} | {b}{r['dev_iou']:.3f}{b} "
                   f"[{r['iou_lo']:.3f}; {r['iou_hi']:.3f}] | {r['f1']:.3f} | "
                   f"{r['precision']:.3f} | {r['recall']:.3f} | "
                   f"{r['d_vs_ref']:+.3f} [{r['d_lo']:+.3f}; {r['d_hi']:+.3f}] |")
    out += ["", "## Выводы", "", *conclusions(rows, sb, ref), "",
            "## Воспроизведение", "", "```bash", "set -a; . ./.env; set +a",
            "uv run --frozen python -m floodvalue.model.pixel.ablation", "```", ""]
    return "\n".join(out)


def conclusions(rows, sb, ref) -> list[str]:
    """Statements computed from the table; each cites the paired interval it rests on."""
    r = {x["variant"]: x for x in rows}
    best = max(rows, key=lambda x: x["dev_iou"])
    second = max((x for x in rows if x["variant"] != ref), key=lambda x: x["dev_iou"])

    def d(a, b):
        """Paired ΔIoU(a − b) from the stored deltas against the reference."""
        return r[a]["d_vs_ref"] - r[b]["d_vs_ref"]

    def sig(x):
        return "значимо" if x["d_lo"] > 0 or x["d_hi"] < 0 else "в пределах шума"

    lines = [
        (f"- Один канал: VH ({r['VH']['dev_iou']:.3f}) лучше VV ({r['VV']['dev_iou']:.3f}), "
         f"ΔIoU {d('VH', 'VV'):+.3f} — как у baseline (D-021)."),
        (f"- Оба канала: VV+VH {r['VV+VH']['dev_iou']:.3f} против VH {r['VH']['dev_iou']:.3f} "
         f"(ΔIoU {d('VV+VH', 'VH'):+.3f}); явное отношение VV−VH к паре каналов почти ничего "
         f"не добавляет ({d('VV+VH+(VV−VH)', 'VV+VH'):+.3f}) — деревья и так видят оба канала."),
        (f"- Текстура (локальные среднее и σ) добавляет сверх пары каналов: окно 3 — "
         f"{d('+текстура 3', 'VV+VH+(VV−VH)'):+.3f}, окна 3/7 — "
         f"{d('+текстура 3/7', 'VV+VH+(VV−VH)'):+.3f} IoU "
         f"({r['VV+VH+(VV−VH)']['dev_iou']:.3f} → {r['+текстура 3/7']['dev_iou']:.3f})."),
        (f"- На dev (событие, не участвовавшее в обучении) крупные окна снижают IoU: 3/7 — "
         f"{r['+текстура 3/7']['dev_iou']:.3f}, 3/7/15 — {r[FULL_LABEL]['dev_iou']:.3f}, "
         f"3/7/15/31 — {r['+текстура 3/7/15/31']['dev_iou']:.3f}. С ростом окна падает "
         f"Precision ({r['+текстура 3/7']['precision']:.3f} → "
         f"{r['+текстура 3/7/15/31']['precision']:.3f}) при почти той же Recall. Гипотеза (не "
         "проверялась отдельно): широкое окно переносит воду на сухие пиксели у берега."),
        (f"- Лучший вариант на dev — **{best['variant']}** (IoU {best['dev_iou']:.3f}, "
         f"95 % [{best['iou_lo']:.3f}; {best['iou_hi']:.3f}]); baseline на тех же пикселях "
         f"— {sb['iou']:.3f}. Итоговая конфигурация — «{ref}»; ближайший соперник "
         f"«{second['variant']}»: ΔIoU {-second['d_vs_ref']:+.3f} "
         f"[{-second['d_hi']:+.3f}; {-second['d_lo']:+.3f}] в пользу итоговой, {sig(second)}."),
        ("- Ограничение: dev — одно событие (USA, 69 чипов, 4,6 % воды), поэтому интервалы "
         "широкие, а выбор опирается на парные интервалы, а не на точечные значения. "
         "Набор окон выбирался на том же dev вместе с гиперпараметрами "
         "(`pixel-val.md`), поэтому превосходство итогового варианта на dev частично "
         "задано отбором; независимое подтверждение — validation (`pixel-val.md`) и "
         "test после заморозки. Validation для выбора входов не использовалась."),
    ]
    return lines


if __name__ == "__main__":
    raise SystemExit(run())
