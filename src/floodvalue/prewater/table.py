"""`python -m floodvalue.prewater.table` — how the pre-event layer relates to the hand labels on
every train and validation chip (#149 p.5) -> docs/verification/prewater/.

Test chips are never read (their labels are sealed until eval-test). The parameters of the
layer are fixed in core.py before this table; nothing here selects or tunes anything.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rasterio

from floodvalue.data import io
from floodvalue.data.split import chip_ids, event_of
from floodvalue.paths import REPO_ROOT, output_dir
from floodvalue.prewater import core
from floodvalue.prewater import run as pw

PARTS = ("train", "validation")
OUT_DIR = REPO_ROOT / "docs" / "verification" / "prewater"
COUNT_KEYS = ("valid_px", "perm_px", "label_water_px", "label_dry_px", "perm_and_water_px",
              "perm_and_dry_px", "jrc_px", "perm_and_jrc_px", "perm_or_jrc_px")


def chip_row(chip_id: str, work: Path) -> dict:
    """Build (or reuse) the layer of one chip and compare it with LabelHand and JRCWaterHand."""
    meta_path = work / pw.META_FILE
    meta = (json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file()
            and (work / pw.PERM_FILE).is_file() else pw.build(chip_id, work))
    row = {"chip_id": chip_id, "event": event_of(chip_id), "status": meta["status"],
           "scenes": len(meta.get("scenes", [])), "failed_scenes": len(meta.get("failed_scenes", []))}
    if meta["status"] != core.STATUS_OK:
        return row | {k: None for k in COUNT_KEYS}
    with rasterio.open(work / pw.PERM_FILE) as r:
        perm = r.read(1)
    label, _ = io.load_label(chip_id)  # train/validation only: raises for test events
    with rasterio.open(io.layer_path(chip_id, io.JRC_LAYER)) as r:
        jrc = r.read(1)
    return row | core.diagnostics(label, perm, jrc)


def pooled(rows: list[dict]) -> dict:
    ok = [r for r in rows if r["status"] == core.STATUS_OK]
    counts = {k: sum(int(r[k]) for r in ok) for k in COUNT_KEYS}
    return {"chips": len(rows), "chips_ok": len(ok),
            "scenes_median": float(np.median([r["scenes"] for r in rows])) if rows else None,
            **counts, **core.ratios(counts)}


def _fmt(v, nd=3):
    return "—" if v is None else f"{v:.{nd}f}".replace(".", ",")


def render_md(groups: dict[str, dict]) -> str:
    lines = [
        "# Вода до события по архиву Sentinel-1 — диагностика на train и validation (#149)",
        "",
        f"Слой `pre_event_water` — вода (VH ≤ {core.VH_WATER_DB} dB, D-021) не менее чем в "
        f"{core.MIN_WATER_SHARE:.0%} наблюдений за {core.WINDOW_DAYS} сут, окно кончается за "
        f"{core.GUARD_DAYS} сут до даты S1 события; < {core.N_MIN_SCENES} сцен — `insufficient`. "
        "Параметры зафиксированы до таблицы и по ней не подбирались. Test не читался; "
        "dev (USA) в таблицу не входит — по условию #149 только train и validation.",
        "",
        "Команда: `uv run --frozen python -m floodvalue.prewater.table`. "
        "Построчно по чипам — `prewater-chips.csv`, сводка — `prewater-summary.json`.",
        "",
        "| Событие | Чипов (ok) | Сцен (медиана) | Доля «воды до» | P(метка=1 \\| до) | "
        "«до» на сухом | Снятая доля воды метки | IoU с JRC |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, g in groups.items():
        lines.append(f"| {name} | {g['chips']} ({g['chips_ok']}) | {_fmt(g['scenes_median'], 0)} | "
                     f"{_fmt(g['perm_share'])} | {_fmt(g['p_water_given_perm'])} | "
                     f"{_fmt(g['perm_on_dry_share'])} | {_fmt(g['removed_water_share'])} | "
                     f"{_fmt(g['iou_jrc'])} |")
    lines += [
        "",
        "Как читать:",
        "- **P(метка=1 | до)** — как часто «вода до события» — вода и на метке события; низкое "
        "значение — ложная «вода до» (пески, сезонная вода, пересохшее).",
        "- **«до» на сухом** — доля сухих пикселей метки, помеченных водой до события.",
        "- **Снятая доля воды метки** — какая часть воды метки уходит из «новой воды»: это вода, "
        "что была и до окна (реки, водоёмы), а не затопление относительно окна.",
        "- **IoU с JRC** — только диагностика: JRC набора, вероятно, собран и по данным после "
        "событий 2016–2018 (CLAUDE.md), поэтому он не эталон и не вход.",
        "",
        "Ограничения слоя — `pre_event_water.json` (`caveats`) и D-053.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="floodvalue.prewater.table")
    p.add_argument("--work", default=None, help="каталог слоёв по чипам (по умолчанию outputs/prewater)")
    p.add_argument("--out", default=str(OUT_DIR))
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--events", default="", help="через запятую: только эти события (проверка)")
    a = p.parse_args(argv)
    work = Path(a.work) if a.work else output_dir() / "prewater"
    only = {e.strip() for e in a.events.split(",") if e.strip()}
    chips = [c for part in PARTS for c in chip_ids(part) if not only or event_of(c) in only]
    with ThreadPoolExecutor(a.workers) as ex:
        rows = list(ex.map(lambda c: chip_row(c, work / c), chips))
    rows.sort(key=lambda r: r["chip_id"])
    groups = {e: pooled([r for r in rows if r["event"] == e])
              for e in sorted({r["event"] for r in rows})}
    for part in PARTS:
        events = {event_of(c) for c in chip_ids(part)}
        groups[part.upper()] = pooled([r for r in rows if r["event"] in events])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "prewater-chips.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows([{k: ("" if v is None else v) for k, v in r.items()} for r in rows])
    summary = {"params": {"vh_water_db": core.VH_WATER_DB, "min_water_share": core.MIN_WATER_SHARE,
                          "n_min_scenes": core.N_MIN_SCENES, "window_days": core.WINDOW_DAYS,
                          "guard_days": core.GUARD_DAYS},
               "parts": list(PARTS), "groups": groups}
    (out / "prewater-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                                               encoding="utf-8")
    (out / "prewater-table.md").write_text(render_md(groups), encoding="utf-8")
    print(render_md(groups))
    return 0


if __name__ == "__main__":
    sys.exit(main())
