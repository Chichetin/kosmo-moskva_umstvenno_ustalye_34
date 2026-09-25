"""`make data-qc`: docs/verification/data-qc.md from $FLOODVALUE_DATA_DIR/interim/qc.csv and
interim/source_manifest_s1.json (both written by `make data`) and splits/split_v1.json.

Pixel statistics (label shares -1/0/1, S1 NaN, VV > 0 dB) are aggregated only over
train/dev/validation/excluded. Test chips are sealed (D-010, D-031): `make data` never reads
their pixels, and the summary refuses a qc.csv where a test chip has pixel statistics.
All chips are 512x512, so the mean of chip shares equals the pixel share of the group.
"""

import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from floodvalue.data.split import SPLIT_PATH, load_split
from floodvalue.paths import REPO_ROOT, data_dir

REPORT_PATH = REPO_ROOT / "docs" / "verification" / "data-qc.md"
OPEN_ORDER = ("train", "dev", "validation", "excluded")
PIXEL_KEYS = ("label_invalid_frac", "label_dry_frac", "label_water_frac", "vv_nan_frac",
              "vh_nan_frac", "vv_pos_frac", "vh_pos_frac", "vv_max", "valid_frac",
              "s1_nan_label_dry_frac", "s1_nan_label_water_frac")


def chip_parts(split: dict) -> dict[str, str]:
    parts = {c: p for p in ("train", "dev", "validation", "test") for c in split.get(p, [])}
    parts.update({c: "excluded" for c in split.get("excluded", {})})
    return parts


def _f(r: dict, key: str) -> float:
    return float(r[key])


def _true(v) -> bool:
    return str(v) == "True"


def _integrity(rows: list[dict]) -> dict:
    return {"n_chips": len(rows),
            "qc_pass": sum(_true(r["qc_pass"]) for r in rows),
            "md5_ok": sum(_true(r["s1_md5_ok"]) and _true(r["label_md5_ok"]) for r in rows),
            "grid_match": sum(_true(r["grid_match"]) for r in rows)}


def _group(name: str, rows: list[dict]) -> dict:
    n = len(rows)
    mean = lambda k: sum(_f(r, k) for r in rows) / n  # noqa: E731
    dry, water = sum(_f(r, "label_dry_frac") for r in rows), sum(_f(r, "label_water_frac")
                                                                 for r in rows)
    return {"group": name, "n_chips": n,
            "invalid": mean("label_invalid_frac"), "dry": mean("label_dry_frac"),
            "water": mean("label_water_frac"),
            "water_of_labelled": water / (dry + water) if dry + water > 0 else None,
            "nan": mean("vv_nan_frac"), "vv_pos": mean("vv_pos_frac"),
            "nan_dry": mean("s1_nan_label_dry_frac"),
            "nan_water": mean("s1_nan_label_water_frac"),
            "chips_nan_labelled": sum(_f(r, "s1_nan_label_dry_frac")
                                      + _f(r, "s1_nan_label_water_frac") > 0 for r in rows),
            "chips_with_nan": sum(_f(r, "vv_nan_frac") > 0 for r in rows),
            "chips_nan_gt_half": sum(_f(r, "vv_nan_frac") > 0.5 for r in rows),
            "chips_vv_pos": sum(_f(r, "vv_pos_frac") > 0 for r in rows),
            "chips_no_water": sum(_f(r, "label_water_frac") == 0 for r in rows),
            "chips_invalid_gt_half": sum(_f(r, "label_invalid_frac") > 0.5 for r in rows),
            "unusable": sum(r["usable"] == "False" for r in rows),
            **_integrity(rows)}


def summarize(rows: list[dict], split: dict) -> dict:
    """Pure summary of qc.csv rows. Raises ValueError for a chip outside the split or for a
    test chip that carries pixel statistics (sealing broken upstream)."""
    parts = chip_parts(split)
    unknown = sorted(r["chip_id"] for r in rows if r["chip_id"] not in parts)
    if unknown:
        raise ValueError(f"чипы вне split: {unknown}")
    by_part: dict[str, list[dict]] = {}
    for r in rows:
        by_part.setdefault(parts[r["chip_id"]], []).append(r)
    test = by_part.get("test", [])
    leaked = sorted(r["chip_id"] for r in test
                    if _true(r.get("s1_checked")) or _true(r.get("label_checked"))
                    or any(r.get(k, "") != "" for k in PIXEL_KEYS))
    if leaked:
        raise ValueError(f"у test-чипов есть пиксельная статистика (запечатывание нарушено): "
                         f"{leaked}")
    events: dict[tuple[str, str], list[dict]] = {}
    for p in ("train", "dev", "validation"):
        for r in by_part.get(p, []):
            events.setdefault((p, r["event_id"]), []).append(r)
    return {
        "total": _integrity(rows),
        "parts": [_group(p, by_part[p]) for p in OPEN_ORDER if by_part.get(p)],
        "events": [{**_group(e, rs), "part": p} for (p, e), rs in sorted(events.items())],
        "test": {**_integrity(test), "pixels_read": 0,
                 "events": sorted({r["event_id"] for r in test})},
    }


def _pct(x: float | None) -> str:
    return "" if x is None else f"{100 * x:.2f}"


def _row(g: dict, first: str) -> str:
    return (f"| {first} | {g['n_chips']} | {_pct(g['invalid'])} | {_pct(g['dry'])} | "
            f"{_pct(g['water'])} | {_pct(g['water_of_labelled'])} | {_pct(g['nan'])} | "
            f"{_pct(g['vv_pos'])} | {_pct(g['nan_dry'])} | {_pct(g['nan_water'])} | "
            f"{g['chips_nan_labelled']} | {g['chips_with_nan']} | {g['chips_nan_gt_half']} | "
            f"{g['chips_vv_pos']} | {g['chips_no_water']} | {g['chips_invalid_gt_half']} | "
            f"{g['unusable']} |")


HEAD = ("| {0} | чипов | −1, % | 0, % | 1, % | вода среди размеченных, % | NaN S1, % | "
        "VV > 0 dB, % | NaN S1 под 0, % | NaN S1 под 1, % | чипов с NaN под 0/1 | "
        "чипов с NaN | NaN > 50 % | чипов с VV > 0 | без воды | −1 > 50 % | "
        "без валидных |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")


def render(summary: dict, events: list[dict], header: dict) -> str:
    t, tt = summary["total"], summary["test"]
    lines = [
        "# QC данных Sen1Floods11 v1.1 HandLabeled (критерий 1)",
        "",
        f"Собрано командой `{header['command']}` на commit `{header['commit']}`"
        + (f", {header['date']}" if header.get("date") else "") + ". Источник чисел — "
        "`$FLOODVALUE_DATA_DIR/interim/qc.csv` и `interim/source_manifest_s1.json` "
        "(`make data`), разбиение — `splits/split_v1.json`. Файл перезаписывается командой, "
        "руками не правится.",
        "",
        "## Целостность и формат — все чипы",
        "",
        "| Проверка | Чипов |",
        "|---|---|",
        f"| всего ручных чипов | {t['n_chips']} |",
        f"| md5 S1 и метки = md5 объекта в бакете | {t['md5_ok']} |",
        f"| сетка S1 = сетка метки (CRS, размер, allclose transform) | {t['grid_match']} |",
        f"| формат пройден (`qc_pass`: 2 канала VV/VH float32 nodata NaN, метка int16, "
        f"EPSG:4326, md5, масштаб dB у открытых чипов) | {t['qc_pass']} |",
        "",
        "## Test запечатан",
        "",
        f"Test — {tt['n_chips']} чипов событий {', '.join(tt['events'])} "
        "(`splits/split_v1.json`, D-016). У них `make data` проверяет только заголовки "
        "(каналы, dtype, nodata, имена, CRS, сетка) и md5/sha256 S1 и метки; **пиксели ни S1, "
        "ни метки не читаются** (D-010, D-031), поэтому ниже для test нет ни одной "
        "пиксельной величины. Оговорка: до разбиения, на разведке набора, сводные доли меток "
        "всех 446 чипов, включая будущий test, один раз смотрели (`docs/data.md` §5, D-016); "
        "разбиение построено правилом по событиям, а не по этим долям. Пиксельный QC test "
        "(NaN, −1, VV > 0 dB, NaN × метка) — задача `eval-test` после заморозки. "
        f"Формат пройден: {tt['qc_pass']}/{tt['n_chips']}, md5: {tt['md5_ok']}/{tt['n_chips']}, "
        f"сетка: {tt['grid_match']}/{tt['n_chips']}. Сводная этого отчёта отказывает, если у "
        "test-чипа в `qc.csv` появилась пиксельная статистика.",
        "",
        "## Доли меток, NaN и VV > 0 dB по частям разбиения",
        "",
        "Доли −1/0/1, NaN и VV > 0 dB — от всех 512×512 пикселей чипа, среднее по чипам "
        "(= доля пикселей части). NaN — по VV (NaN в VV и VH совпадают, `docs/data.md` §11). "
        "«Вода среди размеченных» = Σ(1) / Σ(0 + 1). «Без валидных» — нет ни одного пикселя с "
        "конечным S1 и меткой ≠ −1. «NaN S1 под 0/1» — S1 не конечен хотя бы в одном канале, "
        "а метка 0 или 1 (ловушка набора: маска валидности = S1 конечен **и** метка ≠ −1). "
        "VV > 0 dB — доля от всех пикселей (NaN не > 0). "
        "`excluded` — 5 чипов Ghana, целиком −1 (D-016).",
        "",
        HEAD.format("Часть"),
        *(_row(g, g["group"]) for g in summary["parts"]),
        "",
        "## По событиям (train/dev/validation)",
        "",
        HEAD.format("Событие (часть)"),
        *(_row(g, f"{g['group']} ({g['part']})") for g in summary["events"]),
        "",
        "## Даты событий и лицензия",
        "",
        "Даты — поля `s1_date`/`s2_date` файла `Sen1Floods11_Metadata.geojson` набора v1.1 "
        "(https://storage.googleapis.com/sen1floods11/v1.1/Sen1Floods11_Metadata.geojson, "
        "md5 сверяется с бакетом). Даты — метаданные, не метки, поэтому показаны и для test. "
        "В `source_manifest_s1.json` дата S1 записана в `observation_date` и `event_date` "
        "каждого чипа, таблица — в `events`.",
        "",
        "| Событие | В метаданных | S1 | S2 | чипов |",
        "|---|---|---|---|---|",
        *(f"| {e['event_id']} | {e['metadata_location']} | {e['s1_date']} | "
          f"{e.get('s2_date') or ''} | {e['n_chips']} |" for e in events),
        "",
        f"Лицензия: {header.get('license', '')}",
        "",
        f"{header.get('license_note', '')}",
        "",
    ]
    return "\n".join(lines)


def _commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short=7", "HEAD"], cwd=REPO_ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    out = Path(argv[0]) if argv else REPORT_PATH
    interim = data_dir() / "interim"
    with open(interim / "qc.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    manifest = json.loads((interim / "source_manifest_s1.json").read_text(encoding="utf-8"))
    summary = summarize(rows, load_split(SPLIT_PATH))
    header = {"command": "make data && make data-qc", "commit": _commit(),
              "date": datetime.now(timezone.utc).date().isoformat(),
              "license": manifest.get("license", ""),
              "license_note": manifest.get("license_note", "")}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(summary, manifest.get("events", []), header), encoding="utf-8")
    t = summary["total"]
    print(f"data-qc: {t['n_chips']} чипов, формат {t['qc_pass']}/{t['n_chips']}, "
          f"test запечатан ({summary['test']['n_chips']} чипов) -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
