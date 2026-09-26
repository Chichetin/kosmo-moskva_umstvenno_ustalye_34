"""`python -m floodvalue prewater --chip <id> [--out <dir>]` — pre-event water layer (#149).

Writes pre_event_water.tif / .json and, when flood_probability.tif is already in the chip's
working directory, new_water_probability.tif. No network, HTTP 429 or too few scenes is not an
error: the JSON records the status, a warning is printed and the exit code is 0, so `make all`
and the 12 files never depend on the archive.

Only the S1 chip grid and its date are read — never labels, so a test chip is fine here.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import rasterio

from floodvalue.data.io import layer_path, parse_chip_id, raw_dir
from floodvalue.data.split import METADATA_NAME
from floodvalue.paths import REPO_ROOT, output_dir
from floodvalue.prewater import core, stac
from floodvalue.prewater.stac import COLLECTION, COLLECTION_URL

METADATA = "Sen1Floods11_Metadata.geojson"
METADATA_FALLBACK = REPO_ROOT / "docs" / "sen1floods11" / METADATA
PERM_FILE = "pre_event_water.tif"
META_FILE = "pre_event_water.json"
NEW_WATER_FILE = "new_water_probability.tif"
P_FILE = "flood_probability.tif"
# Upper bound of the whole step: the jury runs `make all` itself and a hanging network must not
# hang it. Past the deadline no more scenes are read; too few scenes -> `unavailable`.
DEADLINE_S = 300

ROLE = "pre_event_context"
MEANING = ("вода на снимках Sentinel-1 до события (окно архива до даты S1 чипа); "
           "«новая вода» = p × (1 − pre_event_water) — вода относительно этого окна, "
           "не доказанное временное затопление")
CAVEATS = [
    "дата начала паводка в наборе нет: окно кончается за фиксированный зазор до даты S1 — "
    "паводок, начавшийся раньше зазора, попадёт в «воду до события»",
    "сезонная вода: окно — другой сезон, чем событие; водоём, пересохший к окну, станет «новым»",
    "сухой песок и аридные равнины тёмные в VH — ложная «вода до события» (Somalia: "
    "P(метка=1 | до) 0,021 на train, docs/verification/prewater/prewater-table.md)",
    "мало сцен в 2016 г. (один спутник): статус insufficient при < N_min сцен",
    "gamma0 RTC Planetary Computer против sigma0 GRD Sen1Floods11: порог D-021 перенесён без "
    "подбора, разница радиометрии не учтена",
]


def s1_date_of(event_id: str) -> date:
    """S1 acquisition date of the event from Sen1Floods11_Metadata.geojson."""
    path = raw_dir() / METADATA
    if not path.is_file():
        path = METADATA_FALLBACK
    location = METADATA_NAME.get(event_id, event_id)
    for f in json.loads(path.read_text(encoding="utf-8"))["features"]:
        p = f["properties"]
        if p.get("location") == location and p.get("s1_date"):
            return date(*map(int, p["s1_date"].split("/")))
    raise KeyError(f"нет s1_date события {event_id} в {path}")


def _grid(chip_id: str) -> dict:
    with rasterio.open(layer_path(chip_id, "S1Hand")) as r:
        return {"crs": r.crs, "transform": r.transform, "width": r.width, "height": r.height,
                "bounds": tuple(r.bounds)}


def _write(path: Path, arr: np.ndarray, grid: dict, nodata) -> None:
    profile = {"driver": "GTiff", "count": 1, "dtype": str(arr.dtype), "nodata": nodata,
               "compress": "deflate", **{k: grid[k] for k in ("crs", "transform", "width", "height")}}
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr, 1)


def source_entry(chip_id: str, scenes: list[dict], license_: str, s1: date, accessed: date) -> dict:
    """SourceEntry of source_manifest.json (contracts §9) for the archive scenes of the chip."""
    last = max(s["date"] for s in scenes)
    return {
        "id": f"pc:sentinel-1-rtc:{chip_id}",
        "file": PERM_FILE,
        "url": COLLECTION_URL,
        "version": f"Planetary Computer STAC {COLLECTION}, {len(scenes)} сцен(ы) "
                   f"{min(s['date'] for s in scenes)}…{last}",
        "observation_date": last,
        "available_date": accessed.isoformat(),
        "license": license_,
        "purpose": "вода до события — слой «новая вода» (доп. файлы, не вход p/маски/EL)",
        "limitations": "; ".join(CAVEATS),
        "applies_to": f"{PERM_FILE}, {NEW_WATER_FILE}",
        "after_event_note": (f"снимки S1 — до события {s1.isoformat()} (последний {last}); "
                             "RTC-обработка Planetary Computer выполнена позже по этим же "
                             "наблюдениям до события, новых сведений о событии не несёт"),
    }


def build(chip_id: str, out: Path, *, archive=None, today: date | None = None,
          deadline_s: float = DEADLINE_S) -> dict:
    """Compute and write the layer for `chip_id` into `out`; returns the JSON written.

    `archive` — the STAC module or a stand-in with the same functions (tests)."""
    archive = archive or stac
    t_end = time.monotonic() + deadline_s
    event_id, _ = parse_chip_id(chip_id)
    s1 = s1_date_of(event_id)
    first, last = core.pre_event_window(s1)
    grid = _grid(chip_id)
    w, s, e, n = grid["bounds"]
    meta = {
        "chip_id": chip_id, "role": ROLE, "used_by_model": False, "used_by_economics": False,
        "meaning": MEANING, "s1_date": s1.isoformat(),
        "window": {"first": first.isoformat(), "last": last.isoformat(),
                   "days": core.WINDOW_DAYS, "guard_days": core.GUARD_DAYS},
        "params": {"vh_water_db": core.VH_WATER_DB, "min_water_share": core.MIN_WATER_SHARE,
                   "n_min_scenes": core.N_MIN_SCENES, "rule": "D-021, фиксированы до таблицы"},
        "archive": COLLECTION_URL, "status": core.STATUS_UNAVAILABLE, "scenes": [],
        "files": [], "sources": [], "caveats": CAVEATS, "warning": None,
    }
    out.mkdir(parents=True, exist_ok=True)
    for name in (PERM_FILE, NEW_WATER_FILE):  # never leave a layer of an earlier run behind
        (out / name).unlink(missing_ok=True)
    try:
        license_ = archive.collection_license()
        items = archive.search((w, s, e, n), first, last)
        # single-polarisation (VV-only / HH) scenes have no VH asset: not an observation here
        groups = archive.acquisitions([it for it in items if "vh" in it.get("assets", {})])
        token = archive.sas_token()
        stack, scenes, failed = [], [], []
        for group in groups:
            if time.monotonic() > t_end:  # keep what was read; too few -> insufficient below
                meta["deadline_hit"] = (f"лимит шага {deadline_s:.0f} с: прочитано сцен "
                                        f"{len(scenes)} из {len(groups)}")
                break
            vh = np.full((grid["height"], grid["width"]), np.nan, dtype=np.float32)
            try:
                for it in group:
                    part = core.linear_to_db(archive.read_vh(
                        it["assets"]["vh"]["href"], token, grid["transform"],
                        (grid["height"], grid["width"])))
                    vh = np.where(np.isfinite(vh), vh, part)
            except Exception as exc:  # noqa: BLE001 — unreadable scene (HTTP, GDAL, decoding)
                failed.append({"id": [it["id"] for it in group], "error": str(exc)[:300]})
                continue  # one unreadable scene is a missing observation, not a failed layer
            if not np.isfinite(vh).any():
                continue  # footprint touches the bbox but not the chip
            stack.append(vh)
            p0 = group[0]["properties"]
            scenes.append({"id": [it["id"] for it in group], "date": p0["datetime"][:10],
                           "datetime": p0["datetime"], "platform": p0.get("platform"),
                           "orbit_state": p0.get("sat:orbit_state"),
                           "relative_orbit": p0.get("sat:relative_orbit"),
                           "orbit_source": p0.get("s1:orbit_source"),
                           "processing_level": p0.get("s1:processing_level"),
                           "product_timeliness": p0.get("s1:product_timeliness"),
                           # the STAC items carry no publication time of the RTC product
                           "published_at": None,
                           "published_at_note": "в STAC-объекте нет даты публикации RTC",
                           "license": license_})
    except Exception as exc:  # noqa: BLE001 — any archive failure must leave `make all` green
        meta["warning"] = f"архив недоступен: {type(exc).__name__}: {exc}"[:500]
        _dump(out, meta)
        return meta
    meta["scenes"] = scenes
    meta["failed_scenes"] = failed
    meta["status"] = core.layer_status(len(scenes))
    if meta["status"] != core.STATUS_OK:
        meta["warning"] = (f"{len(scenes)} сцен(ы) в окне < N_min = {core.N_MIN_SCENES}"
                           + (f"; {meta['deadline_hit']}" if meta.get("deadline_hit") else ""))
        _dump(out, meta)
        return meta
    perm = core.permanent_water(np.stack(stack))
    _write(out / PERM_FILE, perm, grid, core.PERM_NODATA)
    meta["files"].append(PERM_FILE)
    known = perm != core.PERM_NODATA
    meta["pre_event_water_share"] = float((perm == 1).sum() / known.sum()) if known.any() else None
    if (out / P_FILE).is_file():
        with rasterio.open(out / P_FILE) as r:
            p = r.read(1)
        _write(out / NEW_WATER_FILE, core.new_water_probability(p, perm), grid, core.P_NODATA)
        meta["files"].append(NEW_WATER_FILE)
    meta["sources"] = [source_entry(chip_id, scenes, license_, s1, today or date.today())]
    _dump(out, meta)
    return meta


def _dump(out: Path, meta: dict) -> None:
    (out / META_FILE).write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")


def run(args) -> int:
    if not args.chip:
        print("prewater: нужен --chip <chip_id>", file=sys.stderr)
        return 2
    out = Path(args.out) if args.out else output_dir() / args.chip
    meta = build(args.chip, out)
    if meta["status"] == core.STATUS_OK:
        print(f"prewater: {out / PERM_FILE} — {len(meta['scenes'])} сцен "
              f"{meta['window']['first']}…{meta['window']['last']}, вода до события "
              f"{meta['pre_event_water_share']:.3f}; файлы: {', '.join(meta['files'])}")
    else:
        print(f"prewater: ПРЕДУПРЕЖДЕНИЕ — слой не построен ({meta['status']}): {meta['warning']}. "
              "12 файлов от этого не зависят", file=sys.stderr)
    return 0
