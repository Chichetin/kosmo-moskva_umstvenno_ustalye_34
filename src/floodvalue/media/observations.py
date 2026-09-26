"""Satellite observations for an event: found → acquired → prepared → analysed are separate
stages, and a catalog record is not a check. Place AND time must fit: a 2017 chip does not
confirm today's news, and no image is not a refutation.

Local catalog: Sen1Floods11 S1 chips of train/dev/validation events (test events stay sealed;
only raster bounds are read, never labels). External: Copernicus Data Space OData catalogue
(anonymous search); its GRD scenes need preparation the project does not implement yet."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

from shapely.geometry import shape

from floodvalue.media.config import split_parts
from floodvalue.media.geo import geometry_bbox
from floodvalue.media.netfetch import Fetcher
from floodvalue.media.store import dumps, get_kv, loads, now_iso, set_kv

EARLY_TOLERANCE = timedelta(days=1)
LATE_TOLERANCE = timedelta(days=7)
NEARBY_DEG = 0.5
COARSE = ("district", "region", "country")
CDSE_PRODUCTS = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
CDSE_HOST = "catalogue.dataspace.copernicus.eu"
NOT_PREPARED = ("подготовка сцены GRD к формату метода (калибровка, коррекция рельефа, dB, "
                "сетка чипа) в проекте не реализована — анализ этой сцены недоступен")


def _ts(value: str | None) -> datetime | None:
    if not value:
        return None
    value = value.replace("Z", "+00:00")
    dt = datetime.fromisoformat(value if "T" in value else value + "T00:00:00+00:00")
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def assess(event_geometry: dict | None, start: str | None, end: str | None,
           footprint: dict | None, observed: str | None,
           precision: str = "point") -> tuple[str, float | None, str, str]:
    """(spatial_relation, time_offset_h, suitability, reason). Pure function. A point that only
    stands for a district/region/country (e.g. a GDACS centroid) can't be "covered": covering
    the centroid says nothing about the flooded area, so it is at most a partial cover."""
    if not event_geometry or not footprint:
        relation = "unknown"
    else:
        ev, fp = shape(event_geometry), shape(footprint)
        relation = ("covers" if fp.covers(ev) else "intersects" if fp.intersects(ev)
                    else "nearby" if fp.distance(ev) <= NEARBY_DEG else "none")
        if relation == "covers" and ev.geom_type == "Point" and precision in COARSE:
            relation = "intersects"
    s, e, obs = _ts(start), _ts(end) or _ts(start), _ts(observed)
    if relation in ("none", "unknown"):
        return relation, None, "unusable", "наблюдение не покрывает место события"
    if s is None or obs is None:
        return (relation, None, "unknown_time",
                "время события неизвестно — совпадение по времени проверить нельзя")
    offset_h = round((obs - s).total_seconds() / 3600, 1)
    if obs < s - EARLY_TOLERANCE:
        return (relation, offset_h, "too_early",
                f"снимок на {abs(offset_h) / 24:.0f} сут раньше события — текущую воду не наблюдает")
    if obs > e + LATE_TOLERANCE:
        return relation, offset_h, "too_late", "снимок позже окна события более чем на 7 сут"
    if relation == "covers":
        return relation, offset_h, "suitable", "покрывает место и попадает в окно события"
    if relation == "intersects":
        why = ("место известно только до района/региона — снимок покрывает точку-заместитель, "
               "а не всю область" if precision in COARSE else "покрывает место события частично")
        return relation, offset_h, "partial_cover", why
    return relation, offset_h, "unusable", "снимок рядом, но место события не покрывает"


def local_catalog(conn: sqlite3.Connection, data_dir: Path) -> list[dict]:
    """Chips outside the sealed test events with their footprints (bounds cached in kv)."""
    manifest_path = data_dir / "interim" / "source_manifest_s1.json"
    if not manifest_path.exists():
        return []
    key = f"local_catalog_v1:{manifest_path.stat().st_mtime_ns}"
    cached = get_kv(conn, key)
    if cached:
        return json.loads(cached)
    import rasterio  # only raster bounds, read lazily
    sealed = set(split_parts()["test"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    out = []
    for src in manifest["sources"]:
        chip, event = src.get("chip_id"), src.get("event_id")
        if not chip or event in sealed:
            continue
        tif = data_dir / "raw" / "HandLabeled" / "S1Hand" / f"{chip}_S1Hand.tif"
        if not tif.exists():
            continue
        with rasterio.open(tif) as ds:
            w, s, e, n = ds.bounds
        out.append({"ref": chip, "event_id": event, "observed": src["observation_date"],
                    "footprint": {"type": "Polygon", "coordinates": [[[w, s], [e, s], [e, n],
                                                                      [w, n], [w, s]]]}})
    set_kv(conn, key, json.dumps(out))
    return out


def analysed_runs(output_dir: Path) -> dict[str, str]:
    """chip_id → run directory name of a finished FloodValue analysis (read-only)."""
    runs: dict[str, str] = {}
    if not output_dir.exists():
        return runs
    for meta in sorted(output_dir.glob("*/run_metadata.json")):
        try:
            chip = json.loads(meta.read_text(encoding="utf-8")).get("chip_id")
        except (OSError, ValueError):
            continue
        if chip:
            runs[chip] = meta.parent.name
    return runs


def _event_window(event) -> tuple[str | None, str | None]:
    return event["event_time_start"], event["event_time_end"] or event["event_time_start"]


def search_local(conn, event, data_dir: Path, output_dir: Path) -> list[dict]:
    geometry = loads(event["geometry_json"])
    start, end = _event_window(event)
    runs = analysed_runs(output_dir)
    found = []
    for chip in local_catalog(conn, data_dir):
        relation, offset, suitability, reason = assess(geometry, start, end, chip["footprint"],
                                                       chip["observed"], event["geo_precision"])
        if relation in ("none", "unknown"):
            continue
        run = runs.get(chip["ref"])
        found.append({"catalog": "sen1floods11_local", "observation_ref": chip["ref"],
                      "product_type": "S1 GRD чип Sen1Floods11 (VV/VH, dB)",
                      "observation_start": chip["observed"], "footprint": chip["footprint"],
                      "spatial_relation": relation, "time_offset_h": offset,
                      "suitability": suitability, "reason": reason,
                      "stage": "analyzed" if run else "prepared", "analysis_run_id": run})
    return found


def search_cdse(fetcher: Fetcher, event, limit: int = 20) -> list[dict]:
    geometry = loads(event["geometry_json"])
    if not geometry:
        return []
    start, end = _event_window(event)
    anchor_s = _ts(start) or _ts(event["first_detected_at"])
    anchor_e = _ts(end) or anchor_s
    lo = anchor_s - timedelta(days=3)
    hi = min(anchor_e + timedelta(days=7), datetime.now(UTC))
    w, s, e, n = geometry_bbox(geometry)
    wkt = (f"POINT({w} {s})" if (w, s) == (e, n) else
           f"POLYGON(({w} {s},{e} {s},{e} {n},{w} {n},{w} {s}))")
    flt = (f"Collection/Name eq 'SENTINEL-1' and contains(Name,'GRD') and "
           f"OData.CSC.Intersects(area=geography'SRID=4326;{wkt}') and "
           f"ContentDate/Start gt {lo:%Y-%m-%dT%H:%M:%S.000Z} and "
           f"ContentDate/Start lt {hi:%Y-%m-%dT%H:%M:%S.000Z}")
    resp = fetcher.get(CDSE_PRODUCTS, params={"$filter": flt, "$top": str(limit),
                                              "$orderby": "ContentDate/Start asc"})
    out = []
    for p in resp.json().get("value", []):
        footprint = p.get("GeoFootprint")
        observed = (p.get("ContentDate") or {}).get("Start")
        relation, offset, suitability, reason = assess(geometry, start, end, footprint, observed,
                                                       event["geo_precision"])
        if relation == "none":
            continue
        reason = f"{reason}; {NOT_PREPARED}"
        if not p.get("Online", True):
            reason += "; сцена в архиве (offline), получение не мгновенное"
        out.append({"catalog": "cdse", "observation_ref": p["Name"],
                    "product_type": "Sentinel-1 GRD", "observation_start": observed,
                    "footprint": footprint, "spatial_relation": relation,
                    "time_offset_h": offset, "suitability": suitability, "reason": reason,
                    "stage": "found", "analysis_run_id": None})
    return out


def save_matches(conn, event_id: int, matches: list[dict]) -> None:
    for m in matches:
        conn.execute(
            """INSERT INTO observation_matches(event_id, catalog, observation_ref, product_type,
                 observation_start, footprint_json, spatial_relation, time_offset_h, suitability,
                 reason, stage, analysis_run_id, found_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(event_id, catalog, observation_ref) DO UPDATE SET
                 spatial_relation=excluded.spatial_relation, time_offset_h=excluded.time_offset_h,
                 suitability=excluded.suitability, reason=excluded.reason, stage=excluded.stage,
                 analysis_run_id=excluded.analysis_run_id, found_at=excluded.found_at""",
            (event_id, m["catalog"], m["observation_ref"], m["product_type"],
             m["observation_start"], dumps(m["footprint"]), m["spatial_relation"],
             m["time_offset_h"], m["suitability"], m["reason"], m["stage"],
             m["analysis_run_id"], now_iso()))
