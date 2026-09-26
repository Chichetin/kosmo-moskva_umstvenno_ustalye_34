"""Find satellite observations for an event: place AND time, never place alone (spec §З).

The local catalog is the Sen1Floods11 S1 chips listed in the project manifest. Test-split
chips are skipped (the test set stays sealed until eval-test) and no label is read. A
found record is only 'found'/'prepared'; it becomes 'analyzed' only if a result bundle for
that chip already exists in outputs/. Nothing here runs or changes the model.
"""

from __future__ import annotations

import json
import math
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path

import rasterio
from rasterio.warp import transform_bounds

from floodvalue.paths import SPLITS_DIR, data_dir, output_dir

NEARBY_KM = 25.0
# An observation up to this many days after the reported start can still show the water.
AFTER_WINDOW_DAYS = 12.0
BEFORE_SLACK_DAYS = 1.0


def _km_outside(lat: float, lon: float, bounds: tuple[float, float, float, float]) -> float:
    west, south, east, north = bounds
    dlat = max(south - lat, 0.0, lat - north)
    dlon = max(west - lon, 0.0, lon - east)
    return math.hypot(dlat * 110.574, dlon * 111.32 * math.cos(math.radians(lat)))


def _sealed_chips() -> set[str]:
    path = SPLITS_DIR / "split_v1.json"
    if not path.is_file():
        return set()
    return set(json.loads(path.read_text(encoding="utf-8")).get("test", []))


@lru_cache(maxsize=4)
def _catalog(manifest: Path, mtime: float) -> tuple[dict, ...]:
    sealed = _sealed_chips()
    scenes = []
    for source in json.loads(manifest.read_text(encoding="utf-8")).get("sources", []):
        chip_id = source.get("chip_id")
        path = data_dir() / "raw" / "HandLabeled" / "S1Hand" / Path(str(source.get("file", ""))).name
        if not chip_id or chip_id in sealed or not source.get("observation_date") \
                or not path.is_file():
            continue
        with rasterio.open(path) as src:
            if not src.crs:
                continue
            bounds = transform_bounds(src.crs, "EPSG:4326", *src.bounds, densify_pts=21)
        scenes.append({"chip_id": chip_id, "observation_date": source["observation_date"],
                       "bounds": tuple(bounds)})
    return tuple(scenes)


def local_catalog() -> tuple[dict, ...]:
    manifest = data_dir() / "interim" / "source_manifest_s1.json"
    if not manifest.is_file():
        return ()
    return _catalog(manifest, manifest.stat().st_mtime)


def analyzed_runs() -> dict[str, str]:
    """chip_id → run_id of existing result bundles (read-only reference)."""
    runs = {}
    root = output_dir()
    if root.is_dir():
        for metadata in root.glob("*/run_metadata.json"):
            try:
                data = json.loads(metadata.read_text(encoding="utf-8"))
            except ValueError:
                continue
            if data.get("chip_id"):
                runs[str(data["chip_id"])] = str(data.get("run_id", metadata.parent.name))
    return runs


def assess(lat: float, lon: float, start: datetime | None, end: datetime | None,
           scene: dict, runs: dict[str, str]) -> dict | None:
    """One candidate with suitability and a plain reason; None when the scene is far away."""
    outside = _km_outside(lat, lon, scene["bounds"])
    if outside > NEARBY_KM:
        return None
    relation = "covers" if outside == 0 else "nearby"
    observed = datetime.combine(date.fromisoformat(scene["observation_date"]),
                                datetime.min.time(), tzinfo=start.tzinfo if start else None)
    offset = None
    if start is None:
        suitability, reason = "unusable", "Время события неизвестно: сопоставить по времени нельзя."
    else:
        offset = (observed - start).total_seconds() / 86400
        last = ((end or start) - start).total_seconds() / 86400 + AFTER_WINDOW_DAYS
        if offset < -BEFORE_SLACK_DAYS:
            suitability = "too_early"
            reason = (f"Снимок на {abs(offset):.0f} сут. раньше события: показывает состояние "
                      "до него и не подтверждает и не опровергает сообщение.")
        elif offset > last:
            suitability = "too_late"
            reason = f"Снимок через {offset:.0f} сут.: вода могла уйти."
        elif relation != "covers":
            suitability = "partial_cover"
            reason = f"Снимок не покрывает место, ближайший край в {outside:.1f} км."
        else:
            suitability = "suitable"
            reason = ("Снимок покрывает место в окне события. Это ещё не проверка: нужен анализ "
                      "и сравнение аналитиком.")
    run_id = runs.get(scene["chip_id"])
    return {"catalog": "local", "observation_ref": scene["chip_id"], "product_type": "S1 GRD VV/VH dB (чип Sen1Floods11)",
            "observation_at": scene["observation_date"] + "T00:00:00Z",
            "footprint": list(scene["bounds"]), "spatial_relation": relation,
            "distance_km": round(outside, 1), "time_offset_days":
                None if offset is None else round(offset, 1),
            "suitability": suitability, "reason": reason,
            # Local chips are already in the method's format: S1 VV/VH in dB on the chip grid.
            "stage": "analyzed" if run_id else "prepared", "analysis_run_id": run_id}


def search_local(lat: float, lon: float, start: datetime | None,
                 end: datetime | None) -> list[dict]:
    runs = analyzed_runs()
    found = [item for scene in local_catalog()
             if (item := assess(lat, lon, start, end, scene, runs)) is not None]
    order = {"suitable": 0, "partial_cover": 1, "too_late": 2, "too_early": 3, "unusable": 4}
    return sorted(found, key=lambda item: (order[item["suitability"]],
                                           abs(item["time_offset_days"] or 0)))
