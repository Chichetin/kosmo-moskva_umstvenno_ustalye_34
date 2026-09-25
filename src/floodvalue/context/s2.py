"""Offline context layer: Sentinel-2 true colour of a Sen1Floods11 chip (D-042).

Pure functions (`true_color`, `rough_cloud_fraction`, `context_metadata`) plus `build`, which
reads the S2Hand chip (downloading that single file from the public bucket if absent), checks
it lies on the S1 grid and writes `context_s2_rgb.png` + `context_s2.json`.

Never an input to the model or the economics: see floodvalue/context/__init__.py.
"""

from __future__ import annotations

import io
import json
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image

from floodvalue.data.io import layer_path, parse_chip_id, raw_dir

S2_LAYER = "S2Hand"
BUCKET_MEDIA = "https://storage.googleapis.com/sen1floods11/"
BUCKET_PREFIX = "v1.1/data/flood_events/"
METADATA = "Sen1Floods11_Metadata.geojson"
# Sen1Floods11 names the Mekong chips after the event, the metadata after the country.
EVENT_ALIASES = {"Mekong": "Cambodia"}

# S2 L1C band order in S2Hand (0-based): B1 B2 B3 B4 B5 B6 B7 B8 B8A B9 B10 B11 B12.
B2, B3, B4, B10 = 1, 2, 3, 10
REFLECTANCE_SCALE = 10000.0

ROLE = "context_only"
DISCLAIMER = ("Контекст для наглядности, не вход модели и не наблюдение воды на момент "
              "радарного снимка: дата оптического снимка отличается от даты Sentinel-1.")

# Live basemaps for the interface. Tiles are loaded by the viewer, not bulk-downloaded:
# the OSM and Esri usage policies forbid pre-seeding/bulk caching.
ONLINE_BASEMAPS = [
    {"id": "esri_world_imagery", "name": "Esri World Imagery",
     "url": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
     "attribution": "Tiles © Esri — Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community",
     "license": "Esri Master License Agreement / Terms of Use: display with attribution, no bulk download",
     "imagery_date": "разная по месту, обычно годы до/после события — не наблюдение паводка",
     "offline": False},
    {"id": "eox_s2cloudless_2016", "name": "Sentinel-2 cloudless 2016 (EOX)",
     "url": "https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless_3857/default/g/{z}/{y}/{x}.jpg",
     "attribution": "Sentinel-2 cloudless — https://s2maps.eu by EOX IT Services GmbH "
                    "(contains modified Copernicus Sentinel data 2016)",
     "license": "CC BY 4.0 (только выпуск 2016; поздние выпуски — CC BY-NC-SA 4.0)",
     "imagery_date": "мозаика 2016 года", "offline": False},
    {"id": "osm", "name": "OpenStreetMap",
     "url": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
     "attribution": "© OpenStreetMap contributors",
     "license": "ODbL; tile usage policy: без массовой загрузки, с атрибуцией",
     "imagery_date": "векторная карта, текущее состояние", "offline": False},
]


def true_color(s2: np.ndarray, low_pct: float = 2.0, high_pct: float = 98.0,
               gamma: float = 1.0 / 1.4) -> np.ndarray:
    """(13, H, W) S2 L1C digital numbers -> uint8 RGBA (H, W, 4) in natural colour (B4, B3, B2).

    One common percentile stretch over the three bands keeps the colour balance; alpha is 0
    where all three bands are 0 or not finite (no data)."""
    rgb = np.stack([s2[B4], s2[B3], s2[B2]]).astype("float32") / REFLECTANCE_SCALE
    valid = np.isfinite(rgb).all(axis=0) & (rgb != 0).any(axis=0)
    out = np.zeros(rgb.shape[1:] + (4,), dtype=np.uint8)
    if not valid.any():
        return out
    lo, hi = np.percentile(rgb[:, valid], [low_pct, high_pct])
    scaled = np.clip((np.nan_to_num(rgb) - lo) / max(hi - lo, 1e-6), 0.0, 1.0) ** gamma
    out[..., :3] = np.moveaxis((scaled * 255).round().astype(np.uint8), 0, -1)
    out[..., 3] = np.where(valid, 255, 0)
    return out


def rough_cloud_fraction(s2: np.ndarray) -> float | None:
    """Share of valid pixels that look like cloud: bright blue (B2 > 0.3) or cirrus (B10 > 0.01).

    A coarse screen for the caption, not a cloud mask (S2Hand has no QA band)."""
    b2 = s2[B2].astype("float32") / REFLECTANCE_SCALE
    b10 = s2[B10].astype("float32") / REFLECTANCE_SCALE
    valid = np.isfinite(b2) & (b2 != 0)
    if not valid.any():
        return None
    cloud = (b2 > 0.3) | (b10 > 0.01)
    return float(cloud[valid].mean())


def event_dates(event_id: str, metadata: dict) -> dict[str, str | None]:
    """{"s1_date", "s2_date"} of the event as ISO dates from Sen1Floods11_Metadata.geojson."""
    location = EVENT_ALIASES.get(event_id, event_id)
    for f in metadata["features"]:
        p = f["properties"]
        if p.get("location") == location:
            iso = {k: date(*map(int, p[k].split("/"))).isoformat() if p.get(k) else None
                   for k in ("s1_date", "s2_date")}
            return iso
    return {"s1_date": None, "s2_date": None}


def context_metadata(chip_id: str, bounds: tuple[float, float, float, float], crs: str,
                     dates: dict[str, str | None], cloud: float | None) -> dict:
    """Caption and provenance of the offline layer plus the online basemap list."""
    s1, s2 = dates.get("s1_date"), dates.get("s2_date")
    delta = (date.fromisoformat(s2) - date.fromisoformat(s1)).days if s1 and s2 else None
    when = ("дата неизвестна" if delta is None else
            "в день радарного снимка" if delta == 0 else
            f"за {-delta} дн. до радарного снимка" if delta < 0 else
            f"через {delta} дн. после радарного снимка")
    return {
        "chip_id": chip_id,
        "role": ROLE,
        "used_by_model": False,
        "used_by_economics": False,
        "disclaimer": DISCLAIMER,
        "offline": {
            "file": "context_s2_rgb.png",
            "name": "Sentinel-2 L1C, естественные цвета (B4, B3, B2)",
            "source": "Sen1Floods11 v1.1, слой S2Hand",
            "url": BUCKET_MEDIA + BUCKET_PREFIX + f"HandLabeled/{S2_LAYER}/{chip_id}_{S2_LAYER}.tif",
            "acquired": s2,
            "s1_date": s1,
            "days_from_s1": delta,
            "caption": f"Sentinel-2, {s2 or '?'} ({when}) — контекст, не наблюдение воды",
            "cloud_fraction_rough": None if cloud is None else round(cloud, 4),
            "cloud_note": "грубая оценка по B2 и B10, в S2Hand нет маски облаков",
            "license": "не указана авторами (STAC: proprietary); цитирование Bonafilia et al. 2020",
            # [west, south, east, north] in lon/lat (OGC CRS84 order), for an image overlay.
            "bounds_lonlat": [round(v, 9) for v in bounds],
            "crs": crs,
        },
        "online": ONLINE_BASEMAPS,
    }


def _download(chip_id: str, dest: Path) -> None:
    url = BUCKET_MEDIA + urllib.parse.quote(BUCKET_PREFIX + f"HandLabeled/{S2_LAYER}/{chip_id}_{S2_LAYER}.tif")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        f.write(r.read())
    tmp.replace(dest)


def build(chip_id: str, out: Path, *, download: bool = True) -> dict:
    """Write context_s2_rgb.png and context_s2.json for `chip_id` into `out`."""
    event_id, _ = parse_chip_id(chip_id)
    s2_path, s1_path = layer_path(chip_id, S2_LAYER), layer_path(chip_id, "S1Hand")
    if not s2_path.exists():
        if not download:
            raise FileNotFoundError(s2_path)
        _download(chip_id, s2_path)
    with rasterio.open(s2_path) as r2:
        s2 = r2.read()
        transform, shape, crs, b = r2.transform, r2.shape, r2.crs, r2.bounds
    if s1_path.exists():
        with rasterio.open(s1_path) as r1:
            if r1.shape != shape or not r1.transform.almost_equals(transform) or r1.crs != crs:
                raise ValueError(f"{chip_id}: S2Hand grid differs from S1Hand — overlay would be misplaced")
    meta_path = raw_dir() / METADATA
    metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {"features": []}
    meta = context_metadata(chip_id, (b.left, b.bottom, b.right, b.top), str(crs),
                            event_dates(event_id, metadata), rough_cloud_fraction(s2))
    out.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    Image.fromarray(true_color(s2), "RGBA").save(buf, "PNG", optimize=True)
    (out / meta["offline"]["file"]).write_bytes(buf.getvalue())
    (out / "context_s2.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return meta
