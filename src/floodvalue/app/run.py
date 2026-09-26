"""Read-only operator API and the bundled React interface."""

from __future__ import annotations

import csv
import io
import json
import math
import os
import zipfile
from functools import lru_cache
from datetime import date
from decimal import Decimal
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, reproject, transform, transform_bounds
from fastapi import FastAPI, HTTPException, Query
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image

from floodvalue.contracts.files import REQUIRED_FILES
from floodvalue.paths import REPO_ROOT, data_dir, output_dir
from floodvalue.procurement.pricing import load_pricing, price_basket
from floodvalue.procurement.run import plan

STATIC_DIR = Path(__file__).parent / "static"
REQUIRED = REQUIRED_FILES
REBUILD = "соберите комплект: make all CHIP=<chip_id> BUDGET=<руб>"
app = FastAPI(title="FloodValue operator")


def _missing(path: Path) -> list[str]:
    return [name for name in REQUIRED if not (path / name).is_file()]


def _incomplete_runs() -> list[str]:
    root = output_dir()
    if not root.exists():
        return []
    return [f"{path.name} (нет {', '.join(_missing(path))})" for path in sorted(root.iterdir())
            if path.is_dir() and (path / "run_metadata.json").is_file() and _missing(path)]


def _available_runs() -> list[Path]:
    explicit = os.environ.get("FLOODVALUE_APP_RUN_DIR")
    if explicit:
        path = Path(explicit).resolve()
        missing = _missing(path) if path.is_dir() else ["каталога"]
        if missing:
            raise HTTPException(503, f"Неполный комплект {path}: нет {', '.join(missing)}; {REBUILD}")
        return [path]
    root = output_dir()
    candidates = [path for path in root.iterdir() if path.is_dir()
                  and all((path / name).is_file() for name in REQUIRED)] if root.exists() else []
    if candidates:
        # Repeated calculations of one observation are one scene; distinct dates remain visible.
        by_observation: dict[tuple[str, str | None], Path] = {}
        for path in sorted(candidates, key=lambda item: (item / "run_metadata.json").stat().st_mtime):
            metadata = _json(path / "run_metadata.json")
            chip_id = str(metadata.get("chip_id", path.name))
            by_observation[(chip_id, _observation_date(path, metadata))] = path
        return sorted(by_observation.values(), key=lambda item: (item / "run_metadata.json").stat().st_mtime, reverse=True)
    manifest = data_dir() / "interim" / "source_manifest_s1.json"
    if manifest.is_file() and any((data_dir() / "raw" / "HandLabeled" / "S1Hand" /
                                   Path(str(source.get("file", ""))).name).is_file()
                                  for source in _json(manifest).get("sources", [])):
        return []
    return []


def _run_dir(run_id: str | None = None) -> Path:
    runs = _available_runs()
    if run_id is None:
        if not runs:
            skipped = _incomplete_runs()
            raise HTTPException(503, f"Готового комплекта нет в {output_dir()}: {REBUILD}."
                                + (f" Неполные пропущены: {'; '.join(skipped)}." if skipped else ""))
        return runs[0]
    for path in runs:
        metadata = _json(path / "run_metadata.json")
        if metadata.get("run_id", path.name) == run_id:
            return path
    raise HTTPException(404, "Сцена не найдена")


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def _raster_info(path: Path) -> dict:
    with rasterio.open(path) as src:
        if not src.crs:
            return {"bounds": None, "corners": None, "width": src.width,
                    "height": src.height, "crs": None}
        bounds = transform_bounds(src.crs, "EPSG:4326", *src.bounds, densify_pts=21)
        if not all(math.isfinite(value) for value in bounds) or not (-180 <= bounds[0] < bounds[2] <= 180 and -90 <= bounds[1] < bounds[3] <= 90):
            return {"bounds": None, "corners": None, "width": src.width,
                    "height": src.height, "crs": src.crs.to_string()}
        west, south, east, north = bounds
        # Image sources use a north-up WGS84 grid; rotated/projected rasters are warped below.
        corners = [[west, north], [east, north], [east, south], [west, south]]
        return {"bounds": list(bounds), "corners": corners,
                "width": src.width, "height": src.height, "crs": src.crs.to_string()}


def _scene_position(path: Path) -> dict | None:
    try:
        with rasterio.open(path) as src:
            if not src.crs:
                return None
            center_x, center_y = src.transform @ (src.width / 2, src.height / 2)
            longitudes, latitudes = transform(src.crs, "EPSG:4326", [center_x], [center_y])
    except (rasterio.errors.RasterioIOError, ValueError):
        return None
    lon, lat = longitudes[0], latitudes[0]
    if not all(math.isfinite(value) for value in (lat, lon)):
        return None
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    return {"lat": lat, "lon": lon}


def _observation_date(run_dir: Path, metadata: dict) -> str | None:
    manifest_path = run_dir / "source_manifest.json"
    if manifest_path.is_file():
        for source in _json(manifest_path).get("sources", []):
            if source.get("chip_id") == metadata.get("chip_id") and "S1" in str(source.get("source_id", "")):
                return source.get("observation_date")
    return None


@app.get("/api/scenes")
def scenes():
    result = []
    by_chip = {}
    for run_dir in _available_runs():
        metadata = _json(run_dir / "run_metadata.json")
        chip_id = str(metadata.get("chip_id", run_dir.name))
        info = _raster_info(run_dir / "flood_probability.tif")
        item = {
            "run_id": metadata.get("run_id", run_dir.name),
            "fixture": run_dir == REPO_ROOT / "tests" / "fixtures" / "good",
            "chip_id": chip_id,
            "territory": metadata.get("event_id") or chip_id,
            "observation_date": _observation_date(run_dir, metadata),
            "position": _scene_position(run_dir / "flood_probability.tif"),
            "bounds": info["bounds"], "corners": info["corners"],
            "status": "analyzed", "preview_url": None,
        }
        by_chip[chip_id] = item
        result.append(item)
    manifest = data_dir() / "interim" / "source_manifest_s1.json"
    if manifest.is_file():
        for source in _json(manifest).get("sources", []):
            chip_id = source.get("chip_id")
            filename = Path(str(source.get("file", ""))).name
            path = data_dir() / "raw" / "HandLabeled" / "S1Hand" / filename
            if not chip_id or not path.is_file():
                continue
            if chip_id in by_chip:
                by_chip[chip_id]["observation_date"] = source.get("observation_date")
                by_chip[chip_id]["preview_url"] = f"/api/observations/{chip_id}/preview.png"
                continue
            info = _raster_info(path)
            result.append({"run_id": None, "chip_id": chip_id,
                           "fixture": False,
                           "territory": source.get("event_id") or chip_id,
                           "observation_date": source.get("observation_date"),
                           "position": _scene_position(path), "bounds": info["bounds"],
                           "corners": info["corners"], "status": "observation_only",
                           "preview_url": f"/api/observations/{chip_id}/preview.png"})
    return sorted(result, key=lambda item: (item["status"] != "analyzed", item["territory"], item["chip_id"]))


def _observation_file(chip_id: str) -> Path:
    manifest = data_dir() / "interim" / "source_manifest_s1.json"
    if manifest.is_file():
        for source in _json(manifest).get("sources", []):
            if source.get("chip_id") == chip_id:
                path = data_dir() / "raw" / "HandLabeled" / "S1Hand" / Path(str(source.get("file", ""))).name
                if path.is_file():
                    return path
    raise HTTPException(404, "Исходный снимок недоступен")


@app.get("/api/session")
def session(run_id: str | None = None):
    run_dir = _run_dir(run_id)
    metadata = _json(run_dir / "run_metadata.json")
    assets = _json(run_dir / "assets.geojson")
    losses = {row["asset_id"]: row for row in _csv(run_dir / "asset_loss.csv")}
    for feature in assets["features"]:
        feature["properties"].update(losses.get(feature["properties"]["asset_id"], {}))
    candidates = _json(run_dir / "candidate_orders.geojson")
    pricing = load_pricing()
    as_of = date.fromisoformat(metadata["budget"]["decision_at"][:10])
    for feature in candidates["features"]:
        props = feature["properties"]
        item = {**props, "area_km2": Decimal(str(props["area_km2"]))}
        props["solo_cost_rub"] = float(price_basket([item], pricing, as_of=as_of)[0]["cost_rub"])
    s1_name = metadata.get("source", {}).get("s1_files", [""])[0]
    s1_path = data_dir() / "raw" / "HandLabeled" / "S1Hand" / Path(s1_name).name
    layer_rasters = {
        "s1": _raster_info(s1_path) if s1_path.is_file() else None,
        "probability": _raster_info(run_dir / "flood_probability.tif"),
        "mask": _raster_info(run_dir / "flood_mask.tif"),
        "uncertainty": (_raster_info(run_dir / "uncertainty.tif")
                        if (run_dir / "uncertainty.tif").is_file() else None),
    }
    return {
        "run_id": metadata.get("run_id", run_dir.name),
        "fixture": run_dir == REPO_ROOT / "tests" / "fixtures" / "good",
        "metadata": metadata,
        "observation_date": _observation_date(run_dir, metadata),
        "raster": _raster_info(run_dir / "flood_probability.tif"),
        "layer_rasters": layer_rasters,
        "layers": {"s1": s1_path.is_file(), "probability": True, "mask": True,
                   "uncertainty": (run_dir / "uncertainty.tif").is_file()},
        "assets": assets,
        "candidates": candidates,
        "sensitivity": _csv(run_dir / "sensitivity.csv"),
    }


@app.get("/api/plan")
def recalculated_plan(budget: float = Query(ge=0, le=1_000_000_000), run_id: str | None = None):
    try:
        return JSONResponse(jsonable_encoder(plan(_run_dir(run_id), budget)))
    except (FileNotFoundError, ValueError, KeyError) as exc:
        raise HTTPException(422, str(exc)) from exc


def _rgba(layer: str, values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    height, width = values.shape
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    if layer == "s1":
        scaled = np.clip((values + 31) / 27, 0, 1)
        grey = (scaled * 166 + 25).astype(np.uint8)
        rgba[..., :3] = np.stack((grey * 0.86, grey * 1.02, grey * 1.10), axis=-1).astype(np.uint8)
        rgba[..., 3] = np.where(valid, 255, 0)
    elif layer == "probability":
        p = np.clip(values, 0, 1)
        rgba[..., 0] = (11 + p * 89).astype(np.uint8)
        rgba[..., 1] = (31 + p * 191).astype(np.uint8)
        rgba[..., 2] = (45 + p * 173).astype(np.uint8)
        rgba[..., 3] = np.where(valid, 255, 0)
    elif layer == "mask":
        rgba[..., :3] = (86, 228, 225)
        rgba[..., 3] = np.where(valid & (values == 1), 157, 0)
    else:
        p = np.clip(values, 0, 1)
        rgba[..., 0] = (81 + 173 * p).astype(np.uint8)
        rgba[..., 1] = (70 + 87 * (1 - p)).astype(np.uint8)
        rgba[..., 2] = (126 + 71 * p).astype(np.uint8)
        rgba[..., 3] = np.where(valid, 225, 0)
    return rgba


def _visual_values(path: Path, band: int, *, mask: bool = False,
                   preview: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Read a visual grid in WGS84. This never changes model or procurement rasters."""
    with rasterio.open(path) as src:
        width, height = src.width, src.height
        if preview:
            ratio = min(1, 168 / max(width, height))
            width, height = max(1, round(width * ratio)), max(1, round(height * ratio))
        if not src.crs:
            raise HTTPException(422, "У снимка нет геопривязки")
        resampling = Resampling.nearest if mask else Resampling.bilinear
        north_up = (src.crs.to_epsg() == 4326 and src.transform.b == 0 and src.transform.d == 0
                    and src.transform.a > 0 and src.transform.e < 0)
        if north_up:
            values = src.read(band, out_shape=(height, width), resampling=resampling).astype(np.float32)
            valid = src.read_masks(band, out_shape=(height, width), resampling=Resampling.nearest) > 0
        else:
            west, south, east, north = transform_bounds(src.crs, "EPSG:4326", *src.bounds, densify_pts=21)
            destination = from_bounds(west, south, east, north, width, height)
            values = np.full((height, width), np.nan, dtype=np.float32)
            reproject(source=rasterio.band(src, band), destination=values,
                      src_transform=src.transform, src_crs=src.crs,
                      src_nodata=src.nodata, dst_transform=destination, dst_crs="EPSG:4326",
                      dst_nodata=np.nan, resampling=resampling)
            source_mask = src.read_masks(band)
            target_mask = np.zeros((height, width), dtype=np.uint8)
            reproject(source=source_mask, destination=target_mask,
                      src_transform=src.transform, src_crs=src.crs,
                      dst_transform=destination, dst_crs="EPSG:4326",
                      resampling=Resampling.nearest)
            valid = target_mask > 0
        valid &= np.isfinite(values)
        if src.nodata is not None and math.isfinite(src.nodata):
            valid &= values != src.nodata
        return np.where(valid, values, 0), valid


def _png(layer: str, path: Path, band: int, *, preview: bool = False) -> bytes:
    values, valid = _visual_values(path, band, mask=layer == "mask", preview=preview)
    buffer = io.BytesIO()
    Image.fromarray(_rgba(layer, values, valid), "RGBA").save(buffer, format="PNG")
    return buffer.getvalue()


@lru_cache(maxsize=96)
def _preview_png(path: Path) -> bytes:
    return _png("s1", path, 2, preview=True)


@app.get("/api/observations/{chip_id}/preview.png")
def observation_preview(chip_id: str):
    return Response(_preview_png(_observation_file(chip_id)), media_type="image/png",
                    headers={"Cache-Control": "public, max-age=3600"})


@app.get("/api/observations/{chip_id}/s1.png")
def observation_s1(chip_id: str):
    return Response(_png("s1", _observation_file(chip_id), 2), media_type="image/png",
                    headers={"Cache-Control": "public, max-age=3600"})


@app.get("/api/layers/{layer}.png")
def raster_layer(layer: str, run_id: str | None = None):
    run_dir = _run_dir(run_id)
    if layer not in {"s1", "probability", "mask", "uncertainty"}:
        raise HTTPException(404, "Неизвестный слой")
    if layer == "s1":
        metadata = _json(run_dir / "run_metadata.json")
        names = metadata.get("source", {}).get("s1_files", [])
        path = (data_dir() / "raw" / "HandLabeled" / "S1Hand" / Path(names[0]).name) if names else None
        if path is None or not path.is_file():
            raise HTTPException(404, "Исходный S1 недоступен локально")
        band = 2
    else:
        filename = {"probability": "flood_probability.tif", "mask": "flood_mask.tif",
                    "uncertainty": "uncertainty.tif"}[layer]
        path = run_dir / filename
        if not path.is_file():
            raise HTTPException(404, "Слой отсутствует в комплекте")
        band = 1
    return Response(_png(layer, path, band), media_type="image/png")


@app.get("/api/download")
def download(run_id: str | None = None):
    run_dir = _run_dir(run_id)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(run_dir.iterdir()):
            if path.is_file() and path.suffix in {".csv", ".json", ".geojson", ".tif"}:
                archive.write(path, path.name)
    return Response(buffer.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="{run_dir.name}.zip"'})


if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="web")


def run(args) -> int:
    import uvicorn

    if args.out:
        out = Path(args.out).resolve()
        missing = _missing(out) if out.is_dir() else ["каталога"]
        if missing:
            print(f"app: {out} — неполный комплект, нет {', '.join(missing)}; {REBUILD}")
            return 1
        os.environ["FLOODVALUE_APP_RUN_DIR"] = str(out)
    if not STATIC_DIR.exists():
        print("app: собранный React-интерфейс не найден; выполните make web")
        return 2
    port = int(os.environ.get("FLOODVALUE_APP_PORT", "8000"))
    print(f"FloodValue: http://127.0.0.1:{port}")
    uvicorn.run(app, host="127.0.0.1", port=port)
    return 0
