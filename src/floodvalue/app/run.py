"""Read-only operator API and the bundled React interface."""

from __future__ import annotations

import csv
import io
import json
import math
import os
import queue
import subprocess
import sys
import threading
import time
import uuid
import zipfile
from functools import lru_cache
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

from floodvalue.data.split import EVENT_PARTS  # exception (D-050): test sealing, as floodvalue.context
from floodvalue.paths import REPO_ROOT, data_dir, output_dir
from floodvalue.procurement import plan  # public contract §10

STATIC_DIR = Path(__file__).parent / "static"
REQUIRED = (
    "run_metadata.json", "flood_probability.tif", "flood_mask.tif", "assets.geojson",
    "asset_loss.csv", "candidate_orders.geojson", "strategy_plans.json",
    "strategy_comparison.csv", "procurement_plan.csv", "sensitivity.csv",
)
app = FastAPI(title="FloodValue operator")
TEST_EVENTS = frozenset(EVENT_PARTS["test"])


def _sealed(chip_id: str) -> bool:
    """Test chips stay sealed until eval-test (D-010, D-033): the raw catalog neither lists nor
    renders them. A finished analysis of the chip (made after eval-test, D-033) or an explicit
    FLOODVALUE_APP_ALLOW_TEST=1 opens it."""
    return (chip_id.rsplit("_", 1)[0] in TEST_EVENTS
            and os.environ.get("FLOODVALUE_APP_ALLOW_TEST") != "1")


def _available_runs() -> list[Path]:
    explicit = os.environ.get("FLOODVALUE_APP_RUN_DIR")
    if explicit:
        path = Path(explicit).resolve()
        if not path.is_dir() or not all((path / name).is_file() for name in REQUIRED):
            raise HTTPException(503, f"Неполный комплект: {path}")
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
    return []


def _run_dir(run_id: str | None = None) -> Path:
    runs = _available_runs()
    if run_id is None:
        if not runs:
            raise HTTPException(503, "Готового анализа нет. Выберите исходное наблюдение в каталоге.")
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
            if _sealed(chip_id):
                continue
            info = _raster_info(path)
            result.append({"run_id": None, "chip_id": chip_id,
                           "fixture": False,
                           "territory": source.get("event_id") or chip_id,
                           "observation_date": source.get("observation_date"),
                           "position": _scene_position(path), "bounds": info["bounds"],
                           "corners": info["corners"], "status": "observation_only",
                           "preview_url": f"/api/observations/{chip_id}/preview.png",
                           "analyzable": not _s1_empty(chip_id),
                           "not_analyzable_reason": S1_EMPTY_REASON if _s1_empty(chip_id) else None})
    return sorted(result, key=lambda item: (item["status"] != "analyzed", item["territory"], item["chip_id"]))


S1_EMPTY_REASON = "В снимке S1 нет валидных пикселей — объекты разместить негде, анализ невозможен"


@lru_cache(maxsize=4)
def _s1_nan_fractions(qc_path: Path, mtime_ns: int) -> dict[str, float]:
    """chip_id → max(vv_nan_frac, vh_nan_frac) from `make data` QC; sealed test rows are empty."""
    result = {}
    with qc_path.open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            try:
                result[row["chip_id"]] = max(float(row["vv_nan_frac"]), float(row["vh_nan_frac"]))
            except (KeyError, TypeError, ValueError):
                continue
    return result


def _s1_empty(chip_id: str) -> bool:
    """True when QC says no pixel has both S1 bands: assets need valid S1 (portfolio placement).
    Without qc.csv nothing is claimed — the pipeline itself reports the failure then."""
    qc_path = data_dir() / "interim" / "qc.csv"
    if not qc_path.is_file():
        return False
    return _s1_nan_fractions(qc_path, qc_path.stat().st_mtime_ns).get(chip_id, 0.0) >= 1.0


def _observation_file(chip_id: str) -> Path:
    if _sealed(chip_id) and chip_id not in {
            str(_json(path / "run_metadata.json").get("chip_id", path.name)) for path in _available_runs()}:
        raise HTTPException(404, "Снимок из test запечатан до eval-test")
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
    for feature in candidates["features"]:
        # Price of the zone ordered alone — already in the catalog (candidate_orders.geojson).
        feature["properties"]["solo_cost_rub"] = feature["properties"].get("single_order_cost_rub")
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
    except (ValueError, KeyError) as exc:
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


# --- Analysis of a raw observation from the catalog ------------------------------------------
# The same stages as `make all` (data is already prepared), run as subprocesses: the app imports
# no stage internals. One job at a time; the queue is short because the interface may be public.
ANALYSIS_STEPS = (
    ("Карта воды основного метода", ["floodvalue", "predict"]),
    ("Маска baseline", ["floodvalue.baseline.predict", "--mask-only"]),
    ("Объекты и ожидаемый ущерб", ["floodvalue", "assets"]),
    ("Зоны и план съёмки A/B/C", ["floodvalue", "procure"]),
    ("Комплект файлов", ["floodvalue", "export"]),
    ("Проверка 12 файлов", ["floodvalue", "validate"]),
)
ANALYSIS_QUEUE_MAX = 3
ANALYSIS_STEP_TIMEOUT_S = 600
ANALYSIS_JOBS_KEPT = 50  # finished jobs remembered for status polling
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()
_job_queue: queue.Queue[str] = queue.Queue()
_worker: threading.Thread | None = None


def _analysis_commands(chip_id: str, out: Path) -> list[list[str]]:
    commands = []
    for _, (module, *extra) in ANALYSIS_STEPS:
        chip = [] if module == "floodvalue" and extra == ["validate"] else ["--chip", chip_id]
        commands.append([sys.executable, "-m", module, *extra, *chip, "--out", str(out)])
    return commands


def _public(job: dict) -> dict:
    return {key: value for key, value in job.items() if key != "out"}


def _server_paths_hidden(text: str) -> str:
    """The interface may be public: stderr keeps the error, not the server's directories."""
    for path, name in ((output_dir(), "<outputs>"), (data_dir(), "<data>"), (REPO_ROOT, "<repo>")):
        text = text.replace(str(path), name)
    return text[-600:]


def _run_job(job_id: str) -> None:
    """Never leaves a job running: any unexpected error marks it failed and frees the slot."""
    try:
        _run_stages(job_id)
    except Exception as exc:  # noqa: BLE001 — a worker must survive and report
        with _jobs_lock:
            _jobs[job_id].update(status="failed", error="Внутренняя ошибка запуска анализа",
                                 detail=_server_paths_hidden(repr(exc)), finished_at=time.time())


def _run_stages(job_id: str) -> None:
    with _jobs_lock:
        job = _jobs[job_id]
        job.update(status="running", started_at=time.time())
    for index, (command, (label, _)) in enumerate(
            zip(_analysis_commands(job["chip_id"], Path(job["out"])), ANALYSIS_STEPS), start=1):
        with _jobs_lock:
            job.update(step=index, step_label=label)
        try:
            done = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True,
                                  timeout=ANALYSIS_STEP_TIMEOUT_S)
            failed = done.returncode != 0
            detail = _server_paths_hidden((done.stderr or done.stdout or "").strip())
        except (OSError, subprocess.TimeoutExpired) as exc:
            failed, detail = True, _server_paths_hidden(str(exc))
        if failed:
            with _jobs_lock:
                job.update(status="failed", error=f"{label}: этап завершился с ошибкой",
                           detail=detail, finished_at=time.time())
            return
    metadata = Path(job["out"]) / "run_metadata.json"
    with _jobs_lock:
        job.update(status="done", finished_at=time.time(),
                   run_id=_json(metadata).get("run_id") if metadata.is_file() else None)


def _worker_loop() -> None:
    while True:
        _run_job(_job_queue.get())


def _ensure_worker() -> None:
    global _worker
    with _jobs_lock:  # two requests must not start two workers: one job at a time
        if _worker is not None and _worker.is_alive():
            return
        _worker = threading.Thread(target=_worker_loop, name="floodvalue-analysis", daemon=True)
        _worker.start()


def _forget_old_jobs() -> None:
    finished = sorted((job for job in _jobs.values() if job["status"] in ("done", "failed")),
                      key=lambda job: job.get("finished_at") or 0)
    for job in finished[:max(0, len(finished) - ANALYSIS_JOBS_KEPT)]:
        _jobs.pop(job["job_id"], None)


@app.post("/api/analyses")
def start_analysis(chip_id: str = Query(min_length=1, max_length=120)):
    """Queue the analysis of one catalog observation; an existing analysis is returned as is."""
    if os.environ.get("FLOODVALUE_APP_RUN_DIR"):
        raise HTTPException(409, "Открыт один комплект (make app OUT=…): запуск анализа недоступен")
    for path in _available_runs():  # a finished analysis (a test chip too, after eval-test)
        metadata = _json(path / "run_metadata.json")
        if str(metadata.get("chip_id")) == chip_id:
            return {"job_id": None, "chip_id": chip_id, "status": "done",
                    "run_id": metadata.get("run_id", path.name)}
    if chip_id.rsplit("_", 1)[0] in TEST_EVENTS:
        # Never computed from the interface, FLOODVALUE_APP_ALLOW_TEST included: a test run is
        # eval-test of the frozen configuration with a line in test-log.md (D-010, D-033).
        raise HTTPException(403, "Снимок из test: анализ только через make eval-test / make all")
    _observation_file(chip_id)  # 404 when the raw observation is not available
    if _s1_empty(chip_id):
        raise HTTPException(422, S1_EMPTY_REASON)
    with _jobs_lock:
        _forget_old_jobs()
        active = [job for job in _jobs.values() if job["status"] in ("queued", "running")]
        for job in active:
            if job["chip_id"] == chip_id:
                return JSONResponse(_public(job), status_code=202)
        if len(active) >= ANALYSIS_QUEUE_MAX:
            raise HTTPException(429, "Очередь анализа занята, повторите через минуту")
        job_id = uuid.uuid4().hex[:12]
        job = {"job_id": job_id, "chip_id": chip_id, "status": "queued", "step": 0,
               "steps": len(ANALYSIS_STEPS), "step_label": None, "run_id": None, "error": None,
               "detail": None, "created_at": time.time(),
               "out": str(output_dir() / f"{chip_id}_ui_{job_id}")}
        _jobs[job_id] = job
    _job_queue.put(job_id)
    _ensure_worker()
    return JSONResponse(_public(job), status_code=202)


@app.get("/api/analyses/{job_id}")
def analysis_status(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "Задача анализа не найдена")
        return _public(job)


if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="web")


def run(args) -> int:
    import uvicorn

    if args.out:
        os.environ["FLOODVALUE_APP_RUN_DIR"] = args.out
    if not STATIC_DIR.exists():
        print("app: собранный React-интерфейс не найден; выполните make web")
        return 2
    port = int(os.environ.get("FLOODVALUE_APP_PORT", "8000"))
    print(f"FloodValue: http://127.0.0.1:{port}")
    uvicorn.run(app, host="127.0.0.1", port=port)
    return 0
