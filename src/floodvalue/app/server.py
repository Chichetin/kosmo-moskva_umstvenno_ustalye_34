"""Operator API (D-013): reads a result kit `outputs/<run_id>/`, serves JSON, PNG layers and
the built React bundle from `static/`. Budget changes call a pure replan function and never
touch rasters, portfolio or thresholds (criterion 19)."""

from __future__ import annotations

import csv
import importlib
import io
import json
import zipfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import rasterio
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from matplotlib import colormaps
from PIL import Image
from pydantic import BaseModel

from floodvalue.paths import data_dir, output_dir

STATIC = Path(__file__).parent / "static"
LAYERS = ("s1_vv", "s1_vh", "probability", "mask", "uncertainty", "nodata")
KIT_FILES = ("flood_probability.tif", "flood_mask.tif", "assets.geojson", "assets.csv", "asset_loss.csv",
             "candidate_orders.geojson", "procurement_plan.csv", "strategy_plans.json",
             "strategy_comparison.csv", "sensitivity.csv", "run_metadata.json", "source_manifest.json")


def _num(v: str):
    if v == "":
        return None
    if v in ("true", "false"):
        return v == "true"
    try:
        f = float(v)
        return int(f) if f.is_integer() and "." not in v else f
    except ValueError:
        return v


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [{k: _num(v) for k, v in r.items()} for r in csv.DictReader(f)]


def run_dir(run_id: str) -> Path:
    d = (output_dir() / run_id).resolve()
    if d.parent != output_dir().resolve() or not (d / "run_metadata.json").exists():
        raise HTTPException(404, f"комплект {run_id} не найден")
    return d


def load_kit(d: Path) -> dict:
    meta = json.loads((d / "run_metadata.json").read_text())
    with rasterio.open(d / "flood_probability.tif") as r:
        b = r.bounds
    zones = json.loads((d / "candidate_orders.geojson").read_text())
    assets = json.loads((d / "assets.geojson").read_text())
    return {
        "meta": meta,
        "bounds": [b.left, b.bottom, b.right, b.top],
        "assets": assets,
        "zones": zones,
        "loss": read_csv(d / "asset_loss.csv"),
        "plan_rows": read_csv(d / "procurement_plan.csv"),
        "plans": json.loads((d / "strategy_plans.json").read_text()),
        "comparison": read_csv(d / "strategy_comparison.csv"),
        "sensitivity": read_csv(d / "sensitivity.csv"),
        "sources": json.loads((d / "source_manifest.json").read_text()) if (d / "source_manifest.json").exists() else [],
        "files": [f for f in KIT_FILES if (d / f).exists()],
        "layers": [l for l in LAYERS if _layer_available(d, meta, l)],
    }


def _s1_path(meta: dict) -> Path | None:
    p = meta.get("s1_file")
    return data_dir() / p if p and (data_dir() / p).exists() else None


def _layer_available(d: Path, meta: dict, name: str) -> bool:
    if name.startswith("s1_"):
        return _s1_path(meta) is not None
    if name == "uncertainty":
        return (d / "uncertainty.tif").exists()
    return True


def _rgba_png(rgba: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, "PNG", optimize=True)
    return buf.getvalue()


@lru_cache(maxsize=64)
def render_layer(d: str, name: str, mtime: float) -> bytes:
    d_ = Path(d)
    meta = json.loads((d_ / "run_metadata.json").read_text())
    if name.startswith("s1_"):
        with rasterio.open(_s1_path(meta)) as r:
            a = r.read(1 if name == "s1_vv" else 2).astype("float32")
        ok = np.isfinite(a)
        lo, hi = np.nanpercentile(a, [2, 98])
        g = np.clip((a - lo) / (hi - lo + 1e-6), 0, 1)
        rgba = np.zeros(a.shape + (4,), "uint8")
        rgba[..., :3] = (np.nan_to_num(g) * 255)[..., None].astype("uint8")
        rgba[..., 3] = np.where(ok, 255, 0)
        return _rgba_png(rgba)
    if name in ("probability", "uncertainty", "nodata"):
        f = "uncertainty.tif" if name == "uncertainty" else "flood_probability.tif"
        with rasterio.open(d_ / f) as r:
            a = r.read(1).astype("float32")
            ok = a != r.nodata
        if name == "nodata":
            rgba = np.zeros(a.shape + (4,), "uint8")
            stripe = ((np.add.outer(np.arange(a.shape[0]), np.arange(a.shape[1])) // 4) % 2).astype(bool)
            rgba[~ok] = [255, 92, 92, 0]
            rgba[..., 3] = np.where(~ok, np.where(stripe, 200, 90), 0)
            return _rgba_png(rgba)
        v = np.clip(np.where(ok, a, 0), 0, 1)
        cmap = colormaps["cividis" if name == "uncertainty" else "Blues"]
        rgba = (cmap(v) * 255).astype("uint8")
        alpha = np.clip(v * 1.4, 0.08, 1.0) if name == "probability" else np.clip(v, 0.05, 1.0)
        rgba[..., 3] = np.where(ok, (alpha * 235).astype("uint8"), 0)
        return _rgba_png(rgba)
    if name == "mask":
        with rasterio.open(d_ / "flood_mask.tif") as r:
            m = r.read(1)
        rgba = np.zeros(m.shape + (4,), "uint8")
        rgba[m == 1] = [56, 189, 248, 210]
        return _rgba_png(rgba)
    raise HTTPException(404, name)


def replan_fn():
    """Real procurement if merged (D-005 exception), else the demo stand-in."""
    try:
        mod = importlib.import_module("floodvalue.procurement")
        fn = getattr(mod, "plan", None)
        if callable(fn):
            return fn, "procurement"
    except ImportError:
        pass
    from floodvalue.app import demo
    return demo.demo_replan, "demo"


class Replan(BaseModel):
    budget_rub: float


def create_app() -> FastAPI:
    app = FastAPI(title="FloodValue — интерфейс оператора")

    @app.get("/api/runs")
    def runs():
        root = output_dir()
        out = []
        for d in sorted(root.glob("*/run_metadata.json")) if root.exists() else []:
            m = json.loads(d.read_text())
            out.append({"run_id": d.parent.name, "chip_id": m.get("chip_id"), "event_id": m.get("event_id"),
                        "s1_date": m.get("s1_date"), "demo": bool(m.get("demo")), "budget_rub": m.get("budget_rub")})
        return out

    @app.get("/api/runs/{run_id}")
    def kit(run_id: str):
        kit = load_kit(run_dir(run_id))
        _, engine = replan_fn()
        kit["replan_engine"] = engine
        return JSONResponse(kit)

    @app.get("/api/runs/{run_id}/layers/{name}.png")
    def layer(run_id: str, name: str):
        if name not in LAYERS:
            raise HTTPException(404, name)
        d = run_dir(run_id)
        png = render_layer(str(d), name, (d / "run_metadata.json").stat().st_mtime)
        return Response(png, media_type="image/png", headers={"Cache-Control": "max-age=300"})

    @app.post("/api/runs/{run_id}/replan")
    def replan(run_id: str, body: Replan):
        if body.budget_rub < 0:
            raise HTTPException(422, "бюджет не может быть отрицательным")
        d = run_dir(run_id)
        fn, engine = replan_fn()
        if engine == "demo":
            k = load_kit(d)
            from shapely.geometry import shape
            zones = [{**f["properties"], "_geom": shape(f["geometry"])} for f in k["zones"]["features"]]
            assets = [{**f["properties"], "longitude": f["geometry"]["coordinates"][0],
                       "latitude": f["geometry"]["coordinates"][1]} for f in k["assets"]["features"]]
            res = fn({"meta": k["meta"], "zones": zones, "assets": assets, "loss": k["loss"]}, body.budget_rub)
        else:
            res = fn(d, body.budget_rub)
        res["engine"] = engine
        return JSONResponse(res)

    @app.get("/api/runs/{run_id}/export.zip")
    def export(run_id: str):
        d = run_dir(run_id)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(d.iterdir()):
                if f.is_file():
                    z.write(f, f"{run_id}/{f.name}")
        return Response(buf.getvalue(), media_type="application/zip",
                        headers={"Content-Disposition": f'attachment; filename="{run_id}.zip"'})

    if STATIC.exists():
        app.mount("/assets", StaticFiles(directory=STATIC / "assets"), name="assets")

        @app.get("/{path:path}")
        def spa(path: str):
            f = STATIC / path
            if path and f.is_file():
                return FileResponse(f)
            return FileResponse(STATIC / "index.html")

    return app
