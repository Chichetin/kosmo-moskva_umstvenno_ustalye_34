#!/usr/bin/env python3
# Usage: python scripts/check_chain.py <kit_dir> <chip_id> <data_dir>   (from repo root, CPU is enough)
"""Independent chain check: model in memory -> flood_probability.tif -> assets/asset_loss.csv.

Only the model call (predict_proba + calibrator) uses project code; grids, pixel lookup of the
assets, mask rule and NoData are recomputed here with rasterio/numpy directly."""
import csv, json, sys
from pathlib import Path
import numpy as np, rasterio
from rasterio.transform import rowcol

kit = Path(sys.argv[1]); chip = sys.argv[2]; data = Path(sys.argv[3])
fails = []
def check(ok, msg):
    print(("OK   " if ok else "FAIL ") + msg); ok or fails.append(msg)

s1p = data / "raw/HandLabeled/S1Hand" / f"{chip}_S1Hand.tif"
with rasterio.open(s1p) as r: s1 = r.read().astype(np.float32); s1t, s1crs, s1shape = r.transform, r.crs, r.shape
with rasterio.open(kit / "flood_probability.tif") as r: p_file = r.read(1); pt, pcrs, pnod = r.transform, r.crs, r.nodata
with rasterio.open(kit / "flood_mask.tif") as r: m_file = r.read(1); mnod = r.nodata
meta = json.load(open(kit / "run_metadata.json")); thr = float(meta["threshold"])

check(pcrs == s1crs and p_file.shape == s1shape and pt.almost_equals(s1t, 1e-12), f"сетка p = сетка S1 ({pcrs}, {p_file.shape})")
valid = np.isfinite(s1).all(0)
check(bool((p_file[~valid] == pnod).all()) and pnod == -9999, f"NoData p = −9999 ровно там, где S1 не конечен ({int((~valid).sum())} пикс.)")
pv = p_file[valid]
check(bool(np.isfinite(pv).all() and (pv >= 0).all() and (pv <= 1).all()), f"p конечна и в [0,1] на {pv.size} валидных пикселях")
check(bool((m_file[~valid] == mnod).all()) and mnod == 255 and bool(((pv >= thr).astype(np.uint8) == m_file[valid]).all()), f"маска ⇔ p ≥ {thr}, NoData маски = 255")

# model in memory (same dispatcher/calibrator as predict) vs file
import importlib
sys.path.insert(0, str(Path.cwd() / "src"))
from floodvalue.model.predict import METHODS
from floodvalue.calibration import calibrator
from floodvalue.paths import MODELS_DIR
mod = importlib.import_module(METHODS["unet"])
p_raw = mod.predict_proba(s1)
cfg = json.load(open(MODELS_DIR / "unet" / "config.json"))
cal = calibrator.load(MODELS_DIR / "unet" / "calibration.json", cfg["weights_sha256"])
p_mem = calibrator.apply(cal["calibrator"], p_raw)
d = np.abs(p_mem[valid].astype(np.float64) - pv.astype(np.float64))
check(float(d.max()) < 1e-5, f"модель в памяти = файл: max |Δp| = {d.max():.2e}")

# assets: lon/lat -> pixel -> p in file == p_flood in asset_loss.csv
gj = json.load(open(kit / "assets.geojson")); loss = {r["asset_id"]: r for r in csv.DictReader(open(kit / "asset_loss.csv"))}
with rasterio.open(s1p) as r: b = r.bounds
for f in gj["features"]:
    lon, lat = f["geometry"]["coordinates"]; aid = f["properties"]["asset_id"]
    inside = b.left <= lon <= b.right and b.bottom <= lat <= b.top
    row, col = rowcol(s1t, lon, lat)
    pf = float(p_file[row, col]); pl = loss[aid]["p_flood"]
    ok = inside and pl != "" and abs(float(pl) - pf) < 1e-6
    check(ok, f"{aid}: [lon {lon:.5f}, lat {lat:.5f}] в чипе={inside} → пиксель ({row},{col}) p_файл={pf:.6f} p_таблица={pl} p_память={float(p_mem[row, col]):.6f}")
print(f"\nитог: {'все проверки прошли' if not fails else f'{len(fails)} провал(ов)'}")
sys.exit(1 if fails else 0)
