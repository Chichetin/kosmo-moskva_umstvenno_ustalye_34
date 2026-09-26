#!/usr/bin/env python3
# Operator API display check (interface from PR #120): python scripts/check_display.py <api_base_url> <kit_dir>
"""Display chain check for the operator API (PR #120): file values vs what the API serves.
Usage: python api_check.py <base_url> <kit_dir>"""
import io, json, sys, csv, urllib.request
import numpy as np, rasterio
from PIL import Image
from rasterio.transform import rowcol

base, kit = sys.argv[1].rstrip("/"), sys.argv[2]
fails = []
def check(ok, msg):
    print(("OK   " if ok else "FAIL ") + msg); ok or fails.append(msg)
def get(path):
    with urllib.request.urlopen(base + path, timeout=120) as r:
        return r.read()

sess = json.loads(get("/api/session"))
with rasterio.open(f"{kit}/flood_probability.tif") as r:
    p = r.read(1); b = r.bounds; t = r.transform; nod = r.nodata
with rasterio.open(f"{kit}/flood_mask.tif") as r:
    m = r.read(1)
meta = json.load(open(f"{kit}/run_metadata.json"))

rb = sess["raster"]["bounds"]
check(np.allclose(rb, [b.left, b.bottom, b.right, b.top], atol=1e-9), f"границы в API = границы tif {[round(v, 6) for v in rb]}")
c = sess["raster"]["corners"]
check(c == [[rb[0], rb[3]], [rb[2], rb[3]], [rb[2], rb[1]], [rb[0], rb[1]]], "углы наложения: СЗ, СВ, ЮВ, ЮЗ (порядок MapLibre)")
check(float(sess["metadata"]["threshold"]) == float(meta["threshold"]), f"порог в API = порог комплекта ({meta['threshold']})")

loss = {row["asset_id"]: row for row in csv.DictReader(open(f"{kit}/asset_loss.csv"))}
feats = sess["assets"]["features"]
check(len(feats) == 10, f"объектов в API: {len(feats)}")
for f in feats:
    lon, lat = f["geometry"]["coordinates"]; aid = f["properties"]["asset_id"]
    row, col = rowcol(t, lon, lat)
    ok = (str(f["properties"].get("p_flood")) == loss[aid]["p_flood"]
          and str(f["properties"].get("expected_loss_rub")) == loss[aid]["expected_loss_rub"]
          and abs(float(loss[aid]["p_flood"]) - float(p[row, col])) < 1e-6)
    check(ok, f"{aid}: [lon, lat] → пиксель ({row},{col}); p/EL в API = asset_loss.csv = tif")

def png(layer):
    return np.array(Image.open(io.BytesIO(get(f"/api/layers/{layer}.png"))).convert("RGBA"))
prob = png("probability")
check(prob.shape[:2] == p.shape, f"размер PNG вероятности = размер tif {prob.shape[:2]}")
valid = p != nod
check(bool(((prob[..., 3] > 0) == valid).all()), "прозрачность PNG ровно на NoData")
# Colour is linear in p (G = 31 + 191 p); invert and compare pixel by pixel, same orientation.
p_back = (prob[..., 1].astype(float) - 31) / 191
err = np.abs(p_back[valid] - np.clip(p[valid], 0, 1))
check(float(err.max()) <= 1.0 / 191 + 1e-9, f"PNG вероятности = tif попиксельно (max |Δp| {err.max():.4f} ≤ шаг цвета 1/191), без поворота/отражения")
flip = np.abs((prob[::-1, ..., 1].astype(float) - 31) / 191 - np.clip(p, 0, 1))[valid].mean()
check(flip > err.mean() * 5 or flip > 0.01, f"контроль ориентации: отражённый PNG хуже (ср. |Δp| {flip:.3f} против {err.mean():.4f})")
mk = png("mask")
check(bool(((mk[..., 3] > 0) == (m == 1)).all()), "PNG маски непрозрачен ровно там, где mask = 1")
thr = float(meta["threshold"])
check(bool(((m == 1) == (valid & (p >= thr))).all()), f"маска в комплекте ⇔ p ≥ {thr}")

before = get("/api/layers/probability.png")
for budget in (1000, 5000, 50000):
    json.loads(get(f"/api/plan?budget={budget}"))
after = get("/api/layers/probability.png")
sess2 = json.loads(get("/api/session"))
same_assets = [f["properties"]["p_flood"] for f in sess2["assets"]["features"]] == [f["properties"]["p_flood"] for f in feats]
check(before == after and same_assets and sess2["metadata"]["threshold"] == sess["metadata"]["threshold"],
      "пересчёт плана (бюджет 1 000 / 5 000 / 50 000) не меняет слой p, порог и p объектов")
print(f"\nитог: {'все проверки прошли' if not fails else f'{len(fails)} провал(ов)'}")
sys.exit(1 if fails else 0)
