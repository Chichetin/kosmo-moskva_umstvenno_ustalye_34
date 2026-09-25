#!/usr/bin/env python3
"""Скачивание официального Sen1Floods11 (v1.1) из публичного GCS-бакета gs://sen1floods11
без gsutil — через HTTPS JSON API. Не требует авторизации.

Использование:
    python scripts/download_sen1floods11.py [--dest $FLOODVALUE_DATA_DIR/raw] [--no-s2] [--with-weak]

Раскладка повторяет docs/data.md §3: префиксы бакета v1.1/data/flood_events/ и v1.1/data/
отбрасываются, остальное кладётся как есть:
    raw/HandLabeled/<LAYER>/<EVENT>_<ID>_<LAYER>.tif
    raw/perm_water/{S1Perm,JRCPerm}/...   raw/WeaklyLabeled/...
    raw/splits/...   raw/catalog/...   raw/checkpoints/...   raw/Sen1Floods11_Metadata.geojson
Уже скачанные файлы с совпадающим md5 пропускаются, так что повторный запуск = проверка целостности.

По умолчанию качается всё, что нужно для кейса: HandLabeled (S1Hand, LabelHand, JRCWaterHand,
S1OtsuLabelHand, S2Hand), perm_water, splits, метаданные, catalog, checkpoints.
WeaklyLabeled (~33 ГБ) качается только с флагом --with-weak.
"""
import argparse, hashlib, base64, json, os, sys, urllib.request, urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed

BUCKET = "sen1floods11"
API = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o"
MEDIA = f"https://storage.googleapis.com/{BUCKET}/"

PREFIXES_DEFAULT = [
    "v1.1/Sen1Floods11_Metadata.geojson",
    "v1.1/catalog.zip",
    "v1.1/splits/",
    "v1.1/catalog/",
    "v1.1/checkpoints/",
    "v1.1/data/flood_events/HandLabeled/S1Hand/",
    "v1.1/data/flood_events/HandLabeled/LabelHand/",
    "v1.1/data/flood_events/HandLabeled/JRCWaterHand/",
    "v1.1/data/flood_events/HandLabeled/S1OtsuLabelHand/",
    "v1.1/data/perm_water/",
]
PREFIX_S2 = "v1.1/data/flood_events/HandLabeled/S2Hand/"
PREFIX_WEAK = "v1.1/data/flood_events/WeaklyLabeled/"


def list_objects(prefix):
    items, tok = [], None
    while True:
        u = f"{API}?prefix={urllib.parse.quote(prefix)}&maxResults=1000" + (f"&pageToken={tok}" if tok else "")
        d = json.load(urllib.request.urlopen(u, timeout=60))
        items += d.get("items", [])
        tok = d.get("nextPageToken")
        if not tok:
            return items


def local_path(name):
    rel = name[len("v1.1/"):]
    for pre in ("data/flood_events/", "data/"):
        if rel.startswith(pre):
            return rel[len(pre):]
    return rel


def md5_ok(path, md5_b64):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return base64.b64encode(h.digest()).decode() == md5_b64


def fetch(item, dest):
    name, size, md5 = item["name"], int(item["size"]), item.get("md5Hash")
    out = os.path.join(dest, local_path(name))
    if os.path.exists(out) and os.path.getsize(out) == size and (md5 is None or md5_ok(out, md5)):
        return name, "skip"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    url = MEDIA + urllib.parse.quote(name)
    tmp = out + ".part"
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
                for chunk in iter(lambda: r.read(1 << 20), b""):
                    f.write(chunk)
            if md5 and not md5_ok(tmp, md5):
                raise IOError("md5 mismatch")
            os.replace(tmp, out)
            return name, "ok"
        except Exception as e:  # noqa
            err = e
    return name, f"FAIL {err}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=os.path.join(os.environ.get("FLOODVALUE_DATA_DIR", "data"), "raw"))
    ap.add_argument("--with-weak", action="store_true")
    ap.add_argument("--no-s2", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    prefixes = list(PREFIXES_DEFAULT)
    if not a.no_s2:
        prefixes.append(PREFIX_S2)
    if a.with_weak:
        prefixes.append(PREFIX_WEAK)
    items = []
    for p in prefixes:
        items += list_objects(p)
    total = sum(int(i["size"]) for i in items)
    print(f"{len(items)} файлов, {total/1e9:.2f} ГБ -> {a.dest}", flush=True)
    manifest = {i["name"]: {"size": int(i["size"]), "md5Hash": i.get("md5Hash"), "crc32c": i.get("crc32c"),
                            "updated": i.get("updated"), "generation": i.get("generation")} for i in items}
    os.makedirs(a.dest, exist_ok=True)
    with open(os.path.join(a.dest, "bucket_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
    stat = {"ok": 0, "skip": 0, "fail": 0}
    with ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(fetch, i, a.dest) for i in items]
        for n, fu in enumerate(as_completed(futs), 1):
            name, st = fu.result()
            stat["fail" if st.startswith("FAIL") else st] += 1
            if st.startswith("FAIL"):
                print(name, st, flush=True)
            if n % 200 == 0 or n == len(futs):
                print(f"[{n}/{len(futs)}] {stat}", flush=True)
    print("done", stat, flush=True)
    sys.exit(1 if stat["fail"] else 0)


if __name__ == "__main__":
    main()
