"""`make data`: get Sen1Floods11 v1.1 HandLabeled S1Hand + LabelHand + JRCWaterHand, verify
checksums, QC every chip, write interim/qc.csv and interim/source_manifest_s1.json.

Network is touched only when a required file is missing under $FLOODVALUE_DATA_DIR/raw
(layout of docs/data.md §3). A second run downloads nothing: it re-hashes and re-checks.
"""

import base64
import csv
import hashlib
import json
import os
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import numpy as np
import rasterio

from floodvalue.data.io import (
    JRC_LAYER,
    LABEL_LAYER,
    LAYERS,
    S1_LAYER,
    chip_id_from_filename,
    interim_dir,
    layer_relpath,
    parse_chip_id,
    raw_dir,
)
from floodvalue.data.split import OPEN_PARTS, SPLIT_PATH, load_split

BUCKET = "sen1floods11"
API = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o"
MEDIA = f"https://storage.googleapis.com/{BUCKET}/"
VERSION_PREFIX = "v1.1/"
DATASET_VERSION = "Sen1Floods11 v1.1"

METADATA_REL = "Sen1Floods11_Metadata.geojson"
SPLIT_FILES = [f"splits/flood_handlabeled/flood_{s}_data.csv"
               for s in ("train", "valid", "test", "bolivia")]
BASE_FILES = [METADATA_REL, *SPLIT_FILES]
BASE_PREFIXES = [VERSION_PREFIX + METADATA_REL, VERSION_PREFIX + "splits/flood_handlabeled/"]
LAYER_PREFIXES = [f"{VERSION_PREFIX}data/flood_events/HandLabeled/{layer}/" for layer in LAYERS]

# File prefix -> `location` in Sen1Floods11_Metadata.geojson (docs/data.md §6).
EVENT_ALIASES = {"Mekong": "Cambodia"}

# dB check: linear power is >= 0, so dB data has negative medians; values above DB_MAX
# would mean a wrong scale. Values below DB_LOW_WARN occur (Bolivia_233925 VH -103.9 dB) and
# are only reported: models clip them.
DB_MAX = 60.0
DB_LOW_WARN = -100.0
GRID_ATOL_DEG = 1e-9  # S1 vs label pixel step differ by ~1e-17 deg (docs/data.md §4)

LICENSE = ("не указана явно: в репозитории cloudtostreet/Sen1Floods11 нет LICENSE "
           "(GitHub API license=null), в STAC-коллекциях \"license\": \"proprietary\"; "
           "авторы просят цитировать Bonafilia et al. 2020, CVPRW")

QC_FIELDS = [
    "chip_id", "event_id", "s1_file", "s1_sha256", "s1_md5_ok", "label_file", "label_sha256",
    "label_md5_ok", "jrc_file", "jrc_sha256", "width", "height", "crs", "s1_bands", "s1_dtype",
    "s1_nodata", "s1_band_names", "label_bands", "label_dtype", "grid_match",
    "label_checked",
    "vv_nan_frac", "vh_nan_frac", "vv_min", "vv_median", "vv_max", "vh_min", "vh_median",
    "vh_max", "db_scale_ok", "label_values", "label_values_ok", "label_invalid_frac",
    "valid_frac", "qc_pass", "usable", "issues",
]


# --- bucket access -------------------------------------------------------------------------

def bucket_name(rel: str) -> str:
    """raw-relative path -> object name in the bucket."""
    if rel.startswith("HandLabeled/"):
        return f"{VERSION_PREFIX}data/flood_events/{rel}"
    return VERSION_PREFIX + rel


def local_rel(name: str) -> str:
    rel = name[len(VERSION_PREFIX):]
    for pre in ("data/flood_events/", "data/"):
        if rel.startswith(pre):
            return rel[len(pre):]
    return rel


def list_objects(prefix: str) -> list[dict]:
    items, token = [], None
    while True:
        q = {"prefix": prefix, "maxResults": "1000",
             "fields": "items(name,size,md5Hash,crc32c,updated,generation),nextPageToken"}
        if token:
            q["pageToken"] = token
        with urllib.request.urlopen(f"{API}?{urllib.parse.urlencode(q)}", timeout=60) as r:
            d = json.load(r)
        items += d.get("items", [])
        token = d.get("nextPageToken")
        if not token:
            return items


def download(name: str, out: Path, md5_b64: str | None, attempts: int = 5) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")
    err: Exception | None = None
    for _ in range(attempts):
        try:
            with urllib.request.urlopen(MEDIA + urllib.parse.quote(name), timeout=120) as r, \
                    open(tmp, "wb") as f:
                f.writelines(iter(lambda: r.read(1 << 20), b""))
            if md5_b64 and file_hashes(tmp)[1] != md5_b64:
                raise OSError(f"md5 mismatch for {name}")
            os.replace(tmp, out)
            return
        except Exception as e:  # noqa: BLE001 - retry any transport error
            err = e
    raise RuntimeError(f"download failed: {name}: {err}")


def load_bucket_manifest(raw: Path) -> dict:
    p = raw / "bucket_manifest.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def save_bucket_manifest(raw: Path, items: list[dict]) -> dict:
    """Merge listed objects into raw/bucket_manifest.json (same format as the script)."""
    manifest = load_bucket_manifest(raw)
    for i in items:
        manifest[i["name"]] = {"size": int(i["size"]), "md5Hash": i.get("md5Hash"),
                               "crc32c": i.get("crc32c"), "updated": i.get("updated"),
                               "generation": i.get("generation")}
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "bucket_manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
    return manifest


def fetch_missing(raw: Path, rels: list[str], prefixes: list[str], workers: int = 16) -> int:
    """Download the raw-relative files in `rels` that are absent. No network if none are."""
    missing = [r for r in rels if not (raw / r).exists()]
    if not missing:
        return 0
    print(f"data: нет {len(missing)} файлов в {raw}, качаю из gs://{BUCKET}", flush=True)
    items = [i for p in prefixes for i in list_objects(p)]
    manifest = save_bucket_manifest(raw, items)
    names = [bucket_name(r) for r in missing]
    absent = [n for n in names if n not in manifest]
    if absent:
        raise RuntimeError(f"нет в бакете: {absent[:5]}")
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(lambda n: download(n, raw / local_rel(n), manifest[n].get("md5Hash")),
                    names))
    return len(missing)


# --- inventory -----------------------------------------------------------------------------

def chip_inventory(raw: Path) -> list[str]:
    """All HandLabeled chip ids from the authors' four list files (used only as an
    inventory of 446 chips; the project split is splits/split_v1.json)."""
    chips = set()
    for rel in SPLIT_FILES:
        with open(raw / rel, newline="", encoding="utf-8") as f:
            for row in csv.reader(f):
                if not row:
                    continue
                s1, label = row[0].strip(), row[1].strip()
                cid = chip_id_from_filename(s1)
                if chip_id_from_filename(label) != cid:
                    raise ValueError(f"{rel}: pair mismatch {s1} / {label}")
                chips.add(cid)
    return sorted(chips)


def event_dates(raw: Path) -> dict[str, str]:
    """location -> s1_date as ISO date (metadata stores YYYY/MM/DD)."""
    d = json.loads((raw / METADATA_REL).read_text(encoding="utf-8"))
    return {f["properties"]["location"]:
            date(*map(int, f["properties"]["s1_date"].split("/"))).isoformat()
            for f in d["features"]}


def event_s1_date(event_id: str, dates: dict[str, str]) -> str | None:
    return dates.get(EVENT_ALIASES.get(event_id, event_id))


def label_readable_chips(path: Path | None = None) -> set[str]:
    """Chips whose label pixels QC may read: train/dev/validation and `excluded` of the
    frozen split. Test chips are sealed (D-010): QC sees only their header and file hash.
    No split file -> no label pixels are read at all (fail closed)."""
    path = path or SPLIT_PATH
    if not path.exists():
        return set()
    split = load_split(path)
    return {c for part in OPEN_PARTS for c in split[part]} | set(split.get("excluded", {}))


# --- QC ------------------------------------------------------------------------------------

def file_hashes(path: Path) -> tuple[str, str]:
    """(sha256 hex, md5 base64 as in GCS md5Hash) in one pass."""
    sha, md5 = hashlib.sha256(), hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            sha.update(chunk)
            md5.update(chunk)
    return sha.hexdigest(), base64.b64encode(md5.digest()).decode()


def grids_match(a, b) -> bool:
    """Same CRS, shape and transform up to float noise (never `==` on transforms)."""
    return (a.crs == b.crs and a.width == b.width and a.height == b.height
            and np.allclose(tuple(a.transform)[:6], tuple(b.transform)[:6],
                            rtol=0, atol=GRID_ATOL_DEG))


def _num(x: float) -> str:
    return "" if x is None or not np.isfinite(x) else f"{float(x):.6g}"


def _stats(band: np.ndarray) -> tuple[float, float, float, float]:
    finite = band[np.isfinite(band)]
    nan_frac = 1.0 - finite.size / band.size
    if finite.size == 0:
        return nan_frac, np.nan, np.nan, np.nan
    return nan_frac, float(finite.min()), float(np.median(finite)), float(finite.max())


def db_scale_ok(vv: np.ndarray, vh: np.ndarray) -> bool | None:
    """True when values look like dB (negative medians, max <= DB_MAX). Linear power (>= 0)
    fails. None when a band has no finite pixel (nothing to check)."""
    fv, fh = vv[np.isfinite(vv)], vh[np.isfinite(vh)]
    if fv.size == 0 or fh.size == 0:
        return None
    return bool(np.median(fv) < 0 and np.median(fh) < 0 and max(fv.max(), fh.max()) <= DB_MAX)


def qc_chip(chip_id: str, raw: Path, bucket: dict, read_label: bool = True) -> dict:
    """QC one chip. With read_label=False (sealed test chip) the label file is hashed and
    its header (dtype, bands, CRS, grid) checked, but its pixels are never read."""
    event_id, _ = parse_chip_id(chip_id)
    issues: list[str] = []  # format / integrity problems -> qc_pass=False
    notes: list[str] = []  # data properties worth knowing (all-NaN, all -1, extremes)
    row: dict = {"chip_id": chip_id, "event_id": event_id}
    for key, layer in (("s1", S1_LAYER), ("label", LABEL_LAYER), ("jrc", JRC_LAYER)):
        rel = layer_relpath(chip_id, layer)
        row[f"{key}_file"] = rel
        if not (raw / rel).exists():
            issues.append(f"{layer} missing")
            row[f"{key}_sha256"] = ""
            continue
        sha, md5 = file_hashes(raw / rel)
        row[f"{key}_sha256"] = sha
        expected = bucket.get(bucket_name(rel), {}).get("md5Hash")
        if key != "jrc":
            row[f"{key}_md5_ok"] = "" if expected is None else md5 == expected
        if expected is not None and md5 != expected:
            # a corrupted file fails the chip, JRC included; only a missing JRC is tolerated
            issues.append(f"md5 mismatch: {layer}")
    if any(i.endswith("missing") and not i.startswith(JRC_LAYER) for i in issues):
        row.update(qc_pass=False, usable=False, issues="; ".join(issues))
        return row

    with rasterio.open(raw / row["s1_file"]) as s1, rasterio.open(raw / row["label_file"]) as lb:
        row.update(width=s1.width, height=s1.height, crs=str(s1.crs), s1_bands=s1.count,
                   s1_dtype=",".join(sorted(set(s1.dtypes))), s1_nodata=str(s1.nodata),
                   s1_band_names="/".join(str(d) for d in s1.descriptions),
                   label_bands=lb.count, label_dtype=",".join(sorted(set(lb.dtypes))),
                   grid_match=grids_match(s1, lb))
        s1_arr = s1.read()
        label = lb.read(1) if read_label else None
    row["label_checked"] = read_label
    if row["s1_bands"] != 2:
        issues.append(f"S1 bands={row['s1_bands']} (expected 2)")
    if row["s1_dtype"] != "float32":
        issues.append(f"S1 dtype={row['s1_dtype']}")
    if row["label_bands"] != 1:
        issues.append(f"label bands={row['label_bands']}")
    if row["label_dtype"] != "int16":
        issues.append(f"label dtype={row['label_dtype']}")
    if row["s1_nodata"] != "nan":
        issues.append(f"S1 nodata={row['s1_nodata']} (expected nan)")
    if row["s1_band_names"] != "VV/VH":
        issues.append(f"S1 band names={row['s1_band_names']} (expected VV/VH)")
    if row["crs"] != "EPSG:4326":
        issues.append(f"crs={row['crs']}")
    if not row["grid_match"]:
        issues.append("S1 grid != label grid")

    if s1_arr.shape[0] >= 2:
        vv, vh = s1_arr[0].astype(np.float64), s1_arr[1].astype(np.float64)
        for name, band in (("vv", vv), ("vh", vh)):
            nan_frac, lo, med, hi = _stats(band)
            row.update({f"{name}_nan_frac": _num(nan_frac), f"{name}_min": _num(lo),
                        f"{name}_median": _num(med), f"{name}_max": _num(hi)})
        row["db_scale_ok"] = db_scale_ok(vv, vh)
        if row["db_scale_ok"] is False:
            issues.append("S1 values not in dB scale")
        lows = [v for v in (row["vv_min"], row["vh_min"]) if v != ""]
        if lows and min(float(v) for v in lows) < DB_LOW_WARN:
            notes.append(f"S1 min < {DB_LOW_WARN:g} dB (clip)")

    valid = np.isfinite(s1_arr[:2]).all(axis=0)
    if label is not None:
        values = sorted(int(v) for v in np.unique(label))
        row["label_values"] = " ".join(str(v) for v in values)
        row["label_values_ok"] = set(values) <= {-1, 0, 1}
        if not row["label_values_ok"]:
            issues.append(f"label values {values}")
        row["label_invalid_frac"] = _num(float((label == -1).mean()))
        if valid.shape == label.shape:
            valid = valid & (label != -1)
        if (label == -1).all():
            notes.append("label all -1")
        row["valid_frac"] = _num(float(valid.mean()))
    else:
        notes.append("label pixels sealed (test)")
    if not np.isfinite(s1_arr).any():
        notes.append("S1 all NaN")

    # JRC is diagnostics only (D-007): its absence does not fail the chip.
    hard = [i for i in issues if not i.startswith(JRC_LAYER)]
    row["qc_pass"] = not hard
    # usable = format is fine and at least one pixel has finite S1 (and a label, if read).
    row["usable"] = row["qc_pass"] and bool(valid.any())
    row["issues"] = "; ".join(issues + notes)
    return row


def write_qc(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=QC_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k, "")) for k in QC_FIELDS})


# --- source manifest (S1 part) -------------------------------------------------------------

def s1_source_entries(rows: list[dict], dates: dict[str, str], bucket: dict) -> list[dict]:
    """One entry per chip: the S1 file (model input) with its hand label attached.
    `export` copies the entries of the chips it uses into source_manifest.json."""
    entries = []
    for r in rows:
        s1_name, lb_name = bucket_name(r["s1_file"]), bucket_name(r["label_file"])
        s1_obj, lb_obj = bucket.get(s1_name, {}), bucket.get(lb_name, {})
        obs = event_s1_date(r["event_id"], dates)
        entries.append({
            "source_id": f"sen1floods11-v1.1:{S1_LAYER}:{r['chip_id']}",
            "chip_id": r["chip_id"],
            "event_id": r["event_id"],
            "file": Path(r["s1_file"]).name,
            "url": MEDIA + s1_name,
            "version": DATASET_VERSION + (f", GCS generation {s1_obj['generation']}"
                                          if s1_obj.get("generation") else ""),
            "sha256": r["s1_sha256"],
            "md5_base64": s1_obj.get("md5Hash"),
            "observation_date": obs,
            "observation_date_source": "s1_date события в Sen1Floods11_Metadata.geojson",
            "published_at": s1_obj.get("updated"),
            "published_after_event": True,
            "license": LICENSE,
            "purpose": "вход Sentinel-1 VV/VH (dB) для baseline и основного метода",
            "limitations": (
                "Чип — переобработанная авторами копия сцены S1 GRD от даты наблюдения, "
                "опубликована в бакете в 2020 г., после события. В имитации решения считается, "
                "что сама сцена S1 доступна в день наблюдения (оперативная выдача Copernicus); "
                "содержание чипа не использует сведений после даты наблюдения. Значения уже в "
                "dB, nodata = NaN, EPSG:4326, шаг 8.983e-05 град."),
            "label": {
                "file": Path(r["label_file"]).name,
                "url": MEDIA + lb_name,
                "sha256": r["label_sha256"],
                "md5_base64": lb_obj.get("md5Hash"),
                "published_at": lb_obj.get("updated"),
                "purpose": "ручная метка воды (-1 нет метки, 0 не вода, 1 вода) — только "
                           "обучение и оценка, не вход метода",
                "limitations": "размечена по Sentinel-2 и опубликована после события; "
                               "это вся вода, не отдельно паводковая (D-007)",
            },
        })
    return entries


def write_source_manifest(entries: list[dict], path: Path) -> None:
    doc = {"dataset": DATASET_VERSION, "bucket": f"gs://{BUCKET}/{VERSION_PREFIX}",
           "repository": "https://github.com/cloudtostreet/Sen1Floods11",
           "license": LICENSE, "sources": entries}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=1, allow_nan=False), encoding="utf-8")


# --- stage ---------------------------------------------------------------------------------

def run(args=None) -> int:
    raw = raw_dir()
    n_base = fetch_missing(raw, BASE_FILES, BASE_PREFIXES)
    chips = chip_inventory(raw)
    rels = [layer_relpath(c, layer) for c in chips for layer in LAYERS]
    n_chips = fetch_missing(raw, rels, LAYER_PREFIXES)
    print(f"data: {len(chips)} чипов, скачано файлов: {n_base + n_chips}", flush=True)

    bucket = load_bucket_manifest(raw)
    readable = label_readable_chips()
    if not readable:
        print(f"data: нет {SPLIT_PATH.name} — пиксели меток не читаются (test запечатан)",
              flush=True)
    with ThreadPoolExecutor(min(8, os.cpu_count() or 1)) as ex:
        rows = list(ex.map(lambda c: qc_chip(c, raw, bucket, c in readable), chips))
    out = interim_dir()
    write_qc(rows, out / "qc.csv")
    dates = event_dates(raw)
    no_date = sorted({r["event_id"] for r in rows if event_s1_date(r["event_id"], dates) is None})
    if no_date:
        print(f"data: нет даты S1 для событий {no_date}", flush=True)
    write_source_manifest(s1_source_entries(rows, dates, bucket), out / "source_manifest_s1.json")

    failed = [r for r in rows if not r["qc_pass"]]
    unusable = [r for r in rows if r["qc_pass"] and not r["usable"]]
    n_lab = sum(bool(r.get("label_checked")) for r in rows)
    print(f"data: пиксели меток прочитаны у {n_lab} чипов (train/dev/validation/excluded), "
          f"у {len(rows) - n_lab} — только заголовок и хеш", flush=True)
    print(f"data: QC формата {len(rows) - len(failed)}/{len(rows)}; без валидных пикселей "
          f"{len(unusable)}: {' '.join(r['chip_id'] for r in unusable)} -> {out / 'qc.csv'}",
          flush=True)
    for r in failed[:20]:
        print(f"  FAIL {r['chip_id']}: {r['issues']}", flush=True)
    return 1 if failed or no_date else 0
