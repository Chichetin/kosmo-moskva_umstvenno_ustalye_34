"""Sentinel-1 RTC archive of Microsoft Planetary Computer: STAC search, SAS signing, reading VH
of a scene resampled onto the chip grid. The only module of `prewater` that touches the network.
"""

from __future__ import annotations

import time
from datetime import date

import numpy as np
import rasterio
import requests
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-1-rtc"
COLLECTION_URL = f"{STAC_URL}/collections/{COLLECTION}"
SAS_URL = f"https://planetarycomputer.microsoft.com/api/sas/v1/token/{COLLECTION}"
RTC_NODATA = -32768.0
RETRIES = 4
TIMEOUT_S = 60
GDAL_ENV = {"GDAL_HTTP_MAX_RETRY": "5", "GDAL_HTTP_RETRY_DELAY": "2",
            "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
            "GDAL_HTTP_TIMEOUT": str(TIMEOUT_S)}


class ArchiveUnavailable(RuntimeError):
    """The archive could not be reached (no network, HTTP 429/5xx after retries)."""


def _request(method: str, url: str, **kw) -> dict:
    last: Exception | None = None
    for attempt in range(RETRIES):
        try:
            r = requests.request(method, url, timeout=TIMEOUT_S, **kw)
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.HTTPError(f"HTTP {r.status_code}", response=r)
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError) as e:
            last = e
            time.sleep(2 ** attempt)
    raise ArchiveUnavailable(f"{url}: {last}")


def search(bbox: tuple[float, float, float, float], first: date, last: date) -> list[dict]:
    """STAC items of the collection intersecting `bbox` (lon/lat) acquired first..last (UTC days)."""
    body = {"collections": [COLLECTION], "bbox": list(bbox), "limit": 250,
            "datetime": f"{first.isoformat()}T00:00:00Z/{last.isoformat()}T23:59:59Z"}
    items: list[dict] = []
    page = _request("POST", f"{STAC_URL}/search", json=body)
    while True:
        items += page.get("features", [])
        nxt = next((link for link in page.get("links", []) if link.get("rel") == "next"), None)
        if not nxt:
            return items
        page = _request(nxt.get("method", "GET"), nxt["href"],
                        **({"json": nxt["body"]} if nxt.get("body") else {}))


def collection_license() -> str:
    return str(_request("GET", COLLECTION_URL).get("license", "CC-BY-4.0"))


def sas_token() -> str:
    return str(_request("GET", SAS_URL)["token"])


def read_vh(href: str, token: str, transform, shape: tuple[int, int]) -> np.ndarray:
    """VH gamma0 (linear) of one scene on the chip grid (EPSG:4326), NaN outside the scene.

    Averaging 10 m UTM pixels onto the ~10 m geographic chip grid keeps the speckle level of
    the source; the value stays linear power until core.linear_to_db."""
    url = f"{href}?{token}"
    last: Exception | None = None
    for attempt in range(RETRIES):
        try:
            with rasterio.Env(**GDAL_ENV), rasterio.open(url) as src, WarpedVRT(
                    src, crs="EPSG:4326", transform=transform, width=shape[1], height=shape[0],
                    resampling=Resampling.average, src_nodata=RTC_NODATA,
                    nodata=np.nan, dtype="float32") as vrt:
                return vrt.read(1)
        except rasterio.errors.RasterioError as e:
            last = e
            time.sleep(2 ** attempt)
    raise ArchiveUnavailable(f"{href}: {last}")


def acquisitions(items: list[dict]) -> list[list[dict]]:
    """Group items of one pass (same day, platform, relative orbit): adjacent slices of a
    datatake are one observation, not several. Sorted by acquisition time."""
    groups: dict[tuple, list[dict]] = {}
    for it in sorted(items, key=lambda i: i["properties"]["datetime"]):
        p = it["properties"]
        key = (p["datetime"][:10], p.get("platform"), p.get("sat:relative_orbit"))
        groups.setdefault(key, []).append(it)
    return list(groups.values())
