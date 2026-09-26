"""Place resolution with an explicit precision level. Coordinates come only from the source's
structured fields, an eyewitness pin, an analyst or a geocoder — never from text rules or a
model. Unknown stays unknown: no [0, 0], no country centre. GeoJSON order is [lon, lat]."""

from __future__ import annotations

import json
import math
import sqlite3
from typing import Protocol

from floodvalue.media.netfetch import Fetcher, FetchError
from floodvalue.media.store import now_iso

AMBIGUOUS_KM = 50.0
NOMINATIM = "https://nominatim.openstreetmap.org/search"
# Nominatim addresstype → our precision (https://nominatim.org/release-docs/latest/api/Output/)
NOMINATIM_PRECISION = {
    "house": "point", "building": "point", "amenity": "point", "road": "street",
    "hamlet": "locality", "village": "locality", "town": "locality", "city": "locality",
    "suburb": "locality", "neighbourhood": "locality", "municipality": "district",
    "county": "district", "district": "district", "state_district": "district",
    "state": "region", "province": "region", "region": "region", "country": "country",
}


class Geocoder(Protocol):
    def lookup(self, query: str) -> list[dict]: ...


def haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    lon1, lat1, lon2, lat2 = map(math.radians, (*a, *b))
    h = (math.sin((lat2 - lat1) / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2)
    return 2 * 6371.0 * math.asin(math.sqrt(h))


def valid_lon_lat(lon, lat) -> bool:
    return (isinstance(lon, (int, float)) and isinstance(lat, (int, float))
            and math.isfinite(lon) and math.isfinite(lat)
            and -180 <= lon <= 180 and -90 <= lat <= 90 and not (lon == 0 and lat == 0))


def bbox_polygon(west: float, south: float, east: float, north: float) -> dict:
    return {"type": "Polygon", "coordinates": [[[west, south], [east, south], [east, north],
                                                [west, north], [west, south]]]}


def representative_point(geometry: dict | None) -> tuple[float, float] | None:
    if not geometry:
        return None
    if geometry["type"] == "Point":
        return tuple(geometry["coordinates"][:2])
    ring = geometry["coordinates"][0]
    xs, ys = [p[0] for p in ring], [p[1] for p in ring]
    return ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2)


def geometry_bbox(geometry: dict) -> tuple[float, float, float, float]:
    if geometry["type"] == "Point":
        x, y = geometry["coordinates"][:2]
        return x, y, x, y
    ring = geometry["coordinates"][0]
    xs, ys = [p[0] for p in ring], [p[1] for p in ring]
    return min(xs), min(ys), max(xs), max(ys)


class NominatimGeocoder:
    """Public OSM Nominatim under its usage policy: ≤1 request/s, identifying User-Agent,
    results cached, no bulk use. Enabled only by FLOODVALUE_MEDIA_GEOCODER=nominatim."""

    def __init__(self, conn: sqlite3.Connection, fetcher: Fetcher, max_lookups: int = 10):
        self.conn, self.fetcher, self.left = conn, fetcher, max_lookups

    def lookup(self, query: str) -> list[dict]:
        key = query.casefold().strip()
        row = self.conn.execute("SELECT result_json FROM geocode_cache WHERE query = ?",
                                (key,)).fetchone()
        if row:
            return json.loads(row["result_json"])
        if self.left <= 0:
            raise FetchError("лимит геокодирования за цикл исчерпан")
        self.left -= 1
        resp = self.fetcher.get(NOMINATIM, params={"q": query, "format": "jsonv2", "limit": "5",
                                                   "accept-language": "ru,en"})
        out = []
        for r in resp.json():
            lon, lat = float(r["lon"]), float(r["lat"])
            if not valid_lon_lat(lon, lat):
                continue
            south, north, west, east = map(float, r.get("boundingbox", [lat, lat, lon, lon]))
            out.append({"label": r.get("display_name", query), "lon": lon, "lat": lat,
                        "precision": NOMINATIM_PRECISION.get(r.get("addresstype"), "locality"),
                        "bbox": [west, south, east, north]})
        self.conn.execute("INSERT OR REPLACE INTO geocode_cache VALUES(?,?,?)",
                          (key, json.dumps(out, ensure_ascii=False), now_iso()))
        return out


def candidate_geometry(c: dict) -> dict:
    if c["precision"] in ("point", "street", "locality") or not c.get("bbox"):
        return {"type": "Point", "coordinates": [c["lon"], c["lat"]]}
    return bbox_polygon(*c["bbox"])


def resolve(mentions: list[str], geocoder: Geocoder | None) -> dict:
    """One mention → one resolution. Several distant candidates → ambiguous with the list for
    the analyst, never a pick. No geocoder or no hit → missing."""
    label = mentions[0] if mentions else None
    missing = {"label": label, "geometry": None, "precision": "unknown", "status": "missing",
               "candidates": None}
    if not mentions or geocoder is None:
        return missing
    try:
        cands = geocoder.lookup(mentions[0])
    except FetchError:
        return missing
    if not cands:
        return missing
    first = (cands[0]["lon"], cands[0]["lat"])
    if any(haversine_km(first, (c["lon"], c["lat"])) > AMBIGUOUS_KM for c in cands[1:]):
        return {**missing, "status": "ambiguous", "candidates": cands}
    c = cands[0]
    return {"label": c["label"], "geometry": candidate_geometry(c), "precision": c["precision"],
            "status": "resolved", "candidates": None}
