"""Geocoding through Nominatim under its usage policy, with a persistent cache.

Policy (operations.osmfoundation.org/policies/nominatim, checked 2026-09-26): at most one
request per second, an identifying User-Agent, cached results, no search-as-you-type,
attribution «© OpenStreetMap contributors». Coordinates only ever come from here or from
a structured source; text extraction never invents them.
"""

from __future__ import annotations

import json
import re
import sqlite3

from floodvalue.media.net import FetchError, fetch
from floodvalue.media.store import now

NOMINATIM = "https://nominatim.openstreetmap.org/search"
HOSTS = {"nominatim.openstreetmap.org"}
ATTRIBUTION = "© OpenStreetMap contributors, геокодирование Nominatim"
PRECISION = {
    "house": "point", "building": "point", "amenity": "point", "shop": "point",
    "road": "street", "street": "street",
    "city": "locality", "town": "locality", "village": "locality", "hamlet": "locality",
    "suburb": "locality", "neighbourhood": "locality", "quarter": "locality",
    "municipality": "locality", "isolated_dwelling": "locality",
    "county": "district", "district": "district", "city_district": "district",
    "state_district": "district",
    "state": "region", "region": "region", "province": "region",
    "country": "country",
}


def precision_of(item: dict) -> str:
    return PRECISION.get(str(item.get("addresstype") or item.get("type")), "unknown")


def normalize(item: dict) -> dict:
    south, north, west, east = (float(value) for value in item["boundingbox"])
    return {"label": item.get("display_name", ""), "lat": float(item["lat"]),
            "lon": float(item["lon"]), "precision": precision_of(item),
            "bbox": [west, south, east, north], "country_code": None}


def _key(query: str, language: str) -> str:
    return language + "|" + re.sub(r"\s+", " ", query.strip().lower())


def geocode(db: sqlite3.Connection, query: str, *, language: str = "ru", limit: int = 5,
            allow_network: bool = True) -> list[dict] | None:
    """Candidates for a place name; None means 'not in cache and network not allowed'."""
    key = _key(query, language)
    row = db.execute("SELECT result_json FROM geocode_cache WHERE query = ?", (key,)).fetchone()
    if row:
        return json.loads(row["result_json"])
    if not allow_network:
        return None
    response = fetch(NOMINATIM, allowed_hosts=HOSTS, min_interval=1.1, attempts=2,
                     params={"q": query, "format": "jsonv2", "limit": limit,
                             "accept-language": language, "addressdetails": 1})
    try:
        raw = json.loads(response.content)
    except ValueError as exc:
        raise FetchError("геокодер вернул не JSON") from exc
    result = []
    for item in raw:
        entry = normalize(item)
        entry["country_code"] = (item.get("address") or {}).get("country_code")
        result.append(entry)
    db.execute("INSERT OR REPLACE INTO geocode_cache(query, result_json, fetched_at) "
               "VALUES (?, ?, ?)", (key, json.dumps(result, ensure_ascii=False), now()))
    return result
