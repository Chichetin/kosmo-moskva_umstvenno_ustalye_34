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

from floodvalue.media.events import haversine_km
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
            "bbox": [west, south, east, north], "country_code": None,
            "importance": float(item.get("importance") or 0)}


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


PLACE_PRECISIONS = {"locality", "district", "region", "country"}
AMBIGUOUS_KM = 50.0


class Budget:
    """Network calls left in this ingestion cycle."""

    def __init__(self, calls: int):
        self.calls = calls

    def take(self) -> bool:
        if self.calls <= 0:
            return False
        self.calls -= 1
        return True


def pick(results: list[dict], min_importance: float = 0.0) -> dict:
    """Resolved / ambiguous / missing from geocoder candidates; never a guess among equals."""
    places = [item for item in results if item["precision"] in PLACE_PRECISIONS
              and item.get("importance", 0) >= min_importance]
    if not places:
        return {"status": "missing", "chosen": None, "candidates": []}
    top = places[0]
    rivals = [item for item in places[1:]
              if haversine_km(top["lat"], top["lon"], item["lat"], item["lon"]) > AMBIGUOUS_KM
              and item.get("importance", 0) >= 0.6 * top.get("importance", 0)]
    if rivals:
        return {"status": "ambiguous", "chosen": None, "candidates": [top, *rivals][:5]}
    return {"status": "resolved", "chosen": top, "candidates": []}


def resolve(db: sqlite3.Connection, name: str, variants: list[str], budget: Budget,
            min_importance: float = 0.0) -> dict:
    """Try the name and its nominative variants; 'pending' when the budget ran out."""
    outcome = {"status": "missing", "chosen": None, "candidates": []}
    for query in dict.fromkeys([*variants, name]):
        cached = geocode(db, query, allow_network=False)
        if cached is None:
            if not budget.take():
                return {"status": "pending", "chosen": None, "candidates": []}
            cached = geocode(db, query)
        outcome = pick(cached or [], min_importance)
        if outcome["status"] != "missing":
            return outcome
    return outcome


RANK = {"point": 0, "street": 1, "locality": 2, "district": 3, "region": 4, "country": 5}


def most_precise(places: list[dict]) -> list[dict]:
    """Drop a region or country when a more precise resolved place lies inside it."""
    kept = []
    for place in places:
        west, south, east, north = place["bbox"]
        inner = [other for other in places if other is not place
                 and RANK[other["precision"]] < RANK[place["precision"]]
                 and west <= other["lon"] <= east and south <= other["lat"] <= north]
        if not inner:
            kept.append(place)
    return kept
