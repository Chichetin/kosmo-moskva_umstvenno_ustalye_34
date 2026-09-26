"""Source adapters behind one interface: fetch(fetcher, config, cursor) -> (items, new_cursor).
An item is a normalized publication dict; the adapter never decides what the item means
beyond what the source states in structured fields (a bulletin's dates and coordinates)."""

from __future__ import annotations

import hashlib
import html
import json
import re
import xml.etree.ElementTree as ET
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from floodvalue.media.netfetch import Fetcher, FetchError

TRACKING = re.compile(r"^(utm_|fbclid$|gclid$|yclid$|mc_|ref$|ref_src$)")
TAG = re.compile(r"<[^>]+>")
EXCERPT_CHARS = 300


def canonical_url(url: str) -> str:
    parts = urlsplit(url.strip())
    query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
             if not TRACKING.match(k.lower())]
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme.lower(), (parts.hostname or "").lower(), path,
                       urlencode(sorted(query)), ""))


def title_key(title: str) -> str:
    """Reprints keep the headline: casefolded letters/digits only, whitespace collapsed."""
    words = re.findall(r"\w+", title.casefold())
    return " ".join(words)


def content_hash(*parts: str | None) -> str:
    return hashlib.sha256("\n".join(p or "" for p in parts).encode()).hexdigest()


def plain_text(raw: str | None) -> str:
    return re.sub(r"\s+", " ", html.unescape(TAG.sub(" ", raw or ""))).strip()


def iso(dt: datetime) -> str:
    return dt.astimezone(UTC).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_cursor(cursor: str | None, default: timedelta) -> datetime:
    if cursor:
        return datetime.strptime(cursor, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    return datetime.now(UTC) - default


# --- GDACS (bulletin) --------------------------------------------------------------------------

def fetch_gdacs(fetcher: Fetcher, config: dict, cursor: str | None, now: datetime | None = None):
    """GDACS event list API (JSON FeatureCollection). The API filters by the event's start
    date, not by modification, so a cursor would miss updates of long floods: every cycle asks
    for a sliding `lookback_days` window and the register dedups by id + datemodified.
    HTTP 204 is the API's "no events in the window" — an answer, not an error."""
    now = now or datetime.now(UTC)
    since = now - timedelta(days=config.get("lookback_days", 30))
    resp = fetcher.get(config["endpoint"], params={
        "eventlist": "FL", "fromDate": since.strftime("%Y-%m-%d"),
        "toDate": (now + timedelta(days=1)).strftime("%Y-%m-%d")})
    if resp.status_code == 204:
        return [], iso(now)
    try:
        data = resp.json()
        features = data["features"]
    except (ValueError, KeyError, TypeError) as exc:
        raise FetchError(f"GDACS: неожиданный формат ответа ({exc})") from exc
    return [gdacs_item(f) for f in features], iso(now)


def gdacs_item(feature: dict) -> dict:
    p = feature["properties"]
    geom = feature.get("geometry")
    if geom and geom.get("type") == "Point":
        lon, lat = geom["coordinates"][:2]
        if not (-180 <= lon <= 180 and -90 <= lat <= 90) or (lon == 0 and lat == 0):
            geom = None
    key = f"FL-{p['eventid']}-{p.get('episodeid', '')}"
    title = f"{p.get('name') or 'Flood'} ({p.get('alertlevel', '?')})"
    summary = plain_text(p.get("htmldescription") or p.get("description"))
    report_url = (p.get("url") or {}).get("report") or ""
    return {
        "external_id": key,
        "url": report_url,
        "title": title,
        "text_excerpt": summary[:EXCERPT_CHARS],
        "language": "en",
        "source_domain": "gdacs.org",
        "published_at": _gdacs_time(p.get("datemodified")),
        "source_updated_at": _gdacs_time(p.get("datemodified")),
        "structured": {
            "kind": "bulletin",
            "event_key": f"gdacs:FL:{p['eventid']}",
            "report_type": "occurred",
            "time": {"start": _gdacs_time(p.get("fromdate")), "end": _gdacs_time(p.get("todate")),
                     "precision": "day"},
            "place": {"label": p.get("country") or None, "country": p.get("iso3"),
                      "geometry": geom, "precision": "region" if geom else "unknown",
                      "note": "центроид области по GDACS" if geom else None},
            "alert_level": p.get("alertlevel"),
            "provider": p.get("source"),
            "glide": p.get("glide"),
        },
    }


def _gdacs_time(raw: str | None) -> str | None:
    if not raw:
        return None
    return iso(datetime.fromisoformat(raw).replace(tzinfo=UTC))


# --- GDELT DOC 2.0 (news metadata: headline + URL only) -----------------------------------------

def fetch_gdelt(fetcher: Fetcher, config: dict, cursor: str | None, now: datetime | None = None):
    """GDELT DOC 2.0 `mode=artlist&format=json`. `seendate` is when GDELT saw the article,
    which is the closest publication time the API gives; it is marked as such."""
    now = now or datetime.now(UTC)
    since = _parse_cursor(cursor, timedelta(hours=config.get("initial_hours", 24)))
    since -= timedelta(minutes=config.get("overlap_minutes", 30))
    since = max(since, now - timedelta(days=89))  # the API only serves the last 3 months
    resp = fetcher.get(config["endpoint"], params={
        "query": config["query"], "mode": "artlist", "format": "json", "sort": "datedesc",
        "maxrecords": str(min(int(config.get("maxrecords", 100)), 250)),
        "startdatetime": since.strftime("%Y%m%d%H%M%S"),
        "enddatetime": now.strftime("%Y%m%d%H%M%S")})
    text = resp.text.strip()
    if not text:
        return [], iso(now)  # the API answers an empty body when nothing matched
    try:
        articles = json.loads(text).get("articles", [])
    except ValueError as exc:
        raise FetchError(f"GDELT: ответ не JSON ({text[:120]!r})") from exc
    return [gdelt_item(a) for a in articles if a.get("url") and a.get("title")], iso(now)


def gdelt_item(article: dict) -> dict:
    seen = article.get("seendate")
    seen_iso = iso(datetime.strptime(seen, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)) if seen else None
    return {
        "external_id": canonical_url(article["url"]),
        "url": article["url"],
        "title": plain_text(article["title"]),
        "text_excerpt": None,
        "language": article.get("language"),
        "source_domain": article.get("domain"),
        "published_at": seen_iso,
        "source_updated_at": None,
        "time_basis": "gdelt_seendate",
        "structured": None,
    }


# --- Generic RSS 2.0 (off by default: each feed needs its own terms check) ---------------------

def fetch_rss(fetcher: Fetcher, config: dict, cursor: str | None, now: datetime | None = None):
    now = now or datetime.now(UTC)
    resp = fetcher.get(config["endpoint"])
    try:
        root = ET.fromstring(resp.content)
    except ET.ParseError as exc:
        raise FetchError(f"RSS: не XML ({exc})") from exc
    keep_excerpt = config.get("content_storage_policy") in ("summary", "excerpt")
    items = []
    for node in root.iter("item"):
        link, title = (node.findtext("link") or "").strip(), plain_text(node.findtext("title"))
        if not link or not title:
            continue
        published = node.findtext("pubDate")
        try:
            published = iso(parsedate_to_datetime(published)) if published else None
        except (TypeError, ValueError):
            published = None
        items.append({
            "external_id": (node.findtext("guid") or "").strip() or canonical_url(link),
            "url": link, "title": title,
            "text_excerpt": plain_text(node.findtext("description"))[:EXCERPT_CHARS]
            if keep_excerpt else None,
            "language": config.get("language"), "source_domain": urlsplit(link).hostname,
            "published_at": published, "source_updated_at": None, "structured": None,
        })
    return items, iso(now)


ADAPTERS = {"gdacs": fetch_gdacs, "gdelt_doc": fetch_gdelt, "rss": fetch_rss}
