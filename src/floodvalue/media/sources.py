"""Source adapters with one interface: fetch(config, cursor, budget, limit) → Batch.

Formats checked against live responses on 2026-09-26: GDACS event list (GeoJSON),
GDELT DOC 2.0 `mode=artlist&format=json`, RSS 2.0 / Atom. Only what the source's storage
policy allows is kept: link and title always, a short excerpt only where permitted.
A failed request raises FetchError — never an empty 'nothing new' batch.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from floodvalue.media.geo import Budget
from floodvalue.media.net import FetchError, fetch
from floodvalue.media.store import iso, parse_iso

EXCERPT_CHARS = 300
SCRIPT = re.compile(r"<(script|style)\b.*?</\1\s*>", re.IGNORECASE | re.DOTALL)
TRACKING = re.compile(r"^(utm_\w+|fbclid|gclid|at_\w+|ocid|cmpid)$", re.IGNORECASE)


@dataclass
class Item:
    external_id: str
    url: str
    title: str
    excerpt: str | None
    content_level: str
    published_at: str | None
    language: str | None
    structured: dict | None = None


@dataclass
class Batch:
    items: list[Item] = field(default_factory=list)
    http_status: int | None = None
    requests_used: int = 0
    cursor: str | None = None


def canonical_url(url: str) -> str:
    parts = urlsplit(url.strip())
    query = urlencode([(key, value) for key, value in parse_qsl(parts.query)
                       if not TRACKING.match(key)])
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path or "/", query, ""))


def content_hash(title: str, excerpt: str | None) -> str:
    text = re.sub(r"\s+", " ", f"{title} {excerpt or ''}".strip().lower())
    return hashlib.sha256(text.encode()).hexdigest()


def plain(text: str | None, limit: int = EXCERPT_CHARS) -> str | None:
    """Tags stripped and entities decoded; the result is shown only as text, never HTML."""
    if not text:
        return None
    text = SCRIPT.sub(" ", text)
    value = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", text))).strip()
    return (value[: limit - 1] + "…") if len(value) > limit else value or None


def _get(config: dict, budget: Budget, url: str, params: dict | None = None):
    if not budget.take():
        raise BudgetExhausted
    return fetch(url, params=params, allowed_hosts=set(config["allowed_hosts"]),
                 min_interval=float(config.get("min_interval_s", 0)))


class BudgetExhausted(RuntimeError):
    """The cycle's request budget is spent; the source is retried next cycle."""


def fetch_gdacs(config: dict, cursor: str | None, budget: Budget, limit: int) -> Batch:
    now = datetime.now(UTC)
    # Re-read two days before the cursor: episodes are updated after publication.
    since = (parse_iso(cursor) - timedelta(days=2)) if cursor else now - timedelta(days=14)
    response = _get(config, budget, config["url"], {
        "eventlist": "FL", "fromDate": since.date().isoformat(),
        "toDate": now.date().isoformat()})
    try:
        data = json.loads(response.content)
    except ValueError as exc:
        raise FetchError("GDACS вернул не JSON", response.status) from exc
    items = []
    for feature in data.get("features", [])[:limit]:
        props = feature.get("properties", {})
        coords = (feature.get("geometry") or {}).get("coordinates") or [None, None]
        if props.get("eventtype") != "FL":
            continue
        report = (props.get("url") or {}).get("report") or ""
        items.append(Item(
            external_id=f"{props.get('eventid')}-{props.get('episodeid')}",
            url=report, title=str(props.get("name") or "GDACS flood"),
            excerpt=plain(props.get("htmldescription")), content_level="structured",
            published_at=(props.get("datemodified") or props.get("fromdate") or "")[:19] + "Z"
            if props.get("datemodified") or props.get("fromdate") else None,
            language="en",
            structured={
                "eventid": props.get("eventid"), "episodeid": props.get("episodeid"),
                "lon": coords[0], "lat": coords[1], "country": props.get("country"),
                "iso3": props.get("iso3"), "alertlevel": props.get("alertlevel"),
                "fromdate": props.get("fromdate"), "todate": props.get("todate"),
                "basis": props.get("source"), "point_kind": props.get("polygonlabel"),
                "iscurrent": props.get("iscurrent")}))
    return Batch(items, response.status, 1, iso(now))


def fetch_gdelt(config: dict, cursor: str | None, budget: Budget, limit: int) -> Batch:
    now = datetime.now(UTC)
    since = max(parse_iso(cursor) - timedelta(minutes=30), now - timedelta(days=1)) \
        if cursor else now - timedelta(hours=6)
    response = _get(config, budget, config["url"], {
        "query": " ".join(str(config["query"]).split()), "mode": "artlist", "format": "json",
        "maxrecords": min(limit, 250), "sort": "DateDesc",
        "startdatetime": since.strftime("%Y%m%d%H%M%S"),
        "enddatetime": now.strftime("%Y%m%d%H%M%S")})
    try:
        data = json.loads(response.content) if response.content.strip() else {}
    except ValueError as exc:
        raise FetchError(f"GDELT вернул не JSON: {response.content[:120]!r}",
                         response.status) from exc
    items = []
    for article in data.get("articles", [])[:limit]:
        seen = article.get("seendate")
        items.append(Item(
            external_id=canonical_url(article["url"]), url=article["url"],
            title=plain(article.get("title"), 400) or article["url"], excerpt=None,
            content_level="title_only",
            # GDELT gives the time it saw the article, not the outlet's publication time.
            published_at=iso(datetime.strptime(seen, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC))
            if seen else None,
            language=article.get("language"),
            structured={"time_basis": "seendate GDELT", "sourcecountry":
                        article.get("sourcecountry")}))
    return Batch(items, response.status, 1, iso(now))


def _text(node: ET.Element | None) -> str | None:
    return node.text.strip() if node is not None and node.text else None


def parse_feed(content: bytes, *, excerpt_allowed: bool, limit: int) -> list[Item]:
    root = ET.fromstring(content)
    atom = "{http://www.w3.org/2005/Atom}"
    entries = root.findall("./channel/item") or root.findall(f"{atom}entry")
    items = []
    for entry in entries[:limit]:
        if entry.tag.startswith(atom):
            link_node = entry.find(f"{atom}link")
            link = link_node.get("href") if link_node is not None else None
            title, guid = _text(entry.find(f"{atom}title")), _text(entry.find(f"{atom}id"))
            date_text = _text(entry.find(f"{atom}updated")) or _text(entry.find(f"{atom}published"))
            summary = _text(entry.find(f"{atom}summary"))
            published = iso(parse_iso(date_text)) if date_text else None
        else:
            link, title = _text(entry.find("link")), _text(entry.find("title"))
            guid, summary = _text(entry.find("guid")), _text(entry.find("description"))
            date_text = _text(entry.find("pubDate"))
            published = iso(parsedate_to_datetime(date_text)) if date_text else None
        if not link or not title:
            continue
        excerpt = plain(summary) if excerpt_allowed else None
        items.append(Item(external_id=guid or canonical_url(link), url=link, title=plain(title, 400)
                          or link, excerpt=excerpt,
                          content_level="excerpt" if excerpt else "title_only",
                          published_at=published, language=None))
    return items


def fetch_rss(config: dict, cursor: str | None, budget: Budget, limit: int) -> Batch:
    response = _get(config, budget, config["url"])
    try:
        items = parse_feed(response.content, limit=limit,
                           excerpt_allowed=config.get("content_storage_policy")
                           == "link_title_excerpt")
    except ET.ParseError as exc:
        raise FetchError(f"лента не разобрана: {exc}", response.status) from exc
    for item in items:
        item.language = config.get("language")
    return Batch(items, response.status, 1, iso(datetime.now(UTC)))


ADAPTERS = {"gdacs": fetch_gdacs, "gdelt": fetch_gdelt, "rss": fetch_rss}
