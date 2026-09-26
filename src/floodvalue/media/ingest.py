"""One ingestion cycle: sources → publications → extraction → geography → events.

Runs in the separate worker (python -m floodvalue.media worker) or once by hand; never in
the browser. A file lock keeps two cycles from processing at the same time. Each
publication is committed on its own, so stopping mid-cycle loses nothing already written.
"""

from __future__ import annotations

import fcntl
import json
import re
import sqlite3
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import yaml

from floodvalue.media import events as registry
from floodvalue.media import extract, geo
from floodvalue.media.net import FetchError
from floodvalue.media.sources import (
    ADAPTERS,
    Batch,
    BudgetExhausted,
    Item,
    canonical_url,
    content_hash,
)
from floodvalue.media.store import (
    connect,
    get_setting,
    iso,
    media_dir,
    now,
    parse_iso,
    record_history,
    set_setting,
)
from floodvalue.paths import CONFIGS_DIR

SYSTEM = "system"
TIME_HINTS = (("сегодня|утром|этой ночью|today|this morning|tonight", 0),
              ("вчера|yesterday|overnight", 1))


def load_config() -> dict:
    return yaml.safe_load((CONFIGS_DIR / "media.yaml").read_text(encoding="utf-8"))


def sync_sources(db: sqlite3.Connection, config: dict) -> None:
    """Config describes sources; the operator's on/off choice in the database is kept.
    A source without checked terms is always off (spec §Д)."""
    for item in config["sources"]:
        checked = item.get("terms_status") == "checked"
        db.execute(
            "INSERT INTO source(source_id, name, kind, url, language, region_scope, "
            "poll_interval_s, enabled, terms_status, terms_note, attribution, "
            "content_storage_policy, config_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(source_id) DO UPDATE SET name = excluded.name, kind = excluded.kind, "
            "url = excluded.url, language = excluded.language, "
            "region_scope = excluded.region_scope, poll_interval_s = excluded.poll_interval_s, "
            "terms_status = excluded.terms_status, terms_note = excluded.terms_note, "
            "attribution = excluded.attribution, "
            "content_storage_policy = excluded.content_storage_policy, "
            "config_json = excluded.config_json",
            (item["source_id"], item["name"], item["kind"], item.get("url"),
             item.get("language"), item.get("region_scope"), int(item.get("poll_interval_s", 900)),
             int(bool(item.get("enabled")) and checked), item.get("terms_status", "unchecked"),
             item.get("terms_note"), item.get("attribution"),
             item.get("content_storage_policy", "link_title"), json.dumps(item, ensure_ascii=False)))
        if not checked:
            db.execute("UPDATE source SET enabled = 0 WHERE source_id = ?", (item["source_id"],))


def _time_from_text(text: str, published: str | None) -> tuple[str | None, str | None, str, str]:
    """Event interval from the text and publication time; an honest interval, not a guess."""
    if not published:
        return None, None, "unknown", "время публикации неизвестно"
    moment = parse_iso(published)
    for pattern, days_back in TIME_HINTS:
        if re.search(pattern, text, re.IGNORECASE):
            day = (moment - timedelta(days=days_back)).replace(hour=0, minute=0, second=0)
            return iso(day), iso(min(day + timedelta(days=1), moment)), "day", \
                f"в тексте «{pattern.split('|')[0]}» относительно публикации"
    return iso(moment - timedelta(days=3)), iso(moment), "several_days", \
        "не позже публикации; начало оценено как за 3 суток до неё"


def _place_label(label: str) -> str:
    return ", ".join(part.strip() for part in label.split(",")[:2])


def _link_or_create(db: sqlite3.Connection, *, publication_id: int, origin: str, title: str,
                    report_type: str, place: dict | None, geo_status: str,
                    candidates: list[dict], country: str | None, start: str | None,
                    end: str | None, time_precision: str, facts: dict,
                    allow_create: bool = True) -> tuple[int | None, bool]:
    """Returns (event_id, created). Uncertain matches create a separate event marked as a
    possible duplicate — never a silent merge."""
    if place is not None:
        found = registry.merge_candidates(
            db, lat=place["lat"], lon=place["lon"], geo_precision=place["precision"],
            country=country, start=start, end=end, time_precision=time_precision,
            report_type=report_type)
        if found and (found[0]["confidence"] == "high" or not allow_create):
            registry.link(db, found[0]["event_id"], origin=origin, linked_by=SYSTEM,
                          publication_id=publication_id, facts=facts)
            return found[0]["event_id"], False
        duplicate = found[0]["event_id"] if found else None
    else:
        duplicate = None
    if not allow_create:
        return None, False
    event_id = registry.create_event(
        db, SYSTEM, title=title, report_type=report_type,
        place_label=_place_label(place["label"]) if place else (
            facts.get("toponym") if facts.get("toponym") else None),
        country=country, lat=place["lat"] if place else None,
        lon=place["lon"] if place else None,
        bbox_json=json.dumps(place["bbox"]) if place and place.get("bbox")
        and place["precision"] in ("district", "region", "country") else None,
        geo_precision=place["precision"] if place else "unknown",
        geo_status=geo_status,
        geo_candidates_json=json.dumps(candidates, ensure_ascii=False) if candidates else None,
        event_time_start=start, event_time_end=end, time_precision=time_precision,
        possible_duplicate_of=duplicate)
    registry.link(db, event_id, origin=origin, linked_by=SYSTEM,
                  publication_id=publication_id, facts=facts)
    return event_id, True


def _bulletin_extraction(item: Item) -> tuple[extract.Extraction, dict]:
    data = item.structured or {}
    # GDACS flood alerts built on the GloFAS model are alerts, not observed flooding.
    kind = "warning" if str(data.get("basis", "")).upper() == "GLOFAS" else "unclear"
    result = extract.Extraction(
        relevance="relevant", report_type=kind, confidence="medium", toponyms=[],
        signals=[f"сводка GDACS, основание {data.get('basis') or 'не указано'}",
                 f"уровень {data.get('alertlevel') or '—'}"],
        status="ok" if kind == "warning" else "needs_review",
        method="структурированные поля сводки GDACS")
    return result, data


def process_item(db: sqlite3.Connection, source: sqlite3.Row, item: Item,
                 geo_budget: geo.Budget) -> dict:
    stats = {"new": 0, "relevant": 0, "events_new": 0}
    url = canonical_url(item.url) if item.url else ""
    digest = content_hash(item.title, item.excerpt)
    existing = db.execute("SELECT * FROM publication WHERE source_id = ? AND external_id = ?",
                          (source["source_id"], item.external_id)).fetchone()
    if existing:
        if existing["content_hash"] != digest:  # the same publication, updated at the source
            db.execute("UPDATE publication SET title = ?, text_excerpt = ?, content_hash = ?, "
                       "updated_at = ? WHERE publication_id = ?",
                       (item.title, item.excerpt, digest, now(), existing["publication_id"]))
            for row in db.execute("SELECT event_id FROM evidence WHERE publication_id = ?",
                                  (existing["publication_id"],)):
                record_history(db, row["event_id"], SYSTEM, "source_updated",
                               before={"title": existing["title"]}, after={"title": item.title},
                               note=f"публикация обновлена источником {source['name']}")
        return stats
    if url and db.execute("SELECT 1 FROM publication WHERE url = ?", (url,)).fetchone():
        return stats  # the same article already came from another source
    if source["kind"] == "bulletin":
        extraction, structured = _bulletin_extraction(item)
    else:
        extraction, structured = extract.classify(item.title, item.excerpt), item.structured or {}
    cursor = db.execute(
        "INSERT INTO publication(source_id, external_id, url, domain, title, text_excerpt, "
        "content_level, content_hash, language, published_at, fetched_at, updated_at, relevance, "
        "report_type, extraction_json, extraction_method, extraction_confidence, "
        "extraction_status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (source["source_id"], item.external_id, url, (url.split("/")[2] if "//" in url else None),
         item.title, item.excerpt, item.content_level, digest, item.language, item.published_at,
         now(), now(), extraction.relevance, extraction.report_type,
         json.dumps({**extraction.model_dump(), "structured": structured}, ensure_ascii=False),
         extraction.method, extraction.confidence, extraction.status))
    publication_id = int(cursor.lastrowid)
    stats["new"] = 1
    if extraction.relevance != "relevant" or extraction.report_type == "historical":
        return stats
    stats["relevant"] = 1
    kind = extraction.report_type or "unclear"
    origin = "bulletin" if source["kind"] == "bulletin" else "media"
    if origin == "bulletin":
        data = structured
        place = None
        if data.get("lat") is not None and data.get("lon") is not None:
            place = {"lat": float(data["lat"]), "lon": float(data["lon"]), "precision": "region",
                     "label": f"{data.get('country') or 'территория'} (центр области GDACS)",
                     "bbox": None}
        start = (data.get("fromdate") or "")[:19] + "Z" if data.get("fromdate") else None
        end = (data.get("todate") or "")[:19] + "Z" if data.get("todate") else None
        _, created = _link_or_create(
            db, publication_id=publication_id, origin=origin,
            title=f"Сводка GDACS: паводок, {data.get('country') or 'страна не указана'}",
            report_type=kind, place=place, geo_status="resolved" if place else "missing",
            candidates=[], country=data.get("country"), start=start, end=end,
            time_precision="several_days" if start else "unknown",
            facts={"alertlevel": data.get("alertlevel"), "basis": data.get("basis"),
                   "gdacs_eventid": data.get("eventid"), "point_kind": data.get("point_kind")})
        stats["events_new"] += created
        return stats
    start, end, precision, time_basis = _time_from_text(f"{item.title} {item.excerpt or ''}",
                                                        item.published_at)
    resolved, unresolved = [], []
    for name in extraction.toponyms[:3]:
        outcome = geo.resolve(db, name, extract.toponym_variants(name), geo_budget)
        (resolved if outcome["status"] == "resolved" else unresolved).append((name, outcome))
    for name in extraction.weak_toponyms:
        outcome = geo.resolve(db, name, extract.toponym_variants(name), geo_budget,
                              min_importance=0.45)
        if outcome["status"] == "resolved":
            resolved.append((name, outcome))
    precise = geo.most_precise([outcome["chosen"] for _, outcome in resolved])
    resolved = [(name, outcome) for name, outcome in resolved
                if any(outcome["chosen"] is place for place in precise)]
    facts = {"signals": extraction.signals, "time_basis": time_basis}
    allow_create = kind != "negation"  # a denial attaches to an event, it never creates one
    label = {"occurred": "Сообщение СМИ о подтоплении", "forecast": "Прогноз паводка",
             "warning": "Предупреждение о паводке", "negation": "Опровержение",
             "unclear": "Неясное сообщение о паводке"}[kind]
    for name, outcome in resolved:
        chosen = outcome["chosen"]
        _, created = _link_or_create(
            db, publication_id=publication_id, origin=origin,
            title=f"{label}: {_place_label(chosen['label'])}", report_type=kind, place=chosen,
            geo_status="resolved", candidates=[], country=None, start=start, end=end,
            time_precision=precision, facts={**facts, "toponym": name},
            allow_create=allow_create)
        stats["events_new"] += created
    if not resolved and allow_create:
        # Kept in the review list even though it cannot be put on the map.
        name, outcome = unresolved[0] if unresolved else (None, {"status": "missing",
                                                                 "candidates": []})
        status = "ambiguous" if outcome["status"] == "ambiguous" else "missing"
        _, created = _link_or_create(
            db, publication_id=publication_id, origin=origin,
            title=f"{label}: {name or 'место не указано'}", report_type=kind, place=None,
            geo_status=status, candidates=outcome.get("candidates", []), country=None,
            start=start, end=end, time_precision=precision,
            facts={**facts, "toponym": name, "geo_outcome": outcome["status"]})
        stats["events_new"] += created
    if not resolved or unresolved:
        db.execute("UPDATE publication SET extraction_status = 'needs_review' "
                   "WHERE publication_id = ?", (publication_id,))
    return stats


def _due(source: sqlite3.Row, moment: datetime) -> bool:
    last = source["last_attempt_at"]
    return last is None or moment - parse_iso(last) >= timedelta(seconds=source["poll_interval_s"])


def run_cycle(trigger: str = "cli", *, adapters: dict[str, Callable] | None = None,
              force: bool = False, stop: Callable[[], bool] = lambda: False) -> dict:
    """Fetch every enabled, due source once. Returns the ingestion_run summary."""
    adapters = adapters or ADAPTERS
    lock_file = open(media_dir() / "ingest.lock", "w")  # noqa: SIM115 (held for the cycle)
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock_file.close()
        return {"status": "skipped", "reason": "цикл сбора уже выполняется"}
    try:
        return _cycle(trigger, adapters, force or trigger != "schedule", stop)
    finally:
        fcntl.flock(lock_file, fcntl.LOCK_UN)
        lock_file.close()


def _cycle(trigger: str, adapters: dict, force: bool, stop: Callable[[], bool]) -> dict:
    config = load_config()
    settings = config["monitoring"]
    budget = geo.Budget(int(settings["requests_budget"]))
    geo_budget = geo.Budget(min(int(settings["geocode_budget"]), budget.calls))
    started = time.monotonic()
    with connect() as db:
        sync_sources(db, config)
        run_id = int(db.execute(
            "INSERT INTO ingestion_run(trigger, started_at, status, requests_budget) "
            "VALUES (?, ?, 'running', ?)", (trigger, now(), budget.calls)).lastrowid)
        sources = db.execute("SELECT * FROM source WHERE enabled = 1 ORDER BY source_id").fetchall()
    errors, totals = [], {"new": 0, "events_new": 0}
    moment = datetime.now(UTC)
    for source in sources:
        if stop():
            errors.append({"source_id": None, "error": "остановлено по сигналу"})
            break
        if not force and not _due(source, moment):
            continue
        with connect() as db:
            source_run = int(db.execute(
                "INSERT INTO source_run(run_id, source_id, started_at, status) "
                "VALUES (?, ?, ?, 'running')", (run_id, source["source_id"], now())).lastrowid)
            db.execute("UPDATE source SET last_attempt_at = ? WHERE source_id = ?",
                       (now(), source["source_id"]))
        config_item = json.loads(source["config_json"])
        adapter = adapters.get(config_item.get("adapter"))
        stats = {"seen": 0, "new": 0, "relevant": 0, "events_new": 0}
        status, http_status, error, batch = "ok", None, None, Batch()
        calls_before = budget.calls
        try:
            if adapter is None:
                raise FetchError(f"нет адаптера {config_item.get('adapter')!r}")
            batch = adapter(config_item, source["cursor"], budget,
                            int(settings["max_items_per_source"]))
            http_status = batch.http_status
        except BudgetExhausted:
            status, error = "skipped_budget", "исчерпан бюджет запросов цикла"
        except FetchError as exc:
            status, http_status, error = ("skipped_rate_limit" if exc.status == 429 else "failed",
                                          exc.status, str(exc))
        for item in batch.items:
            if stop():
                status, error = "partial", "остановлено по сигналу"
                break
            stats["seen"] += 1
            try:
                with connect() as db:
                    result = process_item(db, source, item, geo_budget)
            except (FetchError, ValueError, KeyError, sqlite3.IntegrityError) as exc:
                status, error = "partial", f"{item.external_id}: {exc}"
                continue
            for key in ("new", "relevant", "events_new"):
                stats[key] += result[key]
        with connect() as db:
            stamp = now()
            db.execute(
                "UPDATE source_run SET finished_at = ?, status = ?, http_status = ?, "
                "items_seen = ?, items_new = ?, items_relevant = ?, requests_used = ?, error = ? "
                "WHERE source_run_id = ?",
                (stamp, status, http_status, stats["seen"], stats["new"], stats["relevant"],
                 calls_before - budget.calls, error, source_run))
            if status in ("ok", "partial"):
                db.execute("UPDATE source SET last_success_at = ?, cursor = COALESCE(?, cursor), "
                           "last_new_item_at = CASE WHEN ? > 0 THEN ? ELSE last_new_item_at END "
                           "WHERE source_id = ?", (stamp, batch.cursor, stats["new"], stamp,
                                                   source["source_id"]))
            if error:
                db.execute("UPDATE source SET last_error_at = ?, last_error = ? "
                           "WHERE source_id = ?", (stamp, error[:500], source["source_id"]))
                errors.append({"source_id": source["source_id"], "error": error[:500]})
        totals["new"] += stats["new"]
        totals["events_new"] += stats["events_new"]
    failed = [item for item in errors if item["source_id"]]
    status = "ok" if not errors else ("failed" if len(failed) == len(sources) and sources
                                      else "partial")
    with connect() as db:
        db.execute("UPDATE ingestion_run SET finished_at = ?, status = ?, items_new = ?, "
                   "events_new = ?, errors_json = ?, duration_s = ?, requests_used = ? "
                   "WHERE run_id = ?",
                   (now(), status, totals["new"], totals["events_new"],
                    json.dumps(errors, ensure_ascii=False) if errors else None,
                    round(time.monotonic() - started, 2),
                    int(settings["requests_budget"]) - budget.calls, run_id))
        set_setting(db, "last_cycle_at", now())
        row = dict(db.execute("SELECT * FROM ingestion_run WHERE run_id = ?", (run_id,)).fetchone())
    return row


def monitoring_settings(db: sqlite3.Connection) -> dict:
    defaults = load_config()["monitoring"]
    return {"enabled": get_setting(db, "monitoring_enabled",
                                   "1" if defaults["enabled"] else "0") == "1",
            "interval_s": int(get_setting(db, "interval_s", str(defaults["interval_s"]))),
            "run_requested": get_setting(db, "run_requested_at") is not None,
            "last_cycle_at": get_setting(db, "last_cycle_at")}

