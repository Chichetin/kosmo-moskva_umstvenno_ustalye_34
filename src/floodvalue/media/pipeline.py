"""One ingestion cycle per source: fetch → store publications without duplicates → extract →
resolve place → link to events. An API error is a failed run, never "no new events"; the
cursor moves only after a complete successful pass, so a gap is refetched, not skipped."""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Callable

from floodvalue.media.config import Settings
from floodvalue.media.events import create_event, find_negation_target, find_similar, link
from floodvalue.media.extract import Extraction, LLMExtractor, extract_rules, validate_llm_answer
from floodvalue.media.geo import Geocoder, NominatimGeocoder, resolve
from floodvalue.media.netfetch import Budget, BudgetExceeded, Fetcher, FetchError
from floodvalue.media.sources import ADAPTERS, canonical_url, content_hash, title_key
from floodvalue.media.store import add_history, dumps, now_iso, transaction

MAX_PLACES_PER_ITEM = 3
MIN_REPRINT_WORDS = 5


def make_fetcher(source: sqlite3.Row, settings: Settings, budget: Budget) -> Fetcher:
    cfg = json.loads(source["config_json"])
    hosts = frozenset(cfg.get("allowed_hosts", []))
    return Fetcher(allowed_hosts=hosts, budget=budget, user_agent=settings.user_agent,
                   min_interval_s={h: float(cfg.get("min_interval_s", 1.0)) for h in hosts})


def make_geocoder(conn, settings: Settings, budget: Budget) -> Geocoder | None:
    if settings.geocoder != "nominatim":
        return None
    host = "nominatim.openstreetmap.org"
    fetcher = Fetcher(allowed_hosts=frozenset({host}), budget=budget,
                      user_agent=settings.user_agent, min_interval_s={host: 1.1})
    return NominatimGeocoder(conn, fetcher)


def recover_interrupted(conn) -> int:
    return conn.execute("UPDATE ingestion_runs SET status='interrupted', finished_at=? "
                        "WHERE status='running'", (now_iso(),)).rowcount


def run_source(conn, settings: Settings, source_id: str, *, budget: Budget | None = None,
               fetcher: Fetcher | None = None, geocoder: Geocoder | None = None,
               llm: LLMExtractor | None = None,
               should_stop: Callable[[], bool] = lambda: False) -> dict:
    source = conn.execute("SELECT * FROM sources WHERE source_id=?", (source_id,)).fetchone()
    if source is None:
        raise KeyError(source_id)
    started = time.monotonic()
    run_id = conn.execute("INSERT INTO ingestion_runs(source_id, started_at, status) "
                          "VALUES(?,?, 'running')", (source_id, now_iso())).lastrowid
    budget = budget or Budget(settings.max_requests_per_cycle)
    fetcher = fetcher or make_fetcher(source, settings, budget)
    stats = {"items_seen": 0, "items_new": 0, "items_updated": 0, "items_relevant": 0,
             "llm_calls": 0}
    status, error, cursor = "ok", None, source["cursor"]
    conn.execute("UPDATE sources SET last_attempt_at=? WHERE source_id=?", (now_iso(), source_id))
    try:
        items, new_cursor = ADAPTERS[source["adapter"]](
            fetcher, json.loads(source["config_json"]), source["cursor"])
        stats["items_seen"] = len(items)
        complete = len(items) <= settings.max_items_per_cycle
        for item in items[: settings.max_items_per_cycle]:
            if should_stop():
                status, error, complete = "interrupted", "остановлено оператором", False
                break
            with transaction(conn):
                outcome = ingest_item(conn, source, item, geocoder=geocoder, llm=llm,
                                      llm_left=settings.llm_max_calls_per_cycle - stats["llm_calls"],
                                      counters=stats)
            stats["items_new"] += outcome == "new"
            stats["items_updated"] += outcome == "updated"
        if not complete and status == "ok":
            error = (f"обработано {settings.max_items_per_cycle} из {len(items)} за цикл; "
                     "окно не сдвинуто, остаток — в следующем цикле")
        if complete:
            cursor = new_cursor
    except BudgetExceeded as exc:
        status, error = "skipped_budget", str(exc)
    except FetchError as exc:
        status = "skipped_rate_limit" if exc.status == 429 else "failed"
        error = str(exc)
    except Exception as exc:  # noqa: BLE001 — a broken adapter must not stop other sources
        status, error = "failed", f"{type(exc).__name__}: {exc}"
    now = now_iso()
    conn.execute(
        """UPDATE ingestion_runs SET finished_at=?, status=?, http_status=?, items_seen=?,
             items_new=?, items_updated=?, items_relevant=?, requests_used=?, llm_calls=?,
             duration_s=?, error=? WHERE run_id=?""",
        (now, status, fetcher.last_status, stats["items_seen"], stats["items_new"],
         stats["items_updated"], stats["items_relevant"], budget.used, stats["llm_calls"],
         round(time.monotonic() - started, 3), error, run_id))
    if status == "ok":
        conn.execute("UPDATE sources SET last_success_at=?, cursor=?, last_new_data_at="
                     "CASE WHEN ? > 0 THEN ? ELSE last_new_data_at END WHERE source_id=?",
                     (now, cursor, stats["items_new"] + stats["items_updated"], now, source_id))
    else:
        conn.execute("UPDATE sources SET last_error_at=?, last_error=? WHERE source_id=?",
                     (now, error, source_id))
    return {"run_id": run_id, "source_id": source_id, "status": status, "error": error,
            "requests_used": budget.used, **stats}


def ingest_item(conn, source, item: dict, *, geocoder=None, llm=None, llm_left: int = 0,
                is_demo: bool = False, counters: dict | None = None) -> str:
    """Returns new / updated / unchanged / irrelevant; counts relevant items and LLM calls."""
    counters = counters if counters is not None else {}
    canon = canonical_url(item["url"]) if item.get("url") else f"urn:{item['external_id']}"
    excerpt = item.get("text_excerpt") if source["content_storage_policy"] != "headline_only" \
        else None
    scope = "headline_only" if not excerpt else source["content_storage_policy"]
    chash = content_hash(item["title"], excerpt, item.get("source_updated_at"))
    existing = conn.execute("SELECT * FROM publications WHERE source_id=? AND external_id=?",
                            (source["source_id"], item["external_id"])).fetchone()
    if existing and existing["content_hash"] == chash:
        return "unchanged"
    fields = (item["url"] or canon, canon, item["title"][:500], excerpt, scope,
              item.get("language"), item.get("source_domain"), item.get("published_at"),
              item.get("source_updated_at"), now_iso(), chash, title_key(item["title"]),
              dumps({"time_basis": item.get("time_basis")}))
    if existing:
        conn.execute("""UPDATE publications SET url=?, canonical_url=?, title=?, text_excerpt=?,
                          content_scope=?, language=?, source_domain=?, published_at=?,
                          source_updated_at=?, content_hash=?, title_key=?,
                          time_hint_json=? WHERE publication_id=?""",
                     (*fields[:9], *fields[10:], existing["publication_id"]))
        pub_id = existing["publication_id"]
        for row in conn.execute("SELECT event_id FROM evidence WHERE publication_id=?", (pub_id,)):
            add_history(conn, row["event_id"], "system", "publication_updated",
                        {"content_hash": existing["content_hash"]}, {"content_hash": chash})
        outcome = "updated"
    else:
        pub_id = conn.execute(
            """INSERT INTO publications(url, canonical_url, title, text_excerpt, content_scope,
                 language, source_domain, published_at, source_updated_at, fetched_at,
                 content_hash, title_key, time_hint_json, source_id, external_id, is_demo)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (*fields, source["source_id"], item["external_id"], int(is_demo))).lastrowid
        outcome = "new"
    structured = item.get("structured")
    if structured:
        _link_bulletin(conn, source, pub_id, structured, is_demo)
        counters["items_relevant"] = counters.get("items_relevant", 0) + 1
        conn.execute("UPDATE publications SET report_type='occurred', extraction_method="
                     "'source_structured', extraction_status='ok', extraction_json=?, "
                     "place_hint_json=? WHERE publication_id=?",
                     (dumps(structured), dumps(structured["place"]), pub_id))
        return outcome
    if existing:
        return outcome  # same article re-published: events stay, history notes the change
    extraction, used_llm = _extract(item, excerpt, llm, llm_left)
    counters["llm_calls"] = counters.get("llm_calls", 0) + used_llm
    status = ("irrelevant" if not extraction.relevant else
              "needs_review" if extraction.needs_review else "ok")
    conn.execute("UPDATE publications SET report_type=?, extraction_method=?, extraction_status=?,"
                 " extraction_json=? WHERE publication_id=?",
                 (extraction.report_type if extraction.relevant else None, extraction.method,
                  status, extraction.model_dump_json(), pub_id))
    if not extraction.relevant:
        return "irrelevant"
    counters["items_relevant"] = counters.get("items_relevant", 0) + 1
    if _link_reprint(conn, source, pub_id, canon, item["title"], extraction):
        return outcome
    _link_news(conn, source, pub_id, item, extraction, geocoder, is_demo)
    return outcome


def _extract(item, excerpt, llm, llm_left) -> tuple[Extraction, bool]:
    rules = extract_rules(item["title"], excerpt)
    if llm is None or llm_left <= 0 or not rules.relevant:
        return rules, False
    try:
        answer = validate_llm_answer(llm.extract_json(item["title"], excerpt), llm.name)
    except Exception:  # noqa: BLE001 — any provider failure means "to manual review"
        answer = None
    if answer is None:
        return rules.model_copy(update={"needs_review": True, "reasons": [
            *rules.reasons, "ответ модели не прошёл схему — на ручную проверку"]}), True
    return answer, True


def _link_bulletin(conn, source, pub_id, s: dict, is_demo: bool) -> None:
    place = s["place"]
    resolved = {"label": place.get("label"), "geometry": place.get("geometry"),
                "precision": place.get("precision", "unknown"),
                "status": "resolved" if place.get("geometry") else "missing", "candidates": None}
    row = conn.execute("SELECT * FROM events WHERE external_key=?", (s["event_key"],)).fetchone()
    if row is None:
        event_id = create_event(conn, title=f"Сводка GDACS: наводнение, {place.get('label') or '?'}",
                                report_type=s["report_type"], place=resolved, time=s["time"],
                                actor=f"system:{source['source_id']}", is_demo=is_demo,
                                external_key=s["event_key"], country=place.get("country"))
    else:
        event_id = row["event_id"]
        new_time = (s["time"]["start"], s["time"]["end"])
        if not row["manual_lock"] and new_time != (row["event_time_start"], row["event_time_end"]):
            conn.execute("UPDATE events SET event_time_start=?, event_time_end=?, updated_at=? "
                         "WHERE event_id=?", (*new_time, now_iso(), event_id))
            add_history(conn, event_id, f"system:{source['source_id']}", "bulletin_time_update",
                        {"start": row["event_time_start"], "end": row["event_time_end"]},
                        {"start": new_time[0], "end": new_time[1]})
    link(conn, event_id, origin="bulletin", is_original=True, link_kind="auto",
         actor=f"system:{source['source_id']}", publication_id=pub_id, report_type="occurred",
         facts={k: s.get(k) for k in ("alert_level", "provider", "glide")})


def _link_reprint(conn, source, pub_id, canon, title, extraction) -> bool:
    """Same URL from another source, or the same headline: link to the original's events as
    a reprint (is_original=0) — it is not an independent confirmation."""
    key = title_key(title)
    params = [canon, pub_id]
    sql = ("SELECT DISTINCT e.event_id FROM publications p JOIN evidence e USING(publication_id) "
           "WHERE (p.canonical_url = ?")
    if len(key.split()) >= MIN_REPRINT_WORDS:
        sql += " OR p.title_key = ?"
        params.insert(1, key)
    sql += ") AND p.publication_id != ?"
    events = [r["event_id"] for r in conn.execute(sql, params)]
    for event_id in events:
        link(conn, event_id, origin=source["kind"], is_original=False, link_kind="auto",
             actor=f"system:{source['source_id']}", publication_id=pub_id,
             report_type=extraction.report_type, facts={"reprint": True})
    return bool(events)


def _link_news(conn, source, pub_id, item, extraction: Extraction, geocoder, is_demo) -> None:
    mentions = extraction.place_mentions[:MAX_PLACES_PER_ITEM] or [None]
    time = extraction.event_time.model_dump()
    time = time if time.get("start") else None
    for mention in mentions:
        place = resolve([mention] if mention else [], geocoder)
        facts = {"place_mention": mention, "quotes": extraction.quotes,
                 "method": extraction.method, "reasons": extraction.reasons}
        actor = f"system:{source['source_id']}"
        if extraction.report_type == "negation":
            target = find_negation_target(conn, place["geometry"], mention)
        else:
            target = find_similar(conn, extraction.report_type, place["geometry"], time)
        if target is None:
            target = create_event(conn, title=item["title"], report_type=extraction.report_type,
                                  place=place, time=time, actor=actor, is_demo=is_demo)
        link(conn, target, origin=source["kind"], is_original=True, link_kind="auto",
             actor=actor, publication_id=pub_id, report_type=extraction.report_type, facts=facts)
