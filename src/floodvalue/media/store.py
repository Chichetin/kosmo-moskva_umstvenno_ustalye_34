"""SQLite register outside git. A source message (publication, eyewitness report) is kept apart
from the event it tells about; evidence links them. Three times never share a column: event
time (events.event_time_*), publication time (publications.published_at), system time
(fetched_at, submitted_at, first_detected_at)."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS sources (
  source_id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  kind TEXT NOT NULL CHECK (kind IN ('news', 'bulletin', 'citizen')),
  adapter TEXT NOT NULL,
  url TEXT,
  enabled INTEGER NOT NULL DEFAULT 0,
  terms_status TEXT NOT NULL,
  terms_url TEXT,
  attribution TEXT,
  content_storage_policy TEXT NOT NULL
    CHECK (content_storage_policy IN ('headline_only', 'summary', 'excerpt')),
  poll_interval_s INTEGER NOT NULL,
  config_json TEXT NOT NULL DEFAULT '{}',
  cursor TEXT,
  last_attempt_at TEXT,
  last_success_at TEXT,
  last_new_data_at TEXT,
  last_error_at TEXT,
  last_error TEXT
);
CREATE TABLE IF NOT EXISTS ingestion_runs (
  run_id INTEGER PRIMARY KEY AUTOINCREMENT,
  source_id TEXT NOT NULL REFERENCES sources(source_id),
  started_at TEXT NOT NULL,
  finished_at TEXT,
  status TEXT NOT NULL CHECK (status IN ('running', 'ok', 'failed', 'skipped_rate_limit',
                                         'skipped_budget', 'interrupted')),
  http_status INTEGER,
  items_seen INTEGER NOT NULL DEFAULT 0,
  items_new INTEGER NOT NULL DEFAULT 0,
  items_updated INTEGER NOT NULL DEFAULT 0,
  items_relevant INTEGER NOT NULL DEFAULT 0,
  requests_used INTEGER NOT NULL DEFAULT 0,
  llm_calls INTEGER NOT NULL DEFAULT 0,
  duration_s REAL,
  error TEXT
);
CREATE TABLE IF NOT EXISTS publications (
  publication_id INTEGER PRIMARY KEY AUTOINCREMENT,
  source_id TEXT NOT NULL REFERENCES sources(source_id),
  external_id TEXT NOT NULL,
  url TEXT NOT NULL,
  canonical_url TEXT NOT NULL,
  title TEXT NOT NULL,
  text_excerpt TEXT,
  content_scope TEXT NOT NULL,
  language TEXT,
  source_domain TEXT,
  published_at TEXT,
  source_updated_at TEXT,
  fetched_at TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  title_key TEXT NOT NULL,
  place_hint_json TEXT,
  time_hint_json TEXT,
  report_type TEXT,
  extraction_method TEXT,
  extraction_status TEXT NOT NULL DEFAULT 'pending'
    CHECK (extraction_status IN ('pending', 'ok', 'needs_review', 'invalid', 'irrelevant')),
  extraction_json TEXT,
  processing_error TEXT,
  is_demo INTEGER NOT NULL DEFAULT 0,
  UNIQUE (source_id, external_id)
);
CREATE INDEX IF NOT EXISTS pub_canonical ON publications(canonical_url);
CREATE INDEX IF NOT EXISTS pub_title_key ON publications(title_key);
CREATE TABLE IF NOT EXISTS citizen_reports (
  report_id TEXT PRIMARY KEY,
  status_token_hash TEXT NOT NULL UNIQUE,
  idempotency_key TEXT NOT NULL UNIQUE,
  category TEXT NOT NULL,
  description TEXT NOT NULL,
  lat REAL,
  lon REAL,
  place_text TEXT,
  location_precision TEXT NOT NULL,
  observed_at TEXT,
  observed_tz TEXT,
  observed_precision TEXT NOT NULL,
  submitted_at TEXT NOT NULL,
  moderation_status TEXT NOT NULL DEFAULT 'pending'
    CHECK (moderation_status IN ('pending', 'accepted', 'rejected', 'spam')),
  public_note TEXT,
  event_id INTEGER REFERENCES events(event_id),
  is_demo INTEGER NOT NULL DEFAULT 0,
  delete_after TEXT NOT NULL,
  CHECK ((lat IS NULL) = (lon IS NULL)),
  CHECK (lat IS NULL OR (lat BETWEEN -90 AND 90 AND lon BETWEEN -180 AND 180))
);
CREATE INDEX IF NOT EXISTS reports_moderation ON citizen_reports(moderation_status);
CREATE TABLE IF NOT EXISTS attachments (
  attachment_id TEXT PRIMARY KEY,
  report_id TEXT NOT NULL REFERENCES citizen_reports(report_id) ON DELETE CASCADE,
  file_name TEXT NOT NULL UNIQUE,
  mime TEXT NOT NULL,
  width INTEGER NOT NULL,
  height INTEGER NOT NULL,
  bytes INTEGER NOT NULL,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
  event_id INTEGER PRIMARY KEY AUTOINCREMENT,
  title TEXT NOT NULL,
  event_type TEXT NOT NULL DEFAULT 'flood',
  report_type TEXT NOT NULL,
  place_label TEXT,
  country TEXT,
  geometry_json TEXT,
  geo_precision TEXT NOT NULL
    CHECK (geo_precision IN ('point', 'street', 'locality', 'district', 'region', 'country',
                             'unknown')),
  geo_status TEXT NOT NULL CHECK (geo_status IN ('resolved', 'ambiguous', 'missing')),
  geo_candidates_json TEXT,
  event_time_start TEXT,
  event_time_end TEXT,
  time_precision TEXT NOT NULL
    CHECK (time_precision IN ('hour', 'day', 'several_days', 'unknown')),
  first_detected_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  review_state TEXT NOT NULL
    CHECK (review_state IN ('new', 'needs_geo', 'ready', 'under_review', 'reviewed',
                            'merged', 'dismissed')),
  verification_result TEXT NOT NULL DEFAULT 'not_checked'
    CHECK (verification_result IN ('not_checked', 'consistent', 'discrepancy',
                                   'insufficient_data')),
  follow_up TEXT NOT NULL DEFAULT 'none'
    CHECK (follow_up IN ('none', 'needs_imagery', 'merge', 'split', 'dismiss')),
  merged_into INTEGER REFERENCES events(event_id),
  independent_origin_count INTEGER NOT NULL DEFAULT 0,
  source_kinds TEXT NOT NULL DEFAULT '',
  external_key TEXT UNIQUE,
  manual_lock INTEGER NOT NULL DEFAULT 0,
  is_demo INTEGER NOT NULL DEFAULT 0,
  is_replay INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS events_state ON events(review_state);
CREATE TABLE IF NOT EXISTS evidence (
  evidence_id INTEGER PRIMARY KEY AUTOINCREMENT,
  event_id INTEGER NOT NULL REFERENCES events(event_id),
  publication_id INTEGER REFERENCES publications(publication_id),
  report_id TEXT REFERENCES citizen_reports(report_id),
  origin TEXT NOT NULL CHECK (origin IN ('news', 'bulletin', 'citizen')),
  is_original INTEGER NOT NULL,
  link_kind TEXT NOT NULL
    CHECK (link_kind IN ('auto', 'manual_merge', 'manual_split', 'moderation', 'manual_link')),
  linked_at TEXT NOT NULL,
  linked_by TEXT NOT NULL,
  report_type TEXT,
  facts_json TEXT,
  CHECK ((publication_id IS NULL) != (report_id IS NULL)),
  UNIQUE (event_id, publication_id),
  UNIQUE (event_id, report_id)
);
CREATE TABLE IF NOT EXISTS event_history (
  history_id INTEGER PRIMARY KEY AUTOINCREMENT,
  event_id INTEGER NOT NULL REFERENCES events(event_id),
  at TEXT NOT NULL,
  actor TEXT NOT NULL,
  action TEXT NOT NULL,
  before_json TEXT,
  after_json TEXT
);
CREATE TABLE IF NOT EXISTS reviews (
  review_id INTEGER PRIMARY KEY AUTOINCREMENT,
  target_kind TEXT NOT NULL CHECK (target_kind IN ('event', 'report')),
  target_id TEXT NOT NULL,
  reviewer TEXT NOT NULL,
  reviewed_at TEXT NOT NULL,
  decision TEXT NOT NULL,
  verification_result TEXT,
  basis TEXT,
  comment TEXT,
  follow_up TEXT,
  state_before TEXT,
  state_after TEXT
);
CREATE TABLE IF NOT EXISTS observation_matches (
  match_id INTEGER PRIMARY KEY AUTOINCREMENT,
  event_id INTEGER NOT NULL REFERENCES events(event_id),
  catalog TEXT NOT NULL,
  observation_ref TEXT NOT NULL,
  product_type TEXT,
  observation_start TEXT,
  footprint_json TEXT,
  spatial_relation TEXT NOT NULL
    CHECK (spatial_relation IN ('covers', 'intersects', 'nearby', 'none', 'unknown')),
  time_offset_h REAL,
  suitability TEXT NOT NULL
    CHECK (suitability IN ('suitable', 'too_early', 'too_late', 'partial_cover', 'unusable',
                           'unknown_time')),
  reason TEXT NOT NULL,
  stage TEXT NOT NULL CHECK (stage IN ('found', 'acquired', 'prepared', 'analyzed')),
  analysis_run_id TEXT,
  found_at TEXT NOT NULL,
  UNIQUE (event_id, catalog, observation_ref)
);
CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS geocode_cache (
  query TEXT PRIMARY KEY, result_json TEXT NOT NULL, fetched_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS rate_hits (bucket TEXT NOT NULL, at REAL NOT NULL);
CREATE INDEX IF NOT EXISTS rate_hits_bucket ON rate_hits(bucket, at);
"""


def now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA)
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection):
    """All-or-nothing for multi-step actions (merge, split, moderation, one ingested item)."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")


def dumps(value) -> str | None:
    return None if value is None else json.dumps(value, ensure_ascii=False, sort_keys=True)


def loads(text: str | None):
    return None if text is None else json.loads(text)


def get_kv(conn: sqlite3.Connection, key: str, default: str | None = None) -> str | None:
    row = conn.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else default


def set_kv(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute("INSERT INTO kv(key, value) VALUES(?, ?) "
                 "ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, value))


def sync_sources(conn: sqlite3.Connection, sources: list[dict]) -> None:
    """Config is the source of truth for connection settings; runtime state (cursor, last
    success, errors) and an operator's enable/disable toggle survive a re-sync."""
    for s in sources:
        conn.execute(
            """INSERT INTO sources(source_id, name, kind, adapter, url, enabled, terms_status,
                 terms_url, attribution, content_storage_policy, poll_interval_s, config_json)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(source_id) DO UPDATE SET name=excluded.name, kind=excluded.kind,
                 adapter=excluded.adapter, url=excluded.url, terms_status=excluded.terms_status,
                 terms_url=excluded.terms_url, attribution=excluded.attribution,
                 content_storage_policy=excluded.content_storage_policy,
                 poll_interval_s=excluded.poll_interval_s, config_json=excluded.config_json""",
            (s["source_id"], s["name"], s["kind"], s["adapter"], s.get("url"),
             int(bool(s.get("enabled")) and s.get("terms_status") == "checked"),
             s["terms_status"], s.get("terms_url"), s.get("attribution"),
             s["content_storage_policy"], int(s.get("poll_interval_s", 900)),
             json.dumps(s.get("config", {}), ensure_ascii=False)))


def add_history(conn: sqlite3.Connection, event_id: int, actor: str, action: str,
                before=None, after=None) -> None:
    conn.execute("INSERT INTO event_history(event_id, at, actor, action, before_json, after_json)"
                 " VALUES(?,?,?,?,?,?)",
                 (event_id, now_iso(), actor, action, dumps(before), dumps(after)))
