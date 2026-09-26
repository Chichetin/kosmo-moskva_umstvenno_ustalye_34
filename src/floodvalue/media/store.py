"""Persistent storage for media monitoring and citizen reports (SQLite from the stdlib).

A raw message (publication or citizen report) and the event it describes are separate
tables linked by evidence rows: one article may describe several events, one event may
collect many articles and reports. Event time, publication time and system receipt time
are separate columns, and so are processing, moderation and satellite-check states.
Nothing here reads or writes the flood-probability rasters, assets or the A/B/C plan.
"""

from __future__ import annotations

import json
import os
import secrets
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

from floodvalue.paths import REPO_ROOT

# Vocabulary follows docs/features/media-monitoring.md (sections Г, Е, З).
GEO_PRECISION = ("point", "street", "locality", "district", "region", "country", "unknown")
REPORT_TYPES = ("occurred", "forecast", "warning", "negation", "historical", "unclear")
REVIEW_STATES = ("new", "needs_geo", "ready", "under_review", "reviewed", "merged", "split",
                 "dismissed")
VERIFICATION_RESULTS = ("not_checked", "consistent", "discrepancy", "insufficient_data")

SCHEMA = """
CREATE TABLE IF NOT EXISTS source (
  source_id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  kind TEXT NOT NULL CHECK (kind IN ('rss', 'api', 'bulletin', 'citizen')),
  url TEXT,
  language TEXT,
  region_scope TEXT,
  poll_interval_s INTEGER NOT NULL DEFAULT 900,
  enabled INTEGER NOT NULL DEFAULT 0,
  terms_status TEXT NOT NULL DEFAULT 'unchecked' CHECK (terms_status IN ('checked', 'unchecked')),
  terms_note TEXT,
  attribution TEXT,
  content_storage_policy TEXT NOT NULL DEFAULT 'link_title'
    CHECK (content_storage_policy IN ('link_title', 'link_title_excerpt')),
  config_json TEXT NOT NULL DEFAULT '{}',
  last_attempt_at TEXT,
  last_success_at TEXT,
  last_new_item_at TEXT,
  last_error_at TEXT,
  last_error TEXT,
  cursor TEXT
);

CREATE TABLE IF NOT EXISTS publication (
  publication_id INTEGER PRIMARY KEY,
  source_id TEXT NOT NULL REFERENCES source(source_id),
  external_id TEXT NOT NULL,
  url TEXT NOT NULL,
  domain TEXT,
  title TEXT NOT NULL,
  text_excerpt TEXT,
  content_level TEXT NOT NULL CHECK (content_level IN ('title_only', 'excerpt', 'structured')),
  content_hash TEXT NOT NULL,
  original_source_hint TEXT,
  language TEXT,
  published_at TEXT,
  fetched_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  relevance TEXT NOT NULL CHECK (relevance IN ('relevant', 'irrelevant', 'unclear')),
  report_type TEXT CHECK (report_type IN ('occurred', 'forecast', 'warning', 'negation',
                                          'historical', 'unclear')),
  extraction_json TEXT,
  extraction_method TEXT NOT NULL,
  extraction_confidence TEXT CHECK (extraction_confidence IN ('high', 'medium', 'low')),
  extraction_status TEXT NOT NULL CHECK (extraction_status IN ('ok', 'invalid', 'needs_review')),
  is_demo INTEGER NOT NULL DEFAULT 0,
  UNIQUE (source_id, external_id)
);
CREATE INDEX IF NOT EXISTS publication_url ON publication(url);
CREATE INDEX IF NOT EXISTS publication_hash ON publication(content_hash);

CREATE TABLE IF NOT EXISTS event (
  event_id INTEGER PRIMARY KEY,
  title TEXT NOT NULL,
  report_type TEXT NOT NULL CHECK (report_type IN ('occurred', 'forecast', 'warning', 'negation',
                                                   'historical', 'unclear')),
  place_label TEXT,
  country TEXT,
  lat REAL,
  lon REAL,
  bbox_json TEXT,
  geo_precision TEXT NOT NULL CHECK (geo_precision IN ('point', 'street', 'locality', 'district',
                                                       'region', 'country', 'unknown')),
  geo_status TEXT NOT NULL CHECK (geo_status IN ('resolved', 'ambiguous', 'missing')),
  geo_candidates_json TEXT,
  event_time_start TEXT,
  event_time_end TEXT,
  time_precision TEXT NOT NULL CHECK (time_precision IN ('hour', 'day', 'several_days', 'unknown')),
  first_detected_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  review_state TEXT NOT NULL CHECK (review_state IN ('new', 'needs_geo', 'ready', 'under_review',
                                                     'reviewed', 'merged', 'split', 'dismissed')),
  verification_result TEXT NOT NULL DEFAULT 'not_checked' CHECK (verification_result IN
    ('not_checked', 'consistent', 'discrepancy', 'insufficient_data')),
  follow_up TEXT NOT NULL DEFAULT 'none' CHECK (follow_up IN
    ('none', 'needs_imagery', 'merge', 'split', 'dismiss')),
  merged_into INTEGER REFERENCES event(event_id),
  possible_duplicate_of INTEGER REFERENCES event(event_id),
  is_demo INTEGER NOT NULL DEFAULT 0,
  is_replay INTEGER NOT NULL DEFAULT 0,
  CHECK ((lat IS NULL) = (lon IS NULL)),
  CHECK (lat IS NULL OR (lat BETWEEN -90 AND 90 AND lon BETWEEN -180 AND 180)),
  CHECK (lat IS NOT NULL OR geo_status != 'resolved')
);
CREATE INDEX IF NOT EXISTS event_review ON event(review_state);

CREATE TABLE IF NOT EXISTS citizen_report (
  report_id INTEGER PRIMARY KEY,
  public_id TEXT NOT NULL UNIQUE,
  client_key TEXT NOT NULL UNIQUE,
  status_token_hash TEXT NOT NULL UNIQUE,
  kind TEXT NOT NULL,
  description TEXT NOT NULL,
  lat REAL,
  lon REAL,
  place_text TEXT,
  location_precision TEXT NOT NULL,
  observed_at TEXT NOT NULL,
  observed_time_precision TEXT NOT NULL,
  observed_tz TEXT NOT NULL,
  submitted_at TEXT NOT NULL,
  moderation_status TEXT NOT NULL DEFAULT 'pending' CHECK (moderation_status IN
    ('pending', 'approved', 'rejected')),
  is_demo INTEGER NOT NULL DEFAULT 0,
  retention_until TEXT NOT NULL,
  CHECK ((lat IS NULL) = (lon IS NULL))
);
CREATE INDEX IF NOT EXISTS report_moderation ON citizen_report(moderation_status);

CREATE TABLE IF NOT EXISTS attachment (
  attachment_id INTEGER PRIMARY KEY,
  report_id INTEGER NOT NULL REFERENCES citizen_report(report_id) ON DELETE CASCADE,
  stored_name TEXT NOT NULL UNIQUE,
  mime TEXT NOT NULL,
  width INTEGER NOT NULL,
  height INTEGER NOT NULL,
  bytes INTEGER NOT NULL,
  created_at TEXT NOT NULL
);

-- Evidence: a publication or a citizen report linked to an event (spec: event_publication).
CREATE TABLE IF NOT EXISTS evidence (
  evidence_id INTEGER PRIMARY KEY,
  event_id INTEGER NOT NULL REFERENCES event(event_id),
  publication_id INTEGER REFERENCES publication(publication_id),
  report_id INTEGER REFERENCES citizen_report(report_id) ON DELETE CASCADE,
  origin TEXT NOT NULL CHECK (origin IN ('media', 'bulletin', 'citizen')),
  link_kind TEXT NOT NULL CHECK (link_kind IN ('auto', 'manual', 'manual_merge', 'manual_split')),
  linked_at TEXT NOT NULL,
  linked_by TEXT NOT NULL,
  is_original INTEGER NOT NULL DEFAULT 1,
  facts_json TEXT,
  CHECK ((publication_id IS NULL) != (report_id IS NULL)),
  UNIQUE (event_id, publication_id),
  UNIQUE (event_id, report_id)
);
CREATE INDEX IF NOT EXISTS evidence_event ON evidence(event_id);

CREATE TABLE IF NOT EXISTS event_history (
  history_id INTEGER PRIMARY KEY,
  event_id INTEGER NOT NULL REFERENCES event(event_id),
  at TEXT NOT NULL,
  actor TEXT NOT NULL,
  action TEXT NOT NULL,
  before_json TEXT,
  after_json TEXT,
  note TEXT
);

CREATE TABLE IF NOT EXISTS review (
  review_id INTEGER PRIMARY KEY,
  subject_type TEXT NOT NULL CHECK (subject_type IN ('report', 'event')),
  subject_id INTEGER NOT NULL,
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

CREATE TABLE IF NOT EXISTS observation_match (
  match_id INTEGER PRIMARY KEY,
  event_id INTEGER NOT NULL REFERENCES event(event_id),
  catalog TEXT NOT NULL CHECK (catalog IN ('local', 'copernicus')),
  observation_ref TEXT NOT NULL,
  product_type TEXT,
  observation_at TEXT,
  footprint_json TEXT,
  spatial_relation TEXT NOT NULL CHECK (spatial_relation IN ('covers', 'intersects', 'nearby',
                                                             'none')),
  distance_km REAL,
  time_offset_days REAL,
  suitability TEXT NOT NULL CHECK (suitability IN ('suitable', 'too_early', 'too_late',
                                                   'partial_cover', 'unusable')),
  reason TEXT NOT NULL,
  stage TEXT NOT NULL CHECK (stage IN ('found', 'retrieved', 'prepared', 'analyzed')),
  analysis_run_id TEXT,
  searched_at TEXT NOT NULL,
  UNIQUE (event_id, catalog, observation_ref)
);

CREATE TABLE IF NOT EXISTS ingestion_run (
  run_id INTEGER PRIMARY KEY,
  trigger TEXT NOT NULL CHECK (trigger IN ('schedule', 'manual', 'cli')),
  started_at TEXT NOT NULL,
  finished_at TEXT,
  status TEXT NOT NULL CHECK (status IN ('running', 'ok', 'partial', 'failed')),
  items_new INTEGER NOT NULL DEFAULT 0,
  events_new INTEGER NOT NULL DEFAULT 0,
  errors_json TEXT,
  duration_s REAL,
  requests_used INTEGER NOT NULL DEFAULT 0,
  requests_budget INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS source_run (
  source_run_id INTEGER PRIMARY KEY,
  run_id INTEGER NOT NULL REFERENCES ingestion_run(run_id),
  source_id TEXT NOT NULL REFERENCES source(source_id),
  started_at TEXT NOT NULL,
  finished_at TEXT,
  status TEXT NOT NULL CHECK (status IN ('running', 'ok', 'partial', 'failed',
                                         'skipped_rate_limit', 'skipped_budget')),
  http_status INTEGER,
  items_seen INTEGER NOT NULL DEFAULT 0,
  items_new INTEGER NOT NULL DEFAULT 0,
  items_relevant INTEGER NOT NULL DEFAULT 0,
  requests_used INTEGER NOT NULL DEFAULT 0,
  error TEXT,
  llm_calls INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS geocode_cache (
  query TEXT PRIMARY KEY,
  result_json TEXT NOT NULL,
  fetched_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS operator_session (
  token_hash TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  created_at TEXT NOT NULL,
  expires_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS setting (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
"""

_initialised: set[Path] = set()


def media_dir() -> Path:
    """Working data of this module (database, photos, secrets); never committed."""
    path = Path(os.environ.get("FLOODVALUE_MEDIA_DIR", REPO_ROOT / "var" / "media"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def db_path() -> Path:
    return media_dir() / "media.sqlite3"


def attachments_dir() -> Path:
    path = media_dir() / "attachments"
    path.mkdir(parents=True, exist_ok=True)
    return path


def secret_file(name: str) -> str:
    """A random server-side secret, created once with owner-only permissions."""
    path = media_dir() / name
    if not path.exists():
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as target:
            target.write(secrets.token_urlsafe(32))
    return path.read_text(encoding="utf-8").strip()


def utc_now() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def iso(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_iso(value: str) -> datetime:
    return datetime.fromisoformat(value)


def now() -> str:
    return iso(utc_now())


def later(days: float) -> str:
    return iso(utc_now() + timedelta(days=days))


@contextmanager
def connect(path: Path | None = None) -> Iterator[sqlite3.Connection]:
    path = path or db_path()
    db = sqlite3.connect(path, timeout=10)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys = ON")
    db.execute("PRAGMA busy_timeout = 10000")
    if path not in _initialised:
        db.execute("PRAGMA journal_mode = WAL")
        db.executescript(SCHEMA)
        _initialised.add(path)
    try:
        yield db
        db.commit()
    except BaseException:
        db.rollback()
        raise
    finally:
        db.close()


def get_setting(db: sqlite3.Connection, key: str, default: str | None = None) -> str | None:
    row = db.execute("SELECT value FROM setting WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else default


def set_setting(db: sqlite3.Connection, key: str, value: str) -> None:
    db.execute("INSERT INTO setting(key, value) VALUES (?, ?) "
               "ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, value))


def record_history(db: sqlite3.Connection, event_id: int, actor: str, action: str,
                   before: dict | None = None, after: dict | None = None,
                   note: str | None = None) -> None:
    db.execute(
        "INSERT INTO event_history(event_id, at, actor, action, before_json, after_json, note) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (event_id, now(), actor, action,
         json.dumps(before, ensure_ascii=False) if before is not None else None,
         json.dumps(after, ensure_ascii=False) if after is not None else None, note))
