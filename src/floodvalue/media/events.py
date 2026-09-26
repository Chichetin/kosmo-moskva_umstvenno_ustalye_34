"""Event registry: create, link evidence, find merge candidates, merge and split with history.

Rules from docs/features/media-monitoring.md §Е: one publication is not necessarily a new
event, ten reprints are not ten independent confirmations, and in doubt two events stay
separate with a 'possible duplicate' mark instead of a silent merge.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
from datetime import datetime, timedelta
from difflib import SequenceMatcher

from floodvalue.media.store import now, parse_iso, record_history

# How far apart two reports may be and still describe one situation, by place precision.
MERGE_RADIUS_KM = {"point": 10, "street": 10, "locality": 20, "district": 40, "region": 120}
# Size of the translucent area drawn when only a district or region is known (no pin).
AREA_RADIUS_KM = {"district": 15, "region": 80}
TIME_SLACK = {"hour": timedelta(hours=6), "day": timedelta(days=1),
              "several_days": timedelta(days=2)}
CLOSED_STATES = ("merged", "split", "dismissed")
EVENT_FIELDS = ("title", "report_type", "place_label", "country", "lat", "lon", "bbox_json",
                "geo_precision", "geo_status", "geo_candidates_json", "event_time_start",
                "event_time_end", "time_precision", "review_state", "verification_result",
                "follow_up", "possible_duplicate_of", "is_demo", "is_replay")


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371.0088 * math.asin(math.sqrt(a))


def time_window(start: str | None, end: str | None,
                precision: str) -> tuple[datetime, datetime] | None:
    if not start or precision not in TIME_SLACK:
        return None
    slack = TIME_SLACK[precision]
    return parse_iso(start) - slack, parse_iso(end or start) + slack


def normalized_title(title: str) -> str:
    text = re.sub(r"[^\w\s]", " ", title.lower())
    return re.sub(r"\s+", " ", text).strip()


def is_reprint(title: str, other: str) -> bool:
    a, b = normalized_title(title), normalized_title(other)
    return bool(a) and (a == b or SequenceMatcher(None, a, b).ratio() >= 0.85)


def _type_family(report_type: str) -> str | None:
    if report_type in ("occurred", "unclear", "negation"):
        return "situation"
    if report_type in ("forecast", "warning"):
        return "alert"
    return None  # historical material never merges into a current event


def merge_candidates(db: sqlite3.Connection, *, lat: float | None, lon: float | None,
                     geo_precision: str, country: str | None, start: str | None,
                     end: str | None, time_precision: str, report_type: str,
                     is_demo: bool = False, exclude: tuple[int, ...] = ()) -> list[dict]:
    """Open events that may describe the same situation, with a high/medium confidence."""
    family = _type_family(report_type)
    if family is None or lat is None or geo_precision not in MERGE_RADIUS_KM:
        return []
    window = time_window(start, end, time_precision)
    rows = db.execute(
        f"SELECT * FROM event WHERE review_state NOT IN {CLOSED_STATES} AND lat IS NOT NULL "
        "AND is_demo = ?", (int(is_demo),)).fetchall()
    found = []
    for row in rows:
        if row["event_id"] in exclude or _type_family(row["report_type"]) != family:
            continue
        if row["geo_precision"] not in MERGE_RADIUS_KM:
            continue
        if country and row["country"] and country.lower() != row["country"].lower():
            continue
        distance = haversine_km(lat, lon, row["lat"], row["lon"])
        radius = max(MERGE_RADIUS_KM[geo_precision], MERGE_RADIUS_KM[row["geo_precision"]])
        if distance > radius:
            continue
        other = time_window(row["event_time_start"], row["event_time_end"],
                            row["time_precision"])
        if window and other and (window[1] < other[0] or other[1] < window[0]):
            continue
        tight = distance <= min(MERGE_RADIUS_KM[geo_precision],
                                MERGE_RADIUS_KM[row["geo_precision"]])
        confidence = "high" if tight and window and other else "medium"
        found.append({"event_id": row["event_id"], "title": row["title"],
                      "distance_km": round(distance, 1), "confidence": confidence})
    return sorted(found, key=lambda item: (item["confidence"] != "high", item["distance_km"]))


def create_event(db: sqlite3.Connection, actor: str, **fields) -> int:
    unknown = set(fields) - set(EVENT_FIELDS)
    if unknown:
        raise ValueError(f"unknown event fields: {sorted(unknown)}")
    stamp = now()
    fields.setdefault("review_state", "new" if fields.get("geo_status") == "resolved"
                      else "needs_geo")
    columns = ", ".join((*fields, "first_detected_at", "updated_at"))
    marks = ", ".join("?" for _ in range(len(fields) + 2))
    cursor = db.execute(f"INSERT INTO event ({columns}) VALUES ({marks})",
                        (*fields.values(), stamp, stamp))
    event_id = int(cursor.lastrowid)
    record_history(db, event_id, actor, "created", after=fields)
    return event_id


def link(db: sqlite3.Connection, event_id: int, *, origin: str, linked_by: str,
         link_kind: str = "auto", publication_id: int | None = None,
         report_id: int | None = None, facts: dict | None = None) -> bool:
    """Attach evidence; a publication whose title repeats a linked one is a reprint."""
    is_original = 1
    if publication_id is not None:
        title = db.execute("SELECT title FROM publication WHERE publication_id = ?",
                           (publication_id,)).fetchone()["title"]
        for other in db.execute(
                "SELECT p.title, p.domain FROM evidence e JOIN publication p "
                "ON p.publication_id = e.publication_id WHERE e.event_id = ? AND e.is_original = 1",
                (event_id,)):
            if is_reprint(title, other["title"]):
                is_original = 0
                db.execute("UPDATE publication SET original_source_hint = ? "
                           "WHERE publication_id = ? AND original_source_hint IS NULL",
                           (other["domain"], publication_id))
                break
    cursor = db.execute(
        "INSERT OR IGNORE INTO evidence(event_id, publication_id, report_id, origin, link_kind, "
        "linked_at, linked_by, is_original, facts_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (event_id, publication_id, report_id, origin, link_kind, now(), linked_by, is_original,
         json.dumps(facts, ensure_ascii=False) if facts else None))
    if cursor.rowcount:
        db.execute("UPDATE event SET updated_at = ? WHERE event_id = ?", (now(), event_id))
    return bool(cursor.rowcount)


def independent_origin_count(db: sqlite3.Connection, event_id: int) -> int:
    """Originals only: reprints and repeated bulletins of one source count once."""
    rows = db.execute(
        "SELECT e.origin, e.report_id, p.source_id, p.title FROM evidence e "
        "LEFT JOIN publication p ON p.publication_id = e.publication_id "
        "WHERE e.event_id = ? AND e.is_original = 1", (event_id,)).fetchall()
    keys = set()
    for row in rows:
        if row["origin"] == "citizen":
            keys.add(("citizen", row["report_id"]))
        elif row["origin"] == "bulletin":
            keys.add(("bulletin", row["source_id"]))
        else:
            keys.add(("media", normalized_title(row["title"])))
    return len(keys)


def update_event(db: sqlite3.Connection, event_id: int, changes: dict, actor: str,
                 action: str = "edited", note: str | None = None) -> dict:
    unknown = set(changes) - set(EVENT_FIELDS)
    if unknown:
        raise ValueError(f"unknown event fields: {sorted(unknown)}")
    row = db.execute("SELECT * FROM event WHERE event_id = ?", (event_id,)).fetchone()
    if row is None:
        raise KeyError(event_id)
    before = {key: row[key] for key in changes}
    if changes:
        assignments = ", ".join(f"{key} = ?" for key in changes)
        db.execute(f"UPDATE event SET {assignments}, updated_at = ? WHERE event_id = ?",
                   (*changes.values(), now(), event_id))
    record_history(db, event_id, actor, action, before=before, after=changes, note=note)
    return before


def merge_events(db: sqlite3.Connection, source_id: int, target_id: int, actor: str,
                 note: str | None = None) -> None:
    if source_id == target_id:
        raise ValueError("нельзя объединить событие с самим собой")
    for event_id in (source_id, target_id):
        row = db.execute("SELECT review_state FROM event WHERE event_id = ?",
                         (event_id,)).fetchone()
        if row is None:
            raise KeyError(event_id)
        if row["review_state"] in CLOSED_STATES:
            raise ValueError(f"событие {event_id} уже закрыто ({row['review_state']})")
    moved = 0
    for evidence in db.execute("SELECT * FROM evidence WHERE event_id = ?",
                               (source_id,)).fetchall():
        moved += link(db, target_id, origin=evidence["origin"], linked_by=actor,
                      link_kind="manual_merge", publication_id=evidence["publication_id"],
                      report_id=evidence["report_id"],
                      facts=json.loads(evidence["facts_json"] or "null"))
    db.execute("DELETE FROM evidence WHERE event_id = ?", (source_id,))
    db.execute("UPDATE event SET review_state = 'merged', merged_into = ?, updated_at = ? "
               "WHERE event_id = ?", (target_id, now(), source_id))
    record_history(db, source_id, actor, "merged_into", after={"merged_into": target_id},
                   note=note)
    record_history(db, target_id, actor, "merged_from",
                   after={"merged_from": source_id, "evidence_moved": moved}, note=note)


def split_event(db: sqlite3.Connection, event_id: int, evidence_ids: list[int], actor: str,
                note: str | None = None) -> int:
    """Move the chosen evidence to a new event with the same place and time for editing."""
    row = db.execute("SELECT * FROM event WHERE event_id = ?", (event_id,)).fetchone()
    if row is None:
        raise KeyError(event_id)
    chosen = db.execute(
        f"SELECT * FROM evidence WHERE event_id = ? AND evidence_id IN "
        f"({', '.join('?' for _ in evidence_ids)})", (event_id, *evidence_ids)).fetchall()
    total = db.execute("SELECT COUNT(*) FROM evidence WHERE event_id = ?",
                       (event_id,)).fetchone()[0]
    if not chosen or len(chosen) != len(set(evidence_ids)) or len(chosen) == total:
        raise ValueError("выберите часть свидетельств этого события, но не все")
    fields = {key: row[key] for key in EVENT_FIELDS
              if key not in ("review_state", "verification_result", "follow_up",
                             "possible_duplicate_of")}
    fields["title"] = f"{row['title']} (выделено)"
    new_id = create_event(db, actor, **fields)
    for evidence in chosen:
        db.execute("UPDATE evidence SET event_id = ?, link_kind = 'manual_split', linked_by = ?, "
                   "linked_at = ? WHERE evidence_id = ?",
                   (new_id, actor, now(), evidence["evidence_id"]))
    record_history(db, event_id, actor, "split",
                   after={"new_event_id": new_id, "evidence_ids": sorted(evidence_ids)}, note=note)
    record_history(db, new_id, actor, "split_from", after={"event_id": event_id}, note=note)
    return new_id
