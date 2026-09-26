"""Event register: create, link evidence, merge repeats, manual merge/split/corrections with
history, moderation of eyewitness reports and analyst review. Reprints are linked with
is_original=0 and never raise independent_origin_count; nothing here is a "credibility" score.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta

from floodvalue.media.geo import haversine_km, representative_point, valid_lon_lat
from floodvalue.media.store import add_history, dumps, loads, now_iso

SAME_PLACE_KM = 25.0
SAME_TIME = timedelta(days=3)
NEGATION_WINDOW = timedelta(days=7)
OPEN_STATES = ("new", "needs_geo", "ready", "under_review", "reviewed")
VERIFICATION = ("not_checked", "consistent", "discrepancy", "insufficient_data")
FOLLOW_UP = ("none", "needs_imagery", "merge", "split", "dismiss")
PRECISIONS = ("point", "street", "locality", "district", "region", "country", "unknown")


class ActionError(ValueError):
    """An analyst action that the rules forbid; the message is shown to the analyst."""


def _ts(value: str | None) -> datetime | None:
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC) if value else None


def _window(row) -> tuple[datetime, datetime]:
    start = _ts(row["event_time_start"]) or _ts(row["first_detected_at"])
    end = _ts(row["event_time_end"]) or start
    return start - SAME_TIME, end + SAME_TIME


def event_snapshot(conn: sqlite3.Connection, event_id: int) -> dict:
    row = conn.execute("SELECT * FROM events WHERE event_id = ?", (event_id,)).fetchone()
    if row is None:
        raise ActionError(f"события {event_id} нет")
    return dict(row)


def create_event(conn, *, title: str, report_type: str, place: dict, time: dict | None,
                 actor: str, is_demo: bool = False, is_replay: bool = False,
                 external_key: str | None = None, country: str | None = None) -> int:
    time = time or {}
    geo_status = place["status"]
    state = "ready" if geo_status == "resolved" else "needs_geo"
    now = now_iso()
    cur = conn.execute(
        """INSERT INTO events(title, report_type, place_label, country, geometry_json,
             geo_precision, geo_status, geo_candidates_json, event_time_start, event_time_end,
             time_precision, first_detected_at, updated_at, review_state, external_key,
             is_demo, is_replay)
           VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (title[:300], report_type, place.get("label"), country, dumps(place.get("geometry")),
         place["precision"], geo_status, dumps(place.get("candidates")), time.get("start"),
         time.get("end"), time.get("precision", "unknown"), now, now, state, external_key,
         int(is_demo), int(is_replay)))
    event_id = cur.lastrowid
    add_history(conn, event_id, actor, "create", None, {"report_type": report_type,
                                                        "place": place.get("label"),
                                                        "geo_status": geo_status})
    return event_id


def link(conn, event_id: int, *, origin: str, is_original: bool, link_kind: str, actor: str,
         publication_id: int | None = None, report_id: str | None = None,
         report_type: str | None = None, facts: dict | None = None) -> bool:
    cur = conn.execute(
        """INSERT OR IGNORE INTO evidence(event_id, publication_id, report_id, origin,
             is_original, link_kind, linked_at, linked_by, report_type, facts_json)
           VALUES(?,?,?,?,?,?,?,?,?,?)""",
        (event_id, publication_id, report_id, origin, int(is_original), link_kind, now_iso(),
         actor, report_type, dumps(facts)))
    if cur.rowcount:
        recount(conn, event_id)
        add_history(conn, event_id, actor, "link_evidence", None,
                    {"publication_id": publication_id, "report_id": report_id,
                     "is_original": is_original, "link_kind": link_kind})
    return bool(cur.rowcount)


def recount(conn, event_id: int) -> None:
    """Independent origins = distinct original publications + accepted eyewitness reports.
    Reprints (is_original=0) and bulletin updates of the same GDACS event count once; a denial
    is attached to the event but is not an origin of the report."""
    rows = conn.execute(
        """SELECT e.origin, e.is_original, p.source_id, p.source_domain, e.report_id,
                  e.report_type
           FROM evidence e LEFT JOIN publications p USING(publication_id)
           WHERE e.event_id = ?""", (event_id,)).fetchall()
    origins = set()
    for r in rows:
        if r["report_type"] == "negation":
            continue
        if r["origin"] == "citizen":
            origins.add(("citizen", r["report_id"]))
        elif r["origin"] == "bulletin":
            origins.add(("bulletin", r["source_id"]))
        elif r["is_original"]:
            origins.add(("news", r["source_domain"]))
    kinds = ",".join(sorted({r["origin"] for r in rows}))
    conn.execute("UPDATE events SET independent_origin_count = ?, source_kinds = ?, "
                 "updated_at = ? WHERE event_id = ?", (len(origins), kinds, now_iso(), event_id))


def find_similar(conn, report_type: str, geometry: dict | None, time: dict | None,
                 exclude: int | None = None) -> int | None:
    """Level 3 of the spec: same type, place within SAME_PLACE_KM, overlapping time. Only
    resolved places; a missing place never merges automatically (doubt → separate events)."""
    point = representative_point(geometry)
    if point is None or report_type == "unclear":
        return None
    probe = {"event_time_start": (time or {}).get("start"),
             "event_time_end": (time or {}).get("end"), "first_detected_at": now_iso()}
    lo, hi = _window(probe)
    for row in conn.execute(
            "SELECT * FROM events WHERE report_type = ? AND geo_status = 'resolved' "
            "AND review_state IN ('new','needs_geo','ready','under_review','reviewed') "
            "AND event_id IS NOT ? ORDER BY event_id", (report_type, exclude)):
        other = representative_point(loads(row["geometry_json"]))
        a, b = _window(row)
        if other and haversine_km(point, other) <= SAME_PLACE_KM and a <= hi and lo <= b:
            return row["event_id"]
    return None


def find_negation_target(conn, geometry: dict | None, label: str | None) -> int | None:
    point = representative_point(geometry)
    since = datetime.now(UTC) - NEGATION_WINDOW
    for row in conn.execute("SELECT * FROM events WHERE report_type = 'occurred' AND "
                            "review_state IN ('new','needs_geo','ready','under_review','reviewed')"
                            " ORDER BY event_id DESC"):
        if _ts(row["first_detected_at"]) < since:
            continue
        other = representative_point(loads(row["geometry_json"]))
        if point and other and haversine_km(point, other) <= SAME_PLACE_KM:
            return row["event_id"]
        if not point and label and (row["place_label"] or "").casefold() == label.casefold():
            return row["event_id"]
    return None


# --- analyst actions ---------------------------------------------------------------------------

def merge_events(conn, source_id: int, target_id: int, actor: str) -> None:
    if source_id == target_id:
        raise ActionError("нельзя объединить событие с самим собой")
    src, dst = event_snapshot(conn, source_id), event_snapshot(conn, target_id)
    if src["review_state"] == "merged" or dst["review_state"] == "merged":
        raise ActionError("одно из событий уже объединено")
    for ev in conn.execute("SELECT * FROM evidence WHERE event_id = ?", (source_id,)).fetchall():
        link(conn, target_id, origin=ev["origin"], is_original=bool(ev["is_original"]),
             link_kind="manual_merge", actor=actor, publication_id=ev["publication_id"],
             report_id=ev["report_id"], report_type=ev["report_type"],
             facts=loads(ev["facts_json"]))
    conn.execute("UPDATE events SET review_state='merged', merged_into=?, updated_at=? "
                 "WHERE event_id=?", (target_id, now_iso(), source_id))
    conn.execute("UPDATE events SET manual_lock=1 WHERE event_id=?", (target_id,))
    conn.execute("UPDATE citizen_reports SET event_id=? WHERE event_id=?", (target_id, source_id))
    add_history(conn, source_id, actor, "merged_into", {"review_state": src["review_state"]},
                {"merged_into": target_id})
    add_history(conn, target_id, actor, "merge_from", None, {"from": source_id})


def split_event(conn, event_id: int, evidence_ids: list[int], actor: str) -> int:
    ev = event_snapshot(conn, event_id)
    rows = [conn.execute("SELECT * FROM evidence WHERE evidence_id=? AND event_id=?",
                         (i, event_id)).fetchone() for i in evidence_ids]
    if not rows or any(r is None for r in rows):
        raise ActionError("выбраны свидетельства не этого события")
    total = conn.execute("SELECT COUNT(*) FROM evidence WHERE event_id=?", (event_id,)).fetchone()[0]
    if len(rows) >= total:
        raise ActionError("в исходном событии должно остаться хотя бы одно свидетельство")
    place = {"label": ev["place_label"], "geometry": None, "precision": "unknown",
             "status": "missing", "candidates": None}
    new_id = create_event(conn, title=f"{ev['title']} (выделено)", report_type=ev["report_type"],
                          place=place, time=None, actor=actor, is_demo=bool(ev["is_demo"]),
                          is_replay=bool(ev["is_replay"]))
    for r in rows:
        conn.execute("DELETE FROM evidence WHERE evidence_id=?", (r["evidence_id"],))
        link(conn, new_id, origin=r["origin"], is_original=bool(r["is_original"]),
             link_kind="manual_split", actor=actor, publication_id=r["publication_id"],
             report_id=r["report_id"], report_type=r["report_type"], facts=loads(r["facts_json"]))
        if r["report_id"]:
            conn.execute("UPDATE citizen_reports SET event_id=? WHERE report_id=?",
                          (new_id, r["report_id"]))
    conn.execute("UPDATE events SET manual_lock=1 WHERE event_id IN (?, ?)", (event_id, new_id))
    recount(conn, event_id)
    add_history(conn, event_id, actor, "split", None,
                {"moved_evidence": evidence_ids, "new_event": new_id})
    return new_id


def set_place(conn, event_id: int, *, label: str, lon: float | None, lat: float | None,
              precision: str, actor: str) -> None:
    if precision not in PRECISIONS:
        raise ActionError(f"неизвестная точность места: {precision}")
    if (lon is None) != (lat is None) or (lon is not None and not valid_lon_lat(lon, lat)):
        raise ActionError("координаты вне допустимых границ или [0, 0]")
    before = event_snapshot(conn, event_id)
    geometry = {"type": "Point", "coordinates": [lon, lat]} if lon is not None else None
    status = "resolved" if geometry else "missing"
    state = before["review_state"]
    if state in ("new", "needs_geo", "ready"):
        state = "ready" if geometry else "needs_geo"
    conn.execute("UPDATE events SET place_label=?, geometry_json=?, geo_precision=?, "
                 "geo_status=?, geo_candidates_json=NULL, review_state=?, manual_lock=1, "
                 "updated_at=? WHERE event_id=?",
                 (label, dumps(geometry), precision if geometry else "unknown", status, state,
                  now_iso(), event_id))
    add_history(conn, event_id, actor, "set_place",
                {k: before[k] for k in ("place_label", "geometry_json", "geo_precision")},
                {"place_label": label, "geometry": geometry, "geo_precision": precision})


def set_time(conn, event_id: int, *, start: str | None, end: str | None, precision: str,
             actor: str) -> None:
    if precision not in ("hour", "day", "several_days", "unknown"):
        raise ActionError(f"неизвестная точность времени: {precision}")
    s, e = _ts(start), _ts(end)
    if s and e and e < s:
        raise ActionError("конец интервала раньше начала")
    before = event_snapshot(conn, event_id)
    conn.execute("UPDATE events SET event_time_start=?, event_time_end=?, time_precision=?, "
                 "manual_lock=1, updated_at=? WHERE event_id=?",
                 (start, end, precision, now_iso(), event_id))
    add_history(conn, event_id, actor, "set_time",
                {k: before[k] for k in ("event_time_start", "event_time_end", "time_precision")},
                {"event_time_start": start, "event_time_end": end, "time_precision": precision})


def review_event(conn, event_id: int, *, verification_result: str, follow_up: str, basis: str,
                 comment: str, actor: str) -> None:
    """consistent / discrepancy need a suitable, analysed observation: a catalog record is not
    a check, and without imagery the honest answers are not_checked or insufficient_data."""
    if verification_result not in VERIFICATION or follow_up not in FOLLOW_UP:
        raise ActionError("недопустимое значение результата проверки")
    ev = event_snapshot(conn, event_id)
    if ev["review_state"] in ("merged", "dismissed") and follow_up != "dismiss":
        raise ActionError("событие закрыто")
    if verification_result in ("consistent", "discrepancy"):
        ok = conn.execute("SELECT 1 FROM observation_matches WHERE event_id=? AND "
                          "suitability='suitable' AND stage='analyzed'", (event_id,)).fetchone()
        if not ok:
            raise ActionError("нет подходящего проанализированного наблюдения: доступны только "
                              "«не проверено» и «недостаточно данных»")
    state = "dismissed" if follow_up == "dismiss" else "reviewed"
    conn.execute("INSERT INTO reviews(target_kind, target_id, reviewer, reviewed_at, decision, "
                 "verification_result, basis, comment, follow_up, state_before, state_after) "
                 "VALUES('event',?,?,?,?,?,?,?,?,?,?)",
                 (str(event_id), actor, now_iso(), state, verification_result, basis, comment,
                  follow_up, ev["review_state"], state))
    conn.execute("UPDATE events SET verification_result=?, follow_up=?, review_state=?, "
                 "updated_at=? WHERE event_id=?",
                 (verification_result, follow_up, state, now_iso(), event_id))
    add_history(conn, event_id, actor, "review",
                {"review_state": ev["review_state"], "verification_result":
                 ev["verification_result"]},
                {"review_state": state, "verification_result": verification_result,
                 "follow_up": follow_up})


REPORT_TYPE_BY_CATEGORY = {"flooding": "occurred", "water_rising": "occurred",
                           "threat": "warning", "other": "unclear"}


def moderate_report(conn, report_id: str, *, decision: str, actor: str, comment: str = "",
                    event_id: int | None = None) -> int | None:
    """accept → link to the given event or a new one; reject/spam → stays out of events.
    Acceptance admits the report to work; it is not a satellite-proven flood."""
    rep = conn.execute("SELECT * FROM citizen_reports WHERE report_id=?", (report_id,)).fetchone()
    if rep is None:
        raise ActionError("обращения нет")
    if decision not in ("accept", "reject", "spam"):
        raise ActionError("решение: accept, reject или spam")
    if rep["moderation_status"] != "pending":
        raise ActionError(f"обращение уже рассмотрено: {rep['moderation_status']}")
    status = {"accept": "accepted", "reject": "rejected", "spam": "spam"}[decision]
    target = None
    if decision == "accept":
        report_type = REPORT_TYPE_BY_CATEGORY.get(rep["category"], "unclear")
        if event_id is not None:
            if event_snapshot(conn, event_id)["review_state"] in ("merged", "dismissed"):
                raise ActionError("событие закрыто, выберите другое")
            target = event_id
        else:
            if rep["lat"] is not None:
                place = {"label": rep["place_text"] or "точка очевидца",
                         "geometry": {"type": "Point", "coordinates": [rep["lon"], rep["lat"]]},
                         "precision": rep["location_precision"], "status": "resolved",
                         "candidates": None}
            else:
                place = {"label": rep["place_text"], "geometry": None, "precision": "unknown",
                         "status": "missing", "candidates": None}
            time = ({"start": rep["observed_at"], "end": rep["observed_at"],
                     "precision": rep["observed_precision"]} if rep["observed_at"] else None)
            target = find_similar(conn, report_type, place["geometry"], time)
            if target is None:
                target = create_event(conn, title=f"Очевидец: {rep['description'][:80]}",
                                      report_type=report_type, place=place, time=time,
                                      actor=actor, is_demo=bool(rep["is_demo"]))
        link(conn, target, origin="citizen", is_original=True, link_kind="moderation",
             actor=actor, report_id=report_id, report_type=report_type,
             facts={"category": rep["category"], "observed_at": rep["observed_at"]})
    conn.execute("UPDATE citizen_reports SET moderation_status=?, event_id=? WHERE report_id=?",
                 (status, target, report_id))
    conn.execute("INSERT INTO reviews(target_kind, target_id, reviewer, reviewed_at, decision, "
                 "comment, state_before, state_after) VALUES('report',?,?,?,?,?,?,?)",
                 (report_id, actor, now_iso(), decision, comment, "pending", status))
    return target
