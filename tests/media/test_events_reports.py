"""Analyst actions with history, review rules, eyewitness validation and photo sanitising."""

import io
from datetime import UTC, datetime, timedelta

import pytest
from PIL import Image
from pydantic import ValidationError

from floodvalue.media import events as ev
from floodvalue.media.reports import (
    ReportError,
    ReportIn,
    create_report,
    public_status,
    purge_expired,
    sanitize_photo,
)
from floodvalue.media.store import transaction

from .test_pipeline import events, ingest, news


def report(**kw):
    base = {"category": "flooding", "description": "Вода подошла к домам на улице Речной",
            "lat": 55.75, "lon": 37.62, "location_precision": "point",
            "idempotency_key": "k" * 20}
    return ReportIn(**{**base, **kw})


def jpeg_with_exif(size=(200, 150)) -> bytes:
    img = Image.new("RGB", size, (40, 90, 160))
    exif = Image.Exif()
    exif[0x010F] = "SecretCam"  # Make
    exif[0x8825] = {1: "N", 2: (55.0, 45.0, 0.0)}  # GPSInfo latitude
    out = io.BytesIO()
    img.save(out, "JPEG", exif=exif)
    return out.getvalue()


# --- events ------------------------------------------------------------------------------------

def test_merge_and_split_keep_history_and_survive_recount(conn):
    ingest(conn, news("1", "Floods hit Kerala, thousands evacuated", domain="a.example"))
    ingest(conn, news("2", "Floods hit Assam, roads washed away", domain="b.example"))
    a, b = events(conn)
    with transaction(conn):
        ev.merge_events(conn, b["event_id"], a["event_id"], "analyst:test")
    assert ev.event_snapshot(conn, b["event_id"])["review_state"] == "merged"
    assert ev.event_snapshot(conn, a["event_id"])["independent_origin_count"] == 2
    moved = conn.execute("SELECT evidence_id FROM evidence WHERE event_id=? AND publication_id=2",
                         (a["event_id"],)).fetchone()[0]
    with transaction(conn):
        new_id = ev.split_event(conn, a["event_id"], [moved], "analyst:test")
    assert ev.event_snapshot(conn, new_id)["manual_lock"] == 1
    actions = [r[0] for r in conn.execute("SELECT action FROM event_history WHERE event_id=?",
                                          (a["event_id"],))]
    assert "merge_from" in actions and "split" in actions
    with pytest.raises(ev.ActionError):
        ev.split_event(conn, new_id, [moved + 999], "analyst:test")


def test_set_place_rejects_zero_and_out_of_range(conn):
    ingest(conn, news("1", "Severe flooding reported, roads closed"))
    (e,) = events(conn)
    for lon, lat in ((0, 0), (200, 10), (10, None)):
        with pytest.raises(ev.ActionError):
            ev.set_place(conn, e["event_id"], label="x", lon=lon, lat=lat, precision="point", actor="a")
    ev.set_place(conn, e["event_id"], label="Kochi", lon=76.27, lat=9.93, precision="locality", actor="a")
    snap = ev.event_snapshot(conn, e["event_id"])
    assert snap["geo_status"] == "resolved" and snap["review_state"] == "ready"


def test_review_cannot_confirm_without_analysed_observation(conn):
    ingest(conn, news("1", "Floods hit Kerala, thousands evacuated"))
    (e,) = events(conn)
    with pytest.raises(ev.ActionError, match="недостаточно"):
        ev.review_event(conn, e["event_id"], verification_result="consistent", follow_up="none",
                        basis="", comment="", actor="a")
    ev.review_event(conn, e["event_id"], verification_result="insufficient_data",
                    follow_up="needs_imagery", basis="каталог пуст", comment="", actor="a")
    snap = ev.event_snapshot(conn, e["event_id"])
    assert snap["verification_result"] == "insufficient_data" and snap["follow_up"] == "needs_imagery"
    conn.execute("INSERT INTO observation_matches(event_id, catalog, observation_ref, spatial_relation, "
                 "suitability, reason, stage, found_at) VALUES(?, 'x', 'r', 'covers', 'suitable', 'r', "
                 "'found', 'now')", (e["event_id"],))
    with pytest.raises(ev.ActionError):  # found in a catalog is not analysed
        ev.review_event(conn, e["event_id"], verification_result="discrepancy", follow_up="none",
                        basis="", comment="", actor="a")


# --- eyewitness reports ------------------------------------------------------------------------

@pytest.mark.parametrize("kw", [
    {"lat": 95.0}, {"lon": None}, {"lat": 0.0, "lon": 0.0}, {"category": "hack"},
    {"description": "short"}, {"lat": None, "lon": None, "place_text": None},
    {"observed_at": datetime.now(UTC) + timedelta(days=1)},
    {"observed_at": datetime(2026, 9, 26, 10, 0)},  # noqa: DTZ001 — no timezone on purpose
    {"idempotency_key": "../../etc"}, {"extra_field": "x"},
])
def test_invalid_reports_rejected(kw):
    with pytest.raises(ValidationError):
        report(**kw)


def test_place_without_coordinates_is_allowed_and_not_point():
    r = report(lat=None, lon=None, place_text="Заречный, у моста")
    assert r.location_precision == "locality"


def test_photo_is_reencoded_without_metadata():
    raw = jpeg_with_exif()
    assert b"SecretCam" in raw
    clean, w, h = sanitize_photo(raw)
    assert b"SecretCam" not in clean and b"Exif" not in clean
    assert Image.open(io.BytesIO(clean)).getexif() == {}
    assert (w, h) == (200, 150)


@pytest.mark.parametrize("raw", [b"<html><script>alert(1)</script></html>", b"GIF89a" + b"\0" * 100,
                                 b"%PDF-1.4 fake"])
def test_non_images_rejected_by_content(raw):
    with pytest.raises(ReportError):
        sanitize_photo(raw)


def test_oversized_dimensions_rejected():
    img = Image.new("L", (9000, 64))
    out = io.BytesIO()
    img.save(out, "PNG")
    with pytest.raises(ReportError, match="размер"):
        sanitize_photo(out.getvalue())


def test_idempotent_submission_and_token_status(conn, settings):
    rid, token, created = create_report(conn, report(), [jpeg_with_exif()], settings.photo_dir, 90)
    rid2, token2, created2 = create_report(conn, report(), [], settings.photo_dir, 90)
    assert created and not created2 and (rid, token) == (rid2, token2)
    assert conn.execute("SELECT COUNT(*) FROM citizen_reports").fetchone()[0] == 1
    assert public_status(conn, token)["status"] == "На проверке у аналитика"
    assert public_status(conn, "0" * 40) is None
    files = list(settings.photo_dir.iterdir())
    assert len(files) == 1 and len(files[0].stem) == 32  # random name, not the upload name


def test_moderation_accept_links_event_reject_does_not(conn, settings):
    rid, *_ = create_report(conn, report(), [], settings.photo_dir, 90)
    rid2, *_ = create_report(conn, report(idempotency_key="z" * 20), [], settings.photo_dir, 90)
    with transaction(conn):
        event_id = ev.moderate_report(conn, rid, decision="accept", actor="a")
        ev.moderate_report(conn, rid2, decision="spam", actor="a")
    e = ev.event_snapshot(conn, event_id)
    assert e["report_type"] == "occurred" and e["geo_status"] == "resolved"
    assert e["verification_result"] == "not_checked"  # accepted ≠ proven by satellite
    assert conn.execute("SELECT COUNT(*) FROM evidence WHERE report_id=?", (rid2,)).fetchone()[0] == 0
    with pytest.raises(ev.ActionError):
        ev.moderate_report(conn, rid, decision="reject", actor="a")


def test_retention_purges_report_and_photo(conn, settings):
    rid, *_ = create_report(conn, report(), [jpeg_with_exif()], settings.photo_dir, 90)
    with transaction(conn):
        ev.moderate_report(conn, rid, decision="accept", actor="a")
    far = (datetime.now(UTC) + timedelta(days=91)).strftime("%Y-%m-%dT%H:%M:%SZ")
    with transaction(conn):
        assert purge_expired(conn, settings.photo_dir, now=far) == 1
    assert not list(settings.photo_dir.iterdir())
    assert conn.execute("SELECT COUNT(*) FROM citizen_reports").fetchone()[0] == 0
