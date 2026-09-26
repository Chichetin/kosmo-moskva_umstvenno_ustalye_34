"""Ingestion end to end on fixtures: no duplicates, reprints are not confirmations, one article
can tell about several places, errors are errors, data survives a restart."""

from contextlib import closing

import pytest

from floodvalue.media import pipeline
from floodvalue.media.netfetch import FetchError
from floodvalue.media.pipeline import ingest_item, run_source
from floodvalue.media.store import connect, transaction

GAZ = {"kerala": (76.27, 10.85), "assam": (92.94, 26.20), "springfield": None}


class Gazetteer:
    def lookup(self, q):
        if q.casefold() == "springfield":  # two far-apart places with one name
            return [{"label": "Springfield, IL", "lon": -89.65, "lat": 39.80, "precision": "locality", "bbox": None},
                    {"label": "Springfield, MA", "lon": -72.59, "lat": 42.10, "precision": "locality", "bbox": None}]
        hit = GAZ.get(q.casefold())
        return [{"label": q, "lon": hit[0], "lat": hit[1], "precision": "region", "bbox": None}] if hit else []


def news(ext, title, url=None, domain="a.example", when="2026-09-26T08:00:00Z"):
    return {"external_id": ext, "url": url or f"https://{domain}/{ext}", "title": title,
            "text_excerpt": None, "language": "en", "source_domain": domain,
            "published_at": when, "source_updated_at": None, "structured": None}


def source(conn, sid="gdelt_floods"):
    return conn.execute("SELECT * FROM sources WHERE source_id=?", (sid,)).fetchone()


GAZETTEER = Gazetteer()


def ingest(conn, item, sid="gdelt_floods", geocoder=GAZETTEER):
    with transaction(conn):
        return ingest_item(conn, source(conn, sid), item, geocoder=geocoder)


def events(conn):
    return conn.execute("SELECT * FROM events ORDER BY event_id").fetchall()


def test_reimport_creates_no_duplicates(conn):
    item = news("1", "Floods hit Kerala, thousands evacuated")
    assert ingest(conn, item) == "new"
    assert ingest(conn, item) == "unchanged"
    assert conn.execute("SELECT COUNT(*) FROM publications").fetchone()[0] == 1
    assert len(events(conn)) == 1


def test_reprints_are_one_event_one_origin(conn):
    ingest(conn, news("1", "Floods hit Kerala, thousands evacuated", domain="a.example"))
    ingest(conn, news("2", "Floods hit Kerala, thousands evacuated", domain="b.example"))
    ingest(conn, news("3", "FLOODS HIT KERALA — thousands evacuated", domain="c.example"))
    (ev,) = events(conn)
    assert ev["independent_origin_count"] == 1
    kinds = [r["is_original"] for r in conn.execute("SELECT is_original FROM evidence ORDER BY evidence_id")]
    assert kinds == [1, 0, 0]


def test_one_article_several_territories(conn):
    ingest(conn, news("1", "Floods hit Kerala and heavy flooding in Assam"))
    labels = sorted(e["place_label"] for e in events(conn))
    assert labels == ["Assam", "Kerala"]
    assert conn.execute("SELECT COUNT(DISTINCT publication_id) FROM evidence").fetchone()[0] == 1


def test_forecast_negation_historical(conn):
    ingest(conn, news("1", "Floods hit Kerala, thousands evacuated"))
    ingest(conn, news("2", "Reports of flooding in Kerala denied by officials", domain="b.example"))
    ingest(conn, news("3", "Heavy rain expected to cause flooding in Assam", domain="c.example"))
    ingest(conn, news("4", "Ten years ago: remembering the flood in Assam", domain="d.example"))
    by_type = {e["report_type"]: e for e in events(conn)}
    assert set(by_type) == {"occurred", "forecast", "historical"}
    occurred = by_type["occurred"]
    neg = conn.execute("SELECT * FROM evidence WHERE report_type='negation'").fetchone()
    assert neg["event_id"] == occurred["event_id"]  # attached, event kept
    assert occurred["verification_result"] == "not_checked" and occurred["review_state"] == "ready"
    assert occurred["independent_origin_count"] == 1  # the denial is not an origin


def test_ambiguous_and_missing_place_stay_in_queue(conn):
    ingest(conn, news("1", "Floods hit Springfield overnight"))
    ingest(conn, news("2", "Severe flooding reported, roads closed", domain="b.example"))
    amb, missing = events(conn)
    assert amb["geo_status"] == "ambiguous" and amb["review_state"] == "needs_geo"
    assert amb["geometry_json"] is None and "Springfield, MA" in amb["geo_candidates_json"]
    assert missing["geo_status"] == "missing" and missing["geometry_json"] is None


def test_no_geocoder_means_missing_not_zero(conn):
    ingest(conn, news("1", "Floods hit Kerala"), geocoder=None)
    (ev,) = events(conn)
    assert ev["geometry_json"] is None and ev["geo_precision"] == "unknown"


def test_update_of_same_publication_keeps_event(conn):
    src = source(conn, "gdacs_floods")
    base = {"external_id": "FL-1-1", "url": "https://www.gdacs.org/report.aspx?eventid=1",
            "title": "Flood in India (Orange)", "text_excerpt": "x", "language": "en",
            "source_domain": "gdacs.org", "published_at": "2026-09-22T06:00:00Z",
            "source_updated_at": "2026-09-22T06:00:00Z",
            "structured": {"kind": "bulletin", "event_key": "gdacs:FL:1", "report_type": "occurred",
                           "time": {"start": "2026-09-09T00:00:00Z", "end": "2026-09-22T00:00:00Z", "precision": "day"},
                           "place": {"label": "India", "country": "IND", "precision": "region",
                                     "geometry": {"type": "Point", "coordinates": [76.6, 9.6]}},
                           "alert_level": "Orange", "provider": "GLOFAS", "glide": None}}
    with transaction(conn):
        assert ingest_item(conn, src, base) == "new"
    later = {**base, "source_updated_at": "2026-09-22T06:10:00Z",
             "structured": {**base["structured"], "time": {**base["structured"]["time"], "end": "2026-09-23T00:00:00Z"}}}
    with transaction(conn):
        assert ingest_item(conn, src, later) == "updated"
    (ev,) = events(conn)
    assert ev["event_time_end"] == "2026-09-23T00:00:00Z"
    assert conn.execute("SELECT COUNT(*) FROM publications").fetchone()[0] == 1
    actions = [r[0] for r in conn.execute("SELECT action FROM event_history WHERE event_id=?", (ev["event_id"],))]
    assert "publication_updated" in actions and "bulletin_time_update" in actions


@pytest.mark.parametrize("error, status", [(FetchError("HTTP 503", 503, True), "failed"),
                                           (FetchError("HTTP 429", 429, True), "skipped_rate_limit")])
def test_source_error_is_not_no_news(conn, settings, monkeypatch, error, status):
    def broken(*a, **k):
        raise error
    monkeypatch.setitem(pipeline.ADAPTERS, "gdelt_doc", broken)
    conn.execute("UPDATE sources SET cursor='2026-09-26T00:00:00Z', last_success_at='2026-09-26T00:00:00Z'")
    result = run_source(conn, settings, "gdelt_floods")
    assert result["status"] == status
    row = source(conn)
    assert row["last_success_at"] == "2026-09-26T00:00:00Z" and row["cursor"] == "2026-09-26T00:00:00Z"
    assert row["last_error"]


def test_other_sources_survive_a_broken_one(conn, settings, monkeypatch):
    def broken(*a, **k):
        raise ValueError("adapter bug")
    monkeypatch.setitem(pipeline.ADAPTERS, "gdelt_doc", broken)
    monkeypatch.setitem(pipeline.ADAPTERS, "gdacs", lambda f, c, cur: ([], "2026-09-26T10:00:00Z"))
    assert run_source(conn, settings, "gdelt_floods")["status"] == "failed"
    assert run_source(conn, settings, "gdacs_floods")["status"] == "ok"
    assert source(conn, "gdacs_floods")["last_success_at"]


def test_cursor_not_moved_when_cycle_truncated(conn, settings, monkeypatch):
    from dataclasses import replace
    items = [news(str(i), f"Floods hit Kerala report number {i}", domain=f"d{i}.example") for i in range(5)]
    monkeypatch.setitem(pipeline.ADAPTERS, "gdelt_doc", lambda f, c, cur: (items, "2026-09-26T10:00:00Z"))
    result = run_source(conn, replace(settings, max_items_per_cycle=2), "gdelt_floods")
    assert result["items_new"] == 2 and "окно не сдвинуто" in result["error"]
    assert source(conn)["cursor"] is None


def test_data_survives_restart(settings):
    with closing(connect(settings.db_path)) as c:
        from floodvalue.media.config import load_sources
        from floodvalue.media.store import sync_sources
        sync_sources(c, load_sources())
        ingest(c, news("1", "Floods hit Kerala, thousands evacuated"))
    with closing(connect(settings.db_path)) as c:
        assert c.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 1
        assert c.execute("SELECT COUNT(*) FROM publications").fetchone()[0] == 1
