"""Ingestion cycle on saved responses, no network (prompt §15, plan examples 6, 8, 10).

Fixture texts are invented; they are not real events.
"""

import fcntl
import hashlib
import json
import socket
from pathlib import Path
from typing import ClassVar

import pytest

from floodvalue.media import geo, ingest, net, sources
from floodvalue.media.net import Fetched, FetchError
from floodvalue.media.store import connect, media_dir

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "media"
CONFIG = {
    "monitoring": {"enabled": True, "interval_s": 900, "max_items_per_source": 40,
                   "requests_budget": 30, "geocode_budget": 10, "llm_budget": 0},
    "sources": [
        {"source_id": "gdacs", "name": "GDACS", "kind": "bulletin", "adapter": "gdacs",
         "url": "https://www.gdacs.org/gdacsapi/api/events/geteventlist/SEARCH",
         "enabled": True, "terms_status": "checked", "allowed_hosts": ["www.gdacs.org"],
         "content_storage_policy": "link_title_excerpt"},
        {"source_id": "feed", "name": "Test feed", "kind": "rss", "adapter": "rss",
         "url": "https://news.example.org/rss.xml", "language": "en", "enabled": True,
         "terms_status": "checked", "allowed_hosts": ["news.example.org"],
         "content_storage_policy": "link_title_excerpt"},
        {"source_id": "gdelt", "name": "GDELT", "kind": "api", "adapter": "gdelt",
         "url": "https://api.gdeltproject.org/api/v2/doc/doc", "query": "flood",
         "enabled": True, "terms_status": "checked", "allowed_hosts": ["api.gdeltproject.org"],
         "content_storage_policy": "link_title"},
        {"source_id": "unchecked", "name": "Unchecked", "kind": "rss", "adapter": "rss",
         "url": "https://x.example/rss", "enabled": True, "terms_status": "unchecked",
         "allowed_hosts": ["x.example"]},
    ],
}
PLACES = {
    "kerala": [{"label": "Kerala, India", "lat": 10.35, "lon": 76.51, "precision": "region",
                "bbox": [74.85, 8.18, 77.42, 12.79], "country_code": "in", "importance": 0.7}],
    "tamil nadu": [{"label": "Tamil Nadu, India", "lat": 10.91, "lon": 78.36,
                    "precision": "region", "bbox": [76.23, 8.07, 80.35, 13.56],
                    "country_code": "in", "importance": 0.7}],
}


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("FLOODVALUE_MEDIA_DIR", str(tmp_path / "media"))
    outputs = tmp_path / "outputs" / "run"
    outputs.mkdir(parents=True)
    (outputs / "flood_probability.tif").write_bytes(b"model output stand-in")
    monkeypatch.setenv("FLOODVALUE_OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setattr(ingest, "load_config", lambda: CONFIG)
    responses = {
        "www.gdacs.org": (FIXTURES / "gdacs_fl.json").read_bytes(),
        "news.example.org": (FIXTURES / "feed.xml").read_bytes(),
        "api.gdeltproject.org": (FIXTURES / "gdelt_artlist.json").read_bytes(),
    }
    failures: dict[str, FetchError] = {}

    def fake_fetch(url, *, allowed_hosts, params=None, min_interval=0.0, **_):
        host = url.split("/")[2]
        assert host in allowed_hosts
        if host in failures:
            raise failures[host]
        return Fetched(url, 200, responses[host], "application/octet-stream")

    def no_network(*_, **__):
        raise AssertionError("geocoder must answer from cache in tests")

    monkeypatch.setattr(sources, "fetch", fake_fetch)
    monkeypatch.setattr(geo, "fetch", no_network)
    with connect() as db:
        for name, result in PLACES.items():
            db.execute("INSERT INTO geocode_cache(query, result_json, fetched_at) "
                       "VALUES (?, ?, '2026-09-26T00:00:00Z')", (f"ru|{name}", json.dumps(result)))
    return {"responses": responses, "failures": failures, "outputs": outputs}


def _events(db):
    return {row["event_id"]: dict(row) for row in db.execute("SELECT * FROM event")}


def test_cycle_builds_events_and_reimport_adds_nothing(env):
    first = ingest.run_cycle("cli")
    assert first["status"] == "ok", first
    with connect() as db:
        events = _events(db)
        kinds = sorted((row["report_type"], row["place_label"]) for row in events.values())
        assert kinds == [("forecast", "Kerala, India"), ("occurred", "Kerala, India"),
                         ("occurred", "Tamil Nadu, India"), ("warning", "Testland (центр области GDACS)")]
        publications = db.execute("SELECT COUNT(*) FROM publication").fetchone()[0]
        assert publications == 1 + 5 + 2
        assert db.execute("SELECT enabled FROM source WHERE source_id = 'unchecked'").fetchone()[0] == 0
    second = ingest.run_cycle("cli")
    assert (second["items_new"], second["events_new"]) == (0, 0)
    with connect() as db:
        assert db.execute("SELECT COUNT(*) FROM publication").fetchone()[0] == publications
        assert len(_events(db)) == 4


def test_reprints_are_not_independent_confirmations(env):
    ingest.run_cycle("cli")
    with connect() as db:
        kerala = db.execute("SELECT event_id FROM event WHERE report_type = 'occurred' "
                            "AND place_label LIKE 'Kerala%'").fetchone()[0]
        rows = db.execute("SELECT e.is_original, p.domain, p.report_type FROM evidence e "
                          "JOIN publication p USING (publication_id) WHERE e.event_id = ?",
                          (kerala,)).fetchall()
        from floodvalue.media.events import independent_origin_count

        reprints = [row for row in rows if not row["is_original"]]
        assert {row["domain"] for row in reprints} == {"one.example.net", "two.example.com"}
        # the negation attaches to the event but does not create one
        assert "negation" in {row["report_type"] for row in rows}
        assert independent_origin_count(db, kerala) == 1


def test_one_source_down_others_continue(env):
    env["failures"]["api.gdeltproject.org"] = FetchError("HTTP 503: unavailable", 503)
    result = ingest.run_cycle("cli")
    assert result["status"] == "partial"
    with connect() as db:
        gdelt = db.execute("SELECT * FROM source WHERE source_id = 'gdelt'").fetchone()
        assert gdelt["last_success_at"] is None and "503" in gdelt["last_error"]
        run = db.execute("SELECT status, http_status FROM source_run WHERE source_id = 'gdelt'"
                         ).fetchone()
        assert (run["status"], run["http_status"]) == ("failed", 503)
        assert db.execute("SELECT last_success_at FROM source WHERE source_id = 'feed'"
                          ).fetchone()[0] is not None


def test_rate_limit_is_not_no_news(env):
    env["failures"]["api.gdeltproject.org"] = FetchError("HTTP 429: limit", 429)
    ingest.run_cycle("cli")
    with connect() as db:
        run = db.execute("SELECT status FROM source_run WHERE source_id = 'gdelt'").fetchone()
        assert run["status"] == "skipped_rate_limit"


def test_updated_publication_is_same_record_with_history(env):
    ingest.run_cycle("cli")
    env["responses"]["news.example.org"] = env["responses"]["news.example.org"].replace(
        b"thousands evacuated", b"tens of thousands evacuated")
    second = ingest.run_cycle("cli")
    assert second["items_new"] == 0
    with connect() as db:
        title = db.execute("SELECT title FROM publication WHERE external_id = 'a1'").fetchone()[0]
        assert "tens of thousands" in title
        actions = [row[0] for row in db.execute("SELECT action FROM event_history")]
        assert "source_updated" in actions


def test_second_cycle_waits_for_lock(env):
    with open(media_dir() / "ingest.lock", "w") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        assert ingest.run_cycle("cli")["status"] == "skipped"


def test_feed_excerpt_is_plain_text_and_url_canonical(env):
    ingest.run_cycle("cli")
    with connect() as db:
        row = db.execute("SELECT url, text_excerpt FROM publication WHERE external_id = 'a1'"
                         ).fetchone()
    assert row["url"] == "https://news.example.org/a1"
    assert "<" not in row["text_excerpt"] and "Rivers burst their banks" in row["text_excerpt"]


def test_model_outputs_untouched(env):
    target = env["outputs"] / "flood_probability.tif"
    before = hashlib.sha256(target.read_bytes()).hexdigest()
    ingest.run_cycle("cli")
    assert hashlib.sha256(target.read_bytes()).hexdigest() == before
    assert sorted(path.name for path in env["outputs"].iterdir()) == ["flood_probability.tif"]


def test_loader_refuses_internal_and_foreign_hosts(monkeypatch):
    addresses = {"good.example": "93.184.216.34", "evil.example": "169.254.169.254",
                 "localhost": "127.0.0.1"}

    def fake_getaddrinfo(host, port, *_, **__):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (addresses[host], port))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    for url, hosts in [("http://localhost/", {"localhost"}),
                       ("https://evil.example/", {"evil.example"}),
                       ("https://other.example/", {"good.example"}),
                       ("file:///etc/passwd", {"good.example"})]:
        with pytest.raises(FetchError):
            net.check_url(url, hosts)
    assert net.check_url("https://good.example/feed", {"good.example"}) == "good.example"

    class Redirect:
        is_redirect, status_code = True, 302
        headers: ClassVar[dict] = {"Location": "http://evil.example/latest/meta-data/"}

        def close(self):
            pass

    class Session:
        def get(self, *_, **__):
            return Redirect()

    monkeypatch.setattr(net, "throttle", lambda *_: None)
    with pytest.raises(FetchError, match="не в списке"):
        net.fetch("https://good.example/feed", allowed_hosts={"good.example"},
                  session=Session(), attempts=1)
