"""One origin (#174): scripts/serve_site.py routes by path to the operator app or to media, refuses
to go public without an analyst token; /api/media/map gives the operator layer without personal data."""

import importlib.util
from dataclasses import replace

import pytest

from floodvalue.media.web import create_app
from floodvalue.paths import REPO_ROOT

from .conftest import LOCAL, REMOTE, call

spec = importlib.util.spec_from_file_location("serve_site", REPO_ROOT / "scripts" / "serve_site.py")
serve_site = importlib.util.module_from_spec(spec)
spec.loader.exec_module(serve_site)

TOKEN = "s3cret-token-for-tests"  # secret-scan: allow — test value
FORM = {"category": "flooding", "description": "Вода у дома 5, звоните +7 900 000-00-00",
        "lat": "55.751234", "lon": "37.618765", "location_precision": "point",
        "observed_precision": "unknown", "place_text": "ул. Речная, 5, кв. 12",
        "idempotency_key": "b" * 24}
PRIVATE = ("+7 900", "Вода у дома", "Речная", "37.618765", "55.751234")


@pytest.mark.parametrize("path,media", [
    ("/report", True), ("/report/status", True), ("/media/analyst", True),
    ("/media/static/report.js", True), ("/api/media/map", True), ("/api/media", True),
    ("/", False), ("/api/scenes", False), ("/api/plan", False), ("/reports", False),
    ("/mediainfo", False), ("/api/mediax", False), ("/assets/index.js", False),
])
def test_paths_split_between_operator_and_media(path, media):
    assert serve_site.is_media(path) is media


def test_one_origin_serves_both_apps(settings, monkeypatch):
    monkeypatch.setenv("FLOODVALUE_APP_RUN_DIR", str(REPO_ROOT / "tests" / "fixtures" / "good"))
    site = serve_site.build(settings)
    assert call(site, "GET", "/api/scenes").status == 200
    report = call(site, "GET", "/report")
    assert report.status == 200 and "Это не вызов экстренных служб" in report.body.decode()
    assert call(site, "GET", "/api/media/report-config").json()["accepting"] is True


@pytest.mark.parametrize("host,env,refused", [
    ("127.0.0.1", {}, False),
    ("0.0.0.0", {}, True),
    ("0.0.0.0", {"FLOODVALUE_MEDIA_ANALYST_TOKEN": TOKEN}, False),
    ("127.0.0.1", {"FLOODVALUE_SITE_PUBLIC": "1"}, True),
    ("127.0.0.1", {"FLOODVALUE_SITE_PUBLIC": "1", "FLOODVALUE_MEDIA_ANALYST_TOKEN": TOKEN}, False),
    ("example.org", {}, True),
])
def test_public_site_needs_analyst_token(host, env, refused):
    assert (serve_site.public_problem(host, env) is not None) is refused


def _submit(app):
    r = call(app, "POST", "/api/media/reports", data=FORM)
    assert r.status == 201, r.body
    return r.json()["report_id"]


def test_map_is_closed_by_default(settings):
    app = create_app(replace(settings, analyst_token=TOKEN))
    assert call(app, "GET", "/api/media/map", client=REMOTE).status == 401
    local = create_app(settings)
    assert call(local, "GET", "/api/media/map", client=REMOTE).status == 403


def test_map_for_analyst_has_exact_point_and_no_personal_data(settings):
    app = create_app(replace(settings, analyst_token=TOKEN, public_reports=True))
    rid = _submit(app)
    r = call(app, "GET", "/api/media/map", client=REMOTE,
             headers={"Authorization": f"Bearer {TOKEN}"})
    assert r.status == 200 and r.json()["exact_locations"] is True
    (feature,) = r.json()["reports"]["features"]
    assert feature["id"] == rid
    assert feature["geometry"]["coordinates"] == [37.618765, 55.751234]
    body = r.body.decode()
    for secret in ("+7 900", "Вода у дома", "status_token", "idempotency"):
        assert secret not in body


def test_public_map_coarsens_reports_and_hides_moderated(settings):
    app = create_app(replace(settings, analyst_token=TOKEN, public_reports=True,
                             operator_map_public=True))
    rid = _submit(app)
    r = call(app, "GET", "/api/media/map", client=REMOTE)
    assert r.status == 200 and r.json()["exact_locations"] is False
    (feature,) = r.json()["reports"]["features"]
    assert feature["geometry"]["coordinates"] == [37.62, 55.75]
    assert feature["properties"]["location_precision"] == "coarsened"
    auth = {"Authorization": f"Bearer {TOKEN}", "X-FloodValue-Media": "1"}
    moderated = call(app, "POST", f"/api/media/reports/{rid}/moderate",
                     json_body={"decision": "accept"}, headers=auth, client=REMOTE)
    assert moderated.status == 200, moderated.body
    analyst_view = call(app, "GET", "/api/media/map", client=LOCAL,
                        headers={"Authorization": f"Bearer {TOKEN}"}).json()
    assert analyst_view["reports"]["features"] == []  # accepted → it is an event now
    (event,) = analyst_view["events"]["features"]
    assert event["id"] == moderated.json()["event_id"]
    assert event["geometry"]["coordinates"] == [37.618765, 55.751234]

    # The event took the eyewitness's text, address and point on accept: the public layer must not.
    public = call(app, "GET", "/api/media/map", client=REMOTE)
    assert public.json()["exact_locations"] is False
    (event,) = public.json()["events"]["features"]
    assert event["geometry"]["coordinates"] == [37.62, 55.75]
    assert event["properties"]["title"] == "Сообщение очевидца"
    assert event["properties"]["place_label"] is None
    assert event["properties"]["geo_precision"] == "coarsened"
    for private in PRIVATE:
        assert private not in public.body.decode()


def test_public_map_behind_proxy_is_not_exact(settings):
    """Behind Caddy every request comes from 127.0.0.1 with X-Forwarded-For: never «local analyst»."""
    app = create_app(replace(settings, public_reports=True, operator_map_public=True))
    _submit(app)
    r = call(app, "GET", "/api/media/map", client=LOCAL,
             headers={"X-Forwarded-For": "203.0.113.7"})
    assert r.status == 200 and r.json()["exact_locations"] is False
    for private in PRIVATE:
        assert private not in r.body.decode()


def test_public_map_hides_eyewitness_text_after_split(settings):
    """Accepted report → event «Очевидец: …»; a news item joins it; the analyst splits the eyewitness
    evidence off. The original event is news-only now but still holds the eyewitness's title, place
    and point — the public layer must not show them for any event (reviewer #175)."""
    from contextlib import closing

    from floodvalue.media import events as ev
    from floodvalue.media.store import connect, transaction

    app = create_app(replace(settings, analyst_token=TOKEN, public_reports=True,
                             operator_map_public=True))
    rid = _submit(app)
    auth = {"Authorization": f"Bearer {TOKEN}", "X-FloodValue-Media": "1"}
    eid = call(app, "POST", f"/api/media/reports/{rid}/moderate", json_body={"decision": "accept"},
               headers=auth, client=REMOTE).json()["event_id"]
    with closing(connect(settings.db_path)) as c, transaction(c):
        src = c.execute("SELECT source_id FROM sources WHERE kind='news' LIMIT 1").fetchone()[0]
        c.execute("INSERT INTO publications(source_id, external_id, url, canonical_url, title, "
                  "content_scope, fetched_at, content_hash, title_key, source_domain) "
                  "VALUES(?,?,?,?,?,?,?,?,?,?)", (src, "x", "http://n/1", "http://n/1",
                  "Подтопление в центре", "title", "2026-09-26T10:00:00Z", "h", "k", "n"))
        pid = c.execute("SELECT last_insert_rowid()").fetchone()[0]
        ev.link(c, eid, origin="news", is_original=True, link_kind="auto", actor="system",
                publication_id=pid, report_type="occurred", facts={})
        cit = c.execute("SELECT evidence_id FROM evidence WHERE event_id=? AND origin='citizen'",
                        (eid,)).fetchone()[0]
    split = call(app, "POST", f"/api/media/events/{eid}/split", json_body={"evidence_ids": [cit]},
                 headers=auth, client=REMOTE)
    assert split.status == 200, split.body
    public = call(app, "GET", "/api/media/map", client=REMOTE)
    events = public.json()["events"]["features"]
    assert len(events) == 2
    assert {e["properties"]["source_kinds"] for e in events} == {"news", "citizen"}
    for event in events:
        assert event["properties"]["place_label"] is None
        if event["geometry"]:  # the split-off part may have no place yet
            assert event["geometry"]["coordinates"] == [37.62, 55.75]
    for private in PRIVATE:
        assert private not in public.body.decode()
