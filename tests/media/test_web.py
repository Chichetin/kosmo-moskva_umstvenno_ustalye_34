"""HTTP scenarios from the prompt (section 15): submit with/without photo, duplicate protection,
invalid input, status only by token, closed attachments, operator actions need authorisation,
moderation puts the event on the layer."""

from dataclasses import replace

import pytest

from floodvalue.media.web import create_app

from .conftest import LOCAL, REMOTE, call
from .test_events_reports import jpeg_with_exif

FORM = {"category": "flooding", "description": "Вода подошла к домам на улице Речной",
        "lat": "55.75", "lon": "37.62", "location_precision": "point",
        "observed_precision": "unknown", "idempotency_key": "a" * 24}
W = {"X-FloodValue-Media": "1"}


@pytest.fixture
def app(settings):
    return create_app(settings)


@pytest.fixture
def token_app(settings):
    return create_app(replace(settings, analyst_token="s3cret-token-for-tests"))  # secret-scan: allow — test value


def test_pages_are_separate_and_locked_down(app):
    r = call(app, "GET", "/report")
    assert r.status == 200 and "Это не вызов экстренных служб" in r.body.decode()
    assert "frame-ancestors 'none'" in r.headers["content-security-policy"]
    assert "analyst" not in r.body.decode()  # the public page does not link the analyst side


def test_submit_without_and_with_photo(app, settings):
    r = call(app, "POST", "/api/media/reports", data=FORM)
    assert r.status == 201, r.body
    token = r.json()["status_token"]
    r2 = call(app, "POST", "/api/media/reports", data={**FORM, "idempotency_key": "b" * 24},
              files={"photos": ("my_home.jpg", jpeg_with_exif(), "image/jpeg")})
    assert r2.status == 201, r2.body
    s = call(app, "POST", "/api/media/reports/status", json_body={"token": token}).json()
    assert s == {"report_id": r.json()["report_id"], "submitted_at": s["submitted_at"],
                 "status": "На проверке у аналитика"}
    assert not any(p.name == "my_home.jpg" for p in settings.photo_dir.iterdir())


def test_duplicate_submit_returns_same_report(app):
    a = call(app, "POST", "/api/media/reports", data=FORM)
    b = call(app, "POST", "/api/media/reports", data=FORM)
    assert (a.status, b.status) == (201, 200)
    assert a.json()["report_id"] == b.json()["report_id"]


@pytest.mark.parametrize("patch, field", [
    ({"lat": "91"}, "lat"), ({"lat": "abc"}, "lat"), ({"observed_at": "2099-01-01T00:00:00+03:00"}, ""),
    ({"observed_at": "2026-09-26T10:00:00"}, ""), ({"category": "<script>"}, "category")])
def test_invalid_input_is_rejected_with_field(app, patch, field):
    r = call(app, "POST", "/api/media/reports", data={**FORM, **patch})
    assert r.status == 422 and field in r.json()["field"]


def test_fake_image_rejected(app):
    r = call(app, "POST", "/api/media/reports", data=FORM,
             files={"photos": ("x.jpg", b"<html><script>alert(1)</script>", "image/jpeg")})
    assert r.status == 422 and r.json()["field"] == "photos"


def test_rate_limit(settings):
    app = create_app(replace(settings, reports_per_hour=2))
    codes = [call(app, "POST", "/api/media/reports", data={**FORM, "idempotency_key": str(i) * 20}).status
             for i in range(3)]
    assert codes == [201, 201, 429]


def test_public_intake_closed_for_internet_by_default(app):
    assert call(app, "POST", "/api/media/reports", data=FORM, client=REMOTE).status == 403
    assert call(app, "GET", "/api/media/report-config", client=REMOTE).json()["accepting"] is False


def test_status_needs_token_not_id(app):
    rid = call(app, "POST", "/api/media/reports", data=FORM).json()["report_id"]
    assert call(app, "POST", "/api/media/reports/status", json_body={"token": rid}).status == 404


def test_operator_actions_closed_to_anonymous(app, token_app):
    # no token configured: only loopback
    assert call(app, "GET", "/api/media/reports", client=REMOTE).status == 403
    assert call(app, "GET", "/api/media/reports", client=LOCAL, headers={"X-Forwarded-For": "1.2.3.4"}).status == 403
    # token configured: loopback is not enough, token or session cookie is
    assert call(token_app, "GET", "/api/media/events").status == 401
    assert call(token_app, "GET", "/api/media/events",
                headers={"Authorization": "Bearer wrong"}).status == 401
    ok = call(token_app, "GET", "/api/media/events",
              headers={"Authorization": "Bearer s3cret-token-for-tests"})
    assert ok.status == 200
    # write needs the custom header too (cross-site forms can't send it)
    assert call(token_app, "POST", "/api/media/monitoring", json_body={"enabled": False},
                headers={"Authorization": "Bearer s3cret-token-for-tests"}).status == 403


def test_login_cookie(token_app):
    bad = call(token_app, "POST", "/media/login", data={"token": "nope"})
    assert bad.status == 401
    good = call(token_app, "POST", "/media/login", data={"token": "s3cret-token-for-tests"})  # secret-scan: allow — test value
    assert good.status == 303
    cookie = good.headers["set-cookie"]
    assert "HttpOnly" in cookie and "SameSite=strict" in cookie
    session = cookie.split(";")[0]
    assert call(token_app, "GET", "/api/media/whoami", cookies=session).status == 200


def test_attachments_closed(app, token_app):
    call(token_app, "POST", "/api/media/reports", data=FORM,
         files={"photos": ("x.jpg", jpeg_with_exif(), "image/jpeg")})
    rep = call(token_app, "GET", "/api/media/reports",
               headers={"Authorization": "Bearer s3cret-token-for-tests"}).json()["reports"][0]
    path = f"/api/media/attachments/{rep['photos'][0]}"
    assert call(token_app, "GET", path).status == 401
    got = call(token_app, "GET", path, headers={"Authorization": "Bearer s3cret-token-for-tests"})
    assert got.status == 200 and got.headers["content-type"] == "image/jpeg"
    assert "sandbox" in got.headers["content-security-policy"]
    assert "status_token_hash" not in rep and "idempotency_key" not in rep


def test_moderation_puts_event_on_layer(app):
    rid = call(app, "POST", "/api/media/reports", data=FORM).json()["report_id"]
    assert call(app, "GET", "/api/media/events").json()["features"] == []  # pending ≠ published
    r = call(app, "POST", f"/api/media/reports/{rid}/moderate", json_body={"decision": "accept"}, headers=W)
    assert r.status == 200
    (f,) = call(app, "GET", "/api/media/events").json()["features"]
    assert f["geometry"] == {"type": "Point", "coordinates": [37.62, 55.75]}  # [lon, lat]
    assert f["properties"]["source_kinds"] == "citizen"
    card = call(app, "GET", f"/api/media/events/{f['id']}").json()
    assert card["event"]["verification_result"] == "not_checked"
    assert any("не проверял" in m for m in card["missing"])
    r = call(app, "POST", f"/api/media/events/{f['id']}/review",
              json_body={"verification_result": "consistent"}, headers=W)
    assert r.status == 409  # no analysed observation
    assert call(app, "GET", "/api/media/events?origin=news").json()["features"] == []


def test_sources_status_reported(app):
    s = call(app, "GET", "/api/media/sources").json()
    states = {x["source_id"]: x["state"] for x in s["sources"]}
    assert states["example_rss"] == "disabled" and states["gdacs_floods"] == "never"
    r = call(app, "POST", "/api/media/sources/example_rss/enabled", json_body={"enabled": True}, headers=W)
    assert r.status == 409  # terms not checked


def test_login_attempts_are_limited(token_app):
    codes = [call(token_app, "POST", "/media/login", data={"token": f"guess{i}"}).status for i in range(11)]
    assert codes[:10] == [401] * 10 and codes[10] == 429
    right = call(token_app, "POST", "/media/login", data={"token": "s3cret-token-for-tests"})  # secret-scan: allow — test value
    assert right.status == 429  # the limit holds even for the right token until the window passes


def test_malformed_requests_are_4xx(app):
    assert call(app, "POST", "/api/media/monitoring", json_body=[1, 2], headers=W).status == 422
    assert call(app, "POST", "/api/media/events/merge", json_body={"source_id": "x", "target_id": 1},
                headers=W).status == 422
    assert call(app, "POST", "/api/media/reports", data=FORM, headers={"content-length": "abc"}).status == 413


def test_local_mode_refuses_foreign_host(app):
    assert call(app, "GET", "/api/media/events", headers={"host": "evil.example"}).status == 400
    assert call(app, "GET", "/api/media/events", headers={"host": "127.0.0.1:8765"}).status == 200


def test_tooltips_never_get_html_strings():
    import re
    from pathlib import Path
    js_path = Path(__file__).resolve().parents[2] / "src/floodvalue/media/static/analyst.js"
    js = js_path.read_text(encoding="utf-8")
    assert ".innerHTML" not in js and "insertAdjacentHTML" not in js
    for call_args in re.findall(r"\.bind(?:Tooltip|Popup)\(([^)]{0,40})", js):
        assert call_args.startswith("tipNode("), call_args
