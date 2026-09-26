"""Citizen intake → moderation queue → event registry, through the HTTP API (prompt §15)."""

import io
import json
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from floodvalue.media import api, store

TOKEN = "test-operator-token-123"
ACTION = {api.ACTION_HEADER: "1"}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("FLOODVALUE_MEDIA_DIR", str(tmp_path / "media"))
    monkeypatch.setenv("FLOODVALUE_OPERATOR_TOKEN", TOKEN)
    for name in ("submit_limit", "status_limit", "login_limit"):
        old = getattr(api, name)
        monkeypatch.setattr(api, name, api.RateLimit(old.limit, old.window))
    app = FastAPI()
    app.include_router(api.router)
    return TestClient(app)


def _report(**overrides):
    body = {
        "client_key": str(uuid.uuid4()), "kind": "flooded_building",
        "description": "Двор затоплен по щиколотку, вода со стороны реки.",
        "lat": 7.05, "lon": 80.12, "location_precision": "point",
        "observed_at": (datetime.now(UTC) - timedelta(hours=2)).isoformat(),
        "observed_time_precision": "approx_hour", "observed_tz": "Asia/Colombo",
    }
    body.update(overrides)
    return body


def _photo_with_gps() -> bytes:
    exif = Image.Exif()
    exif[0x010F] = "SecretCameraMaker"
    exif[0x8825] = {2: (55.0, 45.0, 0.0)}
    buffer = io.BytesIO()
    Image.new("RGB", (80, 60), (20, 80, 140)).save(buffer, "JPEG", exif=exif.tobytes())
    return buffer.getvalue()


def _submit(client, report=None, photos=()):
    files = [("photos", (f"p{i}.jpg", data, "image/jpeg")) for i, data in enumerate(photos)]
    return client.post("/api/public/reports",
                       data={"report": json.dumps(report or _report())}, files=files or None)


def _login(client):
    response = client.post("/api/ops/login", json={"token": TOKEN, "name": "Аналитик 1"})
    assert response.status_code == 200
    return client


def test_submit_without_photo_and_retry_is_not_duplicated(client):
    report = _report()
    first = _submit(client, report)
    assert first.status_code == 201, first.text
    receipt = first.json()
    assert receipt["public_id"].startswith("FV-") and not receipt["duplicate"]
    again = _submit(client, report)
    assert again.status_code == 200
    assert again.json()["public_id"] == receipt["public_id"]
    assert again.json()["status_token"] == receipt["status_token"]
    status = client.post("/api/public/reports/status", json={"token": receipt["status_token"]})
    assert status.json()["status"] == "pending"
    assert "description" not in status.json() and "lat" not in status.json()
    with store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM citizen_report").fetchone()[0] == 1


def test_photo_is_private_and_stripped(client):
    receipt = _submit(client, photos=[_photo_with_gps()]).json()
    _login(client)
    queue = client.get("/api/ops/reports").json()
    url = queue[0]["photos"][0]["url"]
    photo = client.get(url)
    assert photo.status_code == 200 and photo.headers["content-type"] == "image/jpeg"
    assert b"SecretCameraMaker" not in photo.content
    assert photo.headers["x-content-type-options"] == "nosniff"
    anonymous = TestClient(client.app)
    assert anonymous.get(url).status_code == 401
    assert receipt["public_id"] == queue[0]["public_id"]


@pytest.mark.parametrize("overrides", [
    {"lat": 123.0}, {"lat": 0.0, "lon": 0.0}, {"lat": None, "lon": None},
    {"observed_at": (datetime.now(UTC) + timedelta(days=1)).isoformat()},
])
def test_invalid_input_rejected(client, overrides):
    response = _submit(client, _report(**overrides))
    assert response.status_code == 422
    assert response.json()["detail"]


def test_bad_photo_rejected_without_partial_report(client):
    response = _submit(client, photos=[b"<script>alert(1)</script>"])
    assert response.status_code == 422
    assert "Фото 1" in response.json()["detail"]
    with store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM citizen_report").fetchone()[0] == 0
    assert not list(store.attachments_dir().iterdir())


def test_too_many_photos_rejected(client):
    response = _submit(client, photos=[_photo_with_gps()] * 4)
    assert response.status_code == 400


def test_anonymous_cannot_touch_operator_endpoints(client):
    _submit(client)
    for method, url in [("get", "/api/ops/reports"), ("get", "/api/ops/events"),
                        ("post", "/api/ops/reports/1/review"), ("get", "/api/ops/summary")]:
        assert getattr(client, method)(url).status_code == 401
    assert client.post("/api/ops/login", json={"token": "wrong-token-000"}).status_code == 401


def test_operator_action_needs_interface_header(client):
    _submit(client)
    _login(client)
    response = client.post("/api/ops/reports/1/review", json={"decision": "reject"})
    assert response.status_code == 403


def test_approved_report_becomes_event_but_not_confirmed_flood(client):
    token = _submit(client).json()["status_token"]
    _login(client)
    report_id = client.get("/api/ops/reports").json()[0]["report_id"]
    result = client.post(f"/api/ops/reports/{report_id}/review",
                         json={"decision": "approve"}, headers=ACTION).json()
    events = client.get("/api/ops/events").json()
    assert [item["event_id"] for item in events] == [result["event_id"]]
    event = events[0]
    assert event["evidence_counts"]["citizen"] == 1
    assert event["geo_status"] == "resolved" and event["review_state"] == "new"
    assert event["verification_result"] == "not_checked"
    status = client.post("/api/public/reports/status", json={"token": token}).json()
    assert status["status"] == "approved"
    assert "не подтверждение" in status["status_text"]
    card = client.get(f"/api/ops/events/{event['event_id']}").json()
    assert card["evidence"][0]["report"]["description"].startswith("Двор затоплен")
    assert any("Нет подходящего спутникового наблюдения" in gap for gap in card["gaps"])


def test_report_without_coordinates_waits_for_place_then_history_kept(client):
    _submit(client, _report(lat=None, lon=None, place_text="Ратнапура, у рынка"))
    _login(client)
    report_id = client.get("/api/ops/reports").json()[0]["report_id"]
    event_id = client.post(f"/api/ops/reports/{report_id}/review",
                           json={"decision": "approve"}, headers=ACTION).json()["event_id"]
    event = client.get(f"/api/ops/events/{event_id}").json()["event"]
    assert (event["lat"], event["lon"]) == (None, None)
    assert event["geo_status"] == "missing" and event["review_state"] == "needs_geo"
    edited = client.patch(f"/api/ops/events/{event_id}",
                          json={"lat": 6.6828, "lon": 80.3992, "geo_precision": "locality",
                                "note": "по названию рынка"}, headers=ACTION).json()
    assert edited["event"]["review_state"] == "ready"
    assert edited["history"][-1]["action"] == "edited"
    assert json.loads(edited["history"][-1]["before_json"])["lat"] is None


def test_consistent_result_needs_suitable_observation(client):
    _submit(client)
    _login(client)
    event_id = client.post("/api/ops/reports/1/review", json={"decision": "approve"},
                           headers=ACTION).json()["event_id"]
    blocked = client.post(f"/api/ops/events/{event_id}/review",
                          json={"action": "result", "verification_result": "consistent"},
                          headers=ACTION)
    assert blocked.status_code == 409
    allowed = client.post(f"/api/ops/events/{event_id}/review",
                          json={"action": "result", "verification_result": "insufficient_data",
                                "follow_up": "needs_imagery"}, headers=ACTION).json()
    assert allowed["event"]["review_state"] == "reviewed"
    assert allowed["event"]["follow_up"] == "needs_imagery"
    assert allowed["reviews"][-1]["reviewer"] == "Аналитик 1"


def test_merge_and_split_keep_history(client):
    _submit(client)
    _submit(client, _report(lat=7.06, lon=80.13, kind="water_on_road"))
    _login(client)
    first = client.post("/api/ops/reports/1/review", json={"decision": "approve"},
                        headers=ACTION).json()["event_id"]
    candidates = client.get("/api/ops/reports/2/candidates").json()
    assert candidates[0]["event_id"] == first
    second = client.post("/api/ops/reports/2/review", json={"decision": "approve"},
                         headers=ACTION).json()["event_id"]
    merged = client.post(f"/api/ops/events/{second}/merge", json={"into_event_id": first},
                         headers=ACTION).json()
    assert merged["event"]["evidence_counts"]["citizen"] == 2
    assert merged["event"]["independent_origin_count"] == 2
    assert [item["event_id"] for item in client.get("/api/ops/events").json()] == [first]
    evidence_id = merged["evidence"][1]["evidence_id"]
    split = client.post(f"/api/ops/events/{first}/split", json={"evidence_ids": [evidence_id]},
                        headers=ACTION).json()
    actions = [item["action"] for item in client.get(f"/api/ops/events/{first}").json()["history"]]
    assert actions[-2:] == ["merged_from", "split"]
    new_card = client.get(f"/api/ops/events/{split['new_event_id']}").json()
    assert new_card["event"]["evidence_counts"]["citizen"] == 1
    assert new_card["evidence"][0]["link_kind"] == "manual_split"


def test_sender_can_delete_report_and_photos(client):
    token = _submit(client, photos=[_photo_with_gps()]).json()["status_token"]
    assert len(list(store.attachments_dir().iterdir())) == 1
    assert client.post("/api/public/reports/delete", json={"token": token}).status_code == 200
    assert not list(store.attachments_dir().iterdir())
    assert client.post("/api/public/reports/status", json={"token": token}).status_code == 404


def test_other_token_cannot_read_report(client):
    _submit(client)
    guess = "0" * 40
    assert client.post("/api/public/reports/status", json={"token": guess}).status_code == 404
    assert client.post("/api/public/reports/status", json={"token": "1"}).status_code == 422


def test_data_survives_restart(client):
    receipt = _submit(client).json()
    store._initialised.clear()  # a fresh process opens the same file
    status = client.post("/api/public/reports/status", json={"token": receipt["status_token"]})
    assert status.json()["public_id"] == receipt["public_id"]


def test_submission_rate_limited(client):
    codes = [_submit(client).status_code for _ in range(6)]
    assert codes == [201] * 5 + [429]
