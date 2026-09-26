"""Shared fixtures for floodvalue.media: an isolated media dir, settings, a raw ASGI caller
(starlette's TestClient needs httpx2, which the lockfile does not carry)."""

from __future__ import annotations

import asyncio
import json
from contextlib import closing
from dataclasses import replace
from pathlib import Path

import pytest
import requests

from floodvalue.media.config import load_settings, load_sources
from floodvalue.media.store import connect, sync_sources

FIXTURES = Path(__file__).parent / "fixtures"
LOCAL = ("127.0.0.1", 50000)
REMOTE = ("203.0.113.7", 50000)


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.setenv("FLOODVALUE_MEDIA_DIR", str(tmp_path / "media"))
    for name in ("FLOODVALUE_MEDIA_ANALYST_TOKEN", "FLOODVALUE_MEDIA_PUBLIC_REPORTS",
                 "FLOODVALUE_MEDIA_GEOCODER", "FLOODVALUE_MEDIA_POLL_INTERVAL_S"):
        monkeypatch.delenv(name, raising=False)
    return replace(load_settings(), reports_per_hour=1000)


@pytest.fixture
def conn(settings):
    with closing(connect(settings.db_path)) as c:
        sync_sources(c, load_sources())
        yield c


class Response:
    def __init__(self, status, headers, body):
        self.status, self.headers, self.body = status, headers, body

    def json(self):
        return json.loads(self.body)


def call(app, method: str, path: str, *, json_body=None, data=None, files=None, headers=None,
         client=LOCAL, cookies: str | None = None) -> Response:
    """One HTTP request straight into the ASGI app."""
    hdrs = {k.lower(): v for k, v in (headers or {}).items()}
    body = b""
    if json_body is not None:
        body = json.dumps(json_body).encode()
        hdrs["content-type"] = "application/json"
    elif data is not None or files is not None:
        prepared = requests.Request("POST", "http://x/", data=data, files=files).prepare()
        body = prepared.body if isinstance(prepared.body, bytes) else prepared.body.encode()
        hdrs["content-type"] = prepared.headers["Content-Type"]
    if method != "GET":
        hdrs.setdefault("content-length", str(len(body)))
    if cookies:
        hdrs["cookie"] = cookies
    raw_path, _, query = path.partition("?")
    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": method,
             "scheme": "http", "path": raw_path, "raw_path": raw_path.encode(),
             "query_string": query.encode(), "root_path": "",
             "headers": [(k.encode(), v.encode()) for k, v in hdrs.items()],
             "client": client, "server": ("testserver", 80)}
    sent = {"status": None, "headers": [], "body": b""}
    chunks = [body]

    async def receive():
        if chunks:
            return {"type": "http.request", "body": chunks.pop(), "more_body": False}
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.start":
            sent["status"] = message["status"]
            sent["headers"] = [(k.decode().lower(), v.decode()) for k, v in message["headers"]]
        elif message["type"] == "http.response.body":
            sent["body"] += message.get("body", b"")

    asyncio.run(app(scope, receive, send))
    return Response(sent["status"], dict(sent["headers"]), sent["body"])
