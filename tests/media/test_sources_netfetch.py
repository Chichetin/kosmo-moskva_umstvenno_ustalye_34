"""Adapters parse saved real answers; outbound HTTP refuses what it must refuse. No network."""

import json
import socket

import pytest
import requests

from floodvalue.media import netfetch
from floodvalue.media.netfetch import Budget, BudgetExceeded, Fetcher, FetchError
from floodvalue.media.sources import canonical_url, fetch_gdacs, gdacs_item, gdelt_item, title_key

from .conftest import FIXTURES


class FakeResp:
    def __init__(self, status=200, body=b"", headers=None):
        self.status_code, self._content, self.headers = status, body, headers or {}
        self.is_redirect = status in (301, 302, 303, 307, 308)
        self.raw = self

    def read(self, n, decode_content=True):
        return self._content[:n]

    def close(self):
        pass

    def json(self):
        return json.loads(self._content)

    @property
    def text(self):
        return self._content.decode()

    @property
    def content(self):
        return self._content


def fetcher(hosts=("www.gdacs.org",), limit=10):
    f = Fetcher(allowed_hosts=frozenset(hosts), budget=Budget(limit), user_agent="test")
    f.sleep = lambda s: None
    return f


@pytest.fixture
def public_dns(monkeypatch):
    monkeypatch.setattr(netfetch.socket, "getaddrinfo",
                        lambda host, port, proto=0: [(socket.AF_INET, 1, 6, "", ("93.184.216.34", port))])


def test_gdacs_saved_answer_parses():
    data = json.loads((FIXTURES / "gdacs_live_2026-09-26.json").read_text())
    items = [gdacs_item(f) for f in data["features"]]
    assert items and all(i["structured"]["kind"] == "bulletin" for i in items)
    s = items[0]["structured"]
    assert s["event_key"].startswith("gdacs:FL:")
    lon, lat = s["place"]["geometry"]["coordinates"]
    assert -180 <= lon <= 180 and -90 <= lat <= 90
    assert s["place"]["precision"] == "region"  # a centroid, not an address


def test_gdacs_zero_point_is_not_a_place():
    f = {"properties": {"eventid": 1, "episodeid": 1, "name": "Flood in X", "country": "X"},
         "geometry": {"type": "Point", "coordinates": [0, 0]}}
    assert gdacs_item(f)["structured"]["place"]["geometry"] is None


def test_gdacs_204_is_an_answer_not_an_error(public_dns, monkeypatch):
    monkeypatch.setattr(netfetch.requests, "get", lambda *a, **k: FakeResp(204))
    items, cursor = fetch_gdacs(fetcher(), {"endpoint": "https://www.gdacs.org/x"}, None)
    assert items == [] and cursor


def test_gdacs_html_error_page_is_failure(public_dns, monkeypatch):
    monkeypatch.setattr(netfetch.requests, "get", lambda *a, **k: FakeResp(200, b"<html>maintenance"))
    with pytest.raises(FetchError):
        fetch_gdacs(fetcher(), {"endpoint": "https://www.gdacs.org/x"}, None)


def test_gdelt_item_keeps_headline_only_and_marks_time_basis():
    item = gdelt_item({"url": "https://news.example/a?utm_source=x", "title": "Floods hit <b>Kerala</b>",
                       "seendate": "20260926T081500Z", "domain": "news.example", "language": "English"})
    assert item["text_excerpt"] is None and item["title"] == "Floods hit Kerala"
    assert item["published_at"] == "2026-09-26T08:15:00Z" and item["time_basis"] == "gdelt_seendate"
    assert item["external_id"] == "https://news.example/a"


def test_canonical_url_and_title_key():
    assert canonical_url("HTTPS://News.Example/a/?utm_medium=x&b=2#frag") == "https://news.example/a?b=2"
    assert title_key("Floods  hit, KERALA!") == title_key("floods hit kerala")


def test_only_allowlisted_https(public_dns):
    f = fetcher()
    with pytest.raises(FetchError):
        f.check_url("http://www.gdacs.org/x")
    with pytest.raises(FetchError):
        f.check_url("https://evil.example/x")
    with pytest.raises(FetchError):
        f.check_url("https://user:pw@www.gdacs.org/x")


@pytest.mark.parametrize("ip", ["127.0.0.1", "10.0.0.5", "169.254.169.254", "192.168.1.1", "::1"])
def test_internal_addresses_refused(monkeypatch, ip):
    fam = socket.AF_INET6 if ":" in ip else socket.AF_INET
    monkeypatch.setattr(netfetch.socket, "getaddrinfo",
                        lambda host, port, proto=0: [(fam, 1, 6, "", (ip, port))])
    with pytest.raises(FetchError, match="внутренний"):
        fetcher().check_url("https://www.gdacs.org/x")


def test_redirect_to_other_host_refused(public_dns, monkeypatch):
    monkeypatch.setattr(netfetch.requests, "get",
                        lambda *a, **k: FakeResp(302, headers={"Location": "https://169.254.169.254/"}))
    with pytest.raises(FetchError, match="не в списке"):
        fetcher().get("https://www.gdacs.org/x")


def test_retries_with_backoff_then_fails(public_dns, monkeypatch):
    calls, sleeps = [], []
    monkeypatch.setattr(netfetch.requests, "get", lambda *a, **k: calls.append(1) or FakeResp(503))
    f = fetcher()
    f.sleep = sleeps.append
    with pytest.raises(FetchError) as exc:
        f.get("https://www.gdacs.org/x")
    assert exc.value.status == 503 and len(calls) == 3
    backoffs = [s for s in sleeps if s >= 2]
    assert backoffs == sorted(backoffs) and len(backoffs) == 2


def test_network_error_is_retryable_failure(public_dns, monkeypatch):
    def boom(*a, **k):
        raise requests.ConnectTimeout("slow")
    monkeypatch.setattr(netfetch.requests, "get", boom)
    with pytest.raises(FetchError, match="сеть"):
        fetcher().get("https://www.gdacs.org/x")


def test_budget_stops_requests(public_dns, monkeypatch):
    monkeypatch.setattr(netfetch.requests, "get", lambda *a, **k: FakeResp(200, b"{}"))
    f = fetcher(limit=1)
    f.get("https://www.gdacs.org/x")
    with pytest.raises(BudgetExceeded):
        f.get("https://www.gdacs.org/x")
