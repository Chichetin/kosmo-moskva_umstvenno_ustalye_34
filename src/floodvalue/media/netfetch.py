"""Outbound HTTP for sources, geocoder and catalogs: only allow-listed https hosts, never
private/loopback/link-local addresses, redirects re-checked hop by hop, timeouts, bounded
retries with growing pauses, a per-host minimum interval and a per-cycle request budget."""

from __future__ import annotations

import ipaddress
import socket
import time
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlsplit

import requests

MAX_BYTES = 5_000_000
MAX_REDIRECTS = 3
RETRY_STATUSES = {429, 500, 502, 503, 504}


class FetchError(Exception):
    def __init__(self, message: str, status: int | None = None, retryable: bool = False):
        super().__init__(message)
        self.status = status
        self.retryable = retryable


class BudgetExceeded(FetchError):
    pass


@dataclass
class Budget:
    """Requests allowed in one cycle; exceeding it stops the cycle instead of going on."""
    limit: int
    used: int = 0

    def take(self) -> None:
        if self.used >= self.limit:
            raise BudgetExceeded(f"бюджет внешних запросов исчерпан ({self.limit} за цикл)")
        self.used += 1


@dataclass
class Fetcher:
    allowed_hosts: frozenset[str]
    budget: Budget
    user_agent: str
    min_interval_s: dict[str, float] = field(default_factory=dict)
    timeout: tuple[float, float] = (10.0, 30.0)
    attempts: int = 3
    backoff_s: float = 2.0
    sleep = staticmethod(time.sleep)
    _last: dict[str, float] = field(default_factory=dict)
    last_status: int | None = None

    def check_url(self, url: str) -> str:
        parts = urlsplit(url)
        host = (parts.hostname or "").lower()
        if parts.scheme != "https":
            raise FetchError(f"разрешён только https: {url}")
        if host not in self.allowed_hosts:
            raise FetchError(f"хост не в списке разрешённых: {host}")
        if parts.username or parts.password or (parts.port not in (None, 443)):
            raise FetchError(f"недопустимый адрес: {url}")
        for info in socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP):
            ip = ipaddress.ip_address(info[4][0])
            if not ip.is_global:
                raise FetchError(f"{host} указывает на внутренний адрес {ip}")
        return host

    def _pace(self, host: str) -> None:
        gap = self.min_interval_s.get(host, 1.0)
        wait = self._last.get(host, 0.0) + gap - time.monotonic()
        if wait > 0:
            self.sleep(wait)
        self._last[host] = time.monotonic()

    def get(self, url: str, params: dict | None = None) -> requests.Response:
        """One logical request: retries count against the budget too."""
        delay = self.backoff_s
        for attempt in range(1, self.attempts + 1):
            try:
                return self._get_once(url, params)
            except FetchError as exc:
                if not exc.retryable or attempt == self.attempts or isinstance(exc, BudgetExceeded):
                    raise
                self.sleep(delay)
                delay *= 2
        raise AssertionError("unreachable")

    def _get_once(self, url: str, params: dict | None) -> requests.Response:
        for _ in range(MAX_REDIRECTS + 1):
            host = self.check_url(url)
            self.budget.take()
            self._pace(host)
            try:
                resp = requests.get(url, params=params, timeout=self.timeout, stream=True,
                                    allow_redirects=False,
                                    headers={"User-Agent": self.user_agent})
            except requests.RequestException as exc:
                raise FetchError(f"сеть: {type(exc).__name__}: {exc}", retryable=True) from exc
            self.last_status = resp.status_code
            if resp.is_redirect:
                url, params = urljoin(url, resp.headers.get("Location", "")), None
                resp.close()
                continue
            if resp.status_code in RETRY_STATUSES:
                resp.close()
                raise FetchError(f"HTTP {resp.status_code}", resp.status_code, retryable=True)
            if resp.status_code >= 400:
                resp.close()
                raise FetchError(f"HTTP {resp.status_code}", resp.status_code)
            body = resp.raw.read(MAX_BYTES + 1, decode_content=True)
            if len(body) > MAX_BYTES:
                raise FetchError(f"ответ больше {MAX_BYTES} байт")
            resp._content = body
            return resp
        raise FetchError("слишком много перенаправлений")
