"""Outbound HTTP for sources and the geocoder: allowlist, no internal addresses, retries."""

from __future__ import annotations

import fcntl
import ipaddress
import socket
import time
from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit

import requests

from floodvalue.media.store import media_dir

USER_AGENT = "FloodValue-MVP/0.1 (hackathon prototype; media monitoring)"
MAX_BYTES = 5 * 1024 * 1024
MAX_REDIRECTS = 3


class FetchError(RuntimeError):
    """A source did not answer usefully. Never to be read as 'no new events'."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


@dataclass
class Fetched:
    url: str
    status: int
    content: bytes
    content_type: str


def check_url(url: str, allowed_hosts: set[str]) -> str:
    """Reject anything but http(s) to an allowlisted host that resolves to public addresses."""
    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"}:
        raise FetchError(f"схема {parts.scheme!r} запрещена")
    host = (parts.hostname or "").lower().rstrip(".")
    if host not in allowed_hosts:
        raise FetchError(f"хост {host!r} не в списке разрешённых")
    try:
        infos = socket.getaddrinfo(host, parts.port or (443 if parts.scheme == "https" else 80),
                                   type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise FetchError(f"DNS: {host}: {exc}") from exc
    for info in infos:
        address = ipaddress.ip_address(info[4][0])
        if not address.is_global or address.is_multicast:
            raise FetchError(f"{host} указывает на внутренний адрес {address}")
    return host


def throttle(host: str, min_interval: float) -> None:
    """Keep at least `min_interval` seconds between requests to one host across processes."""
    if min_interval <= 0:
        return
    lock_path = media_dir() / f"throttle-{host}.lock"
    with open(lock_path, "a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        lock.seek(0)
        text = lock.read().strip()
        last = float(text) if text else 0.0
        wait = last + min_interval - time.time()
        if wait > 0:
            time.sleep(wait)
        lock.seek(0)
        lock.truncate()
        lock.write(str(time.time()))
        lock.flush()
        fcntl.flock(lock, fcntl.LOCK_UN)


def fetch(url: str, *, allowed_hosts: set[str], params: dict | None = None,
          min_interval: float = 0.0, timeout: tuple[float, float] = (5, 20),
          attempts: int = 3, backoff: float = 2.0, max_bytes: int = MAX_BYTES,
          session: requests.Session | None = None) -> Fetched:
    """GET with bounded retries and exponential backoff; redirects re-checked hop by hop."""
    http = session or requests
    last_error: FetchError | None = None
    for attempt in range(attempts):
        if attempt:
            time.sleep(backoff * 2 ** (attempt - 1))
        current, query = url, params
        try:
            for _ in range(MAX_REDIRECTS + 1):
                host = check_url(current, allowed_hosts)
                throttle(host, min_interval)
                response = http.get(current, params=query, timeout=timeout, stream=True,
                                    allow_redirects=False, headers={"User-Agent": USER_AGENT})
                if response.is_redirect:
                    current, query = urljoin(current, response.headers.get("Location", "")), None
                    response.close()
                    continue
                break
            else:
                raise FetchError("слишком много перенаправлений")
            body = response.raw.read(max_bytes + 1, decode_content=True)
            response.close()
            if len(body) > max_bytes:
                raise FetchError("ответ больше допустимого размера")
            if response.status_code == 429 or response.status_code >= 500:
                last_error = FetchError(f"HTTP {response.status_code}: "
                                        f"{body[:160].decode('utf-8', 'replace')}",
                                        response.status_code)
                continue
            if response.status_code != 200:
                raise FetchError(f"HTTP {response.status_code}", response.status_code)
            return Fetched(current, response.status_code, body,
                           response.headers.get("Content-Type", ""))
        except requests.RequestException as exc:
            last_error = FetchError(f"сеть: {type(exc).__name__}: {exc}")
    assert last_error is not None
    raise last_error
