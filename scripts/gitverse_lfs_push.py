#!/usr/bin/env python3
"""Upload Git LFS objects to GitVerse with Content-Length (no chunked body), #34.

GitVerse's batch response puts ``Transfer-Encoding: chunked`` into ``actions.upload.header``;
git-lfs then sends a chunked body and GitVerse answers 403. The same PUT with
``Content-Length`` succeeds. This script speaks the LFS batch API directly:
batch (upload) -> PUT with Content-Length -> verify, then a second batch must no longer
ask for an upload for any object. Exit 0 only if every object is on the server.

    python3 scripts/gitverse_lfs_push.py --commit origin/main   # LFS files of a commit
    python3 scripts/gitverse_lfs_push.py models/unet/member_0.pt # explicit paths

Credentials: GITVERSE_USER / GITVERSE_TOKEN (Basic). LFS URL: $GITVERSE_URL + /info/lfs.
The token and the Authorization header are never printed; for HTTP errors only the
status code and the first 200 bytes of the response body are shown.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import ipaddress
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

DEFAULT_URL = "https://gitverse.ru/hackrus.experts/kosmo-moskva_umstvenno_ustalye_34.git"
LFS_JSON = "application/vnd.git-lfs+json"
POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"
ATTEMPTS = 3
HINT_403 = ("403 на PUT объекта: проверьте права токена; git-lfs chunked-загрузку GitVerse "
            "отвергает, поэтому используется gitverse_lfs_push.py")


class HttpFail(Exception):
    """HTTP error with a safe message (status + head of body, no headers)."""

    def __init__(self, status: int, body: bytes) -> None:
        self.status = status
        self.body = body[:200]
        super().__init__(f"HTTP {status}")


class NetFail(Exception):
    """Network error after all retries."""


@dataclass
class Obj:
    path: str
    oid: str
    size: int
    source: Path


def _secret() -> str:
    return os.environ.get("GITVERSE_TOKEN", "")


def say(msg: str, err: bool = False) -> None:
    token = _secret()
    if token:
        msg = msg.replace(token, "***")
    print(msg, file=sys.stderr if err else sys.stdout, flush=True)


def _body_head(body: bytes) -> str:
    return body[:200].decode("utf-8", "replace").replace("\n", " ")


def sha256_file(path: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
            size += len(block)
    return h.hexdigest(), size


def parse_pointer(text: bytes) -> tuple[str, int] | None:
    if not text.startswith(POINTER_PREFIX):
        return None
    fields = dict(line.split(" ", 1) for line in text.decode().splitlines() if " " in line)
    oid = fields.get("oid", "")
    if not oid.startswith("sha256:") or "size" not in fields:
        return None
    return oid.removeprefix("sha256:"), int(fields["size"])


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout


def lfs_store() -> Path:
    return Path(_git("rev-parse", "--path-format=absolute", "--git-common-dir").strip()) / "lfs" / "objects"


def _from_pointer(path: str, oid: str, size: int) -> Obj:
    src = lfs_store() / oid[:2] / oid[2:4] / oid
    if not src.is_file():
        raise SystemExit(f"{path}: объекта {oid[:12]} нет локально — сначала git lfs fetch")
    return Obj(path, oid, size, src)


def collect_from_commit(commit: str) -> list[Obj]:
    objs = []
    for path in _git("lfs", "ls-files", "-n", commit).splitlines():
        if not path:
            continue
        ptr = parse_pointer(subprocess.run(["git", "show", f"{commit}:{path}"], check=True,
                                           capture_output=True).stdout)
        if ptr is None:
            raise SystemExit(f"{path}: в {commit} не LFS-указатель")
        objs.append(_from_pointer(path, *ptr))
    return objs


def collect_from_paths(paths: list[str]) -> list[Obj]:
    objs = []
    for p in paths:
        path = Path(p)
        with path.open("rb") as f:
            head = f.read(200)
        if (ptr := parse_pointer(head)) is not None:
            objs.append(_from_pointer(p, *ptr))
        else:
            oid, size = sha256_file(path)
            objs.append(Obj(p, oid, size, path))
    return objs


def check_object(obj: Obj) -> None:
    oid, size = sha256_file(obj.source)
    if (oid, size) != (obj.oid, obj.size):
        raise SystemExit(f"{obj.path}: sha256/размер не совпадают с указателем")


class Client:
    def __init__(self, lfs_url: str, user: str, token: str, delay: float, timeout: float) -> None:
        self.lfs_url = lfs_url.rstrip("/")
        self.auth = "Basic " + base64.b64encode(f"{user}:{token}".encode()).decode()
        self.delay = delay
        self.timeout = timeout
        self.host = urllib.parse.urlsplit(self.lfs_url).hostname or ""

    def _auth_for(self, href: str) -> bool:
        # Basic credentials only go to the LFS host or its parent domain, never elsewhere.
        host = urllib.parse.urlsplit(href).hostname or ""
        if host == self.host:
            return True
        try:
            ipaddress.ip_address(self.host)
            return False
        except ValueError:
            parent = ".".join(self.host.split(".")[-2:])
            return parent.count(".") == 1 and host.endswith("." + parent)

    def _send(self, method: str, url: str, headers: dict[str, str], body_factory) -> bytes:
        last = ""
        for attempt in range(1, ATTEMPTS + 1):
            body = body_factory()
            try:
                req = urllib.request.Request(url, method=method, headers=headers, data=body)
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return resp.read()
            except urllib.error.HTTPError as e:
                status, content = e.code, e.read()
                if status < 500:
                    raise HttpFail(status, content) from None
                last = f"HTTP {status} {_body_head(content)}"
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                reason = getattr(e, "reason", e)
                last = f"{type(reason).__name__}: {reason}"
            finally:
                if hasattr(body, "close"):
                    body.close()
            if attempt < ATTEMPTS:
                say(f"  {method} повтор {attempt}/{ATTEMPTS - 1}: {last}", err=True)
                time.sleep(self.delay)
        raise NetFail(f"{method}: {last}")

    def batch(self, objs: list[Obj]) -> dict[str, dict]:
        payload = json.dumps({"operation": "upload", "transfers": ["basic"],
                              "objects": [{"oid": o.oid, "size": o.size} for o in objs]}).encode()
        headers = {"Accept": LFS_JSON, "Content-Type": LFS_JSON, "Authorization": self.auth}
        raw = self._send("POST", self.lfs_url + "/objects/batch", headers, lambda: payload)
        return {o["oid"]: o for o in json.loads(raw).get("objects", [])}

    def put(self, obj: Obj, action: dict) -> None:
        headers = {k: v for k, v in action.get("header", {}).items()
                   if k.lower() not in ("transfer-encoding", "content-length")}
        if not any(k.lower() == "authorization" for k in headers) and self._auth_for(action["href"]):
            headers["Authorization"] = self.auth
        headers["Content-Type"] = "application/octet-stream"
        headers["Content-Length"] = str(obj.size)
        self._send("PUT", action["href"], headers, lambda: obj.source.open("rb"))

    def verify(self, obj: Obj, action: dict) -> None:
        headers = dict(action.get("header", {}))
        if not any(k.lower() == "authorization" for k in headers) and self._auth_for(action["href"]):
            headers["Authorization"] = self.auth
        headers["Accept"] = LFS_JSON
        headers["Content-Type"] = LFS_JSON
        payload = json.dumps({"oid": obj.oid, "size": obj.size}).encode()
        self._send("POST", action["href"], headers, lambda: payload)


def _obj_error(entry: dict | None) -> str | None:
    if entry is None:
        return "нет в ответе batch"
    err = entry.get("error")
    if err:
        return f"batch error {err.get('code', '?')}: {str(err.get('message', ''))[:200]}"
    return None


def push(client: Client, objs: list[Obj]) -> int:
    failed = 0
    try:
        answer = client.batch(objs)
    except HttpFail as e:
        say(f"[lfs] FAIL batch: HTTP {e.status} {_body_head(e.body)}", err=True)
        return 1
    for obj in objs:
        tag = f"{obj.path} {obj.oid[:12]} {obj.size} байт"
        entry = answer.get(obj.oid)
        if (err := _obj_error(entry)) is not None:
            say(f"[lfs] FAIL {tag}: {err}", err=True)
            failed += 1
            continue
        actions = entry.get("actions", {})
        if "upload" not in actions:
            say(f"[lfs] {tag}: already on server")
            continue
        try:
            client.put(obj, actions["upload"])
            say(f"[lfs] {tag}: PUT ok")
            if "verify" in actions:
                client.verify(obj, actions["verify"])
                say(f"[lfs] {tag}: verify ok")
        except HttpFail as e:
            say(f"[lfs] FAIL {tag}: HTTP {e.status} {_body_head(e.body)}", err=True)
            if e.status == 403:
                say(f"[lfs] {HINT_403}", err=True)
            failed += 1
    if failed:
        return 1
    # Second batch: nothing may still ask for an upload.
    try:
        answer = client.batch(objs)
    except HttpFail as e:
        say(f"[lfs] FAIL контрольный batch: HTTP {e.status} {_body_head(e.body)}", err=True)
        return 1
    for obj in objs:
        entry = answer.get(obj.oid)
        err = _obj_error(entry)
        if err is None and "upload" in entry.get("actions", {}):
            err = "сервер всё ещё просит загрузку"
        if err is not None:
            say(f"[lfs] FAIL контроль {obj.path} {obj.oid[:12]}: {err}", err=True)
            failed += 1
    if failed:
        return 1
    say(f"[lfs] ok: на сервере все объекты ({len(objs)})")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", help="файлы (содержимое или LFS-указатели)")
    ap.add_argument("--commit", default="HEAD", help="commit, чьи LFS-файлы грузить, если пути не заданы")
    args = ap.parse_args(argv)

    user, token = os.environ.get("GITVERSE_USER", ""), os.environ.get("GITVERSE_TOKEN", "")
    if not user or not token:
        say("[lfs] GITVERSE_USER и GITVERSE_TOKEN должны быть заданы", err=True)
        return 2
    lfs_url = os.environ.get("GITVERSE_URL", DEFAULT_URL).rstrip("/") + "/info/lfs"
    client = Client(lfs_url, user, token,
                    delay=float(os.environ.get("GITVERSE_LFS_RETRY_DELAY", "5")),
                    timeout=float(os.environ.get("GITVERSE_LFS_HTTP_TIMEOUT", "300")))
    try:
        objs = collect_from_paths(args.paths) if args.paths else collect_from_commit(args.commit)
        for obj in objs:
            check_object(obj)
    except (OSError, subprocess.CalledProcessError) as e:
        say(f"[lfs] FAIL: не собран список объектов: {type(e).__name__}", err=True)
        return 2
    if not objs:
        say("[lfs] LFS-файлов нет — загружать нечего")
        return 0
    say(f"[lfs] {len(objs)} объектов → {lfs_url}")
    try:
        return push(client, objs)
    except NetFail as e:
        say(f"[lfs] FAIL сеть после {ATTEMPTS} попыток: {e}", err=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
