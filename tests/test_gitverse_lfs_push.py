"""scripts/gitverse_lfs_push.py against a local fake LFS server (no network), #34."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "gitverse_lfs_push.py"
TOKEN = "tok-SECRET-5f3a9c"
USER = "tester"


class FakeLfs:
    """Mimics GitVerse: batch advertises a chunked upload header, chunked PUT -> 403."""

    def __init__(self) -> None:
        self.stored: dict[str, bytes] = {}
        self.put_mode = "ok"  # ok | forbid
        self.drop_puts = 0  # PUTs to drop without a response (network reset)
        self.batch_status = 200
        self.log: list[tuple[str, str, dict]] = []
        self.chunked_puts = 0
        self.ok_puts = 0
        self.verifies = 0
        self.upload_host = ""  # "" = same host as the batch endpoint


def make_handler(state: FakeLfs):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args) -> None:  # keep pytest output clean
            pass

        def _reply(self, status: int, body: bytes, ctype: str = "application/vnd.git-lfs+json") -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _authorized(self) -> bool:
            import base64
            want = "Basic " + base64.b64encode(f"{USER}:{TOKEN}".encode()).decode()
            return self.headers.get("Authorization") == want

        def _read_body(self) -> bytes:
            if self.headers.get("Transfer-Encoding", "").lower() == "chunked":
                data = b""
                while True:
                    size = int(self.rfile.readline().strip(), 16)
                    if size == 0:
                        self.rfile.readline()
                        return data
                    data += self.rfile.read(size)
                    self.rfile.readline()
            return self.rfile.read(int(self.headers.get("Content-Length", "0")))

        def do_POST(self) -> None:
            state.log.append(("POST", self.path, dict(self.headers)))
            body = self._read_body()
            if not self._authorized():
                return self._reply(401, b'{"message":"unauthorized"}')
            base = f"http://{self.headers['Host']}"
            up_base = f"http://{state.upload_host}:{self.server.server_address[1]}" if state.upload_host else base
            if self.path.endswith("/info/lfs/objects/batch"):
                if state.batch_status != 200:
                    return self._reply(state.batch_status, b'{"message":"batch refused"}')
                req = json.loads(body)
                out = []
                for o in req["objects"]:
                    entry = {"oid": o["oid"], "size": o["size"], "authenticated": True}
                    if o["oid"] not in state.stored:
                        entry["actions"] = {
                            "upload": {"href": f"{up_base}/lfs/upload/{o['oid']}",
                                       "header": {"Transfer-Encoding": "chunked"}},
                            "verify": {"href": f"{base}/lfs/verify"},
                        }
                    out.append(entry)
                return self._reply(200, json.dumps({"transfer": "basic", "objects": out}).encode())
            if self.path == "/lfs/verify":
                state.verifies += 1
                oid = json.loads(body)["oid"]
                return self._reply(200 if oid in state.stored else 404, b"{}")
            self._reply(404, b"{}")

        def do_PUT(self) -> None:
            state.log.append(("PUT", self.path, dict(self.headers)))
            if state.drop_puts > 0:
                state.drop_puts -= 1
                self.close_connection = True
                self.connection.shutdown(2)
                return
            chunked = self.headers.get("Transfer-Encoding", "").lower() == "chunked"
            body = self._read_body()
            if chunked:
                state.chunked_puts += 1
                return self._reply(403, b"Forbidden", "text/plain")
            if state.put_mode == "forbid" or not self._authorized():
                return self._reply(403, f"Forbidden for {USER}".encode(), "text/plain")
            oid = self.path.rsplit("/", 1)[-1]
            if hashlib.sha256(body).hexdigest() != oid:
                return self._reply(422, b"bad oid", "text/plain")
            state.stored[oid] = body
            state.ok_puts += 1
            self._reply(200, b"", "text/plain")

    return Handler


@pytest.fixture
def server():
    state = FakeLfs()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(state))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    state.url = f"http://127.0.0.1:{httpd.server_address[1]}/org/repo.git"
    yield state
    httpd.shutdown()
    httpd.server_close()


@pytest.fixture
def blobs(tmp_path: Path) -> list[Path]:
    paths = []
    for i, size in enumerate((1, 70_000, 300_000)):
        p = tmp_path / f"member_{i}.pt"
        p.write_bytes(os.urandom(size))
        paths.append(p)
    return paths


def run(state: FakeLfs, *args: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GITVERSE_")}
    env.update(GITVERSE_USER=USER, GITVERSE_TOKEN=TOKEN, GITVERSE_URL=state.url,
               GITVERSE_LFS_RETRY_DELAY="0", GITVERSE_LFS_HTTP_TIMEOUT="10")
    return subprocess.run([sys.executable, str(SCRIPT), *args], cwd=cwd or REPO_ROOT, env=env,
                          capture_output=True, text=True, timeout=60)


def no_secret(res: subprocess.CompletedProcess) -> None:
    import base64
    basic = base64.b64encode(f"{USER}:{TOKEN}".encode()).decode()
    for stream in (res.stdout, res.stderr):
        assert TOKEN not in stream
        assert basic not in stream
        assert "Authorization" not in stream


def oid(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_uploads_with_content_length_despite_chunked_header(server: FakeLfs, blobs: list[Path]) -> None:
    res = run(server, *map(str, blobs))
    assert res.returncode == 0, res.stderr
    assert server.chunked_puts == 0
    assert server.ok_puts == 3
    assert server.verifies == 3
    assert set(server.stored) == {oid(p) for p in blobs}
    for p in blobs:
        assert server.stored[oid(p)] == p.read_bytes()
    puts = [h for m, _, h in server.log if m == "PUT"]
    assert all(h.get("Content-Length") and "Transfer-Encoding" not in h for h in puts)
    # initial batch + control batch after upload
    assert sum(1 for m, path, _ in server.log if path.endswith("/objects/batch")) == 2
    assert "ok: на сервере все объекты (3)" in res.stdout
    no_secret(res)


def test_already_on_server_skips_put(server: FakeLfs, blobs: list[Path]) -> None:
    for p in blobs:
        server.stored[oid(p)] = p.read_bytes()
    res = run(server, *map(str, blobs))
    assert res.returncode == 0, res.stderr
    assert not [1 for m, _, _ in server.log if m == "PUT"]
    assert res.stdout.count("already on server") == 3
    no_secret(res)


def test_forbidden_put_fails_without_leaking_token(server: FakeLfs, blobs: list[Path]) -> None:
    server.put_mode = "forbid"
    res = run(server, *map(str, blobs))
    assert res.returncode != 0
    assert "HTTP 403" in res.stderr
    assert "проверьте права токена" in res.stderr
    assert "gitverse_lfs_push.py" in res.stderr
    # 4xx: no retries, one PUT per object
    assert sum(1 for m, _, _ in server.log if m == "PUT") == 3
    assert "повтор" not in res.stderr
    # failed PUT -> exit at once, no control batch
    assert sum(1 for m, path, _ in server.log if path.endswith("/objects/batch")) == 1
    no_secret(res)


def test_batch_rejected_fails(server: FakeLfs, blobs: list[Path]) -> None:
    server.batch_status = 403
    res = run(server, str(blobs[0]))
    assert res.returncode != 0
    assert "batch: HTTP 403" in res.stderr
    no_secret(res)


def test_network_reset_is_retried(server: FakeLfs, blobs: list[Path]) -> None:
    server.drop_puts = 2
    res = run(server, str(blobs[1]))
    assert res.returncode == 0, res.stderr
    assert "повтор 1/2" in res.stderr and "повтор 2/2" in res.stderr
    assert oid(blobs[1]) in server.stored
    no_secret(res)


def test_network_reset_gives_up_after_three_attempts(server: FakeLfs, blobs: list[Path]) -> None:
    server.drop_puts = 5
    res = run(server, str(blobs[1]))
    assert res.returncode != 0
    assert sum(1 for m, _, _ in server.log if m == "PUT") == 3
    assert "после 3 попыток" in res.stderr
    no_secret(res)


def test_missing_credentials(server: FakeLfs, blobs: list[Path]) -> None:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GITVERSE_")}
    res = subprocess.run([sys.executable, str(SCRIPT), str(blobs[0])], env=env,
                         capture_output=True, text=True, timeout=30)
    assert res.returncode == 2
    assert not server.log


@pytest.mark.skipif(shutil.which("git-lfs") is None, reason="git-lfs not installed")
def test_commit_mode_reads_local_lfs_store(server: FakeLfs, tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run([*git, "lfs", "install", "--local"], check=True, capture_output=True)
    (repo / ".gitattributes").write_text("*.pt filter=lfs diff=lfs merge=lfs -text\n")
    (repo / "models").mkdir()
    weights = repo / "models" / "w.pt"
    weights.write_bytes(os.urandom(50_000))
    subprocess.run([*git, "add", ".gitattributes", "models/w.pt"], check=True)
    subprocess.run([*git, "commit", "-qm", "w"], check=True)
    res = run(server, "--commit", "HEAD", cwd=repo)
    assert res.returncode == 0, res.stderr
    assert server.stored[oid(weights)] == weights.read_bytes()
    # pointer given as a path is resolved through the local store too
    server.stored.clear()
    server.log.clear()
    content = weights.read_bytes()
    weights.write_bytes(subprocess.run([*git, "show", "HEAD:models/w.pt"], check=True,
                                       capture_output=True).stdout)
    assert weights.read_bytes().startswith(b"version https://git-lfs")
    res = run(server, "models/w.pt", cwd=repo)
    assert res.returncode == 0, res.stderr
    assert server.stored[hashlib.sha256(content).hexdigest()] == content
    no_secret(res)


def test_fake_server_rejects_chunked_put(server: FakeLfs) -> None:
    # the behaviour the loader works around: a chunked body (as git-lfs sends it) gets 403
    import http.client
    port = int(server.url.split(":")[2].split("/")[0])
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    data = b"x" * 10
    conn.request("PUT", f"/lfs/upload/{hashlib.sha256(data).hexdigest()}", body=iter([data]),
                 headers={"Transfer-Encoding": "chunked"}, encode_chunked=True)
    assert conn.getresponse().status == 403
    assert server.chunked_puts == 1


def test_credentials_not_sent_to_foreign_upload_host(server: FakeLfs, blobs: list[Path]) -> None:
    server.upload_host = "localhost"
    res = run(server, str(blobs[0]))
    puts = [h for m, _, h in server.log if m == "PUT"]
    assert puts and all("Authorization" not in h for h in puts)
    assert res.returncode != 0  # the fake demands auth on PUT
    no_secret(res)
