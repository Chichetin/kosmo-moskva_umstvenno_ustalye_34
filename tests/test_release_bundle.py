"""scripts/release_bundle.py — run kit + weights archives for the GitVerse release (#85).

Kit: the reference set tests/fixtures/good (12/12); weights: synthetic files in tmp.
"""

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
GOOD = REPO_ROOT / "tests" / "fixtures" / "good"
SCRIPT = REPO_ROOT / "scripts" / "release_bundle.py"

_spec = importlib.util.spec_from_file_location("release_bundle", SCRIPT)
rb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rb)

LFS_POINTER = (b"version https://git-lfs.github.com/spec/v1\n"
               b"oid sha256:" + b"a" * 64 + b"\nsize 7810843\n")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def models(tmp_path: Path) -> Path:
    root = tmp_path / "models"
    (root / "unet").mkdir(parents=True)
    (root / "baseline").mkdir()
    (root / "unet" / "member_0.pt").write_bytes(b"PK\x03\x04 synthetic weights " * 100)
    (root / "unet" / "config.json").write_text('{"arch": "synthetic"}\n', encoding="utf-8")
    (root / "baseline" / "config.json").write_text('{"threshold_db": -23.0}\n', encoding="utf-8")
    (root / ".gitkeep").write_text("", encoding="utf-8")
    return root


def _build(out: Path, models: Path, dest: Path) -> list[Path]:
    return rb.build(out, models, dest)


def test_build_on_reference_kit_gives_two_zips_and_checkable_sums(models, tmp_path):
    dest = tmp_path / "release"
    paths = _build(GOOD, models, dest)
    names = sorted(p.name for p in paths)
    assert names == ["models_fixture.zip", "run_Synthetic_000001_5000_fixture.zip"]

    sums = (dest / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    assert sorted(sums) == sorted(f"{_sha(p)}  {p.name}" for p in paths)
    if shutil.which("sha256sum"):
        res = subprocess.run(["sha256sum", "-c", "SHA256SUMS"], cwd=dest,
                             capture_output=True, text=True, env={**os.environ, "LC_ALL": "C"})
        assert res.returncode == 0, res.stdout + res.stderr
        assert res.stdout.count(": OK") == 2

    with zipfile.ZipFile(dest / "run_Synthetic_000001_5000_fixture.zip") as z:
        members = z.namelist()
        for name in rb.REQUIRED_FILES:
            assert f"Synthetic_000001_5000_fixture/{name}" in members
        assert len(rb.REQUIRED_FILES) == 12
        assert z.read("Synthetic_000001_5000_fixture/run_metadata.json") == \
            (GOOD / "run_metadata.json").read_bytes()
    with zipfile.ZipFile(dest / "models_fixture.zip") as z:
        assert z.namelist() == sorted(z.namelist())
        assert "models/unet/member_0.pt" in z.namelist()
        assert "models/.gitkeep" in z.namelist()
        assert z.read("models/unet/member_0.pt") == (models / "unet" / "member_0.pt").read_bytes()


def test_rebuild_gives_same_sha256_even_after_mtime_change(models, tmp_path):
    first = {p.name: _sha(p) for p in _build(GOOD, models, tmp_path / "a")}
    # Different mtime and creation order of the weights must not change the archive.
    w = models / "unet" / "member_0.pt"
    data = w.read_bytes()
    w.unlink()
    w.write_bytes(data)
    os.utime(w, (1_000_000_000, 1_000_000_000))
    os.chmod(w, 0o600)
    second = {p.name: _sha(p) for p in _build(GOOD, models, tmp_path / "b")}
    assert first == second


def test_refuses_when_validator_fails(models, tmp_path):
    broken = tmp_path / "kit"
    shutil.copytree(GOOD, broken)
    (broken / "sensitivity.csv").unlink()
    dest = tmp_path / "release"
    with pytest.raises(rb.ReleaseError, match="12/12"):
        _build(broken, models, dest)
    assert not dest.exists() or not any(dest.iterdir())


def test_refuses_on_lfs_pointer_in_models(models, tmp_path):
    (models / "unet" / "member_1.pt").write_bytes(LFS_POINTER)
    dest = tmp_path / "release"
    with pytest.raises(rb.ReleaseError, match="указатель"):
        _build(GOOD, models, dest)
    assert not dest.exists() or not any(dest.iterdir())


def test_refuses_models_without_weight_files(tmp_path):
    root = tmp_path / "models"
    root.mkdir()
    (root / ".gitkeep").write_text("", encoding="utf-8")
    (root / "README.md").write_text("no weights\n", encoding="utf-8")
    with pytest.raises(rb.ReleaseError, match="весов"):
        rb.check_models(root)


def test_weights_in_run_metadata_must_match_archived_files(models):
    w = models / "unet" / "member_0.pt"
    ok = {"methods": {"main": {"weights": [{"path": "models/unet/member_0.pt", "sha256": _sha(w)}]},
                      "baseline": {"weights": []}}}
    rb.check_weights_match(ok, models)

    wrong = json.loads(json.dumps(ok))
    wrong["methods"]["main"]["weights"][0]["sha256"] = "0" * 64
    with pytest.raises(rb.ReleaseError, match="sha256"):
        rb.check_weights_match(wrong, models)

    missing = json.loads(json.dumps(ok))
    missing["methods"]["main"]["weights"][0]["path"] = "models/unet/member_9.pt"
    with pytest.raises(rb.ReleaseError, match="member_9.pt"):
        rb.check_weights_match(missing, models)


def test_cli_prints_check_command_and_never_uploads_by_default(models, tmp_path, monkeypatch):
    monkeypatch.setenv("GITVERSE_TOKEN", "secret-token-value")
    dest = tmp_path / "release"
    res = subprocess.run([sys.executable, str(SCRIPT), "--out", str(GOOD), "--models", str(models),
                          "--dest", str(dest)], capture_output=True, text=True, cwd=REPO_ROOT)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "sha256sum -c SHA256SUMS" in res.stdout
    assert "ЭТАЛОН" in res.stdout  # the fixture is flagged as not a real run
    assert "secret-token-value" not in res.stdout + res.stderr


def test_cli_refuses_without_out(tmp_path):
    res = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True, cwd=REPO_ROOT)
    assert res.returncode != 0


# --- upload (GitVerse REST API, fake local server) ------------------------------------------------

class _FakeGitVerse(BaseHTTPRequestHandler):
    calls: list = []
    existing_assets: list = []
    release_exists = False
    target = "0" * 40

    def log_message(self, *args):
        pass

    def _reply(self, code: int, body: dict) -> None:
        raw = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _record(self) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        type(self).calls.append({"method": self.command, "path": self.path,
                                 "auth": self.headers.get("Authorization"),
                                 "accept": self.headers.get("Accept"),
                                 "ctype": self.headers.get("Content-Type"), "body": body})
        return body

    def do_GET(self):
        self._record()
        if self.path.endswith("/releases/tags/v-test") and type(self).release_exists:
            self._reply(200, {"id": 7, "target_commitish": type(self).target, "assets": [{"name": n} for n in type(self).existing_assets]})
        else:
            self._reply(404, {"message": "not found"})

    def do_POST(self):
        self._record()
        if self.path.endswith("/releases"):
            self._reply(201, {"id": 7, "assets": []})
        elif "/releases/7/assets?name=" in self.path:
            self._reply(201, {"id": 1, "name": self.path.split("name=")[1]})
        else:
            self._reply(404, {"message": "unexpected"})


@pytest.fixture
def fake_api(monkeypatch):
    _FakeGitVerse.calls = []
    _FakeGitVerse.existing_assets = []
    _FakeGitVerse.release_exists = False
    _FakeGitVerse.target = "0" * 40
    server = HTTPServer(("127.0.0.1", 0), _FakeGitVerse)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("GITVERSE_API", f"http://127.0.0.1:{server.server_port}/repos/o/r")
    monkeypatch.setenv("GITVERSE_TOKEN", "secret-token-value")
    yield _FakeGitVerse
    server.shutdown()


def test_upload_creates_release_and_posts_three_assets(models, tmp_path, fake_api):
    dest = tmp_path / "release"
    paths = _build(GOOD, models, dest)
    rb.upload("v-test", paths + [dest / "SHA256SUMS"], target="0" * 40, body="kit")
    methods = [(c["method"], c["path"].split("/repos/o/r")[1]) for c in fake_api.calls]
    assert methods[0] == ("GET", "/releases/tags/v-test")
    assert methods[1] == ("POST", "/releases")
    created = json.loads(fake_api.calls[1]["body"])
    assert created["tag_name"] == "v-test" and created["target_commitish"] == "0" * 40
    uploaded = sorted(p.split("name=")[1] for m, p in methods[2:])
    assert uploaded == sorted(["SHA256SUMS", *(p.name for p in paths)])
    for c in fake_api.calls:
        assert c["auth"] == "Bearer secret-token-value"
        assert c["accept"] == "application/vnd.gitverse.object+json;version=1"
        assert b"secret-token-value" not in c["body"]
        assert "secret-token-value" not in c["path"]
    asset_call = fake_api.calls[2]
    assert asset_call["ctype"].startswith("multipart/form-data; boundary=")
    assert b'name="attachment"' in asset_call["body"]


def test_upload_refuses_to_overwrite_existing_asset(models, tmp_path, fake_api):
    dest = tmp_path / "release"
    paths = _build(GOOD, models, dest)
    fake_api.release_exists = True
    fake_api.existing_assets = ["SHA256SUMS"]
    with pytest.raises(rb.ReleaseError, match="SHA256SUMS"):
        rb.upload("v-test", paths + [dest / "SHA256SUMS"], target="0" * 40, body="kit")
    assert all(c["method"] == "GET" for c in fake_api.calls)


def test_upload_refuses_file_over_limit(tmp_path, fake_api, monkeypatch):
    big = tmp_path / "big.zip"
    big.write_bytes(b"x" * 11)
    monkeypatch.setattr(rb, "MAX_ASSET_BYTES", 10)
    with pytest.raises(rb.ReleaseError, match="100 МБ"):
        rb.upload("v-test", [big], target="0" * 40, body="kit")
    assert fake_api.calls == []


def test_cli_refuses_to_upload_reference_fixture(models, tmp_path, fake_api):
    res = subprocess.run([sys.executable, str(SCRIPT), "--out", str(GOOD), "--models", str(models),
                          "--dest", str(tmp_path / "r"), "--upload", "v-test"],
                         capture_output=True, text=True, cwd=REPO_ROOT)
    assert res.returncode != 0
    assert "эталон tests/fixtures/good в релиз не загружается" in res.stderr
    assert fake_api.calls == []


def test_upload_refuses_existing_tag_on_another_commit(models, tmp_path, fake_api):
    dest = tmp_path / "release"
    paths = _build(GOOD, models, dest)
    fake_api.release_exists = True
    fake_api.target = "1" * 40
    with pytest.raises(rb.ReleaseError, match="1111111"):
        rb.upload("v-test", paths, target="0" * 40, body="kit")
    assert all(c["method"] == "GET" for c in fake_api.calls)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True,
                          text=True, env={**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
                                          "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
                          ).stdout.strip()


def test_models_must_equal_the_commit_that_produced_the_kit(tmp_path):
    repo = tmp_path / "repo"
    (repo / "models" / "unet").mkdir(parents=True)
    _git(repo, "init", "-q")
    w = repo / "models" / "unet" / "member_0.pt"
    w.write_bytes(b"weights v1")
    _git(repo, "add", "models")
    _git(repo, "commit", "-q", "-m", "v1")
    v1 = _git(repo, "rev-parse", "HEAD")
    rb.check_models_committed(repo, repo / "models", v1[:7])

    w.write_bytes(b"weights v2 (uncommitted)")
    with pytest.raises(rb.ReleaseError, match="models/unet/member_0.pt"):
        rb.check_models_committed(repo, repo / "models", v1[:7])

    _git(repo, "commit", "-q", "-am", "v2")
    with pytest.raises(rb.ReleaseError, match="models/unet/member_0.pt"):
        rb.check_models_committed(repo, repo / "models", v1[:7])  # kit made by v1, models are v2

    (repo / ".gitignore").write_text("*.bin\n", encoding="utf-8")
    (repo / "models" / "unet" / "ignored.bin").write_bytes(b"ignored but would be zipped")
    with pytest.raises(rb.ReleaseError, match="ignored.bin"):
        rb.check_models_committed(repo, repo / "models", "HEAD")
    (repo / "models" / "unet" / "ignored.bin").unlink()

    (repo / "models" / "unet" / "extra.pt").write_bytes(b"untracked")
    with pytest.raises(rb.ReleaseError, match="extra.pt"):
        rb.check_models_committed(repo, repo / "models", "HEAD")

    with pytest.raises(rb.ReleaseError, match="не найден"):
        rb.check_models_committed(repo, repo / "models", "deadbeef")


def test_build_checks_repo_models_against_kit_commit(models, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(rb, "REPO_ROOT", tmp_path)  # models fixture = <REPO_ROOT>/models
    monkeypatch.setattr(rb, "is_reference", lambda run_dir, metadata: False)
    monkeypatch.setattr(rb, "check_run", lambda run_dir: {"run_id": "r", "commit": "abc", "methods": {}})
    monkeypatch.setattr(rb, "check_models_committed", lambda *a: calls.append(a))
    rb.build(GOOD, models, tmp_path / "release")
    assert calls == [(tmp_path, models, "abc")]


def test_cli_refuses_to_upload_dirty_kit(models, tmp_path, fake_api):
    kit = tmp_path / "kit"
    shutil.copytree(GOOD, kit)
    meta = json.loads((kit / "run_metadata.json").read_text(encoding="utf-8"))
    meta["commit"] = _git(REPO_ROOT, "rev-parse", "HEAD")
    meta["commit_dirty"] = True
    (kit / "run_metadata.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    monkeypatch_argv = [str(kit), str(models), str(tmp_path / "r")]
    res = subprocess.run([sys.executable, "-c",
                          "import sys, importlib.util as u; s=u.spec_from_file_location('rb', sys.argv[1]);"
                          "m=u.module_from_spec(s); s.loader.exec_module(m);"
                          "m.check_run=lambda d: __import__('json').loads((d/'run_metadata.json').read_text(encoding='utf-8'));"
                          "m.check_models_committed=lambda *a: None;"
                          "sys.exit(m.main(['--out', sys.argv[2], '--models', sys.argv[3], '--dest', sys.argv[4],"
                          " '--upload', 'v-test']))", str(SCRIPT), *monkeypatch_argv],
                         capture_output=True, text=True, cwd=REPO_ROOT)
    assert res.returncode != 0
    assert "commit_dirty" in res.stderr
    assert fake_api.calls == []
