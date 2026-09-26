"""scripts/sync-gitverse.sh — offline checks only (no network, no GitVerse), #34."""

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "sync-gitverse.sh"
TEXT = SCRIPT.read_text(encoding="utf-8")
# PATH lookup, not CreateProcess: on Windows a bare "bash" hits System32\bash.exe (WSL) first.
BASH = shutil.which("bash") or "bash"


def test_bash_syntax() -> None:
    subprocess.run([BASH, "-n", str(SCRIPT)], check=True)


@pytest.fixture
def bare_repo(tmp_path: Path) -> Path:
    """Git repo without .env so the script cannot pick up a real token."""
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "scripts").mkdir()
    shutil.copy(SCRIPT, tmp_path / "scripts" / SCRIPT.name)
    return tmp_path


def _run(repo: Path, *args: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GITVERSE_")}
    return subprocess.run([BASH, "scripts/sync-gitverse.sh", *args], cwd=repo, env=env,
                          capture_output=True, text=True, encoding="utf-8", timeout=30)


def test_unknown_argument_rejected(bare_repo: Path) -> None:
    res = _run(bare_repo, "--bogus")
    assert res.returncode == 2
    assert "неизвестный аргумент" in res.stderr


def test_verify_stops_on_missing_token(bare_repo: Path) -> None:
    res = _run(bare_repo, "--verify")
    assert res.returncode != 0
    assert "GITVERSE_TOKEN не задан" in res.stderr
    assert "неизвестный аргумент" not in res.stderr


@pytest.mark.parametrize("args", [(), ("--code-only",)])
def test_push_modes_refused_after_stop_code(bare_repo: Path, args: tuple[str, ...]) -> None:
    """GitVerse заморожен с 27.09.2026 01:00 МСК: push-режимы отказывают до чтения токена."""
    res = _run(bare_repo, *args)
    assert res.returncode == 1
    assert "GitVerse заморожен" in res.stderr


def test_no_git_tracing_enabled() -> None:
    # tracing prints the Authorization header; the script may only unset these variables
    assert not re.search(r"\b(GIT_TRACE\w*|GIT_CURL_VERBOSE)=", TEXT)
    assert not re.search(r"export\s+(GIT_TRACE|GIT_CURL_VERBOSE)", TEXT)


def test_transport_hardening_present() -> None:
    assert "http.version=HTTP/1.1" in TEXT
    assert "lfs.transfer.maxretries=" in TEXT
    assert "lfs.concurrenttransfers=" in TEXT
    assert re.search(r'timeout -k 30 "\$LFS_TIMEOUT"', TEXT)
    assert "--no-verify" in TEXT
    # zsh-safe refspec: braces around want
    assert '"${want}:refs/heads/' in TEXT
    assert "$want:" not in TEXT


def test_token_not_on_command_line() -> None:
    # the token reaches curl via stdin (-H @-), never as an argument or inside a URL
    assert "-H \"Authorization" not in TEXT
    assert not re.search(r"https://[^\s\"]*\$\{?GITVERSE_TOKEN", TEXT)


def test_lfs_goes_through_content_length_loader() -> None:
    # git-lfs sends chunked bodies that GitVerse rejects with 403; the script must not use it for upload
    assert "python3 scripts/gitverse_lfs_push.py --commit \"$want\"" in TEXT
    assert "lfs push" not in TEXT
    assert "push_opts=(--no-verify)" in TEXT
    assert "нет записи в LFS" not in TEXT
    assert "проверьте права токена" in TEXT
    assert (REPO_ROOT / "scripts" / "gitverse_lfs_push.py").is_file()


def _closed_port() -> int:
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run_api_down(repo: Path, gitverse_url: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GITVERSE_")}
    env.update(GITVERSE_TOKEN="fake-token-for-test", GITVERSE_USER="tester",  # secret-scan: allow
               GITVERSE_URL=gitverse_url, GITVERSE_API_RETRY_DELAY="0",
               GITVERSE_API=f"http://127.0.0.1:{_closed_port()}/repos/x")
    return subprocess.run([BASH, "scripts/sync-gitverse.sh", "--verify"], cwd=repo, env=env,
                          capture_output=True, text=True, encoding="utf-8", timeout=60)


def test_api_down_falls_back_to_symref(bare_repo: Path, tmp_path: Path) -> None:
    remote = tmp_path / "gv.git"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "trunk", str(remote)], check=True)
    subprocess.run(["git", "-C", str(bare_repo), "-c", "user.name=t", "-c", "user.email=t@t",
                    "commit", "-q", "--allow-empty", "-m", "x"], check=True)
    subprocess.run(["git", "-C", str(bare_repo), "push", "-q", str(remote), "HEAD:trunk"], check=True)
    res = _run_api_down(bare_repo, str(remote))
    assert "беру HEAD через ls-remote --symref" in res.stderr
    assert "ветка по умолчанию (ls-remote): trunk\n" in res.stderr
    assert "ветка по умолчанию GitVerse не получена" not in res.stderr
    assert "Traceback" not in res.stderr and "JSONDecodeError" not in res.stderr
    # goes on past branch detection (then stops: the fixture repo has no origin)
    assert res.returncode != 0


def test_api_and_symref_down_fail_cleanly(bare_repo: Path, tmp_path: Path) -> None:
    res = _run_api_down(bare_repo, str(tmp_path / "missing.git"))
    assert res.returncode == 1
    assert "ветка по умолчанию GitVerse не получена ни из API, ни из ls-remote" in res.stderr
    assert "Traceback" not in res.stderr and "JSONDecodeError" not in res.stderr
    assert "fake-token-for-test" not in res.stderr + res.stdout
