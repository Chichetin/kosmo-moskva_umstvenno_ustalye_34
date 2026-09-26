"""#158: CLIs print Cyrillic and «Δ»; a cp1251 console must not crash them (criterion 11)."""

import io
import os
import subprocess
import sys
from pathlib import Path

import floodvalue

REPO = Path(__file__).resolve().parents[1]


def test_utf8_console_switches_cp1251_streams(monkeypatch):
    out = io.TextIOWrapper(io.BytesIO(), encoding="cp1251", newline="\n")
    err = io.TextIOWrapper(io.BytesIO(), encoding="cp1251", newline="\n")
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    floodvalue.utf8_console()
    print("ΔIoU к baseline")
    out.flush()
    assert out.encoding == "utf-8"
    assert err.encoding == "utf-8"
    assert out.buffer.getvalue().decode("utf-8") == "ΔIoU к baseline\n"


def test_utf8_console_leaves_streams_without_reconfigure(monkeypatch):
    stream = io.StringIO()
    monkeypatch.setattr(sys, "stdout", stream)
    floodvalue.utf8_console()
    assert sys.stdout is stream


def test_loeo_table_prints_in_cp1251_console():
    env = os.environ | {"PYTHONIOENCODING": "cp1251", "PYTHONUTF8": "0"}
    res = subprocess.run(
        [sys.executable, "-m", "floodvalue.model.unet.loeo",
         "--table-from", str(REPO / "docs" / "verification" / "loeo")],
        capture_output=True, env=env, cwd=REPO, timeout=300)
    assert res.returncode == 0, res.stderr.decode("utf-8", "replace")
    assert "ΔIoU" in res.stdout.decode("utf-8")
