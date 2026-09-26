"""Pre-event water is context for people, never an input of p, the mask, thresholds, EL or the
plan (#149 p.1, p.4). No module of that chain may import floodvalue.prewater or read its files;
export only copies them as optional files and adds their sources to source_manifest.json."""

import ast
import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "floodvalue"
GUARDED = ["data", "model", "baseline", "calibration", "assets", "procurement"]
FORBIDDEN_TEXT = re.compile(r"floodvalue\.prewater|pre_event_water|new_water_probability|"
                            r"sentinel-1-rtc|planetarycomputer")


def _imports_prewater(tree: ast.AST, package: str) -> list[int]:
    parts = package.split(".")
    lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(a.name == "floodvalue.prewater" or a.name.startswith("floodvalue.prewater.")
                   for a in node.names):
                lines.append(node.lineno)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if node.level == 0:
                full = mod
            else:
                base = parts[: max(0, len(parts) - (node.level - 1))]
                full = ".".join(base + ([mod] if mod else []))
            names = {a.name for a in node.names}
            if (full == "floodvalue.prewater" or full.startswith("floodvalue.prewater.")
                    or (full == "floodvalue" and "prewater" in names)):
                lines.append(node.lineno)
    return lines


def test_pipeline_modules_never_touch_prewater():
    hits = []
    for pkg in GUARDED:
        for f in sorted((SRC / pkg).rglob("*.py")):
            text = f.read_text(encoding="utf-8")
            package = ".".join(("floodvalue",) + f.relative_to(SRC).parent.parts)
            hits += [f"{f.relative_to(SRC)}:{n}: import floodvalue.prewater"
                     for n in _imports_prewater(ast.parse(text), package)]
            hits += [f"{f.relative_to(SRC)}:{i}: {line.strip()}"
                     for i, line in enumerate(text.splitlines(), 1) if FORBIDDEN_TEXT.search(line)]
    assert not hits, "слой «вода до события» попал в конвейер p/EL/плана:\n" + "\n".join(hits)


def test_export_does_not_import_prewater():
    for f in sorted((SRC / "export").rglob("*.py")):
        package = ".".join(("floodvalue",) + f.relative_to(SRC).parent.parts)
        assert not _imports_prewater(ast.parse(f.read_text(encoding="utf-8")), package), f


def test_guard_catches_every_import_spelling():
    for code in ("import floodvalue.prewater", "import floodvalue.prewater.core as c",
                 "from floodvalue import prewater", "from floodvalue.prewater import core",
                 "from ..prewater import core", "from .. import prewater"):
        assert _imports_prewater(ast.parse(code), "floodvalue.model"), code
    assert not _imports_prewater(ast.parse("from floodvalue import contracts"), "floodvalue.model")


def test_guard_actually_scans_code():
    assert all(any((SRC / pkg).rglob("*.py")) for pkg in GUARDED)
    assert "pre_event_water" in (SRC / "prewater" / "run.py").read_text(encoding="utf-8")
