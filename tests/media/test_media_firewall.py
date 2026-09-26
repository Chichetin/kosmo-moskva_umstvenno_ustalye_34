"""Messages are signals for where to look, never inputs of the model or the economics.

Spec docs/features/media-monitoring.md §В/§И: modules that produce p, the mask, the
threshold, EL or the A/B/C plan (and `data`, which feeds them) must not import
`floodvalue.media` or read its storage. `media` itself imports only paths from the
project, so it cannot change those results either.
"""

import ast
import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "floodvalue"
GUARDED = ["data", "model", "baseline", "calibration", "assets", "procurement", "export"]
FORBIDDEN_TEXT = re.compile(r"floodvalue\.media|FLOODVALUE_MEDIA_DIR|media\.sqlite3")
ALLOWED_FROM_MEDIA = {"floodvalue.paths", "floodvalue.contracts"}


def _floodvalue_imports(tree: ast.AST, package: str) -> list[tuple[int, str]]:
    parts = package.split(".")
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [(node.lineno, alias.name) for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                full = node.module or ""
            else:
                base = parts[: max(0, len(parts) - (node.level - 1))]
                full = ".".join(base + ([node.module] if node.module else []))
            found += [(node.lineno, full)]
            if full == "floodvalue":
                found += [(node.lineno, f"floodvalue.{alias.name}") for alias in node.names]
    return [(line, name) for line, name in found if name.startswith("floodvalue")]


def _package(path: Path) -> str:
    return ".".join(("floodvalue",) + path.relative_to(SRC).parent.parts)


def test_pipeline_never_reads_messages():
    hits = []
    for pkg in GUARDED:
        for path in sorted((SRC / pkg).rglob("*.py")):
            text = path.read_text(encoding="utf-8")
            hits += [f"{path.relative_to(SRC)}:{line}: {name}"
                     for line, name in _floodvalue_imports(ast.parse(text), _package(path))
                     if name == "floodvalue.media" or name.startswith("floodvalue.media.")]
            hits += [f"{path.relative_to(SRC)}:{i}: {row.strip()}"
                     for i, row in enumerate(text.splitlines(), 1) if FORBIDDEN_TEXT.search(row)]
    assert not hits, "сообщения попали в конвейер модели или экономики:\n" + "\n".join(hits)


def test_media_imports_only_shared_modules():
    hits = []
    for path in sorted((SRC / "media").rglob("*.py")):
        for line, name in _floodvalue_imports(ast.parse(path.read_text(encoding="utf-8")),
                                              _package(path)):
            if name in ALLOWED_FROM_MEDIA or name.startswith("floodvalue.media"):
                continue
            hits.append(f"{path.relative_to(SRC)}:{line}: {name}")
    assert not hits, "media импортирует внутренности других модулей:\n" + "\n".join(hits)


def test_guard_sees_relative_and_package_imports():
    assert _floodvalue_imports(ast.parse("from ..media import api"), "floodvalue.model") \
        == [(1, "floodvalue.media")]
    assert (1, "floodvalue.media") in _floodvalue_imports(
        ast.parse("from floodvalue import media"), "floodvalue.assets")
    assert all(any((SRC / pkg).rglob("*.py")) for pkg in GUARDED)
