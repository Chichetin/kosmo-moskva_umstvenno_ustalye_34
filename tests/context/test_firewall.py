"""Context imagery is for people, never for the model or the economics (D-042).

No module that produces p, the mask, thresholds, EL or the procurement plan — nor `data`, which
feeds them — may import the context package or reference optical/basemap context. A new
reference here must be a deliberate, reviewed decision (and then S1-only vs S1+S2 has to be
compared, criterion 8), not an accident."""

import ast
import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "floodvalue"
GUARDED = ["data", "model", "baseline", "calibration", "assets", "procurement"]
FORBIDDEN_TEXT = re.compile(
    r"S2Hand|S2_LAYER|load_s2|context_s2|s2cloudless|World_Imagery|tile\.openstreetmap")


def _imports_context(tree: ast.AST, package: str) -> list[int]:
    """Line numbers of imports that reach floodvalue.context, in any spelling. `package` is the
    dotted package of the file (e.g. "floodvalue.model.unet"), used for relative imports."""
    parts = package.split(".")
    lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(a.name == "floodvalue.context" or a.name.startswith("floodvalue.context.")
                   for a in node.names):
                lines.append(node.lineno)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if node.level == 0:
                full = mod
            else:  # relative: level 1 = the file's package, each extra dot one level up
                base = parts[: max(0, len(parts) - (node.level - 1))]
                full = ".".join(base + ([mod] if mod else []))
            names = {a.name for a in node.names}
            if (full == "floodvalue.context" or full.startswith("floodvalue.context.")
                    or (full == "floodvalue" and "context" in names)):
                lines.append(node.lineno)
    return lines


def test_pipeline_modules_never_touch_context_imagery():
    hits = []
    for pkg in GUARDED:
        for f in sorted((SRC / pkg).rglob("*.py")):
            text = f.read_text(encoding="utf-8")
            package = ".".join(("floodvalue",) + f.relative_to(SRC).parent.parts)
            hits += [f"{f.relative_to(SRC)}:{n}: import floodvalue.context"
                     for n in _imports_context(ast.parse(text), package)]
            hits += [f"{f.relative_to(SRC)}:{i}: {line.strip()}"
                     for i, line in enumerate(text.splitlines(), 1) if FORBIDDEN_TEXT.search(line)]
    assert not hits, "контекстные снимки попали в конвейер модели/экономики:\n" + "\n".join(hits)


def test_guard_catches_every_import_spelling():
    for code in ("import floodvalue.context", "import floodvalue.context.s2 as s",
                 "from floodvalue import context", "from floodvalue.context import s2",
                 "from floodvalue.context.s2 import build", "from ..context import s2",
                 "from .. import context"):
        assert _imports_context(ast.parse(code), "floodvalue.model"), code
    for code in ("from ...context import s2", "from ... import context"):
        assert _imports_context(ast.parse(code), "floodvalue.model.unet"), code
    for code in ("from floodvalue import contracts", "import floodvalue.contracts",
                 "from ..contracts import schemas", "from .context_free import x"):
        assert not _imports_context(ast.parse(code), "floodvalue.model"), code
    assert not _imports_context(ast.parse("from ..context import x"), "floodvalue.model.unet")


def test_guard_actually_scans_code():
    # A guard over empty directories would pass forever.
    assert all(any((SRC / pkg).rglob("*.py")) for pkg in GUARDED)
    assert (SRC / "context" / "s2.py").read_text(encoding="utf-8").count("S2Hand") > 0
