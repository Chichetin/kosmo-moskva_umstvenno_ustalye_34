"""Import rules D-005 with the narrow exceptions D-015 (data.io) and D-032 (model.metrics)."""

import ast
from pathlib import Path

import floodvalue

PKG = Path(floodvalue.__file__).parent
# Everything baseline may import from floodvalue: contracts/paths (D-005), the chip reader
# (D-015), the split loader (contracts.md §1b), the shared §7 metrics (D-032), itself.
ALLOWED = {"floodvalue.contracts", "floodvalue.paths", "floodvalue.data.io",
           "floodvalue.data.split", "floodvalue.model.metrics", "floodvalue.baseline"}


def _imports(path: Path) -> set[str]:
    out = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            # `from floodvalue.data import io` -> floodvalue.data.io
            out |= {f"{node.module}.{a.name}" for a in node.names}
    return out


def _floodvalue_imports(path: Path) -> set[str]:
    return {m for m in _imports(path) if m == "floodvalue" or m.startswith("floodvalue.")}


def _allowed(name: str) -> bool:
    return any(name == a or name.startswith(a + ".") for a in ALLOWED)


def test_baseline_imports_only_contracts_paths_and_recorded_exceptions():
    files = sorted((PKG / "baseline").glob("*.py"))
    assert files
    bad = {f.name: sorted(m for m in _floodvalue_imports(f) if not _allowed(m)) for f in files}
    assert not {k: v for k, v in bad.items() if v}


def test_shared_metrics_module_stays_pure():
    """D-032: model.metrics is importable by stages only while it has no I/O and no stage imports."""
    mods = {m.split(".")[0] for m in _imports(PKG / "model" / "metrics.py")}
    assert mods <= {"hashlib", "numpy", "math", "__future__"}
