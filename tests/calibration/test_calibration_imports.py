"""Import rules D-005 with the narrow exceptions D-015, D-032 and D-041 for calibration."""

import ast
from pathlib import Path

import floodvalue

PKG = Path(floodvalue.__file__).parent
# contracts/paths (D-005), chip reader (D-015), split loader (contracts.md §4), shared §7
# metrics (D-032), the method registry (D-041), itself.
ALLOWED = {"floodvalue.contracts", "floodvalue.paths", "floodvalue.data.io",
           "floodvalue.data.split", "floodvalue.model.metrics", "floodvalue.model.predict.METHODS",
           "floodvalue.calibration"}


def _imports(path: Path) -> set[str]:
    out = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, f"{path.name}: relative import"
            out |= {f"{node.module}.{a.name}" for a in node.names}
    return out


def _allowed(name: str) -> bool:
    return any(name == a or name.startswith(a + ".") for a in ALLOWED)


def test_calibration_imports_only_contracts_paths_and_recorded_exceptions():
    files = sorted((PKG / "calibration").glob("*.py"))
    assert files
    bad = {f.name: sorted(m for m in _imports(f)
                          if m.startswith("floodvalue.") and not _allowed(m)) for f in files}
    assert not {k: v for k, v in bad.items() if v}


def test_model_predict_is_the_only_model_module_besides_metrics_that_calibration_takes():
    model = {m for f in (PKG / "calibration").glob("*.py") for m in _imports(f)
             if m.startswith("floodvalue.model")}
    assert model <= {"floodvalue.model.metrics", "floodvalue.model.predict.METHODS"} | {
        m for m in model if m.startswith("floodvalue.model.metrics.")}
