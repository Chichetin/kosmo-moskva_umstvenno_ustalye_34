"""Import rules for the whole package (D-005) with every recorded exception in one table.

Modules talk through files by contract; a module imports only `floodvalue.contracts`,
`floodvalue.paths`, itself and the narrow exceptions below, each with its record. The walk
covers every .py under src/floodvalue, lazy imports inside functions, relative imports and
module-path string literals (the `METHODS`/`STAGES` registries are counted in the file that
declares them). A module imported whole (`from floodvalue.data import io`) is checked by the
attributes used on it (`io.load_s1` -> `floodvalue.data.io.load_s1`), so a narrow entry stays
narrow. Names that open raw files or serve only eval-test are further limited to the files
that use them (FILES): e.g. `data.io.layer_path` could reach LabelHand around the sealed
`load_label` (D-010, D-015), so a new caller must be a reviewed line here.

Not seen by the walk (D-057): what a caller does with a module it got from a registry through
importlib (calibration -> `model.predict.METHODS`, bounded by D-041; model.eval_test ->
`baseline.predict`) and module names built by f-strings. A new cross-module import fails
here until its line is added to ALLOWED (and FILES, if narrowed) with a D-… record.
The older AST tests stay: tests/baseline/test_baseline_imports.py,
tests/calibration/test_calibration_imports.py, tests/context/test_firewall.py (D-042),
tests/media/test_media_firewall.py (D-051).
"""

import ast
import re
from pathlib import Path

import floodvalue

PKG = Path(floodvalue.__file__).parent

BASE = {"floodvalue.contracts", "floodvalue.paths"}  # D-005
# Chip reader, read only (data.io). D-015: load_s1, load_label, valid_mask; D-057: path and
# id helpers and layer names of the same reader.
IO = "floodvalue.data.io."
# Split loader, read only (data.split). contracts.md §4: chip_ids; D-057: the rest.
SPLIT = "floodvalue.data.split."

ALLOWED: dict[str, set[str]] = {
    "__init__": set(),
    "paths": set(),
    "contracts": {"floodvalue.paths"},  # D-005: contracts is the shared base
    "__main__": {"floodvalue.cli.main"},
    # D-057 (project-structure.md): cli loads the stage modules lazily by name (STAGES)
    "cli": {"floodvalue.data.fetch", "floodvalue.data.split", "floodvalue.baseline.run",
            "floodvalue.model.train", "floodvalue.calibration.run", "floodvalue.model.predict",
            "floodvalue.assets.run", "floodvalue.procurement.run", "floodvalue.export.run",
            "floodvalue.export.validate", "floodvalue.app.run", "floodvalue.model.eval_test"},
    "data": BASE,
    "baseline": BASE | {
        IO + "load_s1", IO + "load_label", IO + "valid_mask",  # D-015
        IO + "layer_path", IO + "S1_LAYER",  # D-057
        SPLIT + "chip_ids",  # contracts.md §4
        SPLIT + "load_split",  # D-057
        "floodvalue.model.metrics",  # D-032
    },
    "model": BASE | {
        IO + "load_s1", IO + "load_label", IO + "valid_mask",  # D-015
        IO + "layer_path", IO + "raw_dir", IO + "parse_chip_id",  # D-057
        IO + "S1_LAYER", IO + "JRC_LAYER",  # D-057 (eval-test)
        SPLIT + "chip_ids",  # contracts.md §4
        SPLIT + "load_split", SPLIT + "SealedTestError",  # D-057
        SPLIT + "SPLIT_PATH", SPLIT + "event_of",  # D-057 (eval-test)
        "floodvalue.calibration.calibrator",  # D-041: p_raw -> p, lazy, contracts §3
        "floodvalue.calibration.uncertainty",  # D-039: uncertainty.tif, lazy
        "floodvalue.baseline.predict",  # D-057: METHODS registry (contracts §3), unet report
        "floodvalue.cli.main",  # D-030: eval-test runs the assets stage through the CLI
    },
    "calibration": BASE | {
        IO + "load_s1", IO + "load_label", IO + "valid_mask",  # D-015
        IO + "layer_path", IO + "parse_chip_id",  # D-057
        SPLIT + "chip_ids",  # contracts.md §4
        SPLIT + "load_split",  # D-057
        "floodvalue.model.metrics",  # D-032
        "floodvalue.model.predict.METHODS",  # D-041
    },
    "assets": BASE,
    "procurement": BASE,
    "export": BASE,
    "app": BASE | {
        "floodvalue.procurement.plan",  # contracts.md §10, D-005, D-050
        SPLIT + "EVENT_PARTS",  # D-050: test chips stay sealed in the catalog
    },
    "context": BASE | {
        IO + "layer_path", IO + "raw_dir", IO + "parse_chip_id",  # D-057
        SPLIT + "EVENT_PARTS", SPLIT + "SealedTestError",  # D-050, D-057
    },
    "media": {"floodvalue.paths"},  # D-051: media takes only paths
}

# Entries of ALLOWED usable only from these files (paths relative to src/floodvalue). D-057.
FILES: dict[str, set[str]] = {
    IO + "layer_path": {"baseline/predict.py",  # S1 file name in metadata
                        "calibration/run.py",  # JRCWaterHand, diagnostic only
                        "model/eval_test.py",  # S1Hand, JRCWaterHand
                        "model/pixel/predict.py",  # S1 file name in metadata
                        "context/s2.py"},  # S2Hand, S1Hand (D-042)
    IO + "raw_dir": {"model/unet/loeo.py",  # WeaklyLabeled S1 (transfer experiment)
                     "context/s2.py"},  # bucket manifest, metadata
    IO + "JRC_LAYER": {"model/eval_test.py"},
    SPLIT + "SPLIT_PATH": {"model/eval_test.py"},
    SPLIT + "event_of": {"model/eval_test.py"},
    "floodvalue.cli.main": {"model/eval_test.py", "__main__.py"},
    "floodvalue.baseline.predict": {"model/predict.py",  # METHODS registry, contracts §3
                                    "model/unet/report.py"},  # U-Net vs baseline report
}

MODULE_STRING = re.compile(r"floodvalue(\.[A-Za-z_]\w*)+")


def _is_module(dotted: str) -> bool:
    rel = Path(*dotted.split(".")[1:]) if "." in dotted else Path()
    return (PKG / rel).with_suffix(".py").is_file() or (PKG / rel / "__init__.py").is_file()


def _package_of(path: Path) -> str:
    parts = path.relative_to(PKG).with_suffix("").parts
    parts = parts[:-1]  # the package a relative import resolves against
    return ".".join(("floodvalue", *parts))


def imports_of(text: str, package: str) -> list[tuple[int, str, bool]]:
    """(line, dotted name, is a whole-module import) of every floodvalue reference in `text`.

    `package` is the dotted package of the file, used for relative imports. A whole module
    bound to a name is returned once as the import itself and once per attribute used on it."""
    tree = ast.parse(text)
    out: list[tuple[int, str, bool]] = []
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                out.append((node.lineno, a.name, False))
                if a.asname and _is_module(a.name):
                    aliases[a.asname] = a.name
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                parent = package.rsplit(".", node.level - 1)[0] if node.level > 1 else package
                base = f"{parent}.{base}" if base else parent
            for a in node.names:
                name = f"{base}.{a.name}"
                whole = name.startswith("floodvalue.") and _is_module(name)
                out.append((node.lineno, name, whole))
                if whole:
                    aliases[a.asname or a.name] = name
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if MODULE_STRING.fullmatch(node.value):
                out.append((node.lineno, node.value, False))
    attr_bases = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and node.value.id in aliases):
            attr_bases.add(id(node.value))
            out.append((node.lineno, f"{aliases[node.value.id]}.{node.attr}", False))
    for node in ast.walk(tree):
        if (isinstance(node, ast.Name) and node.id in aliases and id(node) not in attr_bases
                and isinstance(node.ctx, ast.Load)):
            out.append((node.lineno, aliases[node.id], False))  # module passed around bare
    return [(ln, n, w) for ln, n, w in out if n == "floodvalue" or n.startswith("floodvalue.")]


def top_of(path: Path) -> str:
    return path.relative_to(PKG).parts[0].removesuffix(".py")


def allowed(top: str, name: str, whole: bool, rel: str = "") -> bool:
    """`rel` is the file path relative to src/floodvalue, checked against FILES."""
    if name.split(".")[1:2] == [top]:
        return True  # inside its own module
    for a in ALLOWED[top]:
        if name == a or name.startswith(a + "."):
            if a in FILES and rel not in FILES[a]:
                continue
            return True
        if whole and a.startswith(name + "."):
            return True  # narrow entry inside a module imported whole; attributes checked
    return False


def violations(path: Path, text: str | None = None) -> list[str]:
    text = path.read_text(encoding="utf-8") if text is None else text
    top, rel = top_of(path), path.relative_to(PKG).as_posix()
    found = sorted(set(imports_of(text, _package_of(path))))
    return [f"floodvalue/{rel}:{ln} {name}" for ln, name, whole in found
            if not allowed(top, name, whole, rel)]


def _files() -> list[Path]:
    return sorted(p for p in PKG.rglob("*.py") if "__pycache__" not in p.parts)


def test_every_top_level_module_has_a_table():
    assert {top_of(p) for p in _files()} == set(ALLOWED)


def test_package_imports_follow_recorded_exceptions():
    bad = [v for p in _files() for v in violations(p)]
    assert not bad, "imports not recorded in ALLOWED (D-005):\n" + "\n".join(bad)


def test_every_exception_is_still_used():
    """A stale line widens the rule silently: each entry must match a real import."""
    used = {top: set() for top in ALLOWED}
    for p in _files():
        top = top_of(p)
        for _, name, whole in imports_of(p.read_text(encoding="utf-8"), _package_of(p)):
            if whole:
                continue
            used[top] |= {a for a in ALLOWED[top] if name == a or name.startswith(a + ".")}
    unused = {top: sorted(ALLOWED[top] - used[top] - BASE) for top in ALLOWED}
    assert not {k: v for k, v in unused.items() if v}


def test_every_file_limit_is_used_and_listed():
    used = set()
    for p in _files():
        rel = p.relative_to(PKG).as_posix()
        for _, name, whole in imports_of(p.read_text(encoding="utf-8"), _package_of(p)):
            used |= {(a, rel) for a in FILES
                     if not whole and (name == a or name.startswith(a + "."))}
    declared = {(a, rel) for a, rels in FILES.items() for rel in rels}
    assert declared - used == set()
    assert all(any(a in ALLOWED[top] for top in ALLOWED) for a in FILES)


def test_lazy_relative_aliased_and_registry_imports_are_seen():
    path = PKG / "app" / "run.py"
    text = ("def f():\n"
            "    from floodvalue.procurement import strategies\n"
            "    from ..export import assemble\n"
            "    from floodvalue.data import io as chip\n"
            "    return chip.load_s1, chip.build_split\n"
            "REG = {'x': 'floodvalue.model.unet.predict'}\n"
            "from floodvalue.procurement import plan\n"
            "from floodvalue.data.split import EVENT_PARTS\n")
    got = violations(path, text)
    assert got == ["floodvalue/app/run.py:2 floodvalue.procurement.strategies",
                   "floodvalue/app/run.py:3 floodvalue.export.assemble",
                   "floodvalue/app/run.py:4 floodvalue.data.io",
                   "floodvalue/app/run.py:5 floodvalue.data.io.build_split",
                   "floodvalue/app/run.py:5 floodvalue.data.io.load_s1",
                   "floodvalue/app/run.py:6 floodvalue.model.unet.predict"]


def test_module_imported_whole_is_checked_by_attribute():
    path = PKG / "baseline" / "predict.py"
    ok = "from floodvalue.data import io\nx = io.load_s1\n"
    assert violations(path, ok) == []
    bad = "from floodvalue.data import io\nx = io.interim_dir\ny = io\n"
    assert violations(path, bad) == ["floodvalue/baseline/predict.py:2 floodvalue.data.io.interim_dir",
                                     "floodvalue/baseline/predict.py:3 floodvalue.data.io"]


def test_raw_file_helpers_stay_with_their_files():
    """layer_path next to a sealed label must not spread inside an allowed module (D-010)."""
    text = "from floodvalue.data.io import layer_path\np = layer_path('x', 'LabelHand')\n"
    assert violations(PKG / "model" / "unet" / "train.py", text) == [
        "floodvalue/model/unet/train.py:1 floodvalue.data.io.layer_path"]
    assert violations(PKG / "model" / "pixel" / "predict.py", text) == []
