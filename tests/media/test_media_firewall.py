"""Signals decide where to look, never what the model outputs: no FloodValue stage imports
floodvalue.media, and media imports nothing of FloodValue except floodvalue.paths. Media
writes only under FLOODVALUE_MEDIA_DIR (a SQLite file and photos), never into run outputs."""

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "floodvalue"


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module)
            names |= {f"{node.module}.{a.name}" for a in node.names}
    return names


def test_no_stage_imports_media():
    offenders = []
    for path in SRC.rglob("*.py"):
        if "media" in path.relative_to(SRC).parts:
            continue
        if any(n == "floodvalue.media" or n.startswith("floodvalue.media.") for n in _imports(path)):
            offenders.append(str(path))
        if "floodvalue.media" in path.read_text(encoding="utf-8"):
            offenders.append(str(path))
    assert not offenders


def test_media_imports_only_paths():
    for path in (SRC / "media").rglob("*.py"):
        for name in _imports(path):
            if name.startswith("floodvalue.") and not name.startswith("floodvalue.media"):
                assert name in ("floodvalue.paths", "floodvalue.paths.REPO_ROOT",
                                "floodvalue.paths.output_dir", "floodvalue.paths.data_dir"), (path, name)


def test_media_writes_only_its_own_dir(settings, conn, tmp_path, monkeypatch):
    out = tmp_path / "outputs"
    out.mkdir()
    monkeypatch.setenv("FLOODVALUE_OUTPUT_DIR", str(out))
    from floodvalue.media.demo import load_demo
    load_demo(conn, settings)
    assert list(out.iterdir()) == []
    assert settings.db_path.is_relative_to(tmp_path / "media")


def test_form_parsing_dependency_is_direct():
    """/report and /media/login parse multipart forms: python-multipart must not ride on another
    package (it came only via streamlit, which is being removed — PR #169)."""
    import tomllib
    pyproject = tomllib.loads((SRC.parents[1] / "pyproject.toml").read_text(encoding="utf-8"))
    names = [d.split(">")[0].split("<")[0].split("=")[0].strip().lower()
             for d in pyproject["project"]["dependencies"]]
    assert "python-multipart" in names
