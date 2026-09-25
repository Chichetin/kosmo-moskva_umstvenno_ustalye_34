"""Colab wrapper notebooks/loeo_colab.ipynb (#86, D-036): only calls the CLI, no training
logic, no outputs/secrets, never touches the test split; its commands parse with the real
parser of `floodvalue.model.unet.loeo`."""

import json
import re
import shlex

from floodvalue.model.unet import loeo
from floodvalue.paths import REPO_ROOT

NB_PATH = REPO_ROOT / "notebooks" / "loeo_colab.ipynb"


def load_nb() -> dict:
    return json.loads(NB_PATH.read_text(encoding="utf-8"))


def code_cells(nb: dict) -> list[str]:
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def test_notebook_is_clean_nbformat4_plain_python_with_unique_ids():
    nb = load_nb()
    assert nb["nbformat"] == 4
    ids = [c.get("id") for c in nb["cells"]]
    assert all(ids) and len(set(ids)) == len(ids)
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            assert c["outputs"] == [] and c["execution_count"] is None
    for i, src in enumerate(code_cells(nb)):
        compile(src, f"cell{i}", "exec")  # no IPython magics


def test_notebook_calls_only_the_cli_and_never_the_test_split():
    text = NB_PATH.read_text(encoding="utf-8")
    src = "\n".join(code_cells(load_nb()))
    assert "-m floodvalue.model.unet.loeo" in src and "-m floodvalue data" in src
    for forbidden in ("import torch\n", "backward(", "optimizer", "state_dict", "torch.save"):
        assert forbidden not in src, forbidden
    assert "allow_test" not in text and "eval-test" not in text
    assert not re.search(r"gh[pousr]_[A-Za-z0-9]{20,}", text)


def recorded_commands(monkeypatch, tmp_path, mode: str) -> list[str]:
    monkeypatch.setenv("FLOODVALUE_NB_MODE", mode)
    monkeypatch.setenv("FLOODVALUE_NB_WORKDIR", str(tmp_path))
    cells = code_cells(load_nb())
    ns: dict = {}
    for src in cells[:2]:  # parameters + sh helper, no side effects
        exec(compile(src, "cell", "exec"), ns)  # noqa: S102 - notebook cells under test
    calls: list[str] = []
    ns["sh"] = lambda cmd, **kw: calls.append(cmd)
    ns.update(PY="python", ENV={}, REPO_DIR=str(tmp_path))
    (run,) = [s for s in cells if "-m floodvalue.model.unet.loeo --out-dir" in s]
    exec(compile(run, "cell", "exec"), ns)  # noqa: S102
    return calls


def parse(cmd: str):
    argv = shlex.split(cmd.split("2>&1")[0])
    assert argv[:3] == ["python", "-m", "floodvalue.model.unet.loeo"], cmd
    return loeo.parse_args(argv[3:])  # argparse exits on an unknown flag


def test_full_mode_runs_the_default_protocol(monkeypatch, tmp_path):
    run, table = [parse(c) for c in recorded_commands(monkeypatch, tmp_path, "full")]
    assert (run.epochs, run.eval_epochs, run.events, run.max_chips) == (60, "19,60", "", 0)
    assert run.out_dir == str(tmp_path / "outputs" / "loeo")
    assert table.table_from == run.out_dir


def test_smoke_mode_is_small_and_separate(monkeypatch, tmp_path):
    run, _ = [parse(c) for c in recorded_commands(monkeypatch, tmp_path, "smoke")]
    assert (run.events, run.max_chips, run.epochs, run.eval_epochs) == ("USA", 4, 2, "1,2")
    assert run.out_dir.endswith("loeo-smoke")
