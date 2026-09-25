"""Colab training wrapper notebooks/train_colab.ipynb (issue #15, criterion 11).

The notebook must only call the CLI (`python -m floodvalue data`, `python -m floodvalue
train`) — no training logic inside — and must not carry outputs or secrets in git.
"""

import json
import re
import shlex

import pytest

from floodvalue.paths import REPO_ROOT

NB_PATH = REPO_ROOT / "notebooks" / "train_colab.ipynb"


def load_nb() -> dict:
    return json.loads(NB_PATH.read_text(encoding="utf-8"))


def code_cells(nb: dict) -> list[str]:
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def test_notebook_is_valid_nbformat4_without_outputs():
    nb = load_nb()
    assert nb["nbformat"] == 4
    for c in nb["cells"]:
        assert c["cell_type"] in ("code", "markdown")
        if c["cell_type"] == "code":
            assert c["outputs"] == [] and c["execution_count"] is None


def test_code_cells_are_plain_python():
    # No IPython magics (`!`, `%`): cells run the same via exec outside Jupyter.
    for i, src in enumerate(code_cells(load_nb())):
        compile(src, f"cell{i}", "exec")


def test_notebook_calls_the_cli_and_has_no_training_logic():
    src = "\n".join(code_cells(load_nb()))
    assert "-m floodvalue data" in src
    assert "-m floodvalue train" in src
    assert "scripts/setup_gpu_env.sh" in src
    assert "{UV} sync --frozen" in src
    for forbidden in ("import torch\n", "backward(", "optimizer", "state_dict", "torch.save"):
        assert forbidden not in src, forbidden


def test_notebook_has_no_embedded_secret_and_no_test_split():
    text = NB_PATH.read_text(encoding="utf-8")
    assert not re.search(r"gh[pousr]_[A-Za-z0-9]{20,}", text)
    assert "allow_test" not in text and "eval-test" not in text


def run_setup_cells(monkeypatch, tmp_path) -> dict:
    monkeypatch.setenv("FLOODVALUE_NB_WORKDIR", str(tmp_path))
    ns: dict = {}
    for src in code_cells(load_nb())[:2]:  # parameters + sh helper, no side effects
        exec(compile(src, "cell", "exec"), ns)
    return ns


def test_parameters_are_overridable_by_env(monkeypatch, tmp_path):
    monkeypatch.setenv("FLOODVALUE_NB_MODE", "smoke")
    ns = run_setup_cells(monkeypatch, tmp_path)
    assert ns["MODE"] == "smoke"
    assert ns["DATA_DIR"] == str(tmp_path / "data")
    assert ns["GPU_VENV"].startswith(str(tmp_path))  # never the shared $HOME/.cache venv


def test_sh_fails_the_cell_on_error_and_masks_secret(monkeypatch, tmp_path, capsys):
    ns = run_setup_cells(monkeypatch, tmp_path)
    ns["sh"]("echo token-XYZ", secret="token-XYZ")
    out = capsys.readouterr().out
    assert "token-XYZ" not in out and "***" in out
    with pytest.raises(RuntimeError):
        ns["sh"]("false")
    with pytest.raises(RuntimeError):
        ns["sh"]("false | cat")  # pipefail


def cell_with(marker: str) -> str:
    cells = [src for src in code_cells(load_nb()) if marker in src]
    assert len(cells) == 1, marker
    return cells[0]


def recorded_commands(monkeypatch, tmp_path, mode: str) -> list[str]:
    """Execute the data and training cells with `sh` replaced by a recorder."""
    monkeypatch.setenv("FLOODVALUE_NB_MODE", mode)
    ns = run_setup_cells(monkeypatch, tmp_path)
    calls: list[str] = []
    ns["sh"] = lambda cmd, **kw: calls.append(cmd)
    ns.update(PY="python", ENV={})
    for marker in ('sh(f"{PY} -m floodvalue data', 'sh(f"{PY} -m floodvalue train'):
        exec(compile(cell_with(marker), "cell", "exec"), ns)
    return calls


def parse_floodvalue_command(cmd: str):
    """Parse `python -m <module> <args>` with the real argparse parser of that module."""
    from floodvalue import cli

    argv = shlex.split(cmd)
    assert argv[:2] == ["python", "-m"], cmd
    module, rest = argv[2], argv[3:]
    if module == "floodvalue":
        assert rest[0] in cli.STAGES, rest[0]
        return module, cli.build_parser().parse_args(rest)
    assert module == "floodvalue.model.unet.train", module
    from floodvalue.model.unet import train as unet_train

    return module, unet_train.parse_args(rest)


def test_full_mode_commands_match_the_cli_of_main(monkeypatch, tmp_path):
    calls = recorded_commands(monkeypatch, tmp_path, "full")
    parsed = [parse_floodvalue_command(c) for c in calls]  # argparse exits on unknown flags
    assert [(m, a.stage) for m, a in parsed] == [("floodvalue", "data"), ("floodvalue", "train")]
    assert parsed[1][1].out == str(tmp_path / "outputs")


def test_smoke_mode_uses_real_flags_of_unet_train(monkeypatch, tmp_path):
    calls = recorded_commands(monkeypatch, tmp_path, "smoke")
    (_, data), (module, a) = [parse_floodvalue_command(c) for c in calls]
    assert data.stage == "data" and module == "floodvalue.model.unet.train"
    assert (a.epochs, a.max_chips, a.batch) == (1, 4, 2)
    assert a.out_dir == str(tmp_path / "outputs" / "unet-smoke")


def test_unknown_flag_is_rejected_by_the_parser():
    # Guard for the two tests above: a flag that main does not know must fail parsing.
    with pytest.raises(SystemExit):
        parse_floodvalue_command("python -m floodvalue train --epochs 1")
    with pytest.raises(SystemExit):
        parse_floodvalue_command("python -m floodvalue.model.unet.train --smoke 4")
