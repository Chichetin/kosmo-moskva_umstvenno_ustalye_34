from floodvalue import cli


def test_every_documented_stage_is_registered():
    expected = {"data", "split", "baseline", "train", "calibrate", "predict", "assets",
                "procure", "export", "validate", "app", "eval-test"}
    assert set(cli.STAGES) == expected


def test_unimplemented_stage_exits_2_and_names_area(monkeypatch, capsys):
    monkeypatch.setitem(cli.STAGES, "assets", ("floodvalue.assets.not_there", "x"))
    assert cli.main(["assets"]) == 2
    assert "area:assets" in capsys.readouterr().out
