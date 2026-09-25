"""Split v1 (criterion 6): events never cross parts, 446 chips accounted, test sealed."""

import json
from argparse import Namespace

import pytest

from floodvalue.data import split as S
from floodvalue.paths import data_dir

RAW = data_dir() / "raw"
needs_raw = pytest.mark.skipif(
    not (RAW / "HandLabeled" / "LabelHand").is_dir(), reason="Sen1Floods11 raw data not present"
)


@pytest.fixture(scope="module")
def frozen():
    return S.load_split()


def test_file_follows_contract(frozen):
    assert frozen["version"] == 1 and frozen["criterion"] == "event"
    assert frozen["seed"] == S.SEED and frozen["source"] == S.SOURCE
    assert set(frozen["events"]) == set(S.PARTS)
    assert frozen["sha256"] == S.canonical_hash(frozen)


def test_events_do_not_cross_parts(frozen):
    seen: dict[str, str] = {}
    for part in S.PARTS:
        for chip in frozen[part]:
            event = S.event_of(chip)
            assert event in frozen["events"][part], (chip, part)
            assert seen.setdefault(event, part) == part, f"{event} in {seen[event]} and {part}"
    listed = [e for p in S.PARTS for e in frozen["events"][p]]
    assert len(listed) == len(set(listed)) == 11


def test_chips_unique_and_all_446_accounted(frozen):
    chips = [c for p in S.PARTS for c in frozen[p]]
    assert len(chips) == len(set(chips))
    assert not set(chips) & set(frozen["excluded"])
    assert len(chips) + len(frozen["excluded"]) == 446
    assert all(reason for reason in frozen["excluded"].values())


def test_expected_sizes_and_exclusions(frozen):
    sizes = {p: len(frozen[p]) for p in S.PARTS}
    assert sizes == {"train": 237, "dev": 69, "validation": 60, "test": 75}
    assert sorted(frozen["excluded"]) == [
        "Ghana_234935", "Ghana_26376", "Ghana_277", "Ghana_5079", "Ghana_83483"]


def test_hash_matches_decisions_log(frozen):
    assert S.decisions_hash() == frozen["sha256"]


def test_hash_covers_every_part(frozen):
    for part in S.PARTS:
        moved = json.loads(json.dumps(frozen))
        moved[part] = moved[part][1:]
        assert S.canonical_hash(moved) != frozen["sha256"], part


def test_loader_open_parts(frozen):
    for part in S.OPEN_PARTS:
        assert S.chip_ids(part) == frozen[part]


def test_loader_seals_test(frozen):
    with pytest.raises(S.SealedTestError):
        S.chip_ids("test")
    assert S.chip_ids("test", allow_test=True) == frozen["test"]
    with pytest.raises(KeyError):
        S.chip_ids("valid")


def test_loader_rejects_tampered_file(tmp_path, frozen):
    bad = dict(frozen, train=frozen["train"] + [frozen["test"][0]])
    path = tmp_path / "split.json"
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError):
        S.chip_ids("train", path=path)


def test_split_never_reads_test_labels():
    with pytest.raises(S.SealedTestError):
        S.all_invalid(RAW, "Spain_7370579")


def test_unknown_event_is_an_error():
    with pytest.raises(KeyError):
        S.part_of_event("Colombia")


@needs_raw
def test_rebuild_is_deterministic_and_equals_frozen(frozen):
    a, b = S.build_split(seed=S.SEED), S.build_split(seed=S.SEED)
    assert S.dumps(a) == S.dumps(b) == S.SPLIT_PATH.read_text(encoding="utf-8")
    assert a["sha256"] == frozen["sha256"]


@needs_raw
def test_build_does_not_open_test_labels(monkeypatch):
    opened = []
    real = S.all_invalid
    monkeypatch.setattr(S, "all_invalid", lambda raw, c: opened.append(c) or real(raw, c))
    S.build_split()
    assert opened and not [c for c in opened if S.part_of_event(S.event_of(c)) == "test"]


@needs_raw
def test_make_split_verifies(capsys):
    assert S.run(Namespace(chip=None, budget=None, out=None)) == 0
    assert "OK" in capsys.readouterr().out


@needs_raw
def test_event_footprints_disjoint():
    S.check_events_disjoint(RAW)
