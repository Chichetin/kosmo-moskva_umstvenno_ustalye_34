import csv
import hashlib
import json

import numpy as np
import pytest

from floodvalue.data import fetch, io

from .conftest import CHIPS, STEP_LABEL, s1_array, write_bucket_manifest, write_chip


def read_qc(root):
    with open(root / "interim/qc.csv", encoding="utf-8") as f:
        return {r["chip_id"]: r for r in csv.DictReader(f)}


def test_run_offline_on_complete_raw_writes_qc_and_manifest(raw, offline, capsys):
    assert fetch.run(None) == 0
    assert "скачано файлов: 0" in capsys.readouterr().out
    qc = read_qc(raw.parent)
    assert sorted(qc) == sorted(CHIPS)
    r = qc["Sri-Lanka_713926"]
    s1 = raw / r["s1_file"]
    assert r["s1_sha256"] == hashlib.sha256(s1.read_bytes()).hexdigest()
    assert r["event_id"] == "Sri-Lanka"
    assert (r["crs"], r["s1_bands"], r["s1_dtype"], r["label_dtype"]) == \
        ("EPSG:4326", "2", "float32", "int16")
    assert r["grid_match"] == "True"  # 1e-17 step noise tolerated
    assert r["s1_md5_ok"] == r["label_md5_ok"] == "True"
    assert r["label_values"] == "-1 0 1" and r["label_values_ok"] == "True"
    assert float(r["vv_nan_frac"]) == pytest.approx(1 / 64)
    assert float(r["label_invalid_frac"]) == pytest.approx(1 / 64)
    assert float(r["valid_frac"]) == pytest.approx(62 / 64)
    assert r["db_scale_ok"] == r["qc_pass"] == r["usable"] == "True"

    doc = json.loads((raw.parent / "interim/source_manifest_s1.json").read_text())
    by_chip = {e["chip_id"]: e for e in doc["sources"]}
    assert by_chip["Mekong_922373"]["observation_date"] == "2018-08-05"  # Mekong -> Cambodia
    e = by_chip["Sri-Lanka_713926"]
    assert e["observation_date"] == "2017-05-30"
    assert e["sha256"] == r["s1_sha256"] and e["label"]["sha256"] == r["label_sha256"]
    assert e["url"].endswith("v1.1/data/flood_events/HandLabeled/S1Hand/"
                             "Sri-Lanka_713926_S1Hand.tif")
    assert e["published_at"] and e["published_after_event"] is True
    for key in ("file", "version", "license", "purpose", "limitations"):
        assert e[key]


def test_second_run_is_offline_and_identical(raw, offline):
    assert fetch.run(None) == 0
    first = (raw.parent / "interim/qc.csv").read_text()
    assert fetch.run(None) == 0
    assert (raw.parent / "interim/qc.csv").read_text() == first


def test_missing_files_are_downloaded_only(raw, monkeypatch):
    src = raw / "HandLabeled/S1Hand/Mekong_922373_S1Hand.tif"
    payload = src.read_bytes()
    src.unlink()
    manifest = json.loads((raw / "bucket_manifest.json").read_text())
    items = [{"name": n, **m} for n, m in manifest.items()]
    monkeypatch.setattr(fetch, "list_objects",
                        lambda prefix: [i for i in items if i["name"].startswith(prefix)])
    got = []

    def fake_download(name, out, md5):
        got.append(name)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(payload)

    monkeypatch.setattr(fetch, "download", fake_download)
    assert fetch.run(None) == 0
    assert got == ["v1.1/data/flood_events/HandLabeled/S1Hand/Mekong_922373_S1Hand.tif"]


def _with(a, idx, v):
    a[idx] = v
    return a


def _qc_one(raw, **chip_kw):
    write_chip(raw, "USA_1", **chip_kw)
    write_bucket_manifest(raw)
    return fetch.qc_chip("USA_1", raw, fetch.load_bucket_manifest(raw))


def test_qc_passes_good_chip(raw):
    r = _qc_one(raw)
    assert r["qc_pass"] and r["usable"], r["issues"]


@pytest.mark.parametrize("kw, issue", [
    ({"s1": np.abs(s1_array()) / 100}, "dB scale"),         # linear power, not dB
    ({"s1": np.stack([np.abs(s1_array()[0]) / 100, s1_array()[1]])}, "dB scale"),  # VV only
    ({"s1": _with(s1_array(), (0, 3, 3), 80.0)}, "dB scale"),  # above DB_MAX
    ({"s1": s1_array()[:1]}, "bands"),                      # one band
    ({"label": np.full((8, 8), 2, np.int16)}, "label values"),
    ({"dtype": "uint8"}, "label dtype"),
    ({"step": STEP_LABEL * 1.01}, "grid"),                  # real grid mismatch
    ({"origin": (80.0 + 8.983e-05, 7.8)}, "grid"),          # shifted by one pixel
    ({"crs": "EPSG:3857"}, "grid"),
])
def test_qc_flags_broken_chip(raw, kw, issue):
    r = _qc_one(raw, **kw)
    assert not r["qc_pass"]
    assert issue in r["issues"]


def test_qc_flags_md5_mismatch(raw):
    write_chip(raw, "USA_1")
    write_bucket_manifest(raw)
    write_chip(raw, "USA_1", s1=s1_array(seed=1))  # file changed after manifest
    r = fetch.qc_chip("USA_1", raw, fetch.load_bucket_manifest(raw))
    assert r["s1_md5_ok"] is False and not r["qc_pass"]


def test_qc_wrong_s1_crs_fails(raw):
    write_chip(raw, "USA_1")
    from .conftest import write_tif
    write_tif(raw / "HandLabeled/S1Hand/USA_1_S1Hand.tif", s1_array(), "float32",
              crs="EPSG:32644")
    write_bucket_manifest(raw)
    r = fetch.qc_chip("USA_1", raw, fetch.load_bucket_manifest(raw))
    assert not r["qc_pass"] and "crs" in r["issues"]


def test_all_nan_s1_is_valid_format_but_unusable_and_csv_empty(raw, offline):
    s1 = np.full((2, 8, 8), np.nan, np.float32)
    write_chip(raw, "Sri-Lanka_713926", s1=s1)
    write_bucket_manifest(raw)
    assert fetch.run(None) == 0
    r = read_qc(raw.parent)["Sri-Lanka_713926"]
    assert r["qc_pass"] == "True" and r["usable"] == "False"
    assert r["db_scale_ok"] == "" and r["vv_min"] == "" and "S1 all NaN" in r["issues"]


def test_all_minus_one_label_unusable(raw):
    r = _qc_one(raw, label=np.full((8, 8), -1, np.int16))
    assert r["qc_pass"] and not r["usable"] and "label all -1" in r["issues"]


def test_missing_label_fails_chip(raw):
    write_chip(raw, "USA_1")
    io.layer_path("USA_1", "LabelHand").unlink()
    r = fetch.qc_chip("USA_1", raw, {})
    assert not r["qc_pass"] and "LabelHand missing" in r["issues"]


def test_run_returns_1_on_qc_failure(raw, offline):
    write_chip(raw, "Sri-Lanka_713926", label=np.full((8, 8), 3, np.int16))
    write_bucket_manifest(raw)
    assert fetch.run(None) == 1


def test_inventory_rejects_mismatched_pair(raw):
    (raw / "splits/flood_handlabeled/flood_test_data.csv").write_text(
        "USA_1_S1Hand.tif,USA_2_LabelHand.tif\n")
    with pytest.raises(ValueError):
        fetch.chip_inventory(raw)


def test_event_date_alias_and_unknown():
    dates = {"Cambodia": "2018-08-05"}
    assert fetch.event_s1_date("Mekong", dates) == "2018-08-05"
    assert fetch.event_s1_date("Atlantis", dates) is None
