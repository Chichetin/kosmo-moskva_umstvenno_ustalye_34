import csv
import hashlib
import json

import numpy as np
import pytest

from floodvalue.data import fetch, io

from .conftest import (CHIPS, STEP_LABEL, label_array, s1_array, write_bucket_manifest,
                       write_chip)


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

    doc = json.loads((raw.parent / "interim/source_manifest_s1.json").read_text(encoding="utf-8"))
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
    first = (raw.parent / "interim/qc.csv").read_text(encoding="utf-8")
    assert fetch.run(None) == 0
    assert (raw.parent / "interim/qc.csv").read_text(encoding="utf-8") == first


def test_missing_files_are_downloaded_only(raw, monkeypatch):
    src = raw / "HandLabeled/S1Hand/Mekong_922373_S1Hand.tif"
    payload = src.read_bytes()
    src.unlink()
    manifest = json.loads((raw / "bucket_manifest.json").read_text(encoding="utf-8"))
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


@pytest.mark.parametrize("layer", ["S1Hand", "JRCWaterHand"])
def test_qc_flags_md5_mismatch_any_layer(raw, layer):
    write_chip(raw, "USA_1")
    write_bucket_manifest(raw)
    p = raw / f"HandLabeled/{layer}/USA_1_{layer}.tif"
    p.write_bytes(p.read_bytes() + b"x")
    r = fetch.qc_chip("USA_1", raw, fetch.load_bucket_manifest(raw))
    assert not r["qc_pass"] and f"md5 mismatch: {layer}" in r["issues"]


def test_qc_checks_s1_nodata_and_band_names(raw):
    from .conftest import write_tif
    write_chip(raw, "USA_1")
    write_tif(raw / "HandLabeled/S1Hand/USA_1_S1Hand.tif", s1_array(), "float32",
              nodata=-9999.0, descriptions=("HH", "HV"))
    r = fetch.qc_chip("USA_1", raw, {})
    assert not r["qc_pass"] and "nodata" in r["issues"] and "band names" in r["issues"]


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
        "USA_1_S1Hand.tif,USA_2_LabelHand.tif\n", encoding="utf-8")
    with pytest.raises(ValueError):
        fetch.chip_inventory(raw)


def test_event_date_alias_and_unknown():
    dates = {"Cambodia": "2018-08-05"}
    assert fetch.event_s1_date("Mekong", dates) == "2018-08-05"
    assert fetch.event_s1_date("Atlantis", dates) is None


def test_test_chip_label_pixels_are_never_read(raw, offline):
    # Sealed test chip with an impossible label value: QC must not see it.
    write_chip(raw, "Mekong_922373", label=np.full((8, 8), 7, np.int16))
    write_bucket_manifest(raw)
    assert fetch.run(None) == 0
    r = read_qc(raw.parent)["Mekong_922373"]
    assert r["label_checked"] == "False"
    assert r["label_values"] == r["label_values_ok"] == r["label_invalid_frac"] == ""
    assert r["valid_frac"] == "" and "sealed" in r["issues"]
    assert r["label_sha256"] and r["grid_match"] == "True" and r["qc_pass"] == "True"
    assert read_qc(raw.parent)["Sri-Lanka_713926"]["label_checked"] == "True"


def test_test_chip_label_header_still_checked(raw, offline):
    write_chip(raw, "Mekong_922373", dtype="uint8")
    write_bucket_manifest(raw)
    assert fetch.run(None) == 1
    r = read_qc(raw.parent)["Mekong_922373"]
    assert "label dtype" in r["issues"] and r["label_values"] == ""


def test_no_split_file_reads_no_label_pixels(raw, offline, monkeypatch):
    monkeypatch.setattr(fetch, "SPLIT_PATH", raw / "absent.json")
    assert fetch.label_readable_chips() == set()
    assert fetch.run(None) == 0
    assert {r["label_checked"] for r in read_qc(raw.parent).values()} == {"False"}


def test_readable_chips_are_open_parts_and_excluded(tmp_path, monkeypatch):
    from floodvalue.data import split
    doc = {"train": ["A_1"], "dev": ["B_2"], "validation": ["C_3"], "test": ["D_4"],
           "excluded": {"E_5": "all -1"}}
    doc["sha256"] = split.canonical_hash(doc)
    (tmp_path / "s.json").write_text(json.dumps(doc), encoding="utf-8")
    assert fetch.label_readable_chips(tmp_path / "s.json") == {"A_1", "B_2", "C_3", "E_5"}


# --- P23 (#28): label shares, VV > 0 dB, sealed S1, manifest with event dates --------------

def test_open_chip_reports_label_shares_and_positive_db(raw, offline):
    assert fetch.run(None) == 0
    r = read_qc(raw.parent)["Sri-Lanka_713926"]
    # label_array(): rows 0..3 water except one -1 pixel, rows 4..7 dry
    assert float(r["label_invalid_frac"]) == pytest.approx(1 / 64)
    assert float(r["label_water_frac"]) == pytest.approx(31 / 64)
    assert float(r["label_dry_frac"]) == pytest.approx(32 / 64)
    s1 = s1_array()
    # share of ALL pixels (NaN counts as "not > 0") so chip shares average over a part
    assert float(r["vv_pos_frac"]) == pytest.approx(float((s1[0] > 0).sum()) / 64)
    assert float(r["vh_pos_frac"]) == pytest.approx(float((s1[1] > 0).sum()) / 64)
    assert float(r["vv_pos_frac"]) > 0  # fixture VV is uniform(-25, 5): some pixels > 0 dB
    assert r["s1_checked"] == "True"


def test_zero_vv_is_not_positive(raw):
    s1 = s1_array()
    s1[0] = -1.0
    s1[0, 0, :2] = [0.0, 0.5]
    r = _qc_one(raw, s1=s1)
    assert float(r["vv_pos_frac"]) == pytest.approx(1 / 64)


def test_sealed_test_chip_s1_pixels_are_not_read(raw, offline):
    # Out-of-range S1 in a sealed chip is not seen: only header + md5 are checked.
    write_chip(raw, "Mekong_922373", s1=_with(s1_array(), (0, 3, 3), 80.0))
    write_bucket_manifest(raw)
    assert fetch.run(None) == 0
    r = read_qc(raw.parent)["Mekong_922373"]
    assert r["s1_checked"] == "False" and r["qc_pass"] == "True"
    for k in ("vv_nan_frac", "vv_max", "vv_pos_frac", "db_scale_ok", "usable",
              "label_water_frac"):
        assert r[k] == "", k
    assert r["s1_sha256"] and r["s1_band_names"] == "VV/VH" and r["grid_match"] == "True"


def test_sealed_test_chip_s1_header_still_checked(raw, offline):
    from .conftest import write_tif
    write_tif(raw / "HandLabeled/S1Hand/Mekong_922373_S1Hand.tif", s1_array(), "float32",
              nodata=-9999.0, descriptions=("VV", "VH"))
    write_bucket_manifest(raw)
    assert fetch.run(None) == 1
    assert "nodata" in read_qc(raw.parent)["Mekong_922373"]["issues"]


def test_manifest_entries_pass_contract_and_carry_event_dates(raw, offline):
    from floodvalue.contracts.schemas import SourceManifest

    assert fetch.run(None) == 0
    doc = json.loads((raw.parent / "interim/source_manifest_s1.json").read_text())
    SourceManifest.model_validate(doc)  # the contract validator, entries as export copies them
    for e in doc["sources"]:
        SourceManifest.model_validate({"sources": [e]})
        assert e["after_event_note"]  # published in 2020, after every event
        assert e["event_date_source"].startswith(fetch.MEDIA)
    by_chip = {e["chip_id"]: e for e in doc["sources"]}
    m = by_chip["Mekong_922373"]
    assert (m["event_date"], m["event_metadata_location"]) == ("2018-08-05", "Cambodia")
    assert m["event_date"] == m["observation_date"]
    events = {e["event_id"]: e for e in doc["events"]}
    assert events["Mekong"]["s1_date"] == "2018-08-05"
    assert events["Mekong"]["metadata_location"] == "Cambodia"
    assert events["Mekong"]["n_chips"] == 1 and events["Sri-Lanka"]["n_chips"] == 1
    assert "Q-01" in doc["license_note"]


def test_event_table_from_metadata():
    meta = {"features": [{"properties": {"location": "Cambodia", "ISO_CC": "KHM",
                                         "s1_date": "2018/08/05", "s2_date": "2018/08/04"}}]}
    ev = fetch.event_table(meta, {"Mekong": 30})
    assert ev == [{"event_id": "Mekong", "metadata_location": "Cambodia", "iso_cc": "KHM",
                   "s1_date": "2018-08-05", "s2_date": "2018-08-04", "n_chips": 30}]


def test_run_fails_when_manifest_breaks_contract(raw, offline):
    # No publication date for the S1 object -> SourceEntry has neither published_at nor
    # available_date -> the stage must fail rather than write an invalid manifest.
    manifest = json.loads((raw / "bucket_manifest.json").read_text())
    for m in manifest.values():
        m.pop("updated")
    (raw / "bucket_manifest.json").write_text(json.dumps(manifest))
    assert fetch.run(None) == 1


def test_no_pixel_read_call_touches_a_test_chip(raw, offline, monkeypatch):
    # Spy on every raster pixel read during `make data`: test chips (Mekong here) are opened
    # for headers only; train/validation chips are read.
    import rasterio.io

    seen = []

    def spy_on(method):
        orig = getattr(rasterio.io.DatasetReader, method)

        def spy(self, *a, **kw):
            seen.append(self.name)
            return orig(self, *a, **kw)

        monkeypatch.setattr(rasterio.io.DatasetReader, method, spy)

    for method in ("read", "read_masks", "dataset_mask", "sample"):
        spy_on(method)
    assert fetch.run(None) == 0
    assert any("Sri-Lanka_713926" in n for n in seen)
    assert not [n for n in seen if "Mekong_922373" in n]


def test_nan_under_label_is_counted(raw):
    # Dataset trap: under NaN S1 the label is often 0/1, not -1 (docs/data.md §8).
    s1, lab = s1_array(), label_array()     # S1 NaN at (0,0); label there is 1
    s1[1, 5, 5] = np.nan                    # VH-only NaN over a dry pixel
    s1[:, 0, 1] = np.nan                    # NaN over the -1 pixel: not counted
    r = _qc_one(raw, s1=s1, label=lab)
    assert float(r["s1_nan_label_water_frac"]) == pytest.approx(1 / 64)
    assert float(r["s1_nan_label_dry_frac"]) == pytest.approx(1 / 64)
