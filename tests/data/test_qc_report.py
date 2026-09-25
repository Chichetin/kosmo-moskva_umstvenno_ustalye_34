"""docs/verification/data-qc.md is built from interim/qc.csv by a pure summary: test chips
contribute only format/hash counts, never pixel statistics."""

import pytest

from floodvalue.data import qc_report


def row(chip, *, m1=0.0, dry=0.9, water=0.1, nan=0.0, vv_pos=0.0, usable="True",
        qc_pass="True", checked="True", md5="True"):
    event = chip.rsplit("_", 1)[0]
    r = {"chip_id": chip, "event_id": event, "qc_pass": qc_pass, "s1_md5_ok": md5,
         "label_md5_ok": md5, "grid_match": "True", "db_scale_ok": "True",
         "s1_checked": checked, "label_checked": checked, "usable": usable, "issues": ""}
    if checked == "True":
        r.update(label_invalid_frac=str(m1), label_dry_frac=str(dry),
                 label_water_frac=str(water), vv_nan_frac=str(nan), vh_nan_frac=str(nan),
                 vv_pos_frac=str(vv_pos), vv_max="3.0" if vv_pos else "-1.0",
                 valid_frac=str(1 - max(m1, nan)))
    else:
        r.update({k: "" for k in ("label_invalid_frac", "label_dry_frac", "label_water_frac",
                                  "vv_nan_frac", "vh_nan_frac", "vv_pos_frac", "vv_max",
                                  "valid_frac", "db_scale_ok")}, usable="")
    return r


SPLIT = {"train": ["A_1", "A_2"], "dev": [], "validation": ["B_1"], "test": ["T_1"],
         "excluded": {"A_3": "all -1"}}


def rows():
    return [row("A_1", m1=0.2, dry=0.6, water=0.2, nan=0.1, vv_pos=0.01),
            row("A_2", m1=0.0, dry=1.0, water=0.0),
            row("A_3", m1=1.0, dry=0.0, water=0.0, usable="False"),
            row("B_1", m1=0.0, dry=0.6, water=0.4, nan=0.6, vv_pos=0.0),
            row("T_1", checked="False")]


def test_part_of_chip_follows_split():
    parts = qc_report.chip_parts(SPLIT)
    assert parts == {"A_1": "train", "A_2": "train", "B_1": "validation", "T_1": "test",
                     "A_3": "excluded"}


def test_summary_by_part_averages_chip_shares():
    s = {g["group"]: g for g in qc_report.summarize(rows(), SPLIT)["parts"]}
    tr = s["train"]
    assert tr["n_chips"] == 2
    assert tr["invalid"] == pytest.approx(0.1)          # (0.2 + 0.0) / 2
    assert tr["dry"] == pytest.approx(0.8)
    assert tr["water"] == pytest.approx(0.1)
    assert tr["water_of_labelled"] == pytest.approx(0.2 / 1.8)  # sum water / sum (dry+water)
    assert tr["nan"] == pytest.approx(0.05)
    assert tr["vv_pos"] == pytest.approx(0.005)
    assert tr["chips_with_nan"] == 1 and tr["chips_nan_gt_half"] == 0
    assert tr["chips_vv_pos"] == 1 and tr["chips_no_water"] == 1
    assert tr["chips_invalid_gt_half"] == 0 and tr["unusable"] == 0
    va = s["validation"]
    assert va["chips_nan_gt_half"] == 1 and va["chips_vv_pos"] == 0
    ex = s["excluded"]
    assert ex["n_chips"] == 1 and ex["chips_invalid_gt_half"] == 1 and ex["unusable"] == 1


def test_threshold_half_is_strict():
    r = [row("A_1", m1=0.5, dry=0.5, water=0.0, nan=0.5)]
    g = qc_report.summarize(r, {"train": ["A_1"], "test": []})["parts"][0]
    assert g["chips_invalid_gt_half"] == 0 and g["chips_nan_gt_half"] == 0


def test_test_part_has_counts_but_no_pixel_statistics():
    s = qc_report.summarize(rows(), SPLIT)
    assert "test" not in {g["group"] for g in s["parts"]}
    t = s["test"]
    assert t == {"n_chips": 1, "qc_pass": 1, "md5_ok": 1, "grid_match": 1,
                 "pixels_read": 0, "events": ["T"]}


def test_pixels_of_test_chip_in_csv_are_an_error():
    bad = rows()
    bad[-1] = row("T_1")  # a test chip with pixel statistics: sealing was broken upstream
    with pytest.raises(ValueError, match="T_1"):
        qc_report.summarize(bad, SPLIT)


def test_integrity_counts_all_chips():
    r = rows()
    r[1]["s1_md5_ok"] = "False"
    r[1]["qc_pass"] = "False"
    s = qc_report.summarize(r, SPLIT)
    assert s["total"] == {"n_chips": 5, "qc_pass": 4, "md5_ok": 4, "grid_match": 5}


def test_events_are_grouped_inside_open_parts():
    ev = {g["group"]: g for g in qc_report.summarize(rows(), SPLIT)["events"]}
    assert set(ev) == {"A", "B"}  # T (test) absent; A_3 (excluded) is not in event rows
    assert ev["A"]["n_chips"] == 2 and ev["A"]["part"] == "train"


def test_unknown_chip_is_an_error():
    with pytest.raises(ValueError, match="Z_9"):
        qc_report.summarize(rows() + [row("Z_9")], SPLIT)


def test_markdown_mentions_sealing_and_numbers():
    s = qc_report.summarize(rows(), SPLIT)
    md = qc_report.render(s, events=[{"event_id": "A", "metadata_location": "A",
                                      "s1_date": "2018-01-01", "s2_date": "2018-01-02",
                                      "n_chips": 3}],
                          header={"commit": "abc1234", "command": "make data-qc"})
    assert "abc1234" in md and "make data-qc" in md
    assert "запечатан" in md and "| train | 2 |" in md
    assert "2018-01-01" in md and "0,1000" not in md  # decimal point, not comma
