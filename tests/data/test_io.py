import numpy as np
import pytest

from floodvalue.data import io

from .conftest import H, W, label_array, s1_array


def test_parse_chip_id_keeps_hyphenated_event():
    assert io.parse_chip_id("Sri-Lanka_713926") == ("Sri-Lanka", "713926")
    assert io.chip_id_from_filename("Sri-Lanka_713926_S1Hand.tif") == "Sri-Lanka_713926"
    assert io.chip_id_from_filename("x/USA_1_LabelHand.tif") == "USA_1"


@pytest.mark.parametrize("bad", ["Spain", "Spain_", "_123", "Spain_12a"])
def test_parse_chip_id_rejects_malformed(bad):
    with pytest.raises(ValueError):
        io.parse_chip_id(bad)


def test_layer_path_matches_bucket_layout(raw):
    p = io.layer_path("Sri-Lanka_713926", "S1Hand")
    assert p == raw / "HandLabeled/S1Hand/Sri-Lanka_713926_S1Hand.tif"


def test_load_s1_returns_db_values_unchanged_with_nan(raw):
    arr, profile = io.load_s1("Sri-Lanka_713926")
    assert arr.dtype == np.float32 and arr.shape == (2, H, W)
    expected = s1_array()
    np.testing.assert_array_equal(np.isnan(arr), np.isnan(expected))
    np.testing.assert_allclose(arr[np.isfinite(arr)], expected[np.isfinite(expected)])
    assert profile["crs"].to_string() == "EPSG:4326"


def test_load_label_refuses_test_event_without_flag(raw):
    from floodvalue.data.split import SealedTestError
    with pytest.raises(SealedTestError):
        io.load_label("Mekong_922373")
    lab, _ = io.load_label("Mekong_922373", allow_test=True)
    assert lab.dtype == np.int16


def test_load_label_int16(raw):
    lab, _ = io.load_label("Sri-Lanka_713926")
    assert lab.dtype == np.int16
    np.testing.assert_array_equal(lab, label_array())


def test_valid_mask_needs_finite_s1_and_label():
    s1 = np.zeros((2, 2, 2), np.float32)
    s1[1, 0, 0] = np.nan  # NaN in VH only; label under it is 1, not -1
    lab = np.array([[1, -1], [0, 1]], np.int16)
    np.testing.assert_array_equal(io.valid_mask(s1, lab), [[False, False], [True, True]])
    np.testing.assert_array_equal(io.valid_mask(s1), [[False, True], [True, True]])
