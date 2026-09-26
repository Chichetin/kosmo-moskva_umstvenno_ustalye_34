"""Citizen report input: server-side validation and photo sanitising (prompt §7–8)."""

import io
from datetime import UTC, datetime, timedelta

import pytest
from PIL import Image
from pydantic import ValidationError

from floodvalue.media import reports


def _payload(**overrides):
    base = {
        "client_key": "0f8c2a4e-7d1b-4c55-9a3e-1b2c3d4e5f60",
        "kind": "water_on_road",
        "description": "Вода перелилась через дорогу у моста, машины объезжают.",
        "lat": 55.75, "lon": 37.61, "location_precision": "point",
        "observed_at": (datetime.now(UTC) - timedelta(hours=1)).isoformat(),
        "observed_time_precision": "approx_hour",
        "observed_tz": "Europe/Moscow",
    }
    base.update(overrides)
    return base


def test_valid_report_accepted():
    report = reports.ReportIn.model_validate(_payload())
    assert report.lat == 55.75 and report.location_precision == "point"


@pytest.mark.parametrize("overrides", [
    {"lat": 91},
    {"lon": -181},
    {"lat": float("nan")},
    {"lat": 0.0, "lon": 0.0},
    {"lat": 55.0, "lon": None},
    {"lat": None, "lon": None, "place_text": None},
    {"observed_at": (datetime.now(UTC) + timedelta(hours=2)).isoformat()},
    {"observed_at": (datetime.now(UTC) - timedelta(days=45)).isoformat()},
    {"observed_at": "2026-09-26T10:00:00"},
    {"observed_tz": "Mars/Olympus"},
    {"description": "мало"},
    {"kind": "tsunami"},
    {"client_key": "1"},
    {"unexpected": "field"},
])
def test_invalid_report_rejected(overrides):
    with pytest.raises(ValidationError):
        reports.ReportIn.model_validate(_payload(**overrides))


def test_place_without_coordinates_is_kept_with_unknown_precision():
    report = reports.ReportIn.model_validate(
        _payload(lat=None, lon=None, place_text="Орск, ул. Набережная",
                 location_precision="point"))
    assert report.lat is None
    assert report.location_precision == "unknown"


def _image(fmt: str, size=(64, 48), exif: bool = False) -> bytes:
    buffer = io.BytesIO()
    image = Image.new("RGB", size, (30, 90, 160))
    kwargs = {}
    if exif:
        info = Image.Exif()
        info[0x010F] = "SecretCameraMaker"  # Make
        info[0x8825] = {2: (55.0, 45.0, 0.0)}  # GPSInfo
        kwargs["exif"] = info.tobytes()
    image.save(buffer, fmt, **kwargs)
    return buffer.getvalue()


def test_photo_is_reencoded_without_exif():
    original = _image("JPEG", exif=True)
    assert b"SecretCameraMaker" in original
    clean, width, height = reports.sanitize_photo(original)
    assert b"SecretCameraMaker" not in clean
    with Image.open(io.BytesIO(clean)) as image:
        assert image.format == "JPEG"
        assert not image.getexif()
    assert (width, height) == (64, 48)


def test_png_and_webp_accepted_large_downscaled():
    for fmt in ("PNG", "WEBP"):
        clean, width, height = reports.sanitize_photo(_image(fmt, size=(4000, 1000)))
        assert max(width, height) == reports.MAX_SIDE


@pytest.mark.parametrize("data", [
    b"<html><script>alert(1)</script></html>",
    b"GIF89a" + b"\x00" * 64,
    b"",
])
def test_non_images_rejected_by_content_not_extension(data):
    with pytest.raises(ValueError):
        reports.sanitize_photo(data)


def test_gif_rejected_even_if_valid_image():
    with pytest.raises(ValueError):
        reports.sanitize_photo(_image("GIF"))


def test_oversized_photo_rejected():
    with pytest.raises(ValueError):
        reports.sanitize_photo(b"\xff\xd8" + b"\x00" * (reports.MAX_PHOTO_BYTES + 1))


def test_public_id_and_token_are_unpredictable():
    ids = {reports.new_public_id() for _ in range(200)}
    assert len(ids) == 200
    token_a = reports.status_token("secret", "key-a")
    assert token_a == reports.status_token("secret", "key-a")
    assert token_a != reports.status_token("secret", "key-b")
    assert token_a != reports.status_token("other", "key-a")
    assert len(token_a) >= 32
