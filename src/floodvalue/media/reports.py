"""Citizen report validation, photo sanitising and unguessable identifiers."""

from __future__ import annotations

import hashlib
import hmac
import io
import math
import secrets
from datetime import UTC, datetime, timedelta
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

KINDS = {
    "water_on_road": "Вода на дороге",
    "flooded_building": "Подтоплен дом или двор",
    "rising_water": "Быстро поднимается вода",
    "infrastructure": "Пострадала инфраструктура",
    "other": "Другое",
}
MAX_PHOTOS = 3
MAX_PHOTO_BYTES = 8 * 1024 * 1024
MAX_PIXELS = 40_000_000
MAX_SIDE = 2048
ACCEPTED_FORMATS = {"JPEG", "PNG", "WEBP"}
MAX_AGE_DAYS = 30
RETENTION_DAYS = 90
# Crockford-like alphabet without 0/O/1/I/L so a number can be read over the phone.
_ID_ALPHABET = "23456789ABCDEFGHJKMNPQRSTUVWXYZ"


class ReportIn(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    client_key: str = Field(pattern=r"^[A-Za-z0-9-]{16,64}$")
    kind: Literal["water_on_road", "flooded_building", "rising_water", "infrastructure", "other"]
    description: str = Field(min_length=10, max_length=2000)
    lat: float | None = Field(default=None, ge=-90, le=90)
    lon: float | None = Field(default=None, ge=-180, le=180)
    place_text: str | None = Field(default=None, max_length=200)
    location_precision: Literal["point", "street", "locality", "district", "region",
                                "unknown"] = "unknown"
    observed_at: datetime
    observed_time_precision: Literal["exact", "approx_hour", "approx_day"]
    observed_tz: str = Field(max_length=64)

    @field_validator("lat", "lon")
    @classmethod
    def _finite(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("координата должна быть числом")
        return value

    @field_validator("observed_tz")
    @classmethod
    def _known_zone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError("неизвестный часовой пояс") from exc
        return value

    @field_validator("observed_at")
    @classmethod
    def _plausible_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("время наблюдения должно быть с часовым поясом")
        now = datetime.now(UTC)
        if value > now + timedelta(minutes=10):
            raise ValueError("время наблюдения в будущем")
        if value < now - timedelta(days=MAX_AGE_DAYS):
            raise ValueError(f"наблюдение старше {MAX_AGE_DAYS} дней")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def _place(self) -> ReportIn:
        if (self.lat is None) != (self.lon is None):
            raise ValueError("нужны обе координаты или ни одной")
        if self.lat == 0 and self.lon == 0:
            raise ValueError("координаты 0, 0 похожи на пустое значение")
        if self.lat is None:
            if not self.place_text:
                raise ValueError("укажите место на карте или словами")
            # A place name without coordinates waits for the analyst to locate it.
            self.location_precision = "unknown"
        return self


def sanitize_photo(data: bytes) -> tuple[bytes, int, int]:
    """Re-encode an uploaded photo as a fresh JPEG: checks the real format, drops EXIF/GPS."""
    if not data:
        raise ValueError("пустой файл")
    if len(data) > MAX_PHOTO_BYTES:
        raise ValueError("файл больше 8 МБ")
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.format not in ACCEPTED_FORMATS:
                raise ValueError("допустимы только JPEG, PNG и WebP")
            if image.width * image.height > MAX_PIXELS:
                raise ValueError("слишком большое изображение")
            image.load()
            upright = ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError) as exc:
        raise ValueError("файл не является изображением JPEG, PNG или WebP") from exc
    upright.thumbnail((MAX_SIDE, MAX_SIDE))
    clean = io.BytesIO()
    # A new JPEG carries no EXIF, GPS or XMP unless passed explicitly.
    upright.save(clean, "JPEG", quality=85, optimize=True)
    return clean.getvalue(), upright.width, upright.height


def new_public_id() -> str:
    body = "".join(secrets.choice(_ID_ALPHABET) for _ in range(8))
    return f"FV-{body[:4]}-{body[4:]}"


def status_token(secret: str, client_key: str) -> str:
    """Deterministic per submission, so a retried submission gets the same status link."""
    return hmac.new(secret.encode(), client_key.encode(), hashlib.sha256).hexdigest()[:40]


def retention_until(submitted_at: datetime) -> datetime:
    return submitted_at + timedelta(days=RETENTION_DAYS)
