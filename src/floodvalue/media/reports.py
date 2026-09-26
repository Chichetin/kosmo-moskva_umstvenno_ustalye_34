"""Eyewitness reports: server-side validation, photo sanitising, status tokens, retention.
No registration, no name or phone. The place of the event is what the person marks, not
their current position. Photos are re-encoded (EXIF and other metadata dropped) under random
names and are only ever served to an authorised analyst."""

from __future__ import annotations

import hashlib
import hmac
import io
import secrets
import sqlite3
import time
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from floodvalue.media.events import recount
from floodvalue.media.geo import valid_lon_lat
from floodvalue.media.store import add_history, get_kv, now_iso, set_kv

CATEGORIES = {"flooding": "Подтопление или затопление", "water_rising": "Вода быстро поднимается",
              "threat": "Угроза подтопления", "other": "Другое"}
LOCATION_PRECISION = ("point", "street", "locality", "district", "unknown")
TIME_PRECISION = ("hour", "day", "several_days", "unknown")
MAX_PHOTOS = 3
MAX_PHOTO_BYTES = 8 * 1024 * 1024
MAX_PHOTO_SIDE = 8000
MIN_PHOTO_SIDE = 64
STORED_SIDE = 2048
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
MAX_FUTURE = timedelta(minutes=10)
MAX_PAST = timedelta(days=30)
PUBLIC_STATUS = {"pending": "На проверке у аналитика", "accepted": "Принято в работу",
                 "rejected": "Не принято в работу", "spam": "Не принято в работу"}


class ReportError(ValueError):
    """A validation problem the form shows next to the field."""

    def __init__(self, field: str, message: str):
        super().__init__(message)
        self.field = field


class ReportIn(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    category: str
    description: str = Field(min_length=10, max_length=2000)
    lat: float | None = Field(default=None, ge=-90, le=90, allow_inf_nan=False)
    lon: float | None = Field(default=None, ge=-180, le=180, allow_inf_nan=False)
    place_text: str | None = Field(default=None, max_length=200)
    location_precision: str = "unknown"
    observed_at: datetime | None = None
    observed_precision: str = "unknown"
    observed_tz: str | None = Field(default=None, max_length=64)
    idempotency_key: str = Field(min_length=16, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")

    @field_validator("category")
    @classmethod
    def _category(cls, v: str) -> str:
        if v not in CATEGORIES:
            raise ValueError("неизвестный тип сообщения")
        return v

    @field_validator("location_precision")
    @classmethod
    def _loc_precision(cls, v: str) -> str:
        if v not in LOCATION_PRECISION:
            raise ValueError("неизвестная точность места")
        return v

    @field_validator("observed_precision")
    @classmethod
    def _time_precision(cls, v: str) -> str:
        if v not in TIME_PRECISION:
            raise ValueError("неизвестная точность времени")
        return v

    @model_validator(mode="after")
    def _consistency(self):
        if (self.lat is None) != (self.lon is None):
            raise ValueError("координаты: нужны и широта, и долгота")
        if self.lat is not None and not valid_lon_lat(self.lon, self.lat):
            raise ValueError("координаты вне допустимых границ")
        if self.lat is None and not self.place_text:
            raise ValueError("укажите место: точку на карте или название")
        if self.lat is None:
            self.location_precision = ("locality" if self.location_precision == "point"
                                       else self.location_precision)
        if self.observed_at is not None:
            if self.observed_at.tzinfo is None:
                raise ValueError("время наблюдения без часового пояса")
            now = datetime.now(UTC)
            if self.observed_at > now + MAX_FUTURE:
                raise ValueError("время наблюдения в будущем")
            if self.observed_at < now - MAX_PAST:
                raise ValueError("время наблюдения больше 30 дней назад")
        return self


def server_secret(conn: sqlite3.Connection) -> bytes:
    """Per-installation secret for status tokens; created once, kept in the database."""
    value = get_kv(conn, "status_secret")
    if value is None:
        value = secrets.token_hex(32)
        set_kv(conn, "status_secret", value)
    return bytes.fromhex(value)


def status_token(secret: bytes, report_id: str) -> str:
    return hmac.new(secret, report_id.encode(), hashlib.sha256).hexdigest()[:40]


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def sanitize_photo(raw: bytes) -> tuple[bytes, int, int]:
    """Decode by content (not by extension or declared type), bound size and dimensions,
    apply orientation, drop all metadata by re-encoding to plain JPEG."""
    if len(raw) > MAX_PHOTO_BYTES:
        raise ReportError("photos", "фото больше 8 МБ")
    try:
        with Image.open(io.BytesIO(raw)) as probe:
            fmt, (w, h) = probe.format, probe.size
            if fmt not in ALLOWED_FORMATS:
                raise ReportError("photos", "допустимы только JPEG, PNG, WebP")
            if max(w, h) > MAX_PHOTO_SIDE or min(w, h) < MIN_PHOTO_SIDE:
                raise ReportError("photos", "недопустимые размеры изображения")
            probe.verify()
        with Image.open(io.BytesIO(raw)) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
            img.thumbnail((STORED_SIDE, STORED_SIDE))
            out = io.BytesIO()
            img.save(out, "JPEG", quality=85, optimize=True)
            return out.getvalue(), img.width, img.height
    except ReportError:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError,
            Image.DecompressionBombError) as exc:
        raise ReportError("photos", "файл не является допустимым изображением") from exc


def new_report_id() -> str:
    return "R-" + datetime.now(UTC).strftime("%y%m%d") + "-" + secrets.token_hex(3).upper()


def create_report(conn: sqlite3.Connection, data: ReportIn, photos: list[bytes],
                  photo_dir: Path, retention_days: int, is_demo: bool = False) -> tuple[str, str, bool]:
    """Returns (report_id, status_token, created). The same idempotency key returns the
    first report instead of a duplicate."""
    secret = server_secret(conn)
    row = conn.execute("SELECT report_id FROM citizen_reports WHERE idempotency_key=?",
                       (data.idempotency_key,)).fetchone()
    if row:
        return row["report_id"], status_token(secret, row["report_id"]), False
    if len(photos) > MAX_PHOTOS:
        raise ReportError("photos", f"не больше {MAX_PHOTOS} фото")
    cleaned = [sanitize_photo(p) for p in photos]
    report_id = new_report_id()
    token = status_token(secret, report_id)
    now = now_iso()
    delete_after = (datetime.now(UTC) + timedelta(days=retention_days)).strftime(
        "%Y-%m-%dT%H:%M:%SZ")
    observed = (data.observed_at.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
                if data.observed_at else None)
    written: list[Path] = []
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            """INSERT INTO citizen_reports(report_id, status_token_hash, idempotency_key, category,
                 description, lat, lon, place_text, location_precision, observed_at, observed_tz,
                 observed_precision, submitted_at, is_demo, delete_after)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (report_id, token_hash(token), data.idempotency_key, data.category, data.description,
             data.lat, data.lon, data.place_text, data.location_precision, observed,
             data.observed_tz, data.observed_precision if observed else "unknown", now,
             int(is_demo), delete_after))
        photo_dir.mkdir(parents=True, exist_ok=True)
        for jpeg, w, h in cleaned:
            name = uuid.uuid4().hex + ".jpg"
            (photo_dir / name).write_bytes(jpeg)
            written.append(photo_dir / name)
            conn.execute("INSERT INTO attachments VALUES(?,?,?,?,?,?,?,?)",
                         (uuid.uuid4().hex, report_id, name, "image/jpeg", w, h, len(jpeg), now))
        conn.execute("COMMIT")
    except BaseException as exc:
        conn.execute("ROLLBACK")
        for path in written:
            path.unlink(missing_ok=True)
        if isinstance(exc, sqlite3.IntegrityError):  # the same key raced in: return the first
            row = conn.execute("SELECT report_id FROM citizen_reports WHERE idempotency_key=?",
                               (data.idempotency_key,)).fetchone()
            if row:
                return row["report_id"], status_token(secret, row["report_id"]), False
        raise
    return report_id, token, True


def public_status(conn: sqlite3.Connection, token: str) -> dict | None:
    """What the sender may see: number, time, a coarse status. No internals, no other people."""
    row = conn.execute("SELECT report_id, submitted_at, moderation_status FROM citizen_reports "
                       "WHERE status_token_hash=?", (token_hash(token),)).fetchone()
    if row is None:
        return None
    return {"report_id": row["report_id"], "submitted_at": row["submitted_at"],
            "status": PUBLIC_STATUS[row["moderation_status"]]}


def rate_limited(conn: sqlite3.Connection, bucket: str, limit: int, window_s: int) -> bool:
    now = time.time()
    conn.execute("DELETE FROM rate_hits WHERE at < ?", (now - 86400,))
    count = conn.execute("SELECT COUNT(*) FROM rate_hits WHERE bucket=? AND at>=?",
                         (bucket, now - window_s)).fetchone()[0]
    if count >= limit:
        return True
    conn.execute("INSERT INTO rate_hits VALUES(?,?)", (bucket, now))
    return False


def purge_expired(conn: sqlite3.Connection, photo_dir: Path, now: str | None = None) -> int:
    """Delete reports past retention with their photos; linked events keep a history note."""
    now = now or now_iso()
    rows = conn.execute("SELECT report_id, event_id FROM citizen_reports WHERE delete_after < ?",
                        (now,)).fetchall()
    for r in rows:
        for a in conn.execute("SELECT file_name FROM attachments WHERE report_id=?",
                              (r["report_id"],)).fetchall():
            (photo_dir / a["file_name"]).unlink(missing_ok=True)
        affected = [e["event_id"] for e in conn.execute(
            "SELECT event_id FROM evidence WHERE report_id=?", (r["report_id"],))]
        conn.execute("DELETE FROM evidence WHERE report_id=?", (r["report_id"],))
        conn.execute("DELETE FROM attachments WHERE report_id=?", (r["report_id"],))
        conn.execute("DELETE FROM citizen_reports WHERE report_id=?", (r["report_id"],))
        for event_id in affected:
            recount(conn, event_id)
            add_history(conn, event_id, "system", "report_purged", None,
                        {"report_id": r["report_id"], "reason": "срок хранения истёк"})
    return len(rows)
