"""HTTP API: a public citizen-report intake and operator endpoints behind a session.

Public endpoints (/api/public/…) never return other people's reports, photos, the
moderation queue or internal calculations. Operator endpoints (/api/ops/…) need a session
cookie obtained with the operator token; state-changing calls also need a custom header,
which a cross-site form cannot send.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.datastructures import UploadFile

from floodvalue.media import events as registry
from floodvalue.media import geo, reports
from floodvalue.media.net import FetchError
from floodvalue.media.store import (
    attachments_dir,
    connect,
    get_setting,
    iso,
    now,
    parse_iso,
    record_history,
    secret_file,
    utc_now,
)

router = APIRouter()
COOKIE = "fv_ops"
SESSION_HOURS = 12
ACTION_HEADER = "X-FloodValue-Action"
MAX_BODY = reports.MAX_PHOTOS * reports.MAX_PHOTO_BYTES + 64 * 1024
PUBLIC_STATUS = {
    "pending": "Получено. Ждёт проверки аналитиком.",
    "approved": "Принято в работу аналитиком. Это не подтверждение затопления.",
    "rejected": "Проверено и не принято в работу.",
}


class RateLimit:
    """In-process sliding window; one API process is enough for the local MVP."""

    def __init__(self, limit: int, window_s: float):
        self.limit, self.window = limit, window_s
        self.hits: dict[str, deque[float]] = defaultdict(deque)
        self.lock = threading.Lock()

    def allow(self, key: str) -> bool:
        with self.lock:
            stamps, moment = self.hits[key], time.monotonic()
            while stamps and stamps[0] <= moment - self.window:
                stamps.popleft()
            if len(stamps) >= self.limit:
                return False
            stamps.append(moment)
            return True


submit_limit = RateLimit(5, 600)
status_limit = RateLimit(30, 60)
login_limit = RateLimit(10, 600)
geocode_limit = RateLimit(10, 60)


def _client(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def operator_token() -> str:
    return os.environ.get("FLOODVALUE_OPERATOR_TOKEN") or secret_file("operator_token")


def require_operator(request: Request) -> str:
    token = request.cookies.get(COOKIE)
    if not token:
        raise HTTPException(401, "Нужен вход аналитика")
    with connect() as db:
        row = db.execute("SELECT name, expires_at FROM operator_session WHERE token_hash = ?",
                         (_hash(token),)).fetchone()
    if row is None or row["expires_at"] <= now():
        raise HTTPException(401, "Сессия аналитика истекла")
    if request.method not in ("GET", "HEAD") and request.headers.get(ACTION_HEADER) != "1":
        raise HTTPException(403, "Действие без заголовка интерфейса отклонено")
    return row["name"]


# ---------------------------------------------------------------- public intake


FIELD_LABELS = {"kind": "Что произошло", "description": "Описание", "lat": "Широта",
                "lon": "Долгота", "place_text": "Место словами", "observed_at": "Время",
                "observed_tz": "Часовой пояс", "observed_time_precision": "Точность времени",
                "client_key": "Служебный ключ", "location_precision": "Точность места"}
ERROR_TEXT = {"missing": "обязательное поле", "literal_error": "недопустимое значение",
              "string_too_short": "слишком короткий текст",
              "string_too_long": "слишком длинный текст",
              "greater_than_equal": "вне допустимого диапазона",
              "less_than_equal": "вне допустимого диапазона",
              "extra_forbidden": "лишнее поле", "float_parsing": "нужно число",
              "datetime_from_date_parsing": "неверный формат времени",
              "datetime_parsing": "неверный формат времени",
              "string_pattern_mismatch": "неверный формат", "json_invalid": "неверный JSON"}


def _validation_message(error: ValidationError) -> str:
    parts = []
    for item in error.errors():
        field = FIELD_LABELS.get(str(item["loc"][0]) if item["loc"] else "", "Форма")
        text = (item["msg"].removeprefix("Value error, ") if item["type"] == "value_error"
                else ERROR_TEXT.get(item["type"], item["msg"]))
        parts.append(f"{field}: {text}")
    return "; ".join(parts)


def _receipt(row: sqlite3.Row, token: str, duplicate: bool) -> dict:
    return {"public_id": row["public_id"], "status_token": token,
            "submitted_at": row["submitted_at"], "retention_until": row["retention_until"],
            "duplicate": duplicate}


@router.post("/api/public/reports", status_code=201)
async def submit_report(request: Request, response: Response):
    length = request.headers.get("content-length", "")
    if not length.isdigit():
        raise HTTPException(411, "Нужен заголовок Content-Length")
    if int(length) > MAX_BODY:
        raise HTTPException(413, "Слишком большой запрос: до 3 фото по 8 МБ")
    form = await request.form(max_files=reports.MAX_PHOTOS, max_fields=4,
                              max_part_size=16 * 1024)
    raw = form.get("report")
    if not isinstance(raw, str):
        raise HTTPException(422, "Нет полей обращения")
    try:
        data = reports.ReportIn.model_validate_json(raw)
    except ValidationError as exc:
        raise HTTPException(422, _validation_message(exc)) from exc
    token = reports.status_token(secret_file("report_secret"), data.client_key)
    with connect() as db:
        existing = db.execute("SELECT * FROM citizen_report WHERE client_key = ?",
                              (data.client_key,)).fetchone()
    if existing:
        response.status_code = 200
        return _receipt(existing, token, duplicate=True)
    if not submit_limit.allow(_client(request)):
        raise HTTPException(429, "Слишком много обращений подряд. Попробуйте через 10 минут.")
    uploads = [item for item in form.getlist("photos") if isinstance(item, UploadFile)]
    cleaned = []
    for index, upload in enumerate(uploads, 1):
        body = await upload.read(reports.MAX_PHOTO_BYTES + 1)
        try:
            cleaned.append(await run_in_threadpool(reports.sanitize_photo, body))
        except ValueError as exc:
            raise HTTPException(422, f"Фото {index}: {exc}") from exc
    submitted = utc_now()
    written = []
    try:
        for photo, width, height in cleaned:
            name = f"{secrets.token_hex(16)}.jpg"
            fd = os.open(attachments_dir() / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as target:
                target.write(photo)
            written.append((name, width, height, len(photo)))
        with connect() as db:
            cursor = db.execute(
                "INSERT INTO citizen_report(public_id, client_key, status_token_hash, kind, "
                "description, lat, lon, place_text, location_precision, observed_at, "
                "observed_time_precision, observed_tz, submitted_at, retention_until) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (reports.new_public_id(), data.client_key, _hash(token), data.kind,
                 data.description, data.lat, data.lon, data.place_text,
                 data.location_precision, iso(data.observed_at), data.observed_time_precision,
                 data.observed_tz, iso(submitted), iso(reports.retention_until(submitted))))
            for name, width, height, size in written:
                db.execute("INSERT INTO attachment(report_id, stored_name, mime, width, height, "
                           "bytes, created_at) VALUES (?, ?, 'image/jpeg', ?, ?, ?, ?)",
                           (cursor.lastrowid, name, width, height, size, iso(submitted)))
            row = db.execute("SELECT * FROM citizen_report WHERE report_id = ?",
                             (cursor.lastrowid,)).fetchone()
    except sqlite3.IntegrityError:
        for name, *_ in written:
            (attachments_dir() / name).unlink(missing_ok=True)
        with connect() as db:  # a concurrent retry with the same key won the race
            row = db.execute("SELECT * FROM citizen_report WHERE client_key = ?",
                             (data.client_key,)).fetchone()
        if row is None:
            raise
        response.status_code = 200
        return _receipt(row, token, duplicate=True)
    except BaseException:
        for name, *_ in written:
            (attachments_dir() / name).unlink(missing_ok=True)
        raise
    return _receipt(row, token, duplicate=False)


@router.get("/api/public/geocode")
def public_geocode(request: Request, q: Annotated[str, Query(min_length=2, max_length=120)]):
    """Explicit place search for the form (never search-as-you-type, Nominatim policy)."""
    if not geocode_limit.allow(_client(request)):
        raise HTTPException(429, "Слишком много поисков подряд. Подождите минуту.")
    try:
        with connect() as db:
            found = geo.geocode(db, q)
    except FetchError as exc:
        raise HTTPException(503, "Поиск места сейчас недоступен. Отметьте точку на карте "
                                 "или опишите место словами.") from exc
    return {"results": found, "attribution": geo.ATTRIBUTION}


class TokenIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    token: str = Field(pattern=r"^[0-9a-f]{40}$")


def _report_by_token(request: Request, token: str) -> sqlite3.Row:
    if not status_limit.allow(_client(request)):
        raise HTTPException(429, "Слишком много запросов")
    with connect() as db:
        row = db.execute("SELECT * FROM citizen_report WHERE status_token_hash = ?",
                         (_hash(token),)).fetchone()
    if row is None:
        raise HTTPException(404, "Обращение не найдено или удалено")
    return row


@router.post("/api/public/reports/status")
def report_status(body: TokenIn, request: Request):
    row = _report_by_token(request, body.token)
    with connect() as db:
        photos = db.execute("SELECT COUNT(*) FROM attachment WHERE report_id = ?",
                            (row["report_id"],)).fetchone()[0]
    return {"public_id": row["public_id"], "submitted_at": row["submitted_at"],
            "observed_at": row["observed_at"], "kind": reports.KINDS[row["kind"]],
            "status": row["moderation_status"],
            "status_text": PUBLIC_STATUS[row["moderation_status"]],
            "photos": photos, "retention_until": row["retention_until"]}


def delete_report(db: sqlite3.Connection, report_id: int, actor: str, note: str) -> None:
    names = [item["stored_name"] for item in db.execute(
        "SELECT stored_name FROM attachment WHERE report_id = ?", (report_id,))]
    for item in db.execute("SELECT event_id FROM evidence WHERE report_id = ?", (report_id,)):
        record_history(db, item["event_id"], actor, "report_deleted",
                       before={"report_id": report_id}, note=note)
    db.execute("DELETE FROM citizen_report WHERE report_id = ?", (report_id,))
    for name in names:
        (attachments_dir() / name).unlink(missing_ok=True)


@router.post("/api/public/reports/delete")
def delete_own_report(body: TokenIn, request: Request):
    row = _report_by_token(request, body.token)
    with connect() as db:
        delete_report(db, row["report_id"], "sender", "удалено отправителем")
    return {"deleted": row["public_id"]}


# ---------------------------------------------------------------- operator session


class LoginIn(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    token: str = Field(min_length=8, max_length=200)
    name: str = Field(default="аналитик", min_length=1, max_length=60)


@router.post("/api/ops/login")
def login(body: LoginIn, request: Request, response: Response):
    if not login_limit.allow(_client(request)):
        raise HTTPException(429, "Слишком много попыток входа")
    if not hmac.compare_digest(body.token.encode(), operator_token().encode()):
        raise HTTPException(401, "Неверный токен аналитика")
    session = secrets.token_urlsafe(32)
    expires = utc_now() + timedelta(hours=SESSION_HOURS)
    with connect() as db:
        db.execute("DELETE FROM operator_session WHERE expires_at <= ?", (now(),))
        db.execute("INSERT INTO operator_session(token_hash, name, created_at, expires_at) "
                   "VALUES (?, ?, ?, ?)", (_hash(session), body.name, now(), iso(expires)))
    response.set_cookie(COOKIE, session, max_age=SESSION_HOURS * 3600, httponly=True,
                        samesite="strict", secure=request.url.scheme == "https", path="/api")
    return {"name": body.name, "expires_at": iso(expires)}


@router.post("/api/ops/logout")
def logout(request: Request, response: Response, _: str = Depends(require_operator)):
    with connect() as db:
        db.execute("DELETE FROM operator_session WHERE token_hash = ?",
                   (_hash(request.cookies.get(COOKIE, "")),))
    response.delete_cookie(COOKIE, path="/api")
    return {"ok": True}


@router.get("/api/ops/me")
def me(name: str = Depends(require_operator)):
    return {"name": name}


# ---------------------------------------------------------------- moderation queue


def _report_dict(db: sqlite3.Connection, row: sqlite3.Row) -> dict:
    photos = [{"url": f"/api/ops/attachments/{item['stored_name']}", "width": item["width"],
               "height": item["height"]}
              for item in db.execute("SELECT * FROM attachment WHERE report_id = ?",
                                     (row["report_id"],))]
    linked = [item["event_id"] for item in db.execute(
        "SELECT event_id FROM evidence WHERE report_id = ?", (row["report_id"],))]
    return {"report_id": row["report_id"], "public_id": row["public_id"],
            "kind": row["kind"], "kind_label": reports.KINDS[row["kind"]],
            "description": row["description"], "lat": row["lat"], "lon": row["lon"],
            "place_text": row["place_text"], "location_precision": row["location_precision"],
            "observed_at": row["observed_at"],
            "observed_time_precision": row["observed_time_precision"],
            "observed_tz": row["observed_tz"], "submitted_at": row["submitted_at"],
            "moderation_status": row["moderation_status"], "is_demo": bool(row["is_demo"]),
            "retention_until": row["retention_until"], "photos": photos, "event_ids": linked}


@router.get("/api/ops/reports")
def list_reports(status: Literal["pending", "approved", "rejected", "all"] = "pending",
                 _: str = Depends(require_operator)):
    with connect() as db:
        query = "SELECT * FROM citizen_report"
        params: tuple = ()
        if status != "all":
            query, params = query + " WHERE moderation_status = ?", (status,)
        rows = db.execute(query + " ORDER BY submitted_at DESC LIMIT 200", params).fetchall()
        return [_report_dict(db, row) for row in rows]


@router.get("/api/ops/attachments/{name}")
def attachment(name: str, _: str = Depends(require_operator)):
    if not re.fullmatch(r"[0-9a-f]{32}\.jpg", name):
        raise HTTPException(404, "Файл не найден")
    path = attachments_dir() / name
    if not path.is_file():
        raise HTTPException(404, "Файл не найден")
    return Response(path.read_bytes(), media_type="image/jpeg", headers={
        "Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff",
        "Content-Security-Policy": "default-src 'none'", "Content-Disposition": "inline"})


def _report_time(row: sqlite3.Row) -> tuple[str, str]:
    return row["observed_at"], "day" if row["observed_time_precision"] == "approx_day" else "hour"


@router.get("/api/ops/reports/{report_id}/candidates")
def report_candidates(report_id: int, _: str = Depends(require_operator)):
    with connect() as db:
        row = db.execute("SELECT * FROM citizen_report WHERE report_id = ?",
                         (report_id,)).fetchone()
        if row is None:
            raise HTTPException(404, "Обращение не найдено")
        start, precision = _report_time(row)
        return registry.merge_candidates(
            db, lat=row["lat"], lon=row["lon"], geo_precision=row["location_precision"],
            country=None, start=start, end=None, time_precision=precision,
            report_type="occurred", is_demo=bool(row["is_demo"]))


class ReportReviewIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decision: Literal["approve", "reject"]
    comment: str | None = Field(default=None, max_length=1000)
    event_id: int | None = None


@router.post("/api/ops/reports/{report_id}/review")
def review_report(report_id: int, body: ReportReviewIn, name: str = Depends(require_operator)):
    with connect() as db:
        row = db.execute("SELECT * FROM citizen_report WHERE report_id = ?",
                         (report_id,)).fetchone()
        if row is None:
            raise HTTPException(404, "Обращение не найдено")
        if row["moderation_status"] != "pending":
            raise HTTPException(409, "Обращение уже проверено")
        state_after = "approved" if body.decision == "approve" else "rejected"
        event_id = None
        if body.decision == "approve":
            if body.event_id is not None:
                target = db.execute("SELECT review_state FROM event WHERE event_id = ?",
                                    (body.event_id,)).fetchone()
                if target is None or target["review_state"] in registry.CLOSED_STATES:
                    raise HTTPException(409, "Событие для привязки закрыто или не найдено")
                event_id = body.event_id
            else:
                start, precision = _report_time(row)
                located = row["lat"] is not None
                place = row["place_text"] or f"{row['lat']:.4f}, {row['lon']:.4f}"
                event_id = registry.create_event(
                    db, name, title=f"{reports.KINDS[row['kind']]}: {place}",
                    report_type="occurred", place_label=row["place_text"],
                    lat=row["lat"], lon=row["lon"],
                    geo_precision=row["location_precision"] if located else "unknown",
                    geo_status="resolved" if located else "missing",
                    event_time_start=start, time_precision=precision,
                    is_demo=row["is_demo"])
            registry.link(db, event_id, origin="citizen", linked_by=name, link_kind="manual",
                          report_id=report_id, facts={"kind": row["kind"]})
        db.execute("UPDATE citizen_report SET moderation_status = ? WHERE report_id = ?",
                   (state_after, report_id))
        db.execute("INSERT INTO review(subject_type, subject_id, reviewer, reviewed_at, decision, "
                   "comment, state_before, state_after) VALUES ('report', ?, ?, ?, ?, ?, ?, ?)",
                   (report_id, name, now(), body.decision, body.comment,
                    row["moderation_status"], state_after))
        return {"report_id": report_id, "moderation_status": state_after, "event_id": event_id}


# ---------------------------------------------------------------- event registry


def _origins(db: sqlite3.Connection, event_id: int) -> dict:
    counts = {"media": 0, "bulletin": 0, "citizen": 0}
    for row in db.execute("SELECT origin, COUNT(*) AS n FROM evidence WHERE event_id = ? "
                          "GROUP BY origin", (event_id,)):
        counts[row["origin"]] = row["n"]
    return counts


def _event_dict(db: sqlite3.Connection, row: sqlite3.Row) -> dict:
    result = {key: row[key] for key in row.keys()}  # noqa: SIM118 (Row iterates values)
    result["bbox"] = json.loads(row["bbox_json"]) if row["bbox_json"] else None
    result["geo_candidates"] = (json.loads(row["geo_candidates_json"])
                                if row["geo_candidates_json"] else None)
    for key in ("bbox_json", "geo_candidates_json"):
        result.pop(key)
    result["is_demo"], result["is_replay"] = bool(row["is_demo"]), bool(row["is_replay"])
    result["evidence_counts"] = _origins(db, row["event_id"])
    result["independent_origin_count"] = registry.independent_origin_count(db, row["event_id"])
    result["area_radius_km"] = registry.AREA_RADIUS_KM.get(row["geo_precision"])
    return result


@router.get("/api/ops/events")
def list_events(origin: Annotated[list[Literal["media", "bulletin", "citizen"]], Query()] = (),
                report_type: Annotated[list[str], Query()] = (),
                review_state: Annotated[list[str], Query()] = (),
                days: Annotated[int, Query(ge=1, le=3650)] = 30,
                include_closed: bool = False,
                _: str = Depends(require_operator)):
    since = iso(utc_now() - timedelta(days=days))
    with connect() as db:
        rows = db.execute(
            "SELECT * FROM event WHERE COALESCE(event_time_end, event_time_start, "
            "first_detected_at) >= ? OR updated_at >= ? ORDER BY updated_at DESC LIMIT 1000",
            (since, since)).fetchall()
        result = []
        for row in rows:
            if not include_closed and row["review_state"] in registry.CLOSED_STATES:
                continue
            if report_type and row["report_type"] not in report_type:
                continue
            if not report_type and row["report_type"] == "historical":
                continue  # spec §Г: historical material is hidden by default
            if review_state and row["review_state"] not in review_state:
                continue
            item = _event_dict(db, row)
            if origin and not any(item["evidence_counts"][key] for key in origin):
                continue
            result.append(item)
        return result


def _evidence(db: sqlite3.Connection, event_id: int) -> list[dict]:
    items = []
    for row in db.execute("SELECT * FROM evidence WHERE event_id = ? ORDER BY linked_at",
                          (event_id,)):
        item = {"evidence_id": row["evidence_id"], "origin": row["origin"],
                "link_kind": row["link_kind"], "linked_at": row["linked_at"],
                "linked_by": row["linked_by"], "is_original": bool(row["is_original"]),
                "facts": json.loads(row["facts_json"]) if row["facts_json"] else None}
        if row["publication_id"] is not None:
            pub = db.execute(
                "SELECT p.*, s.name AS source_name, s.kind AS source_kind, s.attribution "
                "FROM publication p JOIN source s ON s.source_id = p.source_id "
                "WHERE p.publication_id = ?", (row["publication_id"],)).fetchone()
            item["publication"] = {
                key: pub[key] for key in ("publication_id", "url", "domain", "title",
                                          "text_excerpt", "content_level", "language",
                                          "published_at", "fetched_at", "report_type",
                                          "extraction_method", "extraction_confidence",
                                          "original_source_hint", "source_name",
                                          "source_kind", "attribution")}
        else:
            report = db.execute("SELECT * FROM citizen_report WHERE report_id = ?",
                                (row["report_id"],)).fetchone()
            item["report"] = _report_dict(db, report)
        items.append(item)
    return items


def _gaps(event: dict, evidence: list[dict], matches: list[dict]) -> list[str]:
    gaps = []
    if event["geo_status"] != "resolved":
        gaps.append("Место не определено: уточните его, чтобы искать наблюдения.")
    elif event["geo_precision"] in ("district", "region", "country"):
        gaps.append("Место известно только до района или региона.")
    if event["time_precision"] == "unknown":
        gaps.append("Время события неизвестно.")
    if event["independent_origin_count"] <= 1:
        gaps.append("Один независимый первоисточник: перепечатки не считаются подтверждением.")
    if event["report_type"] in ("forecast", "warning"):
        gaps.append("Это прогноз или предупреждение, а не сообщение о произошедшем.")
    if not any(item["suitability"] == "suitable" for item in matches):
        gaps.append("Нет подходящего спутникового наблюдения по месту и времени.")
    if event["verification_result"] == "not_checked":
        gaps.append("Аналитик ещё не проверял сообщение.")
    return gaps


@router.get("/api/ops/events/{event_id}")
def event_card(event_id: int, _: str = Depends(require_operator)):
    with connect() as db:
        row = db.execute("SELECT * FROM event WHERE event_id = ?", (event_id,)).fetchone()
        if row is None:
            raise HTTPException(404, "Событие не найдено")
        event = _event_dict(db, row)
        evidence = _evidence(db, event_id)
        history = [dict(item) for item in db.execute(
            "SELECT at, actor, action, before_json, after_json, note FROM event_history "
            "WHERE event_id = ? ORDER BY history_id", (event_id,))]
        reviews = [dict(item) for item in db.execute(
            "SELECT * FROM review WHERE subject_type = 'event' AND subject_id = ? "
            "ORDER BY review_id", (event_id,))]
        matches = [dict(item) for item in db.execute(
            "SELECT * FROM observation_match WHERE event_id = ? "
            "ORDER BY suitability != 'suitable', ABS(COALESCE(time_offset_days, 1e9))",
            (event_id,))]
        candidates = registry.merge_candidates(
            db, lat=row["lat"], lon=row["lon"], geo_precision=row["geo_precision"],
            country=row["country"], start=row["event_time_start"], end=row["event_time_end"],
            time_precision=row["time_precision"], report_type=row["report_type"],
            is_demo=bool(row["is_demo"]), exclude=(event_id,))
    return {"event": event, "evidence": evidence, "history": history, "reviews": reviews,
            "observations": matches, "merge_candidates": candidates,
            "gaps": _gaps(event, evidence, matches)}


class EventEditIn(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    place_label: str | None = Field(default=None, max_length=200)
    lat: float | None = Field(default=None, ge=-90, le=90)
    lon: float | None = Field(default=None, ge=-180, le=180)
    geo_precision: Literal["point", "street", "locality", "district", "region", "country",
                           "unknown"] | None = None
    event_time_start: datetime | None = None
    event_time_end: datetime | None = None
    time_precision: Literal["hour", "day", "several_days", "unknown"] | None = None
    report_type: Literal["occurred", "forecast", "warning", "negation", "historical",
                         "unclear"] | None = None
    note: str | None = Field(default=None, max_length=1000)


@router.patch("/api/ops/events/{event_id}")
def edit_event(event_id: int, body: EventEditIn, name: str = Depends(require_operator)):
    changes = body.model_dump(exclude_unset=True, exclude={"note"})
    if ("lat" in changes) != ("lon" in changes):
        raise HTTPException(422, "Нужны обе координаты")
    if changes.get("lat") == 0 and changes.get("lon") == 0:
        raise HTTPException(422, "Координаты 0, 0 похожи на пустое значение")
    for key in ("event_time_start", "event_time_end"):
        if changes.get(key) is not None:
            if changes[key].tzinfo is None:
                raise HTTPException(422, "Время должно быть с часовым поясом")
            changes[key] = iso(changes[key])
    if "lat" in changes:
        located = changes["lat"] is not None
        changes["geo_status"] = "resolved" if located else "missing"
        changes.setdefault("geo_precision", "point" if located else "unknown")
        changes["bbox_json"] = None
    with connect() as db:
        row = db.execute("SELECT review_state FROM event WHERE event_id = ?",
                         (event_id,)).fetchone()
        if row is None:
            raise HTTPException(404, "Событие не найдено")
        if row["review_state"] in registry.CLOSED_STATES:
            raise HTTPException(409, "Событие закрыто")
        if changes.get("geo_status") == "resolved" and row["review_state"] == "needs_geo":
            changes["review_state"] = "ready"
        registry.update_event(db, event_id, changes, name, note=body.note)
    return event_card(event_id, name)


class MergeIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    into_event_id: int
    note: str | None = Field(default=None, max_length=1000)


@router.post("/api/ops/events/{event_id}/merge")
def merge(event_id: int, body: MergeIn, name: str = Depends(require_operator)):
    with connect() as db:
        try:
            registry.merge_events(db, event_id, body.into_event_id, name, body.note)
        except KeyError as exc:
            raise HTTPException(404, "Событие не найдено") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
    return event_card(body.into_event_id, name)


class SplitIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    evidence_ids: list[int] = Field(min_length=1, max_length=500)
    note: str | None = Field(default=None, max_length=1000)


@router.post("/api/ops/events/{event_id}/split")
def split(event_id: int, body: SplitIn, name: str = Depends(require_operator)):
    with connect() as db:
        try:
            new_id = registry.split_event(db, event_id, body.evidence_ids, name, body.note)
        except KeyError as exc:
            raise HTTPException(404, "Событие не найдено") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
    return {"event_id": event_id, "new_event_id": new_id}


class EventReviewIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["start", "result", "dismiss", "reopen"]
    verification_result: Literal["not_checked", "consistent", "discrepancy",
                                 "insufficient_data"] | None = None
    follow_up: Literal["none", "needs_imagery"] = "none"
    basis: str | None = Field(default=None, max_length=1000)
    comment: str | None = Field(default=None, max_length=1000)


@router.post("/api/ops/events/{event_id}/review")
def review_event(event_id: int, body: EventReviewIn, name: str = Depends(require_operator)):
    with connect() as db:
        row = db.execute("SELECT * FROM event WHERE event_id = ?", (event_id,)).fetchone()
        if row is None:
            raise HTTPException(404, "Событие не найдено")
        if row["review_state"] in ("merged", "split"):
            raise HTTPException(409, "Событие закрыто")
        changes: dict = {}
        if body.action == "start":
            changes["review_state"] = "under_review"
        elif body.action == "dismiss":
            if not body.comment:
                raise HTTPException(422, "Укажите причину")
            changes["review_state"] = "dismissed"
        elif body.action == "reopen":
            changes["review_state"] = "ready" if row["geo_status"] == "resolved" else "needs_geo"
        else:
            result = body.verification_result
            if result is None:
                raise HTTPException(422, "Укажите результат проверки")
            if result in ("consistent", "discrepancy"):
                # spec §7: without a suitable observation only insufficient_data is allowed
                suitable = db.execute(
                    "SELECT COUNT(*) FROM observation_match WHERE event_id = ? "
                    "AND suitability = 'suitable'", (event_id,)).fetchone()[0]
                if not suitable:
                    raise HTTPException(409, "Без подходящего наблюдения доступен только "
                                             "результат «недостаточно данных»")
            changes["verification_result"] = result
            changes["review_state"] = "reviewed"
        changes["follow_up"] = body.follow_up
        registry.update_event(db, event_id, changes, name, action=f"review_{body.action}",
                              note=body.comment)
        db.execute("INSERT INTO review(subject_type, subject_id, reviewer, reviewed_at, decision, "
                   "verification_result, basis, comment, follow_up, state_before, state_after) "
                   "VALUES ('event', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                   (event_id, name, now(), body.action, changes.get("verification_result"),
                    body.basis, body.comment, body.follow_up, row["review_state"],
                    changes["review_state"] if "review_state" in changes
                    else row["review_state"]))
    return event_card(event_id, name)


@router.get("/api/ops/summary")
def summary(_: str = Depends(require_operator)):
    with connect() as db:
        pending = db.execute("SELECT COUNT(*) FROM citizen_report "
                             "WHERE moderation_status = 'pending'").fetchone()[0]
        needs_geo = db.execute("SELECT COUNT(*) FROM event "
                               "WHERE review_state = 'needs_geo'").fetchone()[0]
        heartbeat = get_setting(db, "worker_heartbeat_at")
    alive = bool(heartbeat) and utc_now() - parse_iso(heartbeat) < timedelta(seconds=90)
    return {"pending_reports": pending, "needs_geo": needs_geo,
            "worker": {"alive": alive, "heartbeat_at": heartbeat}}


def purge_expired(db: sqlite3.Connection) -> int:
    expired = [row["report_id"] for row in db.execute(
        "SELECT report_id FROM citizen_report WHERE retention_until <= ?", (now(),))]
    for report_id in expired:
        delete_report(db, report_id, "retention", "срок хранения истёк")
    return len(expired)

