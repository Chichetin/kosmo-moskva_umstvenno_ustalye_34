"""HTTP layer: the public eyewitness page (/report) with its own simple layout, a status page by
unguessable token, and analyst endpoints under /api/media/* guarded by authorisation.

Analyst access: FLOODVALUE_MEDIA_ANALYST_TOKEN set → Bearer token or a session cookie from
/media/login; not set → loopback clients only ("local until configured"). Public intake is
open only to loopback unless FLOODVALUE_MEDIA_PUBLIC_REPORTS=1 is set deliberately.
GET /api/media/events is the GeoJSON joint for the globe's future «Сообщения» layer.
GET /api/media/map is the operator map's read-only layer: events plus pending eyewitness reports
without personal data. FLOODVALUE_MEDIA_OPERATOR_MAP_PUBLIC=1 opens it without login (the operator
UI is public for now); then points of reports and events are coarsened to
MAP_PUBLIC_GRID degrees and the hour, and event titles and places are dropped."""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import threading
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from pydantic import ValidationError

from floodvalue.media import events as ev
from floodvalue.media import observations as obs
from floodvalue.media.config import Settings, load_settings, load_sources
from floodvalue.media.netfetch import Budget, Fetcher, FetchError
from floodvalue.media.reports import (
    CATEGORIES,
    MAX_PHOTO_BYTES,
    MAX_PHOTOS,
    ReportError,
    ReportIn,
    create_report,
    public_status,
    rate_limited,
    server_secret,
)
from floodvalue.media.store import connect, get_kv, loads, set_kv, sync_sources, transaction
from floodvalue.paths import data_dir, output_dir

STATIC = Path(__file__).parent / "static"
MAX_BODY = MAX_PHOTOS * MAX_PHOTO_BYTES + 256 * 1024
COOKIE = "fv_media_session"
CSP = ("default-src 'self'; script-src 'self' https://cdn.jsdelivr.net; "
       "style-src 'self' https://cdn.jsdelivr.net; img-src 'self' data: blob: https://cdn.jsdelivr.net "
       "https://tile.openstreetmap.org; connect-src 'self'; frame-ancestors 'none'; "
       "base-uri 'none'; form-action 'self'")
STATUS_LIMIT = (60, 3600)
MAP_PUBLIC_GRID = 0.01  # ≈ 1 km: a public map shows where a report is, not the eyewitness's door


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    app = FastAPI(title="FloodValue media", docs_url=None, redoc_url=None, openapi_url=None)
    if not settings.analyst_token:
        # Local-only mode trusts loopback clients: refuse foreign Host headers (DNS rebinding).
        app.add_middleware(TrustedHostMiddleware,
                           allowed_hosts=list(settings.local_hosts))
    ingest_lock = threading.Lock()
    with closing(connect(settings.db_path)) as conn:
        sync_sources(conn, load_sources(settings.sources_yaml))
        secret = server_secret(conn)

    def db():
        return closing(connect(settings.db_path))

    def is_loopback(request: Request) -> bool:
        host = request.client.host if request.client else ""
        try:
            local = ipaddress.ip_address(host).is_loopback
        except ValueError:
            local = False
        return local and "x-forwarded-for" not in request.headers

    def session_value() -> str:
        return hmac.new(secret, (settings.analyst_token or "").encode(), hashlib.sha256).hexdigest()

    def analyst(request: Request) -> str:
        """Returns the actor name or raises 401/403. A hidden link is not authorisation."""
        if settings.analyst_token:
            auth = request.headers.get("authorization", "")
            if auth.startswith("Bearer ") and hmac.compare_digest(auth[7:], settings.analyst_token):
                return "analyst:token"
            cookie = request.cookies.get(COOKIE, "")
            if cookie and hmac.compare_digest(cookie, session_value()):
                return "analyst:session"
            raise HTTPException(401, "нужен вход аналитика")
        if is_loopback(request):
            return "analyst:local"
        raise HTTPException(403, "операторские действия доступны только локально: "
                                 "задайте FLOODVALUE_MEDIA_ANALYST_TOKEN")

    def analyst_write(request: Request) -> str:
        """State-changing calls also need a custom header, which a cross-site form can't send."""
        actor = analyst(request)
        if request.headers.get("x-floodvalue-media") != "1":
            raise HTTPException(403, "нет заголовка X-FloodValue-Media")
        return actor

    def client_bucket(request: Request, kind: str) -> str:
        host = request.client.host if request.client else "?"
        return kind + ":" + hashlib.sha256(secret + host.encode()).hexdigest()[:24]

    def page(name: str) -> FileResponse:
        return FileResponse(STATIC / name, media_type="text/html; charset=utf-8",
                            headers={"Content-Security-Policy": CSP,
                                     "X-Content-Type-Options": "nosniff",
                                     "Referrer-Policy": "no-referrer",
                                     "Cache-Control": "no-store"})

    # --- public ----------------------------------------------------------------------------

    @app.get("/report")
    def report_page():
        return page("report.html")

    @app.get("/report/status")
    def status_page():
        return page("status.html")

    @app.get("/media/static/{name}")
    def static_file(name: str):
        path = (STATIC / name).resolve()
        if path.parent != STATIC.resolve() or path.suffix not in (".js", ".css") or not path.exists():
            raise HTTPException(404)
        media = "text/javascript" if path.suffix == ".js" else "text/css"
        return FileResponse(path, media_type=media + "; charset=utf-8",
                            headers={"X-Content-Type-Options": "nosniff"})

    @app.get("/api/media/report-config")
    def report_config(request: Request):
        return {"accepting": settings.public_reports or is_loopback(request),
                "categories": CATEGORIES, "max_photos": MAX_PHOTOS,
                "max_photo_mb": MAX_PHOTO_BYTES // (1024 * 1024)}

    @app.post("/api/media/reports")
    async def submit_report(request: Request):
        if not (settings.public_reports or is_loopback(request)):
            raise HTTPException(403, "приём обращений через интернет не включён")
        length = request.headers.get("content-length")
        if length is None:
            raise HTTPException(411, "нужен заголовок Content-Length")
        if not length.isdigit() or int(length) > MAX_BODY:
            raise HTTPException(413, "слишком большой запрос")
        with db() as conn:
            if rate_limited(conn, client_bucket(request, "report"), settings.reports_per_hour,
                            3600):
                raise HTTPException(429, "слишком много обращений, попробуйте позже")
        form = await request.form(max_files=MAX_PHOTOS + 1, max_fields=20,
                                  max_part_size=64 * 1024)
        photos = []
        for upload in form.getlist("photos"):
            if isinstance(upload, str) or not upload.filename:
                continue
            raw = await upload.read(MAX_PHOTO_BYTES + 1)
            if len(raw) > MAX_PHOTO_BYTES:
                return _field_error("photos", "фото больше 8 МБ")
            photos.append(raw)
        fields = {k: v for k, v in form.items() if k != "photos" and isinstance(v, str) and v != ""}
        try:
            data = ReportIn(**fields)
        except ValidationError as exc:
            err = exc.errors()[0]
            return _field_error(".".join(map(str, err["loc"])) or "form",
                                str(err["msg"]).removeprefix("Value error, "))
        try:
            with db() as conn:
                report_id, token, created = create_report(conn, data, photos, settings.photo_dir,
                                                          settings.retention_days)
        except ReportError as exc:
            return _field_error(exc.field, str(exc))
        return JSONResponse({"report_id": report_id, "status_token": token, "created": created},
                            status_code=201 if created else 200)

    @app.post("/api/media/reports/status")
    async def report_status(request: Request):
        with db() as conn:
            if rate_limited(conn, client_bucket(request, "status"), *STATUS_LIMIT):
                raise HTTPException(429, "слишком много запросов")
            body = await _body(request)
            token = str(body.get("token", ""))[:80]
            status = public_status(conn, token) if len(token) == 40 else None
        if status is None:
            raise HTTPException(404, "обращение не найдено")
        return status

    # --- analyst: login --------------------------------------------------------------------

    @app.get("/media/analyst")
    def analyst_page():
        return page("analyst.html")

    @app.post("/media/login")
    async def login(request: Request):
        if not settings.analyst_token:
            raise HTTPException(400, "токен аналитика не настроен — вход только локальный")
        with db() as conn:  # every attempt counts, checked before comparing
            if rate_limited(conn, client_bucket(request, "login"), 10, 600):
                raise HTTPException(429, "слишком много попыток входа, подождите 10 минут")
        form = await request.form(max_files=0, max_fields=2)
        token = str(form.get("token", ""))
        if not hmac.compare_digest(token, settings.analyst_token):
            raise HTTPException(401, "неверный токен")
        resp = RedirectResponse("/media/analyst", status_code=303)
        resp.set_cookie(COOKIE, session_value(), httponly=True, samesite="strict",
                        secure=request.url.scheme == "https", max_age=12 * 3600)
        return resp

    @app.get("/api/media/whoami")
    def whoami(request: Request):
        return {"actor": analyst(request)}

    # --- operator map ----------------------------------------------------------------------

    @app.get("/api/media/map")
    def operator_map(request: Request, include_demo: bool = True):
        """Events and not-yet-moderated eyewitness reports for the operator's «Сообщения» layer.
        Accepted reports are already inside events; rejected and spam are never shown."""
        exact = True
        if not settings.operator_map_public:
            analyst(request)
        else:
            try:
                analyst(request)
            except HTTPException:
                exact = False
        with db() as conn:
            events = _event_features(conn, None, None, None, None, None, False, include_demo)
            reports = _report_features(conn, include_demo, exact)
        if not exact:
            events = [_public_event(f) for f in events]
        return {"events": {"type": "FeatureCollection", "features": events},
                "reports": {"type": "FeatureCollection", "features": reports},
                "exact_locations": exact}

    # --- analyst: reports queue ------------------------------------------------------------

    @app.get("/api/media/reports")
    def list_reports(request: Request, status: str = "pending"):
        analyst(request)
        with db() as conn:
            rows = conn.execute("SELECT * FROM citizen_reports WHERE moderation_status=? "
                                "ORDER BY submitted_at DESC LIMIT 200", (status,)).fetchall()
            out = []
            for r in rows:
                photos = [a["attachment_id"] for a in conn.execute(
                    "SELECT attachment_id FROM attachments WHERE report_id=?", (r["report_id"],))]
                item = {k: r[k] for k in r.keys() if k not in ("status_token_hash",  # noqa: SIM118
                                                               "idempotency_key")}
                out.append({**item, "category_label": CATEGORIES.get(r["category"]),
                            "photos": photos})
        return {"reports": out}

    @app.get("/api/media/attachments/{attachment_id}")
    def attachment(request: Request, attachment_id: str):
        analyst(request)
        with db() as conn:
            row = conn.execute("SELECT file_name FROM attachments WHERE attachment_id=?",
                               (attachment_id,)).fetchone()
        if row is None:
            raise HTTPException(404)
        return FileResponse(settings.photo_dir / row["file_name"], media_type="image/jpeg",
                            headers={"X-Content-Type-Options": "nosniff",
                                     "Content-Security-Policy": "sandbox; default-src 'none'",
                                     "Cache-Control": "private, no-store"})

    @app.post("/api/media/reports/{report_id}/moderate")
    async def moderate(request: Request, report_id: str):
        actor = analyst_write(request)
        body = await _body(request)
        event_id = body.get("event_id")
        with db() as conn, transaction(conn):
            event_id = ev.moderate_report(conn, report_id, decision=str(body.get("decision")),
                                          actor=actor, comment=str(body.get("comment", ""))[:500],
                                          event_id=_int(event_id) if event_id else None)
        return {"event_id": event_id}

    # --- analyst: events -------------------------------------------------------------------

    @app.get("/api/media/events")
    def list_events(request: Request, origin: str | None = None, report_type: str | None = None,
                    review_state: str | None = None, since: str | None = None,
                    until: str | None = None, include_historical: bool = False,
                    include_demo: bool = True):
        analyst(request)
        with db() as conn:
            return {"type": "FeatureCollection",
                    "features": _event_features(conn, origin, report_type, review_state, since,
                                                until, include_historical, include_demo)}

    @app.get("/api/media/events/{event_id}")
    def event_card(request: Request, event_id: int):
        analyst(request)
        with db() as conn:
            return _card(conn, event_id)

    @app.post("/api/media/events/{event_id}/review")
    async def review(request: Request, event_id: int):
        actor = analyst_write(request)
        b = await _body(request)
        with db() as conn, transaction(conn):
            ev.review_event(conn, event_id, verification_result=str(b.get("verification_result")),
                            follow_up=str(b.get("follow_up", "none")),
                            basis=str(b.get("basis", ""))[:1000],
                            comment=str(b.get("comment", ""))[:1000], actor=actor)
        return {"ok": True}

    @app.post("/api/media/events/{event_id}/place")
    async def place(request: Request, event_id: int):
        actor = analyst_write(request)
        b = await _body(request)
        with db() as conn, transaction(conn):
            ev.set_place(conn, event_id, label=str(b.get("label", ""))[:200],
                         lon=_num(b.get("lon")), lat=_num(b.get("lat")),
                         precision=str(b.get("precision", "locality")), actor=actor)
        return {"ok": True}

    @app.post("/api/media/events/{event_id}/time")
    async def set_time(request: Request, event_id: int):
        actor = analyst_write(request)
        b = await _body(request)
        with db() as conn, transaction(conn):
            ev.set_time(conn, event_id, start=_iso_or_none(b.get("start")),
                        end=_iso_or_none(b.get("end")),
                        precision=str(b.get("precision", "day")), actor=actor)
        return {"ok": True}

    @app.post("/api/media/events/merge")
    async def merge(request: Request):
        actor = analyst_write(request)
        b = await _body(request)
        with db() as conn, transaction(conn):
            ev.merge_events(conn, _int(b.get("source_id")), _int(b.get("target_id")), actor)
        return {"ok": True}

    @app.post("/api/media/events/{event_id}/split")
    async def split(request: Request, event_id: int):
        actor = analyst_write(request)
        b = await _body(request)
        with db() as conn, transaction(conn):
            new_id = ev.split_event(conn, event_id, [_int(i) for i in b.get("evidence_ids") or []],
                                    actor)
        return {"event_id": new_id}

    @app.post("/api/media/events/{event_id}/observations")
    async def find_observations(request: Request, event_id: int):
        analyst_write(request)
        b = await _body(request)
        errors = []
        with db() as conn:
            event = conn.execute("SELECT * FROM events WHERE event_id=?", (event_id,)).fetchone()
            if event is None:
                raise HTTPException(404)
            matches = obs.search_local(conn, event, data_dir(), output_dir())
            if b.get("external"):
                fetcher = Fetcher(allowed_hosts=frozenset({obs.CDSE_HOST}), budget=Budget(2),
                                  user_agent=settings.user_agent)
                try:
                    matches += obs.search_cdse(fetcher, event)
                except (FetchError, ValueError) as exc:
                    errors.append(f"Copernicus Data Space: {exc}")
            with transaction(conn):
                obs.save_matches(conn, event_id, matches)
        return {"found": len(matches), "errors": errors}

    # --- analyst: sources and monitoring ---------------------------------------------------

    @app.get("/api/media/sources")
    def sources(request: Request):
        analyst(request)
        with db() as conn:
            return {"monitoring_enabled": get_kv(conn, "monitoring_enabled", "1") == "1",
                    "poll_interval_s": settings.poll_interval_s,
                    "sources": [_source_status(conn, r, settings.poll_interval_s) for r in
                                conn.execute("SELECT * FROM sources ORDER BY source_id")]}

    @app.post("/api/media/sources/{source_id}/enabled")
    async def toggle_source(request: Request, source_id: str):
        analyst_write(request)
        b = await _body(request)
        with db() as conn:
            row = conn.execute("SELECT terms_status FROM sources WHERE source_id=?",
                               (source_id,)).fetchone()
            if row is None:
                raise HTTPException(404)
            if b.get("enabled") and row["terms_status"] != "checked":
                raise HTTPException(409, "условия источника не проверены — включить нельзя")
            conn.execute("UPDATE sources SET enabled=? WHERE source_id=?",
                         (int(bool(b.get("enabled"))), source_id))
        return {"ok": True}

    @app.post("/api/media/monitoring")
    async def monitoring(request: Request):
        analyst_write(request)
        b = await _body(request)
        with db() as conn:
            set_kv(conn, "monitoring_enabled", "1" if b.get("enabled") else "0")
        return {"ok": True}

    @app.post("/api/media/ingest")
    async def ingest_now(request: Request):
        """Manual collection for checking. Same lock as the worker: never two at once."""
        analyst_write(request)
        b = await _body(request)
        from floodvalue.media.worker import run_cycle  # the worker owns the cycle lock
        if not ingest_lock.acquire(blocking=False):
            raise HTTPException(409, "сбор уже идёт")
        try:
            result = run_cycle(settings, only=b.get("source_id"), manual=True)
        finally:
            ingest_lock.release()
        if result is None:
            raise HTTPException(409, "сбор уже идёт в фоновом процессе")
        return {"runs": result}

    @app.exception_handler(ev.ActionError)
    async def action_error(_: Request, exc: ev.ActionError):
        return JSONResponse({"detail": str(exc)}, status_code=409)

    return app


async def _body(request: Request) -> dict:
    """JSON object body or 422 — never a 500 on a malformed request."""
    try:
        body = await request.json()
    except ValueError:
        body = None
    if not isinstance(body, dict):
        raise HTTPException(422, "ожидался JSON-объект")
    return body


def _field_error(field: str, message: str) -> JSONResponse:
    return JSONResponse({"detail": message, "field": field}, status_code=422)


def _int(value) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(422, "ожидался номер") from exc


def _num(value) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ev.ActionError("координаты должны быть числами") from exc


def _iso_or_none(value) -> str | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value))
    except ValueError as exc:
        raise ev.ActionError("время в формате ISO 8601 с часовым поясом") from exc
    if dt.tzinfo is None:
        raise ev.ActionError("укажите часовой пояс времени")
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _source_status(conn, r, override: int | None) -> dict:
    """disabled / never / ok / delayed / error — «no new data» and «source down» differ."""
    now = datetime.now(UTC)
    interval = override or r["poll_interval_s"]
    last_ok = r["last_success_at"]
    if not r["enabled"]:
        state = "disabled"
    elif r["last_error_at"] and (not last_ok or r["last_error_at"] > last_ok):
        state = "error"
    elif not last_ok:
        state = "never"
    else:
        age = (now - datetime.strptime(last_ok, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC))
        state = "delayed" if age.total_seconds() > 2 * interval + 60 else "ok"
    last_run = conn.execute("SELECT * FROM ingestion_runs WHERE source_id=? ORDER BY run_id DESC "
                            "LIMIT 1", (r["source_id"],)).fetchone()
    return {"source_id": r["source_id"], "name": r["name"], "kind": r["kind"], "state": state,
            "enabled": bool(r["enabled"]), "poll_interval_s": interval, "terms_status": r["terms_status"],
            "terms_url": r["terms_url"], "attribution": r["attribution"],
            "content_storage_policy": r["content_storage_policy"],
            "last_attempt_at": r["last_attempt_at"], "last_success_at": last_ok,
            "last_new_data_at": r["last_new_data_at"], "last_error_at": r["last_error_at"],
            "last_error": r["last_error"], "last_run": dict(last_run) if last_run else None}


def _event_features(conn, origin, report_type, review_state, since, until, include_historical,
                    include_demo) -> list[dict]:
    sql = ("SELECT * FROM events WHERE review_state != 'merged' AND EXISTS (SELECT 1 FROM "
           "evidence e WHERE e.event_id = events.event_id)")
    params: list = []
    if not include_historical and report_type != "historical":
        sql += " AND report_type != 'historical'"
    if report_type:
        sql += " AND report_type = ?"
        params.append(report_type)
    if review_state:
        sql += " AND review_state = ?"
        params.append(review_state)
    if origin:
        sql += " AND (',' || source_kinds || ',') LIKE ?"
        params.append(f"%,{origin},%")
    if since:
        sql += " AND COALESCE(event_time_end, event_time_start, first_detected_at) >= ?"
        params.append(since)
    if until:
        sql += " AND COALESCE(event_time_start, first_detected_at) <= ?"
        params.append(until)
    if not include_demo:
        sql += " AND is_demo = 0"
    sql += " ORDER BY updated_at DESC LIMIT 500"
    out = []
    for r in conn.execute(sql, params):
        out.append({"type": "Feature", "id": r["event_id"],
                    "geometry": loads(r["geometry_json"]),
                    "properties": {k: r[k] for k in (
                        "event_id", "title", "report_type", "place_label", "geo_precision",
                        "geo_status", "event_time_start", "event_time_end", "time_precision",
                        "first_detected_at", "updated_at", "review_state", "verification_result",
                        "independent_origin_count", "source_kinds", "is_demo", "is_replay",
                        "follow_up")}})
    return out


def _coarse(lon: float, lat: float) -> list[float]:
    return [round(round(v / MAP_PUBLIC_GRID) * MAP_PUBLIC_GRID, 6) for v in (lon, lat)]


def _coarse_time(value: str | None) -> str | None:
    """ISO 8601 → the UTC hour: a public map need not pin when exactly someone reported."""
    if not value:
        return value
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:00:00Z")


def _public_event(feature: dict) -> dict:
    """Without analyst access every event loses its title and place and gets a coarse point and
    hour. An event's text and point may come from an eyewitness even when its current evidence is
    news only (accepted report, then split or merged), so origin is not guessed from source_kinds.
    Areas (non-Point geometry) come from the geocoder, not from an eyewitness, and stay."""
    props = feature["properties"]
    geometry = feature["geometry"]
    if geometry and geometry.get("type") == "Point":
        geometry = {"type": "Point", "coordinates": _coarse(*geometry["coordinates"][:2])}
    kinds = set((props.get("source_kinds") or "").split(","))
    title = ("Сообщение очевидца" if kinds == {"citizen"} else
             "Сообщения СМИ и очевидцев" if "citizen" in kinds else "Сообщение СМИ")
    times = {k: _coarse_time(props[k]) for k in ("event_time_start", "event_time_end",
                                                  "first_detected_at", "updated_at")}
    return {**feature, "geometry": geometry,
            "properties": {**props, **times, "title": title, "place_label": None,
                           "geo_precision": "coarsened"}}


def _report_features(conn, include_demo: bool, exact: bool) -> list[dict]:
    """Pending reports with a location. No description, contacts, photos or tokens: the free text
    stays in the analyst queue. Without analyst access coordinates snap to MAP_PUBLIC_GRID."""
    sql = ("SELECT report_id, category, lat, lon, location_precision, observed_at, "
           "observed_precision, submitted_at, is_demo FROM citizen_reports "
           "WHERE moderation_status = 'pending' AND lat IS NOT NULL")
    if not include_demo:
        sql += " AND is_demo = 0"
    out = []
    for r in conn.execute(sql + " ORDER BY submitted_at DESC LIMIT 500"):
        lon, lat = (r["lon"], r["lat"]) if exact else _coarse(r["lon"], r["lat"])
        out.append({"type": "Feature", "id": r["report_id"],
                    "geometry": {"type": "Point", "coordinates": [lon, lat]},
                    "properties": {"report_id": r["report_id"], "category": r["category"],
                                   "category_label": CATEGORIES.get(r["category"]),
                                   "location_precision": r["location_precision"] if exact
                                   else "coarsened",
                                   "observed_at": r["observed_at"] if exact
                                   else _coarse_time(r["observed_at"]),
                                   "observed_precision": r["observed_precision"],
                                   "submitted_at": r["submitted_at"] if exact
                                   else _coarse_time(r["submitted_at"]),
                                   "moderation_status": "pending", "is_demo": r["is_demo"]}})
    return out


def _card(conn, event_id: int) -> dict:
    row = conn.execute("SELECT * FROM events WHERE event_id=?", (event_id,)).fetchone()
    if row is None:
        raise HTTPException(404)
    event = dict(row)
    event["geometry"] = loads(event.pop("geometry_json"))
    event["geo_candidates"] = loads(event.pop("geo_candidates_json"))
    evidence = []
    for e in conn.execute(
            """SELECT e.*, p.title, p.url, p.source_id, p.published_at, p.fetched_at,
                      p.content_scope, p.text_excerpt, p.source_domain, p.extraction_method,
                      p.extraction_status, p.time_hint_json, s.name AS source_name,
                      s.attribution, s.kind AS source_kind
               FROM evidence e LEFT JOIN publications p USING(publication_id)
               LEFT JOIN sources s ON s.source_id = p.source_id
               WHERE e.event_id=? ORDER BY e.linked_at""", (event_id,)):
        item = {k: e[k] for k in e.keys()}  # noqa: SIM118 — sqlite3.Row
        item["facts"] = loads(item.pop("facts_json"))
        item["time_hint"] = loads(item.pop("time_hint_json"))
        if e["report_id"]:
            rep = conn.execute("SELECT report_id, category, description, place_text, "
                               "location_precision, observed_at, observed_precision, "
                               "submitted_at, is_demo FROM citizen_reports WHERE report_id=?",
                               (e["report_id"],)).fetchone()
            item["report"] = dict(rep) if rep else None
            item["photos"] = [a["attachment_id"] for a in conn.execute(
                "SELECT attachment_id FROM attachments WHERE report_id=?", (e["report_id"],))]
        evidence.append(item)
    matches = [{**dict(m), "footprint": loads(m["footprint_json"])} for m in conn.execute(
        "SELECT * FROM observation_matches WHERE event_id=? ORDER BY suitability='suitable' DESC,"
        " observation_start", (event_id,))]
    for m in matches:
        m.pop("footprint_json")
    history = [dict(h) for h in conn.execute(
        "SELECT at, actor, action, before_json, after_json FROM event_history WHERE event_id=? "
        "ORDER BY history_id", (event_id,))]
    reviews = [dict(r) for r in conn.execute(
        "SELECT * FROM reviews WHERE target_kind='event' AND target_id=? ORDER BY review_id",
        (str(event_id),))]
    duplicates = []
    if event["place_label"]:
        duplicates = [dict(d) for d in conn.execute(
            "SELECT event_id, title, report_type, review_state FROM events WHERE event_id != ? "
            "AND lower(place_label) = lower(?) AND review_state NOT IN ('merged','dismissed') "
            "LIMIT 10", (event_id, event["place_label"]))]
    return {"event": event, "evidence": evidence, "observations": matches, "history": history,
            "reviews": reviews, "possible_duplicates": duplicates,
            "missing": _missing(event, evidence, matches)}


def _missing(event: dict, evidence: list[dict], matches: list[dict]) -> list[str]:
    out = []
    if event["geo_status"] != "resolved":
        out.append("место не определено — уточните (координаты не подставляются)")
    elif event["geo_precision"] in ("district", "region", "country"):
        level = {"district": "района", "region": "региона", "country": "страны"}[event["geo_precision"]]
        out.append(f"место известно только до уровня {level} — точка на карте условная")
    if event["time_precision"] == "unknown":
        out.append("время события неизвестно — есть только время публикации и получения")
    if all(not e["is_original"] for e in evidence if e["origin"] == "news") and \
            any(e["origin"] == "news" for e in evidence) and \
            not any(e["origin"] in ("bulletin", "citizen") for e in evidence):
        out.append("только перепечатки — независимого первоисточника нет")
    if all(e.get("content_scope") == "headline_only" for e in evidence if e["publication_id"]) \
            and any(e["publication_id"] for e in evidence):
        out.append("у публикаций сохранён только заголовок — текст статьи не анализировался")
    if not matches:
        out.append("наблюдения не искали или не найдены — это не опровержение")
    elif not any(m["suitability"] == "suitable" for m in matches):
        out.append("нет наблюдения, подходящего по месту и времени")
    elif not any(m["suitability"] == "suitable" and m["stage"] == "analyzed" for m in matches):
        out.append("подходящее наблюдение не проанализировано моделью")
    if event["verification_result"] == "not_checked":
        out.append("аналитик сообщение ещё не проверял")
    return out

