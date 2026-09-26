"""Server-side collection: one process, a file lock so a cycle never runs twice in parallel
(worker and a manual run share it), a configurable interval (15 min is a starting setting,
not a detection-latency promise), an on/off switch in the database, per-cycle budgets and a
clean stop on SIGTERM/SIGINT between items — every item is committed as it is processed."""

from __future__ import annotations

import signal
import sys
import time
from contextlib import closing, contextmanager
from datetime import UTC, datetime

from floodvalue.media.config import Settings, load_sources
from floodvalue.media.netfetch import Budget
from floodvalue.media.pipeline import make_geocoder, recover_interrupted, run_source
from floodvalue.media.reports import purge_expired
from floodvalue.media.store import connect, get_kv, sync_sources, transaction

if sys.platform == "win32":  # no fcntl on Windows (#133): lock byte 0 of the lock file
    import msvcrt

    def _try_lock(f) -> bool:
        try:
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True

    def _unlock(f) -> None:
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
else:
    import fcntl

    def _try_lock(f) -> bool:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        return True

    def _unlock(f) -> None:
        fcntl.flock(f, fcntl.LOCK_UN)

_stop = False


def _request_stop(*_):
    global _stop
    _stop = True


@contextmanager
def cycle_lock(settings: Settings):
    path = settings.db_path.with_suffix(".lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        if not _try_lock(f):
            yield False
            return
        try:
            yield True
        finally:
            _unlock(f)


def _due(source, now: datetime, override: int | None) -> bool:
    last = source["last_attempt_at"]
    if not last:
        return True
    age = now - datetime.strptime(last, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    return age.total_seconds() >= (override or source["poll_interval_s"])


def run_cycle(settings: Settings, only: str | None = None, manual: bool = False) -> list | None:
    """Runs due (or, manually, all enabled / the named) sources. None = another cycle holds
    the lock. One request budget is shared by the whole cycle."""
    with cycle_lock(settings) as got:
        if not got:
            return None
        with closing(connect(settings.db_path)) as conn:
            sync_sources(conn, load_sources(settings.sources_yaml))
            if not manual and get_kv(conn, "monitoring_enabled", "1") != "1":
                return []
            budget = Budget(settings.max_requests_per_cycle)
            geocoder = make_geocoder(conn, settings, budget)
            now = datetime.now(UTC)
            results = []
            for source in conn.execute("SELECT * FROM sources WHERE enabled=1 ORDER BY source_id"
                                       ).fetchall():
                if only and source["source_id"] != only:
                    continue
                if not manual and not _due(source, now, settings.poll_interval_s):
                    continue
                if _stop:
                    break
                results.append(run_source(conn, settings, source["source_id"], budget=budget,
                                          geocoder=geocoder, should_stop=lambda: _stop))
            with transaction(conn):
                purge_expired(conn, settings.photo_dir)
            return results


def serve_forever(settings: Settings, tick_s: int = 30) -> None:
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)
    with closing(connect(settings.db_path)) as conn:
        recovered = recover_interrupted(conn)
    if recovered:
        print(f"прерванных запусков помечено: {recovered}", flush=True)
    interval = (f"{settings.poll_interval_s} с для всех источников" if settings.poll_interval_s
                else "poll_interval_s каждого источника (sources.yaml)")
    print(f"worker: интервал — {interval}; проверка расписания каждые {tick_s} с", flush=True)
    while not _stop:
        results = run_cycle(settings)
        for r in results or []:
            print(f"{datetime.now(UTC):%Y-%m-%dT%H:%M:%SZ} {r['source_id']}: {r['status']} "
                  f"новых {r['items_new']}, обновлено {r['items_updated']}, "
                  f"запросов {r['requests_used']}" + (f" — {r['error']}" if r["error"] else ""),
                  flush=True)
        for _ in range(tick_s):
            if _stop:
                break
            time.sleep(1)
    print("worker остановлен, записанные данные сохранены", flush=True)
