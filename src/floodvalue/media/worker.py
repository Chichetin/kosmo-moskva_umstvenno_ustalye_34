"""Background worker: one process, one lock, configurable interval, graceful stop."""

from __future__ import annotations

import fcntl
import signal
import time
from datetime import UTC, datetime, timedelta

from floodvalue.media import api, ingest
from floodvalue.media.store import connect, iso, media_dir, parse_iso, set_setting

TICK_S = 5
PURGE_EVERY = timedelta(hours=1)


def run_forever() -> int:
    lock = open(media_dir() / "worker.lock", "w")  # noqa: SIM115 (held while running)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("worker уже запущен: второй экземпляр не нужен")
        return 1
    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True
        print("остановка: дописываю текущую публикацию и выхожу")

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    last_purge = datetime.min.replace(tzinfo=UTC)
    print(f"worker запущен, хранилище {media_dir()}")
    while not stopping:
        moment = datetime.now(UTC)
        with connect() as db:
            set_setting(db, "worker_heartbeat_at", iso(moment))
            settings = ingest.monitoring_settings(db)
            requested = settings["run_requested"]
            if requested:
                db.execute("DELETE FROM setting WHERE key = 'run_requested_at'")
        last = settings["last_cycle_at"]
        due = settings["enabled"] and (
            last is None or moment - parse_iso(last) >= timedelta(seconds=settings["interval_s"]))
        if requested or due:
            result = ingest.run_cycle("manual" if requested else "schedule",
                                      stop=lambda: stopping)
            print(f"{iso(datetime.now(UTC))} цикл: {result.get('status')}, "
                  f"новых публикаций {result.get('items_new', 0)}, "
                  f"новых событий {result.get('events_new', 0)}")
        if moment - last_purge >= PURGE_EVERY:
            with connect() as db:
                removed = api.purge_expired(db)
            if removed:
                print(f"удалено обращений по сроку хранения: {removed}")
            last_purge = moment
        for _ in range(TICK_S * 2):
            if stopping:
                break
            time.sleep(0.5)
    return 0
