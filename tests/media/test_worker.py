"""One cycle at a time, an off switch, and schedule by interval."""

from datetime import UTC, datetime, timedelta

from floodvalue.media import pipeline, worker
from floodvalue.media.store import set_kv


def test_second_cycle_is_refused_while_first_holds_lock(settings, conn):
    with worker.cycle_lock(settings) as got:
        assert got
        assert worker.run_cycle(settings, manual=True) is None


def test_monitoring_switch_and_schedule(settings, conn, monkeypatch):
    calls = []
    monkeypatch.setitem(pipeline.ADAPTERS, "gdacs", lambda f, c, cur: calls.append("gdacs") or ([], "c"))
    monkeypatch.setitem(pipeline.ADAPTERS, "gdelt_doc", lambda f, c, cur: calls.append("gdelt") or ([], "c"))
    set_kv(conn, "monitoring_enabled", "0")
    assert worker.run_cycle(settings) == []
    set_kv(conn, "monitoring_enabled", "1")
    first = worker.run_cycle(settings)
    assert sorted(r["source_id"] for r in first) == ["gdacs_floods", "gdelt_floods"]
    assert worker.run_cycle(settings) == []  # not due yet (poll_interval_s = 900)
    assert calls == ["gdacs", "gdelt"]
    manual = worker.run_cycle(settings, only="gdacs_floods", manual=True)
    assert [r["source_id"] for r in manual] == ["gdacs_floods"]


def test_due():
    now = datetime(2026, 9, 26, 12, 0, tzinfo=UTC)
    row = {"last_attempt_at": (now - timedelta(minutes=14)).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "poll_interval_s": 900}
    assert not worker._due(row, now, None)
    assert worker._due(row, now, 600)
    assert worker._due({"last_attempt_at": None, "poll_interval_s": 900}, now, None)
