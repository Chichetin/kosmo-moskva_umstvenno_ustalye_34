"""Replay demo, clearly marked: texts are written by the team (fictional), tied to the 2017
Sri Lanka event of the validation split (chip Sri-Lanka_713926), is_demo=1 everywhere. It
never enters notifications, model checks or metrics, and no test chip is used."""

from __future__ import annotations

import json
import secrets

from floodvalue.media.config import Settings
from floodvalue.media.events import set_time
from floodvalue.media.pipeline import ingest_item
from floodvalue.media.reports import ReportIn, create_report
from floodvalue.media.store import transaction

DEMO_SOURCE = "demo_replay"
DEMO_POINT = (81.662, 7.750)  # inside Sri-Lanka_713926 (validation), [lon, lat]
DEMO_ITEMS = [
    ("demo-1", "https://demo.invalid/a1", "ДЕМО: Floods hit Batticaloa, hundreds evacuated"),
    ("demo-2", "https://demo.invalid/b7", "ДЕМО: Floods hit Batticaloa, hundreds evacuated"),
    ("demo-3", "https://demo.invalid/c2",
     "ДЕМО: Reports of flooding in Batticaloa town denied by officials"),
    ("demo-4", "https://demo.invalid/d9",
     "ДЕМО: Heavy rain expected to cause flooding in Batticaloa next week"),
]


class DemoGazetteer:
    """A two-line gazetteer for the replay only; real runs use Nominatim or nothing."""

    def lookup(self, query: str) -> list[dict]:
        if query.casefold() in ("batticaloa", "batticaloa town"):
            return [{"label": "Batticaloa (ДЕМО)", "lon": DEMO_POINT[0], "lat": DEMO_POINT[1],
                     "precision": "locality", "bbox": None}]
        return []


def load_demo(conn, settings: Settings) -> str:
    conn.execute(
        """INSERT OR IGNORE INTO sources(source_id, name, kind, adapter, enabled, terms_status,
             attribution, content_storage_policy, poll_interval_s)
           VALUES(?, 'ДЕМО: вымышленные публикации (replay)', 'news', 'none', 0, 'team_written',
                  'Тексты написаны командой, не реальные публикации', 'headline_only', 0)""",
        (DEMO_SOURCE,))
    source = conn.execute("SELECT * FROM sources WHERE source_id=?", (DEMO_SOURCE,)).fetchone()
    gazetteer = DemoGazetteer()
    for i, (ext, url, title) in enumerate(DEMO_ITEMS):
        item = {"external_id": ext, "url": url, "title": title, "text_excerpt": None,
                "language": "en", "source_domain": f"demo{i % 2}.invalid",
                "published_at": "2017-05-27T06:00:00Z", "source_updated_at": None,
                "structured": None}
        with transaction(conn):
            ingest_item(conn, source, item, geocoder=gazetteer, is_demo=True)
    with transaction(conn):
        for row in conn.execute("SELECT event_id FROM events WHERE is_demo=1 AND "
                                "event_time_start IS NULL").fetchall():
            set_time(conn, row["event_id"], start="2017-05-26T00:00:00Z",
                     end="2017-05-30T00:00:00Z", precision="several_days", actor="demo")
        conn.execute("UPDATE events SET is_replay=1 WHERE is_demo=1")
    report = ReportIn(category="flooding", description="ДЕМО: вода во дворах у дороги, "
                      "текст написан командой для демонстрации", lat=DEMO_POINT[1],
                      lon=DEMO_POINT[0], place_text="Batticaloa (ДЕМО)",
                      location_precision="point", idempotency_key="demo-" + secrets.token_hex(8))
    report_id, _, _ = create_report(conn, report, [], settings.photo_dir,
                                    settings.retention_days, is_demo=True)
    n = conn.execute("SELECT COUNT(*) FROM events WHERE is_demo=1").fetchone()[0]
    return json.dumps({"demo_events": n, "demo_report_pending": report_id}, ensure_ascii=False)


def remove_demo(conn, settings: Settings) -> str:
    with transaction(conn):
        events = [r[0] for r in conn.execute("SELECT event_id FROM events WHERE is_demo=1")]
        marks = ",".join("?" * len(events))
        if events:
            for table in ("evidence", "event_history", "observation_matches"):
                conn.execute(f"DELETE FROM {table} WHERE event_id IN ({marks})", events)
            conn.execute(f"DELETE FROM reviews WHERE target_kind='event' AND target_id IN ({marks})",
                         [str(e) for e in events])
        reports = [r[0] for r in conn.execute("SELECT report_id FROM citizen_reports WHERE is_demo=1")]
        for rid in reports:
            for a in conn.execute("SELECT file_name FROM attachments WHERE report_id=?", (rid,)):
                (settings.photo_dir / a[0]).unlink(missing_ok=True)
            conn.execute("DELETE FROM evidence WHERE report_id=?", (rid,))
            conn.execute("DELETE FROM attachments WHERE report_id=?", (rid,))
            conn.execute("DELETE FROM reviews WHERE target_kind='report' AND target_id=?", (rid,))
        conn.execute("UPDATE citizen_reports SET event_id=NULL WHERE is_demo=1")
        conn.execute("DELETE FROM citizen_reports WHERE is_demo=1")
        if events:
            conn.execute(f"UPDATE events SET merged_into=NULL WHERE merged_into IN ({marks})", events)
            conn.execute(f"DELETE FROM events WHERE event_id IN ({marks})", events)
        conn.execute("DELETE FROM publications WHERE source_id=?", (DEMO_SOURCE,))
    return json.dumps({"removed_events": len(events), "removed_reports": len(reports)})
