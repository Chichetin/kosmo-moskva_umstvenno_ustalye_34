"""Replay demo, clearly marked: texts are written by the team (fictional), tied to past events of
scenes from the train and validation splits — Sri Lanka 2017 (Sri-Lanka_713926, validation), Assam
2016 (India_1018327, train), Nigeria 2018 (Nigeria_1095404, validation); is_demo=1 everywhere. It
never enters notifications, model checks or metrics, and no test chip is used. Loading twice
changes nothing."""

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
# query → (label, lon, lat): the centre of a train/validation scene of the same past event
DEMO_PLACES = {
    "batticaloa": ("Batticaloa (ДЕМО)", *DEMO_POINT),
    "batticaloa town": ("Batticaloa (ДЕМО)", *DEMO_POINT),
    "assam": ("Assam (ДЕМО)", 93.804, 26.607),  # India_1018327 (train), 2016-08-12
    "kwara": ("Kwara (ДЕМО)", 5.680, 8.762),  # Nigeria_1095404 (validation), 2018-09-21
}
# scenes the demo points sit in; none is from test (tests/media/test_site.py checks the split)
DEMO_SCENES = ("Sri-Lanka_713926", "India_1018327", "Nigeria_1095404")
SRI_LANKA = ("2017-05-27T06:00:00Z", "2017-05-26T00:00:00Z", "2017-05-30T00:00:00Z")
ASSAM = ("2016-08-11T05:00:00Z", "2016-08-08T00:00:00Z", "2016-08-12T00:00:00Z")
KWARA = ("2018-09-20T07:00:00Z", "2018-09-17T00:00:00Z", "2018-09-21T00:00:00Z")
DEMO_ITEMS = [  # (external id, url, title, (published, event start, event end))
    ("demo-1", "https://demo.invalid/a1", "ДЕМО: Floods hit Batticaloa, hundreds evacuated",
     SRI_LANKA),
    ("demo-2", "https://demo.invalid/b7", "ДЕМО: Floods hit Batticaloa, hundreds evacuated",
     SRI_LANKA),
    ("demo-3", "https://demo.invalid/c2",
     "ДЕМО: Reports of flooding in Batticaloa town denied by officials", SRI_LANKA),
    ("demo-4", "https://demo.invalid/d9",
     "ДЕМО: Heavy rain expected to cause flooding in Batticaloa next week", SRI_LANKA),
    ("demo-5", "https://demo.invalid/e4",
     "ДЕМО: Floods hit Assam, villages submerged along the Brahmaputra", ASSAM),
    ("demo-6", "https://demo.invalid/f3",
     "ДЕМО: Floods hit Kwara, farmland submerged along the Niger river", KWARA),
]


class DemoGazetteer:
    """A small gazetteer for the replay only; real runs use Nominatim or nothing."""

    def lookup(self, query: str) -> list[dict]:
        found = DEMO_PLACES.get(query.casefold())
        if found is None:
            return []
        label, lon, lat = found
        return [{"label": label, "lon": lon, "lat": lat, "precision": "locality", "bbox": None}]


def load_demo(conn, settings: Settings) -> str:
    conn.execute(
        """INSERT OR IGNORE INTO sources(source_id, name, kind, adapter, enabled, terms_status,
             attribution, content_storage_policy, poll_interval_s)
           VALUES(?, 'ДЕМО: вымышленные публикации (replay)', 'news', 'none', 0, 'team_written',
                  'Тексты написаны командой, не реальные публикации', 'headline_only', 0)""",
        (DEMO_SOURCE,))
    source = conn.execute("SELECT * FROM sources WHERE source_id=?", (DEMO_SOURCE,)).fetchone()
    gazetteer = DemoGazetteer()
    for i, (ext, url, title, (published, start, end)) in enumerate(DEMO_ITEMS):
        item = {"external_id": ext, "url": url, "title": title, "text_excerpt": None,
                "language": "en", "source_domain": f"demo{i % 2}.invalid",
                "published_at": published, "source_updated_at": None,
                "structured": None}
        with transaction(conn):
            ingest_item(conn, source, item, geocoder=gazetteer, is_demo=True)
            for row in conn.execute(
                    "SELECT DISTINCT ev.event_id FROM publications p JOIN evidence x "
                    "USING(publication_id) JOIN events ev USING(event_id) WHERE p.source_id=? "
                    "AND p.external_id=? AND ev.is_demo=1 AND ev.event_time_start IS NULL",
                    (DEMO_SOURCE, ext)).fetchall():
                set_time(conn, row["event_id"], start=start, end=end,
                         precision="several_days", actor="demo")
    with transaction(conn):
        conn.execute("UPDATE events SET is_replay=1 WHERE is_demo=1")
    report_id = None
    if conn.execute("SELECT 1 FROM citizen_reports WHERE is_demo=1").fetchone() is None:
        report = ReportIn(category="flooding", description="ДЕМО: вода во дворах у дороги, "
                          "текст написан командой для демонстрации", lat=DEMO_POINT[1],
                          lon=DEMO_POINT[0], place_text="Batticaloa (ДЕМО)",
                          location_precision="point",
                          idempotency_key="demo-" + secrets.token_hex(8))
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
