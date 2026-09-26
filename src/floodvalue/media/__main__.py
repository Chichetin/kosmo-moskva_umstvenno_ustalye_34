"""python -m floodvalue.media <command> — separate from `make all`; touches no FloodValue output."""

from __future__ import annotations

import argparse
import ipaddress
import json
import sys
from contextlib import closing

from floodvalue.media.config import load_settings, load_sources
from floodvalue.media.store import connect, sync_sources


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m floodvalue.media",
                                     description="Мониторинг сообщений о паводках и обращения очевидцев")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init", help="создать базу и загрузить список источников")
    sub.add_parser("sources", help="состояние источников")
    ing = sub.add_parser("ingest", help="ручной запуск сбора (все включённые или один)")
    ing.add_argument("--source")
    sub.add_parser("worker", help="фоновый сбор по расписанию (Ctrl+C — остановка)")
    srv = sub.add_parser("serve", help="HTTP: /report, /media/analyst, /api/media/*")
    srv.add_argument("--host", default="127.0.0.1")
    srv.add_argument("--port", type=int, default=8765)
    sub.add_parser("purge", help="удалить обращения и фото с истёкшим сроком хранения")
    demo = sub.add_parser("demo", help="загрузить помеченные ДЕМО-сообщения (replay: сцены train/validation — Sri-Lanka, India, Nigeria, Ghana, Pakistan, Paraguay, Somalia)")
    demo.add_argument("--remove", action="store_true", help="удалить демо-данные")
    args = parser.parse_args(argv)
    settings = load_settings()

    if args.cmd == "init":
        with closing(connect(settings.db_path)) as conn:
            sync_sources(conn, load_sources(settings.sources_yaml))
        print(f"база: {settings.db_path}")
    elif args.cmd == "sources":
        with closing(connect(settings.db_path)) as conn:
            sync_sources(conn, load_sources(settings.sources_yaml))
            for r in conn.execute("SELECT source_id, kind, enabled, terms_status, last_success_at,"
                                  " last_error_at, last_error FROM sources ORDER BY source_id"):
                print(json.dumps(dict(r), ensure_ascii=False))
    elif args.cmd == "ingest":
        from floodvalue.media.worker import run_cycle
        results = run_cycle(settings, only=args.source, manual=True)
        if results is None:
            print("сбор уже идёт в другом процессе", file=sys.stderr)
            return 2
        for r in results:
            print(json.dumps(r, ensure_ascii=False))
        return 0 if all(r["status"] == "ok" for r in results) else 1
    elif args.cmd == "worker":
        from floodvalue.media.worker import serve_forever
        serve_forever(settings)
    elif args.cmd == "serve":
        import uvicorn

        from floodvalue.media.web import create_app
        try:
            local = args.host == "localhost" or ipaddress.ip_address(args.host).is_loopback
        except ValueError:
            local = False  # a hostname: treat as non-local
        if not local and not settings.analyst_token:
            print("Отказ: слушать не-локальный адрес без FLOODVALUE_MEDIA_ANALYST_TOKEN нельзя "
                  "(операторские действия остались бы без защиты).", file=sys.stderr)
            return 2
        print(f"форма очевидца: http://{args.host}:{args.port}/report\n"
              f"аналитик:       http://{args.host}:{args.port}/media/analyst")
        uvicorn.run(create_app(settings), host=args.host, port=args.port)
    elif args.cmd == "purge":
        from floodvalue.media.reports import purge_expired
        from floodvalue.media.store import transaction
        with closing(connect(settings.db_path)) as conn, transaction(conn):
            print(f"удалено обращений: {purge_expired(conn, settings.photo_dir)}")
    elif args.cmd == "demo":
        from floodvalue.media.demo import load_demo, remove_demo
        with closing(connect(settings.db_path)) as conn:
            sync_sources(conn, load_sources(settings.sources_yaml))
            print(remove_demo(conn, settings) if args.remove else load_demo(conn, settings))
    return 0


if __name__ == "__main__":
    sys.exit(main())
