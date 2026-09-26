"""`python -m floodvalue.media <command>`: operator token and maintenance, apart from make all."""

import argparse
import json
import sys

from floodvalue.media import api
from floodvalue.media.store import connect, media_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m floodvalue.media")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("token", help="показать токен аналитика для входа в интерфейс")
    sub.add_parser("purge", help="удалить обращения и фото с истёкшим сроком хранения")
    worker = sub.add_parser("worker", help="фоновый сбор источников по расписанию")
    worker.add_argument("--once", action="store_true", help="один цикл сейчас и выход")
    args = parser.parse_args(argv)
    if args.command == "token":
        print(api.operator_token())
        print(f"хранилище: {media_dir()}", file=sys.stderr)
    elif args.command == "worker":
        if not args.once:
            from floodvalue.media.worker import run_forever

            return run_forever()
        from floodvalue.media.ingest import run_cycle

        result = run_cycle("cli")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("status") in ("ok", "partial", "skipped") else 1
    elif args.command == "purge":
        with connect() as db:
            print(f"удалено обращений: {api.purge_expired(db)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
