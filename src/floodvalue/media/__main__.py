"""`python -m floodvalue.media <command>`: operator token and maintenance, apart from make all."""

import argparse
import sys

from floodvalue.media import api
from floodvalue.media.store import connect, media_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m floodvalue.media")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("token", help="показать токен аналитика для входа в интерфейс")
    sub.add_parser("purge", help="удалить обращения и фото с истёкшим сроком хранения")
    args = parser.parse_args(argv)
    if args.command == "token":
        print(api.operator_token())
        print(f"хранилище: {media_dir()}", file=sys.stderr)
    elif args.command == "purge":
        with connect() as db:
            print(f"удалено обращений: {api.purge_expired(db)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
