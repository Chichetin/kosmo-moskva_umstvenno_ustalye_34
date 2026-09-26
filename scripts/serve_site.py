"""One origin for the operator UI, the eyewitness form and media monitoring (#174).

    /                     operator React UI and its read-only API (floodvalue.app.run)
    /report, /report/...  public eyewitness form and status page     (floodvalue.media.web)
    /media/...            analyst page and media static files
    /api/media/...        media API, incl. /api/media/map for the operator's «Сообщения» layer

Lives outside src/floodvalue on purpose: no FloodValue stage may import floodvalue.media
(tests/media/test_media_firewall.py), so the two FastAPI apps are joined by path here. One port,
one TLS certificate, no CORS; the analyst session cookie is shared with the operator UI.

    uv run --frozen python scripts/serve_site.py [--host 127.0.0.1] [--port 8000] [--out DIR]

Collection runs separately: `python -m floodvalue.media worker` (deploy/README.md).
"""

from __future__ import annotations

import argparse
import ipaddress
import os
import sys

MEDIA_EXACT = ("/report", "/media", "/api/media")
MEDIA_PREFIXES = ("/report/", "/media/", "/api/media/")


def is_media(path: str) -> bool:
    return path in MEDIA_EXACT or path.startswith(MEDIA_PREFIXES)


def build(settings=None):
    """ASGI app dispatching by path. Lifespan goes to the operator app (media has no hooks)."""
    from floodvalue.app.run import app as operator_app
    from floodvalue.media.web import create_app

    media_app = create_app(settings)

    async def site(scope, receive, send):
        if scope["type"] in ("http", "websocket") and is_media(scope["path"]):
            await media_app(scope, receive, send)
        else:
            await operator_app(scope, receive, send)

    return site


def public_problem(host: str, env=os.environ) -> str | None:
    """A public site without an analyst token would leave moderation to anyone: refuse."""
    try:
        local = host == "localhost" or ipaddress.ip_address(host).is_loopback
    except ValueError:
        local = False
    public = not local or env.get("FLOODVALUE_SITE_PUBLIC") == "1"
    if public and not env.get("FLOODVALUE_MEDIA_ANALYST_TOKEN"):
        return ("Отказ: публичный сайт без FLOODVALUE_MEDIA_ANALYST_TOKEN — модерация и действия "
                "аналитика остались бы без защиты.")
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FloodValue: оператор + форма очевидца + СМИ")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.environ.get("FLOODVALUE_APP_PORT", "8000")))
    parser.add_argument("--out", help="каталог комплекта для операторского интерфейса")
    args = parser.parse_args(argv)
    problem = public_problem(args.host)
    if problem:
        print(problem, file=sys.stderr)
        return 2
    if args.out:
        os.environ["FLOODVALUE_APP_RUN_DIR"] = args.out
    import uvicorn

    print(f"оператор:   http://{args.host}:{args.port}/\n"
          f"очевидцы:   http://{args.host}:{args.port}/report\n"
          f"аналитик:   http://{args.host}:{args.port}/media/analyst")
    # Behind a reverse proxy on the same host: X-Forwarded-For/Proto from 127.0.0.1 are trusted,
    # so rate limits see the real client and the session cookie gets Secure over https.
    uvicorn.run(build(), host=args.host, port=args.port, proxy_headers=True,
                forwarded_allow_ips="127.0.0.1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
