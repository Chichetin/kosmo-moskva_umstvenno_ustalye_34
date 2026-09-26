"""FloodValue: Sentinel-1 water map → expected loss of 10 assets → EO procurement plan A/B/C."""

import sys

__version__ = "0.1.0"


def utf8_console() -> None:
    """Switch stdout/stderr to UTF-8 when they use another encoding (cp1251 on Windows).

    Every CLI prints Cyrillic and symbols like «Δ»; a Windows console or pipe in cp1251
    would raise UnicodeEncodeError mid-run (#158). Streams without `reconfigure`
    (pytest capture, custom wrappers) are left as they are.
    """
    for stream in (sys.stdout, sys.stderr):
        encoding = (getattr(stream, "encoding", None) or "").lower().replace("-", "")
        if encoding != "utf8" and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


# Runs on import of any floodvalue module, so `python -m floodvalue.<module>` is covered too.
utf8_console()
