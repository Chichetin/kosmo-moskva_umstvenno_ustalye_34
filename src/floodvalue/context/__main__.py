"""`python -m floodvalue.context --chip <id> [--out <dir>]` — offline S2 context layer (D-042)."""

import argparse
import sys
from pathlib import Path

from floodvalue.context.s2 import build
from floodvalue.paths import output_dir


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="floodvalue.context")
    p.add_argument("--chip", required=True, help="chip_id, например Sri-Lanka_713926")
    p.add_argument("--out", help="каталог комплекта (по умолчанию FLOODVALUE_OUTPUT_DIR/<chip>)")
    p.add_argument("--no-download", action="store_true", help="не докачивать S2Hand, если его нет")
    p.add_argument("--allow-test", action="store_true",
                   help="test-чип: только после eval-test (метки набора размечены по этой оптике)")
    a = p.parse_args(argv)
    out = Path(a.out) if a.out else output_dir() / a.chip
    meta = build(a.chip, out, download=not a.no_download, allow_test=a.allow_test)
    o = meta["offline"]
    print(f"context: {out / o['file']} — {o['caption']}; облака (грубо): {o['cloud_fraction_rough']}; "
          f"доступен к сроку решения: {o['available_at_decision']}")
    print(f"context: {out / 'context_s2.json'}; онлайн-подложки: {', '.join(b['id'] for b in meta['online'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
