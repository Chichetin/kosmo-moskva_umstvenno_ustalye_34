"""Per-checkout pytest temp root.

Several sessions and agents run pytest concurrently on one machine. pytest keeps only the
three newest <tmp>/pytest-of-<user>/pytest-N directories, so a parallel run can delete a
directory another run is still writing to (sporadic RasterioIOError). Each checkout gets
its own temp root outside the git tree (some tests need tmp_path outside a repository).
"""

import hashlib
import os
import tempfile
from pathlib import Path


def pytest_configure(config):
    checkout = Path(__file__).resolve().parent.parent
    tag = hashlib.sha256(str(checkout).encode()).hexdigest()[:12]
    root = Path(tempfile.gettempdir()) / f"floodvalue-pytest-{tag}"
    root.mkdir(exist_ok=True)
    os.environ.setdefault("PYTEST_DEBUG_TEMPROOT", str(root))
