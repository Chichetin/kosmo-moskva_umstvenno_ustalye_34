"""`python -m floodvalue train`: dispatch to the main method's training by name.

Method name: env `FLOODVALUE_METHOD` (default `unet`; the CLI has no method flag). Other
methods (e.g. `pixel`) plug in by adding a line to METHODS.
"""

import importlib
import os

METHODS = {"unet": "floodvalue.model.unet.train"}
DEFAULT_METHOD = "unet"


def run(args) -> int:
    method = os.environ.get("FLOODVALUE_METHOD", DEFAULT_METHOD)
    if method not in METHODS:
        print(f"unknown method {method!r}; known: {sorted(METHODS)}")
        return 2
    return int(importlib.import_module(METHODS[method]).run(args) or 0)
