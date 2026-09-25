"""`make app` / `python -m floodvalue app`: start the operator interface (D-013).

Serves the API and the prebuilt React bundle on http://127.0.0.1:8050. If there is no
result kit in FLOODVALUE_OUTPUT_DIR yet, builds the DEMO kit first (marked as demo).
Host/port: FLOODVALUE_APP_HOST, FLOODVALUE_APP_PORT.
"""

import os

import uvicorn

from floodvalue.paths import output_dir


def run(args) -> int:
    root = output_dir()
    if not any(root.glob("*/run_metadata.json")) if root.exists() else True:
        from floodvalue.app import demo

        print("комплектов нет — собираю демо:", demo.build())
    host = os.environ.get("FLOODVALUE_APP_HOST", "127.0.0.1")
    port = int(os.environ.get("FLOODVALUE_APP_PORT", "8050"))
    print(f"FloodValue: http://{host}:{port}")
    from floodvalue.app.server import create_app

    uvicorn.run(create_app(), host=host, port=port, log_level="warning")
    return 0
