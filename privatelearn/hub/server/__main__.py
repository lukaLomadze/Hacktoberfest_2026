"""python -m hub.server [--port 8100] [--db hub/hub.db] [--seed dev|demo|none]   (run from privatelearn/)"""

from __future__ import annotations

import argparse
import logging
import os
import sys


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):  # Windows consoles default to cp1252
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="PrivateLearn platform server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--db", default=None, help="database file (default hub/hub.db or $HUB_DB)")
    ap.add_argument("--seed", default=None, choices=("dev", "demo", "none"),
                    help="seed an empty database (default $HUB_SEED or dev)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    import uvicorn

    from .app import create_app

    app = create_app(args.db, seed_mode=args.seed or os.environ.get("HUB_SEED"))
    print(f"PrivateLearn platform on http://{args.host}:{args.port}  (API /api/v1, docs /api/docs)")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
