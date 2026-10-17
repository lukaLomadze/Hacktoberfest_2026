"""One command: reproduce the run, then replay it in the browser.

    python demo.py            # rebuild data + trace.json, serve, open the player
    python demo.py --no-run   # just serve and replay the committed trace.json

Needs only Python 3.9+ and numpy.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import pathlib
import subprocess
import sys
import threading
import webbrowser

ROOT = pathlib.Path(__file__).resolve().parent


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-run", action="store_true", help="skip regeneration; replay the committed trace")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args(argv)

    if not args.no_run:
        for script in ("src/pipeline/prep_data.py", "src/pipeline/run_loop.py"):
            print(f"$ python {script}")
            subprocess.run([sys.executable, str(ROOT / script)], check=True, cwd=ROOT)

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(ROOT))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    url = f"http://localhost:{args.port}/web/index.html"
    print(f"\nreplaying trace.json at {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
