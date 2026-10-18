"""One command: reproduce the run (with the LLM), then replay it in the browser.

    python demo.py            # rebuild the hospital data + LLM trace, serve, open the player
    python demo.py --banks    # same, but the bank dataset (data/banks)
    python demo.py --no-run   # just serve and replay the committed trace.json

Regeneration runs with --llm and needs DEEPSEEK_API_KEY; without the key it replays the
committed trace instead. `--no-run` always replays the committed trace. Needs only
Python 3.9+ and numpy.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import os
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
    ap.add_argument("--banks", action="store_true",
                    help="run and replay the bank dataset (data/banks) instead of the hospitals")
    args = ap.parse_args(argv)

    if args.banks:
        steps = [("src/pipeline/prep_banks.py", ()), ("src/pipeline/run_banks.py", ("--llm",))]
        trace_rel = "data/banks/trace.json"
    else:
        steps = [("src/pipeline/prep_data.py", ()), ("src/pipeline/run_loop.py", ("--llm",))]
        trace_rel = "trace.json"

    if not args.no_run and not os.environ.get("DEEPSEEK_API_KEY"):
        # regeneration needs the LLM to reconcile differently-named columns; without the
        # key it would fall back to an empty contract, so replay the committed trace instead.
        print("DEEPSEEK_API_KEY is not set — replaying the committed trace instead of regenerating.")
        print('Set $env:DEEPSEEK_API_KEY = "sk-..." and rerun to regenerate (or pass --no-run).')
    elif not args.no_run:
        for script, extra in steps:
            cmd = [sys.executable, str(ROOT / script), *extra]
            print(f"$ python {script} {' '.join(extra)}".rstrip())
            subprocess.run(cmd, check=True, cwd=ROOT)

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(ROOT))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    url = f"http://localhost:{args.port}/web/index.html"
    if args.banks:
        # the player loads ../trace.json by default; point it at the bank trace instead
        url += f"?trace=../{trace_rel}"
    print(f"\nreplaying {trace_rel} at {url}  (Ctrl+C to stop)")
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
