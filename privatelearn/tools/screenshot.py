"""Capture the demo player at the exact demo resolution (1280x720).

Prereqs (run once, on your machine — the agent sandbox cannot run a browser):
    pip install playwright
    playwright install chromium

Usage (server must be running first):
    python -m http.server 8000        # from privatelearn/
    python tools/screenshot.py        # writes shots/*.png

Writes:
    shots/player-initial.png   the page before any round
    shots/player-final.png     the page forced to the last round (chart, log, attack all populated)

Share the shots/ path with the agent so it can read the images and fix visual bugs.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SHOTS = os.path.join(os.path.dirname(HERE), "shots")
URL = "http://localhost:8000/web/index.html"
VIEWPORT = {"width": 1280, "height": 720}


def main():
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        sys.exit("playwright not installed: `pip install playwright && playwright install chromium`")

    os.makedirs(SHOTS, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport=VIEWPORT)

        page.goto(URL, wait_until="networkidle")
        page.wait_for_timeout(500)
        page.screenshot(path=os.path.join(SHOTS, "player-initial.png"))

        # force the final state so the chart, log and attack panel are populated
        page.evaluate(
            "document.getElementById('scrub').value = document.getElementById('scrub').max;"
            "document.getElementById('scrub').dispatchEvent(new Event('input'));"
        )
        page.wait_for_timeout(300)
        page.screenshot(path=os.path.join(SHOTS, "player-final.png"))

        browser.close()

    print("wrote:", os.path.join(SHOTS, "player-initial.png"))
    print("wrote:", os.path.join(SHOTS, "player-final.png"))


if __name__ == "__main__":
    main()
