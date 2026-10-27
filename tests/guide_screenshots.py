#!/usr/bin/env python3
"""Capture the guided tour's screenshots from the running app.

Every slide in `bridge_rna.guide.SLIDES` shows a real frame of the app on the
real corpus, made here and nowhere else, so after an interface change the whole
set is one command away from current:

    /Users/josh/Bridge-RNA/.venv/bin/python tests/guide_screenshots.py \\
        [--port 8074] [--headed] [--only retrieve|map]

It starts its own server, walks the two flows the guide describes, and for each
slide spotlights the feature the slide is about: everything else is dimmed and
the feature is ringed in NASA blue. The spotlight is one fixed-position element
on <body>, injected only here, so no scroll container can clip its ring and
nothing about it ships.

Each frame is a 960x600 or 1440x900 CSS-pixel region around the feature,
encoded by Chrome itself as WebP at exactly `guide.SHOT_WIDTH` x
`guide.SHOT_HEIGHT` pixels: a close region keeps the interface's text at its
real size in the dialog, and the whole viewport is used only where the point
is the whole picture. `docs/guide-screenshots.json` records the commit, date
and viewport they were taken from.

Like the other browser scripts it is named so pytest does not collect it: it
needs the real `cache/` and the 963 MB memmap, and takes a few minutes.
Design: docs/help-and-guide.md.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from playwright.sync_api import sync_playwright

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from bridge_rna import guide  # noqa: E402

PY = os.environ.get("MANIFOLD_PYTHON", sys.executable)
OUT = REPO / "assets" / "guide"
RECORD = REPO / "docs" / "guide-screenshots.json"

VIEWPORT = {"width": 1440, "height": 900}
DSF = 1.5
CLOSE = (960, 600)     # a region at the interface's own size
WHOLE = (1440, 900)    # the whole viewport, for slides about the whole picture
QUALITY = 82

# The query README describes: a mouse left-eye sample flown on OSD-100.
STUDY = "OSD-100"
SAMPLE = "Mmus_C57-6J_EYE_FLT_Rep1_M23"
MATCHES = "30"
FIND = "OSD-100"

SPOT_JS = """(selectors) => {
  document.querySelectorAll('.guide-capture-spot').forEach(n => n.remove());
  const rects = selectors.flatMap(s => [...document.querySelectorAll(s)])
    .map(n => n.getBoundingClientRect()).filter(r => r.width && r.height);
  if (!rects.length) return null;
  const pad = 6;
  const box = {
    left: Math.min(...rects.map(r => r.left)) - pad,
    top: Math.min(...rects.map(r => r.top)) - pad,
    right: Math.max(...rects.map(r => r.right)) + pad,
    bottom: Math.max(...rects.map(r => r.bottom)) + pad,
  };
  const spot = document.createElement('div');
  spot.className = 'guide-capture-spot';
  Object.assign(spot.style, {
    position: 'fixed', zIndex: 2147483647, pointerEvents: 'none',
    left: box.left + 'px', top: box.top + 'px',
    width: (box.right - box.left) + 'px', height: (box.bottom - box.top) + 'px',
    borderRadius: '6px', outline: '3px solid #1c67e3', outlineOffset: '0',
    boxShadow: '0 0 0 4px rgba(28, 103, 227, 0.22), 0 0 0 100vmax rgba(14, 14, 16, 0.30)',
  });
  document.body.appendChild(spot);
  return box;
}"""

RESULTS_READY_JS = """() => {
  const rows = document.querySelectorAll('#results-table-wrap tr.result-row');
  const gd = document.querySelector('#results-plot .js-plotly-plot');
  if (!rows.length || !gd || !gd._fullData) return false;
  return gd._fullData.some(t => t.meta && t.meta.role === 'in_table'
                                && t.x && t.x.length === rows.length);
}"""

MAP_READY_JS = """(minimum) => {
  const sk = document.getElementById('map-skeleton');
  if (sk && !sk.classList.contains('is-done')) return false;
  const gd = document.querySelector('#manifold-graph .js-plotly-plot')
          || document.querySelector('.js-plotly-plot');
  if (!gd || !gd._fullData) return false;
  return gd._fullData.reduce((a, t) => a + ((t.x && t.x.length) || 0), 0) >= minimum;
}"""

# The pixel position of the first OSDR diamond on the 2-D map, from Plotly's
# own axis transforms: WebGL points are not DOM nodes, so this is the only
# honest way to click one.
DIAMOND_JS = """() => {
  const gd = document.querySelector('#manifold-graph .js-plotly-plot')
          || document.querySelector('.js-plotly-plot');
  const fl = gd._fullLayout, xa = fl.xaxis, ya = fl.yaxis;
  const box = gd.getBoundingClientRect();
  for (const t of gd._fullData) {
    const sym = t.marker && t.marker.symbol;
    const isDiamond = (Array.isArray(sym) ? sym[0] : sym || '').toString().startsWith('diamond');
    if (!isDiamond || !t.x || !t.x.length || t.visible === 'legendonly') continue;
    for (let i = 0; i < t.x.length; i += 37) {
      const px = box.left + xa._offset + xa.l2p(t.x[i]);
      const py = box.top + ya._offset + ya.l2p(t.y[i]);
      if (px > box.left + 80 && px < box.right - 260 && py > box.top + 60 && py < box.bottom - 60)
        return [px, py];
    }
  }
  return null;
}"""


def free_port(port: int) -> bool:
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def start_server(port: int, log: Path) -> subprocess.Popen:
    if not free_port(port):
        raise SystemExit(f"port {port} is in use; pass --port")
    handle = log.open("w")
    server = subprocess.Popen([PY, "app.py", "--port", str(port)], cwd=REPO,
                              stdout=handle, stderr=subprocess.STDOUT)
    t0 = time.time()
    while time.time() - t0 < 180:
        if not free_port(port):
            return server
        if server.poll() is not None:
            raise SystemExit(f"server exited; see {log}")
        time.sleep(0.5)
    server.kill()
    raise SystemExit(f"server never listened on {port}; see {log}")


class Shooter:
    def __init__(self, page):
        self.page = page
        self.cdp = page.context.new_cdp_session(page)
        self.written: list[str] = []

    def shot(self, key: str, selectors: list[str], size=CLOSE) -> None:
        page = self.page
        # A field left focused shows its caret and ring in the frame.
        page.evaluate("() => document.activeElement && document.activeElement.blur()")
        # Bring the feature into view: the panels scroll internally, so a
        # feature below the fold is otherwise photographed as nothing.
        page.evaluate("""(s) => { const n = document.querySelector(s); if (!n) return;
          const r = n.getBoundingClientRect();
          if (r.top < 0 || r.bottom > innerHeight)
            n.scrollIntoView({block: r.height > innerHeight * 0.66 ? 'start' : 'center'}); }""",
                      selectors[0])
        page.wait_for_timeout(300)
        # Plotly shows its modebar under a resting cursor; park it on the header.
        page.mouse.move(700, 18)
        page.wait_for_timeout(700)
        box = page.evaluate(SPOT_JS, selectors)
        if box is None:
            raise SystemExit(f"{key}: nothing to spotlight for {selectors}")
        page.wait_for_timeout(250)
        w, h = size
        vw, vh = VIEWPORT["width"], VIEWPORT["height"]
        cx = (box["left"] + box["right"]) / 2
        cy = (box["top"] + box["bottom"]) / 2
        # Centre the feature; one larger than the frame is anchored at its
        # top-left instead, so its heading is what the frame keeps.
        bw, bh = box["right"] - box["left"], box["bottom"] - box["top"]
        x = box["left"] - 12 if bw > w - 24 else cx - w / 2
        y = box["top"] - 12 if bh > h - 24 else cy - h / 2
        x = min(max(0, x), vw - w)
        y = min(max(0, y), vh - h)
        data = self.cdp.send("Page.captureScreenshot", {
            "format": "webp", "quality": QUALITY,
            "clip": {"x": x, "y": y, "width": w, "height": h,
                     # `scale` is relative to CSS pixels and ignores the
                     # device scale factor; the page is rendered at DSF, so
                     # upscaling the region to the shot size stays sharp.
                     "scale": guide.SHOT_WIDTH / w},
        })
        page.evaluate("() => document.querySelectorAll('.guide-capture-spot')"
                      ".forEach(n => n.remove())")
        path = OUT / f"{key}.webp"
        path.write_bytes(base64.b64decode(data["data"]))
        self.written.append(key)
        print(f"  {key:<28} {path.stat().st_size // 1024:>4} KB", flush=True)


def choose(page, dropdown_id: str, text: str, exact: bool = False) -> None:
    """Set a Dash 4 dropdown, tolerating the value it already holds."""
    if text in (page.locator(f"#{dropdown_id}").inner_text() or ""):
        return
    page.locator(f"#{dropdown_id}").click()
    page.wait_for_timeout(500)
    page.locator(".dash-dropdown-content").get_by_text(text, exact=exact).first.click()
    page.wait_for_timeout(800)
    if page.locator(".dash-dropdown-content").count():
        page.keyboard.press("Escape")
    page.wait_for_timeout(800)


def first_option(page, dropdown_id: str) -> None:
    page.locator(f"#{dropdown_id}").click()
    page.wait_for_timeout(500)
    page.locator(".dash-dropdown-content [role='option']").first.click()
    page.wait_for_timeout(800)
    if page.locator(".dash-dropdown-content").count():
        page.keyboard.press("Escape")
    page.wait_for_timeout(800)


def set_segment(page, group_label: str, option_text: str) -> None:
    group = page.locator(".bm-group", has=page.locator(
        f".bm-group-label:text-is('{group_label}')"))
    group.locator(
        f".bm-seg .dash-options-list-option:has-text('{option_text}')").first.click()


def set_colorby(page, text: str) -> None:
    if text in (page.locator("#color-by").inner_text() or ""):
        return
    page.locator("#color-by").click()
    page.wait_for_timeout(400)
    page.locator(".dash-dropdown-content").get_by_text(text, exact=False).first.click()
    page.wait_for_timeout(600)


def wait_map(page, minimum=1, timeout=240_000) -> None:
    page.wait_for_function(MAP_READY_JS, arg=minimum, timeout=timeout)
    page.wait_for_timeout(1800)


def capture_retrieve(s: Shooter, base: str) -> None:
    page = s.page
    print("\n== Sample to map ==", flush=True)
    page.goto(f"{base}/", wait_until="load")
    page.wait_for_selector(".sample-preview", timeout=60_000)
    page.wait_for_timeout(1200)
    choose(page, "study-dropdown", STUDY, exact=True)
    choose(page, "sample-dropdown", SAMPLE)
    page.wait_for_function(
        "n => { const el = document.querySelector('.sample-preview');"
        " return el && el.innerText.includes(n); }", arg=SAMPLE, timeout=60_000)
    s.shot("retrieve-01-sample", ["#study-group", "#mode-panel-sample"])

    page.locator("#topk-input input, input#topk-input").first.fill(MATCHES)
    page.keyboard.press("Enter")
    page.wait_for_timeout(800)
    s.shot("retrieve-02-search", [".topk-control", "#search-button"])

    page.locator("#search-button").click()
    page.wait_for_function(RESULTS_READY_JS, timeout=180_000)
    page.wait_for_timeout(1500)
    s.shot("retrieve-03-table", ["#results-table-wrap"])
    s.shot("retrieve-04-filter", ["#results-filters"])
    s.shot("retrieve-10-export", ["#results-download-button"])

    top = page.locator("#results-table-wrap tr.result-row").first
    top.locator("td.col-title").click()
    page.wait_for_function(
        "() => { const d = document.querySelector('#details-panel');"
        " return d && d.innerText.includes('GSM'); }", timeout=90_000)
    page.wait_for_function(
        "() => { const d = document.querySelector('#details-panel');"
        " return d && /^(publication|study context)$/im.test(d.innerText); }",
        timeout=90_000)
    page.wait_for_timeout(1500)
    s.shot("retrieve-05-inspect", ["#details-panel"])
    s.shot("retrieve-06-geo", ["#details-panel a[href*='geo/query']"])

    page.locator("#results-plot-kind label:has-text('Series')").first.click()
    page.wait_for_timeout(2500)
    s.shot("retrieve-07-plots", ["#plot-panel"])
    page.locator("#results-plot-kind label:has-text('Score by rank')").first.click()
    page.wait_for_timeout(1500)

    page.locator("#mode-tab-cohort").click()
    page.wait_for_timeout(2500)
    page.wait_for_function(
        "() => { const c = document.querySelector('#cohort-card');"
        " return c && c.innerText.trim().length > 0; }", timeout=60_000)
    first_option(page, "cohort-compare-dropdown")
    page.wait_for_timeout(1500)
    s.shot("retrieve-08-cohort", ["#mode-panel-cohort"])

    page.locator("#mode-tab-upload").click()
    page.wait_for_timeout(1200)
    s.shot("retrieve-09-upload", ["#mode-panel-upload"])
    page.locator("#mode-tab-sample").click()
    page.wait_for_timeout(1200)

    page.locator(".app-nav-item:has-text('Map')").click()
    page.wait_for_selector("#retrieval-group", state="visible", timeout=240_000)
    wait_map(page, minimum=100_000)
    s.shot("retrieve-11-map", ["#retrieval-group"], size=WHOLE)

    page.locator("#frame-retrieval").click()
    page.wait_for_selector("#neighborhood-drawer", state="visible", timeout=60_000)
    page.wait_for_timeout(3500)
    s.shot("retrieve-12-neighborhood", ["#neighborhood-drawer"], size=WHOLE)


def capture_map(s: Shooter, base: str) -> None:
    page = s.page
    print("\n== Explore the map ==", flush=True)
    page.goto(f"{base}/map", wait_until="load")
    wait_map(page, minimum=100_000)
    set_colorby(page, "Tissue")
    wait_map(page, minimum=100_000)
    s.shot("map-01-overview", [".bm-plot-wrap, #manifold-graph"], size=WHOLE)
    s.shot("map-02-projection", ["[aria-labelledby='projection-label']"])

    set_colorby(page, "Species")
    wait_map(page, minimum=100_000)
    s.shot("map-04-color", ["[aria-labelledby='color-by-label']"], size=WHOLE)
    set_colorby(page, "Tissue")
    wait_map(page, minimum=100_000)
    s.shot("map-05-points", ["[aria-labelledby='budget-label']"])

    set_segment(page, "Dimensions", "3D")
    wait_map(page, minimum=1000)
    page.wait_for_timeout(6000)
    s.shot("map-03-3d", ["[aria-labelledby='dims-label']"], size=WHOLE)
    set_segment(page, "Dimensions", "2D")
    wait_map(page, minimum=100_000)

    point = page.evaluate(DIAMOND_JS)
    if point is None:
        raise SystemExit("map-07-select: no OSDR diamond on screen to click")
    page.mouse.click(*point)
    page.wait_for_function(
        "() => [...document.querySelectorAll('.bm-group-label')]"
        ".some(n => n.textContent.trim() === 'Selected sample')", timeout=60_000)
    page.wait_for_timeout(1200)
    s.shot("map-07-select",
           ["#selected-panel, .bm-picked-panel, :has(> .bm-picked)"], size=WHOLE)

    page.locator("#find-input").fill(FIND)
    page.keyboard.press("Enter")
    page.wait_for_function(
        "() => { const b = document.querySelector('#frame-find');"
        " return b && b.offsetParent !== null; }", timeout=60_000)
    page.wait_for_timeout(2500)
    s.shot("map-06-find", ["#find-group"], size=WHOLE)

    page.locator("#frame-find").click()
    page.wait_for_selector("#view-group", state="visible", timeout=60_000)
    page.wait_for_timeout(3000)
    s.shot("map-08-reset", ["#view-group"])



def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8074)
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--only", choices=("retrieve", "map"))
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    log = Path(os.environ.get("TMPDIR", "/tmp")) / "guide-screenshots-server.log"
    server = start_server(args.port, log)
    errors: list[str] = []
    written: list[str] = []
    try:
        base = f"http://127.0.0.1:{args.port}"
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=not args.headed)
            for name, capture in (("retrieve", capture_retrieve), ("map", capture_map)):
                if args.only and name != args.only:
                    continue
                # A context per chapter: the map chapter starts from a clean
                # map, not one carrying the retrieval the first left behind.
                ctx = browser.new_context(viewport=VIEWPORT, device_scale_factor=DSF)
                # The tour opens itself on a first visit; not while it is
                # being photographed.
                ctx.add_init_script(
                    "try { localStorage.setItem('bridge-rna-guide-seen', 'capture') }"
                    " catch (e) {}")
                page = ctx.new_page()
                page.on("pageerror", lambda e: errors.append(str(e)))
                shooter = Shooter(page)
                capture(shooter, base)
                written += shooter.written
                ctx.close()
            browser.close()
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()

    expected = {s.key for s in guide.SLIDES
                if not args.only or s.key.startswith(args.only)}
    missing = sorted(expected - set(written))
    revision = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                              capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=REPO,
                                capture_output=True, text=True).stdout.strip())
    record = json.loads(RECORD.read_text()) if RECORD.exists() else {}
    record.update({
        "captured_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "revision": revision + ("+uncommitted" if dirty else ""),
        "viewport": VIEWPORT, "device_scale_factor": DSF,
        "image": [guide.SHOT_WIDTH, guide.SHOT_HEIGHT], "webp_quality": QUALITY,
        "query": {"study": STUDY, "sample": SAMPLE, "matches": int(MATCHES)},
    })
    RECORD.write_text(json.dumps(record, indent=2) + "\n")
    print(f"\nwrote {len(written)} images; record in {RECORD.relative_to(REPO)}")
    if errors:
        print("page errors:\n  " + "\n  ".join(errors[:10]))
    if missing:
        print("MISSING: " + ", ".join(missing))
    return 1 if (missing or errors) else 0


if __name__ == "__main__":
    sys.exit(main())
