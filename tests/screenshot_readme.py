#!/usr/bin/env python3
"""Capture the two README hero images, sized so that nothing is clipped.

A sibling of `screenshots.py` with a narrower job. `screenshots.py` walks both
views and composes a gallery at one fixed 1680x1010 viewport; this one captures
exactly the two frames README.md embeds - one retrieval, one map - and it does
not accept a fixed viewport. Both views are fixed-height instruments that scroll
internally (`assets/01-shell.css`), so a viewport shorter than the content does
not produce a scrollable page: it produces a *silently clipped* one. The
shipped `docs/bridge-rna-interface.png` was clipped that way, losing 410 px of
the inspector and the bottom of the retrieval network.

So both ways a frame can be cut are measured rather than assumed, and they need
different instruments.

A *panel* clipping its content is visible to the DOM: `fit_viewport` grows the
window until no scroll container reports overflow, re-measuring after each step
because a taller window changes what the panels lay out.

A *figure* running off its canvas is not visible to the DOM at all, since a
Plotly canvas is exactly as big as its container whether or not the drawing
inside it fits. `edge_ink` re-renders the figure through `Plotly.toImage` and
counts the ink in its outer band, and `frame_3d_camera` uses that to pick the
largest 3-D framing that leaves the band clean.

Both predicates are re-run after the shot and reported as the exit code, so a
frame that could not be fitted fails loudly instead of shipping cropped.

Two things it inherits from `screenshots.py` because they are obvious in the
result: the cursor is parked in the header before each frame, since Plotly
reveals its modebar on hover and a stray cursor puts a floating toolbar in the
shot; and every wait is on what the page reports about itself - glyph counts,
a populated inspector - rather than on a fixed sleep.

Named so pytest does not collect it, for the same reason `e2e_check.py` is: it
needs the real `cache/` and takes about three minutes.

Design, measurements and rejected alternatives: `docs/design-notes.md#readme-screenshots`.

    /Users/josh/Bridge-RNA/.venv/bin/python tests/screenshot_readme.py \\
        [--out docs] [--port 8072] [--headed] [--only retrieval|map]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

REPO = Path(__file__).resolve().parent.parent
PY = os.environ.get("MANIFOLD_PYTHON", sys.executable)

# The query README.md's prose describes: a mouse left-eye sample flown on
# OSD-100, whose neighbours are Earth retina studies. Changing it means
# changing that paragraph too.
STUDY = "OSD-100"
SAMPLE = "Mmus_C57-6J_EYE_FLT_Rep1_M23"

WIDTH = 1680
START_HEIGHT = 1010
MAX_HEIGHT = 2400
FIT_STEPS = 6

# Two bands, because "cut off" and "uncomfortably close" are different faults
# and only one of them is a defect. Ink in the outermost `CUT_BAND` pixels means
# the drawing continues past the boundary - a glyph or a tick numeral with its
# other half missing - and that is the hard failure. `COMFORT_BAND` is the
# margin the framing aims for so nothing is merely grazing the edge; failing it
# costs a wider camera, not a failed run.
CUT_BAND = 3
COMFORT_BAND = 14
EDGE_TOL = 30
EDGE_INK_MAX = 0.0002

# The camera is a unit direction times a distance, and only the distance is
# searched. The direction is a compositional choice and is fixed here: slightly
# above the cloud and off-square, so it reads as a volume rather than as a flat
# sheet. It is deliberately not chosen by the fill measure below - the three
# directions worth using score within 0.01% of each other, which is noise, and
# letting noise pick the angle makes the frame change shape between runs.
#
# The scatter3d default `eye` is (1.25, 1.25, 1.25): distance 2.17 along an
# equal-thirds direction. It always fits, and it always leaves the cloud small.
CAMERA_DIRECTION = (0.68, 0.68, 0.28)
CAMERA_DISTANCES = [1.60, 1.75, 1.90, 2.05, 2.20, 2.40]

# Every scroll container that is actually overflowing, in CSS pixels.
#
# Two classes of false positive have to be excluded or the fit never converges.
# `.visually-hidden` clips a 1x1 box on purpose - it is how a control that Dash
# renders without a labelable element gets an accessible name - and Dash's own
# checkbox wrappers are 1x1 for the same reason. Both report overflow forever
# and neither is visible, so anything under 40 px tall is not a panel.
OVERFLOW_JS = """() => {
  const out = [];
  for (const el of document.querySelectorAll('*')) {
    const cs = getComputedStyle(el);
    if (!/auto|scroll|hidden/.test(cs.overflowY + cs.overflowX)) continue;
    const r = el.getBoundingClientRect();
    if (r.height < 40 || r.width < 40) continue;
    const dy = el.scrollHeight - el.clientHeight;
    if (dy <= 1) continue;
    out.push({dy, tag: el.tagName.toLowerCase(), id: el.id,
              cls: (el.className || '').toString().slice(0, 60)});
  }
  return out;
}"""

COUNT_JS = """() => {
  const gd = document.querySelector('.js-plotly-plot');
  if (!gd || !gd._fullData) return null;
  let n = 0;
  for (const t of gd._fullData) n += (t.x && t.x.length) || 0;
  return n;
}"""

# The share of a figure's outermost band that carries ink rather than paper.
#
# A DOM overflow check cannot see this one. A Plotly canvas is exactly as big as
# its container whether or not the drawing inside it fits, so a 3-D camera that
# has dollied in too far reports no overflow at all while cutting the point
# cloud and the axis ticks off at the canvas edge - which is what the first
# capture of the map did. Render through `Plotly.toImage` rather than reading
# the live canvas: it re-renders the figure to a plain image, so the answer is
# about the figure and not about the floating key and badges sitting over it.
EDGE_INK_JS = """async ([selector, band, tol]) => {
  const gd = document.querySelector(selector);
  if (!gd || !gd._fullLayout) return null;
  const w = Math.round(gd.clientWidth), h = Math.round(gd.clientHeight);
  const url = await Plotly.toImage(gd, {format: 'png', width: w, height: h});
  const img = new Image();
  await new Promise((res, rej) => { img.onload = res; img.onerror = rej; img.src = url; });
  const c = document.createElement('canvas');
  c.width = img.width; c.height = img.height;
  const ctx = c.getContext('2d', {willReadFrequently: true});
  ctx.drawImage(img, 0, 0);
  const d = ctx.getImageData(0, 0, c.width, c.height).data;
  const at = (x, y) => { const i = (y * c.width + x) * 4; return [d[i], d[i+1], d[i+2]]; };
  // The paper colour, taken from the figure rather than assumed, so this works
  // on the map's navy canvas and the retrieval network's white one alike.
  const bg = at(1, 1);
  let ink = 0, total = 0, worst = null, allInk = 0;
  for (let y = 0; y < c.height; y++) {
    const edgeRow = y < band || y >= c.height - band;
    for (let x = 0; x < c.width; x++) {
      const p = at(x, y);
      const dist = Math.abs(p[0]-bg[0]) + Math.abs(p[1]-bg[1]) + Math.abs(p[2]-bg[2]);
      const inked = dist > tol;
      if (inked) allInk++;
      if (!edgeRow && x >= band && x < c.width - band) continue;
      total++;
      if (inked) { ink++; if (!worst) worst = {x, y, dist}; }
    }
  }
  // `fill` is how much of the whole canvas carries drawing rather than paper.
  // Framing maximizes it subject to the edge band staying clean, which is what
  // "as large as it goes without being cut" means as a number.
  return {ink, total, frac: ink / total, worst, w: c.width, h: c.height,
          fill: allInk / (c.width * c.height)};
}"""


def overflow_report(page) -> list[dict]:
    return page.evaluate(OVERFLOW_JS)


def describe(over: list[dict]) -> str:
    return ", ".join(f"{o['tag']}#{o['id'] or o['cls']} +{o['dy']}px" for o in over)


def fit_viewport(page, settle, height: int = START_HEIGHT) -> int:
    """Grow the window until nothing on the page is clipped; return the height.

    `settle` is called after each resize: the caller knows what "laid out" means
    for its view (a repainted Plotly canvas, a populated panel), and a fixed
    sleep would either be too short on the first pass or wasted on the last.
    """
    for step in range(FIT_STEPS):
        page.set_viewport_size({"width": WIDTH, "height": height})
        settle(page)
        over = overflow_report(page)
        if not over:
            print(f"     fits at {WIDTH}x{height} (step {step})", flush=True)
            return height
        need = max(o["dy"] for o in over)
        print(f"     {WIDTH}x{height}: clipped -> {describe(over)}", flush=True)
        # A margin over the measured shortfall, because a taller window can
        # reveal content that was not laid out at all (a panel that had
        # collapsed) and because the last few pixels are usually a border.
        height = min(MAX_HEIGHT, height + need + 24)
        if height >= MAX_HEIGHT:
            break
    page.set_viewport_size({"width": WIDTH, "height": height})
    settle(page)
    return height


def edge_ink(page, selector=".js-plotly-plot", band=CUT_BAND, tol=EDGE_TOL):
    return page.evaluate(EDGE_INK_JS, [selector, band, tol])


def set_camera(page, dist: float) -> None:
    page.evaluate(
        """([u, d]) => {
          const gd = document.querySelector('.js-plotly-plot');
          return Plotly.relayout(gd, {'scene.camera.eye':
            {x: u[0]*d, y: u[1]*d, z: u[2]*d}});
        }""", [list(CAMERA_DIRECTION), dist])


def frame_3d_camera(page):
    """Frame the 3-D scene as large as it goes without touching the canvas.

    The camera is set outright rather than dollied with the wheel: a wheel step
    is a fixed fraction of the current distance, so a loop of wheel events
    cannot ask for a particular framing and cannot be repeated after a resize -
    which is how the first capture of this frame ended up with the cloud and the
    bottom row of tick numerals running off the canvas.

    Every candidate is rendered and measured, and the winner is the one with the
    most ink on the canvas among those that leave the outer band clean. That is
    "as large as it fits" as a number rather than as a judgement, and it
    re-derives itself if the corpus, the palette or the rail ever changes.
    """
    clean, best_dirty = [], None
    for dist in CAMERA_DISTANCES:
        set_camera(page, dist)
        page.wait_for_timeout(2400)
        m = edge_ink(page, band=COMFORT_BAND)
        mark = "clean" if m["frac"] <= EDGE_INK_MAX else "TOUCHES EDGE"
        print(f"     camera d={dist:.2f}: fill {m['fill'] * 100:5.2f}%, "
              f"{m['frac'] * 100:6.3f}% ink in the outer {COMFORT_BAND} px  "
              f"{mark}", flush=True)
        if m["frac"] <= EDGE_INK_MAX:
            clean.append((m["fill"], dist))
        elif best_dirty is None or m["frac"] < best_dirty[0]:
            best_dirty = (m["frac"], dist)
    if clean:
        fill, dist = max(clean)
        print(f"     -> d={dist:.2f}, the largest of {len(clean)} clean framings "
              f"({fill * 100:.2f}% of the canvas carries ink)", flush=True)
    else:
        _, dist = best_dirty
        print(f"     -> no distance cleared the margin; taking the clearest, "
              f"d={dist:.2f}", flush=True)
    set_camera(page, dist)
    page.wait_for_timeout(3000)
    return dist


def shoot(page, path: Path) -> None:
    # Plotly reveals its modebar on hover, so a cursor left over the canvas from
    # the last interaction puts a floating toolbar in the frame.
    page.mouse.move(WIDTH // 2, 20)
    page.wait_for_timeout(900)
    page.screenshot(path=str(path))
    box = page.viewport_size
    print(f"  wrote {path.name}  {box['width']}x{box['height']} logical, "
          f"{path.stat().st_size // 1024} KB", flush=True)


def choose(page, dropdown_id: str, text: str, exact: bool = False) -> None:
    """Set a Dash 4 dropdown, tolerating the value it already holds.

    Clicking the option that is already selected leaves the Radix popover open
    rather than closing it, so the next click lands on the overlay instead of
    on the page. Skip the interaction when the trigger already reads right, and
    dismiss the popover if it survived the click.
    """
    if text in (page.locator(f"#{dropdown_id}").inner_text() or ""):
        return
    page.locator(f"#{dropdown_id}").click()
    page.wait_for_timeout(500)
    page.locator(".dash-dropdown-content").get_by_text(text, exact=exact).first.click()
    page.wait_for_timeout(1000)
    if page.locator(".dash-dropdown-content").count():
        page.keyboard.press("Escape")
    page.wait_for_timeout(1200)


def wait_points(page, minimum=1, timeout=240_000):
    page.wait_for_function(
        "m => { const gd = document.querySelector('.js-plotly-plot');"
        " if (!gd || !gd._fullData) return false;"
        " return gd._fullData.reduce((a,t)=>a+((t.x&&t.x.length)||0),0) >= m; }",
        arg=minimum, timeout=timeout)
    return page.evaluate(COUNT_JS)


def set_segment(page, group_label: str, option_text: str):
    group = page.locator(".bm-group", has=page.locator(
        f".bm-group-label:text-is('{group_label}')"))
    group.locator(
        f".bm-seg .dash-options-list-option:has-text('{option_text}')"
    ).first.click()


def set_colorby(page, text: str):
    if text in (page.locator("#color-by").inner_text() or ""):
        return
    page.locator("#color-by").click()
    page.wait_for_timeout(400)
    page.locator(".dash-dropdown-content").get_by_text(text, exact=False).first.click()
    page.wait_for_timeout(600)


def top_hit_xy(page):
    """Pixel centre of the highest-scoring ARCHS4 hit in the network."""
    return page.evaluate(
        """() => {
          const gd = document.querySelector('#network-graph .js-plotly-plot');
          if (!gd || !gd._fullData) return null;
          const t = gd._fullData[gd._fullData.length - 1];
          if (!t || !t.customdata) return null;
          for (let i = 0; i < t.customdata.length; i++) {
            if (t.customdata[i][0] !== 'gsm') continue;
            const xa = gd._fullLayout.xaxis, ya = gd._fullLayout.yaxis;
            const r = gd.getBoundingClientRect();
            return {x: r.left + xa._offset + xa.l2p(t.x[i]),
                    y: r.top + ya._offset + ya.l2p(t.y[i]),
                    label: t.customdata[i][1]};
          }
          return null;
        }""")


def capture_retrieval(page, base: str, out: Path) -> list[str]:
    print("\n== retrieve: pick the query README describes ==", flush=True)
    page.goto(f"{base}/", wait_until="load")
    page.wait_for_selector(".sample-preview", timeout=60_000)
    page.wait_for_timeout(1500)
    choose(page, "study-dropdown", STUDY, exact=True)
    choose(page, "sample-dropdown", SAMPLE)
    page.wait_for_function(
        "n => { const el = document.querySelector('.sample-preview');"
        " return el && el.innerText.includes(n); }", arg=SAMPLE, timeout=60_000)

    print("== retrieve: run the search ==", flush=True)
    t0 = time.time()
    page.locator("#search-button").click()
    page.wait_for_function(
        "() => { const gd = document.querySelector('#network-graph .js-plotly-plot');"
        " if (!gd || !gd._fullData) return false;"
        " const t = gd._fullData[gd._fullData.length - 1];"
        " return !!(t && t.customdata && t.customdata.length > 3); }",
        timeout=180_000)
    page.wait_for_timeout(1500)
    status = page.locator("#search-status").inner_text().replace("\n", " ")
    print(f"     {time.time() - t0:.1f}s -> {status[:160]}", flush=True)

    print("== retrieve: open the top hit in the inspector ==", flush=True)
    pos = top_hit_xy(page)
    if not pos:
        raise SystemExit("no ARCHS4 hit node in the network")
    print(f"     clicking {pos['label']}", flush=True)
    page.mouse.click(pos["x"], pos["y"])
    page.wait_for_function(
        "() => { const d = document.querySelector('#details-panel');"
        " return d && d.innerText.includes('GSM'); }", timeout=90_000)
    # The GEO enrichment for the opened hit arrives after the panel does, and it
    # is most of the panel's height - fitting before it lands fits the wrong
    # content. Wait for the record itself, not for a duration.
    page.wait_for_function(
        "() => { const d = document.querySelector('#details-panel');"
        " return d && /PUBLICATION|STUDY CONTEXT/.test(d.innerText); }",
        timeout=90_000)
    page.wait_for_timeout(2500)

    def settle(p):
        p.wait_for_timeout(1600)

    print("== retrieve: fit the window to the content ==", flush=True)
    fit_viewport(page, settle)
    hits = page.evaluate(
        """() => { const gd = document.querySelector('#network-graph .js-plotly-plot');
             const t = gd._fullData[gd._fullData.length - 1];
             return t.customdata.filter(c => c[0] === 'gsm').map(c => c[1]); }""")
    print(f"     hits on screen: {', '.join(hits)}", flush=True)
    shoot(page, out / "bridge-rna-interface.png")
    return overflow_report(page), edge_ink(page, "#network-graph .js-plotly-plot")


def capture_map(page, base: str, out: Path) -> list[str]:
    print("\n== map: the whole corpus, 3-D UMAP, coloured by tissue ==", flush=True)
    t0 = time.time()
    page.goto(f"{base}/map", wait_until="load")
    n = wait_points(page, minimum=900_000)
    print(f"     {n:,} glyphs in {time.time() - t0:.1f}s", flush=True)
    page.wait_for_timeout(2500)
    set_colorby(page, "Tissue")
    wait_points(page, minimum=900_000)
    page.wait_for_timeout(2000)

    def settle(p):
        p.wait_for_timeout(2500)

    print("== map: fit the window to the content ==", flush=True)
    fit_viewport(page, settle)

    print("== map: switch to 3-D and frame the cloud ==", flush=True)
    set_segment(page, "Dimensions", "3D")
    wait_points(page, minimum=1000)
    page.wait_for_timeout(7000)
    # 3-D lays out a different rail, so re-fit the window before framing the
    # camera - the canvas the camera is fitted against has to be the final one.
    fit_viewport(page, settle, height=page.viewport_size["height"])
    frame_3d_camera(page)
    badges = page.locator(".bm-plot-badges").inner_text().replace("\n", " ")
    print(f"     badges: {badges}", flush=True)
    shoot(page, out / "bridge-rna-map.png")
    return overflow_report(page), edge_ink(page)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8072)
    ap.add_argument("--out", default=str(REPO / "docs"))
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--only", choices=("retrieval", "map"),
                    help="capture one frame instead of both")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []

    server = subprocess.Popen(
        [PY, "app.py", "--port", str(args.port)], cwd=REPO,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        t0 = time.time()
        while time.time() - t0 < 180:
            line = server.stdout.readline()
            if not line:
                break
            print("    [server] " + line.rstrip(), flush=True)
            if "serving on" in line:
                break
        else:
            print("server never announced itself")
            return 1

        base = f"http://127.0.0.1:{args.port}"
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=not args.headed)
            leftover = {}
            frames = [("retrieval", capture_retrieval), ("map", capture_map)]
            for name, capture in frames:
                if args.only and name != args.only:
                    continue
                # A context each, so the map frame is a clean map rather than
                # one carrying the retrieval the previous frame left in
                # `hits-store` on the shell.
                ctx = browser.new_context(
                    viewport={"width": WIDTH, "height": START_HEIGHT},
                    device_scale_factor=2)
                page = ctx.new_page()
                page.on("pageerror", lambda e: errors.append(str(e)))
                page.on("console", lambda m: errors.append(m.text)
                        if m.type == "error" else None)
                leftover[name] = capture(page, base, out)
                ctx.close()
            browser.close()
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()

    print("\n" + "=" * 62)
    bad = False
    for name, (over, ink) in leftover.items():
        if over:
            bad = True
            print(f"FAIL {name}: a panel is clipped -> {describe(over)}")
        else:
            print(f"ok   {name}: no panel on the page is clipped")
        frac = (ink or {}).get("frac")
        if frac is None:
            bad = True
            print(f"FAIL {name}: could not measure the figure's edge band")
        elif frac > EDGE_INK_MAX:
            bad = True
            print(f"FAIL {name}: the figure runs off its canvas "
                  f"({frac * 100:.3f}% of the outer {CUT_BAND} px is ink, "
                  f"first at {ink['worst']})")
        else:
            print(f"ok   {name}: nothing in the figure reaches its canvas edge "
                  f"({frac * 100:.3f}% ink in the outer {CUT_BAND} px)")

    real = [e for e in errors
            if "favicon" not in e.lower() and "_dash-component-suites" not in e]
    if real:
        print(f"\n{len(real)} console error(s):")
        for e in real[:10]:
            print("  " + e[:200])
    else:
        print("ok   no console errors")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
