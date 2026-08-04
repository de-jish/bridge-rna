#!/usr/bin/env python3
"""Drive the real Bridge RNA app and capture the presentation screenshots.

A sibling of `e2e_check.py` rather than a test: same harness, opposite purpose.
`e2e_check.py` asserts and returns an exit code; this one composes. It boots
app.py against the real `cache/` and walks both views the way a user does,
capturing a retina-scale PNG (1680x1010 at 2x) wherever the app is showing
something worth showing.

Two independent retrieval walkthroughs (a spaceflight mouse liver, OSD-137, and
a spaceflight mouse kidney, OSD-102), each: pick a study and sample, run the
search, open a hit in the inspector, then jump to the map and frame the
viewport on the query and its neighbours so the retrieved network and its
position in the whole corpus are shown side by side. The liver query also
widens to top-20 before that. Then a clean map pass: default UMAP view coloured
by tissue and by species, the OSDR-only coverage case, a zoomed neighbourhood,
a hover card, and all three projections (PCA, UMAP, t-SNE) in both 2-D and
3-D.

Named so pytest does not collect it, for the same reason `e2e_check.py` is: it
needs the real artifacts and takes about ten minutes.

    /Users/josh/Bridge-RNA/.venv/bin/python tests/screenshots.py \\
        [--out screenshots] [--port 8071] [--headed]

Two things it does that are easy to leave out and obvious in the result. It
parks the cursor in the header before each frame, because Plotly reveals its
modebar on hover and a stray cursor puts a floating toolbar in the shot. And it
waits on what the page reports about itself - glyph counts, populated panels -
rather than on fixed sleeps, so a slow render produces a late screenshot rather
than a blank one.
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

# Two cached-tier queries, deliberately different studies and tissues so the
# two retrieval walkthroughs read as distinct examples rather than the same
# shot twice: a 39-day spaceflight mouse liver, and a flight mouse kidney.
QUERIES = [
    {"study": "OSD-137", "sample": "Mmus_BAL-TAL_LVR_FLT_Rep1_F1",
     "slug": "liver", "label": "OSD-137 spaceflight mouse liver"},
    {"study": "OSD-102", "sample": "Mmus_C57-6J_KDN_FLT_Rep1_M23",
     "slug": "kidney", "label": "OSD-102 spaceflight mouse kidney"},
]

COUNT_JS = """() => {
  const gd = document.querySelector('.js-plotly-plot');
  if (!gd || !gd._fullData) return null;
  let n = 0;
  for (const t of gd._fullData) n += (t.x && t.x.length) || 0;
  return n;
}"""


class Shooter:
    def __init__(self, out: Path):
        self.out = out
        self.n = 0
        self.taken: list[str] = []

    def shot(self, page, name: str, park: bool = True) -> None:
        # Plotly reveals its modebar on hover, so a cursor left over the canvas
        # from the last interaction puts a floating toolbar in the frame. Park
        # it in the header first unless the shot is deliberately hovering.
        if park:
            page.mouse.move(830, 20)
            page.wait_for_timeout(700)
        self.n += 1
        path = self.out / f"{self.n:02d}-{name}.png"
        page.screenshot(path=str(path))
        kb = path.stat().st_size // 1024
        print(f"  shot {path.name}  ({kb} KB)", flush=True)
        self.taken.append(path.name)


def wait_points(page, timeout=240_000, minimum=1):
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
    page.locator("#color-by").click()
    page.locator(".dash-dropdown-content").get_by_text(
        text, exact=False).first.click()
    page.wait_for_timeout(400)


def node_xy(page, kind: str, which: int = 0):
    """Pixel centre of the `which`-th network node of `kind` (query/gsm/gse).

    Plotly's axis objects expose the data->pixel transform, so this clicks the
    node a user would click rather than dispatching a synthetic event.
    """
    return page.evaluate(
        """([kind, which]) => {
          const gd = document.querySelector('#network-graph .js-plotly-plot');
          if (!gd || !gd._fullData) return null;
          const t = gd._fullData[gd._fullData.length - 1];
          if (!t || !t.customdata) return null;
          let seen = -1;
          for (let i = 0; i < t.customdata.length; i++) {
            if (t.customdata[i][0] !== kind) continue;
            if (++seen !== which) continue;
            const xa = gd._fullLayout.xaxis, ya = gd._fullLayout.yaxis;
            const r = gd.getBoundingClientRect();
            return {x: r.left + xa._offset + xa.l2p(t.x[i]),
                    y: r.top + ya._offset + ya.l2p(t.y[i]),
                    label: t.customdata[i][1]};
          }
          return null;
        }""", [kind, which])


def osdr_glyph_xy(page):
    """Pixel centre of an OSDR glyph currently on screen, if there is one.

    The ARCHS4 cloud carries no customdata - it was stripped as dead payload -
    so a trace that has customdata is the OSDR overlay, which is exactly the
    layer whose hover card is worth showing.
    """
    return page.evaluate(
        """() => {
          const gd = document.querySelector('.js-plotly-plot');
          if (!gd || !gd._fullData) return null;
          const xa = gd._fullLayout.xaxis, ya = gd._fullLayout.yaxis;
          if (!xa || !ya) return null;
          const r = gd.getBoundingClientRect();
          for (const t of gd._fullData) {
            if (!t.customdata || !t.x || !t.x.length) continue;
            for (let i = 0; i < t.x.length; i++) {
              const px = r.left + xa._offset + xa.l2p(t.x[i]);
              const py = r.top + ya._offset + ya.l2p(t.y[i]);
              if (px > r.left + 160 && px < r.right - 580
                  && py > r.top + 160 && py < r.bottom - 160) {
                return {x: px, y: py, key: t.customdata[i][0]};
              }
            }
          }
          return null;
        }""")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8071)
    ap.add_argument("--out", default=str(REPO / "screenshots"))
    ap.add_argument("--headed", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    s = Shooter(out)
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
            ctx = browser.new_context(viewport={"width": 1680, "height": 1010},
                                      device_scale_factor=2)
            page = ctx.new_page()
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.on("console", lambda m: errors.append(m.text)
                    if m.type == "error" else None)

            # ---------------- Retrieve: two independent examples ----------------
            for qi, q in enumerate(QUERIES):
                print(f"\n== retrieve[{q['slug']}]: pick a study and a sample ==",
                      flush=True)
                page.goto(f"{base}/", wait_until="load")
                page.wait_for_selector(".sample-preview", timeout=60_000)
                page.wait_for_timeout(1500)

                page.locator("#study-dropdown").click()
                page.locator(".dash-dropdown-content").get_by_text(
                    q["study"], exact=True).first.click()
                page.wait_for_timeout(1200)
                page.locator("#sample-dropdown").click()
                page.locator(".dash-dropdown-content").get_by_text(
                    q["sample"], exact=False).first.click()
                page.wait_for_function(
                    "n => { const el = document.querySelector('.sample-preview');"
                    " return el && el.innerText.includes(n); }",
                    arg=q["sample"], timeout=60_000)
                page.wait_for_timeout(1200)
                s.shot(page, f"retrieve-{q['slug']}-01-query-selected")

                print(f"== retrieve[{q['slug']}]: run the search ==", flush=True)
                t0 = time.time()
                page.locator("#search-button").click()
                # The status banner reports which retrieval path ran; wait for it
                # to stop saying "select a sample" and for the network to have
                # nodes.
                page.wait_for_function(
                    "() => { const gd = document.querySelector('#network-graph .js-plotly-plot');"
                    " if (!gd || !gd._fullData) return false;"
                    " const t = gd._fullData[gd._fullData.length - 1];"
                    " return !!(t && t.customdata && t.customdata.length > 3); }",
                    timeout=180_000)
                page.wait_for_timeout(1500)
                status = page.locator("#search-status").inner_text().replace("\n", " ")
                print(f"     search took {time.time() - t0:.1f}s -> {status[:140]}",
                      flush=True)
                s.shot(page, f"retrieve-{q['slug']}-02-network")

                print(f"== retrieve[{q['slug']}]: open a hit in the inspector ==",
                      flush=True)
                pos = node_xy(page, "gsm", 0)
                if not pos:
                    print("     no GSM node found", flush=True)
                else:
                    print(f"     clicking {pos['label']}", flush=True)
                    page.mouse.click(pos["x"], pos["y"])
                    page.wait_for_function(
                        "() => { const d = document.querySelector('#details-panel');"
                        " return d && d.innerText.includes('GSM'); }", timeout=90_000)
                    page.wait_for_timeout(2500)  # GEO enrichment fetch for this hit
                    s.shot(page, f"retrieve-{q['slug']}-03-hit-inspector")

                if qi == 0:
                    print("== retrieve: widen to 20 neighbours ==", flush=True)
                    # Dash 4 renders the slider on Radix: the paired number input
                    # is hidden, so drive the thumb with the keyboard the way a
                    # keyboard user would, one step per press from 5 up to 20.
                    thumb = page.locator("#topk-slider [role=slider]").first
                    thumb.click()
                    for _ in range(15):
                        thumb.press("ArrowRight")
                        page.wait_for_timeout(60)
                    page.wait_for_timeout(800)
                    print("     top-k now "
                          + page.locator("#topk-slider [role=slider]").first
                                .get_attribute("aria-valuenow"), flush=True)
                    page.locator("#search-button").click()
                    page.wait_for_function(
                        "() => { const gd = document.querySelector('#network-graph .js-plotly-plot');"
                        " if (!gd || !gd._fullData) return false;"
                        " const t = gd._fullData[gd._fullData.length - 1];"
                        " return !!(t && t.customdata"
                        "   && t.customdata.filter(c => c[0] === 'gsm').length >= 20); }",
                        timeout=180_000)
                    page.wait_for_timeout(2000)
                    s.shot(page, "retrieve-liver-04-network-top20")
                    # Reset back to 5 so the map cross-reference below frames the
                    # same tight neighbourhood the network view showed by default.
                    thumb = page.locator("#topk-slider [role=slider]").first
                    thumb.click()
                    for _ in range(15):
                        thumb.press("ArrowLeft")
                        page.wait_for_timeout(60)
                    page.locator("#search-button").click()
                    page.wait_for_function(
                        "() => { const gd = document.querySelector('#network-graph .js-plotly-plot');"
                        " if (!gd || !gd._fullData) return false;"
                        " const t = gd._fullData[gd._fullData.length - 1];"
                        " return !!(t && t.customdata && t.customdata.length > 3); }",
                        timeout=180_000)
                    page.wait_for_timeout(1500)

                print(f"== retrieve[{q['slug']}]: see this retrieval on the map, "
                      "framed on the query and its neighbours ==", flush=True)
                page.locator("#see-on-map").click()
                wait_points(page, minimum=900_000)
                page.wait_for_selector("#retrieval-group", state="visible",
                                        timeout=60_000)
                page.wait_for_timeout(3000)
                summary = page.locator("#retrieval-summary").inner_text().replace(
                    "\n", " ")
                print(f"     rail says: {summary[:140]}", flush=True)
                s.shot(page, f"retrieve-{q['slug']}-05-map-whole-corpus")

                page.locator("#frame-retrieval").click()
                page.wait_for_timeout(7000)
                wait_points(page, minimum=10)
                page.wait_for_timeout(2000)
                s.shot(page, f"retrieve-{q['slug']}-06-map-framed-on-hit")

            # ---------------- Map, clean (no retrieval in the store) ----------
            print("\n== map: default view, whole corpus ==", flush=True)
            clean = browser.new_context(viewport={"width": 1680, "height": 1010},
                                        device_scale_factor=2)
            mp = clean.new_page()
            mp.on("pageerror", lambda e: errors.append(str(e)))
            mp.on("console", lambda m: errors.append(m.text)
                  if m.type == "error" else None)
            t0 = time.time()
            mp.goto(f"{base}/map", wait_until="load")
            n = wait_points(mp, minimum=900_000)
            print(f"     {n:,} glyphs in {time.time() - t0:.1f}s", flush=True)
            mp.wait_for_timeout(2500)
            s.shot(mp, "map-umap-tissue")

            print("== map: colour by species ==", flush=True)
            set_colorby(mp, "Species")
            wait_points(mp, minimum=900_000)
            mp.wait_for_timeout(3000)
            s.shot(mp, "map-umap-species")

            print("== map: an OSDR-only field draws ARCHS4 as context ==", flush=True)
            set_colorby(mp, "Flight vs Ground")
            wait_points(mp, minimum=900_000)
            mp.wait_for_timeout(3000)
            cov = mp.locator("#coverage").inner_text().replace("\n", " ")
            print(f"     coverage: {cov[:120]}", flush=True)
            s.shot(mp, "map-coverage-osdr-only")

            print("== map: back to tissue, zoom into a neighbourhood ==", flush=True)
            set_colorby(mp, "Tissue")
            wait_points(mp, minimum=900_000)
            mp.wait_for_timeout(2500)
            box = mp.locator(".js-plotly-plot .nsewdrag").first.bounding_box()
            if box:
                # Centre the wheel on an OSDR glyph, so the detail view lands
                # on a neighbourhood with NASA samples in it rather than on
                # whatever happens to sit at the middle of the canvas.
                target = osdr_glyph_xy(mp) or {
                    "x": box["x"] + box["width"] * 0.45,
                    "y": box["y"] + box["height"] * 0.45}
                mp.mouse.move(target["x"], target["y"])
                for _ in range(16):
                    mp.mouse.wheel(0, -260)
                    mp.wait_for_timeout(320)
                mp.wait_for_timeout(8000)
                live = mp.locator(".bm-plot-badges").inner_text().replace("\n", " ")
                print(f"     zoomed badges: {live}", flush=True)
                s.shot(mp, "map-zoom-detail")

                # Hover an OSDR glyph so a shot shows what one point carries.
                # Keep well clear of the floating legend, which sits above the
                # plot and would otherwise cover the hover card.
                pos = osdr_glyph_xy(mp)
                if pos:
                    print(f"     hovering {pos['key']}", flush=True)
                    mp.mouse.move(pos["x"] - 60, pos["y"] - 60)
                    mp.wait_for_timeout(300)
                    mp.mouse.move(pos["x"], pos["y"])
                    mp.wait_for_timeout(1800)
                    if mp.locator(".hoverlayer .hovertext").count():
                        s.shot(mp, "map-hover-osdr-point", park=False)
                    else:
                        print("     no hover card appeared", flush=True)
                mp.locator(".modebar-btn[data-title='Reset axes']").first.click()
                mp.wait_for_timeout(5000)
                wait_points(mp, minimum=900_000)

            def shoot_3d(name: str):
                set_segment(mp, "Dimensions", "3D")
                wait_points(mp, minimum=1000)
                mp.wait_for_timeout(6000)
                # The default scatter3d camera leaves the cloud small in a lot
                # of empty axis box, so dolly in and turn it a little
                # off-square.
                box = mp.locator(".js-plotly-plot").first.bounding_box()
                if box:
                    cx = box["x"] + box["width"] * 0.45
                    cy = box["y"] + box["height"] / 2
                    mp.mouse.move(cx, cy)
                    for _ in range(2):
                        mp.mouse.wheel(0, -240)
                        mp.wait_for_timeout(500)
                    mp.mouse.move(cx, cy)
                    mp.mouse.down()
                    mp.mouse.move(cx - 70, cy + 25, steps=20)
                    mp.mouse.up()
                    mp.wait_for_timeout(4000)
                s.shot(mp, name)
                set_segment(mp, "Dimensions", "2D")
                mp.wait_for_timeout(3000)
                wait_points(mp, minimum=900_000)

            print("== map: UMAP 3-D ==", flush=True)
            shoot_3d("map-umap-3d")

            print("== map: PCA 2-D and 3-D ==", flush=True)
            set_segment(mp, "Projection", "PCA")
            wait_points(mp, minimum=900_000)
            mp.wait_for_timeout(4000)
            print("     params: "
                  + mp.locator("#method-params").inner_text().replace("\n", " "),
                  flush=True)
            s.shot(mp, "map-pca-2d")
            shoot_3d("map-pca-3d")

            print("== map: t-SNE 2-D and 3-D ==", flush=True)
            set_segment(mp, "Projection", "t-SNE")
            wait_points(mp, minimum=900_000)
            mp.wait_for_timeout(5000)
            print("     params: "
                  + mp.locator("#method-params").inner_text().replace("\n", " "),
                  flush=True)
            s.shot(mp, "map-tsne-2d")
            shoot_3d("map-tsne-3d")
            clean.close()

            ctx.close()
            browser.close()
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()

    real = [e for e in errors
            if "favicon" not in e.lower() and "_dash-component-suites" not in e]
    print("\n" + "=" * 60)
    print(f"{len(s.taken)} screenshots in {out}")
    for name in s.taken:
        print("  " + name)
    if real:
        print(f"\n{len(real)} console error(s):")
        for e in real[:10]:
            print("  " + e[:200])
    else:
        print("no console errors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
