#!/usr/bin/env python3
"""End-to-end browser check of the running app, the way a user meets it.

Deliberately NOT part of the pytest suite, and named so pytest does not collect
it. Everything else in `tests/` runs against a synthetic corpus in a temp
directory and finishes in about a second; this boots the real Dash app against
the real `cache/` and drives a real browser, so it needs artifacts a fresh
clone does not have and takes about a minute.

It exists because "160 tests pass" says nothing about whether 942,563 WebGL
glyphs actually arrive in a browser and stay interactive. It asserts on what
the page reports about itself rather than on what the server intended to send,
and it is what caught that the density underlay was really gone from the DOM,
that the budget tiers draw exactly the counts they claim, and that the whole
corpus reaches the client in one figure.

    /Users/josh/Bridge-RNA/.venv/bin/python tests/e2e_check.py [--port 8062] [--headed]

One trap is baked into the counting helper below: under plotly 6 the
coordinates arrive as base64 typed-array specs, so `gd.data[i].x` has no
`.length` and naive counting silently yields NaN, which looks exactly like an
empty plot. Read `_fullData`.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

REPO = Path(__file__).resolve().parent.parent
PY = os.environ.get("MANIFOLD_PYTHON", sys.executable)
SHOTS = Path(os.environ.get("MANIFOLD_E2E_SHOTS",
                            Path(tempfile.gettempdir()) / "bm-e2e-shots"))

# How Plotly exposes what it actually drew.
#
# Read `_fullData`, not `data`. Under plotly 6 the coordinates arrive as base64
# typed-array specs, so `gd.data[i].x` is a `{dtype, bdata}` object with no
# `.length` at all - counting it silently yields NaN and looks like an empty
# plot. `_fullData` holds the decoded `Float32Array` plus every defaulted
# attribute, so it is what the browser is actually rendering.
COUNT_JS = """() => {
  const gd = document.querySelector('.js-plotly-plot');
  if (!gd || !gd._fullData) return null;
  let n = 0, traces = [];
  for (const t of gd._fullData) {
    const len = (t.x && t.x.length) || 0;
    n += len;
    traces.push({name: t.name, type: t.type, n: len,
                 color: t.marker && t.marker.color,
                 opacity: t.marker && t.marker.opacity});
  }
  return {total: n, traces: traces,
          images: (gd.layout.images || []).length,
          webgl: !!gd.querySelector('canvas')};
}"""


class Checks:
    def __init__(self):
        self.failures: list[str] = []
        self.notes: list[str] = []

    def ok(self, cond: bool, msg: str) -> bool:
        print(("  OK   " if cond else "  FAIL ") + msg, flush=True)
        if not cond:
            self.failures.append(msg)
        return cond

    def note(self, msg: str) -> None:
        print("  ..   " + msg, flush=True)
        self.notes.append(msg)


def wait_for_points(page, timeout=180_000):
    """Wait until the graph reports a non-zero glyph count, then return it."""
    page.wait_for_function(
        "() => { const gd = document.querySelector('.js-plotly-plot');"
        " return gd && gd._fullData && gd._fullData.length > 0"
        " && gd._fullData.some(t => t.x && t.x.length > 0); }",
        timeout=timeout)
    return page.evaluate(COUNT_JS)


def set_segment(page, group_label: str, option_text: str):
    """Click an option in one of the segmented radio controls, by its label.

    Dash 4 renders each RadioItems option as a label carrying its own structural
    class, which is what the stylesheet targets, so match on that class rather
    than on the tag name.
    """
    group = page.locator(".bm-group", has=page.locator(
        f".bm-group-label:text-is('{group_label}')"))
    group.locator(
        f".bm-seg .dash-options-list-option:has-text('{option_text}')"
    ).first.click()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8062)
    ap.add_argument("--headed", action="store_true")
    args = ap.parse_args()
    SHOTS.mkdir(parents=True, exist_ok=True)
    c = Checks()

    server = subprocess.Popen(
        [PY, "app.py", "--port", str(args.port)], cwd=REPO,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        # Wait for the server to announce itself rather than sleeping blind.
        t0 = time.time()
        while time.time() - t0 < 120:
            line = server.stdout.readline()
            if not line:
                break
            print("    [server] " + line.rstrip(), flush=True)
            if "serving on" in line:
                break
        else:
            print("server never announced itself")
            return 1

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=not args.headed)
            page = browser.new_page(viewport={"width": 1680, "height": 1000})
            console_errors: list[str] = []
            page.on("console", lambda m: console_errors.append(m.text)
                    if m.type == "error" else None)
            page.on("pageerror", lambda e: console_errors.append(str(e)))

            print("\n=== 1. first paint, default controls ===")
            t0 = time.time()
            page.goto(f"http://127.0.0.1:{args.port}/map", wait_until="load")
            info = wait_for_points(page)
            first_paint = time.time() - t0
            c.note(f"first interactive frame in {first_paint:.1f}s")
            print(f"     total glyphs: {info['total']:,}  traces: {len(info['traces'])}  "
                  f"webgl canvas: {info['webgl']}  layout images: {info['images']}")
            c.ok(info["total"] > 900_000,
                 f"the default view draws the whole corpus ({info['total']:,} glyphs)")
            c.ok(info["images"] == 0, "no density underlay image is present")
            c.ok(info["webgl"], "the plot rendered onto a WebGL canvas")
            c.ok(first_paint < 60, f"first paint under 60s ({first_paint:.1f}s)")
            page.screenshot(path=str(SHOTS / "01-default-tissue.png"))

            print("\n=== 2. the control rail ===")
            budget_labels = page.locator(
                ".bm-group:has(.bm-group-label:text-is('ARCHS4 point budget')) "
                ".bm-seg .dash-options-list-option").all_inner_texts()
            print(f"     budget tiers: {budget_labels}")
            c.ok(len(budget_labels) == 4, f"four budget tiers offered: {budget_labels}")
            c.ok(any("All" in b for b in budget_labels), "an 'All' tier is offered")
            layer_labels = page.locator(
                ".bm-group:has(.bm-group-label:text-is('Layers')) "
                ".bm-checklist .dash-options-list-option").all_inner_texts()
            print(f"     layers: {layer_labels}")
            c.ok(not any("ensity" in t for t in layer_labels),
                 "the Layers control no longer offers a density underlay")
            c.ok(len(layer_labels) == 2, f"two layer toggles remain: {layer_labels}")

            # Nothing may overflow the 268px rail; a wrapped segmented control
            # is the most likely casualty of going from three tiers to four.
            overflow = page.evaluate(
                "() => { const r = document.querySelector('.bm-rail');"
                " return {scroll: r.scrollWidth, client: r.clientWidth}; }")
            c.ok(overflow["scroll"] <= overflow["client"] + 1,
                 f"the control rail does not overflow horizontally ({overflow})")
            seg_h = page.evaluate(
                "() => document.querySelector(\"[id='budget']\").getBoundingClientRect().height")
            c.ok(seg_h < 60, f"the budget control is a single row ({seg_h:.0f}px tall)")
            # The Projection control went from two pills to three, so it is now
            # the one most likely to wrap.
            method_labels = page.locator(
                ".bm-group:has(.bm-group-label:text-is('Projection')) "
                ".bm-seg .dash-options-list-option").all_inner_texts()
            print(f"     projections: {method_labels}")
            c.ok(len(method_labels) == 3,
                 f"three projections offered: {method_labels}")
            method_h = page.evaluate(
                "() => document.querySelector(\"[id='method']\").getBoundingClientRect().height")
            c.ok(method_h < 60,
                 f"the projection control is a single row ({method_h:.0f}px tall)")

            print("\n=== 3. an OSDR-only field draws context, not a grey category ===")
            page.locator("#color-by").click()
            page.locator(".dash-dropdown-content").get_by_text(
                "Flight vs Ground", exact=False).first.click()
            page.wait_for_timeout(2500)
            info = wait_for_points(page)
            ctx = [t for t in info["traces"] if t["color"] == "#43597c"]
            print(f"     traces: {[(t['name'], t['n']) for t in info['traces']][:6]}")
            c.ok(bool(ctx), "an ARCHS4 context trace is drawn in the context colour")
            if ctx:
                c.ok(ctx[0]["opacity"] < 0.5,
                     f"the context cloud is faint (opacity {ctx[0]['opacity']})")
                c.ok(ctx[0]["n"] > 900_000,
                     f"the context cloud carries the whole ARCHS4 corpus ({ctx[0]['n']:,})")
            c.ok(info["images"] == 0, "still no underlay image")
            badges = page.locator(".bm-plot-badges").inner_text()
            print(f"     badges: {badges!r}")
            c.ok("context only" in badges, "the plot badge says 'context only'")
            coverage = page.locator("#coverage").inner_text()
            print(f"     coverage: {coverage!r}")
            c.ok("context" in coverage.lower(),
                 "the coverage readout explains what happens to uncoloured points")
            page.screenshot(path=str(SHOTS / "02-osdr-only-context.png"))

            print("\n=== 4. budget tiers re-render ===")
            page.locator("#color-by").click()
            page.locator(".dash-dropdown-content").get_by_text(
                "Tissue", exact=False).first.click()
            page.wait_for_timeout(3000)
            # Wait for the count to reach the tier's *expected* value, not merely
            # to exceed a floor. A floor is already satisfied by whatever is on
            # screen, so the wait returns instantly and the check silently
            # measures nothing - which is exactly what it did on the first run.
            n_osdr = 2_108
            for label, expected in (("100k", 100_000 + n_osdr),
                                    ("500k", 500_000 + n_osdr),
                                    ("All", 940_455 + n_osdr)):
                t0 = time.time()
                set_segment(page, "ARCHS4 point budget", label)
                page.wait_for_function(
                    "e => { const gd = document.querySelector('.js-plotly-plot');"
                    " if (!gd || !gd._fullData) return false;"
                    " const n = gd._fullData.reduce((a,t)=>a+((t.x&&t.x.length)||0),0);"
                    " return Math.abs(n - e) <= 20; }",
                    arg=expected, timeout=120_000)
                dt = time.time() - t0
                got = page.evaluate(COUNT_JS)["total"]
                c.note(f"budget {label}: {got:,} glyphs in {dt:.1f}s")
                c.ok(abs(got - expected) <= 20,
                     f"budget {label} drew {got:,}, expected about {expected:,}")
                c.ok(dt < 45, f"budget {label} re-rendered in under 45s ({dt:.1f}s)")

            print("\n=== 5. 3-D view ===")
            set_segment(page, "Dimensions", "3D")
            page.wait_for_timeout(4000)
            info = wait_for_points(page)
            print(f"     3-D glyphs: {info['total']:,}  images: {info['images']}")
            c.ok(info["total"] > 0, "the 3-D view draws points")
            c.ok(info["images"] == 0, "3-D has no underlay image")
            c.ok(all(t["type"] in ("scatter3d",) for t in info["traces"] if t["n"]),
                 "3-D traces are scatter3d")
            # 3-D must not offer a budget it cannot honour: the tiers cap at
            # 40,000 with no "All" pill, and the cloud is drawn at that cap.
            budget_3d = page.locator(
                ".bm-group:has(.bm-group-label:text-is('ARCHS4 point budget')) "
                ".bm-seg .dash-options-list-option").all_inner_texts()
            print(f"     3-D budget tiers: {budget_3d}")
            c.ok(not any("All" in b for b in budget_3d),
                 f"3-D drops the 'All' tier: {budget_3d}")
            c.ok(bool(budget_3d) and all(b.endswith("k") for b in budget_3d),
                 f"3-D tiers are all capped k-values: {budget_3d}")
            c.ok(info["total"] <= 40_000 + 2_108 + 50,
                 f"3-D caps the ARCHS4 cloud near 40k ({info['total']:,} glyphs)")
            page.screenshot(path=str(SHOTS / "03-3d.png"))

            print("\n=== 6. PCA and zoom ===")
            set_segment(page, "Dimensions", "2D")
            page.wait_for_timeout(2500)
            budget_2d = page.locator(
                ".bm-group:has(.bm-group-label:text-is('ARCHS4 point budget')) "
                ".bm-seg .dash-options-list-option").all_inner_texts()
            c.ok(any("All" in b for b in budget_2d),
                 f"2-D restores the 'All' tier: {budget_2d}")
            set_segment(page, "Projection", "PCA")
            page.wait_for_timeout(4000)
            info = wait_for_points(page)
            c.ok(info["total"] > 0, f"PCA renders ({info['total']:,} glyphs)")
            c.ok(info["images"] == 0, "PCA has no underlay image")
            page.screenshot(path=str(SHOTS / "04-pca.png"))

            print("\n=== 6b. t-SNE, and the parameter readout on the rail ===")

            def params_text() -> str:
                return page.locator("#method-params").inner_text().replace("\n", " ")

            pca_params = params_text()
            print(f"     PCA:       {pca_params}")
            c.ok("eigendecomposition" in pca_params,
                 f"the rail names how PCA was fit: {pca_params!r}")

            set_segment(page, "Projection", "t-SNE")
            page.wait_for_timeout(5000)
            info = wait_for_points(page)
            c.ok(info["total"] > 0, f"t-SNE renders ({info['total']:,} glyphs)")
            tsne_2d = params_text()
            print(f"     t-SNE 2-D: {tsne_2d}")
            c.ok("perplexity" in tsne_2d,
                 f"the rail names the perplexity: {tsne_2d!r}")
            # The two t-SNE maps are built by different algorithms, because
            # openTSNE's interpolation accelerator refuses three dimensions.
            # The rail has to say which one it is showing.
            c.ok("FIt-SNE" in tsne_2d and "Barnes-Hut" not in tsne_2d,
                 f"2-D t-SNE is named as the interpolation fit: {tsne_2d!r}")
            page.screenshot(path=str(SHOTS / "05-tsne.png"))

            set_segment(page, "Dimensions", "3D")
            page.wait_for_timeout(6000)
            tsne_3d = params_text()
            print(f"     t-SNE 3-D: {tsne_3d}")
            c.ok("Barnes-Hut" in tsne_3d and "FIt-SNE" not in tsne_3d,
                 f"3-D t-SNE is named as the Barnes-Hut fit: {tsne_3d!r}")
            page.screenshot(path=str(SHOTS / "06-tsne-3d.png"))
            set_segment(page, "Dimensions", "2D")
            page.wait_for_timeout(3000)

            print("\n=== 6c. zoom ===")
            set_segment(page, "Projection", "UMAP")
            page.wait_for_timeout(4000)
            box = page.locator(".js-plotly-plot .nsewdrag").first.bounding_box()
            if box:
                page.mouse.move(box["x"] + box["width"] / 2,
                                box["y"] + box["height"] / 2)
                t0 = time.time()
                for _ in range(4):
                    page.mouse.wheel(0, -240)
                    page.wait_for_timeout(400)
                page.wait_for_timeout(4000)
                dt = time.time() - t0
                c.note(f"four scroll-zoom steps settled in {dt:.1f}s")
                info = page.evaluate(COUNT_JS)
                c.ok(info["total"] > 0, f"the map still has points after zooming "
                                        f"({info['total']:,})")
                page.screenshot(path=str(SHOTS / "07-zoomed.png"))

            print("\n=== 7. console ===")
            real = [e for e in console_errors
                    if "favicon" not in e.lower() and "_dash-component-suites" not in e]
            for e in real[:10]:
                print(f"     {e[:160]}")
            c.ok(not real, f"no console errors ({len(real)} seen)")

            browser.close()
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()

    print("\n" + "=" * 62)
    for n in c.notes:
        print("NOTE: " + n)
    if c.failures:
        for f in c.failures:
            print("FAIL: " + f)
        print(f"E2E FAILED ({len(c.failures)} checks)")
        return 1
    print(f"ALL E2E CHECKS PASSED (screenshots in {SHOTS})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
