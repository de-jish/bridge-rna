#!/usr/bin/env python3
"""Browser checks for the "i" explanations and the guided tour.

    /Users/josh/Bridge-RNA/.venv/bin/python tests/e2e_help_check.py [--port 8075] [--headed]

Starts its own server. Each check prints PASS or FAIL and the script exits
non-zero if any failed, like the other e2e suites. What it covers, and why a
unit test cannot: the popover's focus and dismissal rules, a route change
while one is open, the guide opening by itself only once a page has drawn
(the map waits for its first figure), blocked storage, keyboard-only use,
reduced motion, and the phone header.

The guide skips auto-open under automation (`navigator.webdriver`), so the
first-visit checks pretend not to be automated.
Design: docs/help-and-guide.md.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tests"))
from guide_screenshots import start_server  # noqa: E402

NOT_AUTOMATED = "Object.defineProperty(navigator, 'webdriver', {get: () => false});"
SEEN = "try { localStorage.setItem('bridge-rna-guide-seen', 'x') } catch (e) {}"
BLOCKED_STORAGE = """Object.defineProperty(window, 'localStorage', {
  get() { throw new DOMException('blocked', 'SecurityError'); } });"""

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, bool(ok), detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail and not ok else ""),
          flush=True)


def popover_state(page) -> dict:
    return page.evaluate("""() => { const p = document.getElementById('info-popover');
      const a = document.activeElement;
      return { open: !!p && !p.hidden, key: p && p.getAttribute('data-key'),
               focus: a && (a.id || a.getAttribute('data-info') || a.className) }; }""")


def guide_state(page) -> dict:
    return page.evaluate("""() => { const d = document.getElementById('guide');
      const c = d.querySelector('.guide-chapter[data-current]');
      return { open: d.open, chapter: c && c.dataset.chapter,
               counter: d.querySelector('.guide-counter').textContent }; }""")


def checks_popover(ctx, base: str) -> None:
    print("\n== the i explanations ==", flush=True)
    page = ctx.new_page()
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(base + "/", wait_until="load")
    page.wait_for_selector("[data-info='osdr-sample']", timeout=60_000)
    page.wait_for_timeout(1500)

    n = page.locator(".info-btn[data-info]").count()
    check("the retrieval view carries at least 15 explanations", n >= 15, str(n))

    button = page.locator("[data-info='number-of-matches']")
    k_before = page.locator("#topk-input").input_value()
    button.click()
    s = popover_state(page)
    check("clicking an i opens its explanation", s["open"] and s["key"] == "number-of-matches", str(s))
    check("focus moves into the popover", s["focus"] == "info-popover", str(s))
    check("the click did not reach the control beside it",
          page.locator("#topk-input").input_value() == k_before)
    check("the explanation shows a code excerpt naming its file",
          "bridge_rna/retrieval.py" in page.locator("#info-popover").inner_text())
    box = page.locator("#info-popover").bounding_box()
    vp = page.viewport_size
    check("the popover is inside the viewport",
          box["x"] >= 0 and box["y"] >= 0 and box["x"] + box["width"] <= vp["width"]
          and box["y"] + box["height"] <= vp["height"], str(box))

    page.keyboard.press("Escape")
    s = popover_state(page)
    check("Esc closes it and returns focus to the button",
          not s["open"] and s["focus"] == "number-of-matches", str(s))

    page.keyboard.press("Enter")
    check("Enter on the focused button reopens it", popover_state(page)["open"])
    page.keyboard.press("Tab")
    page.keyboard.press("Tab")
    page.keyboard.press("Tab")
    s = popover_state(page)
    check("tabbing out of the popover closes it rather than trapping focus",
          not s["open"], str(s))

    button.click()
    page.locator("#results-panel").click(position={"x": 5, "y": 5})
    check("an outside click closes it", not popover_state(page)["open"])

    button.click()
    page.locator("[data-info='osdr-study']").click()
    s = popover_state(page)
    expanded = page.evaluate("""() => [...document.querySelectorAll('.info-btn[aria-expanded="true"]')]
      .map(b => b.dataset.info)""")
    check("only one is open at a time", s["key"] == "osdr-study" and expanded == ["osdr-study"],
          f"{s} {expanded}")

    scrolled = page.evaluate("""() => { const el = [...document.querySelectorAll('aside, main, section, div')]
      .find(n => n.scrollHeight > n.clientHeight + 20 && /(auto|scroll)/.test(getComputedStyle(n).overflowY));
      if (!el) return false; el.scrollTop += 40; return true; }""")
    page.wait_for_timeout(300)
    check("found a panel to scroll", scrolled)
    s = popover_state(page)
    check("a scroll closes it", not s["open"])
    check("and focus that was in it returns to its button", s["focus"] == "osdr-study", str(s))

    page.locator("#mode-tab-cohort").click()
    page.wait_for_timeout(1200)
    page.locator("[data-info='cohort']").click()
    check("an explanation inside a mode panel opens", popover_state(page)["key"] == "cohort")

    # Navigate with a popover open: the header and view are replaced.
    page.locator(".app-nav-item:has-text('Map')").click()
    page.wait_for_selector("[data-info='color-by']", timeout=120_000)
    page.wait_for_timeout(800)
    check("a route change closes the open explanation", not popover_state(page)["open"])
    page.locator("[data-info='color-by']").click()
    check("explanations open on the map after navigating", popover_state(page)["key"] == "color-by")
    page.keyboard.press("Escape")
    page.locator(".app-nav-item:has-text('Retrieve')").click()
    page.wait_for_selector("[data-info='osdr-sample']", timeout=60_000)
    page.locator("[data-info='osdr-sample']").click()
    check("and again after coming back", popover_state(page)["key"] == "osdr-sample")
    page.keyboard.press("Escape")

    # Inspector entries are rendered by a callback.
    page.locator("#mode-tab-sample").click()
    page.locator("#search-button").click()
    page.wait_for_selector("#results-table-wrap tr.result-row", timeout=180_000)
    page.locator("#results-table-wrap tr.result-row td.col-title").first.click()
    page.wait_for_selector("[data-info='match-details']", timeout=60_000)
    page.locator("[data-info='match-details']").click()
    check("the inspector's explanation opens", popover_state(page)["key"] == "match-details")
    head = page.evaluate("""() => document.querySelector('.details-heading').getBoundingClientRect().height""")
    check("the inspector heading stays on one line", head < 32, f"{head:.0f}px")
    check("no page errors", not errors, "; ".join(errors[:3]))
    page.close()


def checks_guide(browser, base: str) -> None:
    print("\n== the guided tour ==", flush=True)

    def context(*scripts, **kw):
        ctx = browser.new_context(viewport=kw.pop("viewport", {"width": 1440, "height": 900}), **kw)
        for s in scripts:
            ctx.add_init_script(s)
        return ctx

    # Automated browsers are left alone.
    ctx = context()
    page = ctx.new_page()
    page.goto(base + "/", wait_until="load")
    page.wait_for_selector("#results-panel", timeout=60_000)
    page.wait_for_timeout(2500)
    check("an automated browser is not shown the tour", not guide_state(page)["open"])
    ctx.close()

    # First visit to the retrieval view.
    ctx = context(NOT_AUTOMATED)
    page = ctx.new_page()
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(base + "/", wait_until="load")
    page.wait_for_function("() => document.getElementById('guide')?.open", timeout=40_000)
    s = guide_state(page)
    check("a first visit opens the tour at the chapter for the page",
          s["chapter"] == "retrieve" and s["counter"].startswith("Step 1 of 12"), str(s))
    loaded = page.evaluate("() => [...document.querySelectorAll('#guide img')].filter(i => i.getAttribute('src')).length")
    check("only the current and next screenshots are fetched", loaded == 2, str(loaded))
    focus = page.evaluate("() => document.activeElement.className")
    check("focus starts on Next", "guide-next" in focus, focus)
    inert = page.evaluate("""() => { const b = document.querySelector('#search-button');
      const r = b.getBoundingClientRect();
      const top = document.elementFromPoint(r.left + 5, r.top + 5);
      return !b.contains(top); }""")
    check("the page behind is covered", inert)
    page.keyboard.press("ArrowRight")
    check("ArrowRight moves on", guide_state(page)["counter"].startswith("Step 2 of 12"))
    page.keyboard.press("Tab")
    page.keyboard.press("Shift+Tab")
    page.keyboard.press("Shift+Tab")
    page.keyboard.press("Shift+Tab")
    inside = page.evaluate("() => document.getElementById('guide').contains(document.activeElement)")
    check("focus stays inside the open tour", inside)
    page.locator("#guide-tab-map").click()
    s = guide_state(page)
    check("a chapter tab switches chapter and restarts it",
          s["chapter"] == "map" and s["counter"].startswith("Step 1 of 8"), str(s))
    page.keyboard.press("Escape")
    page.wait_for_timeout(300)
    s = guide_state(page)
    focus = page.evaluate("() => document.activeElement.className")
    check("Esc closes it and focus returns to How it works",
          not s["open"] and "app-guide-btn" in focus, focus)
    page.reload(wait_until="load")
    page.wait_for_selector("#results-panel", timeout=60_000)
    page.wait_for_timeout(3000)
    check("it opens by itself only once", not guide_state(page)["open"])

    # The header button opens it from the map too, across a route change.
    page.locator(".app-nav-item:has-text('Map')").click()
    page.wait_for_selector("#retrieval-group, [data-info='color-by']", timeout=120_000)
    page.locator(".app-guide-btn").click()
    page.wait_for_timeout(400)
    s = guide_state(page)
    check("How it works opens the map chapter on the map", s["open"] and s["chapter"] == "map", str(s))
    for _ in range(7):
        page.keyboard.press("ArrowRight")
    check("the last slide offers Done", page.locator("#guide .guide-next").inner_text().strip() == "Done")
    page.locator("#guide .guide-next").click()
    page.wait_for_timeout(300)
    check("Done closes it", not guide_state(page)["open"])
    check("no page errors", not errors, "; ".join(errors[:3]))
    ctx.close()

    # A cold /map waits for its first figure.
    ctx = context(NOT_AUTOMATED)
    page = ctx.new_page()
    page.goto(base + "/map", wait_until="load")
    t0 = time.time()
    page.wait_for_function("() => document.getElementById('guide')?.open", timeout=60_000)
    ready = page.evaluate("() => document.getElementById('map-skeleton').classList.contains('is-done')")
    check("on a cold /map it waits for the first figure", ready, f"{time.time() - t0:.1f}s")
    check("and opens at the map chapter", guide_state(page)["chapter"] == "map")
    ctx.close()

    # Interacting first cancels it.
    ctx = context(NOT_AUTOMATED)
    page = ctx.new_page()
    page.goto(base + "/map", wait_until="load")
    page.wait_for_selector(".app-guide-btn", timeout=60_000)
    page.mouse.click(700, 18)
    page.wait_for_function("() => document.getElementById('map-skeleton')?.classList.contains('is-done')",
                           timeout=120_000)
    page.wait_for_timeout(1500)
    check("a reader who starts clicking first is not interrupted", not guide_state(page)["open"])
    ctx.close()

    # Blocked storage: never opens by itself, still opens by hand, no errors.
    ctx = context(NOT_AUTOMATED, BLOCKED_STORAGE)
    page = ctx.new_page()
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(base + "/", wait_until="load")
    page.wait_for_selector("#results-panel", timeout=60_000)
    page.wait_for_timeout(3000)
    check("with storage blocked it does not open by itself", not guide_state(page)["open"])
    page.locator(".app-guide-btn").click()
    page.wait_for_timeout(300)
    check("with storage blocked it still opens by hand", guide_state(page)["open"])
    check("with storage blocked there are no page errors", not errors, "; ".join(errors[:3]))
    ctx.close()

    # Reduced motion.
    ctx = context(SEEN, reduced_motion="reduce")
    page = ctx.new_page()
    page.goto(base + "/", wait_until="load")
    page.wait_for_selector("#results-panel", timeout=60_000)
    page.locator(".app-guide-btn").click()
    anim = page.evaluate("""() => [getComputedStyle(document.getElementById('guide')).animationName,
      getComputedStyle(document.querySelector('#guide .guide-slide[data-current]')).animationName]""")
    check("under reduced motion nothing animates", anim == ["none", "none"], str(anim))
    ctx.close()

    # A very short window: the words scroll rather than being cut off.
    ctx = context(SEEN, viewport={"width": 640, "height": 300})
    page = ctx.new_page()
    page.goto(base + "/", wait_until="load")
    page.wait_for_selector("#results-panel", timeout=60_000)
    page.locator(".app-guide-btn").click()
    for _ in range(7):
        page.keyboard.press("ArrowRight")
    page.wait_for_timeout(400)
    reach = page.evaluate("""() => { const b = document.querySelector('#guide .guide-body');
      const t = document.querySelector('#guide .guide-slide[data-current] .guide-slide-body');
      b.scrollTop = b.scrollHeight; const r = t.getBoundingClientRect(), f = b.getBoundingClientRect();
      return r.bottom <= f.bottom + 1; }""")
    check("in a very short window the slide's words can be scrolled to", reach)
    ctx.close()

    # Phone.
    ctx = context(SEEN, viewport={"width": 375, "height": 812}, is_mobile=True, has_touch=True)
    page = ctx.new_page()
    page.goto(base + "/", wait_until="load")
    page.wait_for_selector("#results-panel", timeout=60_000)
    page.wait_for_timeout(800)
    header = page.evaluate("""() => ({ h: document.querySelector('.app-title').getBoundingClientRect().height,
      sw: document.documentElement.scrollWidth,
      name: document.querySelector('.app-guide-btn').innerText })""")
    check("on a phone the title stays on one line and nothing scrolls sideways",
          header["h"] < 30 and header["sw"] <= 375, str(header))
    page.locator(".app-guide-btn").tap()
    page.wait_for_timeout(500)
    box = page.locator("#guide").bounding_box()
    check("on a phone the tour fits the screen",
          guide_state(page)["open"] and box["width"] <= 375 and box["height"] <= 812, str(box))
    ctx.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8075)
    ap.add_argument("--headed", action="store_true")
    args = ap.parse_args()
    log = Path(os.environ.get("TMPDIR", "/tmp")) / "e2e-help-server.log"
    server = start_server(args.port, log)
    try:
        base = f"http://127.0.0.1:{args.port}"
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=not args.headed)
            ctx = browser.new_context(viewport={"width": 1440, "height": 900})
            ctx.add_init_script(SEEN)
            checks_popover(ctx, base)
            ctx.close()
            checks_guide(browser, base)
            browser.close()
    finally:
        server.terminate()
        server.wait(timeout=10)
    failed = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
