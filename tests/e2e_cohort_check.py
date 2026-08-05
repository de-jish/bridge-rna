#!/usr/bin/env python3
"""End-to-end browser check of cohort retrieval, the way a user meets it.

A sibling of `e2e_check.py` and `e2e_upload_check.py`: same harness, same
deliberate exclusion from the pytest suite (which never touches the real
artifacts), and the same reason for existing. `tests/test_cohorts.py` proves the
estimator and the grouping against a synthetic corpus in a fraction of a second;
`precompute/validate_cohorts.py` proves the science against the real memmap.
Neither says whether a person can define a cohort, see how far to trust it, pool
it, compare two arms, and find the result on the map.

It asserts on what the page reports about itself, not on what the server meant
to send. The specific failures it was built to catch are in the checks below,
and two of them are already regressions this feature shipped and had fixed:
callbacks firing at page load so the canvas greeted a visitor with "Cohort
retrieval failed", and the legend continuing to advertise GSE nodes while a
two-cohort comparison that draws none was on screen.

    /Users/josh/Bridge-RNA/.venv/bin/python tests/e2e_cohort_check.py \\
        [--port 8064] [--headed]
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

REPO = Path(__file__).resolve().parent.parent
PY = os.environ.get("MANIFOLD_PYTHON", sys.executable)
SHOTS = Path(os.environ.get("MANIFOLD_E2E_SHOTS",
                            Path(tempfile.gettempdir()) / "bm-cohort-shots"))

# A study with a real two-arm contrast in one tissue, so the comparison has
# something to compare. Verified against the shipped metadata: OSD-137 carries
# Liver in Basal Control, Ground Control and Space Flight, six animals each.
STUDY = "OSD-137"

# Other studies to sweep, each paired with a different retrieval depth, so the
# feature is exercised across studies, cohort shapes and top-k rather than only
# against the one study whose two-arm contrast the comparison needs. Each of
# these carries at least one cohort of two or more under the default
# definition; the picker opens on its largest, which is what gets searched.
SWEEP = [("OSD-101", 5), ("OSD-104", 12), ("OSD-168", 25)]

NETWORK_READY_JS = """() => {
  const gd = document.querySelector('#network-graph .js-plotly-plot');
  if (!gd || !gd._fullData || !gd._fullData.length) return false;
  return gd._fullData.some(t => t.customdata && t.customdata.length);
}"""

# Every node the network drew, by kind, read out of what Plotly is rendering
# rather than out of what the callback returned.
NODES_JS = """() => {
  const gd = document.querySelector('#network-graph .js-plotly-plot');
  if (!gd || !gd._fullData) return null;
  const out = {};
  for (const t of gd._fullData) {
    if (!t.customdata) continue;
    for (const row of t.customdata) {
      if (!Array.isArray(row)) continue;
      out[row[0]] = (out[row[0]] || 0) + 1;
    }
  }
  return out;
}"""

QUERY_NODE_JS = """() => {
  const gd = document.querySelector('#network-graph .js-plotly-plot');
  if (!gd || !gd._fullData) return null;
  for (const t of gd._fullData) {
    if (!t.customdata) continue;
    for (let i = 0; i < t.customdata.length; i++) {
      if (t.customdata[i][0] !== 'query') continue;
      const xa = gd._fullLayout.xaxis, ya = gd._fullLayout.yaxis;
      const r = gd.getBoundingClientRect();
      return {x: r.left + xa._offset + xa.l2p(t.x[i]),
              y: r.top + ya._offset + ya.l2p(t.y[i])};
    }
  }
  return null;
}"""

# How many OSDR query glyphs the map drew. A pooled cohort has no single
# position in the space, so it must draw one per member.
MAP_QUERY_JS = """() => {
  const gd = document.querySelector('.js-plotly-plot');
  if (!gd || !gd._fullData) return null;
  let halo = 0, query = 0;
  for (const t of gd._fullData) {
    const n = (t.x && t.x.length) || 0;
    if (t.name === 'query halo') halo += n;
    if (t.name === 'query') query += n;
  }
  return {halo: halo, query: query};
}"""

# The retrieval overlay in full, one entry per trace, so a check can ask which
# cohort a glyph belongs to rather than only how many were drawn. Points are
# rounded and stringified so a shared hit - which receives both cohorts' marks
# at identical coordinates - can be found by set intersection.
MAP_OVERLAY_JS = """() => {
  const gd = document.querySelector('.js-plotly-plot');
  if (!gd || !gd._fullData) return null;
  const out = {members: [], hits: []};
  for (const t of gd._fullData) {
    if (t.name !== 'query' && t.name !== 'retrieved hit') continue;
    const pts = [];
    for (let i = 0; i < (t.x || []).length; i++) {
      pts.push(t.x[i].toFixed(4) + ',' + t.y[i].toFixed(4));
    }
    const entry = {colour: t.marker.color, symbol: t.marker.symbol,
                   size: t.marker.size, n: pts.length, points: pts,
                   hasText: (t.mode || '').indexOf('text') >= 0};
    (t.name === 'query' ? out.members : out.hits).push(entry);
  }
  return out;
}"""


class Checks:
    def __init__(self):
        self.failures: list[str] = []
        self.ran = 0

    def ok(self, cond: bool, msg: str) -> bool:
        # Counted, because the documented totals used to be hand-written and
        # drifted: the number in the docs was never the number this file ran.
        self.ran += 1
        print(("  OK   " if cond else "  FAIL ") + msg, flush=True)
        if not cond:
            self.failures.append(msg)
        return bool(cond)

    def note(self, msg: str) -> None:
        print("  ..   " + msg, flush=True)


def shot(page, name: str) -> None:
    page.wait_for_timeout(250)
    page.screenshot(path=str(SHOTS / f"{name}.png"))


def open_mode(page, key: str) -> None:
    tab = page.locator(f"#mode-tab-{key}")
    if "is-active" not in (tab.get_attribute("class") or ""):
        tab.click()
        page.wait_for_selector(f"#mode-panel-{key}", state="visible", timeout=30_000)
        page.wait_for_timeout(500)


def choose(page, dropdown_id: str, text: str, exact: bool = False) -> None:
    page.locator(f"#{dropdown_id}").click()
    page.wait_for_timeout(400)
    page.locator(".dash-dropdown-content").get_by_text(text, exact=exact).first.click()
    page.wait_for_timeout(1800)


def banner(page) -> str:
    return page.locator("#search-status").inner_text().replace("\n", " ")


def _topk_handle(page):
    """Dash 4 renders dcc.Slider as a radix slider plus a *hidden* number input.

    So the number box cannot be filled - Playwright refuses to type into an
    invisible element, correctly - and the handle is driven by keyboard instead,
    which is exact where dragging by pixel offset is not.
    """
    return page.locator("#topk-slider [role=slider]").first


def topk_value(page) -> int:
    return int(_topk_handle(page).get_attribute("aria-valuenow"))


def set_topk(page, k: int) -> None:
    handle = _topk_handle(page)
    handle.focus()
    current = topk_value(page)
    key = "ArrowRight" if k > current else "ArrowLeft"
    for _ in range(abs(k - current)):
        page.keyboard.press(key)
    page.wait_for_timeout(800)
    got = topk_value(page)
    if got != k:
        raise AssertionError(f"could not set top-k to {k}, stuck at {got}")


def offer_count(text: str) -> int:
    """The number in "See N hits on the map"."""
    m = re.search(r"(\d+)", text or "")
    return int(m.group(1)) if m else 0


def shared_count(page) -> int:
    """How many hits the status banner says both cohorts retrieved.

    Read this while still on the retrieval view. The router destroys a view when
    you leave it, so `#search-status` does not exist once the map is open - only
    `hits-store`, which lives on the shell, survives the trip.
    """
    m = re.search(r"share (\d+) of", banner(page))
    return int(m.group(1)) if m else -1


def wait_for_map(page) -> None:
    page.wait_for_function(
        "() => { const gd = document.querySelector('.js-plotly-plot');"
        " return gd && gd._fullData && gd._fullData.some("
        "t => t.name === 'query'); }", timeout=180_000)
    page.wait_for_timeout(3000)


def run_cohort_search(page, timeout: int = 180_000) -> float:
    """Click Search and wait for *this* search, not for evidence of any search.

    The obvious predicates - "the network has nodes", "the running indicator is
    empty" - are both already true the moment a second search starts, because
    the first one left a network on screen and an idle spinner. Waiting on them
    returns immediately and the caller then reads the *previous* result's
    banner, which is exactly how the comparison step first appeared to fail
    while the app was doing the right thing.

    So this waits for the banner to actually change. The running indicator is
    watched too, but only as a secondary settle.
    """
    before = page.locator("#search-status").inner_text()
    t0 = time.time()
    page.locator("#cohort-search-button").click()
    page.wait_for_function(
        "prev => { const el = document.querySelector('#search-status');"
        " return el && el.innerText !== prev; }", arg=before, timeout=timeout)
    page.wait_for_function(NETWORK_READY_JS, timeout=timeout)
    page.wait_for_function(
        "() => { const el = document.querySelector('#cohort-running-indicator');"
        " return el && !el.innerText.trim(); }", timeout=timeout)
    page.wait_for_timeout(900)
    return time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8064)
    ap.add_argument("--headed", action="store_true")
    args = ap.parse_args()
    SHOTS.mkdir(parents=True, exist_ok=True)
    c = Checks()
    base = f"http://127.0.0.1:{args.port}"

    server = subprocess.Popen(
        [PY, "app.py", "--port", str(args.port)], cwd=REPO,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
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
            page = browser.new_page(viewport={"width": 1680, "height": 1010})
            console_errors: list[str] = []
            page.on("console", lambda m: console_errors.append(m.text)
                    if m.type == "error" else None)
            page.on("pageerror", lambda e: console_errors.append(str(e)))

            # ---- 1. the rail opens on Sample and stays quiet ----------------
            print("\n=== 1. a cold load runs nothing ===")
            page.goto(f"{base}/", wait_until="load")
            page.wait_for_selector(".sample-preview", timeout=60_000)
            page.wait_for_timeout(1500)

            c.ok(page.locator(".mode-tab").count() == 3,
                 "the rail offers three query sources")
            c.ok("is-active" in (page.locator("#mode-tab-sample")
                                 .get_attribute("class") or ""),
                 "it opens on Sample")
            c.ok(page.locator("#mode-panel-sample").is_visible()
                 and not page.locator("#mode-panel-cohort").is_visible()
                 and not page.locator("#mode-panel-upload").is_visible(),
                 "exactly one query panel is on screen")
            c.ok(page.locator("#action-slot-sample").is_visible()
                 and not page.locator("#action-slot-cohort").is_visible(),
                 "exactly one action button is on screen")
            # The regression: restyling the action slots on load remounted the
            # buttons inside them, and Dash fires a callback when an input
            # component newly appears - so both the cohort and the upload search
            # ran at n_clicks 0 and the canvas said "Cohort retrieval failed".
            msg = banner(page)
            c.ok("failed" not in msg.lower() and "Upload a counts file" not in msg,
                 f"no search ran at page load: {msg[:70]!r}")
            c.ok("Select" in msg, "the banner invites a search instead")
            shot(page, "01-sample-mode")

            # ---- 2. defining a cohort --------------------------------------
            print("\n=== 2. defining a cohort ===")
            open_mode(page, "cohort")
            c.ok(page.locator("#mode-panel-cohort").is_visible(),
                 "the cohort panel opens")
            c.ok(not page.locator("#mode-panel-sample").is_visible(),
                 "the sample panel closes")
            c.ok(page.locator("#study-group").is_visible(),
                 "the study picker is shared with Sample mode")

            chips = page.locator(".facet-chip")
            c.ok(chips.count() == 3, f"three facets offered, saw {chips.count()}")
            labels = {chips.nth(i).inner_text().strip() for i in range(chips.count())}
            c.ok(labels == {"Study", "Tissue", "Spaceflight arm"},
                 f"and they are the curated grouping, nothing else: {sorted(labels)}")
            study_chip = page.locator("button.facet-chip", has_text="Study").first
            c.ok(study_chip.is_disabled(), "Study is pinned and cannot be unticked")
            c.ok(bool(study_chip.get_attribute("title")),
                 "the pinned chip carries the reason it is pinned")
            on = page.locator(".facet-chip.is-on")
            c.ok(on.count() == chips.count(),
                 f"every offered facet is on by default, saw {on.count()}")

            summary = page.locator("#cohort-facet-summary").inner_text()
            c.ok("212 cohorts" in summary,
                 f"the definition reports its consequence: {summary[:80]!r}")
            c.ok("2,105 of 2,108" in summary,
                 "and says how many samples it groups")

            choose(page, "study-dropdown", STUDY, exact=True)
            picked = page.locator("#cohort-dropdown").inner_text()
            c.ok("samples" in picked, f"a cohort is selected: {picked!r}")
            shot(page, "02-cohort-defined")

            # ---- 3. the confidence card ------------------------------------
            print("\n=== 3. how far to trust it ===")
            card = page.locator("#cohort-card").inner_text()
            c.ok("samples pooled into one query" in card,
                 "the card states the pooled size")
            c.ok("RESULT STABILITY" in card.upper(),
                 "the card quotes result stability")
            m = re.search(r"0\.\d\d", card)
            c.ok(bool(m), f"stability is a quoted number: {card[:60]!r}")
            c.ok("0.16" in card,
                 "and it is stated against what one sample alone scores")
            c.ok(page.locator(".cohort-meter-fill").count() == 1,
                 "the stability meter is drawn")
            # R-bar was removed on 2026-08-05: median 0.9991 over all 212 real
            # cohorts and no lower at k=2 than at k=30, so it graded nothing
            # while looking like a grade. Stability is the only headline now.
            c.ok("GROUP TIGHTNESS" not in card.upper() and "R̄" not in card,
                 "and no group-tightness figure sits beside it")

            page.locator(".cohort-members-summary").click()
            page.wait_for_timeout(700)
            rows = page.locator(".member-list label")
            c.ok(rows.count() >= 2, f"the member list opens: {rows.count()} rows")
            c.ok(re.search(r"0\.\d{4}", rows.first.inner_text()) is not None,
                 "each member shows its leave-one-out cosine")
            shot(page, "03-members")

            # ---- 4. a pooled search ----------------------------------------
            print("\n=== 4. pooling and searching ===")
            secs = run_cohort_search(page)
            msg = banner(page)
            c.note(f"the pooled search took {secs:.1f}s")
            c.ok(secs < 30, f"a pooled query costs one memmap pass ({secs:.1f}s)")
            c.ok("pooled mean" in msg,
                 f"the banner names the path that answered: {msg[:90]!r}")
            c.ok("pooled samples" in msg, "and says how many samples went in")
            c.ok(STUDY in msg, "and names the cohort rather than one animal")

            nodes = page.evaluate(NODES_JS) or {}
            c.ok(nodes.get("query", 0) == 1,
                 f"one pooled query node, saw {nodes.get('query')}")
            c.ok(nodes.get("gsm", 0) >= 5, f"the hits are drawn: {nodes.get('gsm')}")
            legend = page.locator("#graph-legend").inner_text()
            c.ok("Pooled cohort query" in legend,
                 f"the legend says a cohort is the query: {legend[:60]!r}")
            subtitle = page.locator("#canvas-subtitle").inner_text()
            c.ok("Pooled OSDR cohort" in subtitle,
                 f"so does the subtitle: {subtitle[:60]!r}")
            shot(page, "04-pooled-search")

            # ---- 5. the inspector ------------------------------------------
            print("\n=== 5. the inspector describes a group ===")
            pos = page.evaluate(QUERY_NODE_JS)
            if c.ok(pos is not None, "the query node is locatable"):
                page.mouse.click(pos["x"], pos["y"])
                page.wait_for_timeout(1500)
            details = page.locator("#details-panel").inner_text()
            c.ok("POOLED OSDR COHORT" in details.upper(),
                 "the inspector opens it as a cohort, not as one blank sample")
            c.ok("Grouped by" in details, "it states the definition that made it")
            c.ok("Samples pooled" in details, "and the size")
            c.ok("POOLED MEMBERS" in details.upper(),
                 "and lists every member by name")
            c.ok("How this query was built" in details,
                 "and explains the estimator")
            shot(page, "05-inspector")

            # ---- 6. excluding a member -------------------------------------
            print("\n=== 6. excluding a member changes the numbers ===")
            before = page.locator("#cohort-card").inner_text().split("\n")[0]
            page.locator(".member-list input[type=checkbox]").first.uncheck()
            page.wait_for_timeout(1800)
            after = page.locator("#cohort-card").inner_text().split("\n")[0]
            c.ok(before != after,
                 f"the card restates the pooled size: {before!r} -> {after!r}")
            c.ok("excluded" in page.locator("#cohort-members-summary").inner_text(),
                 "and the disclosure says one was excluded")
            page.locator(".member-list input[type=checkbox]").first.check()
            page.wait_for_timeout(1500)

            # ---- 7. two arms -----------------------------------------------
            print("\n=== 7. comparing two arms ===")
            compare_hint = page.locator("#cohort-compare-hint").inner_text()
            c.ok("Not a difference vector" in compare_hint,
                 "the comparison says what it is not")
            choose(page, "cohort-compare-dropdown", "differs by")
            chosen = page.locator("#cohort-compare-dropdown").inner_text()
            c.ok("differs by" in chosen,
                 f"only one-facet-apart siblings are offered: {chosen!r}")

            run_cohort_search(page)
            msg = banner(page)
            c.ok("Two pooled queries" in msg,
                 f"the banner reports two queries: {msg[:100]!r}")
            c.ok("Jaccard overlap" in msg, "and quantifies their overlap")
            c.ok("differing by" in msg, "and names the facet they differ in")

            nodes = page.evaluate(NODES_JS) or {}
            c.ok(nodes.get("query", 0) == 1 and nodes.get("query2", 0) == 1,
                 f"two query nodes are drawn: {nodes}")
            c.ok(nodes.get("gse", 0) == 0,
                 "the comparison draws no GSE column")
            # The second regression: the strip above the plot kept advertising a
            # GSE node the comparison does not draw.
            legend = page.locator("#graph-legend").inner_text()
            c.ok("GSE study" not in legend,
                 f"and the legend stops advertising one: {legend[:80]!r}")
            c.ok("Pooled cohort queries (2)" in legend,
                 "the legend names two pooled queries")
            c.ok("which cohort retrieved it" in legend,
                 "and explains what the colours mean")
            subtitle = page.locator("#canvas-subtitle").inner_text()
            c.ok("Two pooled cohorts" in subtitle, "so does the subtitle")
            shot(page, "06-comparison")

            # ---- 8. the map draws BOTH cohorts ------------------------------
            # Section 7 left a comparison in the store, so this is the two-arm
            # case. It used to draw cohort A alone and say nothing about the
            # other, which is the failure this section exists to prevent.
            print("\n=== 8. the map draws both cohorts ===")
            # Read off the retrieval view before the router destroys it.
            n_shared = shared_count(page)
            c.ok(n_shared >= 0, f"the banner reports a shared count: {n_shared}")
            c.ok(page.locator("#see-on-map").is_visible(),
                 "the map is offered once there is something to show")
            offer = page.locator("#see-on-map").inner_text()
            c.ok(offer_count(offer) > topk_value(page),
                 f"the offer counts both cohorts' hits: {offer!r}")
            page.locator("#see-on-map").click()
            wait_for_map(page)
            drawn = page.evaluate(MAP_QUERY_JS) or {}
            c.ok(drawn.get("query", 0) >= 4,
                 f"both cohorts' members are drawn: {drawn}")
            c.ok(drawn.get("halo", 0) == drawn.get("query", 0),
                 "each member gets its halo")

            ov = page.evaluate(MAP_OVERLAY_JS) or {}
            members, hits = ov.get("members", []), ov.get("hits", [])
            c.ok(len(members) == 2, f"two member traces, one per cohort: {len(members)}")
            c.ok(len({m["colour"] for m in members}) == 2,
                 f"the two cohorts' members differ by hue: "
                 f"{[m['colour'] for m in members]}")
            c.ok(len({m["symbol"] for m in members}) == 1,
                 "and not by symbol, which already means member-versus-hit")
            c.ok(len(hits) == 2, f"two hit traces, one per cohort: {len(hits)}")
            c.ok(len({h["symbol"] for h in hits}) == 2,
                 f"hits differ by ring shape: {[h['symbol'] for h in hits]}")
            c.ok(len({str(h["colour"]).lower() for h in hits}) == 1,
                 "and every ring stays white, which is what keeps it visible "
                 "over any tissue colour")
            c.ok(not any(h["hasText"] for h in hits),
                 "rank numerals are dropped when two rank sets would compete")

            if c.ok(len(hits) == 2, "both hit traces are readable"):
                shared_drawn = set(hits[0]["points"]) & set(hits[1]["points"])
                c.ok(len(shared_drawn) == n_shared,
                     f"a hit both cohorts retrieved carries both marks: "
                     f"{len(shared_drawn)} drawn against {n_shared} reported")

            badges = page.locator(".bm-plot-badges").inner_text().replace("\n", " ")
            c.ok("2" in badges and "cohorts" in badges,
                 f"the badge counts both: {badges[:90]!r}")
            c.ok("retrieved by both" in badges,
                 "and names the number the comparison is about")

            key = page.locator(".bm-retrieval-key").inner_text()
            c.ok(page.locator(".bm-retrieval-swatch").count() == 2,
                 "the rail key carries one swatch per cohort")
            c.ok("ring inside a square" in key,
                 f"and says what a shared hit looks like: {key[:80]!r}")
            shot(page, "07-cohort-on-map")

            # ---- 8b. one tick per cohort -----------------------------------
            print("\n=== 8b. unticking an arm ===")
            ticks = page.locator("#show-retrieval input[type=checkbox]")
            c.ok(ticks.count() == 2,
                 f"the single show-it tick became one per cohort: {ticks.count()}")
            labels = page.locator("#show-retrieval").inner_text()
            c.ok("Show it on the map" not in labels,
                 f"each tick carries its cohort's own name: {labels[:70]!r}")
            ticks.nth(1).uncheck()
            page.wait_for_timeout(2500)
            solo = page.evaluate(MAP_OVERLAY_JS) or {}
            c.ok(len(solo.get("members", [])) == 1,
                 "unticking an arm removes it from the map")
            c.ok(any(h["hasText"] for h in solo.get("hits", [])),
                 "and the rank numerals come back, with one list to number")
            shot(page, "08-one-arm")
            ticks.nth(1).check()
            page.wait_for_timeout(2500)
            c.ok(len((page.evaluate(MAP_OVERLAY_JS) or {}).get("members", [])) == 2,
                 "and reticking brings it back")

            # ---- 9. several cohorts, several depths -------------------------
            print("\n=== 9. other studies, cohorts and depths ===")

            # The regression this sweep found: walking back from the map and
            # clicking Cohort straight away opened the cohort panel and then
            # closed it again. The router was repainting the whole view on top
            # of a correct answer, so the click was thrown away with it. No
            # settle here on purpose - the settle is what used to hide it.
            page.goto(f"{base}/")
            page.wait_for_selector(".sample-preview", timeout=60_000)
            page.locator("#mode-tab-cohort").click()
            page.wait_for_timeout(2500)
            c.ok("is-active" in (page.locator("#mode-tab-cohort")
                                 .get_attribute("class") or ""),
                 "arriving from the map and clicking Cohort stays on Cohort")
            c.ok(page.locator("#cohort-search-button").is_visible(),
                 "and the Search button comes with it")

            for study, topk in SWEEP:
                page.goto(f"{base}/")
                page.wait_for_selector(".sample-preview", timeout=60_000)
                open_mode(page, "cohort")
                choose(page, "study-dropdown", study, exact=True)
                set_topk(page, topk)
                label = page.locator("#cohort-dropdown").inner_text()
                secs = run_cohort_search(page)
                msg = banner(page)
                c.ok("pooled mean" in msg,
                     f"{study} at k={topk} answered from the pooled path "
                     f"({secs:.1f}s): {label.strip()[:44]!r}")
                nodes = page.evaluate(NODES_JS) or {}
                c.ok(nodes.get("gsm", 0) == topk,
                     f"and drew exactly {topk} hits, saw {nodes.get('gsm')}")
                page.locator("#see-on-map").click()
                wait_for_map(page)
                drawn = page.evaluate(MAP_QUERY_JS) or {}
                c.ok(drawn.get("query", 0) >= 2,
                     f"and the whole cohort reached the map: {drawn}")
            shot(page, "09-sweep")

            # ---- 10. nothing broke on the way out --------------------------
            print("\n=== 10. console ===")
            noise = [e for e in console_errors
                     if "favicon" not in e.lower()
                     and "ResizeObserver" not in e]
            c.ok(not noise, f"no console errors ({len(noise)}): {noise[:2]}")

            browser.close()
    finally:
        server.terminate()
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()

    print(f"\nscreenshots in {SHOTS}")
    if c.failures:
        print(f"\n{len(c.failures)} of {c.ran} FAILED:")
        for f in c.failures:
            print("  - " + f)
        return 1
    print(f"\nall {c.ran} cohort checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
