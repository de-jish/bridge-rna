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
CHROME = os.environ.get("MANIFOLD_CHROME")
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

# Whether the stability panel actually contains what it is rendering. Measured
# against the *content* box and the scroll box, because the border box is what
# the first version of this check used and it is 20px of padding too generous -
# see the comment at the assertion.
PANEL_FIT_JS = """() => {
  const p = document.querySelector('#stability-panel');
  if (!p) return null;
  const blocks = [...p.querySelectorAll('.stability-cohort')];
  if (!blocks.length) return null;
  const padBottom = parseFloat(getComputedStyle(p).paddingBottom);
  const contentBottom = p.getBoundingClientRect().bottom - padBottom;
  const last = blocks[blocks.length - 1].getBoundingClientRect();
  const boxes = blocks.map(b => b.getBoundingClientRect());
  const near = (xs) => Math.max(...xs) - Math.min(...xs) <= 1;
  const valueTops = [...p.querySelectorAll('.stability-value')]
    .map(n => +n.getBoundingClientRect().top.toFixed(1));
  return {
    scrollH: p.scrollHeight,
    clientH: p.clientHeight,
    overflow: p.scrollHeight - p.clientHeight,
    lastBlockOverrun: +(last.bottom - contentBottom).toFixed(1),
    tops: boxes.map(b => +b.top.toFixed(1)),
    heights: boxes.map(b => +b.height.toFixed(1)),
    sameTop: near(boxes.map(b => b.top)),
    sameHeight: near(boxes.map(b => b.height)),
    valueTops,
    valuesShareABaseline: valueTops.length > 1 && near(valueTops),
  };
}"""

# The same panel below 1180px, where the app grid collapses to one column and
# the document scrolls instead of the panels.
NARROW_FIT_JS = """() => {
  const p = document.querySelector('#stability-panel');
  const d = document.querySelector('#details-panel');
  if (!p || !d) return null;
  const boxes = [...p.querySelectorAll('.stability-cohort')]
    .map(b => b.getBoundingClientRect());
  const near = xs => xs.length < 2 ? true
    : Math.max(...xs) - Math.min(...xs) <= 1;
  return {
    detailsH: +d.getBoundingClientRect().height.toFixed(1),
    detailsContent: d.scrollHeight,
    detailsHidden: d.scrollHeight - d.clientHeight,
    armH: boxes.map(b => +b.height.toFixed(1)),
    sameHeight: near(boxes.map(b => b.height)),
    overflowX: p.scrollWidth - p.clientWidth > 1,
  };
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


NEIGHBORHOOD_EVIDENCE_JS = """() => {
  const gd = document.querySelector('#manifold-graph .js-plotly-plot');
  const raw = (gd && gd.data || []).filter(
    t => t.name === '512-D evidence neighbor');
  const full = (gd && gd._fullData || []).filter(
    t => t.name === '512-D evidence neighbor');
  return {
    sourceTraces: raw.length,
    renderedTraces: full.length,
    marks: full.length === 1 && full[0].x ? full[0].x.length : 0,
  };
}"""


STABLE_MAP_RANGE_JS = """() => {
  const gd = document.querySelector('#manifold-graph .js-plotly-plot');
  const x = gd && gd._fullLayout && gd._fullLayout.xaxis
    && gd._fullLayout.xaxis.range;
  const y = gd && gd._fullLayout && gd._fullLayout.yaxis
    && gd._fullLayout.yaxis.range;
  if (!x || !y || x.length !== 2 || y.length !== 2) return null;
  const ranges = {x: Array.from(x), y: Array.from(y)};
  const value = JSON.stringify(ranges);
  const now = performance.now();
  const prior = window.__bmCohortRangeStability;
  if (!prior || prior.value !== value) {
    window.__bmCohortRangeStability = {value, since: now};
    return null;
  }
  return now - prior.since >= 750 ? ranges : null;
}"""


def _figure_response_for(changed_prop: str):
    """Match the completed map-figure callback for one Dash interaction."""
    def matches(response) -> bool:
        if (response.request.method != "POST"
                or "/_dash-update-component" not in response.url):
            return False
        try:
            payload = response.request.post_data_json
        except Exception:
            return False
        return ("manifold-graph.figure" in str(payload.get("output", ""))
                and changed_prop in payload.get("changedPropIds", []))

    return matches


def _response_evidence_gsms(response) -> list[str]:
    """Read the exact evidence identity from the response being awaited."""
    figure = response.json()["response"]["manifold-graph"]["figure"]
    traces = [trace for trace in figure.get("data", [])
              if trace.get("name") == "512-D evidence neighbor"]
    if len(traces) != 1:
        raise AssertionError(
            f"figure response contained {len(traces)} evidence traces")
    return [str(row[2] or "") for row in traces[0].get("customdata", [])]


def _wait_for_evidence_gsms(page, expected: list[str]) -> dict:
    """Wait until Plotly rendered the response's exact 250-sample identity."""
    page.wait_for_function(
        """expected => {
          const gd = document.querySelector('#manifold-graph .js-plotly-plot');
          const raw = (gd && gd.data || []).filter(
            t => t.name === '512-D evidence neighbor');
          const full = (gd && gd._fullData || []).filter(
            t => t.name === '512-D evidence neighbor');
          const rows = raw.length === 1 && Array.isArray(raw[0].customdata)
            ? raw[0].customdata : [];
          return raw.length === 1 && full.length === 1
            && full[0].x && full[0].x.length === 250
            && rows.length === expected.length
            && rows.every((row, index) => String(row[2] || '') === expected[index]);
        }""", arg=expected, timeout=90_000)
    return page.evaluate(NEIGHBORHOOD_EVIDENCE_JS)


def _stable_map_span(page) -> float:
    """Read the x span only after Plotly's responsive relayout has settled."""
    page.evaluate("() => { delete window.__bmCohortRangeStability; }")
    ranges = page.wait_for_function(
        STABLE_MAP_RANGE_JS, timeout=90_000).json_value()
    return float(ranges["x"][1] - ranges["x"][0])


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


#: Which pass of `--loops` is running. Screenshots are namespaced by it, so a
#: failure on the second pass does not land under the first pass's filename.
_RUN = 1


def shot(page, name: str) -> None:
    page.wait_for_timeout(250)
    suffix = "" if _RUN == 1 else f"-run{_RUN}"
    page.screenshot(path=str(SHOTS / f"{name}{suffix}.png"))


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


def stability_values(page) -> list[float]:
    """Every measured stability on screen, read off the marks that carry them.

    By class rather than by regex over the panel's text. The number lost its
    in-line "RESULT STABILITY" label when the heading took over that job, and a
    text pattern anchored on the label would have gone quietly empty rather than
    failing loudly - which is exactly how a two-arm comparison could come to
    report one number without anything noticing.
    """
    el = page.locator("#stability-panel .stability-value")
    return [float(el.nth(i).inner_text().strip()) for i in range(el.count())]


def open_stability(page) -> None:
    """Read the optional diagnostics using the native keyboard disclosure."""
    disclosure = page.locator("#stability-panel details.stability-details")
    if disclosure.get_attribute("open") is None:
        summary = disclosure.locator("summary").first
        summary.focus()
        summary.press("Enter")
    page.locator("#stability-panel .stability-value").first.wait_for(state="visible")


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


def run_checks(page, c: "Checks", base: str, console_errors: list[str],
               network_errors: list[str], dash_changes: list[str]) -> None:
    """Every check, against one page. Extracted so `--loops` can repeat it.

    One pass proves the feature works. Repeating it is what catches the class of
    bug this suite was built around: both regressions it exists for were
    callbacks firing when they should not have, and both depended on what was on
    screen a moment earlier. State that survives one iteration into the next - a
    stale hits-store, a stability panel that never cleared, a search that ran at
    remount - is invisible to a run that only ever starts from a cold load.
    """
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
    c.ok(not msg.strip(), "the status area stays empty before a search")
    c.ok("Select an OSDR sample, then run a search." in
         page.locator("#network-graph").inner_text(),
         "the empty network gives the next step")
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

    # ---- 3. the rail before a search -------------------------------
    print("\n=== 3. the rail says only what it knows yet ===")
    card = page.locator("#cohort-card").inner_text()
    c.ok("samples pooled into one query" in card,
         "the card states the pooled size")
    # Result stability used to be quoted here, read out of a curve by cohort
    # size. It is a population average, and printing it beside one cohort's
    # name got it read as a property of that cohort: measured live, a cohort
    # of 7 scores 0.316 and one of 6 scores 0.849, and the curve told both
    # of them 0.72. It is measured during the search now and reported on the
    # right afterwards. See docs/design-notes.md#live-stability.
    c.ok("STABILITY" not in card.upper(),
         f"and nothing about result stability: {card[:70]!r}")
    c.ok(page.locator("#cohort-card .cohort-meter").count() == 0,
         "no meter on the rail, because there is no number for it to fill")
    c.ok("0.16" not in card,
         "and no precomputed single-sample baseline either")
    # R-bar was removed on 2026-08-05: median 0.9991 over all 212 real
    # cohorts and no lower at k=2 than at k=30, so it graded nothing
    # while looking like a grade.
    c.ok("GROUP TIGHTNESS" not in card.upper() and "R̄" not in card,
         "and no group-tightness figure either")
    c.ok(not page.locator("#stability-panel").is_visible(),
         "and the stability panel is not on screen before a search")

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

    # ---- 4b. stability, measured on the query that just ran ---------
    #
    # The number the rail stopped quoting. It is the same statistic
    # validate_cohorts.py measures over all 212 cohorts, computed here for
    # this cohort alone, at the depth on screen, in the pass that fetched
    # the hits. It cannot exist before the search, which is why it is here
    # and not under the picker.
    print("\n=== 4b. stability is measured, not looked up ===")
    c.ok(page.locator("#stability-panel").is_visible(),
         "the stability panel appears once there is a result to describe")
    c.ok(page.locator("#stability-panel details").get_attribute("open") is None,
         "optional stability diagnostics start collapsed")
    c.ok(not page.locator("#stability-panel .stability-value").first.is_visible(),
         "the detailed measurement does not compete with the results initially")
    open_stability(page)
    c.ok(page.locator("#stability-panel .stability-value").first.is_visible(),
         "Enter on the summary opens the complete diagnostics")
    # Upper-cased throughout, because `inner_text` reports what is rendered and
    # several labels carry `text-transform: uppercase`. Matching the source
    # casing here passed nothing and would have hidden the panel being absent.
    panel = page.locator("#stability-panel").inner_text().upper()
    c.ok("MEASURED ON THIS QUERY" in panel,
         f"and says the number is a measurement: {panel[:70]!r}")
    values = stability_values(page)
    c.ok(len(values) == 1, f"it quotes one number: {values}")
    if values:
        c.ok(0.0 <= values[0] <= 1.0,
             f"which is a share, not a score: {values[0]}")
        c.note(f"measured stability {values[0]:.2f} at top-{topk_value(page)}")
    c.ok(page.locator("#stability-panel .cohort-meter").count() == 1,
         "the meter moved here with the number it fills")
    depth = topk_value(page)
    c.ok(f"THESE {depth} HITS" in panel,
         f"and it names the depth it measured at (top-{depth}): {panel[:140]!r}")
    c.ok("ANY ONE POOLED SAMPLE IS DROPPED" in panel,
         "and what dropping one of them did")
    # The scale, measured on this cohort in the same pass, which is what
    # replaced the fixed 0.16 constant the rail used to quote.
    c.ok("OVERLAPS ANOTHER BY" in panel or "AGREE ON A HIT ALONE" in panel,
         f"and what one sample alone would have overlapped: {panel[:170]!r}")
    # The heading names the statistic; a per-block label would repeat it twice
    # over and push cohort B's measurement below the fold on a comparison.
    c.ok(panel.count("RESULT STABILITY") == 1,
         f"and the statistic is named once: {panel.count('RESULT STABILITY')}")
    c.ok(page.locator("#stability-panel").count() == 1
         and page.locator("#details-panel").count() == 1,
         "the panel sits beside the inspector rather than replacing it")
    shot(page, "04b-stability")

    # ---- 5. the inspector ------------------------------------------
    print("\n=== 5. the inspector describes a group ===")
    pos = page.evaluate(QUERY_NODE_JS)
    if c.ok(pos is not None, "the query node is locatable"):
        page.mouse.click(pos["x"], pos["y"])
        page.wait_for_timeout(1500)
    members = page.locator("#details-panel summary", has_text="Pooled members")
    c.ok(members.count() == 1, "executed query members have one disclosure")
    if members.count():
        c.ok(members.locator("..").get_attribute("open") is None,
             "the executed member list starts collapsed")
        members.focus()
        members.press("Enter")
    details = page.locator("#details-panel").inner_text()
    c.ok("POOLED OSDR COHORT" in details.upper(),
         "the inspector opens it as a cohort, not as one blank sample")
    c.ok("Grouped by" in details, "it states the definition that made it")
    c.ok("Samples pooled" in details, "and the size")
    # One number, one home. The Definition section carried a "Result
    # stability" row too, directly under the panel that now reports it.
    c.ok("Result stability" not in details,
         "and does not restate the stability the panel above it carries")
    c.ok(page.locator("#stability-panel").is_visible(),
         "which is still on screen with a hit node open")
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
    c.ok("top-k the two share" in compare_hint,
         f"the comparison says what it reports: {compare_hint[:70]!r}")
    choose(page, "cohort-compare-dropdown", "differs by")
    chosen = page.locator("#cohort-compare-dropdown").inner_text()
    c.ok("differs by" in chosen,
         f"only one-facet-apart siblings are offered: {chosen!r}")

    # A comparison runs TWO independent pooled queries, and the rail
    # used to describe the size of only one of them while giving the
    # other a color in the figure and on the map. Cohort B can easily be
    # the smaller and shakier arm, which is exactly what decides how much
    # of the overlap number to believe.
    card_a = page.locator("#cohort-card").inner_text()
    card_b = page.locator("#cohort-compare-card").inner_text()
    c.ok("COHORT A" in card_a.upper(),
         f"the selected cohort names its arm: {card_a[:40]!r}")
    c.ok("COHORT B" in card_b.upper(),
         f"and the compared one names its own: {card_b[:40]!r}")
    c.ok("differs by" in card_b,
         "the second card states the one facet they differ in")
    for name, card in (("A", card_a), ("B", card_b)):
        c.ok("samples pooled into one query" in card,
             f"cohort {name}'s card states its pooled size")
        c.ok("STABILITY" not in card.upper(),
             f"and cohort {name}'s card still predicts nothing about the result")
    c.ok(page.locator(".cohort-card.is-a").count() == 1
         and page.locator(".cohort-card.is-b").count() == 1,
         "each card carries its own role color")
    # Arming the comparison must not leave the previous single-cohort
    # measurement on screen describing a query nobody ran.
    c.ok(page.locator("#stability-panel .stability-cohort").count() == 1,
         "and the panel still describes the one query that has run")

    run_cohort_search(page)
    msg = banner(page)
    c.ok("Two pooled queries" in msg,
         f"the banner reports two queries: {msg[:100]!r}")
    c.ok("Jaccard overlap" in msg, "and quantifies their overlap")
    c.ok("differing by" in msg, "and names the facet they differ in")

    # Two pooled queries, two measurements. STABILITY_BY_K was a function
    # of size, so an overlap of 0.25 between two arms of twelve is not the
    # same finding as 0.25 between one of twelve and one of two - and now
    # neither arm's number is a function of size at all.
    stab = page.locator("#stability-panel")
    open_stability(page)
    c.ok(stab.locator(".stability-cohort").count() == 2,
         f"both arms are measured: {stab.locator('.stability-cohort').count()}")
    panel = stab.inner_text().upper()
    c.ok("COHORT A" in panel and "COHORT B" in panel,
         f"each measurement names its arm: {panel[:80]!r}")
    values = stability_values(page)
    c.ok(len(values) == 2, f"two numbers, one per arm: {values}")
    # Both have to be readable without scrolling, or measuring the second arm
    # bought nothing.
    #
    # This is the second version of this check, and the first one passed while
    # cohort B's last row was clipped at every viewport the app is used at. It
    # compared the last block's bottom against `bounding_box()`, which is the
    # *border* box, so a block was allowed to run through the panel's own 20px
    # bottom padding and stop 1px short of its border. At 1600x1000 the last
    # block ended 9.9px below the content box and 10.1px above the border box,
    # which is to say the assertion was satisfied by exactly the margin that hid
    # the failure. It also never compared scrollHeight against clientHeight, so
    # 11px of unreachable content was invisible to it.
    #
    # `PANEL_FIT_JS` measures the content box and the scroll box instead. Both
    # matter: a panel can scroll internally while every block's border box still
    # sits inside it.
    fit = page.evaluate(PANEL_FIT_JS)
    c.ok(bool(fit) and fit["overflow"] <= 1,
         f"the panel does not scroll internally: {fit and fit['scrollH']}px of "
         f"content in {fit and fit['clientH']}px")
    c.ok(bool(fit) and fit["lastBlockOverrun"] <= 1,
         f"and the second arm's last row is inside the panel's content box, "
         f"not its padding: overruns by {fit and fit['lastBlockOverrun']}px")
    # An even split is a measurement, not an intention. The two arms have to
    # start on the same line and be the same height, or one of them is again the
    # footnote to the other.
    c.ok(bool(fit) and fit["sameTop"] and fit["sameHeight"],
         f"the two arms are an even split: tops {fit and fit['tops']}, "
         f"heights {fit and fit['heights']}")
    c.ok(bool(fit) and fit["valuesShareABaseline"],
         f"and the two numbers sit on one baseline, which is the whole reason "
         f"to measure two: {fit and fit['valueTops']}")
    c.ok(stab.locator(".stability-cohort.is-a").count() == 1
         and stab.locator(".stability-cohort.is-b").count() == 1,
         "and each carries the role color its card and its glyphs carry")
    # The facet moved from cohort B's role line into the header when the two
    # arms went side by side: it describes the pair, and hanging it under B's
    # letter started B's name a line below A's.
    c.ok("DIFFER BY" in panel and panel.count("DIFFER BY") == 1,
         "with the facet the pair differs in stated once, above both arms")
    c.ok(page.locator("#stability-panel .stability-pair.is-pair").count() == 1,
         "and the two arms are laid out as one even pair rather than stacked")

    # The same panel at the widths where the app grid collapses to one column
    # and the document scrolls. Two things are checked that only exist there.
    #
    # The details panel must size to its content, not to a leftover. `flex: 1 1
    # 0` is right in the fixed-height desktop column and wrong the moment the
    # column's height comes from its contents, because a zero-basis item adds
    # nothing to that height - which pinned this panel to its 120px floor with
    # 372px hidden. That regression shipped in the same change as the even split
    # and was invisible to every desktop measurement of it.
    #
    # And 320px is the width where the "moves it most" label and its score stop
    # fitting on one line in a column, so the label has to wrap rather than run
    # into the other arm.
    was = page.viewport_size
    for width in (900, 390, 320):
        page.set_viewport_size({"width": width, "height": 950})
        page.wait_for_timeout(700)
        narrow = page.evaluate(NARROW_FIT_JS)
        c.ok(bool(narrow) and narrow["detailsHidden"] <= 1,
             f"at {width}px the page scrolls and the inspector is not clipped: "
             f"details panel {narrow and narrow['detailsH']}px holds "
             f"{narrow and narrow['detailsContent']}px")
        c.ok(bool(narrow) and narrow["sameHeight"] and not narrow["overflowX"],
             f"and the two arms are still an even split at {width}px: "
             f"{narrow and narrow['armH']}")
    page.set_viewport_size(was)
    page.wait_for_timeout(700)
    shot(page, "07b-two-measurements")

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
         "and explains what the colors mean")
    subtitle = page.locator("#canvas-subtitle").inner_text()
    c.ok("Two pooled cohorts" in subtitle, "so does the subtitle")

    # The inspector opens on cohort A. Without the letter it reads as
    # *the* pooled query rather than as one of two, with nothing saying
    # the other star on the canvas leads to its twin.
    # .details-kicker is uppercased by CSS, so read it case-insensitively
    # the way this file's other kicker checks already do.
    details = page.locator("#details-panel").inner_text()
    c.ok("COHORT A" in details.upper(),
         f"the inspector says which arm it opened on: {details[:60]!r}")
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
    c.ok(offer == "View results on map",
         f"the map link does not double-count shared comparison samples: {offer!r}")
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

    # The key sits on the plot now, beside the marks it decodes, and it
    # is grouped by ROLE rather than by cohort - so the two member rows
    # are adjacent and differ only in hue, and the two hit rows are
    # adjacent and differ only in shape. That layout is the explanation;
    # no sentence has to assert the rule.
    key = page.locator("#legend-retrieval").inner_text()
    c.ok("POOLED MEMBERS" in key.upper() and "RETRIEVED HITS" in key.upper(),
         f"the key groups the marks by role: {key[:60]!r}")
    c.ok("retrieved by both" in key,
         f"and rows the doubled mark: {key[:90]!r}")
    # Exact class tokens, not substrings: "is-hit-b" is a prefix of
    # "is-hit-both", so a substring test counts the doubled mark as a
    # cohort-B row and reports three hit rows as two.
    glyphs = page.eval_on_selector_all(
        "#legend-retrieval .bm-key-glyph",
        "els => els.map(e => [...e.classList])")
    shapes = [c for g in glyphs for c in g if c.startswith("is-")]
    c.ok(shapes.count("is-star") == 2,
         f"two member rows, same shape: {shapes}")
    c.ok(shapes.count("is-hit-a") == 1 and shapes.count("is-hit-b") == 1,
         f"two hit rows, different shapes: {shapes}")
    c.ok(shapes == ["is-star", "is-star", "is-hit-a", "is-hit-b",
                    "is-hit-both"],
         f"grouped by role, so each pair varies in one channel: {shapes}")
    fills = page.eval_on_selector_all(
        "#legend-retrieval .bm-key-glyph-fill",
        "els => els.map(e => getComputedStyle(e).backgroundColor)")
    c.ok(len(set(fills)) == 2,
         f"the member rows carry the two cohort hues: {fills}")
    names = page.eval_on_selector_all(
        "#legend-retrieval .bm-key-label",
        "els => els.map(e => e.textContent)")
    c.ok(names.count(names[0]) == 2,
         f"each cohort is named in both of its rows: {names}")
    shot(page, "07-cohort-on-map")

    # ---- 8a. each comparison arm owns its evidence neighborhood -----
    #
    # Wait for the *figure response caused by the arm input*, then require the
    # browser's raw source trace to carry that response's exact GSM ordering.
    # A generic "one trace exists" wait can succeed on arm A while arm B is
    # still in flight; tying both sides to the response prevents that stale
    # Plotly read from turning a label-only switch into a passing check.
    print("\n=== 8a. comparison evidence switches independently ===")
    requested_before = page.evaluate(MAP_OVERLAY_JS) or {}
    runs_before_neighborhood = {
        prop: dash_changes.count(prop)
        for prop in ("cohort-search-button.n_clicks", "search-button.n_clicks")
    }

    with page.expect_response(
            _figure_response_for("neighborhood-open-store.data"),
            timeout=90_000) as response_info:
        page.locator("#frame-retrieval").click()
    page.locator("#neighborhood-drawer").wait_for(state="visible", timeout=30_000)
    evidence_gsms_a = _response_evidence_gsms(response_info.value)
    evidence_a = _wait_for_evidence_gsms(page, evidence_gsms_a)
    page.wait_for_selector("#neighborhood-arm input[value='b']", timeout=30_000)
    arm_a = page.locator(
        "#neighborhood-arm .dash-options-list-option:has(input[value='a'])")
    arm_b = page.locator(
        "#neighborhood-arm .dash-options-list-option:has(input[value='b'])")
    arm_label_a = arm_a.inner_text().strip()
    arm_label_b = arm_b.inner_text().strip()
    heading_a = page.locator("#neighborhood-heading").inner_text().strip()
    meta_a = page.locator("#neighborhood-meta").inner_text().strip()
    c.ok(arm_label_a.startswith("A · ") and arm_label_b.startswith("B · ")
         and arm_label_a != arm_label_b,
         f"the explorer names two distinct comparison arms: "
         f"{arm_label_a!r}, {arm_label_b!r}")
    c.ok(page.get_by_role("radio", name=re.compile(r"^A · ")).is_checked(),
         "the comparison explorer opens on arm A")
    c.ok(f"{depth} requested hits" in meta_a
         and "exact top-250 cosine neighborhood in 512-D" in meta_a,
         f"arm A states its requested depth and exact evidence context: {meta_a!r}")
    c.ok(evidence_a["sourceTraces"] == 1
         and evidence_a["renderedTraces"] == 1
         and evidence_a["marks"] == 250
         and len(evidence_gsms_a) == 250,
         "arm A owns exactly one complete 250-point evidence trace "
         f"({evidence_a['sourceTraces']} source, "
         f"{evidence_a['renderedTraces']} rendered, {evidence_a['marks']} marks)")

    page.locator(
        "#neighborhood-tab .dash-options-list-option:has(input[value='samples'])"
    ).click()
    page.wait_for_function(
        """expected => {
          const rows = [...document.querySelectorAll(
            '.bm-neighborhood-sample .bm-neighborhood-row-primary')]
            .map(row => row.textContent.trim());
          return rows.length === expected.length
            && rows.every((gsm, index) => gsm === expected[index]);
        }""", arg=evidence_gsms_a, timeout=30_000)
    rows_a = page.locator(
        ".bm-neighborhood-sample .bm-neighborhood-row-primary").all_inner_texts()

    with page.expect_response(
            _figure_response_for("neighborhood-arm.value"),
            timeout=90_000) as response_info:
        arm_b.click()
    evidence_gsms_b = _response_evidence_gsms(response_info.value)
    evidence_b = _wait_for_evidence_gsms(page, evidence_gsms_b)
    page.wait_for_function(
        """state => {
          const heading = document.querySelector('#neighborhood-heading');
          const rows = [...document.querySelectorAll(
            '.bm-neighborhood-sample .bm-neighborhood-row-primary')]
            .map(row => row.textContent.trim());
          return heading && heading.textContent.trim() !== state.heading
            && rows.length === state.gsms.length
            && rows.every((gsm, index) => gsm === state.gsms[index]);
        }""", arg={"heading": heading_a, "gsms": evidence_gsms_b},
        timeout=30_000)
    heading_b = page.locator("#neighborhood-heading").inner_text().strip()
    meta_b = page.locator("#neighborhood-meta").inner_text().strip()
    rows_b = page.locator(
        ".bm-neighborhood-sample .bm-neighborhood-row-primary").all_inner_texts()
    c.ok(page.get_by_role("radio", name=re.compile(r"^B · ")).is_checked(),
         "switching A to B updates the selected arm's radio state")
    c.ok(f"{depth} requested hits" in meta_b
         and "exact top-250 cosine neighborhood in 512-D" in meta_b,
         f"arm B states its requested depth and exact evidence context: {meta_b!r}")
    c.ok(heading_a != heading_b,
         f"switching A to B changes the active arm heading: "
         f"{heading_a!r} -> {heading_b!r}")
    c.ok(evidence_b["sourceTraces"] == 1
         and evidence_b["renderedTraces"] == 1
         and evidence_b["marks"] == 250
         and len(evidence_gsms_b) == 250,
         "arm B replaces A with one complete 250-point evidence trace "
         f"({evidence_b['sourceTraces']} source, "
         f"{evidence_b['renderedTraces']} rendered, {evidence_b['marks']} marks)")
    c.ok(rows_a == evidence_gsms_a and rows_b == evidence_gsms_b
         and rows_a != rows_b,
         "the complete 250-row Samples list changes with the selected arm")
    c.ok(set(evidence_gsms_a) != set(evidence_gsms_b),
         f"the exact evidence point set changes with the arm "
         f"({len(set(evidence_gsms_a) & set(evidence_gsms_b))} shared)")
    c.ok((page.evaluate(MAP_OVERLAY_JS) or {}) == requested_before,
         "arm switching preserves both cohorts' requested-hit traces")
    c.ok({prop: dash_changes.count(prop) for prop in runs_before_neighborhood}
         == runs_before_neighborhood,
         "opening and switching evidence does not rerun either retrieval path")
    shot(page, "11-neighborhood-comparison")

    with page.expect_response(
            _figure_response_for("neighborhood-open-store.data"),
            timeout=90_000):
        page.get_by_role("button", name="Close neighborhood explorer").click()
    page.locator("#neighborhood-drawer").wait_for(state="hidden", timeout=30_000)
    page.wait_for_function(
        """() => {
          const gd = document.querySelector('#manifold-graph .js-plotly-plot');
          return gd && !(gd._fullData || []).some(t =>
            ['512-D evidence neighbor', 'focused evidence neighbor'].includes(t.name));
        }""", timeout=90_000)
    c.ok((page.evaluate(MAP_OVERLAY_JS) or {}) == requested_before,
         "Close removes comparison evidence but preserves requested-hit traces")

    # The claim that dropping the rank numerals is safe because "the
    # hover says strictly more" was untrue: two traces at one coordinate
    # resolve to one tooltip, so a shared hit named one arm only.
    shared_hover = page.evaluate("""() => {
      const gd = document.querySelector('.js-plotly-plot');
      const hits = gd._fullData.filter(t => t.name === 'retrieved hit');
      if (hits.length !== 2) return null;
      const key = t => t.x.map((x, i) => x.toFixed(4) + ',' + t.y[i].toFixed(4));
      const a = key(hits[0]), b = new Set(key(hits[1]));
      const i = a.findIndex(p => b.has(p));
      return i < 0 ? null : hits[0].customdata[i][1];
    }""")
    if c.ok(shared_hover is not None, "a shared hit is locatable"):
        c.ok(shared_hover.count("512-d rank") == 2,
             f"its hover carries both arms' ranks: {shared_hover[:120]!r}")
        c.ok(shared_hover.count("cosine") == 2,
             "and both cosines, which is what the numerals gave up for")

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
    # This map already holds that a key is read as "what am I looking
    # at" rather than "what exists", which is why the colour legend
    # recounts itself per figure. A key still counting hits for an arm
    # that is not drawn would be the same error by another route.
    key = page.locator("#legend-retrieval").inner_text()
    c.ok("hidden" in key,
         f"the key reports the hidden arm as hidden: {key[:90]!r}")
    c.ok(page.locator(".bm-key-row.is-hidden").count() == 2,
         "in both of its rows, so the hue and the shape it owns are "
         "still named while nothing of it is drawn")
    c.ok("retrieved by both" not in key,
         "and the doubled mark leaves the key, because with one arm "
         "hidden it cannot exist")
    solo_badges = page.locator(".bm-plot-badges").inner_text().replace("\n", " ")
    c.ok("2 cohorts" not in solo_badges,
         f"and the badge stops claiming two: {solo_badges[:80]!r}")
    shot(page, "08-one-arm")

    # The panel used to be shown or hidden on the color items alone, so a state
    # that draws no colored category would have taken the whole key off screen,
    # retrieval and all, while the retrieval was still drawn.
    #
    # That state used to be reached with an OSDR-only color-by and the OSDR
    # layer unticked. All nine OSDR-only fields were removed on 2026-08-11, so
    # it is reached here by unticking *both* corpus layers - which is a stronger
    # version of the same case: the retrieval overlay is drawn outside the layer
    # toggles entirely, so this leaves the map with nothing on it but the two
    # cohorts and no legend row to hang the panel on.
    for i in (0, 1):
        page.locator("#layers input[type=checkbox]").nth(i).uncheck()
    page.wait_for_timeout(3500)
    c.ok(page.locator("#legend").is_visible(),
         "the key survives a state with no colored category on screen")
    c.ok(page.locator("#legend-retrieval .bm-key-row").count() > 0,
         "and still decodes the retrieval that is still drawn")
    c.ok(not page.locator("#legend-color").is_visible(),
         "while the empty color section takes itself off")
    for i in (0, 1):
        page.locator("#layers input[type=checkbox]").nth(i).check()
    page.wait_for_timeout(3000)

    # The ticks are the user's, and only a new retrieval may reset them.
    # This callback also fires on a dimensionality switch, because the
    # frame button is 2-D only, so reasserting the value there would
    # re-tick a hidden arm the moment you looked at it in 3-D.
    page.locator("#dims label", has_text="3D").first.click()
    page.wait_for_timeout(4000)
    c.ok(not ticks.nth(1).is_checked(),
         "switching to 3-D leaves the hidden arm hidden")
    c.ok(len((page.evaluate(MAP_OVERLAY_JS) or {}).get("members", [])) == 1,
         "and 3-D draws the one arm, with no unknown-symbol error")
    # Scatter3d rejects `star`, so a member draws as a diamond there.
    # A key that kept showing a star would be asserting a mark 3-D does
    # not draw, which is worse than the silence it replaced.
    glyphs_3d = [c for g in page.eval_on_selector_all(
        "#legend-retrieval .bm-key-glyph",
        "els => els.map(e => [...e.classList])") for c in g
        if c.startswith("is-")]
    c.ok("is-diamond" in glyphs_3d and "is-star" not in glyphs_3d,
         f"the key follows 3-D's diamond substitution: {glyphs_3d}")
    # 3-D used to discard the OSDR overlay's symbol and its white ring
    # outright, so 2,108 spaceflight samples arrived as plain circles in
    # the same palette hue as the 940,455 they sit among.
    osdr_3d = page.evaluate("""() => {
      const gd = document.querySelector('.js-plotly-plot');
      const t = gd._fullData.find(t => t.customdata && t.customdata.length
                && Array.isArray(t.customdata[0])
                && String(t.customdata[0][0]).includes('|'));
      return t ? {symbol: t.marker.symbol,
                  line: t.marker.line && t.marker.line.width} : null;
    }""")
    if c.ok(osdr_3d is not None, "the OSDR overlay is locatable in 3-D"):
        c.ok(osdr_3d["symbol"] == "diamond",
             f"and keeps its diamond in 3-D: {osdr_3d}")
        c.ok((osdr_3d["line"] or 0) > 0,
             f"and its white ring: {osdr_3d}")
    page.locator("#dims label", has_text="2D").first.click()
    page.wait_for_timeout(4000)

    ticks.nth(1).check()
    page.wait_for_timeout(2500)
    c.ok(len((page.evaluate(MAP_OVERLAY_JS) or {}).get("members", [])) == 2,
         "and reticking brings it back")

    # ---- 8c. framing a comparison, and the one control that undoes it ----
    #
    # Framing a retrieval and framing a find write the same `viewport-store`,
    # so one reset undoes both - which is the whole reason there is one control
    # rather than a button beside each. The find half is checked in
    # `e2e_check.py`; this is the retrieval half, on the payload with the most
    # to lose, because a reset that cleared the wrong store would take two
    # pooled queries and twelve member marks with it.
    print("\n=== 8c. framing a comparison, and the reset that undoes it ===")

    drawn = page.evaluate(MAP_OVERLAY_JS) or {}
    whole = _stable_map_span(page)
    c.ok(not page.locator("#reset-view").is_visible(),
         "the map is unframed until something frames it")
    with page.expect_response(
            _figure_response_for("viewport-store.data"), timeout=90_000):
        page.locator("#frame-retrieval").click()
    page.locator("#neighborhood-drawer").wait_for(
        state="visible", timeout=30_000)
    framed = _stable_map_span(page)
    c.ok(framed < whole * 0.9,
         f"framing a comparison narrows the view ({framed:.2f} of {whole:.2f})")
    c.ok(page.locator("#reset-view").is_visible(),
         "and the reset appears once it has")
    with page.expect_response(
            _figure_response_for("neighborhood-open-store.data"),
            timeout=90_000):
        page.get_by_role("button", name="Close neighborhood explorer").click()
    page.locator("#neighborhood-drawer").wait_for(
        state="hidden", timeout=30_000)
    framed_closed = _stable_map_span(page)
    c.ok(abs(framed_closed - framed) < max(framed * 0.02, 0.001),
         f"closing the drawer preserves the explicit frame ({framed_closed:.2f})")
    with page.expect_response(
            _figure_response_for("viewport-store.data"), timeout=90_000):
        page.locator("#reset-view").click()
    back = _stable_map_span(page)
    c.ok(abs(back - whole) < whole * 0.02,
         f"the reset restores the whole map ({back:.2f})")
    c.ok(not page.locator("#reset-view").is_visible(),
         "and takes itself off again")
    after = page.evaluate(MAP_OVERLAY_JS) or {}
    c.ok(len(after.get("members", [])) == len(drawn.get("members", [])),
         f"both cohorts survive the reset: {len(after.get('members', []))} arms")
    c.ok(len(after.get("hits", [])) == len(drawn.get("hits", [])),
         "and so does every hit they retrieved")
    with page.expect_response(
            _figure_response_for("viewport-store.data"), timeout=90_000):
        page.locator("#frame-retrieval").click()
    page.locator("#neighborhood-drawer").wait_for(
        state="visible", timeout=30_000)
    reframed = _stable_map_span(page)
    c.ok(abs(reframed - framed) < max(framed * 0.02, 0.001),
         f"re-framing gives the same window, not a stale one ({reframed:.2f})")
    with page.expect_response(
            _figure_response_for("neighborhood-open-store.data"),
            timeout=90_000):
        page.get_by_role("button", name="Close neighborhood explorer").click()
    page.locator("#neighborhood-drawer").wait_for(
        state="hidden", timeout=30_000)
    with page.expect_response(
            _figure_response_for("viewport-store.data"), timeout=90_000):
        page.locator("#reset-view").click()
    _stable_map_span(page)

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
        # The measurement follows the slider. The curve it replaced was fixed
        # at top-5 whatever the reader was actually looking at, which is the
        # other half of why it could not describe the list on screen.
        open_stability(page)
        panel = page.locator("#stability-panel").inner_text().upper()
        c.ok(f"THESE {topk} HITS" in panel,
             f"and measured stability over those {topk}: {panel[:90]!r}")
        page.locator("#see-on-map").click()
        wait_for_map(page)
        drawn = page.evaluate(MAP_QUERY_JS) or {}
        c.ok(drawn.get("query", 0) >= 2,
             f"and the whole cohort reached the map: {drawn}")
    shot(page, "09-sweep")

    # A single query has one tick, and it hides the whole overlay. The
    # key ignored it at first, so the plot lost its star and its rings
    # while the key went on counting them - the failure the count rule
    # exists to prevent, in the commonest state the map has.
    print("\n=== 9b. unticking a single query ===")
    ticks = page.locator("#show-retrieval input[type=checkbox]")
    c.ok(ticks.count() == 1,
         f"a single query gets one tick: {ticks.count()}")
    before = page.evaluate(MAP_OVERLAY_JS) or {}
    c.ok(before.get("members") and before.get("hits"),
         "the overlay is drawn to begin with")
    ticks.first.uncheck()
    page.wait_for_timeout(3000)
    after = page.evaluate(MAP_OVERLAY_JS) or {}
    c.ok(not after.get("members") and not after.get("hits"),
         f"unticking takes the whole overlay off the plot: {after}")
    key = page.locator("#legend-retrieval").inner_text()
    c.ok("hidden" in key,
         f"and the key says so rather than counting on: {key[:70]!r}")
    c.ok(page.locator("#legend-retrieval .bm-key-row.is-hidden").count()
         == page.locator("#legend-retrieval .bm-key-row").count(),
         "with every row receded, because none of them is drawn")
    summary = page.locator("#retrieval-summary").inner_text()
    c.ok("Not drawn" in summary,
         f"and the rail agrees with the plot: {summary[:60]!r}")
    ticks.first.check()
    page.wait_for_timeout(2500)
    c.ok(bool((page.evaluate(MAP_OVERLAY_JS) or {}).get("members")),
         "and reticking brings it back")

    # ---- 10. nothing broke on the way out --------------------------
    print("\n=== 10. console ===")
    noise = [e for e in console_errors
             if "favicon" not in e.lower()
             and "ResizeObserver" not in e]
    c.ok(not noise, f"no console errors ({len(noise)}): {noise[:2]}")
    c.ok(not network_errors,
         f"no failed requests or HTTP errors ({len(network_errors)}): "
         f"{network_errors[:2]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8064)
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--loops", type=int, default=1,
                    help="run the whole suite this many times against one "
                         "server, each in a fresh page. A second pass is what "
                         "catches state carried between runs and callbacks that "
                         "fire on remount rather than on a click, which is what "
                         "both of the regressions this file exists for were.")
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
            browser = p.chromium.launch(
                headless=not args.headed, executable_path=CHROME)
            global _RUN
            for run in range(1, max(1, args.loops) + 1):
                _RUN = run
                if args.loops > 1:
                    print(f"\n{'=' * 60}\n  PASS {run} of {args.loops}\n{'=' * 60}",
                          flush=True)
                # A fresh page per pass, but the *same* server. The point of a
                # second pass is to exercise a warm process: caches populated,
                # the cohort metadata already loaded, and every lru_cache in the
                # retrieval path holding whatever the previous pass left in it.
                page = browser.new_page(viewport={"width": 1680, "height": 1010})
                console_errors: list[str] = []
                network_errors: list[str] = []
                dash_changes: list[str] = []
                page.on("console", lambda m: console_errors.append(m.text)
                        if m.type == "error" else None)
                page.on("pageerror", lambda e: console_errors.append(str(e)))

                def record_response(response):
                    if response.status >= 400:
                        network_errors.append(
                            f"HTTP {response.status} {response.url}")

                def record_failed_request(request):
                    network_errors.append(
                        f"{request.failure or 'request failed'} {request.url}")

                def record_dash_changes(request):
                    if (request.method != "POST"
                            or "/_dash-update-component" not in request.url):
                        return
                    try:
                        payload = request.post_data_json
                    except Exception:
                        return
                    dash_changes.extend(
                        str(change)
                        for change in payload.get("changedPropIds", []))

                page.on("response", record_response)
                page.on("requestfailed", record_failed_request)
                page.on("request", record_dash_changes)
                try:
                    run_checks(page, c, base, console_errors,
                               network_errors, dash_changes)
                finally:
                    page.close()
            browser.close()
    finally:
        server.terminate()
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()

    passes = max(1, args.loops)
    over = f" over {passes} passes" if passes > 1 else ""
    print(f"\nscreenshots in {SHOTS}")
    if c.failures:
        print(f"\n{len(c.failures)} of {c.ran} FAILED{over}:")
        for f in c.failures:
            print("  - " + f)
        return 1
    print(f"\nall {c.ran} cohort checks passed{over} "
          f"({c.ran // passes} per pass)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
