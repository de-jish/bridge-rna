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


# The retrieval network is drawn as one node trace after the edge traces, and
# its customdata carries [kind, node_id, hover] per node - so the last trace
# having several nodes is the signal that a search has landed.
NETWORK_READY_JS = """() => {
  const gd = document.querySelector('#network-graph .js-plotly-plot');
  if (!gd || !gd._fullData || !gd._fullData.length) return false;
  const t = gd._fullData[gd._fullData.length - 1];
  return !!(t && t.customdata && t.customdata.length > 3);
}"""

# Pixel centre of the first GSM node, via Plotly's own data->pixel transform,
# so the check clicks the node a user would click.
GSM_NODE_JS = """() => {
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
            id: t.customdata[i][1]};
  }
  return null;
}"""


MAP_RETRIEVAL_READY_JS = """() => {
  const gd = document.querySelector('#manifold-graph .js-plotly-plot');
  return !!(gd && gd._fullData &&
            gd._fullData.some(t => t.name === 'retrieved hit'));
}"""


STABLE_MAP_X_RANGE_JS = """() => {
  const gd = document.querySelector('#manifold-graph .js-plotly-plot');
  const range = gd && gd._fullLayout && gd._fullLayout.xaxis
    && gd._fullLayout.xaxis.range;
  if (!range || range.length !== 2) return null;
  const value = JSON.stringify(Array.from(range));
  const now = performance.now();
  const prior = window.__bmE2ERangeStability;
  if (!prior || prior.value !== value) {
    window.__bmE2ERangeStability = {value, since: now};
    return null;
  }
  return now - prior.since >= 750 ? Array.from(range) : null;
}"""


EVIDENCE_ABSENT_JS = """() => {
  const gd = document.querySelector('#manifold-graph .js-plotly-plot');
  return !!(gd && gd._fullData && !gd._fullData.some(t =>
    ['512-D evidence neighbor', 'focused evidence neighbor'].includes(t.name)));
}"""


STABLE_DRAWER_GEOMETRY_JS = """expectedWidth => {
  const read = () => {
    const rect = selector => {
      const el = document.querySelector(selector);
      if (!el) return null;
      const r = el.getBoundingClientRect();
      return {left: +r.left.toFixed(1), right: +r.right.toFixed(1),
              top: +r.top.toFixed(1), bottom: +r.bottom.toFixed(1),
              width: +r.width.toFixed(1), height: +r.height.toFixed(1)};
    };
    const overlaps = (a, b) => !!(a && b && a.left < b.right &&
      a.right > b.left && a.top < b.bottom && a.bottom > b.top);
    const drawer = rect('#neighborhood-drawer');
    const plot = rect('.bm-plot-wrap');
    const modebar = rect('.bm-plot-wrap .modebar');
    const close = rect('#neighborhood-close');
    const tabs = rect('#neighborhood-tab');
    const search = rect('#neighborhood-search');
    const inside = child => !!(child && drawer &&
      child.left >= drawer.left - 1 && child.right <= drawer.right + 1 &&
      child.top >= drawer.top - 1 && child.bottom <= drawer.bottom + 1);
    return {
      viewport: {width: innerWidth, height: innerHeight},
      drawer, plot, modebar,
      horizontalOverflow: document.scrollingElement.scrollWidth > innerWidth + 1,
      controlsInside: inside(close) && inside(tabs) && inside(search),
      modebarObscured: overlaps(modebar, drawer),
    };
  };
  const current = read();
  if (!current.drawer || !current.plot
      || current.viewport.width !== expectedWidth) return null;
  const value = JSON.stringify(current);
  const now = performance.now();
  const prior = window.__bmE2EGeometryStability;
  if (!prior || prior.value !== value) {
    window.__bmE2EGeometryStability = {value, since: now};
    return null;
  }
  return now - prior.since >= 250 ? {...current, stable: true} : null;
}"""


def _figure_response_for(changed_prop: str):
    """Match the completed Dash figure callback caused by one interaction."""
    def matches(response) -> bool:
        if (response.request.method != "POST"
                or "/_dash-update-component" not in response.url):
            return False
        try:
            payload = response.request.post_data_json
        except Exception:
            return False
        output = str(payload.get("output", ""))
        return ("manifold-graph.figure" in output
                and changed_prop in payload.get("changedPropIds", []))

    return matches


def _map_x_range(page) -> list[float]:
    page.evaluate("() => { delete window.__bmE2ERangeStability; }")
    return page.wait_for_function(
        STABLE_MAP_X_RANGE_JS, timeout=90_000).json_value()


def _wait_for_evidence(page, trace_type: str | None = None) -> dict:
    """Wait for one complete source trace and Plotly's decoded render."""
    page.wait_for_function(
        """expectedType => {
          const gd = document.querySelector('#manifold-graph .js-plotly-plot');
          const raw = (gd && gd.data || []).filter(
            t => t.name === '512-D evidence neighbor');
          const full = (gd && gd._fullData || []).filter(
            t => t.name === '512-D evidence neighbor');
          return raw.length === 1 && full.length === 1
            && full[0].x && full[0].x.length === 250
            && (!expectedType || full[0].type === expectedType);
        }""",
        arg=trace_type, timeout=90_000)
    return _trace_state(page)


def _tab_until(page, predicate_js: str, *, reverse: bool = False,
               max_presses: int = 200) -> bool:
    """Reach a control with genuine Tab presses, never synthetic focus."""
    key = "Shift+Tab" if reverse else "Tab"
    for _ in range(max_presses):
        page.keyboard.press(key)
        if page.evaluate(predicate_js):
            return True
    return False


def _trace_state(page) -> dict:
    """Read the semantic overlays Plotly is actually rendering."""
    return page.evaluate("""() => {
      const gd = document.querySelector('#manifold-graph .js-plotly-plot');
      const data = (gd && gd.data) || [];
      const full = (gd && gd._fullData) || [];
      const evidence = full.filter(t => t.name === '512-D evidence neighbor');
      const focus = full.filter(t => t.name === 'focused evidence neighbor');
      const requested = full.filter(t => t.name === 'retrieved hit');
      return {
        evidenceTraces: data.filter(
          t => t.name === '512-D evidence neighbor').length,
        evidenceMarks: evidence.reduce((n, t) => n + ((t.x && t.x.length) || 0), 0),
        evidenceTypes: evidence.map(t => t.type),
        focusTraces: focus.length,
        focusMarks: focus.reduce((n, t) => n + ((t.x && t.x.length) || 0), 0),
        requestedTraces: requested.length,
        requestedColours: requested.map(t => t.marker && t.marker.color),
      };
    }""")


def _drawer_geometry(page, width: int) -> dict:
    """Wait until responsive layout settles, then measure its behavior."""
    page.evaluate("() => { delete window.__bmE2EGeometryStability; }")
    return page.wait_for_function(
        STABLE_DRAWER_GEOMETRY_JS, arg=width, timeout=30_000).json_value()


def check_neighborhood_explorer(page, c: "Checks", dash_changes: list[str]) -> None:
    """Exercise the explorer from one real cached OSDR retrieval."""
    page.locator("#see-on-map").click()
    page.wait_for_url(re.compile(r"/map$"), timeout=30_000)
    page.wait_for_function(MAP_RETRIEVAL_READY_JS, timeout=180_000)

    print("\n=== 7b. exact retrieval neighborhood explorer ===")
    explore = page.get_by_role("button", name="Explore neighborhood")
    c.ok(explore.is_visible(), "Explore neighborhood is offered for a real retrieval")
    reached_explore = _tab_until(
        page, "() => document.activeElement.id === 'explore-neighborhood'")
    if not c.ok(reached_explore,
                "Tab reaches Explore neighborhood without synthetic focus"):
        return
    before = _map_x_range(page)
    retrieval_runs = dash_changes.count("search-button.n_clicks")
    t0 = time.perf_counter()
    with page.expect_response(
            _figure_response_for("neighborhood-open-store.data"),
            timeout=90_000):
        page.keyboard.press("Enter")
    page.locator("#neighborhood-drawer").wait_for(state="visible", timeout=30_000)
    traces = _wait_for_evidence(page, "scattergl")
    first_open = time.perf_counter() - t0
    c.note(f"first neighborhood open and evidence render in {first_open:.2f}s")

    after = _map_x_range(page)
    c.ok(max(abs(a - b) for a, b in zip(before, after)) < 0.05,
         f"Explore opens without moving the viewport ({before} -> {after})")
    heading = page.locator("#neighborhood-heading").inner_text().strip()
    c.ok(bool(heading), f"the drawer names the active query: {heading!r}")
    c.ok(page.get_by_role("complementary", name=heading).count() == 1,
         "the drawer is a named complementary landmark")
    c.ok(page.get_by_role("button", name="Close neighborhood explorer").count() == 1,
         "Close is a named button")
    meta = page.locator("#neighborhood-meta").inner_text()
    c.ok("250 nearest" in meta and "512-D" in meta,
         f"the drawer names its exact evidence depth and space: {meta!r}")
    c.ok(traces["evidenceTraces"] == 1,
         "the exact evidence neighborhood is drawn once")
    c.ok(traces["evidenceMarks"] == 250,
         f"that evidence trace contains all 250 neighbors ({traces['evidenceMarks']})")

    # Opening focuses the announced heading. Reverse-tab reaches Close; a
    # forward Tab then enters the radio-tab group. Native radio groups use one
    # Tab stop and arrow keys between choices, which is their expected keyboard
    # contract rather than three redundant Tab stops.
    c.ok(page.evaluate("() => document.activeElement.id") == "neighborhood-heading",
         "opening focuses the drawer heading for announcement")
    reached_close = _tab_until(
        page, "() => document.activeElement.id === 'neighborhood-close'",
        reverse=True, max_presses=4)
    if not c.ok(reached_close,
                "Shift+Tab reaches Close without synthetic focus"):
        return
    page.keyboard.press("Tab")
    active_radio = page.evaluate("""() => ({
      type: document.activeElement.type,
      value: document.activeElement.value,
    })""")
    c.ok(active_radio == {"type": "radio", "value": "overview"},
         f"Tab reaches the explorer tabs at Overview ({active_radio})")
    overview_radio = page.get_by_role("radio", name="Overview", exact=True)
    c.ok(overview_radio.count() == 1 and overview_radio.is_checked(),
         "Overview has its accessible name and selected radio state")

    page.keyboard.press("ArrowRight")
    page.wait_for_selector(".bm-neighborhood-study", timeout=30_000)
    studies_radio = page.get_by_role("radio", name=re.compile(r"^Studies\s+\d+"))
    c.ok(studies_radio.count() == 1 and studies_radio.is_checked(),
         "the Studies tab has a radio role, count-bearing name, and keyboard state")
    study_count_text = page.locator(
        "#neighborhood-tab .dash-options-list-option:has(input[value='studies'])"
    ).inner_text()
    represented = int(re.search(r"(\d[\d,]*)", study_count_text).group(1)
                      .replace(",", ""))
    study_rows = page.locator(".bm-neighborhood-study")
    c.ok(study_rows.count() == represented and represented > 0,
         f"all {represented} represented studies are available")
    first_gse = study_rows.first.locator(
        ".bm-neighborhood-row-primary").inner_text().strip()
    c.ok(page.get_by_role(
        "button", name=re.compile(rf"^{re.escape(first_gse)}")).count() == 1,
        f"the first study row is a named button ({first_gse})")
    page.keyboard.press("Tab")
    c.ok(page.evaluate(
        "() => document.activeElement.classList.contains('bm-neighborhood-study')"),
         "Tab reaches the first study row")

    runs_before_focus = dash_changes.count("search-button.n_clicks")
    with page.expect_response(
            _figure_response_for("neighborhood-focus-store.data"),
            timeout=90_000):
        page.keyboard.press("Enter")
    page.wait_for_function(
        "() => { const gd = document.querySelector('#manifold-graph .js-plotly-plot');"
        " return gd && (gd._fullData || []).some("
        "t => t.name === 'focused evidence neighbor'); }", timeout=90_000)
    focused = _trace_state(page)
    c.ok(focused["focusTraces"] == 1 and focused["focusMarks"] > 0,
         f"selecting a study adds one focus trace ({focused['focusMarks']} marks)")
    c.ok(dash_changes.count("search-button.n_clicks") == runs_before_focus,
         "study focus does not rerun retrieval")

    reached_studies = _tab_until(page, """() => {
      const active = document.activeElement;
      return active && active.type === 'radio' && active.value === 'studies'
        && !!active.closest('#neighborhood-tab');
    }""", max_presses=220)
    if not c.ok(
            reached_studies,
            "Tab traversal returns from study rows to the selected Studies tab"):
        return
    page.keyboard.press("ArrowRight")
    samples_radio = page.get_by_role("radio", name=re.compile(r"^Samples\s+250$"))
    page.wait_for_selector(".bm-neighborhood-sample", timeout=30_000)
    c.ok(samples_radio.count() == 1 and samples_radio.is_checked(),
         "the Samples tab has a radio role, exact count-bearing name, and keyboard state")
    sample_rows = page.locator(".bm-neighborhood-sample")
    c.ok(sample_rows.count() == 250,
         f"the complete evidence list contains 250 samples ({sample_rows.count()})")
    first_gsm = sample_rows.first.locator(
        ".bm-neighborhood-row-primary").inner_text().strip()
    c.ok(page.get_by_role(
        "button", name=re.compile(rf"^{re.escape(first_gsm)}")).count() == 1,
        f"the first sample row is a named button ({first_gsm})")
    search = page.get_by_role("searchbox", name="Search neighborhood samples")
    c.ok(search.count() == 1 and search.is_visible(),
         "the sample search is a named searchbox")
    page.keyboard.press("Tab")
    c.ok(page.evaluate("() => document.activeElement.id") == "neighborhood-search",
         "Tab reaches sample search from Samples")
    page.keyboard.press("Tab")
    c.ok(page.evaluate(
        "() => document.activeElement.classList.contains('bm-neighborhood-sample')"),
         "Tab reaches the first sample row")
    page.keyboard.press("Shift+Tab")
    c.ok(page.evaluate("() => document.activeElement.id") == "neighborhood-search",
         "Shift+Tab returns to sample search")
    page.keyboard.type(first_gsm)
    page.wait_for_function(
        "() => document.querySelectorAll('.bm-neighborhood-sample').length === 1",
        timeout=30_000)
    c.ok(sample_rows.count() == 1,
         f"the complete list is searchable to one exact GSM ({first_gsm})")

    print("\n=== 7c. close, reopen, frame, and 3-D ===")
    with page.expect_response(
            _figure_response_for("neighborhood-open-store.data"),
            timeout=90_000):
        page.get_by_role("button", name="Close neighborhood explorer").click()
    page.locator("#neighborhood-drawer").wait_for(state="hidden", timeout=30_000)
    page.wait_for_function(EVIDENCE_ABSENT_JS, timeout=90_000)
    closed = _trace_state(page)
    white = {"#fff", "#ffffff", "rgb(255, 255, 255)", "white"}
    c.ok(closed["evidenceTraces"] == 0 and closed["focusTraces"] == 0,
         "Close removes evidence and focus marks")
    c.ok(closed["requestedTraces"] > 0
         and all(str(colour).lower() in white
                 for colour in closed["requestedColours"]),
         "Close preserves the requested white hit rings")

    before_reopen = _map_x_range(page)
    with page.expect_response(
            _figure_response_for("neighborhood-open-store.data"),
            timeout=90_000):
        explore.click()
    page.locator("#neighborhood-drawer").wait_for(state="visible", timeout=30_000)
    _wait_for_evidence(page, "scattergl")
    after_reopen = _map_x_range(page)
    c.ok(max(abs(a - b) for a, b in zip(before_reopen, after_reopen)) < 0.05,
         "Explore reopens without changing the viewport")
    with page.expect_response(
            _figure_response_for("neighborhood-open-store.data"),
            timeout=90_000):
        page.get_by_role("button", name="Close neighborhood explorer").click()
    page.locator("#neighborhood-drawer").wait_for(state="hidden", timeout=30_000)
    page.wait_for_function(EVIDENCE_ABSENT_JS, timeout=90_000)

    before_frame = _map_x_range(page)
    with page.expect_response(
            _figure_response_for("viewport-store.data"), timeout=90_000):
        page.get_by_role("button", name="Frame the retrieval").click()
    page.locator("#neighborhood-drawer").wait_for(state="visible", timeout=30_000)
    _wait_for_evidence(page, "scattergl")
    before_frame_span = abs(before_frame[1] - before_frame[0])
    framed = _map_x_range(page)
    c.ok(abs(framed[1] - framed[0]) < before_frame_span,
         f"Frame opens the drawer and narrows the 2-D view ({framed})")
    with page.expect_response(
            _figure_response_for("neighborhood-open-store.data"),
            timeout=90_000):
        page.get_by_role("button", name="Close neighborhood explorer").click()
    page.locator("#neighborhood-drawer").wait_for(state="hidden", timeout=30_000)
    page.wait_for_function(EVIDENCE_ABSENT_JS, timeout=90_000)

    set_segment(page, "Dimensions", "3D")
    page.wait_for_function(
        "() => { const gd = document.querySelector('#manifold-graph .js-plotly-plot');"
        " return gd && (gd._fullData || []).some(t => t.name === 'query'"
        " && t.type === 'scatter3d'); }", timeout=90_000)
    c.ok(not page.get_by_role("button", name="Frame the retrieval").is_visible(),
         "3-D hides Frame, whose 2-D ranges its camera would ignore")
    c.ok(explore.is_visible(), "Explore remains visible in 3-D")
    with page.expect_response(
            _figure_response_for("neighborhood-open-store.data"),
            timeout=90_000):
        explore.click()
    page.locator("#neighborhood-drawer").wait_for(state="visible", timeout=30_000)
    three_d = _wait_for_evidence(page, "scatter3d")
    c.ok(three_d["evidenceTraces"] == 1
         and three_d["evidenceTypes"] == ["scatter3d"],
         f"Explore draws the evidence once in 3-D ({three_d['evidenceTypes']})")

    set_segment(page, "Dimensions", "2D")
    page.wait_for_function(
        "() => { const gd = document.querySelector('#manifold-graph .js-plotly-plot');"
        " return gd && (gd._fullData || []).some(t => "
        "t.name === '512-D evidence neighbor' && t.type === 'scattergl'); }",
        timeout=90_000)
    samples_radio = page.get_by_role("radio", name=re.compile(r"^Samples\s+250$"))
    page.get_by_text(re.compile(r"^Samples\s+250$"), exact=True).click()
    page.wait_for_selector("#neighborhood-search", state="visible", timeout=30_000)

    print("\n=== 7d. docked drawer and bottom sheet ===")
    for width, height in ((1440, 1000), (1024, 900), (768, 900), (320, 780)):
        page.set_viewport_size({"width": width, "height": height})
        page.locator("#neighborhood-drawer").scroll_into_view_if_needed()
        geom = _drawer_geometry(page, width)
        narrow = width <= 900
        c.ok(geom["stable"], f"{width}px responsive geometry settles")
        c.ok(not geom["horizontalOverflow"],
             f"{width}px has no horizontal page overflow")
        c.ok(geom["controlsInside"],
             f"{width}px keeps Close, tabs, and search inside the drawer")
        c.ok(geom["modebar"] is not None and not geom["modebarObscured"],
             f"{width}px leaves the Plotly modebar unobscured")
        d = geom["drawer"]
        v = geom["viewport"]
        c.ok(d["left"] >= -1 and d["right"] <= v["width"] + 1
             and d["top"] >= -1 and d["bottom"] <= v["height"] + 1,
             f"the {width}px drawer stays within the viewport ({d})")
        if narrow:
            c.ok(abs(d["left"]) <= 1 and abs(d["right"] - width) <= 1
                 and geom["plot"]["top"] < d["top"] < geom["plot"]["bottom"],
                 f"the {width}px drawer behaves as a full-width bottom sheet")
        else:
            c.ok(d["left"] >= geom["plot"]["right"] - 1,
                 f"the {width}px drawer docks beside, not over, the map")
        page.screenshot(path=str(SHOTS / f"10-neighborhood-{width}.png"))

    c.ok(dash_changes.count("search-button.n_clicks") == retrieval_runs,
         "opening, closing, focusing, filtering, framing, and resizing do not rerun retrieval")


class Checks:
    def __init__(self):
        self.failures: list[str] = []
        self.ran = 0
        self.notes: list[str] = []

    def ok(self, cond: bool, msg: str) -> bool:
        # Counted, because the documented totals used to be hand-written and
        # drifted: the number in the docs was never the number this file ran.
        self.ran += 1
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
            browser = p.chromium.launch(
                headless=not args.headed, executable_path=CHROME)
            page = browser.new_page(viewport={"width": 1680, "height": 1000})
            console_errors: list[str] = []
            network_errors: list[str] = []

            def record_response(response):
                if response.status >= 400:
                    network_errors.append(f"HTTP {response.status} {response.url}")

            def record_failed_request(request):
                network_errors.append(
                    f"{request.failure or 'request failed'} {request.url}")

            def watch_runtime(runtime_page):
                runtime_page.on(
                    "console", lambda m: console_errors.append(m.text)
                    if m.type == "error" else None)
                runtime_page.on(
                    "pageerror", lambda e: console_errors.append(str(e)))
                runtime_page.on("response", record_response)
                runtime_page.on("requestfailed", record_failed_request)

            watch_runtime(page)

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
                ".bm-group:has(.bm-group-label:text-is('Number of ARCHS4 points')) "
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

            # A whole-map field says nothing under the control: the full bar is
            # the answer, and "Colours all 942,563 points." only restated it.
            cov = page.locator("#coverage").inner_text().strip()
            c.ok(cov == "",
                 f"a whole-map field leaves the coverage readout silent: {cov!r}")
            c.ok(page.locator(".bm-coverage-bar").count() == 1,
                 "while the bar itself stays, so the rail does not jump")

            print("\n=== 2b. the key names the shapes, not only the hues ===")
            # That a diamond is one of the 2,108 spaceflight samples has been
            # true since the map was built and was stated nowhere a viewer could
            # read it. The colour legend keys hue; this keys shape.
            corpus = page.locator("#legend-corpus").inner_text().replace("\n", " ")
            print(f"     corpus key: {corpus!r}")
            c.ok("ARCHS4" in corpus and "OSDR" in corpus,
                 f"both corpora are keyed by shape: {corpus!r}")
            shapes = page.eval_on_selector_all(
                "#legend-corpus .bm-key-glyph", "els => els.map(e => e.className)")
            c.ok(any("corpus-archs4" in s for s in shapes)
                 and any("corpus-osdr" in s for s in shapes),
                 f"each with its own glyph: {shapes}")
            # A key is what you are looking at, so an untickable layer leaves it.
            page.locator("#layers input[type=checkbox]").nth(1).uncheck()
            page.wait_for_timeout(2500)
            corpus = page.locator("#legend-corpus").inner_text()
            c.ok("OSDR" not in corpus,
                 f"unticking a layer drops its row from the key: {corpus!r}")
            page.locator("#layers input[type=checkbox]").nth(1).check()
            page.wait_for_timeout(2500)

            print("\n=== 3. two color-bys, and the nine that were removed ===")
            # This section used to pick "Flight vs Ground" and assert that the
            # ARCHS4 cloud degraded to a faint context layer. All nine OSDR-only
            # fields were removed on 2026-08-11, so that state is no longer
            # reachable from a full cache at all - the only route into it is
            # Tissue on a machine that never fetched the GEO join, which this
            # suite cannot produce and `tests/test_render.py` covers directly.
            # What is checkable here is the removal itself, and that both
            # survivors really do color the whole corpus in every projection.
            page.locator("#color-by").click()
            page.wait_for_timeout(500)
            options = page.locator(
                ".dash-dropdown-content .dash-options-list-option").all_inner_texts()
            print(f"     color-by options: {options}")
            c.ok(len(options) == 2, f"two color-bys are offered: {options}")
            removed = [name for name in
                       ("Flight vs Ground", "Spaceflight arm", "Strain", "Sex",
                        "Genotype", "Study", "Habitat", "Mission duration", "Diet")
                       if any(name in option for option in options)]
            c.ok(not removed, f"no removed color-by is back in the menu: {removed}")
            c.ok(all("whole map" in option for option in options),
                 "every offered field colors the whole map")
            page.keyboard.press("Escape")
            page.wait_for_timeout(400)

            # Every projection x every color-by, because a stale legend or a
            # stale color plan would only show up on the second combination.
            for method in ("t-SNE", "PCA", "UMAP"):
                set_segment(page, "Projection", method)
                page.wait_for_timeout(1500)
                for field in ("Species", "Tissue"):
                    page.locator("#color-by").click()
                    page.wait_for_timeout(400)
                    page.locator(".dash-dropdown-content").get_by_text(
                        field, exact=False).first.click()
                    page.wait_for_timeout(2500)
                    info = wait_for_points(page)
                    title = page.locator("#legend-title").inner_text()
                    rows = page.locator("#legend-list .bm-legend-item").count()
                    print(f"     {method:6} / {field:8} {info['total']:>9,} glyphs, "
                          f"{rows:2} legend rows, title {title!r}")
                    c.ok(info["total"] > 900_000,
                         f"{method}/{field} draws the whole corpus ({info['total']:,})")
                    c.ok(rows > 0, f"{method}/{field} produced a legend")
                    # The heading is small-caps by stylesheet, and `inner_text`
                    # reports the transformed text, so this compares folded.
                    c.ok(field.lower() in title.lower(),
                         f"{method}/{field} legend is not stale: {title!r}")
                    c.ok(info["images"] == 0, "still no underlay image")
            page.screenshot(path=str(SHOTS / "02-two-color-bys.png"))

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
                set_segment(page, "Number of ARCHS4 points", label)
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
                ".bm-group:has(.bm-group-label:text-is('Number of ARCHS4 points')) "
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
                ".bm-group:has(.bm-group-label:text-is('Number of ARCHS4 points')) "
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

            print("\n=== 7. the retrieval view fits its window ===")
            # Both views are fixed-height instruments that scroll internally,
            # and nothing in the Python can tell you whether they still do.
            # This caught the shell growing to fit its tallest child: opening a
            # hit whose GEO record ran long pushed the page 410 px below the
            # fold, took the AI panel off screen, and stretched the network
            # canvas out of the window with it. The page height is the check,
            # because that is the thing that must not move.
            rp = browser.new_page(viewport={"width": 1680, "height": 1010})
            watch_runtime(rp)
            dash_changes: list[str] = []

            def record_dash_changes(request):
                if (request.method != "POST"
                        or "/_dash-update-component" not in request.url):
                    return
                try:
                    payload = request.post_data_json
                except Exception:
                    return
                dash_changes.extend(str(change)
                                    for change in payload.get("changedPropIds", []))

            rp.on("request", record_dash_changes)
            rp.goto(f"http://127.0.0.1:{args.port}/", wait_until="load")
            rp.wait_for_selector(".sample-preview", timeout=60_000)
            rp.wait_for_timeout(1500)
            fits = "() => document.scrollingElement.scrollHeight" \
                   " <= document.scrollingElement.clientHeight"
            c.ok(rp.evaluate(fits), "the view fits the window before a search")

            rp.locator("#search-button").click()
            rp.wait_for_function(NETWORK_READY_JS, timeout=180_000)
            rp.wait_for_timeout(1500)
            search_status = rp.locator("#search-status").inner_text()
            c.ok("precomputed OSDR embedding" in search_status,
                 f"the real OSDR query used the cached path: {search_status[:90]!r}")
            c.ok(dash_changes.count("search-button.n_clicks") == 1,
                 "the requested retrieval ran exactly once")
            c.ok(rp.evaluate(fits), "the view still fits once the network is drawn")

            # Open the hit whose record is longest, since the bug only showed
            # up on a hit with enough metadata to overflow the inspector.
            pos = rp.evaluate(GSM_NODE_JS)
            if pos:
                rp.mouse.click(pos["x"], pos["y"])
                rp.wait_for_function(
                    "() => { const d = document.querySelector('#details-panel');"
                    " return d && d.innerText.includes('GSM'); }", timeout=90_000)
                rp.wait_for_timeout(3000)
                # The overflow is forced rather than hoped for. This used to
                # rely on the opened hit's GEO record being long enough to
                # overspill the panel by itself - but that record is fetched
                # live from NCBI inside the callback and the fetch fails
                # closed, so a rate-limited run produced a short record, no
                # overflow, and a red "the inspector scrolls its own overflow"
                # that was really a network hiccup. A tall filler makes the
                # condition the invariant is about, so the invariant is what
                # gets measured. Removed again straight after.
                geom = rp.evaluate(
                    """() => {
                      const d = document.querySelector('.details-panel');
                      const ai = document.querySelector('.ai-panel');
                      const read = () => ({
                        sh: document.scrollingElement.scrollHeight,
                        ch: document.scrollingElement.clientHeight,
                        scrolls: d.scrollHeight > d.clientHeight,
                        aiBottom: Math.round(ai.getBoundingClientRect().bottom)});
                      const asFound = read();
                      const filler = document.createElement('div');
                      filler.style.height = '2000px';
                      d.appendChild(filler);
                      const forced = read();
                      filler.remove();
                      return {...asFound, overflowY: getComputedStyle(d).overflowY,
                              forced};
                    }""")
                print(f"     {geom}")
                c.ok(geom["sh"] <= geom["ch"],
                     f"an open inspector does not grow the page ({geom['sh']} "
                     f"vs {geom['ch']})")
                c.ok(geom["overflowY"] in ("auto", "scroll")
                     and geom["forced"]["scrolls"],
                     "the inspector scrolls its own overflow instead")
                c.ok(geom["forced"]["sh"] <= geom["forced"]["ch"],
                     "and 2000 px more of it still does not grow the page "
                     f"({geom['forced']['sh']} vs {geom['forced']['ch']})")
                c.ok(geom["aiBottom"] <= geom["ch"],
                     f"the AI panel stays on screen (bottom at {geom['aiBottom']})")
            else:
                c.ok(False, "no GSM node to open in the inspector")
            check_neighborhood_explorer(rp, c, dash_changes)
            rp.close()

            print("\n=== 8. finding a sample on the map ===")
            page.goto(f"http://127.0.0.1:{args.port}/map", wait_until="networkidle")
            wait_for_points(page)
            # `wait_for_points` returns as soon as one trace has coordinates,
            # which is before the view has finished hydrating - and the first
            # interaction after a navigation is the one that finds out. Let it
            # settle before typing into a control that was mounted a moment ago.
            page.wait_for_timeout(2500)

            # The pause between filling the box and pressing Enter is not
            # padding, and it is not a workaround for a defect in the app.
            #
            # The find box no longer debounces, so its `value` is published on
            # every keystroke and Enter now means only `n_submit`. Under
            # `debounce=True` the two were one event - Dash's Input flushes the
            # pending value on Enter *only* when debounce is literally `true` -
            # so `fill` immediately followed by `press` used to be atomic and no
            # longer is. `page.fill` and `page.press` are two CDP calls with no
            # event-loop tick between them, which a browser cannot produce from
            # real input: measured on this app, a gap of 0 ms commits the
            # previous text and **every gap from 5 ms up commits the right one**,
            # against a fastest realistic keypress-to-keypress of tens of
            # milliseconds. So this restores the ordering a real keyboard has,
            # rather than hiding anything.
            def find_it(term):
                page.fill("#find-input", "")
                # `type`, not a second `fill`. Filling replaces the whole value
                # in one synthetic event; typing produces one input event per
                # character, which is what a keyboard does and what the
                # suggestion callback is written against.
                page.type("#find-input", term, delay=20)
                page.wait_for_timeout(250)
                page.press("#find-input", "Enter")
                # Wait for the answer, not for a guess at how long it takes.
                #
                # A fixed sleep was enough while Enter was the *only* thing the
                # box did. It is not any more: Enter now runs a search whose
                # first invocation also builds the 460 ms accession index, and
                # the figure it redraws carries 942,563 glyphs - so the first
                # find of a session lands well after the third of a second the
                # later ones need, and only the first one failed. That is this
                # file's stated discipline anyway: wait on what the page reports
                # about itself, so a slow render produces a late read rather
                # than a wrong one.
                try:
                    page.wait_for_function(
                        "() => (document.getElementById('find-status')"
                        ".innerText || '').trim().length > 0", timeout=20_000)
                except Exception:
                    pass  # an empty query legitimately says nothing; read it anyway
                page.wait_for_timeout(1500)
                return page.evaluate("""() => {
                    const gd = document.querySelector('#manifold-graph .js-plotly-plot');
                    const t = (gd._fullData || []).find(t => t.name === 'found');
                    return {
                      marks: t ? (t._length !== undefined ? t._length : t.x.length) : 0,
                      type: t ? t.type : null,
                      status: (document.querySelector('#find-status') || {}).innerText || '',
                      key: (document.querySelector('#legend-found') || {}).innerText || '',
                      badges: (document.querySelector('#plot-badges') || {}).innerText || '',
                      panel: (document.querySelector('#picked-group') || {}).innerText || '',
                      frame: !!document.querySelector('#frame-find[style*="display: none"]')
                             ? false : !!document.querySelector('#frame-find'),
                    };
                }""")

            one = find_it("GSE143281")
            print(f"     GSE143281 -> {one['marks']} mark(s), {one['type']}, "
                  f"status {' '.join(one['status'].split())[:70]!r}")
            c.ok(one["marks"] >= 1, f"a series marks its samples ({one['marks']})")
            c.ok(one["type"] == "scattergl",
                 f"the found layer is WebGL in 2-D, not the non-gl Scatter ({one['type']})")
            c.ok("GSE143281" in one["key"], "the found mark is named in the key")
            c.ok("GSE143281" in one["badges"], "the plot badge names what was found")
            c.ok("GSE143281" in one["panel"],
                 "the record panel names the study that was found")
            c.ok(one["frame"], "framing is offered for a found study")

            # The narrowing of 2026-08-13: the box takes studies, so a sample
            # accession is a shape miss and not a one-point hit.
            sample = find_it("GSM4256053")
            c.ok(sample["marks"] == 0, "a sample accession marks nothing")
            c.ok("Color by" in sample["status"],
                 f"and is refused as a shape: {sample['status'][:80]}")

            series = find_it("GSE228590")
            print(f"     GSE228590 -> {series['marks']} marks, status: "
                  f"{' '.join(series['status'].split())[:70]}")
            c.ok(series["marks"] == 500,
                 f"a huge series is capped at 500 marks ({series['marks']})")
            c.ok("8,764" in series["status"] and "500" in series["status"],
                 "the cap states what it dropped rather than being silent")
            c.ok("8,764" in series["badges"] or "500" in series["badges"],
                 "the badge reports the cap too")
            c.ok("500" in series["key"] and "8,764" not in series["key"],
                 "the key counts what is drawn, not what exists")

            # A refusal must not read as a breakage: "liver" is the color-by's
            # question and the message has to say so.
            word = find_it("liver")
            c.ok(word["marks"] == 0, "free text marks nothing")
            c.ok("Color by" in word["status"],
                 f"free text is pointed at the color-by: {word['status'][:80]}")
            c.ok(not word["frame"], "no framing is offered when nothing was found")

            missing = find_it("GSE9999999")
            c.ok("not on this map" in missing["status"],
                 f"an absent accession says so: {missing['status'][:80]}")
            c.ok("Color by" not in missing["status"],
                 "an absent accession is not confused with free text")

            # Framing is a button, never automatic.
            before = page.evaluate("() => document.querySelector"
                                   "('#manifold-graph .js-plotly-plot')"
                                   "._fullLayout.xaxis.range.slice()")
            find_it("OSD-100")
            after_find = page.evaluate("() => document.querySelector"
                                       "('#manifold-graph .js-plotly-plot')"
                                       "._fullLayout.xaxis.range.slice()")
            c.ok(abs(after_find[1] - after_find[0]) > 20,
                 f"a search does not move the viewport on its own ({after_find})")
            page.click("#frame-find")
            page.wait_for_timeout(3000)
            framed = page.evaluate("() => document.querySelector"
                                   "('#manifold-graph .js-plotly-plot')"
                                   "._fullLayout.xaxis.range.slice()")
            print(f"     x range {[round(v, 2) for v in before]} -> "
                  f"{[round(v, 2) for v in framed]}")
            c.ok(abs(framed[1] - framed[0]) < abs(after_find[1] - after_find[0]),
                 f"the frame button zooms in ({framed})")
            page.screenshot(path=str(SHOTS / "07-find.png"))

            # Find works in 3-D; framing does not, and does not pretend to.
            set_segment(page, "Dimensions", "3D")
            page.wait_for_timeout(6000)
            three = page.evaluate("""() => {
                const gd = document.querySelector('#manifold-graph .js-plotly-plot');
                const t = (gd._fullData || []).find(t => t.name === 'found');
                return {marks: t ? t.x.length : 0, type: t ? t.type : null,
                        frame: !!document.querySelector('#frame-find[style*="display: none"]')
                               ? false : !!document.querySelector('#frame-find')};
            }""")
            c.ok(three["marks"] == 12,
                 f"the found set is still marked in 3-D ({three['marks']})")
            c.ok(three["type"] == "scatter3d", "the 3-D found layer is scatter3d")
            c.ok(not three["frame"],
                 "3-D hides the frame button, which its camera would ignore")
            set_segment(page, "Dimensions", "2D")
            page.wait_for_timeout(3000)

            print("\n=== 9. the find box completes what is being typed ===")
            # The placeholder is the only thing an empty box says, and a
            # four-grammar wording clipped mid-word at "sample na…" for as long
            # as the box took four grammars. Nothing in Python can catch that -
            # it is a text
            # measurement in the rail's own font - so it is measured here, the
            # way the browser measures it.
            fit = page.evaluate("""() => {
                const el = document.getElementById('find-input');
                const cs = getComputedStyle(el);
                const c = document.createElement('canvas').getContext('2d');
                c.font = `${cs.fontStyle} ${cs.fontWeight} ${cs.fontSize} `
                       + `${cs.fontFamily}`;
                return {text: el.placeholder, avail: el.clientWidth,
                        width: Math.round(c.measureText(el.placeholder).width)};
            }""")
            print(f"     placeholder {fit['text']!r}: {fit['width']}px "
                  f"in {fit['avail']}px")
            c.ok(fit["width"] <= fit["avail"],
                 f"the placeholder fits its box ({fit['width']} of {fit['avail']}px)")
            for grammar in ("GSE", "OSD-100"):
                c.ok(grammar in fit["text"],
                     f"and still names {grammar}: {fit['text']!r}")
            # The suggestions exist because the input stopped debouncing, so the
            # first thing to establish is that typing still costs nothing: the
            # search, and the 942,563-point figure that hangs off it, must not
            # run until Enter, blur or a chosen row says so.
            page.fill("#find-input", "")
            page.wait_for_timeout(120)
            page.press("#find-input", "Enter")
            page.wait_for_timeout(2500)
            page.click("#find-input")
            page.type("#find-input", "GSE1432", delay=45)
            page.wait_for_timeout(2500)
            typed = page.evaluate("""() => {
                const box = document.getElementById('find-input');
                const list = document.getElementById('find-suggestions');
                const rows = [...list.querySelectorAll('[role="option"]')];
                const r = list.getBoundingClientRect();
                return {n: rows.length, height: +r.height.toFixed(1),
                        scrolls: list.scrollHeight > r.height + 2,
                        role: box.getAttribute('role'),
                        expanded: box.getAttribute('aria-expanded'),
                        first: rows.length ? rows[0].innerText.split('\\n')[0] : '',
                        status: document.getElementById('find-status').innerText,
                        marked: !!(document.querySelector(
                            '#manifold-graph .js-plotly-plot')._fullData || [])
                            .find(t => t.name === 'found')};
            }""")
            print(f"     {typed['n']} rows in {typed['height']}px, first {typed['first']!r}")
            c.ok(typed["n"] > 1, f"a partial accession is completed ({typed['n']} rows)")
            c.ok(typed["first"].startswith("GSE1432"),
                 f"the completions extend what was typed: {typed['first']!r}")
            c.ok(typed["height"] <= 200,
                 f"the list is bounded rather than growing ({typed['height']}px)")
            c.ok(typed["scrolls"], "and scrolls inside that bound")
            c.ok(typed["role"] == "combobox" and typed["expanded"] == "true",
                 f"the box is an expanded combobox ({typed['role']}, {typed['expanded']})")
            c.ok(typed["status"].strip() == "",
                 f"typing alone runs no search: {typed['status'][:60]!r}")
            c.ok(not typed["marked"], "and marks nothing on the map")
            page.screenshot(path=str(SHOTS / "08-find-autocomplete.png"))

            # Arrow keys move a virtual cursor and scroll it into view; Enter
            # takes the row rather than the half-typed text under it.
            for _ in range(6):
                page.keyboard.press("ArrowDown")
                page.wait_for_timeout(120)
            keyed = page.evaluate("""() => {
                const list = document.getElementById('find-suggestions');
                const rows = [...list.querySelectorAll('[role="option"]')];
                const i = rows.findIndex(r => r.classList.contains('is-active'));
                const lr = list.getBoundingClientRect();
                const rr = i >= 0 ? rows[i].getBoundingClientRect() : null;
                return {i: i, scrollTop: list.scrollTop,
                        inView: rr ? (rr.top >= lr.top - 1 && rr.bottom <= lr.bottom + 1)
                                   : false,
                        aria: document.getElementById('find-input')
                              .getAttribute('aria-activedescendant'),
                        label: i >= 0 ? rows[i].innerText.split('\\n')[0] : ''};
            }""")
            c.ok(keyed["i"] == 5, f"six Downs land on the sixth row ({keyed['i']})")
            c.ok(keyed["inView"], "the active row is scrolled into view")
            c.ok(keyed["scrollTop"] > 0, f"the list scrolled ({keyed['scrollTop']}px)")
            c.ok(bool(keyed["aria"]), "aria-activedescendant follows the cursor")
            page.keyboard.press("Enter")
            page.wait_for_timeout(3500)
            chosen = page.evaluate("""() => ({
                value: document.getElementById('find-input').value,
                open: document.getElementById('find-suggestions')
                      .getBoundingClientRect().height > 0,
                status: document.getElementById('find-status').innerText,
                panel: document.getElementById('picked-group').innerText,
            })""")
            print(f"     Enter chose {chosen['value']!r}")
            c.ok(chosen["value"] == keyed["label"],
                 f"Enter commits the active row ({chosen['value']!r})")
            c.ok(not chosen["open"], "and closes the list")
            c.ok(chosen["value"] in chosen["status"],
                 f"the search ran for it: {chosen['status'][:60]!r}")
            c.ok(chosen["value"] in chosen["panel"] or chosen["panel"].strip(),
                 "the record panel opened for it")

            # Escape closes without clearing; free text is still refused.
            page.click("#find-input")
            page.type("#find-input", "0", delay=40)
            page.wait_for_timeout(2000)
            page.keyboard.press("Escape")
            page.wait_for_timeout(400)
            escaped = page.evaluate("""() => ({
                open: document.getElementById('find-suggestions')
                      .getBoundingClientRect().height > 0,
                value: document.getElementById('find-input').value,
                expanded: document.getElementById('find-input')
                          .getAttribute('aria-expanded'),
            })""")
            c.ok(not escaped["open"], "Escape closes the list")
            c.ok(escaped["expanded"] == "false", "and says so to a screen reader")
            c.ok(escaped["value"].endswith("0"), "without clearing what was typed")

            page.fill("#find-input", "")
            page.type("#find-input", "liver", delay=40)
            page.wait_for_timeout(2200)
            refused = page.evaluate("""() => {
                const list = document.getElementById('find-suggestions');
                return {rows: list.querySelectorAll('[role="option"]').length,
                        empty: !!list.querySelector('.bm-suggest-empty'),
                        open: list.getBoundingClientRect().height > 0};
            }""")
            c.ok(refused["rows"] == 0 and not refused["open"],
                 "free text is refused here exactly as it is in the search")
            c.ok(not refused["empty"],
                 "and gets no 'no matches' row to contradict the sentence below")

            print("\n=== 10. one reset undoes every kind of framing ===")
            page.fill("#find-input", "")
            page.wait_for_timeout(120)
            page.press("#find-input", "Enter")
            page.wait_for_timeout(2500)
            whole = page.evaluate("() => document.querySelector"
                                  "('#manifold-graph .js-plotly-plot')"
                                  "._fullLayout.xaxis.range.slice()")
            c.ok(not page.locator("#reset-view").is_visible(),
                 "the reset is not offered on an unframed map")

            def x_span():
                r = page.evaluate("() => document.querySelector"
                                  "('#manifold-graph .js-plotly-plot')"
                                  "._fullLayout.xaxis.range.slice()")
                return abs(r[1] - r[0])

            find_it("OSD-100")
            page.click("#frame-find")
            page.wait_for_timeout(3000)
            narrowed = x_span()
            c.ok(narrowed < abs(whole[1] - whole[0]), f"framing narrows ({narrowed:.2f})")
            c.ok(page.locator("#reset-view").is_visible(),
                 "the reset appears once the map is framed")
            page.screenshot(path=str(SHOTS / "09-framed-and-reset.png"))
            page.click("#reset-view")
            page.wait_for_timeout(3000)
            back = x_span()
            print(f"     x span {abs(whole[1] - whole[0]):.2f} -> {narrowed:.2f} "
                  f"-> {back:.2f}")
            # `uirevision="keep"` means Plotly holds the user's zoom unless the
            # figure changes the attribute, so a reset that merely omits the
            # range leaves the map where it was. This is that assertion.
            c.ok(abs(back - abs(whole[1] - whole[0])) < 0.05,
                 f"the reset restores the whole map ({back:.2f})")
            c.ok(not page.locator("#reset-view").is_visible(),
                 "and takes itself off again")
            c.ok(page.input_value("#find-input") == "OSD-100",
                 "the query survives the reset")
            c.ok(page.locator("#frame-find").is_visible(),
                 "and so does the result, ready to be framed again")
            page.click("#frame-find")
            page.wait_for_timeout(3000)
            c.ok(abs(x_span() - narrowed) < 0.05,
                 f"re-framing gives the same window, not a stale one ({x_span():.2f})")
            page.click("#reset-view")
            page.wait_for_timeout(2500)

            # A plain scroll zoom is the same state by the time it is stored, so
            # the same control undoes it - which is the reason there is one.
            page.mouse.move(900, 500)
            for _ in range(4):
                page.mouse.wheel(0, -240)
                page.wait_for_timeout(400)
            page.wait_for_timeout(2500)
            c.ok(x_span() < abs(whole[1] - whole[0]), "a scroll zoom narrows the view")
            c.ok(page.locator("#reset-view").is_visible(),
                 "and offers the same reset")
            page.click("#reset-view")
            page.wait_for_timeout(3000)
            c.ok(abs(x_span() - abs(whole[1] - whole[0])) < 0.05,
                 f"which restores the whole map too ({x_span():.2f})")

            set_segment(page, "Dimensions", "3D")
            page.wait_for_timeout(5000)
            c.ok(not page.locator("#reset-view").is_visible(),
                 "3-D hides the reset, whose axis ranges its camera ignores")
            set_segment(page, "Dimensions", "2D")
            page.wait_for_timeout(3000)

            print("\n=== 11. console and network ===")
            real = [e for e in console_errors
                    if "favicon" not in e.lower() and "_dash-component-suites" not in e]
            for e in real[:10]:
                print(f"     {e[:160]}")
            c.ok(not real, f"no console errors ({len(real)} seen)")
            for error in network_errors[:10]:
                print(f"     {error[:200]}")
            c.ok(not network_errors,
                 f"no failed browser requests or HTTP errors ({len(network_errors)} seen)")

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
        print(f"E2E FAILED ({len(c.failures)} of {c.ran} checks)")
        return 1
    print(f"ALL {c.ran} E2E CHECKS PASSED (screenshots in {SHOTS})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
