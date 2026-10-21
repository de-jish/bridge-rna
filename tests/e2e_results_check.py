#!/usr/bin/env python3
"""End-to-end browser check of the retrieval results view, the way a user meets it.

A sibling of `e2e_check.py`, `e2e_cohort_check.py` and `e2e_upload_check.py`:
same harness, same exclusion from pytest, and it runs against the real corpus.
`tests/test_results.py` and `tests/test_results_view.py` prove the filter rules
and the wiring on hand-built rows; this proves a person can retrieve 1,000
matches, narrow them, read one in the inspector, switch plots, export exactly
what is on screen, leave for the map and come back to the same table.

It asserts on what the page reports about itself. The query is the OSD-100
left-eye sample the design was measured on, because its top 30 carries six
profiles deposited twice (GSE88819 and GSE124745), which is what the identical
profile marks and the merge option exist for.

    /Users/josh/Bridge-RNA/.venv/bin/python tests/e2e_results_check.py \\
        [--port 8065] [--headed]
"""
from __future__ import annotations

import argparse
import io
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
from playwright.sync_api import sync_playwright

REPO = Path(__file__).resolve().parent.parent
PY = os.environ.get("MANIFOLD_PYTHON", sys.executable)
CHROME = os.environ.get("MANIFOLD_CHROME")
SHOTS = Path(os.environ.get("MANIFOLD_E2E_SHOTS",
                            Path(tempfile.gettempdir()) / "bm-results-shots"))

STUDY = "OSD-100"
SAMPLE = "Mmus_C57-6J_EYE_FLT_Rep1_M23"
# One of the six re-deposited eye profiles in that query's top 30.
TWIN_SERIES = ("GSE88819", "GSE124745")
COHORT_STUDY = "OSD-137"  # Liver in flight and ground, six animals each


class Checks:
    def __init__(self):
        self.failures: list[str] = []
        self.ran = 0

    def ok(self, cond: bool, msg: str) -> bool:
        self.ran += 1
        print(("  OK   " if cond else "  FAIL ") + msg, flush=True)
        if not cond:
            self.failures.append(msg)
        return bool(cond)

    def note(self, msg: str) -> None:
        print("  ..   " + msg, flush=True)


def shot(page, name: str) -> None:
    page.wait_for_timeout(300)
    page.screenshot(path=str(SHOTS / f"{name}.png"))


def settle(page, ms: int = 900) -> None:
    """Let Dash finish the callbacks a change set off."""
    page.wait_for_timeout(ms)
    page.wait_for_function(
        "() => !document.querySelector('[data-dash-is-loading=\"true\"]')",
        timeout=60_000)


def choose(page, dropdown_id: str, text: str, exact: bool = False) -> None:
    """Pick an option by typing it into the dropdown's search box.

    Typed, not scrolled to: a series list at k=1,000 holds 140 options and
    Dash 4 renders only the visible part of it, so the text of an option far
    down the list is simply not in the page until the list is narrowed.
    """
    page.locator(f"#{dropdown_id}").click()
    page.wait_for_timeout(400)
    search = page.locator(".dash-dropdown-content .dash-dropdown-search")
    if search.count():
        search.first.fill(text)
        page.wait_for_timeout(400)
    page.locator(".dash-dropdown-content").get_by_text(text, exact=exact).first.click()
    page.keyboard.press("Escape")
    settle(page, 1200)


def summary(page) -> str:
    return page.locator("#results-summary").inner_text().strip()


def ranks(page, column: int = 0) -> list[int]:
    cells = page.locator("#results-table-wrap tbody tr").evaluate_all(
        f"rows => rows.map(r => r.children[{column}].innerText.trim())")
    return [int(c.replace(",", "")) for c in cells if c and c[0].isdigit()]


def column_text(page, header: str) -> list[str]:
    headers = page.locator("#results-table-wrap thead th").all_inner_texts()
    i = [h.strip() for h in headers].index(header)
    return page.locator("#results-table-wrap tbody tr").evaluate_all(
        f"rows => rows.map(r => r.children[{i}].innerText.trim())")


def set_k(page, k: int) -> None:
    box = page.locator("#topk-input")
    box.fill(str(k))
    box.press("Enter")
    settle(page, 600)


def search(page, button: str = "#search-button") -> float:
    t0 = time.time()
    page.locator(button).click()
    page.wait_for_function(
        "() => document.querySelector('#search-status')"
        " && /Retrieved|Two pooled|failed/.test("
        "document.querySelector('#search-status').innerText)",
        timeout=180_000)
    page.wait_for_selector("#results-table-wrap tbody tr", timeout=60_000)
    settle(page, 1500)
    return time.time() - t0


def plot_traces(page) -> list[dict]:
    return page.evaluate("""() => {
      const gd = document.querySelector('#results-plot .js-plotly-plot');
      return (gd && gd.data || []).map(t => ({
        type: t.type || 'scatter', mode: t.mode || '', name: t.name || '',
        n: (t.x || t.z || []).length,
        custom: (t.customdata || []).length,
      }));
    }""")


def plot_layout(page) -> dict:
    return page.evaluate("""() => {
      const gd = document.querySelector('#results-plot .js-plotly-plot');
      const l = gd && gd._fullLayout || {};
      return {xtype: l.xaxis && l.xaxis.type, ytitle: l.yaxis && l.yaxis.title
              && l.yaxis.title.text};
    }""")


def pick_plot(page, label: str) -> None:
    page.locator("#results-plot-kind").get_by_text(label, exact=True).click()
    settle(page, 1500)


def no_page_overflow(page) -> bool:
    return page.evaluate(
        "() => document.documentElement.scrollWidth <= window.innerWidth + 1")


def run_checks(page, c: Checks, base: str, console_errors: list[str]) -> None:
    print("\n[load]")
    # A generous first load: Dash fetches its async component bundles here, and
    # on a machine running other servers that alone has taken over 30 s.
    page.goto(base + "/", wait_until="domcontentloaded", timeout=180_000)
    page.wait_for_selector("#results-table-wrap", timeout=180_000)
    page.wait_for_selector("#study-dropdown", timeout=180_000)
    settle(page)
    c.ok("Run a search" in page.locator("#results-table-wrap").inner_text(),
         "before a search the table invites one")
    c.ok(page.locator("#results-download-button").is_disabled(),
         "Download is disabled with nothing to export")
    c.ok(page.locator("#topk-input").input_value() == "5",
         "the number of matches still opens at 5")

    print("\n[number of matches]")
    set_k(page, 1000)
    pos = float(page.locator("#topk-slider [role=slider]").first.get_attribute(
        "aria-valuenow"))
    c.ok(abs(pos - 3.0) < 1e-3, f"typing 1000 moves the log slider to its end ({pos})")
    set_k(page, 5000)
    c.ok(page.locator("#topk-input").input_value() == "1000",
         "a k above the maximum is clamped to 1,000")

    print("\n[a 1,000-match search]")
    choose(page, "study-dropdown", STUDY, exact=True)
    choose(page, "sample-dropdown", SAMPLE)
    took = search(page)
    c.note(f"search answered in {took:.1f} s")
    text = summary(page)
    c.ok(text.startswith("1,000 matches for"), f"summary names the count: {text!r}")
    c.ok("not biological equivalence" in text, "summary keeps the equivalence caveat")
    c.ok(page.locator("#results-page-label").inner_text() == "Rows 1-50 of 1,000",
         "the pager reads Rows 1-50 of 1,000")
    first = ranks(page)
    c.ok(first == list(range(1, 51)), "page one holds ranks 1-50 in order")
    c.ok(not page.locator("#results-download-button").is_disabled(),
         "Download is enabled once there are rows")
    shot(page, "01-results-k1000")

    print("\n[paging and sorting]")
    page.locator("#results-next").click()
    settle(page)
    c.ok(ranks(page)[:1] == [51], "Next shows rank 51 first")
    page.locator("#results-table-wrap thead").get_by_text("Series", exact=True).click()
    settle(page)
    series = column_text(page, "Series")
    c.ok(series == sorted(series, key=str.lower), "Series sorts ascending")
    c.ok(page.locator("#results-page-label").inner_text().startswith("Rows 1-50"),
         "a new sort returns to the first page")
    page.locator("#results-table-wrap thead").get_by_text("Series", exact=True).click()
    settle(page)
    series = column_text(page, "Series")
    c.ok(series == sorted(series, key=str.lower, reverse=True),
         "a second click sorts descending")
    c.ok(page.locator("#results-table-wrap thead th[aria-sort=descending]").count() == 1,
         "the sorted column says so to assistive technology")
    page.locator("#results-table-wrap thead").get_by_text("Rank", exact=True).click()
    settle(page)

    print("\n[identical profiles]")
    twin_rows = page.locator("#results-table-wrap .result-twin")
    c.ok(twin_rows.count() >= 2, f"identical profiles are marked ({twin_rows.count()} on page one)")
    page.locator("#filter-merge").get_by_text("Merge identical profiles").click()
    settle(page)
    text = summary(page)
    merged = re.search(r"([\d,]+) identical profiles? merged", text)
    c.ok(bool(merged), f"merging states how many it hid: {text!r}")
    c.ok("positions among all" in text, "merging explains why ranks now skip")
    page.locator("#filter-merge").get_by_text("Merge identical profiles").click()
    settle(page)

    print("\n[series filter]")
    choose(page, "filter-series", TWIN_SERIES[0])
    gses = column_text(page, "Series")
    c.ok(gses and set(gses) == {TWIN_SERIES[0]}, f"Show only keeps {TWIN_SERIES[0]}")
    text = summary(page)
    c.ok(bool(re.match(r"[\d,]+ of 1,000 matches shown", text)), f"count line: {text!r}")
    kept = ranks(page)
    c.ok(kept == sorted(kept) and kept != list(range(1, len(kept) + 1)),
         "filtered ranks keep their true positions")
    c.ok(page.locator("#filter-clear").is_visible(), "Clear filters appears")
    page.locator("#filter-series-mode").get_by_text("Hide", exact=True).click()
    settle(page)
    c.ok(TWIN_SERIES[0] not in column_text(page, "Series"), "Hide drops that series")
    shot(page, "02-series-hidden")
    page.locator("#filter-clear").click()
    settle(page)
    c.ok(summary(page).startswith("1,000 matches for"), "Clear filters restores all rows")
    c.ok(not page.locator("#filter-clear").is_visible(), "Clear filters hides again")

    print("\n[text and score filters]")
    page.locator("#filter-text").fill("retina")
    settle(page, 1200)
    text = summary(page)
    c.ok(bool(re.match(r"[\d,]+ of 1,000", text)), f"text filter narrows: {text!r}")
    page.locator("#filter-text").fill("zzzz-no-such-sample")
    settle(page, 1200)
    c.ok("No matches pass these filters" in page.locator("#results-table-wrap").inner_text(),
         "a filter that keeps nothing says so")
    c.ok(page.locator("#results-download-button").is_disabled(),
         "and there is nothing to download")
    page.locator("#filter-text").fill("")
    page.locator("#filter-min-score").fill("0.99")
    settle(page, 1300)
    scores = [float(s) for s in column_text(page, "Score")]
    c.ok(scores and min(scores) >= 0.99, "minimum score holds on every shown row")
    page.locator("#filter-clear").click()
    settle(page)

    print("\n[inspector]")
    row = page.locator("#results-table-wrap tbody tr").nth(2)
    gsm = row.locator(".result-select").inner_text()
    row.locator("td").nth(4).click()  # the title cell, not the button
    settle(page, 1500)
    details = page.locator("#details-panel").inner_text()
    c.ok(gsm in details, f"clicking a row opens {gsm} in the inspector")
    c.ok(row.get_attribute("aria-current") == "true", "the row is marked current")
    row.locator(".result-select").focus()
    page.keyboard.press("Enter")
    settle(page, 800)
    c.ok(gsm in page.locator("#details-panel").inner_text(),
         "the accession button opens it from the keyboard")

    print("\n[plots]")
    layout = plot_layout(page)
    c.ok(layout["xtype"] == "log", "score by rank is the default, on a log rank axis")
    traces = plot_traces(page)
    c.ok(any(t["custom"] for t in traces), "its points are clickable")
    caption = page.locator("#plot-caption").inner_text()
    c.ok("depth 1,000" in caption, f"caption states the depth: {caption!r}")
    pick_plot(page, "Similarity")
    traces = plot_traces(page)
    c.ok(any(t["type"] == "heatmap" for t in traces), "Similarity draws a heatmap")
    c.ok("first 50 of 1,000" in page.locator("#plot-caption").inner_text(),
         "Similarity says it shows the first 50")
    shot(page, "03-similarity")
    pick_plot(page, "Series")
    c.ok(any(t["type"] == "bar" for t in plot_traces(page)), "Series draws bars")
    c.ok(bool(re.search(r"Showing 20 of [\d,]+ series",
                        page.locator("#plot-caption").inner_text())),
         "Series says how many it shows")
    pick_plot(page, "Network")
    c.ok(page.locator("#graph-legend").is_visible(), "Network brings its legend")
    gsm_nodes = page.evaluate("""() => {
      const gd = document.querySelector('#results-plot .js-plotly-plot');
      return (gd.data || []).flatMap(t => t.customdata || [])
        .filter(d => d && d[0] === 'gsm').length; }""")
    c.ok(gsm_nodes == 30, f"Network is capped at 30 matches ({gsm_nodes})")
    pick_plot(page, "Score by rank")
    c.ok(not page.locator("#graph-legend").is_visible(), "the legend leaves with it")

    print("\n[export]")
    choose(page, "filter-series", TWIN_SERIES[1])
    expected = int(re.match(r"([\d,]+) of", summary(page)).group(1).replace(",", ""))
    with page.expect_download() as dl:
        page.locator("#results-download-button").click()
    path = dl.value.path()
    raw = Path(path).read_text()
    frame = pd.read_csv(io.StringIO(raw), comment="#")
    c.ok(len(frame) == expected, f"the CSV holds exactly the {expected} rows shown")
    c.ok(set(frame["gse"]) == {TWIN_SERIES[1]}, "and only the filtered series")
    provenance = "\n".join(l for l in raw.splitlines() if l.startswith("#"))
    c.ok(SAMPLE in provenance and TWIN_SERIES[1] in provenance,
         "the # header records the query and the filter")
    c.ok(dl.value.suggested_filename.endswith(".csv"), "it downloads as .csv")
    page.locator("#filter-clear").click()
    settle(page)

    print("\n[a visit to the map and back]")
    page.locator("#see-on-map").click()
    page.wait_for_selector("#manifold-graph .js-plotly-plot", timeout=120_000)
    page.wait_for_timeout(1500)
    page.locator(".app-nav").get_by_text("Retrieve").click()
    page.wait_for_selector("#results-table-wrap tbody tr", timeout=60_000)
    settle(page)
    c.ok(summary(page).startswith("1,000 matches for"),
         "returning from the map repaints the same results")

    print("\n[narrow layout]")
    for width in (1180, 900, 390):
        page.set_viewport_size({"width": width, "height": 900})
        settle(page, 700)
        c.ok(no_page_overflow(page), f"no horizontal page scroll at {width}px")
        shot(page, f"04-width-{width}")
    page.set_viewport_size({"width": 1680, "height": 1010})
    settle(page, 700)

    print("\n[a cohort comparison]")
    page.locator("#mode-tab-cohort").click()
    settle(page, 800)
    choose(page, "study-dropdown", COHORT_STUDY, exact=True)
    set_k(page, 100)
    compare = page.locator("#cohort-compare-dropdown")
    compare.click()
    page.wait_for_timeout(400)
    page.locator(".dash-dropdown-content .dash-options-list-option").first.click()
    settle(page, 1500)
    search(page, "#cohort-search-button")
    headers = [h.strip() for h in
               page.locator("#results-table-wrap thead th").all_inner_texts()]
    c.ok({"Rank A", "Rank B", "Score A", "Score B"} <= set(headers),
         f"a comparison shows both cohorts' ranks: {headers}")
    c.ok(page.locator("#filter-retrieved-by-group").is_visible(),
         "Retrieved by appears for a comparison")
    page.locator("#filter-retrieved-by").get_by_text("Both", exact=True).click()
    settle(page)
    both_a = ranks(page, headers.index("Rank A"))
    both_b = ranks(page, headers.index("Rank B"))
    c.ok(len(both_a) == len(both_b) > 0, "Both keeps rows ranked by both cohorts")
    shot(page, "05-comparison-both")

    print("\n[console]")
    real = [e for e in console_errors if "favicon" not in e]
    c.ok(not real, f"no console errors ({real[:3]})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8065)
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
            browser = p.chromium.launch(headless=not args.headed, executable_path=CHROME)
            page = browser.new_page(viewport={"width": 1680, "height": 1010},
                                    accept_downloads=True)
            console_errors: list[str] = []
            page.on("console", lambda m: console_errors.append(m.text)
                    if m.type == "error" else None)
            page.on("pageerror", lambda e: console_errors.append(str(e)))
            try:
                run_checks(page, c, base, console_errors)
            finally:
                page.close()
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
    print(f"\nall {c.ran} results checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
