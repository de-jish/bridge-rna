#!/usr/bin/env python3
"""End-to-end browser check of file ingestion, run in a loop.

Deliberately NOT collected by pytest, for the same reason as `e2e_check.py`:
it boots the real Dash app against the real `cache/`, drives a real browser,
and embeds real samples through the real checkpoint. `tests/test_upload_ingestion.py`
already pins the science (a live embed reproduces the corpus vector at cosine
1.0) and the wiring; what no unit test can tell you is whether a user can
actually drop a file on the page, pick a column, and get the right answer -
twice in a row, without reloading.

The loop is the point. A one-shot upload check passes on a page that leaks
state between runs, and the two failure modes worth catching here are exactly
that: a stale `upload-store` serving the previous file's path, and a stale
column value serving the previous file's column. So every fixture is run
`--cycles` times through one long-lived page, and every cycle must produce
byte-identical hits.

The anchor for correctness is not a golden file. It is the catalog path: the
example CSV is two real columns of OSD-100, both of which are cached OSDR
samples, so the uploaded result must equal what the sample picker returns for
the same sample. Any drift in preprocessing, gene order, or encode settings
breaks that equality, and nothing else in the browser would show it.

    /Users/josh/Bridge-RNA/.venv/bin/python tests/e2e_upload_check.py
        [--port 8063] [--headed] [--cycles 2]
"""
from __future__ import annotations

import argparse
import gzip
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import pandas as pd
from playwright.sync_api import sync_playwright

from e2e_target import add_target_args, credentials, patience, target

REPO = Path(__file__).resolve().parent.parent
PY = os.environ.get("MANIFOLD_PYTHON", sys.executable)
EXAMPLE = REPO / "examples" / "osdr_upload_example.csv"
SHOTS = Path(os.environ.get("MANIFOLD_E2E_SHOTS",
                            Path(tempfile.gettempdir()) / "bm-upload-e2e-shots"))

# The two columns of the shipped example, and the OSDR sample_ids they are.
# Both are in `cache/osdr_metadata.parquet`, which is what makes the
# catalog-vs-upload equality check possible.
FLT_COL = "Mmus_C57-6J_EYE_FLT_Rep1_M23"
GC_COL = "Mmus_C57-6J_EYE_GC_Rep1_M33"
FLT_ID = f"OSD-100|{FLT_COL}"
GC_ID = f"OSD-100|{GC_COL}"


# Every GSM node the retrieval network drew, with the score parsed out of the
# hover text. The last trace is the node trace and its customdata is
# [kind, node_id, hover] - the same contract `e2e_check.py` relies on.
HITS_JS = """() => {
  const gd = document.querySelector('#network-graph .js-plotly-plot');
  if (!gd || !gd._fullData || !gd._fullData.length) return null;
  const t = gd._fullData[gd._fullData.length - 1];
  if (!t || !t.customdata) return null;
  const out = {gsm: [], query: null};
  for (const row of t.customdata) {
    if (row[0] === 'gsm') out.gsm.push([row[1], row[2]]);
    if (row[0] === 'query') out.query = row[2];
  }
  return out;
}"""

FITS_JS = ("() => document.scrollingElement.scrollHeight"
           " <= document.scrollingElement.clientHeight")


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


# --- fixtures ---------------------------------------------------------------

def build_fixtures(dest: Path) -> dict[str, Path]:
    """Every accepted variation and every rejected one, from the shipped example.

    Generated rather than committed: the repo carries one example file, and the
    seven variants below are mechanical transforms of it, so committing them
    would be seven more things to keep in sync with a format that has one
    source of truth.
    """
    dest.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(EXAMPLE, index_col=0)
    one = df[[FLT_COL]]

    paths: dict[str, Path] = {"example": EXAMPLE}

    p = dest / "single_column.csv"
    one.to_csv(p)
    paths["single"] = p

    p = dest / "versioned_ids.csv"
    v = one.copy()
    v.index = [f"{g}.{(i % 9) + 1}" for i, g in enumerate(v.index)]
    v.index.name = "gene_id"
    v.to_csv(p)
    paths["versioned"] = p

    p = dest / "tab_separated.tsv"
    one.to_csv(p, sep="\t")
    paths["tsv"] = p

    p = dest / "gzipped.csv.gz"
    with gzip.open(p, "wt") as fh:
        one.to_csv(fh)
    paths["gzip"] = p

    # Rejections. Human Ensembl IDs are the realistic mistake: a well-formed
    # matrix of the right shape that maps zero orthologs.
    p = dest / "human_ids.csv"
    h = one.copy()
    h.index = [g.replace("ENSMUSG", "ENSG") for g in h.index]
    h.index.name = "gene_id"
    h.to_csv(p)
    paths["human"] = p

    p = dest / "gene_symbols.csv"
    s = one.copy()
    s.index = [f"Gene{i}" for i in range(len(s))]
    s.index.name = "symbol"
    s.to_csv(p)
    paths["symbols"] = p

    p = dest / "no_sample_columns.csv"
    p.write_text("gene_id\nENSMUSG00000000001\nENSMUSG00000000003\n")
    paths["nocols"] = p

    return paths


# --- browser helpers --------------------------------------------------------

PATIENCE = 1  # set from the target in main(); 1 locally


def _wait_run(page, indicator: str, timeout: int = 300_000) -> float:
    """Wait out one Dash `running=` cycle on `indicator`, returning its seconds.

    Waiting on the running indicator rather than on the figure changing is
    deliberate: the loop asserts that two consecutive runs produce *identical*
    hits, so "wait until the figure differs" would hang on exactly the case
    this file exists to check.
    """
    t0 = time.time()
    page.wait_for_function(
        f"() => (document.querySelector('{indicator}')||{{}}).innerText",
        timeout=30_000 * PATIENCE)
    page.wait_for_function(
        f"() => !((document.querySelector('{indicator}')||{{}}).innerText || '').trim()",
        timeout=timeout * PATIENCE)
    page.wait_for_timeout(400)
    return time.time() - t0


def banner(page) -> tuple[str, str]:
    el = page.locator("#search-status .status-banner").first
    if el.count() == 0:
        return "", ""
    return el.inner_text().strip(), (el.get_attribute("class") or "")


def read_hits(page) -> tuple[list[tuple[str, str]], str]:
    """[(gsm, score-as-shown)] plus the query node's hover text."""
    info = page.evaluate(HITS_JS)
    if not info:
        return [], ""
    out = []
    for gsm, hover in info["gsm"]:
        m = re.search(r"Score:\s*([0-9.]+)", hover or "")
        out.append((str(gsm), m.group(1) if m else "?"))
    return out, str(info.get("query") or "")


def open_mode(page, key: str) -> None:
    """Bring one query source on screen.

    The rail used to stack the sample picker and the upload dropzone; it is a
    Sample / Cohort / Upload tablist now, so a check that drives the upload
    controls has to open Upload first, exactly as a user does. Clicking an
    already-active tab is harmless, so this is safe to call unconditionally.
    """
    tab = page.locator(f"#mode-tab-{key}")
    if "is-active" not in (tab.get_attribute("class") or ""):
        tab.click()
        page.wait_for_selector(f"#mode-panel-{key}", state="visible",
                               timeout=30_000 * PATIENCE)
        page.wait_for_timeout(400)


def upload(page, path: Path, timeout: int = 180_000,
           settle: int = 1200) -> None:
    """Put a file in the dropzone and wait until the app says it read it.

    The retry is not defensive padding. `dcc.Upload` lazy-loads its own
    JavaScript bundle, and setting the file input before that bundle lands is a
    silent no-op: no request is made, no preview appears, and the run continues
    against a file the app never saw. Locally the bundle arrives in
    milliseconds and a fixed 1.2 s sleep hid this completely; against a
    deployment it does not, and the failure presented as the column picker
    staying hidden forever rather than as an upload error.

    The settle stays a fixed wait rather than a condition, deliberately. The
    slot holds a preview on success and a banner on rejection, and several
    fixtures here are *meant* to be rejected, so there is no single "ready"
    string to wait for - and "the preview is non-empty" is satisfied by the
    previous upload's own text, which returns before this file is read at all.

    What is worth retrying is the one failure that is silent: an entirely empty
    slot means the component never reacted, and nothing downstream will ever
    recover from that.
    """
    open_mode(page, "upload")
    deadline = time.time() + (timeout * PATIENCE) / 1000
    while True:
        page.set_input_files("#upload-counts input[type=file]", str(path))
        page.wait_for_timeout(settle)
        el = page.locator("#upload-preview")
        if el.count() and (el.inner_text() or "").strip():
            return
        if time.time() >= deadline:
            raise AssertionError(
                f"the app never acknowledged the uploaded file {path.name}")


def wait_for_selected_sample(page, sample_id: str, timeout: int = 120_000) -> None:
    """Wait until a `/?q=` link has settled on *this* sample.

    `.sample-preview` mounts before the dropdowns are populated, so a fixed
    pause after navigation is a race: the search fires with nothing selected
    and the banner answers "Select a sample to start", which reads like a
    retrieval that returned nothing rather than one that never ran.

    It has to match the requested sample rather than merely being non-empty.
    The page is reused across navigations, so the previous sample's name is
    already sitting in the control and would satisfy any weaker condition
    instantly - which is the same stale-state trap this file exists to catch,
    reintroduced in the harness that checks for it.
    """
    wanted = sample_id.split("|", 1)[-1].replace("\\", "\\\\").replace("'", "\\'")
    page.wait_for_function(
        "() => ((document.querySelector('#sample-dropdown .dash-dropdown-value')"
        f"||{{}}).innerText||'').includes('{wanted}')",
        timeout=timeout * PATIENCE)
    # The control showing the sample and the app being ready to search on it
    # are different callbacks, and this one is *in addition to* the settle
    # rather than instead of it. Replacing the settle with this condition made
    # the second catalog search fire mid-render and answer "Select a sample to
    # start", which is a failure that reads like an empty retrieval.
    page.wait_for_timeout(1200)


def upload_notice(page) -> tuple[str, str]:
    """The rail's upload slot: a preview on success, a banner on rejection.

    Upload-time failures (undecodable, oversized, unparseable, no sample
    columns) land here rather than in `#search-status`, because they happen
    before any search exists to report on.
    """
    el = page.locator("#upload-preview")
    if el.count() == 0:
        return "", ""
    inner = el.locator(".status-banner").first
    cls = (inner.get_attribute("class") or "") if inner.count() else ""
    return el.inner_text().strip(), cls


def pick_column(page, column: str) -> None:
    """Choose a sample column from the dropdown, the way a user does.

    Two Dash 4 details make this fussier than it looks. The open menu is
    portaled to a `.dash-dropdown-content` at the end of the body rather than
    nested in the component, so typing at the closed trigger is a silent no-op.
    And each option is a `<label>` wrapping a `<span>`, so Playwright's
    `:text-is()` never matches it - that engine matches only the deepest element
    holding the text. Compare the option's own text instead.
    """
    page.locator("#upload-sample-column .dash-dropdown-trigger").click()
    # Wait for an option, not just for the portal: the container mounts a frame
    # before its options do, and catching it in between reads as an empty menu.
    page.wait_for_selector(".dash-dropdown-content .dash-options-list-option",
                           timeout=10_000 * PATIENCE)
    page.wait_for_timeout(250)
    opts = page.locator(".dash-dropdown-content .dash-options-list-option")
    for i in range(opts.count()):
        if opts.nth(i).inner_text().strip() == column:
            opts.nth(i).click()
            page.wait_for_timeout(400)
            return
    raise AssertionError(
        f"column {column!r} not offered; menu has "
        f"{[opts.nth(i).inner_text().strip() for i in range(opts.count())]}")


def picked_column(page) -> str:
    return page.locator("#upload-sample-column .dash-dropdown-value").inner_text().strip()


def upload_search(page) -> float:
    open_mode(page, "upload")
    page.locator("#upload-search-button").click()
    return _wait_run(page, "#upload-running-indicator")


def catalog_search(page) -> float:
    open_mode(page, "sample")
    page.locator("#search-button").click()
    return _wait_run(page, "#query-running-indicator")


# --- the run ----------------------------------------------------------------

def _report(c) -> int:
    print("\n" + "=" * 62)
    for n in c.notes:
        print("NOTE: " + n)
    if c.failures:
        for f in c.failures:
            print("FAIL: " + f)
        print(f"UPLOAD E2E FAILED ({len(c.failures)} checks)")
        return 1
    print(f"ALL UPLOAD E2E CHECKS PASSED (screenshots in {SHOTS})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8063)
    ap.add_argument("--headed", action="store_true")
    add_target_args(ap)
    ap.add_argument("--cycles", type=int, default=2,
                    help="How many times to run the whole fixture set.")
    args = ap.parse_args()
    SHOTS.mkdir(parents=True, exist_ok=True)
    c = Checks()

    global PATIENCE
    PATIENCE = patience(args)

    if not EXAMPLE.exists():
        print(f"missing {EXAMPLE}")
        return 1

    tmp = Path(tempfile.mkdtemp(prefix="bm-upload-fixtures-"))
    fx = build_fixtures(tmp)
    print(f"fixtures in {tmp}")

    with target(args, PY, REPO, timeout=180) as base:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=not args.headed)
            page = browser.new_page(viewport={"width": 1680, "height": 1010},
                                    http_credentials=credentials(args))
            console_errors: list[str] = []
            page.on("console", lambda m: console_errors.append(m.text)
                    if m.type == "error" else None)
            page.on("pageerror", lambda e: console_errors.append(str(e)))

            # --- truth: what the catalog path returns for the same samples ---
            print("\n=== 0. the catalog answers, which the uploads must match ===")
            truth: dict[str, list[tuple[str, str]]] = {}
            for sid in (FLT_ID, GC_ID):
                page.goto(f"{base}/?q={sid}", wait_until="load")
                page.wait_for_selector(".sample-preview", timeout=60_000 * PATIENCE)
                wait_for_selected_sample(page, sid)
                secs = catalog_search(page)
                hits, _ = read_hits(page)
                truth[sid] = hits
                msg, kind = banner(page)
                c.ok(len(hits) > 0, f"catalog search for {sid} drew {len(hits)} hits "
                                    f"in {secs:.1f}s")
                c.ok("precomputed OSDR embedding" in msg,
                     f"the catalog banner names the cached path: {msg[:70]}")
                c.ok("status-good" in kind, "the catalog banner is a success banner")

            c.ok(truth[FLT_ID] != truth[GC_ID],
                 "the two example columns are genuinely different samples")

            # --- the loop ------------------------------------------------------
            # Every result of every cycle, keyed by step, so a later cycle can be
            # held against cycle 1 as a whole rather than at one arbitrary point.
            transcripts: dict[int, list[tuple[str, Any]]] = {}
            for cycle in range(1, args.cycles + 1):
                print(f"\n=== cycle {cycle} of {args.cycles} ===")
                tape: list[tuple[str, Any]] = []
                transcripts[cycle] = tape

                # 1. the shipped example, both columns, no reload between them.
                print("\n--- 1. the shipped example, column picker, both columns ---")
                upload(page, fx["example"])
                preview = page.locator("#upload-preview").inner_text()
                c.ok("2 sample columns" in preview,
                     f"the preview reports both columns: {preview.strip()[:60]!r}")
                c.ok(page.locator("#upload-column-control").is_visible(),
                     "the column picker is shown when there is a choice")
                c.ok(not page.locator("#upload-search-button").is_disabled(),
                     "the search button is armed once a file is loaded")

                for column, sid in ((FLT_COL, FLT_ID), (GC_COL, GC_ID)):
                    pick_column(page, column)
                    held = picked_column(page)
                    c.ok(held == column,
                         f"the column picker holds {column} (shows {held!r})")
                    secs = upload_search(page)
                    hits, qhover = read_hits(page)
                    msg, kind = banner(page)
                    c.ok(hits == truth[sid],
                         f"uploaded {column} returns exactly the catalog result for "
                         f"{sid} ({len(hits)} hits, {secs:.1f}s)")
                    c.ok("status-good" in kind and "uploaded counts matrix live" in msg,
                         f"the banner names the uploaded path: {msg[:80]}")
                    c.ok(f"UPLOAD|{EXAMPLE.name}::{column}" in qhover,
                         f"the query node is identified as the uploaded column: "
                         f"{qhover[:70]!r}")
                    c.ok(page.evaluate(FITS_JS),
                         "the view still fits the window after an uploaded search")
                    tape.append((f"example::{column}", hits))
                if cycle == 1:
                    page.screenshot(path=str(SHOTS / "01-upload-result.png"))

                # 2. every accepted variation must give the FLT answer exactly.
                print("\n--- 2. accepted format variations ---")
                for name in ("single", "versioned", "tsv", "gzip"):
                    upload(page, fx[name])
                    if name == "single":
                        c.ok(not page.locator("#upload-column-control").is_visible(),
                             "the column picker hides again for a one-sample file")
                    secs = upload_search(page)
                    hits, _ = read_hits(page)
                    c.ok(hits == truth[FLT_ID],
                         f"{name}: identical to the catalog answer ({secs:.1f}s)")
                    tape.append((name, hits))

                # 3. rejections say why, and draw nothing.
                print("\n--- 3. rejections ---")
                for name, expect in (("human", "ortholog"),
                                     ("symbols", "ortholog"),
                                     ("nocols", "sample columns")):
                    upload(page, fx[name])
                    if name == "nocols":
                        # Rejected before any search: the notice is in the rail's
                        # upload slot, not in the search-status banner.
                        msg, kind = upload_notice(page)
                        c.ok("status-error" in kind and expect in msg.lower(),
                             f"{name}: rejected at upload time: {msg[:80]}")
                        c.ok(page.locator("#upload-search-button").is_disabled(),
                             f"{name}: the search button stays disabled")
                        c.ok(not page.locator("#upload-column-control").is_visible(),
                             f"{name}: no column picker is offered for a rejected file")
                        tape.append((f"reject::{name}", msg))
                        if cycle == 1:
                            page.screenshot(path=str(SHOTS / "02-upload-rejected.png"))
                        continue
                    upload_search(page)
                    msg, kind = banner(page)
                    hits, _ = read_hits(page)
                    c.ok("status-error" in kind and expect in msg.lower(),
                         f"{name}: rejected with the reason: {msg[:90]}")
                    c.ok(not hits, f"{name}: no hits were drawn for a rejected file")
                    c.ok(page.evaluate(FITS_JS),
                         f"{name}: the view still fits the window after a rejection")
                    tape.append((f"reject::{name}", msg))

                # 4. recovery: a good file after a rejected one still works.
                print("\n--- 4. recovery after a rejection ---")
                upload(page, fx["single"])
                upload_search(page)
                hits, _ = read_hits(page)
                c.ok(hits == truth[FLT_ID],
                     "a valid upload right after a rejected one is correct again")
                tape.append(("recovery", hits))

                # 5. the whole cycle, held against the first one.
                if cycle > 1:
                    same = tape == transcripts[1]
                    c.ok(same, f"cycle {cycle} reproduces cycle 1 step for step "
                               f"({len(tape)} recorded steps)")
                    if not same:
                        for (k1, v1), (k2, v2) in zip(transcripts[1], tape):
                            if (k1, v1) != (k2, v2):
                                print(f"     drift at {k1}: {v1} -> {v2}")

            # --- staged uploads must not accumulate --------------------------
            # Measured while the server is still up, because that is the
            # property that matters: one session doing ~20 uploads must leave
            # one staged file, not twenty. (The first version of this feature
            # left every one of them, and they outlived the process.)
            print("\n=== staging ===")
            staged = sorted(Path(tempfile.gettempdir())
                            .glob("bridge_rna_uploads_*/upload_*"))
            n_up = args.cycles * 10
            print(f"     {len(staged)} staged file(s) after ~{n_up} uploads")
            c.ok(len(staged) <= 1,
                 f"one session's uploads leave at most one staged file "
                 f"({len(staged)} found after ~{n_up})")
            c.ok(not list(Path(tempfile.gettempdir()).glob("bridge_upload_*")),
                 "nothing is staged outside the app's own upload directory")

            print("\n=== console ===")
            real = [e for e in console_errors
                    if "favicon" not in e.lower() and "_dash-component-suites" not in e]
            for e in real[:10]:
                print(f"     {e[:160]}")
            c.ok(not real, f"no console errors ({len(real)} seen)")

            browser.close()

    # `terminate()` is SIGTERM, which is how a supervisor stops a server and is
    # exactly the shutdown `atexit` does not cover, so the killed server's
    # staging directory is expected to still be there. What must hold is that
    # the next run reaps it - which is the guarantee, checked against a real
    # abandoned directory rather than a synthetic one.
    #
    # This reads *this* machine's temp directory, so it is only a statement
    # about a server this process started. Against a deployment there is no
    # local server and no local staging directory, and asserting on one would
    # be a check that passes for the wrong reason.
    if args.base_url:
        print("\n  ..   skipping the staging-directory reap check: "
              "it describes a locally started server, and this run drove "
              f"{args.base_url}")
        return _report(c)

    time.sleep(1.0)
    sys.path.insert(0, str(REPO))
    from bridge_rna.callbacks import _sweep_abandoned_upload_dirs

    left = sorted(Path(tempfile.gettempdir()).glob("bridge_rna_uploads_*"))
    c.note(f"the SIGTERMed server left {len(left)} staging dir(s), as expected")
    reaped = _sweep_abandoned_upload_dirs()
    still = sorted(Path(tempfile.gettempdir()).glob("bridge_rna_uploads_*"))
    c.ok(bool(reaped) and not still,
         f"the next run reaps the killed server's staging directory "
         f"(reaped {len(reaped)}, {len(still)} left)")

    return _report(c)


if __name__ == "__main__":
    sys.exit(main())
