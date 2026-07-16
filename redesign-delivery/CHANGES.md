# bridge-rna web app redesign - change log

This folder is a self-contained copy of everything that changed during the UI/UX redesign of `app_osdr_dash.py`, plus screenshots of every UI state and this writeup.
Nothing here has been pushed anywhere.

## What's in this folder

- `changed-files/app_osdr_dash.py` - the redesigned app (drop-in replacement for the repo's `app_osdr_dash.py`).
- `changed-files/assets/style.css` - **new file**; the entire design system lives here.
  Dash auto-loads anything in an `assets/` folder next to the app, so no wiring is needed beyond placing this file there.
- `screenshots/` - before/after and every validated state (initial, loading, populated graph, details, error, top-k=30).

To apply: copy `changed-files/app_osdr_dash.py` over the existing one and copy `changed-files/assets/style.css` to `assets/style.css` next to it.
Only two files changed. No other repo files, data, or model artifacts were touched.

## How it was validated

Every state below was driven in a real headless Chromium against the running app and screen-shotted, then reviewed for overlap and alignment:

- Initial load, loading state (real retrieval subprocess), error state, populated network graph, query/GSM/GSE details, expanded long-text blocks, and the 30-node case.
- The populated-data states were exercised with a synthetic-hits harness because the live retrieval pipeline needs `archs4py` + `biopython`, which are not installed in this environment.
  The data flow, callbacks, and rendering are the real ones; only the `search_hits` data source was mocked for the screenshots.
- The loading and error states were exercised against the **real** retrieval path (the error path is what a user hits in this env, since `archs4py`/`Bio` are missing).

## 1. Bug fix - subprocess tracebacks no longer dumped into the UI

This was the wall of monospace traceback text visible in the old UI.

**Before:** on a failed retrieval, `run_real_retrieval` raised `RuntimeError(f"Demo retrieval failed: {msg[:500]}")` where `msg` was raw subprocess stderr (a full Python traceback).
`run_search` caught it, prepended `"Real retrieval failed: "`, and rendered the whole thing - stack frames and all - into the status area.

**After:**
- `run_real_retrieval` now logs the **full** stderr server-side (`print(..., file=sys.stderr)`) for debugging, and raises only the **last non-empty line** of stderr (almost always the real exception message, e.g. `RuntimeError: Requested OSDR sample not found`).
  A new helper `_last_nonempty_line()` extracts that line.
- A new lightweight exception `RetrievalError(message, detail=...)` carries the clean one-line message *and* the full raw output, so nothing is lost.
- `run_search` renders failures through a new `build_status_banner(message, kind="error", detail=...)` helper: a clean one-line red banner with a collapsed **"Show details"** disclosure that expands the full traceback in a scrollable block.
  The stack trace is available for debugging but never occupies the primary viewport.

See `screenshots/06_error_banner.png`.

## 2. CSS architecture

- Added `assets/style.css`.
  All styling now lives in one stylesheet keyed on CSS custom properties.
- Removed essentially every inline `style={...}` dict from the layout and the `build_*` functions, replacing them with `className` + CSS rules.
  Inline styles remain only where values are genuinely data-driven (per-node marker color/size/symbol computed from the retrieval, and the one `height:100%` the graph container needs).
- Loaded **Inter** and **JetBrains Mono** from Google Fonts via a custom `app.index_string` (allowed here - this is a local dev app, not a sandboxed artifact).

## 3. Design system (light theme)

Per your choice, the theme is **light** but built on the same token architecture the brief proposed, so it can be re-skinned (including back to dark) by editing ~20 variables in `:root`.

Key tokens: `--bg-canvas #eef2f7`, `--bg-panel #ffffff`, `--accent #2b7fff`, `--accent-warm #e07a1f` (GSE/study), `--status-good`, `--status-error`, a 3-step type scale (`--text-lg/base/sm` = 18/14/12 px) plus `--text-xl` for the app title, and a spacing/radius scale.

Dash 4.4 ships its own `--Dash-*` design tokens that drive the built-in dropdown/slider/checklist/input widgets.
Rather than fighting their inline styles, the stylesheet **overrides those tokens** (e.g. `--Dash-Fill-Interactive-Strong` -> `--accent`) so the slider fill, checkbox, focus rings, and dropdowns all pick up the accent color and the light surfaces automatically.

## 4. Layout restructure

Replaced the old equal-weight `290px 1fr 360px` grid of three identical white shadow-cards with a proper three-zone hierarchy plus a header:

- **Top header bar** - brand mark + "bridge-rna" title + tagline + an `ARCHS4 · 940k samples` chip.
  Gives the tool an identity for a NASA/collaborator demo.
- **Left sidebar** - a persistent tool panel (flush, full-height, hard right border, no drop shadow), with controls grouped into labeled subsections (**Query sample**, **Retrieval**, **Metadata enrichment**) separated by dividers.
- **Center workspace** - the network graph as the "main event": a card with a thin accent top border and a colored dot next to its title.
- **Right inspector** - the details panel and the AI-hypothesis panel stacked, reading as contextual cards "in front of" the workspace.

The whole app is a fixed-height, no-page-scroll dashboard; the sidebar, details panel, and AI output scroll internally instead.

## 5. Component work

**Search controls (sidebar).**
Controls grouped and labeled; the disabled **"Compare sample (later)"** dropdown was **removed entirely** (per your choice) so nothing reads as an unfinished stub.
The Search button is now the accent-colored primary action.
The "query running" indicator has an inline spinner and a clear loading message; the Search button disables while running.

**Network graph (`build_network_figure`).**
`paper_bgcolor`/`plot_bgcolor` now match the panel surface (white) instead of the old light-gray `#f7f8fb`.
Nodes recolored for the light theme - query = accent blue star, GSM = steel blue circle, GSE = warm orange diamond (shape encoding preserved).
Node label font now matches the UI sans font.
The redundant in-plot title was removed (the panel header covers it).
A custom **horizontal legend strip** above the graph explains the three node types and states that **edge width = similarity score** - previously a first-time viewer had to guess.
**Label decluttering:** above 12 GSM hits, individual GSM labels are hidden (query + GSE labels stay) so the 30-node case doesn't collide; ids remain on hover.
See `screenshots/07_topk30_declutter.png`.

The graph colors are mirrored in a `GRAPH_THEME` dict in Python (Plotly can't read CSS variables); it's commented to stay in sync with `:root`.

**Details panel (`build_details_panel`) - the biggest structural change.**
Was a flat list of `html.P()` tags with no grouping.
Now built from reusable helpers (`_detail_row`, `_detail_section`, `_detail_text_block`, `_details_head`) into labeled key/value groups:
**Identity**, **Biology**, **Platform & series**, **Study context**, **Publication**.
Each row is a `label | value` pair with consistent alignment; empty values render as a muted em-dash instead of a blank.
GSM hits show a score badge in the header.
Long-form fields (GEO summary, overall design, OSDR study description/publication/protocol) are collapsible `<details>` blocks, collapsed by default, so they no longer dominate the panel.
The FTP link is now a real clickable link.
See `screenshots/03`, `04`, `05`.

**AI summary panel.**
Renamed to "AI hypothesis" with a small "Beta" chip and a colored dot in the header.
The button is a secondary style; while generating, it disables and shows an inline spinner (driven by a loading class toggled via Dash's `running=`, so the spinner only shows during actual generation, not for idle hint text).
The markdown output has its own scrollable region with a max height, so a long summary can't push the details panel off-screen.

**Bar chart (`build_bar_figure`).**
Confirmed it is **not referenced anywhere in the layout** (dead code).
Left it in place but restyled its background/fonts to match the new theme, so it's consistent if it ever gets wired up.

## 6. Also changed

- `app.run(...)` now passes `dev_tools_ui=False` so Dash's floating debug toolbar doesn't overlap the UI during a demo; hot-reload is still on (set `DASH_DEBUG=0` to disable).
- `app.title` updated to `bridge-rna · OSDR → ARCHS4 Explorer`.

## 7. Explicitly left alone

Per the brief, the data flow was not rewritten: `search_hits`, `run_real_retrieval`'s subprocess mechanics, `run_precomputed_query_retrieval`, and the embedding/retrieval logic are unchanged apart from the error-message truncation in section 1.
The callback wiring (`run_search`, `select_node`, `render_details`, `generate_ai_summary`) is structurally the same; only its outputs' presentation changed.

## Note on running it

`.venv/bin/python app_osdr_dash.py` serves on `http://0.0.0.0:8050`.
In this environment a real search reaches the error banner because `archs4py` and `biopython` aren't installed (the retrieval subprocess itself runs; it fails at the GEO-metadata step).
On a machine with those deps and the checkpoint/memmap present, the same UI renders the populated graph and details shown in the screenshots.
