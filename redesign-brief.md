# bridge-rna UI/UX Redesign Brief

Context for whoever (whatever) is implementing this: `app_osdr_dash.py` is a single-file Dash app,
~1718 lines, currently styled entirely with inline Python `style={...}` dicts. There is no
`assets/` folder and no external stylesheet. This brief covers a full visual redesign plus one
real bug fix found while reviewing the code. Audience is "demo to NASA/collaborators," so this
should look like a credible scientific tool, not a toy.

## 0. Bug fix (do this regardless of anything else below)

**Location:** `run_real_retrieval` (~line 1074) and `run_search` (~line 1595-1607).

`run_real_retrieval` shells out to `demo_osdr_top5.py` via subprocess and, on failure, does:
```python
raise RuntimeError(f"Demo retrieval failed: {msg[:500]}")
```
where `msg` is raw subprocess stderr - i.e. a Python traceback. `run_search` catches this and
prepends `"Real retrieval failed: "` and shows the whole thing, traceback and all, in the
`search-status` div. This is what's producing the wall of monospace traceback text visible in
the current UI.

Fix:
- In `run_real_retrieval`, when the subprocess fails, log the full stderr server-side
  (`print`/`logging`, whatever's already used in this repo) and raise `RuntimeError` with **just
  the last non-empty line of stderr** (that's almost always the actual exception message, e.g.
  `RuntimeError: Requested OSDR sample not found`), not the full traceback.
- In `run_search`'s except block, render this as a clean one-line status message (see the new
  `.status-banner.status-error` class below), not string concatenation into a paragraph.
- Optional but recommended: keep the full raw error in a `dcc.Store` and add a small "Show
  details" toggle that expands a `<pre>` with the full text, collapsed by default. This preserves
  debuggability without putting a stack trace in the primary viewport.

## 1. CSS architecture

- Create `assets/style.css`. Dash auto-loads anything in `assets/` - no `external_stylesheets`
  wiring needed, just create the folder next to `app_osdr_dash.py`.
- Remove essentially all inline `style={...}` dicts from `app.layout` and the `build_*` functions.
  Replace with `className` and CSS rules. Inline styles should remain only for values that are
  genuinely dynamic/data-driven (e.g. per-node marker color computed from a score) - static
  layout/spacing/color/typography should all move to CSS.
- Google Fonts is fine to pull in via a `<link>` in `assets/` (Dash supports a custom
  `index_string` on the `Dash()` app object for this) - no CDN restrictions here since this is a
  local dev app, not a sandboxed artifact.

## 2. Design system

Suggested palette - dark slate/navy base with a single accent, avoiding the current "generic
pastel dashboard" look:

```css
:root {
  --bg-canvas: #0f1620;      /* page background - deep space-adjacent navy, not pure black */
  --bg-panel: #161f2c;       /* card backgrounds */
  --bg-panel-raised: #1d2836;/* hover/active panel state */
  --border-subtle: #2a3646;
  --text-primary: #e8edf3;
  --text-secondary: #8fa0b3;
  --accent: #4fb0ff;         /* primary interactive accent - links, active states, query node */
  --accent-dim: #2c5c80;
  --accent-warm: #ff9f45;    /* secondary accent for GSE/study-level elements, sparingly */
  --status-good: #3ecf8e;
  --status-error: #ff6b6b;
  --font-sans: 'Inter', 'Segoe UI', -apple-system, sans-serif;
  --font-mono: 'JetBrains Mono', 'SF Mono', Consolas, monospace;
  --radius-sm: 6px;
  --radius-md: 10px;
  --space-1: 4px;
  --space-2: 8px;
  --space-3: 12px;
  --space-4: 20px;
  --space-5: 32px;
}
```

Rationale: this is scientific instrument data (space biology, gene expression), a dark
technical theme reads more credible for a NASA demo than the current light "generic SaaS
dashboard" gradient. If Josh prefers a light theme instead, keep the same token structure
(`--bg-canvas`, `--bg-panel`, etc.) but swap values - the point is centralizing every color/space
value into named tokens instead of scattered inline hex codes, so the whole theme can be
re-skinned by changing ~15 variables instead of hunting through 1700 lines.

Typography scale: 3 sizes total. `--text-lg` (18px, panel titles), `--text-base` (14px, body/
labels), `--text-sm` (12px, metadata/secondary). Currently the app uses `html.H3`/`H4`/`P` with no
consistent scale - standardize on these three.

## 3. Layout restructure

Current: CSS grid `290px 1fr 360px` (controls | graph | details), all three columns equal visual
weight via identical white-card-with-shadow treatment.

New structure - same three-zone idea, different visual hierarchy:
- **Left sidebar** (fixed ~280px): search controls. Should read as a persistent tool panel, not a
  card floating among other cards - full height, subtly different background from canvas
  (`--bg-panel` vs `--bg-canvas`), no drop shadow, a hard right border instead.
- **Center canvas**: the network graph is the primary artifact - give it the most horizontal
  space and a clear visual anchor (a thin top border in `--accent`, or a small colored dot next to
  the panel title) to establish it as the "main event." Add a compact horizontal legend strip
  above or below the graph: colored dot + label for query/GSM/GSE, plus a note on what edge
  thickness encodes ("edge width = similarity score").
- **Right sidebar**: details panel + AI summary panel, stacked. These can keep card-style
  treatment (they're secondary/contextual) but should visually read as "in front of" the canvas
  rather than equal-weight siblings.

Concretely: give sidebar and canvas different background tokens so the eye immediately parses
"tool panel vs. workspace vs. inspector," matching how tools like Figma or Blender organize
persistent controls vs. canvas vs. inspector.

## 4. Component-specific work

### Search controls (left sidebar)
- Dash's default `dcc.Dropdown`, `dcc.Slider`, `dcc.Input`, `dcc.Checklist` all render their own
  internal DOM/class names (`.Select-control`, `.rc-slider`, etc.) - style these via CSS overrides
  targeting those classes, not by fighting Dash's inline styles. Check what Dash version is
  pinned (see `requirements.txt`/`pyproject.toml` if present) since dropdown internals changed
  between `dash-core-components` versions.
- Group related controls visually: e.g. put "OSDR study" + "OSDR sample" in one labeled subsection,
  "Top-k" + "Entrez email" + "Biopython toggle" in another, separated by a subtle divider
  (`border-top: 1px solid var(--border-subtle)`), rather than one undifferentiated vertical stack.
- The "Compare sample (later)" disabled dropdown reads as unfinished/broken to an outside viewer.
  For a NASA demo, either hide it entirely (feature-flag it out until it's real) or give it a small
  "Coming soon" badge so it reads as intentional roadmap, not a forgotten stub.
- Search button: make the primary action visually distinct (accent-colored, not a plain bordered
  button) so it's unambiguous as *the* button to press.

### Network graph (`build_network_figure`)
- Set `paper_bgcolor`/`plot_bgcolor` to `--bg-canvas`'s value (currently `#f7f8fb`, a light
  gray - needs to change if going dark theme) so the Plotly chart doesn't sit in a mismatched
  white/light box against a dark surrounding UI.
- Node colors: currently query=black, GSM=blue (`#1f77b4`), GSE=purple (`#6a3d9a`) on white. On a
  dark theme, recolor to `--accent` (query), a lighter blue-white for GSM, `--accent-warm` for GSE
  - keep the shape encoding (star/circle/diamond) as-is, it already works.
- Add an actual Plotly legend or a custom HTML legend strip (simpler, more control) instead of
  `showlegend=False` everywhere - right now a first-time viewer has to guess what shapes/colors
  mean.
- Text labels (`node_df["label"]`) currently sit directly on the plot in default Plotly font -
  set `textfont` to match `--font-sans` and a size that doesn't clutter when there are 30 GSM
  nodes (top-k slider goes to 30). Consider hiding labels by default and showing them only on
  hover/click at high node counts, since 30 always-on text labels will visually collide.

### Bar chart (`build_bar_figure`)
Currently uses Plotly's `Blues` colorscale - fine to keep, but should match the new
`paper_bgcolor`/`plot_bgcolor` and title font treatment for consistency with the network graph.
Note this function exists but I don't see it called anywhere in the layout - confirm whether it's
dead code or wired up elsewhere before styling it.

### Details panel (`build_details_panel`)
This is the biggest structural change. Currently a flat list of `html.P()` tags, one per field, no
grouping (visible in the screenshots - Study Title, Study ID, Sample ID, Tissue, Condition,
Strain, Sex, Duration, Study description all render as an undifferentiated vertical list).

Restructure into labeled subsections with a consistent `.detail-row` pattern:
```
.detail-row { display: flex; justify-content: space-between; padding: 6px 0;
              border-bottom: 1px solid var(--border-subtle); }
.detail-row .label { color: var(--text-secondary); font-size: var(--text-sm); }
.detail-row .value { color: var(--text-primary); font-size: var(--text-base); text-align: right; }
```
Group into: **Identity** (Sample ID, Study ID, GSM/GSE), **Biology** (Species, Tissue, Condition,
Strain, Sex, Duration), **Study context** (Title, Description, Protocol - these are long-form text,
render as a separate full-width block below the key/value rows, not squeezed into the same
row pattern), **Publication** (PubMed fields). Long text fields (`geo_summary`, `geo_design`,
study description/protocol) should be collapsible (`<details>`/`<summary>` or a Dash-friendly
equivalent) since they're multi-paragraph and currently dominate the panel via `html.Pre`.

### AI Summary panel
Currently a button + status text + `dcc.Markdown` output, all in one card. Give the button a
loading state (Dash's `running=[...]` already toggles `disabled` and status text - lean into that
with a small inline spinner rather than just button-disable). The markdown output should get its
own scrollable sub-region with a max-height so a long AI summary doesn't push the details panel
off-screen.

## 5. Things to leave alone

- The core callback wiring (`run_search`, `select_node`, `render_details`, `generate_ai_summary`)
  is functionally sound - this is a visual/structural pass, not a rewrite of the data flow.
- Don't touch `search_hits`, `run_real_retrieval`'s subprocess mechanics, or the embedding/
  retrieval logic beyond the error-message truncation described in section 0.

## 6. Suggested execution order

1. Bug fix (section 0) - independent of everything else, do it first.
2. `assets/style.css` + CSS variables + font loading.
3. Layout restructure (section 3) - get the three-zone skeleton right before polishing components.
4. Search controls styling.
5. Network graph recolor + legend.
6. Details panel restructure (biggest chunk of work).
7. AI summary panel polish.
8. Bar chart, if confirmed it's actually used.
