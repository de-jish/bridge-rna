# Bridge RNA NASA visual refinement

This worktree now previews the existing app layout with a shared NASA visual theme. The prior three structural prototypes are preserved at commit `22031a0`; their adapter, script and browser check are historical, and are no longer loaded by the launcher.

```sh
cd /Users/josh/Bridge-RNA/.worktrees/hds-exploration
/Users/josh/Bridge-RNA/.venv/bin/python prototypes/hds/preview.py
```

Open http://127.0.0.1:8065 . The launcher calls the normal `app.build_app()` without layout decoration or callback overrides. Real local embeddings, retrieval, cohort comparison and upload inference are used. No results are simulated. Initial state and selections follow the existing app.

## Design sources and adaptations

- NASA homepage and Glenn: https://www.nasa.gov/ and https://www.nasa.gov/glenn/ . Rendered component styles verified: black header with white text; Inter headings and action labels; Public Sans Web body; `#d83933` homepage action. Computed-style evidence is in `.lavish/nasa-refinement/nasa-reference-styles.json`.
- NASA HDS Core **0.10.0**, pinned in package.json and package-lock.json. Official custom properties and selected font assets are vendored locally; their license files are under `assets/fonts`. Source: https://github.com/nasa/hds-core and https://nasa.github.io/hds-core/ . No full external CSS framework or NASA website stylesheet is loaded.
- Supporting accessibility guidance: https://designsystem.digital.gov/components/button/ . Existing semantic controls, keyboard behavior, visible focus and non-color error signals are retained.

The shared adapter is `assets/00-tokens.css`. It uses HDS neutral surfaces, square panels and 2px control corners, with compact original spacing. Search uses red as explicitly requested, a deliberate departure from HDS's on-page blue action role. Its accessible `#d83933` matches the rendered NASA homepage action instead of the brighter HDS red token. Focus uses a dark blue outline on light surfaces and white in the header. Chart category hues and the map's validated dark canvas are retained independently of brand colors.

The user explicitly overrode the repository's older navy-header/teal-rule visual guidance. Original grid, breakpoints, control ordering, visible sample metadata, details placement and callback behavior remain. At 1440 × 1000 the header is 65.59px high, query rail 288px, workspace 768px and inspector 384px. The graph is unchanged at 686 × 746.41px. Only long rendered query labels gain line breaks to prevent laptop clipping; coordinates, identifiers in hover/customdata and axis calculations are unchanged.

## Verification

```sh
/Users/josh/Bridge-RNA/.venv/bin/python -m pytest tests/test_app.py tests/test_retrieval.py tests/test_cohorts.py tests/test_render.py -q
/Users/josh/Bridge-RNA/.venv/bin/python prototypes/hds/check_refinement.py
```

291 focused tests pass, including shared palette/contrast tests (now resolving HDS CSS aliases). Browser checks cover original geometry, empty/populated retrieval, result metadata, keyboard Search and retrieval depth, PNG export, cohort comparison, upload validation and real inference, Map navigation and 2D/3D, desktop/laptop/narrow layouts. Evidence and screenshots are in `.lavish/nasa-refinement/`. The existing Chrome wrapper returned a missing-pageId error; verification used installed Playwright Chromium.

AI generation was not exercised. External metadata depends on NCBI availability. A pre-existing navigation issue was reproduced on both original port 8050 and refined port 8065: after returning from Map, the empty Upload mode can show an enabled Embed & search button. First-load and invalid-upload disabled states work; callback logic was left unchanged under the styling-only scope. The map keeps its scientific palette and dark canvas; this is a presentation change, not a redesign of its controls or projection behavior.

Branch: `prototype/hds-exploration`. Worktree: `/Users/josh/Bridge-RNA/.worktrees/hds-exploration`. Original source checkout remains untouched; hashes in `baseline.json` verify its preserved tracked state. Local corpus/cache links and the shared Python environment are required to run the real-data preview. No deployment, merge or push was performed.


## Cohesive component refinement

The second pass is based on commit `9b9cfba`, the prior NASA refinement. Before/after evidence at 1440 × 1000 and 1280 × 800 is in `.lavish/nasa-cohesion/`.

The incomplete theme came from four sources: `.status-good` still imposed a green alert and checkmark; the broad `--accent` family served links, selected controls and focus indiscriminately; Map maintained hardcoded blue-gray overlay colors and competing modebar selectors; Dash's inner controls were not fully themed (including two nested dropdown borders and a separate legend-search input). The map spinner also retained an inline turquoise color. The line below Color by is a real coverage indicator, including a degraded partial-coverage branch; it was retained with neutral full coverage and amber partial coverage, with calculations and explanatory text unchanged.

Shared roles now live in `assets/00-tokens.css`, with native and Dash controls in `assets/02-controls.css`. View styles own composition. HDS 0.10.0 supplies typography, neutral color ramps and shape tokens. App-specific adaptations are charcoal selections, neutral outlined secondary actions, neutral routine completion summaries, red primary Search and thin navigation indicators, and stronger blue keyboard focus. These are adaptations for research work, not claims that HDS prescribes these exact controls. True source links retain blue. Semantic errors retain text and warning marks. Dark overlays use neutral HDS surfaces, with high-contrast text and focus; the scientific map canvas remains navy.

Scientific hues, category order, symbol meanings, customdata, hover identifiers, node coordinates, projection logic and retrieval callbacks are unchanged. Network markers now explicitly use opacity 1.0 rather than Plotly's automatic 0.7 for color arrays, matching their existing solid legend swatches. White hit rings, teal query/cohort-A marks, orange network cohort-B/study marks, yellow Map cohort-B marks and the tissue/species palettes are preserved. Map hover labels use neutral chrome rather than category-colored tooltips.

Validation: 295 focused tests, including exact preservation of completion-message content and AA contrast for text, selected controls and dark overlays; 44 real-data browser checks plus 9 interaction-state checks. The latter includes a real delayed Map response to expose loading, without mocked callback results. Run `prototypes/hds/check_cohesion.py` and `prototypes/hds/capture_cohesion.py` with the same shared Python interpreter. Desktop/laptop screenshots use the same OSD-100 query and five hits. 390px layouts also pass overflow checks. AI generation and external enrichment were not exercised. The pre-existing empty-upload enabled state after returning from Map remains outside this presentation scope.
