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
