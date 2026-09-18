# Bridge RNA HDS exploration prototypes

Three layouts on one separately launched Dash app. Nothing changes the production entry point.

```sh
cd /Users/josh/Bridge-RNA/.worktrees/hds-exploration
/Users/josh/Bridge-RNA/.venv/bin/python prototypes/hds/preview.py
```

Open http://127.0.0.1:8065/?variant=b. Options: `a` visualization first, `b` research workspace, `c` guided exploration. Switching uses the same mounted controls and in-memory state. Initial sample: `OSD-100|Mmus_C57-6J_EYE_FLT_Rep1_M23`, five real matches. Reload starts a fresh session.

The existing `app.build_app()` registers all scientific callbacks. `preview.py` decorates the layout and adds a read-only result-list projection of `hits-store`; selecting a row opens the existing inspector. Figure wrappers change presentation only. `preview.js` owns layout selection, expansion, anchor navigation, and explicitly labeled UI state examples. There are no simulated scientific results.

## HDS source

Installed with `npm install --save-exact --prefix prototypes/hds @nasa-hds/core@0.10.0`. The lockfile pins its automatically installed USWDS peer. Reinstall with `npm ci --prefix prototypes/hds`.

Authority: https://github.com/nasa/hds-core and https://nasa.github.io/hds-core/ . Verified package exports, `_custom-properties.scss`, `_button.scss`, and bundled font assets. Supporting guidance: https://designsystem.digital.gov/components/accordion/ . NASA insignia: https://www.nasa.gov/wp-content/themes/nasa/assets/images/nasa-logo.svg .

Actual HDS CSS and fonts load locally. No second CSS framework is installed. The original app CSS supports the existing markup, with one shared adaptation stylesheet. On-page actions use HDS secondary blue, navigational review links use HDS red. Adaptations: existing navy and teal identity; existing scientific colors; compact type and full-width scientific workspace; Dash native widgets themed to HDS; stronger blue keyboard outline in place of HDS's documented low-contrast Carbon 30 focus treatment.

## Scope and limitations

All sample, cohort, upload, map, metadata and optional AI capabilities remain registered. The map itself is not redesigned. AI generation needs the configured provider; no AI generation was performed for this design review. NCBI metadata depends on its service. Loading, empty and error examples are clearly labeled presentation overlays and do not alter production error handling.

The local cache and embedding index are symlinked to the source checkout. They are read by existing scientific code. Model/data tracked with Git LFS must be present for upload embedding. The shared Python environment is `/Users/josh/Bridge-RNA/.venv`.

The source checkout had existing uncommitted changes. Commit `6d7c8f4` captures a copy of its relevant tracked state as this branch's baseline. `baseline.json` records source hashes. Prototype changes are the subsequent files under `prototypes/hds` and `.lavish`; scientific modules are unchanged relative to that baseline. Nothing has been merged or deployed.

## Verification

```sh
/Users/josh/Bridge-RNA/.venv/bin/python -m pytest tests/test_app.py tests/test_retrieval.py tests/test_cohorts.py -q
/Users/josh/Bridge-RNA/.venv/bin/python prototypes/hds/check_preview.py
```

230 focused tests and 43 browser checks passed. Final screenshots additionally checked 1280 px laptop layouts. Browser checks and screenshots live in `.lavish/`. The Chrome DevTools wrapper failed with a missing `pageId` validation error, so browser checks use the already-installed Playwright Chromium.

The comparison artifact is `.lavish/hds-comparison.html`, opened through Lavish for choice and annotation. Use the preview switcher for state-preserving comparison. Standalone links start independent sessions.

Recommendation: research workspace for repeat scientific queries; visualization first for inspecting established results; guided exploration for initial onboarding. Awaiting the user's direction before any broader redesign.
