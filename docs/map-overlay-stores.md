# The map's derived retrieval stores

Status: built 2026-10-21 on `feature/retrieval-table`.

## The problem

`hits-store` lives on the shell and is the source of truth for a retrieval.
At k=1,000 it is about 472 KB of JSON for one sample and about 988 KB for a cohort comparison, because every hit carries its full GEO record (title, characteristics, summary, design, PubMed ids) and each arm carries a 250-row evidence neighborhood.

Dash posts every Input and State with every callback call.
The retrieval view had already been changed so nothing that fires per interaction takes `hits-store`.
The map had not.
Measured in the browser before this change, after a real 1,000-match single-sample search:

| interaction on /map | callbacks posting `hits-store` | upload |
| --- | --- | --- |
| arriving | 6 | about 2.8 MB |
| one zoom | 2 (viewport, figure) | about 945 KB |
| a cohort tick | 2 (figure, rail summary) | about 945 KB |
| Fit results | 5 | about 2.4 MB |
| a drawer tab or one search keystroke | 2 | about 1.0 MB |
| switching to 3-D | 3 | about 1.4 MB |

A comparison doubles every number.

## What the map actually reads

Of the 12 fields a hit carries, the map reads three: `archs4_index` (where to draw it), `gsm` (its hover title) and `score` (its hover cosine).
Of the 10 fields an evidence row carries, the figure reads seven (`archs4_index`, `rank`, `gsm`, `gse`, `score`, `tissue`, `species`); only the drawer reads the long text (`title`, `source_name`, `characteristics`).
The rail reads almost nothing: how many cohorts there are, their names, two counts, and - when Fit results is pressed - one window.

Measured on a real k=1,000 comparison: per arm, the hit list is 380 KB of which those three fields are 42 KB, and the neighborhood is 83 KB of which the figure's seven fields are about 21 KB.

## The design

One server callback, `derive_map_stores`, reads `hits-store` and writes three stores that live in the map view.
Each per-interaction callback reads only the store cut to what it reads.

| store | contents | read by |
| --- | --- | --- |
| `map-overlay-store` | the payload cut to the fields above, in the payload's own shape | the figure |
| `map-evidence-store` | both arms' neighborhoods in full, plus the labels and mode the drawer prints | the drawer |
| `map-retrieval-store` | cohort roles and names, the distinct-hit and shared counts, the query label, and Fit results' window for every 2-D projection and tick combination | the rail, the viewport, the drawer's open state |

Two callbacks only need to know *that* a new retrieval arrived (the drawer closes and its focus resets).
They take `hits-store.modified_timestamp`, a number, which changes exactly when the data does.

### Why the cut stores keep the payload's shape

`map-overlay-store` and `map-evidence-store` are the hits-store payload with fields removed, never a new structure.
So `_retrieval_overlay`, `_neighborhood_overlays`, `neighborhood_drawer_state` and `_frame_for` run unchanged on them, and the proof that the map draws what it drew before is a plain equality: the same function on the cut store and on the full payload returns the same value.
`tests/test_map_stores.py` asserts that for a single sample, an upload, a cohort and a comparison, for every tick combination, with the drawer open and with a focus, and finally that `render.build_figure` produces byte-identical figure JSON from either.
A new field read by the map has to be added to `MAP_HIT_FIELDS` or `MAP_NEIGHBOR_FIELDS`; the figure-equality test fails until it is.

### Why Fit results' window is precomputed

`update_viewport` fires on every relayout, which is every zoom and pan, and it needs the retrieval only when the trigger is Fit results.
Taking even the hit indices as State would post them on every zoom.
So `derive_map_stores` runs the existing `_frame_for` for each of the three 2-D projections and each of the three possible tick sets (A, B, both), and `update_viewport` looks the window up.
Nine windows of four floats; the lookup is asserted equal to `_frame_for` on the full payload for every combination.
3-D has no window, as before.

### Why the stores live in the map view rather than on the shell

On the shell, the derivation would run on every search, on the retrieval view, whether or not anyone opens the map, and on every full page load.
In the view, it runs once each time the map mounts and never on the retrieval view.
It takes `map-workspace.id` as an Input for the same reason `derive_results` takes `results-panel.id`: after a client-side route change Dash reruns a callback only when one of its inputs is newly mounted, and `hits-store` never remounts.
The figure does not build twice on arrival: its Input is the derivation's Output, and Dash holds a callback until the callbacks producing its inputs have answered.

### Server-side derivation, not clientside

A clientside projection would avoid even the one upload per visit.
It would also put a second copy of "which fields the map reads" in JavaScript, where pytest cannot hold it equal to the Python that reads them.
One upload per visit, against the per-interaction uploads above, is the trade worth making.

## What it measures now

Measured in the browser after the change, on the same real 1,000-match searches: the OSD-100 left-eye sample, and the first OSD-137 cohort comparison the picker offers.
Bytes are the JSON of the retrieval stores each request carried.

| what | single sample | comparison |
| --- | --- | --- |
| `hits-store` in session storage | 472 KB | 945 KB |
| arriving: the derivation, the one call that carries `hits-store` | 502 KB | 1,006 KB |
| one zoom: the viewport callback | 0.8 KB | 1.1 KB |
| one zoom or tick: the figure | 114 KB | 229 KB |
| one tick: the rail summary | 0.8 KB | 1.1 KB |
| one drawer tab or search keystroke | 88 KB | 174 KB |
| Fit results: the drawer's open state | 0.8 KB | 1.1 KB |
| switching to 3-D: every callback but the figure | 0.8 KB | 1.1 KB |

Every map callback except the figure and the drawer now carries at most about 1 KB of the retrieval, independent of k.
A zoom went from about 945 KB of upload to about 115 KB for one sample, and from about 1.9 MB to about 230 KB for a comparison.
The figure still builds once on arrival, not twice.

For scale, the figure callback's *response* for that single-sample search is 11.2 MB, so the 114 KB it now uploads is about 1% of that callback's traffic.
The cut is larger than the 63 KB its values alone would suggest because JSON repeats each field name on every row; a columnar layout would roughly halve it, and was not taken because it would give up the payload shape the equality proof rests on, for about 0.5% of the callback's traffic.

`tests/e2e_results_check.py` asserts the bounds on every run: on arrival `hits-store` is posted once and only to the derivation; no zoom or tick posts it; every callback but the figure carries at most 4 KB of retrieval data; the figure carries at most a third of the payload; and the map still draws all 1,000 rings.

## The floor, and what was not done to go under it

The figure callback rebuilds the whole figure on every zoom, so it has to receive every hit it draws.
At k=1,000 its values are about 42 KB per arm, plus about 21 KB of evidence rows, and no stateless design posts less while keeping the hover exact.
The drawer is the same: it searches and prints 250 rows of GEO text per arm, so it receives them.
Three ways under it were considered and rejected:

- **A server-side cache keyed by a digest**, so requests carry only a key.
  Production runs two gunicorn workers (`deploy/apply.py`), so a process-local cache misses half the time; a shared on-disk cache adds eviction and a cold-miss path that would draw a map without its retrieval.
- **Patching only the cloud on a zoom** (`dash.Patch`), leaving the retrieval traces in the browser.
  That needs a second writer to `manifold-graph.figure`, which `test_every_output_has_exactly_one_writer` forbids for the race reasons recorded in `CLAUDE.md`.
- **A packed encoding** (base64 int32 indices and float32 scores, GSM labels re-derived from the index).
  It saves about 60% of the remainder, but it changes the payload's shape, and re-deriving labels from the index makes the overlay depend on a second join rather than on what the retrieval returned.
