# Point selection: finding a sample on the map, and inspecting the one you clicked

Implementation plan for two features that turn out to be one.

**Find** (idea 3) answers "where is OSD-100 on this map?".
The map draws 942,563 points and has no way to ask about a specific one.

**Probe** (idea 4) answers "what is this point I am looking at?".
The ARCHS4 cloud is 940,455 of those 942,563 points and none of them can be inspected: only the OSDR overlay carries `customdata`, so `pick_osdr_point` offers a retrieval for a clicked diamond and returns nothing for everything else.

They are one feature with two entry points, and building them as one is what keeps this from being clutter.
A found point and a clicked point are the same thing: **the point you are currently looking at**.
That gives one mark on the plot, one key row, one panel, and one module that turns a gesture into a set of point indices.
Built separately they would be two marks, two panels and two answers to "what is selected".

## What was measured first, and what it killed

Five spikes were run against the real corpus and the running app before any of the design below was written.
Two of them killed the obvious implementation, which is the reason this document exists.

**1. The ARCHS4 cloud emits no click event.**
`render._scatter` sets `hoverinfo="skip"` on any trace with no hover lines, and `skip` suppresses click picking as well as hover.
Measured by attaching a `plotly_click` listener and clicking a dense ARCHS4 pixel 19.3 data units from the nearest OSDR diamond: zero events.
The same click on an OSDR diamond returns a point.
So the probe cannot simply read `clickData`.

**2. Turning hover on to get the click costs 240 ms per mouse move.**
With `hoverinfo="x+y"` restyled onto all 13 cloud traces the point becomes clickable, and synchronous mousemove handling goes to a **median of 239.8 ms, max 243.7 ms** at 942,563 points, sustained rather than a one-time index build.
That is the cost the `hoverinfo="skip"` comment in `render.py` is about, now with a number on it.
It makes the map feel frozen under the cursor and is not acceptable at any budget the 2-D view offers.
`hoverinfo="none"` was tested on the theory that it keeps events while dropping the label: it does not restore click picking either.

**3. A pixel can be converted back to a data coordinate accurately.**
Plotly's own `xaxis.p2d()` round-trips a point's coordinate through pixels with an error of 1.4e-4 data units, against a full-zoom scale of 0.0317 data units per pixel.
That is 0.004 px.
So a plain DOM click listener on the drag layer can produce the coordinate that Plotly's picking would have produced, at no per-frame cost, and the server can resolve it.

**4. Resolving a coordinate to a point is cheap, and restricting it to what is drawn is cheap.**
Brute-force nearest point over all 942,563 coordinates is **0.9 ms**; over the 102,108 drawn at a 100k budget it is **0.5 ms**.
Recomputing the drawn subset with `render._archs4_sample_indices` costs **19.2 ms at the full corpus, 9.8 ms at 100k, 2.8 ms zoomed**, and is deterministic across calls (`seed=7`), so the resolver and the figure agree about what is on screen without sharing state.

**5. A click at full-corpus zoom is genuinely ambiguous, and the number is large.**
The end-to-end spike clicked a known point at full zoom and again after zooming to a 3-unit window.
Both resolved to within 0.09 px and 0.26 px of the click.
But **758 drawn points sit within 3 px of the first click** and 35 within 3 px of the second.
The resolved point was not the one aimed at, and could not have been: at 0.0404 data units per pixel, a point's nearest neighbours are 0.0019 to 0.0092 units away, so hundreds of samples share a pixel.

That last measurement is the one that shapes the design.
A probe that prints one accession for a click on 758 overlapping samples is the same class of error as a stability curve quoted beside a single cohort's name: the number is real and the label is wrong.

## The design

### One module owns "gesture to point indices"

`manifold/selection.py` is new and holds both entry points.
It opens no Dash object and builds no figure, so it is testable against the fixture corpus with no browser, in the same way `bridge_rna/cohorts.py` is testable with neither the embedding nor the memmap.

```python
Selection = dict     # {"points": [int], "source": "find"|"click",
                     #  "query": str, "crowd": int}

def find(query: str) -> Selection            # identity lookup
def resolve_click(x, y, units_per_px, drawn) -> Selection
```

`drawn` is the candidate index array, passed in rather than computed here, so the module never has to know about budgets, viewports or layers.
The callback computes it from the same `render._archs4_sample_indices` the figure used.

### The probe resolves against what is drawn, and says how crowded the click was

`resolve_click` takes the candidate set and returns the nearest point in it, plus `crowd`: how many candidates fall inside the tolerance radius, **counting the resolved point itself**, so `crowd == 1` is an unambiguous click and the panel's "nearest of *n*" quotes `crowd` directly.
Tolerance is **3 px**, converted to data units by the `units_per_px` the clientside listener sends, so it means the same thing at every zoom.

Three rules follow, and each prevents a specific lie:

- **Only drawn points are candidates.** At a 100k budget, 89% of ARCHS4 is not on screen. Returning a record for a point nobody can see would be a click that invents its own target. The candidate set is the ARCHS4 sample the figure actually drew, plus the OSDR block when its layer is ticked, and nothing when a layer is unticked.
- **`crowd > 1` is reported, not hidden.** The panel says "nearest of 758 samples within 3 px" and that is the cue to zoom. At `crowd == 1` it says nothing extra, because an unambiguous click needs no caveat.
- **A click that lands more than 3 px from any drawn point clears the selection.** Clicking empty space means "nothing", not "the nearest thing in this direction".

### The click arrives through a clientside listener, not through `clickData`

This is the first clientside callback in the repository, and it exists because measurement 2 ruled out the alternative.

A clientside callback keyed on `Input("manifold-graph", "figure")` attaches one idempotent `click` listener to the graph's `.nsewdrag` layer.
The listener converts the event's pixel position through `xaxis.p2d()` / `yaxis.p2d()`, reads the data units per pixel off the same axes, and writes `{x, y, ux, uy}` into `plot-click-store` with `dash_clientside.set_props` (Dash 4.4.0, confirmed available).

Cost per frame: zero.
Payload added to the figure: zero.
The 600 KB of ARCHS4 `customdata` that was deliberately removed stays removed, and the docstring in `pick_osdr_point` explaining why stays true.

**Guard.** A test asserts the ARCHS4 traces still carry `hoverinfo="skip"`, so a future change that "fixes" picking by enabling hover fails in the suite rather than in someone's hand at 240 ms per mouse move.

### The probe is 2-D only, and the rail says so

`p2d` has no 3-D equivalent: a 3-D scene has a camera, not an axis-to-pixel map.
The alternative there is Plotly's own picking, which in 3-D is GPU-based and **costs 0.1 ms per mouse move at the 40,000-point cap**, so unlike 2-D it would be affordable.
But a real click on a `Scatter3d` cloud point could not be made to fire `plotly_click` in a headless driver during the spike, and a mechanism that was not made to work is not a mechanism to plan around.

So the probe is 2-D, and in 3-D the panel states that point inspection is 2-D only.
That is the precedent `show_retrieval_group` already sets for the frame button, which is hidden in 3-D because it "would have been a click with no visible effect - the thing this map removed the lasso for".

**Find is not affected and works in both.**
An identity lookup involves no picking, so searching for a sample in 3-D marks it exactly as it does in 2-D.
Confirming the 3-D click is a follow-up spike, not a blocker.

### Find accepts identity, and deliberately not free text

| input | resolves to | measured scale |
| --- | --- | --- |
| `GSM…` | one ARCHS4 point | 940,455 accessions, all `GSM`-prefixed, no duplicate numbers |
| `GSE…` | every sample in that series | 51,284 series; median 9, p95 55, max 8,764 |
| `OSD-###` | every sample in that study | 70 studies; median 20, max 192 |
| an OSDR sample name, or a full `<accession>\|<name>` key | one OSDR point | 2,108 keys, unique |

**Free-text search over titles and characteristics is deliberately not offered.**
`archs4_metadata.parquet` carries `title`, `source_name` and `characteristics`, and a substring match over 940,455 rows is affordable.
It is still wrong here: it would return hundreds of rows for "liver" and be read as a biological query when it is a string match, on exactly the map whose whole design is that a color-by declares what it does and does not describe.
"Where is liver" is the Tissue color-by's question and it already answers it over both corpora.

A query that resolves to nothing says so on the rail and changes nothing on the plot.

### The identity index is integer-keyed

Both accession spaces are a fixed prefix and digits, so the index is the parsed integer, sorted once, queried with `np.searchsorted`.

Measured: **90 ms to build, 15.0 MB, zero duplicate GSM numbers.**
The obvious alternative, `pd.Index(accessions).get_loc`, is **754 ms to build the hashtable and 96.8 MB retained**, which more than doubles the app's 80.8 MB working set for a lookup that runs once per keystroke-submit.
It is rejected on that measurement.

The index is `@lru_cache`d and built on first search, so a session that never searches never pays the 90 ms.
`global_index` is verified to equal row position in `archs4_metadata.parquet`, so an ARCHS4 accession's row **is** its point index, by the same join the rest of the app rests on.

A `GSE` with 8,764 samples is marked and framed in full, with the count stated.
There is no silent cap: if a cap is ever added it states what it dropped, per the rule the map's numerals and the comparison network's labels already follow.

### One mark, keyed once

The selection draws as **`x-open` in white** (`x` in 3-D, which is in `Scatter3d`'s eight-symbol vocabulary).

White rather than a new hue, for the reason `theme.py` already records: no hue clears 3:1 against the worst categorical tissue bucket on the navy canvas, which is why the hit ring is white.
Adding a twelfth hue to sit beside eleven validated ones and two cohort colors would be re-opening a question that was already answered with measurements.
Shape is the free channel, and an X is not a circle, a square, a diamond or a star, so it collides with nothing already drawn.

Per `docs/map_key.md`, the mark gets a key row and the shape gets a `.bm-key-glyph.is-selected` rule.
`test_the_key_glyph_shapes_all_have_a_stylesheet_rule` gains `"selected"`, and `test_the_second_cohorts_hit_symbol_is_valid_in_three_dimensions` gains a sibling for the selection symbol.

The row reads "found sample" or "selected point" with the count beside it, following `_key_row`'s existing rule that the count is what is drawn.

### Where the record renders

The existing hidden `picked-group` on the rail becomes the selection panel, keeping the rail's standing rule that the fact qualifying a control sits under that control: the find input is the control, the panel is its result, and a clicked point lands in the same place.

| selection | panel shows |
| --- | --- |
| OSDR point | sample key, study, tissue, spaceflight arm, and the existing "Retrieve its Earth analogs" link |
| ARCHS4 point | GSM, GSE, title, source name, characteristics, tissue, and a link to the GEO record |
| a set (series or study) | the identifier, how many samples, and the fields they share |
| crowded click | the above, plus "nearest of *n* samples within 3 px" |
| nothing | hidden, exactly as `picked-group` is today |

The GEO link is built through `bridge_rna.geo`'s existing accession normalization rather than by formatting a URL here.
That normalization is the fix for the "Platform 21103" defect and there is no reason for a second spelling of it.

### Wiring, and the single-writer rule

`test_every_output_has_exactly_one_writer` is a real constraint, so nothing below adds a second writer to an existing output.

- `plot-click-store` (new) is written only by the clientside listener.
- `selection-store` (new) is written by one server callback taking the click store, the find submit, and the state the candidate set depends on (budget, viewport, dims, layers).
- `update_figure` gains `Input("selection-store", "data")` and draws the mark. It already owns the figure.
- `update_viewport` gains `Input("selection-store", "data")` and frames **only** when `source == "find"`. A click needs no framing, because you clicked something already on screen. It already owns the viewport, which is why this is an Input there rather than a new callback.
- The selection panel is its own callback off `selection-store`.

`pick_osdr_point` is deleted, and its behaviour is absorbed: an OSDR click now arrives through the same resolver as everything else and produces the same panel, with the retrieval link still on it.

**A selection is a set of point indices, so it survives a projection change** and the mark moves to the new coordinates.
The viewport still resets on that change, as it does today.

## Build order

Each step ends with something runnable, and the two that carry risk come first.

1. **`manifold/selection.py` with the identity index and the click resolver.**
   Pure functions, unit-tested against the fixture corpus.
   No Dash, no browser.
2. **The clientside listener and `plot-click-store`.**
   Verified in a browser that a click on the cloud produces coordinates, and that the resolved point is the nearest drawn one.
   This is the new mechanism, so it is proven before anything is built on it.
3. **`selection-store` and the panel.** Click a cloud point, read its GEO record on the rail.
4. **The mark, the key row, and the CSS.** Click a point, see it marked.
5. **The find input, and framing.** Type `GSE143281`, land on it.
6. **The 3-D pass.** Find works; the probe states that it is 2-D only.
7. **Docs.** `CLAUDE.md`'s package-layout block gains `selection.py`, `progress.md` gains the entry, and this file is updated to record what was actually built.

## Tests

Unit, against the fixture corpus:

- a GSM resolves to its own point, and an OSDR sample name resolves without its accession prefix
- a series resolves to every sample in it; a study resolves to every sample in it
- an unknown identifier resolves to nothing rather than to a guess
- free text that appears in a title resolves to nothing, pinning the scoping decision
- a click resolves to the nearest **drawn** point, and ignores points the budget is not drawing
- a click ignores the ARCHS4 block when its layer is unticked, and the OSDR block when that one is
- a click more than 3 px from any drawn point selects nothing
- the crowd count is reported and matches a brute-force count at the same tolerance
- the selection survives a projection change, because indices are projection-independent
- **the ARCHS4 traces still carry `hoverinfo="skip"`** - the 240 ms guard
- the selection symbol is valid in `Scatter3d`
- the key glyph has a stylesheet rule
- every output still has exactly one writer

Browser, added to `tests/e2e_check.py`:

- clicking a cloud point opens the panel with an accession that matches the coordinate
- clicking empty space clears it
- finding a GSM frames and marks one point; finding a GSE marks its whole set
- the mark is in the key, with the right count
- the probe is absent in 3-D and find still works there

Then the standing suites: 346 pytest, 51 + 70 + 156 browser checks, and `tests/screenshot_readme.py`, since the rail gains a control and the map screenshot is measured against its own content height.

## Rejected alternatives

**Attach `customdata` to the ARCHS4 traces.**
The direct way to make a click identifiable, and the reason it was removed still holds: about 600 KB of dead payload per figure, on every zoom step.
The coordinate resolver gets the same answer for nothing.

**Enable hover on the cloud, permanently or behind a toggle.**
Measured at 240 ms per mouse move.
A toggle would confine the cost to users who asked for it and would still be a control whose honest label is "make the map slow".

**Reconstruct the point from `curveNumber` and `pointNumber`.**
Works in principle, since the drawn subset is deterministic, but it makes the click handler depend on the exact trace order `_categorical_traces` emits, including the residual-first ordering.
The coordinate is the stable identifier; trace order is an implementation detail of the renderer.

**Free-text search over titles and characteristics.**
Affordable and wrong.
See above.

**A separate "found" mark and a separate "picked" mark.**
Two marks, two key rows and two panels for one idea.
The unification is the point.

## Open questions

- **The 3-D click.** Hover picking there is GPU-based and costs 0.1 ms, so the probe is probably affordable in 3-D; the spike could not make `plotly_click` fire on a `Scatter3d` cloud point in a headless driver. Worth one more attempt with a real pointer before accepting the 2-D-only scope permanently.
- **Whether `crowd` should suppress the record entirely** above some threshold, rather than showing the nearest with a caveat. Showing it with the count stated is the proposal, on the grounds that the record is still a real sample and the count is the honest qualifier. If a reader is observed treating the caveat as decoration, suppression is the fallback.
