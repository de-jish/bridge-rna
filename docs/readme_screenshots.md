# The two README screenshots, and why they are measured rather than framed by hand

`README.md` embeds two images, one per view: `docs/bridge-rna-interface.png` and `docs/bridge-rna-map.png`.
`tests/screenshot_readme.py` captures exactly those two and nothing else.
It is a sibling of `tests/screenshots.py` with a narrower job: that one walks both views and composes a fourteen-frame gallery at a fixed 1680x1010, this one produces the two frames the README ships and refuses to accept a fixed viewport.

## The defect this was written to fix

The shipped `docs/bridge-rna-interface.png`, captured 2026-07-22, was cut off.
The inspector ended mid-record, so the top hit's publication, journal and DOI were missing from the frame, and the retrieval network's lowest node was sliced in half by the bottom of the image.

That is not a framing mistake, it is a property of the layout meeting a viewport that was too short.
Both views are fixed-height instruments that scroll internally: `assets/01-shell.css` gives the shell `height: 100%`, and every panel under it carries `min-height: 0` with its own `overflow-y: auto`.
A page like that never grows past the window, so a window shorter than the content does not produce a scrollable screenshot.
It produces a silently clipped one, and the clipping is invisible in the capture script's output.

Measured on the real app at the viewport the old capture used:

| viewport | `.sidebar` | `#details-panel` |
| --- | --- | --- |
| 1680x1010 | 26 px hidden | 410 px hidden |
| 1680x1400 | fits | 20 px hidden |
| 1680x1444 | fits | fits |

The 410 px is the same 410 px named in the comment at `assets/retrieve.css:21-28`, which records the day the row track became `minmax(0, 1fr)` so a long GEO record would scroll the inspector instead of pushing the whole page below the fold.
The CSS fix was correct and is still in place.
The screenshot simply predated it and was never retaken.

## Two ways a frame can be cut, and only one of them is visible to the DOM

**A panel can clip its content.** `scrollHeight > clientHeight` on any scroll container says so exactly, and `fit_viewport` grows the window until no container reports it.
The window is re-measured after each step rather than computed once, because a taller window changes what the panels lay out.

Two false positives have to be excluded or the loop never converges.
`.visually-hidden` clips a 1x1 box on purpose, which is how a Dash control that renders no labelable element gets an accessible name, and Dash's own checkbox wrappers are 1x1 for the same reason.
Both report overflow forever and neither is visible, so the check ignores anything under 40 px in either dimension.

**A figure can run off its canvas, and the DOM cannot see it at all.**
A Plotly canvas is exactly as big as its container whether or not the drawing inside it fits.
The first capture of the map made this concrete: a 3-D camera dollied in by two mouse-wheel events reported no overflow anywhere on the page while the point cloud and the bottom row of tick numerals ran off the bottom edge.

So the figure is measured as pixels.
`EDGE_INK_JS` re-renders it through `Plotly.toImage` and counts how much of the outer band differs from the paper colour, which is read out of the corner of the image rather than assumed so the same code works on the map's navy canvas and the retrieval network's white one.
Rendering through `toImage` rather than reading the live canvas is deliberate: it returns the figure alone, so the floating key and the plot badges sitting over the map do not count as ink.

Two bands, because "cut off" and "uncomfortably close" are different faults and only one is a defect:

- `CUT_BAND`, 3 px. Ink here means the drawing continues past the boundary, which is a glyph or a numeral with its other half missing. This is the hard failure, asserted after the shot is taken.
- `COMFORT_BAND`, 14 px. This is the margin the framing aims for. Failing it costs a wider camera, not a failed run.

## Framing the 3-D scene

The camera is set outright with `Plotly.relayout`, not dollied with wheel events.
A wheel step is a fixed fraction of the current distance, so a loop of wheel events cannot ask for a particular framing and cannot be repeated after a resize, which is exactly how the first attempt overshot.

`eye` is a unit direction times a distance, and only the distance is searched.
The winner is the framing with the most ink on the canvas among those leaving the comfort band clean, which is "as large as it fits" stated as a number rather than as a judgement.
Measured at 1680x1010, tissue colouring, 40k ARCHS4 budget:

| distance | canvas carrying ink | ink in the outer 14 px | |
| --- | --- | --- | --- |
| 1.60 | 7.97% | 0.247% | touches the edge |
| 1.75 | 7.20% | 0.166% | touches the edge |
| 1.90 | 6.52% | 0.126% | touches the edge |
| 2.05 | 5.94% | 0.089% | touches the edge |
| **2.20** | **5.44%** | **0.000%** | **chosen** |
| 2.40 | 4.87% | 0.000% | smaller for nothing |

The direction is fixed at `(0.68, 0.68, 0.28)` rather than searched.
It was searched first, over three candidates at three canvas heights, and the three scored within 0.01% of each other on fill, which is noise.
Letting noise pick the camera angle means the frame changes shape between runs for no reason, so the angle is a stated compositional choice: slightly above the cloud and off-square, so the corpus reads as a volume rather than as a flat sheet.

Canvas height was searched at the same time and does not earn its cost.
Growing the map window from 1010 px to 1450 px moves fill from 5.44% to 5.83%, because what limits the 3-D frame is the sprawl of the x and y tick numerals across the bottom, not the height of the canvas.
The map is therefore captured at its natural fitted height and the retrieval frame is the tall one.

**Only what a user could do.** The camera is a user action, since rotating and zooming the scene is what the modebar and the drag handle are for.
`scene.domain`, axis visibility and marker sizes are not, so none of them are touched.
A screenshot that reframes the app by editing the figure is no longer a screenshot of the app.

## What the two frames land at

| frame | viewport | image | edge band |
| --- | --- | --- | --- |
| `bridge-rna-interface.png` | 1680x1444 | 3360x2888 | 0.000% |
| `bridge-rna-map.png` | 1680x1010 | 3360x2020 | 0.000% |

Both at `device_scale_factor=2`, so the type is retina-sharp at the width a README renders them.
The two heights differ because the two views need different amounts of room, and forcing them to match would mean either clipping the inspector again or padding the map with empty canvas.

Each frame is captured in its own browser context, so the map is a clean map rather than one carrying the retrieval the previous frame left in `hits-store` on the shell.

## One truncation that is not a defect

The rail's OSDR sample dropdown reads `Mmus_C57-6J_EYE_FLT_Re...`.
That is a `text-overflow: ellipsis` inside a fixed 288 px rail, not a clipped frame: sample keys are arbitrarily long, the control is doing the right thing, and the full key `Mmus_C57-6J_EYE_FLT_Rep1_M23` is printed in full in the query card immediately below it and again as the query node's label in the network.
Do not widen the rail to make it go away.

## Rejected

**`full_page=True`.** It captures the document, and on a `height: 100%` shell with `overflow: hidden` the document *is* the viewport. It would have reproduced the clipped frame exactly.

**Cropping or scaling the image afterwards.** A README screenshot is evidence about the app. Anything that edits the pixels after capture makes it evidence about the editing.

**Shrinking the page with CSS `zoom` to make the content fit.** It fits the content by making the app's type smaller than the app's type, which misrepresents the interface at exactly the moment the image is meant to represent it.

**Capturing the map in 2-D, which fills its canvas far better.** It does, and it is a different claim: the README's map paragraph is about a corpus you can rotate. 2-D is one pill away in the app and the README already says so.

## Running it

```bash
/Users/josh/Bridge-RNA/.venv/bin/python tests/screenshot_readme.py           # both, straight into docs/
/Users/josh/Bridge-RNA/.venv/bin/python tests/screenshot_readme.py --only map
/Users/josh/Bridge-RNA/.venv/bin/python tests/screenshot_readme.py --out /tmp/shots --headed
```

It boots `app.py` against the real `cache/`, so it needs the artifacts and the LFS objects, and it takes about three minutes.
It exits non-zero if either frame ends up with a clipped panel, a figure touching its canvas edge, or a console error, so a layout change that breaks the images fails the capture rather than shipping a cut one.
