# In-app help: "i" explanations and the guided tour

Requested by Joshua on 2026-10-27.
This is the implementation doc: the plan, the decisions and the tradeoffs, kept current as the work lands.

## What he asked for

1. Small "i" icons throughout the app, each opening a short description of how that feature works plus a couple of lines of the code behind it.
   Many is fine; he will remove the ones he finds unnecessary later.
2. One large entry point that opens a centered, multi-slide walkthrough of a typical flow: selecting a sample, getting its matches, reading metadata, opening GEO, and viewing the result on the map.

## Decisions from the interview

| Question | Answer |
| --- | --- |
| What an "i" shows | Plain explanation (2-3 sentences) plus the real lines of code, each naming its file |
| How an "i" opens | Click or tap to open a popover beside it; click outside or Esc to close |
| Guide style | Slides with real screenshots, Back/Next, never touches the live page |
| How the guide opens | A "How it works" button in the header on both pages, plus once automatically on a first visit |
| Guide flows | "Sample to map" and "Explore the map" |
| Guide scope | One guide with chapters; it opens at the chapter for the page you are on |

## Design

An independent assessment by Codex before building changed four things, recorded where each applies below: the popover's DOM ownership, its positioning, how excerpts are checked, and when the guide may open by itself.

### The "i" popovers

- **One registry, `bridge_rna/help.py`.**
  Every entry is a key, a title, a short explanation, and optional code excerpts naming their files.
  `help.info(key)` returns only a button carrying `data-info="<key>"`; removing an explanation from the page is deleting that one call.
- **The explanations render once, on the shell, and the script copies one into a popover it owns.**
  `help.library()` renders every entry, hidden, beside the header and view slots in `app.py`.
  `assets/info-popover.js` creates a single popover on `<body>` and fills it from the library with `cloneNode`.
  This answers Codex's DOM-ownership point: the script never reparents, rewrites or removes a node Dash rendered, so React never reconciles over it.
  It also means a key can appear twice on a page without duplicate ids.
  The first design rendered a hidden panel beside every button; that would have been clipped by every scroll container a button sits in.
- **The popover is `position: fixed` on `<body>`, so no ancestor can clip it.**
  It opens below its button, flips above when there is more room there, is clamped to the viewport, and caps its height to the room it has.
  It closes on any scroll or resize rather than chasing its button, on an outside click, and when its button leaves the page, which is what a Dash re-render looks like from the script.
  Codex flagged that clamping alone cannot beat `overflow: hidden` ancestors; a body-level node can.
- **The code is real code, and a test keeps it real.**
  An excerpt is blocks of lines separated by `...`.
  `tests/test_help.py` requires each block to appear in its cited file as consecutive lines, indentation aside, and the blocks to appear in the order quoted.
  Checking each line on its own, the first plan, was rejected on Codex's review: it lets an excerpt stitch unrelated lines into code that never existed.
  Extracting lines by number at runtime was rejected because it breaks silently on any edit above them and would ship source files the server does not otherwise need.
- **Accessible by construction.**
  The trigger is a real `<button>` named "About <feature>" with `aria-haspopup="dialog"`; the script sets `aria-expanded` and `aria-controls`, which Dash never sets.
  The button sits beside its label, never inside a `<label>` or `<summary>`, and a test walks both views to enforce that.
  The popover is a non-modal dialog labelled by its title: focus moves into it, Esc and the close button return focus to the button, and tabbing out of it closes it.
  A wide code block gets a tab stop so a keyboard can scroll it.
  The visible ring is 16 px; the target is 24 px (WCAG 2.5.8).
- **Placed beside 32 controls and headings** across the retrieval rail, the results table and its filters, the plot switch, the inspector, the AI summary, and the map rail and neighborhood explorer.
  `tests/test_help.py` fails if an entry is not placed anywhere or a placement names a missing entry.

### The guide

- **A native `<dialog>` on the shell (`bridge_rna/guide.build_guide`)**, beside the header and view slots, so a route change cannot destroy it while it is open.
  `showModal()` makes the page behind inert, holds focus in the dialog, and gives Esc for free.
- **The header's "How it works" button opens it on both routes.**
  The header is rebuilt on every navigation, so the script finds the button through event delegation and looks it up again when restoring focus on close.
  On a phone the words are visually hidden and stay the button's accessible name.
- **Two chapters, one story.**
  "Sample to map" (12 slides) runs from picking a sample through the table, filters, the inspector, GEO, the plots, cohorts, upload, export, the map and the neighborhood explorer.
  "Explore the map" (8 slides) covers the overview, projections, 3-D, color, point count, find a study, selecting a spaceflight sample, and reset.
  It opens at the chapter for the current page, and chapter tabs follow the tabs pattern (arrow keys move between them).
- **Slides are data**: a title, one or two sentences, alt text and a key in `guide.SLIDES`.
- **The script owns everything that changes, and Dash sets none of it**: `data-current` on the shown chapter and slide (CSS hides the rest with `display: none`, which also takes them out of the accessibility tree), `aria-selected` on the tabs, the counter, the dots, Back and Next.
  A polite status line announces "Step n of m: title" on every change.
- **Screenshots load when their slide is shown.**
  The server renders `data-src`; the script sets `src` for the current slide and the next.
  Twenty images are not fetched on every page view by readers who never open the guide; Dash's `html.Img` has no `loading` attribute, which is why this is done by hand.
- **The screenshot takes the space the words leave.**
  The dialog has a definite height, and the image shrinks to fit above the text rather than pushing the text below the fold on a laptop.
- **First visit.**
  It opens once, for the page you landed on, only after that page has drawn: the boot skeleton gone, and on the map the first figure in place.
  It polls for that every 250 ms for at most 30 s, gives up if the reader starts clicking or typing first, and records the visit only after it has really opened.
  `localStorage` is wrapped in try/catch; a blocked store means it never opens by itself, which is the safe failure.
  It never opens by itself in an automated browser (`navigator.webdriver`): a modal over a test suite makes the page it drives inert, and every existing browser suite starts from a fresh profile.
  Codex's point that "Dash has mounted" is not "the map is ready" is why the map waits on its own skeleton.
- **Motion**: the dialog fades and rises 8 px on open and slides cross-fade; the popover fades 4 px from its side.
  All of it is off under reduced motion.

### The screenshots

- **Made only by `tests/guide_screenshots.py`**, which starts its own server and walks both flows on the real corpus with the README's query (OSD-100, a flown left eye, 30 matches).
- **Each frame spotlights its feature**: one fixed-position element on `<body>`, injected only during capture, dims everything else and rings the feature in NASA blue.
  Being on `<body>`, the ring is never clipped by a scroll container.
- **A close region at the interface's own size**: most frames are a 960 x 600 CSS-pixel region around the feature, captured at 1.5x and encoded by Chrome as WebP at exactly 1440 x 900.
  Slides about the whole picture use the whole viewport.
  The text in a frame is therefore about the size it is in the app.
- **Freshness is recorded, not assumed**: `docs/guide-screenshots.json` holds the commit, time, viewport and query of the last capture.
- A spaceflight diamond is clicked through Plotly's own axis transforms, since WebGL points are not DOM nodes.

### Shipping

The images are binary assets in a subfolder, which the deploy allowlist accepts only by exact name.
Each is listed in `deploy/common.STATIC_ASSETS` and `deploy/runtime-files.json`, and `tests/test_guide.py` requires the two lists and `guide.SLIDES` to agree.
`deploy/ship.py`'s pre-transfer audit used to treat only `.woff2` as binary, so replacing a screenshot already on the server would have failed while decoding it as text; it now records WebP changes by hash, and a test covers it.
The new Python, CSS and JS files are listed too, and the existing payload test now also requires every guide image to be shipped.

### What Codex's review of the build changed

- Scroll and resize dismissal now return focus to the button when it was inside the popover; before, a keyboard reader lost their place.
- The guide body scrolls when even a fully shrunk screenshot leaves the words no room, so a very short window cannot cut a slide's text off.
- The excerpt validator gained negative cases (a silent skip, reordered blocks, an invented or edited line), so it cannot be weakened unnoticed.
- Its point that readiness and cancellation lacked behavioral tests was already answered by `tests/e2e_help_check.py`, written after the review was requested.
  That suite then caught a real race of its own: on a cold `/map`, Dash can drop its loading node a beat before the view mounts, and the first readiness check read "no skeleton" as "no map" and opened the tour over a blank canvas.
  Readiness now needs the first figure, or the explicit `#map-unavailable` view.

### Verification

- `tests/test_help.py` (excerpts, placement, labels) and `tests/test_guide.py` (every slide's image exists at the reserved size, no orphans, the allowlist matches, the dialog sits outside the replaced slots).
- `tests/e2e_help_check.py`, 45 browser checks with its own server: popover focus, Esc, Tab, outside click, scroll, one at a time, route round trips with a popover open, the callback-rendered inspector entry; the tour's first-visit opening on both routes, waiting for the map's first figure, cancelling on early interaction, opening only once, blocked storage, keyboard, focus return, reduced motion and the phone header.
- Checked by eye at 1440 x 900, 1280 x 720 and 375 x 812.

## Tradeoffs

- **Screenshots go stale as the interface changes.**
  They are regenerated by one script rather than hand-made, and a test checks that every slide's image exists.
- **Many "i" icons add visual noise.**
  They are small and quiet until hovered, and each is a single call that is easy to remove.
- **Real code excerpts expose implementation detail to scientists.**
  That is what Joshua asked for.
  Every excerpt sits under a plain explanation that stands on its own.

## Status

- [x] Registry, popover component and script, with tests
- [x] "i" icons placed across Retrieve and Map
- [x] Guide content, modal, script and header button
- [x] Screenshot capture script and images
- [x] Deploy allowlist, browser checks, docs
- [x] Codex review and fixes
- [ ] Ship to the NASA server (waiting on Joshua)
