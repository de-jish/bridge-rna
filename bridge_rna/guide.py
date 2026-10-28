"""The guided tour: what a reader does with Bridge RNA, one screenshot at a time.

Slides are data. `build_guide()` renders them into one native `<dialog>` on the
shell, and `assets/guide.js` owns everything that moves: which chapter and
slide is current, Back and Next, the keys, and opening it once on a first
visit. The screenshots are real captures of the running app made by
`tests/guide_screenshots.py`, which reads `SLIDES` for the file names, so a
slide added here without an image fails `tests/test_guide.py` until the script
is rerun.

Design: docs/help-and-guide.md.
"""

from __future__ import annotations

from dataclasses import dataclass

from dash import html

# Every capture is this size, so the dialog can reserve the image's box before
# it loads and nothing jumps when a slide changes.
SHOT_WIDTH, SHOT_HEIGHT = 1440, 900


@dataclass(frozen=True)
class Slide:
    key: str
    title: str
    body: str
    alt: str

    @property
    def image(self) -> str:
        return f"assets/guide/{self.key}.webp"


@dataclass(frozen=True)
class Chapter:
    key: str
    label: str
    route: str
    slides: tuple[Slide, ...]


CHAPTERS: tuple[Chapter, ...] = (
    Chapter("retrieve", "Sample to map", "/", (
        Slide("retrieve-01-sample",
              "Pick a spaceflight sample",
              "Choose an OSDR study, then one of its samples. All 2,108 "
              "searchable samples have their embeddings computed in advance, "
              "so a search takes about a second.",
              "The search panel with an OSDR study and sample chosen."),
        Slide("retrieve-02-search",
              "Choose how many matches, then search",
              "Ask for anywhere from 3 to 1,000 of the closest GEO samples. "
              "Search compares your sample against all 940,455 of them.",
              "The number-of-matches control and the Search button."),
        Slide("retrieve-03-table",
              "Read the ranked matches",
              "Each row is a GEO sample, ranked by how similar its embedding is "
              "to yours. Rank 1 is the closest of all 940,455.",
              "The matches table with rank, score, GSM, series and tissue "
              "columns."),
        Slide("retrieve-04-filter",
              "Filter without renumbering",
              "Narrow the list by series, tissue, species, score or text. Ranks "
              "stay positions among all 940,455 samples, so a filter never "
              "renumbers them.",
              "The filter row above the matches table."),
        Slide("retrieve-05-inspect",
              "Inspect a match's metadata",
              "Click a row to open its record: series, title, source, tissue, "
              "species and the GEO study's own description.",
              "The inspector showing the metadata of a selected match."),
        Slide("retrieve-06-geo",
              "Go to the GEO record",
              "Every GSM and GSE accession links to its page on NCBI GEO, so the "
              "original study is one click away.",
              "A GSM accession link in the inspector."),
        Slide("retrieve-07-plots",
              "See the results as a plot",
              "Score by rank shows how quickly similarity falls off. Series "
              "counts matches per study, and Network draws the query, its "
              "matches and their studies.",
              "The plot panel showing matches counted per GEO series."),
        Slide("retrieve-08-cohort",
              "Search a whole group at once",
              "Cohort pools one study's animals from one organ and spaceflight "
              "arm into a single query, which gives far more repeatable matches "
              "than one sample. Compare against runs the other arm as a second "
              "search.",
              "The cohort panel with a cohort and a comparison chosen."),
        Slide("retrieve-09-upload",
              "Or bring your own counts",
              "Upload mouse RNA-seq counts and pick a column. It is embedded "
              "with the same model and searched against the same index.",
              "The upload panel with its file drop area."),
        Slide("retrieve-10-export",
              "Export what you see",
              "Download CSV writes exactly the rows on screen, with the query "
              "and filters recorded at the top of the file.",
              "The Download CSV button beside the matches."),
        Slide("retrieve-11-map",
              "View the matches on the map",
              "Open Map: your sample is drawn as a star and its matches as teal "
              "circles. Fit results zooms to them.",
              "The map with a retrieval drawn and the Your retrieval controls."),
        Slide("retrieve-12-neighborhood",
              "Explore the neighborhood",
              "The neighborhood explorer lists the same matches as the table. "
              "Pick one to read its details and see where it sits.",
              "The neighborhood explorer open beside the framed matches."),
    )),
    Chapter("map", "Explore the map", "/map", (
        Slide("map-01-overview",
              "Every sample on one map",
              "All 942,563 samples, placed by their embeddings: GEO samples as "
              "dots and NASA spaceflight samples as diamonds. Nearby points "
              "tend to have similar expression.",
              "The whole map colored by tissue."),
        Slide("map-02-projection",
              "Three ways to flatten it",
              "UMAP, t-SNE and PCA each squeeze 512 dimensions into two. UMAP "
              "and t-SNE keep each sample's near neighbors together; PCA keeps "
              "the largest overall directions of variation.",
              "The projection control on the map's control rail."),
        Slide("map-03-3d",
              "Flat or 3-D",
              "Switch to 3-D and drag to rotate. It draws fewer GEO points so "
              "rotation stays smooth.",
              "The map drawn in three dimensions."),
        Slide("map-04-color",
              "Color by tissue or species",
              "Tissue sorts every sample into one of 37 categories with the same "
              "rules on both sides; Species separates human and mouse.",
              "The map colored by species with the Color by control."),
        Slide("map-05-points",
              "Draw more or fewer points",
              "Fewer points draw faster. Zooming in redraws the visible area at "
              "full detail.",
              "The number-of-points control on the control rail."),
        Slide("map-06-find",
              "Find a study",
              "Type a GEO series (GSE...) or an OSDR study (OSD-...) to mark its "
              "samples with an X, then Fit study to zoom to them.",
              "A study found on the map, its samples marked."),
        Slide("map-07-select",
              "Click a spaceflight sample",
              "Click a diamond to read its record, then search from it with "
              "Find related ARCHS4 samples.",
              "The selected-sample panel after clicking an OSDR diamond."),
        Slide("map-08-reset",
              "Back to the whole map",
              "Reset view undoes any zoom, from Fit study, Fit results or your "
              "own scrolling.",
              "The Reset view button on the control rail."),
    )),
)

SLIDES: tuple[Slide, ...] = tuple(s for c in CHAPTERS for s in c.slides)


def _slide(chapter: Chapter, index: int, slide: Slide) -> html.Article:
    total = len(chapter.slides)
    return html.Article(
        className="guide-slide",
        **{"data-slide": str(index), "aria-roledescription": "slide",
           "aria-label": f"{index + 1} of {total}: {slide.title}"},
        children=[
            # No `src` until the slide is shown: `assets/guide.js` sets it
            # from `data-src` for the current slide and the next, so twenty
            # screenshots are not fetched on every page view by readers who
            # never open the guide. (Dash's Img has no `loading` attribute.)
            html.Figure(className="guide-shot", children=html.Img(
                alt=slide.alt, width=SHOT_WIDTH, height=SHOT_HEIGHT,
                **{"data-src": f"/{slide.image}"})),
            html.Div(className="guide-copy", children=[
                html.P(f"Step {index + 1} of {total}", className="guide-step"),
                html.H3(slide.title, className="guide-slide-title"),
                html.P(slide.body, className="guide-slide-body"),
            ]),
        ],
    )


def build_guide() -> html.Dialog:
    """The tour, closed. Lives on the shell beside the header and view slots,
    never inside them, so a route change cannot destroy it while it is open."""
    return html.Dialog(
        id="guide", className="guide",
        **{"aria-labelledby": "guide-title"},
        children=[
            html.Div(className="guide-head", children=[
                html.H2("How Bridge RNA works", id="guide-title",
                        className="guide-title"),
                html.Div(
                    className="guide-tabs", role="tablist",
                    **{"aria-label": "Guide chapters"},
                    children=[
                        html.Button(
                            c.label, type="button", className="guide-tab",
                            id=f"guide-tab-{c.key}", role="tab",
                            **{"aria-controls": f"guide-chapter-{c.key}",
                               "data-guide-chapter": c.key},
                        )
                        for c in CHAPTERS
                    ],
                ),
                html.Button(type="button", className="guide-close",
                            **{"aria-label": "Close guide"}),
            ]),
            html.Div(className="guide-body", children=[
                html.Section(
                    id=f"guide-chapter-{c.key}", className="guide-chapter",
                    role="tabpanel", tabIndex="-1",
                    **{"aria-labelledby": f"guide-tab-{c.key}",
                       "data-chapter": c.key, "data-route": c.route},
                    children=[_slide(c, i, s) for i, s in enumerate(c.slides)],
                )
                for c in CHAPTERS
            ]),
            html.Div(className="guide-foot", children=[
                html.Div(className="guide-dots", **{"aria-hidden": "true"}),
                html.P(className="guide-counter", role="status"),
                html.Div(className="guide-nav", children=[
                    html.Button("Back", type="button",
                                className="btn-ghost guide-back"),
                    html.Button("Next", type="button",
                                className="btn-secondary guide-next"),
                ]),
            ]),
        ],
    )


def guide_button() -> html.Button:
    """The header's "How it works". Opened by `assets/guide.js` through event
    delegation, because the header is rebuilt on every route change."""
    return html.Button(
        type="button", className="app-guide-btn",
        **{"data-guide-open": "", "aria-haspopup": "dialog"},
        children=[
            html.Span("?", className="app-guide-icon", **{"aria-hidden": "true"}),
            html.Span("How it works", className="app-guide-text"),
        ],
    )
