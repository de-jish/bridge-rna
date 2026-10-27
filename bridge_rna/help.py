"""The "i" explanations: what each feature does, and the code behind it.

Every entry is a short plain-language explanation plus, where it helps, a few
lines of the code that does the work, naming the file they come from. Joshua
asked for many of them so he can remove the ones he finds unnecessary: an
explanation is one `info(key)` call beside a label, and removing it is deleting
that call (the entry can stay or go).

**The code is quoted, never paraphrased, and a test keeps it honest.**
`tests/test_help.py` checks that every excerpt line other than `...` appears in
the file it cites, so an explanation can never show code the app no longer
runs. Excerpts are abridged with `...` and may drop indentation; they may not
rewrite a line.

The explanations are rendered once, hidden, by `library()` on the shell, and
`assets/info-popover.js` shows one in a body-level popover entirely in the
browser: opening an explanation never posts to the server.
Design: docs/help-and-guide.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from dash import html


@dataclass(frozen=True)
class Excerpt:
    """A few lines quoted from one file. `...` marks lines left out."""

    path: str
    lines: tuple[str, ...]


@dataclass(frozen=True)
class Entry:
    title: str
    body: tuple[str, ...]
    code: tuple[Excerpt, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        # A one-paragraph body written without its trailing comma is a str,
        # which would render one paragraph per character.
        if isinstance(self.body, str):
            object.__setattr__(self, "body", (self.body,))


def _x(path: str, *lines: str) -> Excerpt:
    return Excerpt(path, tuple(lines))


ENTRIES: dict[str, Entry] = {
    # --- Retrieve: the query --------------------------------------------------
    "query-source": Entry(
        "Query source",
        ("Three ways to ask the same question, \"which Earth samples look most "
         "like this?\": one OSDR sample, a whole experimental group pooled into "
         "one query, or a counts file you upload.",
         "All three end in the same cosine scan over the 940,455 ARCHS4 samples, "
         "so their results share one format."),
    ),
    "osdr-study": Entry(
        "OSDR study",
        ("The NASA Open Science Data Repository study (OSD-###) to pick a "
         "sample or cohort from. Sample and Cohort share it, so a study chosen "
         "once serves both."),
    ),
    "osdr-sample": Entry(
        "OSDR sample",
        ("The spaceflight sample to search with. Each of the 2,108 searchable "
         "samples already has its 512-dimensional embedding computed, so a "
         "search reads it from disk instead of running the model.",
         "Samples that cannot be searched are listed, disabled, with the "
         "reason."),
        (_x("bridge_rna/retrieval.py",
            "def cached_query_vector(sample_id: str) -> np.ndarray | None:"),),
    ),
    "number-of-matches": Entry(
        "Number of matches",
        ("How many of the 940,455 GEO samples to return, best first, from 3 "
         "to 1,000. Each match's score is the cosine similarity between its "
         "embedding and your sample's.",
         "Ranks are positions among all 940,455 samples and never change when "
         "you filter. An exact tie goes to the lower reference row, so the "
         "same search always returns the same order."),
        (_x("bridge_rna/retrieval.py",
            "x /= (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)",
            "block = (x @ Qt).T[:m]                          # (m, rows in block)",
            "...",
            "order = np.lexsort((best_idx, -best_score), axis=-1)"),),
    ),
    "metadata-options": Entry(
        "Metadata options",
        ("Optional enrichment from NCBI. Every match already carries its GEO "
         "series, title, source and tissue from the cached ARCHS4 metadata; "
         "these settings only add live lookups such as study summaries and "
         "publications."),
    ),
    # --- Retrieve: cohorts ----------------------------------------------------
    "group-by": Entry(
        "Group by",
        ("Which OSDR samples count as one experimental group. A cohort is "
         "always one study and one organ; Spaceflight arm (flight, ground, "
         "basal, vivarium) is the one choice.",
         "Different organs never pool: the average of a liver and a brain "
         "profile resembles no real sample. Left and right retina, heart and "
         "right ventricle, and gastrocnemius and quadriceps were measured to "
         "be interchangeable and pool together."),
        (_x("bridge_rna/cohorts.py",
            "SAME_ORGAN: dict[str, str] = {",
            "\"left retina\": \"Retina\",",
            "\"right retina\": \"Retina\","),),
    ),
    "cohort": Entry(
        "Cohort",
        ("A group of animals searched as one query. Each member's embedding "
         "is scaled to length 1 and the group's average direction is the "
         "query, so every animal gets one equal vote.",
         "Pooling is what makes results repeatable: two replicates share about "
         "16% of their top-5 matches, a pooled cohort about 75%."),
        (_x("bridge_rna/cohorts.py",
            "u = _unit_rows(arr)",
            "m = u.mean(axis=0)",
            "norm = float(np.linalg.norm(m))"),),
    ),
    "cohort-members": Entry(
        "Members",
        ("The samples pooled into this cohort. Untick one to leave it out of "
         "the next search. Each shows how closely it agrees with the rest "
         "(its leave-one-out cosine), so an animal unlike its group stands "
         "out."),
    ),
    "compare-against": Entry(
        "Compare against",
        ("Runs a second cohort that differs in exactly one thing, the "
         "spaceflight arm, as its own pooled search, then reports how many "
         "matches the two share.",
         "It is two independent searches, never a subtraction: a flight-minus-"
         "ground vector is not a transcriptome and cannot be ranked against "
         "real samples."),
        (_x("bridge_rna/callbacks.py",
            "shared = set(hits_df[\"gsm\"]) & set(other_hits[\"gsm\"])",
            "union = set(hits_df[\"gsm\"]) | set(other_hits[\"gsm\"])",
            "overlap = len(shared) / len(union) if union else 0.0"),),
    ),
    # --- Retrieve: upload -----------------------------------------------------
    "upload-counts": Entry(
        "Upload counts",
        ("Search with a sample the corpus has never seen. Upload mouse RNA-seq "
         "counts (Ensembl gene IDs in the first column, one sample per "
         "column); the model embeds the column you choose, then it is scored "
         "against the same 940,455 samples.",
         "The genes must line up with the model's exact gene order, checked "
         "by a fingerprint, or the file is refused rather than embedded "
         "wrongly."),
        (_x("precompute/embed_upload.py",
            "digest = sym[\"canonical_gene_order_digest\"](canonical_genes)",
            "expected = sym[\"CANONICAL_GENES_SHA256\"]",
            "if digest != expected:"),),
    ),
    # --- Retrieve: results ----------------------------------------------------
    "matches": Entry(
        "Matches",
        ("The GEO samples most similar to your query, ranked by cosine "
         "similarity of their embeddings. Similarity here means the model "
         "places the two expression profiles close together, not that they "
         "are the same biology.",
         "Some GEO samples are the same data uploaded twice under two "
         "accessions; each copy keeps its own row, marked \"Same data as\"."),
        (_x("bridge_rna/similarity.py",
            "IDENTICAL_PROFILE_DISTANCE = 1e-6"),),
    ),
    "filter-series": Entry(
        "Series",
        ("Show only the matches from the GEO series (GSE) you pick. The count "
         "beside each series is how many of your matches it holds."),
    ),
    "filter-tissue": Entry(
        "Tissue category",
        ("Each GEO sample's tissue, read from its free-text GEO description "
         "and sorted into one of 37 categories by ordered keyword rules. The "
         "same rules sort the OSDR samples, so a liver is \"Liver\" on both "
         "sides.",
         "\"Other\" means a tissue was recorded but matched no rule; "
         "\"Unknown\" means none was recorded."),
        (_x("manifold/tissue.py",
            "def canonical_tissue(value: object) -> str:"),),
    ),
    "filter-species": Entry(
        "Species",
        ("Human or mouse. ARCHS4 holds both; OSDR samples are mouse, and "
         "mouse genes are mapped to the model's human gene space before "
         "embedding."),
    ),
    "filter-min-score": Entry(
        "Minimum score",
        ("Hide matches whose cosine similarity is below this value. Scores at "
         "the top of a 940,455-sample index are close together, so small "
         "differences in this number matter."),
    ),
    "filter-text": Entry(
        "Text contains",
        ("Keep matches whose GSM, series, title or source name contains this "
         "text, ignoring case. Tissue is filtered by its own control."),
    ),
    "filter-retrieved-by": Entry(
        "Retrieved by",
        ("In a comparison, which cohort found each match: either, both, or "
         "only one of them."),
    ),
    "download-csv": Entry(
        "Download CSV",
        ("Exports exactly the rows on screen, in the table's order, with the "
         "query, filters and sort recorded as # comment lines at the top. "
         "Text from GEO is written as deposited."),
        (_x("bridge_rna/results.py",
            "def apply_filters(rows: Sequence[Mapping[str, Any]],"),),
    ),
    "plot-switch": Entry(
        "Plots",
        ("One plot of the rows the table shows. Score by rank shows how fast "
         "similarity falls from rank 1 (on a linear axis to ten times your "
         "number of matches); Series counts matches per GEO series; Network "
         "draws the query, its matches and their studies (up to 30)."),
        (_x("bridge_rna/result_plots.py",
            "depth = max(SCORE_DEPTH_MIN, SCORE_DEPTH_FACTOR * k)"),),
    ),
    # --- Retrieve: the inspector ---------------------------------------------
    "query-details": Entry(
        "Query sample",
        ("The OSDR sample you searched with: its study, tissue, flight or ground "
         "arm and the other conditions OSDR recorded for it. The study links to "
         "its page on the NASA Open Science Data Repository.",),
    ),
    "match-details": Entry(
        "Match details",
        ("Everything recorded about the selected GEO sample: its series, title, "
         "source, tissue and species from the ARCHS4 metadata, plus the study "
         "abstract and publication fetched from NCBI when you open it.",
         "The GSM and GSE accessions open the sample's and the study's own pages "
         "on NCBI GEO in a new tab."),
        (_x("bridge_rna/panels.py",
            "elif re.fullmatch(r\"G(?:SM|SE)\\d+\", accession):",
            "url = f\"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={accession}\""),),
    ),
    "study-details": Entry(
        "GEO study",
        ("One GEO series and the matches it contributed. A series that "
         "contributes many matches is a study whose samples, as a group, "
         "resemble your query."),
    ),
    "ai-summary": Entry(
        "AI summary",
        ("A language model reads your query's metadata and the matches in view, "
         "with their GEO study text, and writes a short interpretation. It sees "
         "the same rows the table shows, so filtering changes what it reads.",
         "Check its claims against the records it cites: similar expression does "
         "not establish a shared mechanism."),
    ),
    # --- Map ------------------------------------------------------------------
    "projection": Entry(
        "Projection",
        ("Every sample's 512-dimensional embedding squeezed onto a flat (or "
         "3-D) map, by one of three methods fit on all 942,563 points: UMAP, "
         "t-SNE or PCA. Each vector is scaled to length 1 first.",
         "Nearby points tend to be similar, but distances on the map are not "
         "the cosine scores used to rank matches."),
        (_x("precompute/build_projections.py",
            "reducer = umap.UMAP(",
            "n_components=n_components,",
            "n_neighbors=n_neighbors,",
            "min_dist=0.1,",
            "metric=\"cosine\","),),
    ),
    "dimensions": Entry(
        "Dimensions",
        ("Flat 2-D or rotatable 3-D. 3-D draws fewer background points so it "
         "stays responsive while you rotate it."),
    ),
    "color-by": Entry(
        "Color by",
        ("Color every point by tissue category or species. The bar underneath "
         "shows how much of the map the field can actually color; a field "
         "that does not describe a corpus draws it as faint context, never as "
         "a category."),
        (_x("manifold/colorby.py",
            "def covers(self) -> tuple[str, ...]:",
            "\"\"\"Corpora this field can color on this machine, right now.\"\"\""),),
    ),
    "layers": Entry(
        "Layers",
        ("Turn the 940,455 GEO samples (dots) and the 2,108 OSDR spaceflight "
         "samples (diamonds) on or off."),
    ),
    "point-budget": Entry(
        "Number of ARCHS4 points",
        ("How many GEO points to draw. Fewer points draw faster; the sample "
         "keeps the human and mouse proportions, and zooming in redraws the "
         "visible area at full detail."),
        (_x("manifold/render.py",
            "mask = sampling.viewport_mask(coords_xy[:n_archs4], viewport)",
            "return sampling.stratified_archs4_sample(species, budget, seed=7, mask=mask)"),),
    ),
    "find-study": Entry(
        "Find a study",
        ("Type a GEO series (GSE…) or an OSDR study (OSD-…) to mark its "
         "samples with an X. Fit study zooms to them. Free text is not "
         "searched; use Color by for tissues."),
        (_x("manifold/find.py",
            "_GSE = re.compile(r\"^GSE(\\d+)$\")",
            "...",
            "_OSD = re.compile(r\"^OSD[\\s-]*(\\d+)$\")"),),
    ),
    "your-retrieval": Entry(
        "Your retrieval",
        ("The last search you ran, drawn on the map: your sample (star) and "
         "its matches (small teal circles; squares for a second cohort). Fit "
         "results zooms to them and opens the neighborhood explorer, which "
         "lists exactly those matches."),
    ),
    "neighborhood": Entry(
        "Neighborhood explorer",
        ("A list of exactly the matches drawn on the map, the same rows as the "
         "results table. Pick one to see its details and where it sits; in a "
         "comparison, A and B switch between the two cohorts' matches."),
        (_x("manifold/callbacks.py",
            "return neighborhoods.build_payload(pd.DataFrame(list(hits or [])),",
            "label=str(stored.get(\"label\") or \"\"))"),),
    ),
    "reset-view": Entry(
        "Reset view",
        ("Undo any zoom, whether from Fit study, Fit results or your own "
         "scrolling, and show the whole map again."),
        (_x("manifold/callbacks.py",
            "return {\"xaxis\": dict(autorange=True), \"yaxis\": dict(autorange=True)}"),),
    ),
}


def info(key: str) -> html.Button:
    """The "i" button for `key`, to sit beside (never inside) a label.

    It carries only the key. The explanation itself is rendered once, in
    `library()` on the shell, and `assets/info-popover.js` copies it into one
    body-level popover when the button is pressed - so a key can appear any
    number of times without duplicate ids, and no panel is clipped by the
    scroll container its button happens to sit in.
    """
    entry = ENTRIES[key]
    return html.Button(
        "i", type="button", className="info-btn",
        **{"aria-label": f"About {entry.title}", "aria-haspopup": "dialog",
           "data-info": key},
    )


def _entry_children(key: str, entry: Entry) -> list:
    blocks: list = [html.H3(entry.title, className="info-title")]
    blocks += [html.P(p, className="info-text") for p in entry.body]
    for excerpt in entry.code:
        blocks.append(html.Figure(className="info-code", children=[
            html.Figcaption(excerpt.path, className="info-code-path"),
            html.Pre(html.Code("\n".join(excerpt.lines))),
        ]))
    return blocks


def library() -> html.Div:
    """Every explanation, hidden, once per page. Lives on the shell so it
    survives route changes; the popover script reads from it and never writes
    to it."""
    return html.Div(
        id="info-library", hidden=True,
        children=[html.Div(_entry_children(k, e), **{"data-info-entry": k})
                  for k, e in ENTRIES.items()],
    )
