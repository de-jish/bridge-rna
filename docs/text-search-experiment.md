# Natural-language search over the frozen embeddings: experiment record

Started 2026-11-05 on branch `cc/experiment-text-search-2b87f3`.
This is an experiment, not a feature: no app code (`app.py`, `bridge_rna/`, `manifold/`, `assets/`) is edited.
Scripts live under `experiments/text_search/`, large intermediates under the gitignored `exp_data/`.
The literature survey is written by another session; this record covers the placement walkthrough and the experiment itself.

## Placement

Walked on 2026-11-05 against the deployed code (`feature/retrieval-table` at 2ba9fd9, the main checkout), served on :8061 with the real cache.
Screenshots are in [`docs/text-search/`](text-search/).

### What the walkthrough showed

- **Retrieve has three query-vector sources in one tab strip: Sample, Cohort, Upload** ([01](text-search/01-retrieve-sample-results.jpg), [03](text-search/03-upload-tab.jpg), [04](text-search/04-cohort-tab.jpg)).
  Each produces one 512-d query vector, then the same whitened scan, results table, score-by-rank plot, network and map overlay.
  The strip already fills the 140 px rail at three tabs.
- **The results table already has a "Text contains" filter** over GSM, series, title and source ([01](text-search/01-retrieve-sample-results.jpg)).
  It is a substring filter over *retrieved* hits' GEO metadata, so the app already answers "which of my matches mention X" by string match.
- **The query panel on the right describes an OSDR query from curated metadata** (study, tissue, condition, OSDR study description).
  For an **uploaded** sample it shows only Identity, because `bridge_rna/panels._build_osdr_query_metadata_block` deliberately renders nothing for the synthetic "Uploaded file" study.
  An uploaded sample is therefore the one query the app cannot describe at all.
- **The AI summary** ([02](text-search/02-inspector-and-ai-summary.jpg)) is built from the query's metadata and the hits' GEO context, sent to a local Ollama model (`gemma3:4b` by default; on this Mac Ollama is running with no model pulled) or a configured Bedrock endpoint.
  For an upload it has no query description to work from.
- **The map's find box refuses free text by design** ([05](text-search/05-map-find-refuses-free-text.jpg)): "mouse liver" answers "Not a study identifier ... To find a tissue or a condition instead, use Color by above."

### The candidates

| placement | what it would do | what it displaces or costs | fit |
| --- | --- | --- | --- |
| **A. Generated description in the query inspector** (samples -> text) | Under the query sample, a short predicted description from its expression alone: tissue bucket with confidence and runner-up, species, optionally the controlled terms its nearest training samples share | Nothing on screen; fills the empty panel an upload gets today. For an OSDR query it sits beside the curated tissue, which turns a disagreement into a visible check | **Best.** No new control, contextual disclosure, and it serves exactly the query that metadata search cannot |
| **B. A fourth Retrieve tab, "Describe"** (text -> samples, the sixth query-vector source) | Type a description; it becomes one 512-d query vector, then the cached path's scan, annotation and map join unchanged | A fourth tab in a strip that is full at three; and it changes the view's premise, which is "start from a spaceflight sample". A text query has no spaceflight sample, so it is a different question | Architecturally clean, product-wise a new question |
| **C. The map's find box** (text -> samples, marked on the map) | Type a description; the top-N most similar samples are marked, like a found study | `manifold/find.py` refuses free text on purpose (see below), and an exact identifier match and a ranked fuzzy set would share one box and one mark | Weakest fit as specified today |
| **D. Feeding the AI summary** (samples -> text as prompt context) | The predicted description goes into the summary prompt for an upload | Nothing; it is plumbing behind A | Follows from A |

### The find-box conflict, stated for Josh to decide

CLAUDE.md records why free text is refused: a substring scan of 940,455 titles "would return hundreds of rows for 'liver' and read as a biological query when it is a string match", and "Where is liver" is the Tissue color-by's question.
A text-to-expression similarity query is a different object from that substring scan: it ranks samples by their *expression*, so it is a biological query in the sense the objection says a string match is not.
That answers the first half of the objection.
It does not answer the second half for bucket-level queries: "liver" is still the Tissue color-by's question, and the color-by answers it exactly over all 942,563 points.
What a similarity query adds is the space between the buckets ("skeletal muscle after unloading", "activated T cells"), which no color-by can express.
It also introduces something the box has never had: a ranked, fuzzy result set with no natural cut, where a find today is exact and complete.
Whether that argument is good enough to reopen the box is Josh's product decision; this record does not decide it.

### How placement shapes the evaluation

Because placement A is the strongest fit, **samples -> text on samples whose own metadata is hidden** is the criterion that matters most, and within it the OSDR spaceflight samples (the out-of-distribution, mouse, curated-tissue test) matter more than held-out ARCHS4.
So the samples -> text criterion weights OSDR at 0.6 and held-out ARCHS4 at 0.4 (the brief required at least half).
Text -> samples stays the largest single criterion (35%), because placements B and C both need it and it is the harder, more novel half.
