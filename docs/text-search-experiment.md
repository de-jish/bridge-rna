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

## Setup and fixed procedure

The experiment clock started at **12:04 EST on 2026-11-05** (Step 0 not counted).
The budget is 4 hours of wall clock, so the deadline is 16:04 EST, with the last 30 minutes reserved for final re-evaluation, the write-up and pushing.
On 2026-11-05 Josh asked to finish the experiment in a cloud session.
The clock keeps running from 12:04 and is not restarted by the move.

### Data the experiment reads

- The ARCHS4 index, `archs4_sample_embeddings_full/sample_embeddings.float16.mmap` (940,455 x 512 float16, row = global index), and `sample_locations.parquet` (species: 0 human 510,709, 1 mouse 429,746). Both are Git LFS objects, which GitHub serves for this public repository.
- Snapshots of four gitignored cache files in `experiments/text_search/data/`, committed to this branch with Josh's approval on 2026-11-05 so a cloud session can run without the Mac: `archs4_metadata.parquet`, `osdr_sample_embeddings.float32.npy`, `osdr_metadata.parquet` (rows align with the embeddings), `whitening.npz`. Checksums in `data/SHA256SUMS`.
- CellWhisperer's curated captions, read once with HTTP range requests from `full_data.h5ad` (163 GB) by `fetch_cellwhisperer_captions.py`: only `obs/geo_id`, `obs/organism` and `obs/natural_language_annotation`, 197 MB transferred in 67 s. 705,430 samples, all human (32 "human semen metagenome"), 428,353 distinct captions, median 111 characters; **381,018 join our ARCHS4 index by GSM**.
  The site states no data license. The code repository is GPL-3.0. Treat the captions as research-use only until the license is confirmed with the authors (cellwhisperer at bocklab.org); they are not committed.

### Split and leakage control (`prep.py`, fixed)

- Split by GEO `series_id`: a series is held out when `md5(f"{seed}:{series_id}")` falls in the lowest 15%. The 839 rows with no series are train-only. No `series_id` holds more than one series.
- Gallery: 40,000 held-out samples drawn at random (seed) from held-out rows whose tissue bucket is not Other or Unknown.
- A gallery sample whose plain cosine distance to **any** training sample is below 1e-6 is dropped (exact float64 recheck of every candidate within 1e-4).
- Caption pool: up to 5,000 gallery samples, one per series, with text unique in the gallery.
- OSDR: all 2,108 samples, curated tissue through `manifold/tissue.canonical_tissue`. Their max cosine to any ARCHS4 training sample is recorded, because GeneLab studies are also deposited in GEO.
- Sample text is `title. source_name. characteristics` from the metadata parquet. At test time a held-out sample is represented only by its L2-normalised expression vector.

### Free-text queries

`queries.py` holds 32 free-text queries with their relevance definition (tissue bucket set, optional species), written at 12:05 before any result existed. Do not edit them.
Bucket queries use each of the 37 named buckets as text (`bucket_query_text`).

### Scoring Z (fixed in the baseline round, never changed after)

Raw metrics, all reported:

- **T2S** (text -> sample), gallery = held-out pool expression vectors. AP@100 per query = sum over the top 100 of precision@k at each relevant rank, divided by min(R, 100). Macro mAP over bucket queries with at least 20 relevant gallery samples, and separately over the free-text queries. Null: the same rankings scored against 3 label permutations of the gallery.
  Per species: bucket queries re-run inside the human-only and mouse-only gallery.
  Species collapse: for each species-agnostic bucket query, |human share of the top 100 - human share of its relevant set|.
- **S2T** (sample -> text), classification over the 37 named buckets as candidate labels. Balanced accuracy (mean per-class recall over classes with at least 20 samples) on the held-out ARCHS4 gallery, per species too, and on all 2,108 OSDR samples (every class present). Chance = 1/37.
- **CAP** (caption retrieval), within the caption pool: each sample's own GEO text as the query; recall@10 and median rank in both directions (text -> sample, sample -> text). Random recall@10 = 10/N.
- **Serving**: torch needed at query time; text-encoder kind; measured CPU latency of one text query against all 940,455 rows (encode plus every full scan it needs); added artifact size.

Criterion scores (each 0-100) and the overall:

- T2S = 100 x (0.5 mAP_bucket + 0.5 mAP_free)
- S2T = 100 x (0.4 BA_archs4 + 0.6 BA_osdr)
- CAP = 100 x (0.5 mean(R@10 both directions) + 0.5 mean over directions of max(0, 1 - median_rank / (N/2)))
- SERV = 0 if torch is needed at serving; else 100 x (0.5 latency + 0.3 dependency + 0.2 size), latency 1 if under 1 s, 0.5 if under 2 s, else 0; dependency 1 numpy-only (model2vec or precomputed vocabulary), 0.7 onnxruntime, 0.5 the configured LLM endpoint; size 1 if at most 100 MB added, 0.5 if at most 500 MB, else 0
- HON = 0 if the leakage check fails; else 100 x mean of (margin = min, clipped to [0, 1], of (mAP_bucket - null)/(1 - null) and (BA - chance)/(1 - chance) for ARCHS4 and OSDR; species balance = mean of min/max per-species ratios of S2T BA and T2S bucket mAP; anti-collapse = 1 - mean species-collapse gap)
- **Overall = 0.35 T2S + 0.30 S2T + 0.15 CAP + 0.10 SERV + 0.10 HON**

A metadata-only text search (encode the held-out samples' own GEO text) is measured with the same harness as a reference ceiling; it is not eligible as an artifact by itself.

### Baseline B0 (not yet run)

Frozen `minishlab/potion-base-32M` (model2vec, 512-d, numpy-only at query time) over each training sample's GEO text, plus a ridge map (lambda 100, fit on training series only) from text space to L2-normalised expression space.
T2S scores a mapped query against the gallery by cosine; S2T maps the 37 bucket texts and takes the argmax cosine; CAP uses the same similarity matrix in both directions.

## Status at the move to the cloud (2026-11-05, about 12:40 EST)

- [x] Placement walkthrough, committed
- [x] Fixed queries, `prep.py`, CellWhisperer caption fetch (captions in gitignored `exp_data/`, not committed; re-run the fetch, about 70 s)
- [x] Cache snapshots committed in `experiments/text_search/data/`
- [ ] `prep.py --seed 0` has not finished a run (the Mac session ended mid-run); no gallery exists yet
- [ ] Text embedding cache (`encode_texts.py`, not yet written), evaluation harness, Codex harness review, rounds, write-up
