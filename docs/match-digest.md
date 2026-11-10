# Shared terms: what a set of matches has in common

Started 2026-11-10 on branch `feature/match-digest`, cut from `feature/retrieval-table` at 7c2358d.
It is a demo branch for the 2026-11-10 mentor meeting and is not shipped.

## Why

CellWhisperer (Schaefer et al., Nature Biotechnology 2025) lets a reader select a group of samples and press "Describe selected cells".
A chat model then writes a paragraph about the group.
In a hands-on test it called a skeletal-muscle selection "cardiac muscle myoblasts" and cited ACTA1 and MYL1, two skeletal-muscle genes, as strong support.
It cites no sample, count or score.
The scoping record is `cellwhisperer-scope.md` in the 2026-11-10 side-session notes.

Bridge RNA had no way to say what a result set has in common beyond the tissue column.
An uploaded sample is the sharpest case: it has no curated metadata, so the inspector shows only its identity.

This feature describes a result set with counts a reader can check, and needs no model.

## What it shows

A "Shared terms" row sits between the filters and the match table.
It lists up to eight words that appear in the GEO titles and source names of the matches much more often than across all of GEO.
Each word shows how many of the matches' series use it (for example `eye 10/22`) and the ratio to its corpus rate (for example `20x`).
Pressing a word puts it in the "Text contains" filter, so the table shows the rows behind the count.
In a comparison there is one row per cohort, each named and marked with its cohort dot, because the two result sets are independent searches.

## Decisions

1. **Counted by GEO series, not by sample.**
   One series often deposits dozens of samples with near-identical titles ("P30 CTL1", "P30 CTL2").
   Counting samples would let a single submitter's wording dominate.
   A word counts once per series among the matches and once per series in the corpus.
   This is the same batch reasoning that keeps GSE out of the map's color-bys (a 333x series lift).
   A match with no series counts as its own series.
2. **Only the fields the text filter searches: title and source name.**
   `results._TEXT_SEARCH_FIELDS` is GSM, series, title and source name.
   Using the same text means pressing a word always shows at least the rows it was counted in.
   `characteristics` carries more signal, but a word found only there would filter the table to nothing, which reads as a broken claim.
3. **A word must appear in at least 2 of the matches' series and be at least 3x its corpus rate.**
   The ratio is `observed series / expected series`, where expected is `m * S_t / N`.
   Here `m` is the matches' series count, `S_t` is the number of corpus series using the word, and `N` is the corpus series count.
   The two sides are not perfectly matched, and the bias runs both ways.
   The corpus side counts a series if any of its samples uses the word, while the match side looks only at the matched samples, which lowers the observed count.
   But a large series is both more likely to appear among the matches and more likely to use any given word, which raises it.
   So the ratio is a description of how unusual the wording is, not a test.
   No p-value is shown for the same reason: the counts are small, many words are tried at once, and a p-value would read as a finding.
4. **Ordered by how many series share the word, then by ratio.**
   That puts the widely shared words first and is easy to explain.
   Ordering by ratio alone favours a rare word used by two series.
5. **The corpus table is built offline.**
   `precompute/build_term_index.py` reads `cache/archs4_metadata.parquet` and writes `cache/archs4_terms.parquet`, one row per word with its series count.
   It keeps the offline/online split: the serving app only reads that file and counts words in at most 1,000 matches, which takes milliseconds.
   The tokenizer lives in `bridge_rna/digest.py` and the build script imports it, so the two sides cannot tokenize differently.
6. **The artifact is optional.**
   Without it the row says which command builds it, the way the map's Tissue color-by does, rather than disappearing.
7. **It never changes a match, a rank or a score.**
   It reads the hits payload the table already uses.

## Honesty wording

The row and its "i" explanation say that this is a string match over what GEO submitters wrote, counted by series.
It describes the matches' metadata, not their biology.

## Rejected

- **An LLM description of the group**: it is the CellWhisperer behaviour that motivated this feature.
  The existing AI summary remains available and separate.
- **Including `characteristics`**: see decision 2.
- **Bigrams**: worth trying later; single words already read clearly on the four queries measured.

## Measured (2026-11-10, real corpus, whitened matching)

The corpus table holds 33,215 words over 52,122 series (51,283 GEO series plus 839 samples with none) from 940,455 samples.
It is 0.3 MB and builds in 4 to 75 s depending on machine load.
Describing a result set takes under 1 ms at k=100.

| Query | k | Series | Shared terms (series, ratio) |
| --- | --- | --- | --- |
| OSD-100 left eye, flight | 5 | 3 | none: no word is shared by two of the 3 series |
| OSD-100 left eye, flight | 100 | 22 | eye 8/22, rpe 6/22, eyeball 3/22, c57bl 3/22, retina 2/22 (18x) |
| OSD-137 liver, basal | 30 | 15 | liver 13/15 (19x) |
| OSD-137 liver, basal | 100 | 38 | liver 31/38 (18x), diet, c57bl, tbt, fed, hfd |
| OSD-164 spleen, flight | 100 | 36 | spleen 33/36 (47x), non-dox, spl |
| OSD-244 thymus, basal | 100 | 48 | adipose 13/48 (29x), inguinal 12/48, mammary, iwat, white, gland, aorta, subcutaneous |
| Uploaded eye counts (`examples/osdr_upload_example.csv`) | 100 | 33 | retina 14/33 (85x), eye, rpe, rd10, c57bl, eyeball, choroid, retinal |

Ratios above 100x are shown as ">100x": they come from expected counts far below one series, so their digits are noise.
The hover text keeps the exact figure.

The thymus row is the instructive one.
A basal-control thymus sample's nearest Earth analogs are mostly inguinal white fat, and the row says so in one line.
That is a fact about the neighbours' metadata.
Whether it reflects thymic involution, dissection or batch is a question for a biologist, and the row does not answer it.

## Verified in the running app

- Sample search at k=5 and k=110, the press-to-filter toggle both ways (31 rows for "rpe", which was counted in 7 series), and the empty-case sentence.
- Cohort comparison (OSD-100 left eye, Ground Control against Space Flight): one row per cohort with its dot and name.
- The row survives a server restart (session storage) and a visit to the map and back.
- 375 px layout: the row wraps, no horizontal page scroll.
- No console errors.
- `tests/e2e_results_check.py` gained a `[shared terms]` block (9 checks).

## The AI summary

The AI summary prompt gains a "Words Shared by the Hits' GEO Titles and Source Names" section, filled by `digest.prompt_text` over the matches being summarized.
The prompt tells the model that the words describe what submitters wrote, not biology, and to cite them only as metadata.
It could not be exercised end to end on the demo machine, because Ollama has no model installed there; a test checks that the template's placeholders are exactly the ones the callback fills.

## Not done here

- Shipping: `cache/archs4_terms.parquet` must be built on, or copied to, the server, and `deploy/` has no artifact step for it yet.
  `bridge_rna/digest.py` is already in `deploy/runtime-files.json`, and without the artifact the row names the build command instead of failing.
