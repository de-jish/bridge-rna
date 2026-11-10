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
   The corpus side counts every sample of a series and the match side only the matched samples, so the expected count is an upper bound and the ratio is conservative.
   No p-value is shown: the counts are small, many words are tried at once, and a p-value would read as a finding.
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

## Measured

Filled in after the build: see the end of this file.
