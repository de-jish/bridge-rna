# File ingestion: embed an uploaded OSDR sample live and retrieve its Earth analogs

## What this adds

The Retrieve view can already answer for the 2,108 OSDR samples the manifold precompute embedded.
This feature lets a user bring an OSDR sample the corpus has never seen - upload its counts, embed it live, and get the identical output (network graph + inspector + optional LLM summary) the picker produces, computed against the same 940,455-sample ARCHS4 index.

## The one idea that makes this small

`bridge_rna/retrieval.py` is already built around a single fact: the cosine scan (`_topk_cosine_from_memmap`) is shared, and the three existing paths (cached, precomputed, demo) differ *only in where the 512-d query vector comes from*.
File ingestion is a **fourth query-vector source**.
Everything downstream - the top-k scan, the offline annotation (`_annotate_from_cache`), the `archs4_index` column, the figure, the inspector, the summary - is reused unchanged.

So the output for an uploaded sample is the same schema as the cached path, annotated from the same local `archs4_metadata.parquet` (gse / title / source_name / characteristics / tissue / species), with the same `archs4_index` that lets a hit be located on the Map.

## Where the embedding runs, and why it is a subprocess

Invariant: the serving app never imports `torch` at module scope, and `tests/test_app.py::test_the_serving_app_does_not_import_the_scientific_stack` pins that.
The existing `demo` path already embeds live, and it does so by shelling out (`run_real_retrieval` -> `demo_osdr_top5.py`).
File ingestion follows the same pattern: a new CLI, `precompute/embed_upload.py`, loads the checkpoint, embeds one counts file, writes a 512-d `.npy`, and exits.
`bridge_rna/retrieval.run_uploaded_retrieval` invokes it with `sys.executable`, loads the vector, and runs the shared scan.
No torch import enters the serving process, and the app still starts on a machine with no model.

## The preprocessing is the audited one, not a new one

Embedding an OSDR sample wrong is a silent scientific error, so `embed_upload.py` reuses the exact symbols already funnelled through `manifold/bridge_rna.py` (`load_bridge_rna_symbols`) rather than re-implementing preprocessing:

1. Read counts CSV, `index_col=0` (mouse Ensembl gene IDs), strip version suffix.
2. Map mouse Ensembl -> human ortholog symbol (`build_mouse_to_human_maps`), keep mappable, sum duplicates.
3. Reindex to the 15,165 canonical genes, fill 0.
4. TPM-normalize (`normalize_counts_to_tpm_single`), then `log1p(max(., 0))`.
5. `model.encode(x, None, normalize=False)` -> 512-d, exactly as `embed_osdr.py` and `demo_osdr_top5.py` do (species passed as `None`; the OSDR embedding path does not use the species embedding).

**Invariant 1 (gene-digest gate) is enforced**: `embed_upload.py` computes `canonical_gene_order_digest(genes)` and aborts unless it equals `CANONICAL_GENES_SHA256`.
This is byte-for-byte the same gate `embed_osdr.py` runs, so an uploaded sample is embedded in the same gene order as the corpus it is compared against, or the build refuses.

## Input contract

- Format: CSV (optionally `.tsv`/gzip), genes in rows, samples in columns, first column = mouse Ensembl gene IDs (version suffixes tolerated), matching the OSDR counts matrices already in `data/osdr`.
- Species: mouse. OSDR spaceflight data is Mus musculus (the shipped metadata is 100% mouse), and the preprocessing maps mouse Ensembl -> human ortholog space. A human-indexed matrix is out of scope for this pass and rejected with a clear message rather than silently mis-mapped.
- Sample column: if the matrix has one sample column it is used; if several, the user picks which column, defaulting to the first.
- A file that maps zero genes through the ortholog table (e.g. human Ensembl IDs, or symbols) is rejected with the reason, never embedded into a meaningless vector.
- Size: capped at `MAX_UPLOAD_BYTES`, 200 MB, which is generous headroom over the few MB a whole-transcriptome counts file actually costs.
  The cap is enforced twice, because one of the two places is too late on its own: as Flask's `MAX_CONTENT_LENGTH` (`app.py`), so an oversized body fails at the request layer with a 413 rather than arriving truncated and being parsed into a malformed matrix; and again in the callback after the base64 decode, where the user is told the actual size and the limit.

**No metadata is required or accepted alongside the counts, so this path collects nothing about the sample or about whoever uploaded it.**
That is a consequence of the design rather than a policy bolted on: the query vector is a pure function of one counts column, so tissue, flight-vs-ground and accession could only fill the inspector and the summary prompt, never a hit or a score.
Combined with the default loopback bind, an upload in the normal local setup does not cross a network at all.

## Failure handling

Every failure is surfaced as one clean line in the status banner, with the full detail logged server-side (mirrors `run_real_retrieval`):
- missing model prerequisites (checkpoint / orthologs / canonical genes / exon lengths not resolved, or an unresolved LFS pointer),
- unreadable or empty counts file,
- no ortholog-mappable genes,
- gene-digest mismatch (refuses rather than embedding wrong).

## The retrieval mode

`search_hits`-style contract: the uploaded path returns mode `"uploaded"`.
The status banner must name it, exactly as the existing invariant requires the interface to always say which path ran.
This is added to the banner's mode->label map, not special-cased.

## Files

| file | change |
| --- | --- |
| `precompute/embed_upload.py` | NEW - embed one counts file -> 512-d npy, gene-digest gated |
| `bridge_rna/config.py` | add `UPLOAD_EMBED_SCRIPT_PATH`, upload size cap |
| `bridge_rna/retrieval.py` | add `embed_uploaded_counts` + `run_uploaded_retrieval`, mode `"uploaded"` |
| `bridge_rna/layout.py` | `dcc.Upload` + sample-column control in the Retrieve rail |
| `bridge_rna/callbacks.py` | upload -> temp file -> retrieval -> `hits-store` -> same render; banner label |
| `tests/test_upload_ingestion.py` | NEW - preprocessing parity, gene-digest gate, mode, annotation schema |
| docs | IMPLEMENTATION.md, REFERENCE.md, progress.md, CLAUDE.md |

## Testing

- **Preprocessing parity**: an uploaded file built from a known OSDR sample's own counts column must embed to the same 512-d vector (to float tolerance) as that sample's precomputed cached vector - the strongest possible check that the live path matches the corpus.
- **Gene-digest gate**: a shuffled canonical gene order aborts the embed.
- **Annotation schema**: uploaded hits carry the same columns as cached hits, including `archs4_index`.
- **Mode**: the uploaded path reports `"uploaded"` and the banner names it.
- **Serving-import invariant still holds**: `embed_upload.py` is in `precompute/`, so the app-import test is unaffected.

## Verified in a browser, in a loop (2026-07-30)

`tests/e2e_upload_check.py` drives the real app through the real flow: 97 checks, three cycles through one long-lived page, about eight minutes.
The loop is the design. A one-shot upload check passes on a page that leaks state between runs, so every fixture runs three times and every cycle must reproduce cycle 1 step for step.
Correctness is anchored to the catalog path rather than to a golden file: `examples/osdr_upload_example.csv` is two columns of OSD-100, both cached OSDR samples, so an uploaded column must return exactly what the picker returns for that sample. Format variations (single-column, version-suffixed IDs, TSV, gzip) must return that same answer, rejections must draw nothing and say why, and a valid upload straight after a rejection must be correct again.

Two defects it found, both fixed:

- **Staged uploads accumulated forever.** Every upload wrote a `delete=False` temp file that nothing removed; one looped run left 32 files and 29 MB behind, surviving process exit. See `_upload_dir` / `_discard_upload` / `_sweep_abandoned_upload_dirs` in `bridge_rna/callbacks.py`, and the note there on why the reaping is PID-tagged rather than signal-based.
- **The inspector claimed an OSDR study for an uploaded sample**, repeating the "Uploaded file" study ID already under Identity and sending a lookup for a study that does not exist.

## Staging, and why it is not just a temp file

The embedder is a subprocess and takes a path, so an upload has to reach disk and survive past the callback that received it. That is the whole reason it cannot live in a `with` block. Three bounds keep it from being a leak: one process-owned directory removed at exit, a session's previous file unlinked when its next upload arrives, and PID-tagged reaping of directories whose owner is gone. Steady state is one file per active session.
