# CLAUDE.md

Guidance for Claude Code (claude.ai/code) and other agents working in this repository.

## What this project does

Bridge RNA (package slug `bridge-rna`) is an OSDR→ARCHS4 transcriptomic retrieval system.
It maps a NASA OSDR space-biology RNA-seq sample to its most similar Earth-based GEO/ARCHS4 samples, then uses an LLM to generate biological hypotheses about what the retrieval implies for spaceflight biology.

The core idea: a trained `ExpressionPerformer` model turns a gene-expression vector into a 512-dim embedding.
All ~940k ARCHS4 samples were pre-embedded into a float16 memmap; a query OSDR sample is embedded the same way and matched by cosine similarity.

This project is independent research and is not affiliated with or endorsed by NASA.
It uses NASA's publicly available Open Science Data Repository (OSDR) data.

## Environment & commands

Python 3.11 in `.venv/`.
Activate with `source .venv/bin/activate` or call `.venv/bin/python` directly.
`requirements.txt` covers retrieval (including `torch`, `pyarrow`, and `biopython`).
`requirements-optional.txt` holds `archs4py`, which is only needed for ARCHS4 HDF5 metadata enrichment and is inert without the separately downloaded `.h5` files.
Both enrichment paths degrade with an explanatory warning rather than failing.

All default paths in `demo_osdr_top5.py` are anchored to the repository root via `ROOT`, so the CLI works from any working directory.
Verify the large Git LFS artifacts with `python3 fetch_artifacts.py --verify-only` (standard library only).

Run the Dash web app (serves on `http://127.0.0.1:8050`; `--host`/`--port`/`--debug` override, and `--debug` off loopback is refused):

```bash
.venv/bin/python app_osdr_dash.py
```

Run the CLI retrieval demo (embeds one OSDR sample, prints top-k ARCHS4 hits + metadata):

```bash
.venv/bin/python demo_osdr_top5.py --topk 5 --device cpu
# Use a specific sample instead of random, and save a report:
.venv/bin/python demo_osdr_top5.py --osdr-sample-name "<id.sample name>" --save-report-prefix ./reports/run1
# Enrich hits with live GEO/PubMed metadata (needs network + email):
.venv/bin/python demo_osdr_top5.py --biopython-metadata --entrez-email you@example.com --biopython-pubmed
```

Regenerate ARCHS4 embeddings from sharded parquet (GPU-scale batch job, rarely run locally):

```bash
.venv/bin/python generate_archs4_embeddings.py --checkpoint checkpoints_performer/r7hnr92k/best_model.pt --overwrite
```

There is no test suite, linter config, or build step in this repo.

## Architecture

The retrieval flow spans five modules.
Data flows: **OSDR counts → human-ortholog TPM vector → ExpressionPerformer embedding → cosine top-k over ARCHS4 memmap → GEO metadata → LLM summary.**

- **`generate_archs4_embeddings.py`** — Defines `ExpressionPerformer`, the **canonical deployed model**, and the batch pipeline that writes the embedding memmap + `sample_locations.parquet` metadata + `embedding_manifest.json`.
  `ExpressionPerformer.encode()` (mean-pool over gene positions) is what both the batch job and the demo import and call.
  It supports two attention backends selected by the checkpoint's `feature_type`: `"flash"` (`FlashTransformerLayer`, PyTorch SDPA) or a SLiM/Performer linear-attention layer imported lazily from `slim_performer_model.py`.
  **The deployed checkpoint uses `flash`.**

- **`slim_performer_model.py`** + **`numerator_and_denominator.py`** — Google-Research SLiMPerformer linear-attention implementation.
  `numerator_and_denominator.py` is a **local inference-only reimplementation** of the prefix-sum numerator/denominator ops (`num_iter`/`den_iter` etc.); the `_ps`/`parallel` variants just delegate to the iterative path.
  Only relevant if a checkpoint uses a `favor+`/`sqr`/`relu` feature type rather than `flash`.

- **`demo_osdr_top5.py`** — Standalone CLI retrieval.
  Imports `ExpressionPerformer` from `generate_archs4_embeddings.py`.
  Does the full OSDR→query-vector transform (`load_random_osdr_sample_vector`), embeds it, runs `topk_search` against the memmap, and enriches hits via `archs4py` (HDF5 metadata) and optional Biopython Entrez GEO/PubMed lookups.
  `--select-best N` samples N random OSDR candidates and keeps the one with the highest top-1 similarity.

- **`app_osdr_dash.py`** — Dash single-file web app.
  **It shells out to `demo_osdr_top5.py` via `subprocess`** (`run_real_retrieval`) rather than importing it, and can also use precomputed query embeddings (`run_precomputed_query_retrieval`).
  Renders a Plotly network graph (query ↔ GSE ↔ GSM) and bar chart, and generates the AI hypothesis summary.
  `preflight_retrieval_requirements` validates that checkpoint gene count / attention config match the canonical gene list before running.

- **`osdr_metadata.py`** — Thin client for the OSDR REST API (`visualization.osdr.nasa.gov/biodata/api/v2`) to fetch study titles/descriptions/protocols.

### AI summary backends

`_call_ai_summary` (in `app_osdr_dash.py`) dispatches to either **Ollama** (`_call_ollama_summary`, default `http://127.0.0.1:11434`, auto-picks an available model) or **AWS Bedrock** via an API-Gateway endpoint (`_call_bedrock_summary`, `BEDROCK_API_URL`).
The prompt template lives in `prompts/ai_summary_prompt.txt` and is filled with OSDR query metadata, the hits table, and GEO study context.
Config is env-driven: `BEDROCK_API_URL`, `BEDROCK_API_KEY`, `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `ENTREZ_EMAIL`, `NCBI_API_KEY`.

## Critical domain details

- **Species mapping is central.**
  OSDR samples used here are **mouse** (`Mus musculus`); the model operates in **human** gene space.
  `build_mouse_to_human_maps` uses `data/ensembl/orthologs_one2one.txt` (one-to-one orthologs only) to map mouse Ensembl IDs → human gene symbols, then reindexes onto the checkpoint's canonical gene list.

- **Normalization must match the checkpoint.**
  Counts are converted to **TPM using mouse exon lengths** (`data/gencode/gencode_v49_mouse_gene_exon_lengths.csv`) then **`log1p`**.
  The checkpoint's `normalization` field is `log1p_tpm`; query-side normalization in the demo must reproduce this exactly or embeddings won't align.

- **Embeddings are stored un-normalized.**
  The manifest has `l2_normalize: false`; L2 normalization is applied at **search time** (`topk_search` / `_topk_cosine_from_memmap`), so cosine == dot product after normalizing both sides.

- **Embedding index facts** (`archs4_sample_embeddings_full/embedding_manifest.json`): 940,455 samples × 512 dims, float16 memmap, `feature_type: flash`.
  Paths inside the manifest are from the original NAS training host (`/nobackupp17/...`) — ignore them; the app resolves files relative to repo root via `EMBEDDING_DIR`.

- **The canonical gene list is present and authoritative.**
  `data/archs4/train_orthologs/canonical_genes.csv` holds 15,165 genes, columns `token_id,gene_symbol`, ordered alphabetically from `A1CF` to `ZZZ3`.
  It is a strict subset of `protein_coding_ortholog_genes.txt` (15,734) that drops 569 scattered genes, and its row count matches the checkpoint's `gene_embedding` rows exactly.
  It commits as an ordinary Git file; no `.gitattributes` pattern routes it to LFS.

  The gene list defines the row order of the expression vector and must match the order used to build the ARCHS4 index.
  `ExpressionPerformer._encode_hidden` indexes `gene_embedding` by position (`torch.arange(genes)`), so slot `i` **is** gene `i` and the model has no other channel for gene identity.
  A list of the right length in the wrong order pairs every gene's embedding with another gene's expression value and still yields plausible-looking cosine scores.

  **Authenticity is checked by content, not by path.**
  `CANONICAL_GENES_SHA256` in `generate_archs4_embeddings.py` is the SHA-256 of the gene ordering (symbols joined by newline, so the `token_id` column and file formatting do not affect it).
  `resolve_canonical_genes` in `demo_osdr_top5.py` and `_canonical_gene_order_is_authoritative` in `app_osdr_dash.py` both hash the list they load and compare against it.
  Judging by path was the original blind spot: it trusts whatever happens to sit at the authoritative location. Do not reintroduce a path-identity or count-only test as the gate for validity.

  If the digest does not match, the CLI prints a warning and stamps saved reports, and the app shows a persistent banner.
  `_infer_canonical_genes_from_checkpoint` still synthesizes `data/ensembl/canonical_genes.inferred.csv` as a last-resort stand-in, but it now fails the content check rather than satisfying the preflight.
  `_canonical_matches_checkpoint` remains count-only and is retained solely as a coarse fallback ordering criterion.

## Data layout

- `checkpoints_performer/r7hnr92k/best_model.pt` — trained checkpoint; `torch.load(...)["config"]` holds `hidden_dim`, `num_heads`, `feature_type`, `normalization`, etc.
- `archs4_sample_embeddings_full/` — the memmap (`sample_embeddings.float16.mmap`), `sample_locations.parquet` (per-sample GEO accession + shard location), `embedding_manifest.json`.
- `data/osdr/metadata/selected_sample_metadata.tsv` — OSDR sample table; `data/osdr/raw/*.csv` — per-study unnormalized count matrices.
- `data/archs4/train_orthologs/canonical_genes.csv` — the authoritative gene list; defines expression-vector row order.
- `data/ensembl/` — orthologs, protein-coding gene lists, and `canonical_genes.inferred.csv` (a stand-in, not the canonical list).
- `data/gencode/` — mouse exon lengths.
- `docs/` — README images.
- `prompts/ai_summary_prompt.txt` — the LLM prompt template.

The large model, embedding, and data files are stored in **Git LFS**.
Run `git lfs pull` after cloning to fetch them.

## Conventions

- Model definitions are **duplicated by import, not copied**: `demo_osdr_top5.py` and any consumer import `ExpressionPerformer` / `_strip_module_prefix` from `generate_archs4_embeddings.py`.
  Change the model in one place.
- `.venv/`, the checkpoint, the memmap, and `*Zone.Identifier` sidecar files are large/local artifacts — don't commit or modify them.
- `--device cuda` is the default in the scripts but they fall back to CPU automatically when CUDA is unavailable.
- The UI design system is fully tokenized in `assets/style.css` (`:root` custom properties); re-skin the whole app by editing those tokens.
