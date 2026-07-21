# Bridge RNA

**What does spaceflight look like in a cell? Ask 940,000 Earth samples.**

Bridge RNA takes a tissue flown on a NASA mission and finds its closest biological relatives on the ground.

Developed by the **Space Biosciences Research Branch at NASA Ames Research Center**.

NASA flies rare biology.
A rodent mission returns tens of tissues, not the tens of thousands that terrestrial studies accumulate, and every one of them is expensive, irreplaceable, and hard to interpret in isolation.
Meanwhile the published human transcriptome keeps growing.
Bridge RNA connects the two: it treats every NASA spaceflight sample as a query into that entire corpus, so a tissue flown aboard the International Space Station lands next to the Earth-based studies that look most like it - no manual literature search, no prior guess about what it should resemble.

Hand it a mouse RNA-seq sample from NASA's Open Science Data Repository (OSDR).
It returns the most transcriptomically similar human samples out of the ~940,000-sample ARCHS4/GEO collection, draws the matches as an interactive network, and asks a language model what the retrieval implies for spaceflight biology.

![The Bridge RNA web interface, showing a mouse eye spaceflight sample from OSD-100 matched against its nearest ARCHS4 neighbours](docs/bridge-rna-interface.png)

**The screenshot above is a real retrieval, and it makes the case better than any description.**
The query is a mouse left-eye sample flown on NASA's Rodent Research-1 mission (OSD-100).
The three GEO series it pulls back are all mouse retina studies: sub-RPE deposit accumulation in retinal dystrophy (GSE210492), `Mertk` loss-of-function traits (GSE205070), and the retina transcriptome after `UXT` knockout (GSE143281).

Nothing in the pipeline is told what tissue the query came from.
It found the retina on its own, out of 940,000 candidates, from expression alone.
That is the capability NASA Rodent Research data has been waiting for.

## How it works

```
NASA OSDR counts → human-ortholog TPM vector → ExpressionPerformer embedding
                 → cosine top-k over the ARCHS4 embedding index → GEO metadata → AI summary
```

A trained `ExpressionPerformer` model turns a gene-expression vector into a 512-dimensional embedding.
Every ARCHS4 sample was pre-embedded into a memory-mapped index, so a query sample from NASA's OSDR can be embedded the same way and matched by cosine similarity in one pass.
The mouse query is mapped into human gene space with one-to-one orthologs and normalized (`log1p` of TPM) to match how the model was trained.

The result is a search engine over Earth biology, keyed by a NASA spaceflight sample.
A researcher studying a NASA flight tissue gets back the published human studies that look most like it transcriptomically, ranked, annotated, and summarized.

## Requirements

- **Python 3.11**, 64-bit.
  Some scientific dependencies do not publish wheels for every Python version, so 3.11 is the supported target.
- **Git** and **Git LFS**.
  The model checkpoint, embedding index, and the raw NASA OSDR count matrices (~2 GB total) are stored in Git LFS.
  Gene annotations, ortholog tables, and the NASA OSDR sample metadata are ordinary Git files and arrive with the clone.
- Optional: a local [Ollama](https://ollama.com) install for AI summaries (see below).

## Quickstart

### 1. Clone and fetch the large files

```bash
git lfs install
git clone https://github.com/de-jish/bridge-rna.git
cd bridge-rna
git lfs pull
```

`git lfs pull` downloads roughly 2 GB.

Then verify that the large artifacts arrived intact:

```bash
python3 fetch_artifacts.py --verify-only
```

This checks the model checkpoint and embedding index against the SHA-256 digests recorded in `artifacts.json`, and exits non-zero if anything is missing, truncated, or corrupt.
It needs no dependencies beyond the Python standard library, so you can run it before creating the virtual environment.

> **If `git lfs pull` fails with a bandwidth or quota error:** this repository's LFS payload exceeds GitHub's free allowance, so cloning may not fetch the large files.
> If that happens, ask the maintainer for the artifact download links, add them to the `url` fields in `artifacts.json`, and run `python3 fetch_artifacts.py` to fetch and verify them directly.

### 2. Create a virtual environment and install dependencies

**macOS / Linux:**

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

**Windows (PowerShell):**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Calling the environment's Python directly (as above) avoids activation and execution-policy issues.
The first install is large because it includes PyTorch and PyArrow.

**Optional metadata enrichment.**
`requirements.txt` covers everything needed for retrieval.
To also label hits with tissue and study metadata read from the ARCHS4 HDF5 files, install the extras:

```bash
.venv/bin/python -m pip install -r requirements-optional.txt
```

This pulls in `archs4py`, which additionally needs the ARCHS4 HDF5 files from <https://archs4.org/download>.
Those files are several GB each and are **not** bundled here.
Without them, retrieval still works and hits are reported as bare GSM accessions.

### 3. Run the app

```bash
# macOS / Linux
.venv/bin/python app_osdr_dash.py

# Windows
.\.venv\Scripts\python.exe .\app_osdr_dash.py
```

Open <http://localhost:8050> and stop the server with `Ctrl+C`.

If anything retrieval depends on is missing, the app still starts and shows an amber banner naming exactly what is wrong and how to fix it, rather than failing only once you press Search.
A clone whose Git LFS payload never arrived is the usual cause.

By default the app binds to `127.0.0.1`, so it is reachable only from your own machine, and the debugger is off.
`--host`, `--port`, and `--debug` (or `DASH_HOST`, `DASH_PORT`, `DASH_DEBUG`) change that:

```bash
.venv/bin/python app_osdr_dash.py --port 8060      # different port
.venv/bin/python app_osdr_dash.py --debug          # hot reload, loopback only
.venv/bin/python app_osdr_dash.py --host 0.0.0.0   # expose on your network
```

`--debug` enables the Werkzeug debugger, which runs arbitrary Python for anyone who can reach the port.
Combining it with a non-loopback `--host` is refused rather than warned about.
This is Flask's development server either way, so put a real WSGI server in front of it for anything beyond local or lab use.

### Command-line retrieval

The web app shells out to the same retrieval workflow you can run directly:

```bash
# Random eligible NASA OSDR sample, top-5 hits, on CPU
.venv/bin/python demo_osdr_top5.py --topk 5 --device cpu

# A specific NASA OSDR sample, saving a report
.venv/bin/python demo_osdr_top5.py --osdr-sample-name "<id.sample name>" --save-report-prefix ./reports/run1
```

## Configuration

The app runs with no credentials.
All configuration is read from environment variables in the process that starts the app; nothing is loaded from a file automatically.
See `.env.example` for the full list, and never commit real keys.

### AI summaries (Ollama, default)

Summaries use a local Ollama server by default, which needs no account or API key.
The default model is `gemma3:4b`.

```bash
# macOS (Homebrew)
brew install ollama
brew services start ollama
ollama pull gemma3:4b
```

The app connects to `http://127.0.0.1:11434` automatically.
Set `OLLAMA_MODEL` to use a different installed model, or `OLLAMA_BASE_URL` for a remote server.
If no Ollama server is reachable, retrieval and the rest of the interface still work; only the AI summary is skipped.

### Optional metadata enrichment (NCBI)

Live GEO/PubMed enrichment uses NCBI Entrez, which asks callers to identify themselves:

```bash
export ENTREZ_EMAIL="you@example.com"
export NCBI_API_KEY="your-key"   # optional, raises the rate limit
```

### Optional AWS Bedrock backend

Set `BEDROCK_API_URL` (and `BEDROCK_API_KEY` if your gateway requires one) to route AI summaries through an API-Gateway-fronted Bedrock endpoint instead of Ollama.

## Project layout

| Path | What it is |
| --- | --- |
| `app_osdr_dash.py` | Dash web app, served on port 8050. |
| `demo_osdr_top5.py` | Command-line retrieval workflow (also called by the web app). |
| `generate_archs4_embeddings.py` | `ExpressionPerformer` model + the batch job that builds the embedding index. |
| `slim_performer_model.py`, `numerator_and_denominator.py` | Linear-attention backend, used only for non-flash checkpoints. |
| `osdr_metadata.py` | Client for the NASA OSDR REST API. |
| `fetch_artifacts.py` | Downloads and checksum-verifies the large artifacts in `artifacts.json`. |
| `assets/style.css` | Fully tokenized UI design system. |
| `checkpoints_performer/` | Trained model checkpoint (Git LFS). |
| `archs4_sample_embeddings_full/` | Precomputed ARCHS4 embedding index and metadata (Git LFS). |
| `data/archs4/train_orthologs/canonical_genes.csv` | Authoritative gene list (15,165 genes) defining expression-vector row order. |
| `data/` | NASA OSDR counts (`osdr/raw/`, Git LFS) plus sample metadata, orthologs, and gene annotations (ordinary Git files). |
| `docs/` | README images. |
| `prompts/` | The AI summary prompt template. |

## Architecture

Retrieval spans five modules, and data flows in one direction through them, starting at NASA OSDR:

```
NASA OSDR counts → human-ortholog TPM vector → ExpressionPerformer embedding
                 → cosine top-k over the ARCHS4 memmap → GEO metadata → LLM summary
```

**`generate_archs4_embeddings.py`** defines `ExpressionPerformer`, the deployed model, along with the batch job that writes the embedding memmap, `sample_locations.parquet`, and `embedding_manifest.json`.
`ExpressionPerformer.encode()` mean-pools over gene positions, and is what both the batch job and the CLI call.
Two attention backends are selected by the checkpoint's `feature_type`: `flash` (`FlashTransformerLayer`, PyTorch SDPA) or a SLiM/Performer linear-attention layer imported lazily from `slim_performer_model.py`.
The deployed checkpoint uses `flash`.

Rebuilding the index from sharded parquet is a GPU-scale batch job and is rarely run locally:

```bash
.venv/bin/python generate_archs4_embeddings.py \
  --checkpoint checkpoints_performer/r7hnr92k/best_model.pt --overwrite
```

**`slim_performer_model.py`** and **`numerator_and_denominator.py`** are the Google Research SLiMPerformer linear-attention implementation.
The latter is a local inference-only reimplementation of the prefix-sum numerator/denominator ops, where the `_ps` and `parallel` variants delegate to the iterative path.
Neither is reached unless a checkpoint uses a `favor+`, `sqr`, or `relu` feature type instead of `flash`.

**`demo_osdr_top5.py`** is the standalone CLI.
It performs the whole NASA OSDR to query-vector transform in `load_random_osdr_sample_vector`, embeds it, runs `topk_search` against the memmap, and enriches hits through `archs4py` and optional Biopython Entrez lookups.
`--select-best N` samples N random NASA OSDR candidates and keeps whichever has the highest top-1 similarity.

**`app_osdr_dash.py`** is the single-file Dash app.
It shells out to `demo_osdr_top5.py` through `subprocess` rather than importing it, renders the Plotly network graph and bar chart, and generates the AI summary.
`preflight_retrieval_requirements` checks every required path, the checkpoint's attention config, and the canonical gene list before any retrieval runs.

**`osdr_metadata.py`** is a thin client for the NASA OSDR REST API, used to fetch NASA study titles, descriptions, and protocols.

`_call_ai_summary` dispatches to either Ollama or an API-Gateway-fronted Bedrock endpoint.
The prompt template in `prompts/ai_summary_prompt.txt` is filled with the NASA OSDR query metadata, the hits table, and GEO study context.

## Implementation notes

**Species mapping is central.**
The NASA OSDR samples here are mouse, flown on NASA rodent missions, and the model operates in human gene space.
Crossing that species boundary is what lets a NASA flight tissue be compared against human ground studies at all.
`build_mouse_to_human_maps` uses `data/ensembl/orthologs_one2one.txt`, restricted to one-to-one orthologs, to map mouse Ensembl IDs onto human gene symbols, then reindexes onto the canonical gene list.

**Normalization has to match the checkpoint.**
Counts are converted to TPM using mouse exon lengths from `data/gencode/`, then `log1p`.
The checkpoint's `normalization` field is `log1p_tpm`, and query-side normalization must reproduce it exactly or the embeddings will not align.

**Embeddings are stored un-normalized.**
The manifest records `l2_normalize: false`, and L2 normalization is applied at search time, so cosine similarity equals the dot product once both sides are normalized.

**Index facts.**
940,455 samples, 512 dimensions, float16 memmap, `feature_type: flash`.
The paths recorded inside `embedding_manifest.json` point at the original training host and should be ignored; everything resolves relative to the repository root at runtime.

**Conventions.**
The model is shared by import rather than copied, so `ExpressionPerformer` changes in exactly one place.
`--device cuda` is the default in both scripts, and both fall back to CPU automatically when CUDA is unavailable.
The UI design system is fully tokenized in `assets/style.css` as `:root` custom properties, so the whole app can be re-skinned by editing those tokens.

## The canonical gene list

The model consumes a fixed-length expression vector whose row order is defined by a canonical gene list, at `data/archs4/train_orthologs/canonical_genes.csv`.
This matters more than it sounds: `ExpressionPerformer` indexes its `gene_embedding` table by *position*, so slot `i` of the vector simply is gene `i`.
The model carries no other notion of which gene a slot holds.

A list with the right length but the wrong order therefore pairs every gene's learned embedding with a different gene's expression value.
The resulting cosine scores still land in a plausible range and every visualization still renders, so the failure is silent by construction.

That is not hypothetical.
This repository previously shipped without the real list and fell back to a stand-in synthesized from the checkpoint's gene count, taking the alphabetical prefix of `protein_coding_ortholog_genes.txt`.
It agreed with the true ordering on **18 of 15,165 positions**, and every retrieval published before the list was restored was meaningless.

The list is now present and committed as an ordinary Git file (no `git lfs pull` needed).
Its integrity is pinned by content rather than by path: `CANONICAL_GENES_SHA256` in `generate_archs4_embeddings.py` records the SHA-256 of the gene ordering, and both entry points hash the list they load and compare.
A stand-in, a corrupted copy, or a wrong-order file sitting at the authoritative path all fail that check, and the CLI prints a warning while the web app shows a banner.
Checking the path instead of the content was the original blind spot, so the check is deliberately content-based.

If you want to confirm the ordering yourself, embed a query and compare its mean cosine against the whole index.
A correct gene order puts the query on the index manifold (~0.84 for the sample above); a scrambled one leaves it off-manifold (~0.58) while the top-1 score barely moves.

## Known limitations

- `_canonical_matches_checkpoint` in `app_osdr_dash.py` still compares gene counts only. It is now backed by the content check above, but on its own it cannot tell a correct list from a wrong-order list of the same length.
- If a precomputed query-embedding parquet is ever added (see `PRECOMPUTED_QUERY_EMBEDDING_CANDIDATES`), it takes precedence over live retrieval and carries no record of which gene ordering produced it.

**Other known issues**

- `demo_osdr_top5.py` loads the entire embedding index into memory (~1.9 GB, ~3.9 GB peak). `--select-best N` repeats that work per candidate. The web app streams the index in chunks and does not have this problem.
- The UI loads webfonts from `fonts.googleapis.com`, so first paint needs network access.
- There is no automated test suite.

## Licensing

The **code** in this repository is released under the [MIT License](LICENSE).

The **bundled data is not covered by that license** and carries its own terms.
The spaceflight data is from NASA's [Open Science Data Repository (OSDR)](https://osdr.nasa.gov/), NASA's open repository for space biology and life sciences data.
ARCHS4 is from the [Ma'ayan Lab](https://maayanlab.cloud/archs4/).
Gene annotations come from [Ensembl](https://www.ensembl.org/) and [GENCODE](https://www.gencodegenes.org/).
Review the terms of each source before redistributing the bundled data.

## Citing

If you use Bridge RNA in published work, see [`CITATION.cff`](CITATION.cff).
GitHub renders it as a "Cite this repository" link in the sidebar.

Bridge RNA was developed by the Space Biosciences Research Branch at NASA Ames Research Center, which studies how living systems respond to the space environment.
