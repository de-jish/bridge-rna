# Bridge RNA

Bridge RNA finds the closest Earth-based analogs for NASA spaceflight RNA-seq samples.

Give it a mouse RNA-seq sample from NASA's Open Science Data Repository (OSDR), and it retrieves the most transcriptomically similar human samples from the ~940,000-sample ARCHS4/GEO collection, visualizes the matches as an interactive network, and asks a language model to suggest what the retrieval implies for spaceflight biology.

> Bridge RNA is independent research and is **not affiliated with or endorsed by NASA**.
> It uses NASA's publicly available OSDR data.

> [!WARNING]
> **Retrieval results are currently not scientifically valid.**
> The authoritative gene list that defines the model's input ordering
> (`data/archs4/train_orthologs/canonical_genes.csv`) is missing from this
> repository, and the code falls back to a stand-in derived from the
> checkpoint. The stand-in reproduces the gene *count* but not the training
> gene *order*, so query vectors are built in a different gene space than the
> ARCHS4 index they are compared against.
>
> Similarity scores still look plausible - they fall in a normal range and the
> visualizations render - but they are **not meaningful and must not be
> interpreted biologically**. The pipeline, plumbing, and UI are functional and
> reviewable; only the retrieval output is affected. See
> [Known limitations](#known-limitations).

## How it works

```
OSDR counts → human-ortholog TPM vector → ExpressionPerformer embedding
            → cosine top-k over the ARCHS4 embedding index → GEO metadata → AI summary
```

A trained `ExpressionPerformer` model turns a gene-expression vector into a 512-dimensional embedding.
Every ARCHS4 sample was pre-embedded into a memory-mapped index, so a query OSDR sample can be embedded the same way and matched by cosine similarity in one pass.
The mouse query is mapped into human gene space with one-to-one orthologs and normalized (`log1p` of TPM) to match how the model was trained.

## Requirements

- **Python 3.11**, 64-bit.
  Some scientific dependencies do not publish wheels for every Python version, so 3.11 is the supported target.
- **Git** and **Git LFS**.
  The model checkpoint, embedding index, and the raw OSDR count matrices (~2 GB total) are stored in Git LFS.
  Gene annotations, ortholog tables, and the OSDR sample metadata are ordinary Git files and arrive with the clone.
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
# Random eligible sample, top-5 hits, on CPU
.venv/bin/python demo_osdr_top5.py --topk 5 --device cpu

# A specific sample, saving a report
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
| `osdr_metadata.py` | Client for the OSDR REST API. |
| `fetch_artifacts.py` | Downloads and checksum-verifies the large artifacts in `artifacts.json`. |
| `assets/style.css` | Fully tokenized UI design system. |
| `checkpoints_performer/` | Trained model checkpoint (Git LFS). |
| `archs4_sample_embeddings_full/` | Precomputed ARCHS4 embedding index and metadata (Git LFS). |
| `data/` | OSDR counts (`osdr/raw/`, Git LFS) plus sample metadata, orthologs, and gene annotations (ordinary Git files). |
| `prompts/` | The AI summary prompt template. |

## Known limitations

**The canonical gene list is missing, and retrieval output is invalid until it is restored.**

The model consumes a fixed-length expression vector whose row order is defined by a canonical gene list.
The ARCHS4 embedding index was built with the list produced by the training pipeline, at `data/archs4/train_orthologs/canonical_genes.csv`.
That file was never committed to this repository.

When it is absent, the app synthesizes `data/ensembl/canonical_genes.inferred.csv` by taking the first N entries of the alphabetically sorted `protein_coding_ortholog_genes.txt`, where N is the checkpoint's gene-embedding row count (15,165).
The result is an exact alphabetical prefix: it truncates at `WDTC1` and drops the 569 genes from there through `ZZZ3`.
It matches the model's expected input *shape* but not its expected gene *order*.

Consequences:

- Query and index vectors occupy different gene spaces, so cosine similarity between them is not meaningful.
- The failure is silent by construction. Scores land in a plausible range, the network graph renders, and the AI summary confidently describes biology downstream of a scrambled mapping.
- The existing preflight cannot catch it, because it compares gene counts only, and the synthesized file is built to match the count.
- The true ordering is not recoverable from the checkpoint, whose `config` stores no gene list. It has to be retrieved from the training host.

The CLI prints a warning whenever it falls back, and the web app shows a banner.
Everything except the retrieval results - data loading, normalization, ortholog mapping, the embedding model, the index, the UI, and the AI summary plumbing - is unaffected and reviewable.

**Other known issues**

- `demo_osdr_top5.py` loads the entire embedding index into memory (~1.9 GB, ~3.9 GB peak). `--select-best N` repeats that work per candidate. The web app streams the index in chunks and does not have this problem.
- The UI loads webfonts from `fonts.googleapis.com`, so first paint needs network access.
- There is no automated test suite.

## Licensing

The **code** in this repository is released under the [MIT License](LICENSE).

The **bundled data is not covered by that license** and carries its own terms.
OSDR data is from NASA's [Open Science Data Repository](https://osdr.nasa.gov/).
ARCHS4 is from the [Ma'ayan Lab](https://maayanlab.cloud/archs4/).
Gene annotations come from [Ensembl](https://www.ensembl.org/) and [GENCODE](https://www.gencodegenes.org/).
Review the terms of each source before redistributing the bundled data.

## Citing

If you use Bridge RNA in published work, see [`CITATION.cff`](CITATION.cff).
GitHub renders it as a "Cite this repository" link in the sidebar.
