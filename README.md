# NASA Bio

NASA Bio is a Dash website for querying NASA OSDR RNA-seq samples against
precomputed ARCHS4 transcriptomic embeddings and exploring the closest GEO
samples and studies.

## Run on Windows

Use 64-bit Python 3.11 and PowerShell. Git LFS is required because this
repository contains the trained checkpoint, embedding index, and source data.
Git for Windows normally includes Git LFS.

```powershell
git clone https://github.com/de-jish/NASA-Bio.git
cd NASA-Bio
git lfs install
git lfs pull

py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

$env:DASH_DEBUG = "0"
python .\app_osdr_dash.py
```

Open <http://localhost:8050> in a browser. The first dependency installation
and Git LFS download are large and may take a while. CPU retrieval works; an
NVIDIA GPU is used automatically when a compatible PyTorch/CUDA setup is
available.

If `git lfs` is unavailable, install it and retry:

```powershell
winget install --id GitHub.GitLFS -e
git lfs install
git lfs pull
```

## Optional configuration

The website runs without API credentials. AI summaries use a local Ollama
server by default; retrieval and the rest of the interface do not require it.
Environment variables are read only from the current process, so do not commit
real keys to the repository.

PowerShell example for NCBI metadata enrichment:

```powershell
$env:ENTREZ_EMAIL = "you@example.com"
$env:NCBI_API_KEY = "your-key"
python .\app_osdr_dash.py
```

For optional settings, see `.env.example`. The application does not
automatically load that file; set values in PowerShell or your preferred local
environment manager.

## Main files

- `app_osdr_dash.py` — Dash website, served on port 8050.
- `demo_osdr_top5.py` — retrieval command-line workflow used by the website.
- `checkpoints_performer/r7hnr92k/best_model.pt` — trained model checkpoint.
- `archs4_sample_embeddings_full/` — precomputed ARCHS4 embedding index and metadata.
- `data/` — OSDR counts, sample metadata, orthologs, and gene annotations.

The root `best_model.pt` is an identical compatibility copy of the checkpoint.
Git LFS stores the identical content only once.
