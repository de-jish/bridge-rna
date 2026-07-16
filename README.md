# NASA Bio

NASA Bio is a Dash website for querying NASA OSDR RNA-seq samples against
precomputed ARCHS4 transcriptomic embeddings and exploring the closest GEO
samples and studies.

## Windows setup and run instructions

Use **64-bit Python 3.11**. Newer or older Python versions may not have wheels
for every scientific dependency. Run the commands below in PowerShell.

### 1. Install the prerequisites

```powershell
winget install --id Git.Git -e --source winget
winget install --id Python.Python.3.11 -e --source winget
```

Close and reopen PowerShell after installation. Confirm the commands work:

```powershell
git --version
git lfs version
py -3.11 --version
```

Git for Windows normally includes Git LFS. If `git lfs version` fails, install
Git LFS separately, then reopen PowerShell:

```powershell
winget install --id GitHub.GitLFS -e --source winget
```

### 2. Clone the repository and download the large data files

```powershell
git lfs install
git clone https://github.com/de-jish/NASA-Bio.git
cd NASA-Bio
git lfs pull
```

The LFS download is approximately 2 GB and contains the model checkpoint,
embedding index, and OSDR source data. Verify that LFS downloaded real files
instead of leaving pointer files:

```powershell
git lfs ls-files
Get-Item .\best_model.pt, .\archs4_sample_embeddings_full\sample_embeddings.float16.mmap | Select-Object Name, Length
```

`best_model.pt` should be hundreds of megabytes and the `.mmap` file should be
close to one gigabyte. If either file is only about 130 bytes, rerun
`git lfs pull`.

### 3. Create a clean Python virtual environment

The commands below deliberately call the environment's `python.exe` directly.
Activation is not required, so this avoids PowerShell execution-policy errors
and prevents packages from being installed into the wrong Python environment.

If a previous `.venv` is broken, remove only that environment first:

```powershell
if (Test-Path .venv) { Remove-Item -Recurse -Force .venv }
```

Create the environment and install the dependencies:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install --only-binary=:all: -r requirements.txt
```

Verify that the environment and important packages load correctly:

```powershell
.\.venv\Scripts\python.exe -c "import sys, dash, numpy, pandas, plotly, pyarrow, requests, torch; print(sys.version); print('Dash', dash.__version__); print('PyTorch', torch.__version__)"
```

The first installation is large because it includes PyTorch and PyArrow.

### 4. Start the website

```powershell
$env:DASH_DEBUG = "0"
.\.venv\Scripts\python.exe .\app_osdr_dash.py
```

Keep that PowerShell window open and visit <http://localhost:8050>. You can
also open it from another PowerShell window:

```powershell
Start-Process http://localhost:8050
```

Stop the website with `Ctrl+C`. CPU retrieval works; an NVIDIA GPU is used
automatically when a compatible PyTorch/CUDA setup is available.

### Optional: activate the environment

Directly calling `.venv\Scripts\python.exe` as shown above is sufficient. If
you prefer activation, use:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python .\app_osdr_dash.py
```

### Common errors

- **`py` is not recognized:** install Python 3.11 with the `winget` command
  above, reopen PowerShell, and run `py -0p` to list installed interpreters.
- **`No matching distribution found`:** check that `py -3.11 --version` reports
  Python 3.11 and that Windows/Python are 64-bit.
- **`No module named dash`, `torch`, or `pyarrow`:** rerun the two pip commands
  using `.\.venv\Scripts\python.exe`; do not use a different global `pip`.
- **Model, embedding, or Parquet file is missing:** run `git lfs install` and
  `git lfs pull` from the `NASA-Bio` folder.
- **PowerShell says scripts are disabled:** use the direct `python.exe`
  commands above, which do not require activation.
- **Port 8050 is already in use:** find and stop the old process with:

  ```powershell
  Get-NetTCPConnection -LocalPort 8050 -State Listen | ForEach-Object { Stop-Process -Id $_.OwningProcess }
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
.\.venv\Scripts\python.exe .\app_osdr_dash.py
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
