# Hosted Bridge RNA. See docs/deployment.md for what this carries and why.
#
# Layer order is deliberate: dependencies, then the ~1.6 GB of artifacts, then
# the source. Artifacts change only when the corpus is rebuilt, so a code-only
# deploy reuses their cached layers instead of re-uploading a gigabyte.

# 3.11 to match the venv the app was verified against.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# --- Dependencies ------------------------------------------------------------
# torch comes from the CPU wheel index: this container has no GPU, and the
# default index would pull the CUDA build and several gigabytes of nvidia
# libraries with it. --extra-index-url keeps PyPI available for everything else.
COPY requirements-deploy.txt ./
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements-deploy.txt

# --- Artifacts (large, and stable across code changes) -----------------------
# .dockerignore drops the OSDR raw counts and the expression intermediate; the
# app opens neither. Copied before the source so their layers stay cached.
COPY cache/ ./cache/
COPY archs4_sample_embeddings_full/ ./archs4_sample_embeddings_full/
COPY checkpoints_performer/ ./checkpoints_performer/
COPY data/ ./data/

# --- Source ------------------------------------------------------------------
COPY app.py wsgi.py ./
COPY bridge_rna/ ./bridge_rna/
COPY manifold/ ./manifold/
COPY assets/ ./assets/
COPY prompts/ ./prompts/
# precompute/embed_upload.py is the live-embedding subprocess the upload path
# shells out to, which is why precompute/ ships even though no build runs here.
COPY precompute/ ./precompute/
# embed_upload.py reaches these through manifold/bridge_rna.py's path shim:
# generate_archs4_embeddings and demo_osdr_top5, plus what they import.
COPY *.py ./

# Fail the build rather than the first request if an artifact arrived as an
# unresolved Git LFS pointer stub, which is exactly what the memmap and the
# checkpoint do on a checkout without `git lfs pull`.
RUN python -c "\
import sys; \
from pathlib import Path; \
from manifold.preflight import is_lfs_pointer; \
from manifold import paths; \
bad = [str(p) for p in (paths.ARCHS4_MMAP, paths.CHECKPOINT, paths.POINTS_META_PARQUET, paths.COORDS_PCA2) if not Path(p).exists() or is_lfs_pointer(Path(p))]; \
sys.exit('unusable artifacts in image: ' + ', '.join(bad)) if bad else print('artifact check ok')"

EXPOSE 8050

# One worker, threads for concurrency, and a timeout long enough for the upload
# path to load a 522 MB checkpoint in its subprocess. wsgi.py explains the
# single worker: peak RSS was measured at 1,852 MB in one process.
CMD ["gunicorn", \
     "--bind", "0.0.0.0:8050", \
     "--workers", "1", \
     "--threads", "4", \
     "--worker-class", "gthread", \
     "--timeout", "300", \
     "--graceful-timeout", "30", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "wsgi:application"]
