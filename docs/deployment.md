# Deploying Bridge RNA

How the app is hosted, what the hosted image carries, and which choices were made deliberately.

The app was until now a localhost instrument: `README.md` and `MEETING_QA.md` both describe running it as `.venv/bin/python app.py` and opening `http://127.0.0.1:8050`.
This document covers turning that into a URL a colleague can open.

## Platform: Fly.io

Bridge RNA is a stateful long-running process that memory-maps a 963 MB file and holds ~1.9 GB resident at peak.
That rules out the serverless hosts outright: Vercel and Netlify cap a function bundle at 250 MB and give it no persistent process to hold a memmap open between requests.

Fly.io runs an ordinary container with a real filesystem and configurable memory, and it was already authenticated on this machine.
The app is deployed to region `sjc` (San Jose), which is the closest Fly region to Ames.

## What the image carries, and what it deliberately does not

The serving app reads its artifacts from the repository root, so the container is a copy of the repo with the parts it never opens removed.

Carried, about 1.6 GB:

| Artifact | Size | Why |
| --- | --- | --- |
| `archs4_sample_embeddings_full/sample_embeddings.float16.mmap` | 963 MB | the index every cached retrieval scans |
| `archs4_sample_embeddings_full/sample_locations.parquet` | 14 MB | accession per memmap row |
| `checkpoints_performer/r7hnr92k/best_model.pt` | 522 MB | only the upload path needs it |
| `cache/` minus the expression matrix | 107 MB | coordinates, identity table, GEO metadata join, OSDR embeddings |
| `data/` minus the raw counts | 4.6 MB | ortholog map, exon lengths, canonical genes, OSDR metadata TSV |

Left out, about 658 MB:

**`data/osdr/raw/` (536 MB).**
These counts matrices are only read by the `subprocess` retrieval tier, and that tier is measurably empty whenever the cache exists: `retrieval.sample_tier` returns `cached` before it ever looks at a counts path, and the 788 unavailable samples are already unavailable for reasons that do not involve reading the file.
Only 55 of them reach `_counts_columns`, which returns an empty frozenset for a missing file and therefore still classifies them `unavailable`.
Excluding the directory changes no sample's tier.
The consequence to be aware of: without the cache the hosted app would have no fallback, where a local checkout does.

**`cache/osdr_expression.float32.npy` (122 MB).**
A precompute intermediate that makes a re-embed cheap. The serving app never opens it, and the container never re-embeds the corpus.

**The scientific stack.**
`umap-learn`, `openTSNE`, `pynndescent` and `scikit-learn` are precompute-only and are already pinned out of the serving path by a test, so the image installs `requirements-deploy.txt` rather than `requirements.txt`.
`torch` is installed from the CPU wheel index, which is the difference between roughly 200 MB and the multi-gigabyte CUDA build; the container has no GPU.

## Serving: gunicorn, one worker

`app.py`'s `app.run()` is the Flask development server and is single-threaded and unsuitable for anything shared, so the container runs gunicorn against `wsgi.py`.

**One worker, not several.** Peak resident memory was measured on the real corpus at 1,852 MB with all six coordinate sets, the tissue color-by, and a cached retrieval resident in one process.
A second worker would not share those pages (they are ordinary heap, not the memmap) and would exceed the machine.
Concurrency comes from threads instead: the workload is numpy and file IO, both of which release the GIL.

**A long timeout.** The default 30 s kills the upload path, which shells out to a subprocess that loads a 522 MB checkpoint before it embeds anything.

The machine is `shared-cpu-2x` with 4096 MB, which leaves headroom above the measured 1.85 GB peak for the upload subprocess running alongside the web worker.

## Access control

The public URL is gated behind HTTP basic auth, and this is not decoration.

The app exposes an unauthenticated file-upload endpoint that accepts up to 200 MB and spawns a torch subprocess per submission.
On an open URL that is a free denial-of-service primitive, and the LLM summary endpoint would likewise be an open proxy to whatever provider is configured.

`bridge_rna/auth.py` installs a `before_request` guard when `BRIDGE_RNA_BASIC_AUTH` is set to `user:password`, and does nothing when it is not, so local development is unaffected.
The credential is stored as a Fly secret rather than in `fly.toml`.
It is applied in `wsgi.py` rather than in `build_app`, keeping the test suite and the local entry point untouched.

## The AI summary is inert in the container

`AI_SUMMARY_PROVIDER` defaults to `ollama` at `http://127.0.0.1:11434`, which does not exist inside the container.
The summary button therefore reports that it could not reach a provider, which is honest degradation rather than a broken feature, and it is the same message a local run without Ollama produces.
Pointing the deployment at a real provider is a matter of setting `AI_SUMMARY_PROVIDER=bedrock` plus `BEDROCK_API_URL` and `BEDROCK_API_KEY` as Fly secrets; no image change is needed.

## Deploying

```bash
fly deploy                      # build remotely and release
fly secrets set BRIDGE_RNA_BASIC_AUTH='user:password'
fly logs
fly status
```

The first deploy uploads the full build context and is slow, because the artifacts are baked into the image.
Later code-only deploys reuse the cached artifact layers, which is why the Dockerfile copies artifacts before it copies source.

### Why artifacts are baked in rather than kept on a volume

A Fly volume would make code deploys cheaper and would survive image rebuilds, but it has to be seeded once over the network and it pins the app to a single machine in a single region.
Baking the artifacts in keeps the image reproducible and self-describing: any machine that pulls it can serve, with no out-of-band data step that a future deploy could forget.
The artifacts change only when the corpus is rebuilt, and the layer ordering means they are uploaded once rather than per deploy.

The alternative worth revisiting is the one `artifacts.json` already anticipates: publish the checkpoint and the memmap to Zenodo, give them a DOI, and have the image fetch them at build time.
That would shrink the build context to the repository itself and is the right move if the artifacts are ever published alongside a paper.
It was not done here because those URLs do not exist yet.
