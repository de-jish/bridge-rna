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

**Eight threads, not four, and a 900 s timeout.** Both are the upload path's doing, and both started out wrong.

A live embed holds one thread for minutes while the browser keeps firing short Dash callbacks behind it.
Sizing the pool to roughly the core count starves the interface for the whole duration of an upload, and threads share the worker's heap, so the extra four cost almost nothing: the 1,852 MB is paid once.

The timeout matters for the same reason.
An in-app embed was measured at **300 s** on this machine, which is exactly where the original 300 s timeout kills the worker mid-request and leaves the browser holding the previous figure - a wrong answer rather than an error.

**The proxy's concurrency limits are not the thread count.**
They were `soft_limit = 4` / `hard_limit = 8`, chosen to mirror the worker, and that was a mistake with a visible symptom.
While an upload occupies a thread the in-flight request count sails past eight, and past the hard limit the Fly proxy **rejects** rather than queues.
A rejected Dash callback is a figure that never updates, which presented as an uploaded search returning the *previous* column's result - a plausible-looking wrong answer, which is the worst kind.
They are 20 and 40 now.
A concurrency limit should describe when to shed load; one slow callback is not a reason to turn away the short ones around it.

**`--keep-alive 75`, which is not a tuning knob but a bug fix.**
gunicorn defaults it to 2 seconds, which is shorter than the idle timeout of the proxy in front of it.
The proxy then occasionally reuses a connection at the moment gunicorn is closing it, gets `ECONNRESET`, and serves a 502.

This is recorded because it was found the hard way and because of what it looked like.
The first browser run against the deployed app failed two checks the local run passes: the rail's parameter readout showed UMAP's settings while PCA was selected, and the console carried a 502 plus a callback error.
The evidence pointed at the connection rather than the app - exactly one reset among hundreds of successful requests, no worker crash, no OOM, no traceback - and it was not reproducible locally, where the browser talks to gunicorn directly with no proxy in between.

The lesson generalizes past this one flag.
A stale parameter readout is precisely the failure invariant 7 exists to prevent, and it arrived anyway, by a route invariant 7 cannot see: the readout was correct and the transport dropped it.
An invariant enforced in the application does not survive the network on its own, which is the argument for checking the deployment rather than only the code.

The machine is `performance-2x` with 4096 MB, which leaves headroom above the measured 1.85 GB peak for the upload subprocess, itself measured at 1,036 MB resident.

### The CPU must be dedicated, and the upload path is what decides it

This started as `shared-cpu-2x` and had to change, on a measurement rather than a hunch.

| | one live embed of `examples/osdr_upload_example.csv` |
| --- | --- |
| laptop | 10.4 s wall, 28 s CPU |
| `shared-cpu-2x` | **13 m 36 s wall**, 1 m 55 s CPU |
| `performance-2x` | **1 m 35 s wall**, 1 m 38 s CPU |

The shared machine's numbers are the tell: 115 seconds of CPU spread across 816 seconds of wall clock is **14% utilization** on a process that is pegging both vCPUs.
That is Fly throttling a shared vCPU down to its sustained baseline once the burst credits are spent, and no amount of application tuning touches it.
On dedicated CPU the same work runs at 103% utilization, which is the same job no longer being held back.

The map and cached retrieval were tolerable on the shared machine, so it would have been easy to ship it and never notice.
A thirteen-minute upload is not a slow feature, it is a broken one - it exceeds the gunicorn timeout, the browser's patience, and any reviewer's.

Two things are worth knowing before someone tries to tune this further.
The container needs about 3.5x the CPU-seconds the laptop does for identical output (98 s against 28 s), which is Apple Silicon's matrix units against a generic x86 build, not a misconfiguration.
And the embed only uses about one core even when two are available, so `performance-4x` would buy very little; the work does not parallelize past what it already uses.

`swap_size_mb` was set at first and then removed, because it silently did nothing - the running machine reported `SwapTotal: 0` - and the memory it was insuring against never materialized: 1.9 GB stayed available with the embed at full size. A setting that has no effect is worse than no setting, because it reads as protection that is not there.

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

## Verifying a deployment

All three browser suites take `--base-url` and `--http-auth`, so the checks that verify a local run verify the hosted one:

```bash
.venv/bin/python tests/e2e_check.py        --base-url https://bridge-rna.fly.dev --http-auth user:password
.venv/bin/python tests/e2e_cohort_check.py --base-url https://bridge-rna.fly.dev --http-auth user:password
.venv/bin/python tests/e2e_upload_check.py --base-url https://bridge-rna.fly.dev --http-auth user:password
```

This is not a formality, and it is worth being blunt about why.
**Every defect in this document was found by running these against the deployment, and not one of them was visible locally.**
A proxy sits between the browser and gunicorn, the CPU is different, and the machine may be asleep; none of that exists when the checks drive a subprocess on the same laptop.

`e2e_target.patience()` multiplies every wait by 5 for a remote target and by 1 locally.
That single knob is the honest shape of the difference: a page paints in milliseconds locally and over a network hosted, an embed is 10 seconds locally and minutes hosted, and a stopped machine adds a cold start on top. These are constant factors on the same waits, not different waits.

Two harness bugs it exposed are worth remembering, because both are the same mistake and both produced *plausible wrong answers* rather than errors.

A dropdown's displayed value updates in the browser immediately, while the callback carrying that choice to the server has not landed.
So "the column picker holds the GC column" passed while the search still ran on the previous column, and the uploaded result looked like a correct answer to the wrong question.

And waiting for the upload slot to be merely non-empty is satisfied by the *previous* upload's text, so the run continued with a stale `upload-store` pointing at a file the app had already unlinked.
That surfaced as a rejection with the wrong reason - "Uploaded counts file not found", naming the previous fixture.

The lesson both times: **wait for a condition that names the thing you just did.** A condition that the previous step also satisfies is not a wait at all, and on a fast local machine it looks exactly like one.

### Where this stands, honestly

`e2e_check.py` (45) and `e2e_cohort_check.py` (60) pass against the deployment reliably.

`e2e_upload_check.py` does not yet pass end to end against it *reliably*, and the reason is latency rather than correctness.
One full run got through 23 consecutive checks with no failures, including the check that matters most - both columns of the example file embedded live and each matching the catalog path exactly:

```
FLT column   5 hits, 248.1 s, matches OSD-100|Mmus_C57-6J_EYE_FLT_Rep1_M23
GC  column   5 hits, 272.6 s, matches OSD-100|Mmus_C57-6J_EYE_GC_Rep1_M33
```

That is the equality the whole file is built around, and it holds on the deployment.
Later runs of the same suite failed elsewhere, differently each time, against a machine that had auto-stopped and cold-started in between.

So: **the upload feature is verified on the deployment; the upload suite is not yet a reliable gate for it.**
Anyone picking this up should start from the cheap fix - pin a machine warm (`min_machines_running = 1`) for the duration of the run - before adding any more patience to the harness. Four to five minutes per uploaded search against a suite that performs a dozen of them is the real constraint, and no wait multiplier removes it.

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
