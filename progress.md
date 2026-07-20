# Progress

## 2026-07-20 - Pre-handoff audit

Full audit of the repository ahead of sending it to an outside researcher.
Findings are numbered as in the audit and referenced by number below.

### Verified healthy

- No secrets in tracked files.
  Earlier regex hits were gene sequences in `data/ensembl/protein_coding_genes.csv`, not credentials.
- No junk committed: no `.DS_Store`, `__pycache__`, `.env`, or `*Zone.Identifier` files are tracked.
- Git history is clean: 9 commits, 9 MiB pack.
  The 1.9 GB `.git` directory is local Git LFS cache, not repository bloat.
- All three Git LFS artifacts are materialized and checksum-verified, not pointer stubs.
- `requirements.txt` pins are all real, unyanked, and have cp311 wheels.
  An earlier suspicion that the versions looked implausible was wrong.
- The Dash app serves HTTP 200 and its preflight resolves every path.
- Scientific paths confirmed correct: L2 normalization is applied to both query and index at search time (consistent with `l2_normalize: false`); TPM conversion divides by kb, scales by 1e6, and guards a non-positive denominator; ortholog mapping filters to one-to-one, strips Ensembl version suffixes, and sums collapsed duplicate symbols.

### Fixed this session

**#2 - CLI path defaults (commit `06753ba`).**
Three argparse defaults in `demo_osdr_top5.py` pointed at directories that were never committed (`prepared_data/`, `data/archs4/`), so both documented CLI commands failed on a fresh clone.
Only the Dash app worked, because it overrides those paths on the subprocess command line.
All defaults are now anchored to `ROOT` rather than the process working directory, so the CLI behaves identically from any cwd.
Verified by running the documented command from both the repo root and an unrelated directory.

**#3 - Licensing (commit `587e695`).**
Added the MIT `LICENSE` and a `CITATION.cff`.
Without a license, default copyright applied and the researcher had no legal right to use or build on the code.
The README's licensing section now separates code (MIT) from bundled data (OSDR, ARCHS4, Ensembl, GENCODE upstream terms).

**#5 - Optional dependencies (commit `aaa5be1`).**
`archs4py` was imported but declared nowhere, so metadata enrichment silently produced bare GSM accessions with no explanation.
It is genuinely optional - it pulls in h5py, s3fs, biomart, and xalign, and is inert without the multi-GB ARCHS4 HDF5 files - so it now lives in `requirements-optional.txt` rather than burdening every install.
Both degradation paths now explain what is missing and how to fix it, and distinguish "package not installed" from "package installed but no HDF5 file present".

**#4 - Large-artifact distribution (commit `f23c5ea`).**
Added `artifacts.json` (SHA-256, size, and download URL per artifact) and `fetch_artifacts.py`.
Standard library only, so it runs before the virtualenv exists.
Size is checked before hashing; downloads land in a `.part` file and are only moved into place after the checksum matches; transfers resume via HTTP range requests and correctly restart when a server ignores `Range` and replies 200 instead of 206.
Validated against the three real artifacts plus a sandbox covering missing, corrupt, truncated, oversized-partial, resumed, unhosted, and unreachable-host cases.

### Second round: adversarial verification of the above

A verification pass (four review lenses plus a fresh-clone simulation, each finding refuted by execution before counting) found real defects in the first round's own work.

**`fetch_artifacts.py` had three defects, all contradicting its own docstring (commit `b9050fd`).**
The first-round testing only ever exercised a well-behaved server, which is why it missed them.

- `download()` ended with an unconditional `os.replace`, and verification ran afterwards, so corrupt bytes were installed at the real artifact path before anything checked them.
- A truncated body reads as a clean EOF rather than raising, so a dropped connection promoted a half-written file and destroyed the `.part`. The resume feature therefore never engaged for the case it exists to handle.
- `--force` combined with `--verify-only` reported byte-perfect artifacts as FAILED.

Now verified against a deliberately hostile server: short transfer keeps its `.part` and resumes; wrong-content-correct-size is discarded with the destination untouched; a valid local file survives a failed `--force`.

**The invalidity warning did not reach anyone actually looking at results (commit `50ddb97`).**
Commit `e7be19d` made the demo emit the warning on the app's code path but did not make the app display it - `run_real_retrieval` captures stdout and reads it only on failure, so a successful run discarded it. Saved reports had the same gap, and a report forwarded to a colleague was indistinguishable from a valid one. The README described what the tool retrieves with no caveat at all.
Fixed in three places, with the wording centralized in `INVALID_GENE_ORDER_NOTICE` so they cannot drift: a persistent web-app banner, a warning block plus sibling `.INVALID_RESULTS.txt` in saved reports, and a README callout with a Known limitations section.

**Un-pulled LFS clones and exhausted candidate loops (commit `313a850`).**
Preflight tested existence only, so a clone without `git lfs pull` reported nothing missing and then failed inside `torch.load` with an error never mentioning Git LFS. Separately, a `--select-best` run where every candidate failed produced a `TypeError` about NoneType multiplication. Both now report their actual cause.

### Outstanding

**#1 - BLOCKING: the authoritative canonical gene list is missing.**
`data/archs4/train_orthologs/canonical_genes.csv` was never committed.
Both entry points fall back to `data/ensembl/canonical_genes.inferred.csv`, which is generated by taking the first 15,165 entries of the alphabetically sorted `protein_coding_ortholog_genes.txt`.
That reproduces the gene count but not the training gene order: it is an exact alphabetical prefix truncating at `WDTC1`, dropping the 569 genes through `ZZZ3`.
Because the ARCHS4 index was built with the true ordering, query and index sit in different gene spaces, and retrievals are plausible-looking but meaningless.
`_canonical_matches_checkpoint` compares counts only, so the synthesized file satisfies the preflight instead of tripping it.
The ordering is not recoverable from the checkpoint - its `config` stores no gene list - so it must be retrieved from the training host (`/nobackupp17/woalvara/bridge-rna/data/archs4/train_orthologs/`).
Owner: Joshua. The demo now warns loudly when using the fallback.

**#4 follow-up - publishing the artifacts.**
Decision taken: host the checkpoint and embedding index on Zenodo for a citable DOI, keeping the OSDR CSVs in LFS.
Still to do: upload the files, fill in the `url`, `record_url`, and `doi` fields in `artifacts.json`, and add the DOI to `CITATION.cff`.
The recorded checksums are host-independent and already correct.
Removing the files from LFS to reclaim quota would require a history rewrite and has deliberately not been done.

**#6, #7, #8 - not yet addressed** (audited, fixes not authorized this session; all are now disclosed in the README's Known limitations section):

- #6: `app_osdr_dash.py` has no argparse, so `--help` boots the server and blocks instead of printing help. `DASH_DEBUG` defaults on with `host="0.0.0.0"`, exposing the interactive traceback console on all interfaces.
- #7: `demo_osdr_top5.py` loads the entire index into RAM (~1.93 GB, ~3.9 GB peak during normalization), and `--select-best N` re-normalizes the full index once per candidate. The Dash app already solves this with 25k-row chunking.
- #8: `load_state_dict(strict=False)` with no reporting hides checkpoint mismatch; `feature_type` falls back to `"sqr"` when the deployed checkpoint is `"flash"`; query normalization hardcodes `log1p` instead of reading the checkpoint's `normalization` field.

### Lesson worth keeping

Both rounds of defects in this session's own work shared one cause: testing only the path where things go right.
`fetch_artifacts.py` passed eleven sandbox cases against a cooperative server and still had three bugs that a hostile one exposed immediately.
The gene-list warning was verified as "printed by the demo" without checking whether anything downstream displayed it.
For work that ships to someone else, verify the failure path and the delivery path, not just the happy path.

### Notes

- The repository has no test suite. Every fix this session was validated by executing the real code path rather than by inspection.
- Local `.venv/bin/pip` is broken: its shebang points at a stale `/Users/josh/NASA Bio/` path. Use `.venv/bin/python -m pip` instead. Local-only issue; `.venv/` is gitignored and does not affect a fresh clone.
- The Dash UI loads Google Fonts from `fonts.googleapis.com`, so first paint depends on network access.
