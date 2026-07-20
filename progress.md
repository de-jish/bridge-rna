# Progress

## 2026-07-20 - Canonical gene list restored

Joshua supplied the missing `canonical_genes.csv`, which clears blocking item #1 below.
Committed as `78aeca5`.

### What the file is

15,165 genes, columns `token_id,gene_symbol`, ordered alphabetically from `A1CF` to `ZZZ3`.
Verified before use: row count matches the checkpoint's `gene_embedding` rows exactly, `token_id` is contiguous 1..15165 in row order, no duplicate symbols, and it is a strict subset of `protein_coding_ortholog_genes.txt` dropping 569 scattered genes.
SHA-256 of the file is `1dbe9a5753e5982b3ddb383f35e92ce5795c21fc054fff685514831c8dadbb8d`; SHA-256 of the gene ordering is `3f887ac8d329dce3c54d26448964904c07a345940cd3d9ebab18dd1f603194c5`.

It commits as an ordinary Git file.
No `.gitattributes` pattern matches `data/archs4/train_orthologs/*.csv`, so a fresh clone gets it without `git lfs pull`, and the failure mode that produced this whole incident cannot recur from a missing LFS fetch.

Worth recording: the true ordering is alphabetical, but over a 15,165-gene *subset*.
The old stand-in assumed an alphabetical *prefix* of the 15,734-gene superset, which is why it diverged at index 18 (`AARS2` where the real list has `AARS1`) and agreed on only 18 of 15,165 positions, or 0.1%.

### Evidence that retrieval is now correct

Embedded one OSDR sample three ways and compared the resulting neighbourhoods.

| gene order | top-1 | corpus mean cosine | distinct shards in top 50 |
| --- | --- | --- | --- |
| real | 0.9964 | 0.836 | 15 |
| inferred stand-in | 0.9796 | 0.626 | 21 |
| scrambled control | 0.9739 | 0.577 | 23 |

The diagnostic signal is the corpus mean, not the top-1 score.
With the real ordering the query lands on the index manifold; both wrong orderings leave it off-manifold while the top-1 score barely moves, which is exactly why the original failure was invisible.
The three neighbourhoods share zero of 50 hits.

Biological check on OSD-100 (mouse left eye, Rodent Research-1): all three retrieved GEO series are mouse retina studies.
GSE210492 (sub-RPE deposits in retinal dystrophy), GSE205070 (`Mertk` loss-of-function), GSE143281 (retina transcriptome after `UXT` knockout).
Nothing in the pipeline is told the query tissue.

### Validity is now checked by content, not by path

The old `resolve_canonical_genes` decided authenticity by comparing the resolved path against the authoritative one, and `build_gene_list_banner` cleared the banner on a bare `Path.exists()`.
Both would trust any file sitting at the right path, including a wrong-order one, which is the same unchecked assumption that let the stand-in through.

`CANONICAL_GENES_SHA256` and `canonical_gene_order_digest` now live in `generate_archs4_embeddings.py`, the module that owns the deployed index contract.
The digest hashes the symbol sequence rather than the file bytes, so column layout and line endings do not affect it.
`demo_osdr_top5.resolve_canonical_genes` and `app_osdr_dash._canonical_gene_order_is_authoritative` both hash what they load and compare.

Verified all three cases by execution: the real file passes and emits no warning; the inferred stand-in warns and sets `USING_FALLBACK_GENE_LIST`; and a reversed-order list with the correct count passes `_canonical_matches_checkpoint` (`True`) while failing the content check (`False`).
That last case is the exact failure mode and it is now caught.

Also fixed a dead branch: preflight took the first *existing* candidate at line 462, which made the validating loop below it unreachable.
It now prefers a candidate whose ordering verifies, falling back to first-existing only when none does.

### UI fix found while screenshotting

The retrieval-network query label rendered clipped (`_C57-6J_EYE_FLT_Rep1_M23`, missing its `Mmus` prefix).
Node labels are centred on their node and the query sits at `x=0.0`, the left edge of the data extent, so Plotly's autorange left no room for the overhang.
The x-axis range is now padded by the half-width of the widest label anchored on each side, with `cliponaxis=False` and wider margins as a backstop for narrow viewports.

Added `docs/bridge-rna-interface.png`, a real retrieval captured from the running app, to the README.

### Next steps

- Item #4 follow-up (Zenodo upload) is unchanged and still outstanding.
- Consider recording the gene-list digest in `artifacts.json` so `fetch_artifacts.py --verify-only` covers it too.
- If a precomputed query-embedding parquet is ever generated, stamp it with the gene-order digest. That path takes precedence over live retrieval and currently carries no provenance.

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

**#1 - RESOLVED (see the 2026-07-20 gene-list entry at the top of this file).**
`data/archs4/train_orthologs/canonical_genes.csv` was never committed, so both entry points fell back to a stand-in that reproduced the gene count but not the training gene order, leaving query and index in different gene spaces.
Joshua supplied the real list and it is committed as `78aeca5`.
Validity is now gated on a content digest rather than on a path comparison, so a stand-in or a wrong-order file at the authoritative path is caught rather than trusted.

**#4 follow-up - publishing the artifacts.**
Decision taken: host the checkpoint and embedding index on Zenodo for a citable DOI, keeping the OSDR CSVs in LFS.
Still to do: upload the files, fill in the `url`, `record_url`, and `doi` fields in `artifacts.json`, and add the DOI to `CITATION.cff`.
The recorded checksums are host-independent and already correct.
Removing the files from LFS to reclaim quota would require a history rewrite and has deliberately not been done.

**#6 - fixed.**
`app_osdr_dash.py` had no argparse, so `--help` booted the server and blocked, and `DASH_DEBUG` defaulted on with `host="0.0.0.0"` - the Werkzeug debugger, which runs arbitrary Python for any client that reaches it, was exposed on every interface by default.
Defaults are now loopback-only with the debugger off, `--host`/`--port`/`--debug` (and `DASH_HOST`/`DASH_PORT`/`DASH_DEBUG`) are supported, and `--debug` on a non-loopback host is refused rather than warned about, since it is not a combination anyone wants by accident.
Binding `0.0.0.0` deliberately still warns.
Verified: default binds `127.0.0.1` only and is refused from the LAN address; `--debug` works on loopback; `--debug --host 0.0.0.0` exits with an explanation; `--help` prints and exits.

**#7, #8 - not yet addressed** (audited, fixes not authorized this session; both are disclosed in the README's Known limitations section):

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
