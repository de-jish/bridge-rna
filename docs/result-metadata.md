# Retrieved-match metadata

Implemented on `feature/result-metadata`, using the existing result inspector,
network, Dash controls and styles. Production has not been contacted or activated.

For the implementation walkthrough, design rationale and limitations, see
[Bridge RNA metadata upgrade explained](metadata-upgrade-explained.md).

## Source and validation

The earlier comparison is at
`/Users/josh/Bridge-RNA-analysis/2026-10-04-box-361046303309/report.md`.
Its Additional Metadata evidence identified [bridge_metadata.sqlite in Box](https://app.box.com/file/2499909479489).
The MCP download was still blocked on October 6; the user supplied the complete
local file instead. No Box original was changed and no database copy was made.
The local file is outside Git at `/Users/josh/Downloads/bridge_metadata.sqlite`.
Disk space was checked before validation; low available disk space was another
reason to validate it in place. Production should retain it under ignored `cache/`.

- Box file **2499909479489**, version **2773286356289**.
- Exactly **4,371,132,416 bytes**; SHA-1 **b207d54b15a3a21f662cd78407794731fe338c85**.
- SQLite `integrity_check`: `ok`; foreign key violations: **0**.
- Assertion schema: `bridge-metadata-v1`.
- All **940,455** samples match the existing locations parquet, in embedding
  order, on GSM, global index, species ID, shard index/file and row in shard.
- GSM and global index are unique; canonical metadata has exactly one row per
  sample. Study memberships are fetched separately, so **210,140** samples with
  multiple studies cannot multiply result rows. All **36** recorded conflicts
  remain available in details and CSV.

Actual useful coverage, after treating exact placeholder annotations as unknown:

| Field | Samples | Coverage |
| --- | ---: | ---: |
| Organ | 404,266 | 42.986% |
| Disease | 297,603 | 31.645% |
| Treatment | 94,990 | 10.100% |
| Experimental setting | 438,119 | 46.586% |
| Perturbation | 438,119 | 46.586% |
| Study summary through any membership | 931,581 | 99.056% |

Treatment has 96,172 non-null entries but 1,182 are exact NA/N/A/NULL variants.
These show as Unknown; their stored values and attribution remain accessible.
Normal, No, vehicle and untreated are meaningful values. Combined source terms
are preserved rather than split into inferred diagnoses or treatments.
Organ/disease coverage is concentrated in human samples. Mouse records have
20,063 organ annotations, 479 disease annotations and no experimental-setting
annotations. Study summaries are study-level context, not sample-level evidence
that an intervention or disease applies. GEO protocols are shown when present.
Validation-only AI assertions are excluded; active source assertions, methods,
evidence, ontology IDs, conflicting values and study retrieval provenance remain.

## Behavior and files

- `bridge_rna/result_metadata.py`: bounded read-only SQLite lookups by GSM with
  embedding-index checks. A validation receipt must match the file's device,
  inode, size, modification time and change time before Box provenance is used.
  Missing/stale/invalid metadata leaves the original search usable and displays
  an unavailable message. Revalidate after moving, replacing or modifying a file.
- `bridge_rna/results.py`: ordered result rows, exact filters and CSV serialization.
- `bridge_rna/results_ui.py`: compact controls, ranked sample selector and
  expandable source/study details. `panels.py`, `layout.py`, `callbacks.py` and
  `assets/retrieve.css` connect these to the existing interface.
- `precompute/validate_result_metadata.py`: offline full hash, integrity, schema,
  coverage and sample-mapping checks; writes only a small JSON receipt/report.

Filters cover organ, disease, treatment, experimental setting and study accession.
Options come only from the current retrieved matches, including Unknown.
Filters combine with AND; a study matches any membership. The count is **X of Y
matches shown**. Reset and each new search restore the complete retrieved set.
The ranked selector, retrieval network and CSV use the same filtered identities.
The map, similarity profile and AI summary retain the full search, as explained
beside the filters. This is not retrieval within a metadata subset.

Source organ annotations do not replace derived tissue. No model, embedding,
preprocessing, similarity, ranking, cohort grouping or projection code changed.
For comparisons, rows use A's original ranking followed by B-only matches in B's
ranking, with one row per GSM and both original ranks/scores retained. CSV follows
that exact visible order, preserves full numeric scores, includes source values,
study/context/conflict/assertion JSON and Box provenance. Formula-leading text is
escaped for spreadsheet safety. Empty results disable export.

## Preview and checks

From the repository, reuse the existing environment:

```sh
rtk proxy .venv/bin/python precompute/validate_result_metadata.py --database "$HOME/Downloads/bridge_metadata.sqlite"
rtk proxy env BRIDGE_METADATA_DB="$HOME/Downloads/bridge_metadata.sqlite" BRIDGE_METADATA_VALIDATION="$PWD/reports/result-metadata-validation.json" .venv/bin/python app.py --port 8066
```

Open <http://127.0.0.1:8066>. Search the default OSD-100 eye sample with 30 matches;
expand Filter retrieved matches, try Eye, Unknown or a treatment, select a ranked
match, expand Metadata sources and study context, and export the visible rows.

Verification evidence is local and ignored by Git:
`reports/result-metadata-validation.json`, `reports/result-metadata-performance.json`,
and screenshots/checks in `.lavish/metadata-upgrade/`.

```sh
rtk proxy .venv/bin/python -m pytest -q
rtk proxy .venv/bin/python tests/e2e_result_metadata.py --url http://127.0.0.1:8066
rtk proxy env BRIDGE_METADATA_DB="$HOME/Downloads/bridge_metadata.sqlite" BRIDGE_METADATA_VALIDATION="$PWD/reports/result-metadata-validation.json" python3.14 deploy/ship.py build
```

The unit suite passes 542 tests. Real-data browser verification covers sample,
cohort/comparison, upload, map, metadata, filters/reset/Unknown/empty states and
CSV. Desktop, 390px and 320px layouts retain accessible details without horizontal
overflow or overlapping network/profile panels. Existing broad browser suites
passed 205 general, 178 cohort and 41 upload checks. The focused metadata suite
passed 30 checks, comparing displayed/exported identities and full scores to the
raw retrieval, including both comparison arms.

The final metadata benchmark for 30 real matches, including receipt validation
and source assertions, took 23.9 ms first lookup and 5.0–6.3 ms thereafter while
Python allocation tracing was enabled. Peak traced Python allocations were
373 KB; SQLite's page cache is capped at 2 MiB per connection and closed after
each lookup. JSON grew from 12 KB to 87 KB for this query. These are representative
local measurements, not a guarantee for every study. The database is never read
at application import or loaded into a full-corpus cache. Full validation took
26.4 seconds separately from serving. Clean Python 3.14 staging with the existing
production dependency lock passed real-data preflight and upload/reference cosine
1.00000000; no new dependency was introduced.
The final validated local bundle is
`.ships/shipments/20261006T190118Z-20db138f/`; its receipt records
`result_metadata_checked: true`. It was built from metadata commit `a14529c`.

## Later deployment, only when authorized

Follow [deployment.md](deployment.md). This task prepared and validated code
locally only. No server file, process, environment or Box original was changed.
The deployment allowlist also now includes the previously omitted `profile.py`;
that independent packaging fix must survive any metadata rollback.

1. Revalidate the local file and run the build command above. When deployment is
   authorized, run the same environment-prefixed command with `ship` instead of
   `build`. Review its permitted source/server diffs and backups; only rerun with
   `--reviewed-server-diff` after that review. Record the printed **SHIPMENT_ID**.
   Shipping transfers code exclusively into `.ship/incoming/SHIPMENT_ID` through
   SFTP via `nasa-server`. Transferred is not applied or confirmed running.
2. Transfer the database separately into a new, uniquely named directory such as
   `.ship/incoming/metadata-2773286356289-SHIPMENT_ID`. Do **not** add it to the code
   payload: bundle verification rejects unlisted files. Check remote free space
   first with SFTP `df /home/ubuntu/fm_viz_new`; allow at least the 4.37 GB file plus
   normal operational headroom. Use the existing `deploy.sftp_client.SFTP.checked`
   path checks to refuse symlinks at every parent and ensure the destination does
   not exist; create only the new incoming directory, using the actual shipment
   ID in this local command:

   ```sh
   rtk proxy python3 - SHIPMENT_ID <<'PY'
   import sys
   from deploy.common import valid_id
   from deploy.sftp_client import SFTP
   shipment = sys.argv[1]
   valid_id(shipment)
   destination = '.ship/incoming/metadata-2773286356289-' + shipment
   with SFTP() as sftp:
       if sftp.checked(destination, missing=True) is not None:
           raise SystemExit('Destination exists; use a fresh incoming ID.')
       sftp.mkdir(destination)
   PY
   ```

   Do not read private config,
   uploads or paths outside this application. With no concurrent directory changes,
   transfer using the SFTP subsystem, never a remote shell:

   ```text
   sftp nasa-server
   put /Users/josh/Downloads/bridge_metadata.sqlite /home/ubuntu/fm_viz_new/.ship/incoming/metadata-2773286356289-SHIPMENT_ID/bridge_metadata.sqlite
   ls -l /home/ubuntu/fm_viz_new/.ship/incoming/metadata-2773286356289-SHIPMENT_ID/bridge_metadata.sqlite
   bye
   ```

   Confirm the size is 4,371,132,416 bytes. Do not overwrite/retry onto an existing
   partial destination; use a fresh incoming ID. Keep the local original.
3. **The user, on NASA**, chooses an unused final file under real `cache/`, moves
   the incoming database there on the same filesystem, and validates at that
   final location. The user must confirm these paths are regular directories/files,
   not symlinks. Example (substitute the actual shipment ID):

   ```sh
   cd /home/ubuntu/fm_viz_new
   export BRIDGE_METADATA_DB="$PWD/cache/bridge_metadata-2773286356289.sqlite"
   export BRIDGE_METADATA_VALIDATION="$PWD/cache/bridge_metadata-2773286356289.validation.json"
   test ! -e "$BRIDGE_METADATA_DB" && test ! -L "$BRIDGE_METADATA_DB" && mv -n .ship/incoming/metadata-2773286356289-SHIPMENT_ID/bridge_metadata.sqlite "$BRIDGE_METADATA_DB"
   fm_viz_env/bin/python .ship/incoming/SHIPMENT_ID/precompute/validate_result_metadata.py --database "$BRIDGE_METADATA_DB" --locations "$PWD/archs4_sample_embeddings_full/sample_locations.parquet" --report "$BRIDGE_METADATA_VALIDATION"
   ```

   Stop if any check fails. Persist the two absolute environment settings in the
   user's normal startup configuration; the agent must not access that configuration.
   The receipt is local to this file placement and is not copied from the Mac.
4. **The user, on NASA**, with those two variables set, runs the exact printed
   apply command. Its preflight requires both verified metadata and unchanged
   result identities/scores. After **READY**, the user runs the existing Gunicorn
   commands from deployment.md. The agent never runs remote helpers, installers
   or process controls. Complete the usual user-run confirmation and subsequent
   SFTP receipt check. Verify filters and a CSV in the live app after manual start.

   ```sh
   python3 ~/fm_viz_new/.ship/incoming/SHIPMENT_ID/deploy/apply.py apply
   # Only after READY, in the shell with both metadata variables set:
   cd ~/fm_viz_new
   source fm_viz_env/bin/activate
   gunicorn --workers 2 --threads 4 --timeout 120 --bind 0.0.0.0:8000 wsgi:application
   ```

   If the observer times out, the user runs
   `python3 ~/fm_viz_new/deploy/apply.py confirm --wait 60`. On the Mac,
   `rtk proxy python3.14 deploy/ship.py confirm` reads the resulting receipt via SFTP.

## Rollback

For interrupted/failed apply, the user runs the documented incoming
`deploy/apply.py recover`, then the normal manual Gunicorn startup. Recovery is
not the rollback command for a successfully applied release.

For a completed metadata release, create a new reviewed shipment of the prior
application state while retaining the independent profile packaging fix. In a
clean local checkout, revert metadata commit **a14529c** while retaining profile
packaging fix **f774e7d**:

```sh
rtk git switch -c rollback/result-metadata
rtk git revert a14529c
rtk proxy .venv/bin/python -m pytest -q
rtk proxy python3.14 deploy/ship.py build
```

Then use the same authorized ship/review/user-apply/manual-start/confirm sequence.
The unchanged model, embeddings and reference data require no migration or restore.
Leave the validated database/receipt in cache for investigation; previous code
does not load them. Do not delete data or restore private configuration.

Future work, outside this update: search within a metadata subset; display
validated precomputed pathways; add verified missing spaceflight studies.
