# Ship to NASA

The canonical application is `/home/ubuntu/fm_viz_new`; its sole production
environment is `fm_viz_env`. The local command uses SFTP over the existing
OpenSSH client/configuration (`nasa-server`). It requests only the SFTP
subsystem, never an SSH shell/exec command. There is no rsync, remote Python,
package install, health request or process control from the Mac.

## Normal update

**On the Mac, in this repository:**

```sh
python3.14 deploy/ship.py ship
```

Or tell the agent **“ship my latest changes to NASA.”** The agent audits source
and permitted server files, reviews any drift, validates and transfers. First
reconciliation or unexpected server edits require reviewing the saved diff,
then `ship --reviewed-server-diff`. This is a local review flag, not permission
to read additional server paths. The reviewed local source is authoritative.

**On NASA:** run the exact apply command printed by shipping, for example:

```sh
python3 ~/fm_viz_new/.ship/incoming/SHIPMENT_ID/deploy/apply.py apply
```

Wait for **READY**, then run your unchanged startup commands:

```sh
cd ~/fm_viz_new
source fm_viz_env/bin/activate
gunicorn --workers 2 --threads 4 --timeout 120 --bind 0.0.0.0:8000 wsgi:application
```

The apply command prepares and validates before stopping the intended Gunicorn
master; it never launches Gunicorn. Do not manually start another instance while
apply is running. It checks owner, cwd, interpreter/environment, exact exec
arguments, parentage and PID/start/boot identity, then uses a Linux pidfd to stop
only that master and waits for its workers. It refuses ambiguous or hidden
process identities, never kills a shell/group and never forces a stuck process.
The serving port must be free before canonical files change.

The apply worker has a private session, closed stdin and a persistent
`~/fm_viz_new/.ship/apply.log`. Disconnecting its viewing terminal does not
cancel it. After reconnecting, inspect the log and
`python3 ~/fm_viz_new/deploy/apply.py status` (use the incoming helper path on the
first apply). Do not start Gunicorn until the status says ready.

An observer waits up to ten minutes for your manual start. It checks the real
canonical master plus repeated HTTP 200 responses on `/`, `/map`, and
`/__release`, all with the expected captured shipment ID. Only then does it
write `.ship/running.json`. If the ten-minute window expires, run on NASA:

```sh
python3 ~/fm_viz_new/deploy/apply.py confirm --wait 60
```

**On the Mac:** `python3.14 deploy/ship.py confirm` reads that receipt through
SFTP. It records verification at the receipt's timestamp; it is not a fresh
remote health check. The next shipment also imports a matching receipt.
Transferred, applied and confirmed running are separate states.

## First reconciliation and Python compatibility

The authorized SFTP comparison on 2026-09-05 found production PyArrow **22.0.0**,
Gunicorn **26.1.0**, and Python 3.14 package directories. The path/configuration
modules match local source. The other reviewed differences are older application
features/copy on the server. Small gene reference files match exactly.
The earlier 3.11 target was a lock choice, not a syntax requirement. We now
preserve the server's 3.14/PyArrow/Gunicorn profile and pin its runtime transitive
versions in `deploy/constraints.txt`. `requirements.lock` has hashes and platform
markers. `requirements.txt` in the payload is a runtime-only pointer to that lock;
the repository's development/precompute dependency sections stay local.

No one-time interpreter migration is expected. The existing interpreter link
points through `python3` to `/usr/bin/python3`; only its link text was read through SFTP, and
that external target was not accessed by the agent. SFTP distribution names are only
metadata evidence: the user-run apply still checks the actual interpreter,
installed versions, `pip check`, and real application/data preflight.

If dependencies already match, apply leaves `fm_viz_env` intact. For a future
intentional dependency change, the same apply command downloads hashed Ubuntu
wheels, installs a temporary candidate environment and validates it while the
original is still serving. Only after that does it stop the master, retain the
old environment temporarily, recreate `fm_viz_env` at its final path and install
from those already downloaded wheels. Venvs are not moved into their final path,
so activation scripts and executable shebangs are valid. Failed application
restores the old code/environment and leaves them ready for your startup command.
Success removes the temporary candidate, old environment and staging directory.
Expect a longer stop/start interval if the environment must be recreated.

All code/data writes and temporary environments are inside `fm_viz_new`. This
**user-run** helper also invokes the host's Python/pip, reads Linux `/proc` to
identify processes, and contacts localhost for verification; dependency changes
contact PyPI. Those actions are performed by you, never by the transfer agent.
The earlier separate deployment is not inspected, used, changed or cleaned up.
The SSH account itself is not technically restricted by these workflow checks.

For a new Mac, create only a local validation environment:

```sh
python3.14 -m venv .ships/validation-env
.ships/validation-env/bin/python -m pip install --require-hashes --only-binary=:all: -r deploy/requirements.lock
```

After intentional dependency changes, update `requirements.txt`,
`deploy/requirements.in` and constraints, regenerate, then reinstall locally:

```sh
uv pip compile deploy/requirements.in -c deploy/constraints.txt --python-version 3.14 --universal --generate-hashes -o deploy/requirements.lock --quiet
```

## Packaging, comparison and recovery

`deploy/runtime-files.json` is the explicit runtime allowlist. The app is Dash;
there are no templates or frontend compiler steps. The payload includes Python
application modules, six CSS/JS assets, `prompts/ai_summary_prompt.txt`, three
small gene reference tables, dependency definitions and the minimal user-run
helpers. `wsgi.py` captures `ship.json` once per worker to expose its identity.

Dynamic dependencies are retained: upload inference imports
`generate_archs4_embeddings.py`, `demo_osdr_top5.py` and
`precompute/embed_upload.py`; the demo script is also the fallback retrieval
subprocess. `osdr_metadata.py` supplies study summaries. The conditional attention
import retains `slim_performer_model.py` and `numerator_and_denominator.py`.
This does not prove every function is live. Offline precompute tools, downloaders,
examples, Markdown instructions/plans, tests, editors, old archives, logs and
virtual environments are excluded. No runtime Markdown input was found; the
runtime text prompt and required text reference table are included deliberately.

Every shipment retains `.ships/shipments/ID/payload`, `server-before` (prior
permitted code/dependency definitions), readable diffs, hashes, source commit,
dirty flag, dependency/file comparisons, validation report and transfer receipt.
Back up `.ships`; it is ignored by Git and contains no server secrets or uploads.
`last-transferred.json` is the successful read-back transfer baseline;
`last-confirmed-running.json` is separately recorded. The server's current
manifest and actual permitted file hashes are compared independently.

The SFTP client checks every path component with LSTAT, refuses symlinks, reads
only allowed small runtime files/receipts and writes only newly created staging
files. Existing files are never truncated. These checks do not provide kernel
isolation against a simultaneous malicious filesystem change; nobody should
restructure the production directories during shipping. They do not constrain
the server account's other capabilities. No private keys, credentials, `.env`,
private configuration or uploads are downloaded. No permission/account changes
are performed by shipping.

All uploads go to `.ship/incoming/ID`, even if the app is thought to be stopped.
Every transferred file is read back through SFTP and SHA-256 compared. Apply
rechecks those hashes and the recorded original files before making changes.
Untracked files are preserved. Obsolete files are removed **only** if the previous
managed manifest recorded them and they are approved code/asset paths. Existing
reference data cannot be changed automatically. Earlier untracked offline tools
remain on the server; the transfer agent does not clean them up.

Before stopping production, apply saves `.ship/transaction.json` and an exact
local-to-server-workspace backup of the files it can replace. On error it restores
them. If interrupted uncatchably during a switch, use the still-present incoming
helper on NASA:

```sh
python3 ~/fm_viz_new/.ship/incoming/SHIPMENT_ID/deploy/apply.py recover
```

Then run the usual Gunicorn startup commands. Do not delete transaction files to
bypass an error. Failed staging/work directories are kept for diagnosis; only
successful application removes its staging/work directory. There is one canonical
code installation and one environment after success, plus small `.ship` records.
No versioned server release trees or `current` symlink are introduced.

## Persistent artifacts and validation limits

These existing paths remain under `~/fm_viz_new` and are never transferred:

- `data/osdr`: `metadata/selected_sample_metadata.tsv` and referenced raw counts.
- `archs4_sample_embeddings_full`: embedding manifest, locations parquet and
  float16 memmap.
- `checkpoints_performer/r7hnr92k/best_model.pt`.
- `cache`: point/OSDR/GEO metadata, OSDR vectors, PCA/UMAP coordinates and
  projection stats; existing t-SNE coordinates are optional.
- `fm_viz_env`, server configuration/secrets and existing uploads.

Application paths default to this canonical root; existing environment overrides
remain configuration. Runtime uploads/reports default to `.runtime/tmp` after
application; existing externally configured temporary storage is not migrated.
The apply preflight uses temporary links to these data directories, removes the
links afterward, and never rebuilds or migrates scientific artifacts.

Local preflight checks formats, shapes, gene order, row/sample alignment, finite
vectors, real upload/reference cosine, sample/cohort retrieval and HTTP startup
routes from a clean staging copy. Required small references are canonical genes,
one-to-one orthologs and mouse exon lengths. Large scientific artifacts are not
downloaded just to validate the server. The user-run server preflight checks its
own actual data. Full Ubuntu installation, process control and live startup have
not been exercised by this transfer-only workflow; transfer success cannot prove
them. Schema, model, index or data changes require separate compatibility review.
