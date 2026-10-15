# Documented flight/ground comparisons

Compare extends the existing retrieval view with fixed comparison definitions and
a shared table of reference evidence. It answers: **is this reference sample also
close to the ground group?** It does not establish a spaceflight effect, explain
the genes behind a match, or test enrichment.

## Use

1. Open **Compare** and select a documented comparison.
2. Inspect **Definition and source** for exact membership and limitations.
3. Click **Compare groups**. The default result is **Paired evidence**.
4. Read the same sample's flight and ground ranks and absolute cosine scores.
   Expand sample annotations and study context to inspect the source records.
   The table pages through 30 samples at a time, preserving the complete set.
5. Change evidence depth between 30, 100 and 250 without rerunning retrieval.
   **Network** shows the existing comparison graph; **View results on map**
   transfers both member sets and their independent top-250 neighborhoods.

The number-of-matches slider controls the network, not evidence depth. Ordinary
Sample, Cohort and Upload searches remain available. An ordinary successful
search replaces the comparison table; a failed documented comparison clears its
previous results rather than presenting stale evidence.

## Source review and fixed membership

`bridge_rna/reviewed.py` is the version-controlled definition. The review date is
2026-10-15. This is a source/data review by the software agent, **not independent
biological validation or domain-expert approval**. The presets remain exploratory.

| Preset | Flight / ground | Source review | Important limits |
|---|---:|---|---|
| OSD-100 left eye | 6 / 6 | Frozen local NASA sample export, count-column identities and [NASA study](https://osdr.nasa.gov/bio/repo/data/studies/OSD-100); [NASA portal](https://visualization.genelab.nasa.gov/data/OSD-100) lists 12 samples and flight/ground groups. | Female C57BL/6J, 37 days. Diet is unknown in the installed export. Whole left eye is not interchangeable with retina or eyecup. |
| OSD-104 soleus, both sides | 6 / 6 | Frozen local sample export and [NASA study](https://osdr.nasa.gov/bio/repo/data/studies/OSD-104); [NASA portal](https://visualization.genelab.nasa.gov/data/OSD-104) enumerates the exact 12 sample names. | Female C57BL/6J, 37 days. Left/right muscles were combined within an animal, not counted as separate replicates. |
| OSD-194 left retina | 5 / 3 | Frozen local sample export, left-retina count-column identities and [NASA study](https://osdr.nasa.gov/bio/repo/data/studies/OSD-194). The [NASA portal](https://visualization.genelab.nasa.gov/data/OSD-194) describes 13 study samples; the preset deliberately contains only these 8 flight/ground left retinas. | Female BALB/c, durations 41–42 days. Unequal, small groups. Basal controls and right retinas are excluded. |

For OSD-100 and OSD-104, flight animal IDs are M23–M28 and ground IDs M33–M38,
with the EYE or SLS tissue code respectively. They are RR-1 tissue datasets and
must not be treated as independent animal cohorts. For OSD-194, the explicit
flight IDs are F6–F10 and ground IDs G6, G8 and G9. Replicate numbering does not
pair flight and ground animals. Full sample keys are available in the UI and code.

Every execution checks unique sample keys, exact membership, study, tissue,
spaceflight arm, strain, sex, diet, genotype, habitat and allowed duration values
against the installed OSDR metadata. Missing embeddings or inconsistent metadata
refuse the comparison. No sample is silently removed. These checks cannot detect
an error already present in the source export or establish causal comparability.

OSD-238 is intentionally not a preset. Its coarse flight group combines 1G/uG
and JC/JCwFOS sample labels. A flight/ground label alone would lose the factorial
design. Additional studies require a separate source review and explicit member
list, not automatic selection through the current cohort grouping.

## Calculation and identities

Each animal's 512-D vector is L2-normalized, then the cohort mean is normalized,
exactly as in the existing cohort estimator. Both queries are scored in one
blockwise pass through the same frozen ARCHS4 index. The scanner optionally
retains two float32 score rows: approximately 7.5 MB for 940,455 references.
Ordinary retrieval continues using its bounded top-k merge without retaining
these full score rows. Scanner normalization owns a copy for both float16 and
float32 inputs and never modifies the reference array.

Full score arrays are sorted independently to obtain exact ranks. Equal float32
scores break ties by global reference row index. The display candidates are the
union of each arm's top 250; a selected depth of 30 or 100 chooses the corresponding
prefix union. Every candidate retains its actual full-index score and rank for
both arms. Being absent from one prefix never becomes a missing/low score.
The legacy argpartition scanner has unspecified tie ordering; comparisons now
have an explicit row-index tie convention. Last-bit floating-point variation
across BLAS/hardware remains possible, especially for nearly tied scores.

The table groups each GSM once by the installed cache's selected GSE. Group
ordering follows best rank, not sample-count dominance. This is an inspection
grouping, **not a claim of independent studies or corroboration**. Additional GSE
memberships remain visible when source context is configured; no many-to-many
join multiplies reference rows. No tissue/species filter changes eligibility.
The reference collection includes spaceflight-related studies and has not been
screened for OSDR cross-deposition or shared biological material. Call it
ARCHS4/GEO reference evidence, not a verified Earth-only or independent collection.

No cosine difference statistic, significance value, confidence badge, pathway
claim or stability panel is added. In particular, subtracting two normalized
centroid cosines is not the same estimand as a difference of mean per-animal
cosines. A future contrast statistic needs an explicit scientific specification.

## Optional study context

Set `BRIDGE_RNA_STUDY_CONTEXT_DB=/absolute/path/to/bridge_metadata.sqlite` to use
the existing local sidecar. It is opened with SQLite `mode=ro` and
`PRAGMA query_only=ON`; the application never creates, migrates or writes it.
Every candidate's GSM/global-index pair must match before any sidecar context is
used. A mismatch refuses the context, not retrieval. Missing files, unsupported
schemas and read failures fall back to the existing local sample annotations.

The sidecar supplies attributed study titles, summaries, experimental designs
and all recorded GSE memberships. It supplies neither embeddings nor eligibility
filters. The UI distinguishes raw source/characteristics fields from derived
tissue categories and reports missing information. No LLM generates these labels.
The 4.37 GB sidecar is not committed, required for startup, or included in a
shipment. New Python/CSS modules are included in the runtime manifest; no NASA
deployment was performed as part of this PR.

## Validation and reproduction

```bash
rtk .venv/bin/python -m pytest tests/ -q
rtk .venv/bin/python precompute/validate_comparisons.py
rtk .venv/bin/python app.py --port 8067
rtk .venv/bin/python tests/e2e_comparisons.py --url http://127.0.0.1:8067
```

The real-corpus validator checks fixed membership, unique identities, top-30
agreement and cosine differences from both legacy cohort searches. It reports
the displayed sample/GSE concentration at depths 30, 100 and 250 without calling
it significance. On the local corpus all three presets reproduced the legacy
scores with maximum absolute difference **0.0** in the validation run.

Unit tests compare all ranks with an independent direct cosine calculation,
exercise exact ranks far beyond the other prefix, deterministic ties, invalid
vectors, metadata drift, missing/duplicate sample keys and sidecar identity
failure. Identical-query negative controls produce identical scores/ranks;
swapping labels swaps them. SQLite tests preserve the input bytes and multiple
GSE memberships. These are implementation controls, not biological validation.

The browser suite runs all three presets, source disclosures, evidence depth,
network switching, map transfer, clearing by a new sample search and desktop/
narrow overflow checks. It writes the real-app screenshots under
`docs/images/comparisons/` for the PR. The ordinary sample, cohort, upload and
export browser suites remain the regression checks.

For the reproducible OSD-194 example, GSM6661548, GSM6661550, GSM6661549 and
GSM6661553 are the first four flight hits and rank 1, 3, 2 and 4 against ground.
All four belong to GSE216208. A disease-related study title does not establish
a flight-specific analogy, and "treated as control" in that study does not mean
healthy wild type. This is an evidence-inspection example, not a new discovery.

Before making stronger claims, a domain researcher must review the experimental
units and contrasts, source/reference independence, selection-aware negative
controls, animal-level or pairing-aware uncertainty, and sensitivity to dominant
reference study families. Those research conclusions have not been established
by this software PR.
