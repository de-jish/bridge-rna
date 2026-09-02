# Retrieval Neighborhood Explorer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a map-linked drawer that summarizes and lists the 250 true nearest ARCHS4 samples for an active retrieval while preserving the existing requested top-k overlay.

**Architecture:** The retrieval layer retains a larger ranked prefix from the same cosine scan and stores a compact, JSON-safe neighborhood beside the existing hits. A pure neighborhood module owns aggregation; the map translates the stored rows into a uniform highlight trace and renders an adaptive Overview / Studies / Samples drawer. Map viewport geometry never defines the evidence set.

**Tech Stack:** Python 3.11, Dash 4.4, Plotly 6.8, pandas 3.0, NumPy 2.4, pytest, Playwright, project CSS tokens.

**Spec:** `docs/superpowers/specs/2026-09-02-retrieval-neighborhood-explorer-design.md`

## Global Constraints

- `NEIGHBORHOOD_DEPTH` is fixed at 250 in version one; requested retrieval depth remains 3–30.
- Requested hits and the evidence neighborhood come from one cosine scan; no second ARCHS4 memmap pass is allowed.
- Requested hits are the exact ranked prefix of the evidence neighborhood for the same query.
- The wider map marks use one uniform size, opacity, and color and never draw a line, hull, enclosing ring, or rank ramp.
- Summary percentages state both coverage and denominator; missing metadata is not folded silently into “Other.”
- Cached OSDR, upload, cohort, and comparison paths are supported. Unsupported stale/demo payloads degrade without breaking the existing overlay.
- The million-point ARCHS4 cloud keeps no hover payload; only the capped 250-point evidence trace gains metadata.
- No new runtime dependency, configurable depth, export, AI-written summary, viewport statistics, or new route is added.
- Preserve and do not stage the pre-existing local changes in `.env.example`, `app.py`, `requirements.txt`, and `wsgi.py` unless the user separately asks to include them.

## File Structure

- Create `bridge_rna/neighborhoods.py`: constants, JSON-safe payload construction, deterministic summaries, study grouping, and sample filtering.
- Create `tests/test_neighborhoods.py`: pure domain tests for counts, coverage, ordering, filtering, and stale payloads.
- Modify `bridge_rna/retrieval.py`: one-scan requested/evidence result helpers for sample, upload, and cohort retrieval.
- Modify `bridge_rna/callbacks.py`: put neighborhood payloads into `hits-store` for sample, upload, cohort, and comparison searches.
- Modify `manifold/theme.py`: shared evidence/focus mark constants.
- Modify `manifold/render.py`: evidence and focus traces behind/around existing requested-hit traces.
- Modify `manifold/layout.py`: static explorer shell, tabs, arm control, and accessible row builders.
- Modify `manifold/callbacks.py`: payload selection, open/close state, drawer rendering, focus state, and map wiring.
- Modify `assets/map.css`: docked drawer, bottom-sheet breakpoint, list/table, focus, and empty-state styling.
- Modify `tests/test_retrieval.py`, `tests/test_cohorts.py`, `tests/test_render.py`, and `tests/test_app.py`: integration and callback invariants.
- Modify `tests/e2e_check.py`: real-browser explorer flow, keyboard, 3-D, and responsive checks.
- Modify `README.md`, `docs/design-notes.md`, and `progress.md`: shipped behavior, scientific boundary, and verification record.

---

### Task 1: Pure neighborhood model and deterministic summaries

**Files:**
- Create: `bridge_rna/neighborhoods.py`
- Create: `tests/test_neighborhoods.py`

**Interfaces:**
- Consumes: ranked `pandas.DataFrame` rows with any subset of `gsm`, `gse`, `title`, `source_name`, `characteristics`, `tissue`, `species`, `score`, and `archs4_index`.
- Produces: `NEIGHBORHOOD_DEPTH: int`, `build_payload(frame, *, label="", depth_requested=NEIGHBORHOOD_DEPTH) -> dict`, `summarize(payload) -> dict`, `study_groups(payload) -> list[dict]`, and `sample_rows(payload, query="") -> list[dict]`.

- [ ] **Step 1: Write failing payload and aggregation tests**

Create `tests/test_neighborhoods.py` with representative known, missing, and unassigned rows:

```python
from __future__ import annotations

import math

import pandas as pd

from bridge_rna import neighborhoods as N


def _frame() -> pd.DataFrame:
    return pd.DataFrame([
        {"gsm": "GSM1", "gse": "GSE10", "title": "Retina A",
         "tissue": "Eye / retina", "species": "Mus musculus",
         "score": 0.99, "archs4_index": 4},
        {"gsm": "GSM2", "gse": "GSE10", "title": "Retina A",
         "tissue": "Eye / retina", "species": "Mus musculus",
         "score": 0.98, "archs4_index": 7},
        {"gsm": "GSM3", "gse": "GSE20", "title": "Brain B",
         "tissue": "Brain / CNS", "species": "Homo sapiens",
         "score": 0.96, "archs4_index": 9},
        {"gsm": "GSM4", "gse": "", "title": "",
         "tissue": None, "species": float("nan"),
         "score": 0.95, "archs4_index": None},
    ])


def test_payload_is_ranked_json_safe_and_explicit_about_depth():
    payload = N.build_payload(_frame(), label="OSD-100 · eye")
    assert payload["depth_requested"] == 250
    assert payload["depth_returned"] == 4
    assert payload["metric"] == "cosine"
    assert payload["space"] == "embedding-512d"
    assert [r["rank"] for r in payload["hits"]] == [1, 2, 3, 4]
    assert payload["hits"][3]["species"] == ""
    assert payload["hits"][3]["archs4_index"] is None
    assert not any(isinstance(v, float) and math.isnan(v)
                   for row in payload["hits"] for v in row.values())


def test_summary_uses_metadata_coverage_as_the_denominator():
    summary = N.summarize(N.build_payload(_frame()))
    assert summary["depth"] == 4
    assert summary["tissue"]["covered"] == 3
    assert summary["tissue"]["items"][0] == {
        "label": "Eye / retina", "count": 2, "share": 2 / 3}
    assert summary["species"]["covered"] == 3
    assert summary["study_count"] == 2
    assert summary["top_three_study_samples"] == 3
    assert "2 of 3" in summary["sentence"]


def test_studies_group_unassigned_rows_and_sort_deterministically():
    groups = N.study_groups(N.build_payload(_frame()))
    assert [g["gse"] for g in groups] == ["GSE10", "GSE20", ""]
    assert groups[0]["sample_count"] == 2
    assert groups[0]["best_rank"] == 1
    assert groups[0]["dominant_tissue"] == "Eye / retina"
    assert groups[-1]["display_gse"] == "No GSE recorded"


def test_sample_filter_matches_accession_study_tissue_species_and_title():
    payload = N.build_payload(_frame())
    assert [r["gsm"] for r in N.sample_rows(payload, "gsm3")] == ["GSM3"]
    assert [r["gsm"] for r in N.sample_rows(payload, "retina")] == ["GSM1", "GSM2"]
    assert len(N.sample_rows(payload, "")) == 4


def test_missing_or_unavailable_payload_has_honest_empty_structures():
    unavailable = N.unavailable_payload("Run this retrieval again.")
    assert N.summarize(unavailable)["available"] is False
    assert N.study_groups(None) == []
    assert N.sample_rows({}, "") == []
```

- [ ] **Step 2: Run the focused tests and verify the missing-module failure**

Run:

```bash
rtk pytest tests/test_neighborhoods.py -q
```

Expected: collection fails with `ImportError` for `bridge_rna.neighborhoods`.

- [ ] **Step 3: Implement the pure neighborhood module**

Create the module with a deliberately small serialization schema and pure aggregators:

```python
from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from .util import _safe_str

NEIGHBORHOOD_DEPTH = 250
ROW_TEXT_FIELDS = (
    "gsm", "gse", "title", "source_name", "characteristics",
    "tissue", "species",
)


def _index(value: Any) -> int | None:
    try:
        return None if pd.isna(value) else int(value)
    except (TypeError, ValueError):
        return None


def build_payload(frame: pd.DataFrame | None, *, label: str = "",
                  depth_requested: int = NEIGHBORHOOD_DEPTH) -> dict:
    rows = []
    source = frame if frame is not None else pd.DataFrame()
    for rank, (_, raw) in enumerate(source.reset_index(drop=True).iterrows(), 1):
        row = {field: _safe_str(raw.get(field)) for field in ROW_TEXT_FIELDS}
        score = pd.to_numeric(raw.get("score"), errors="coerce")
        row.update({
            "rank": rank,
            "score": None if pd.isna(score) else float(score),
            "archs4_index": _index(raw.get("archs4_index")),
        })
        rows.append(row)
    return {
        "available": True,
        "reason": "",
        "label": _safe_str(label),
        "depth_requested": int(depth_requested),
        "depth_returned": len(rows),
        "metric": "cosine",
        "space": "embedding-512d",
        "hits": rows,
    }


def unavailable_payload(reason: str) -> dict:
    return {
        "available": False, "reason": _safe_str(reason), "label": "",
        "depth_requested": NEIGHBORHOOD_DEPTH, "depth_returned": 0,
        "metric": "cosine", "space": "embedding-512d", "hits": [],
    }
```

Implement category counts so the denominator contains only non-empty values,
study groups so blank GSEs form one explicit final group, score statistics with
`numpy.median/min/max`, and sample filtering over the concatenated lowercase
text fields. Sort category rows by `(-count, label)`, studies by
`(-sample_count, best_rank, display_gse)`, and samples by stored rank. Construct
the summary sentence from the leading tissue only when tissue coverage is
nonzero; otherwise state that no tissue metadata is available for the returned
depth.

- [ ] **Step 4: Run the pure tests**

Run:

```bash
rtk pytest tests/test_neighborhoods.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit the domain slice**

```bash
rtk git add bridge_rna/neighborhoods.py tests/test_neighborhoods.py
rtk git diff --cached --check
rtk git commit -m "feat: model retrieval evidence neighborhoods"
```

---

### Task 2: Retain requested hits and top-250 evidence in one scan

**Files:**
- Modify: `bridge_rna/retrieval.py`
- Modify: `tests/test_retrieval.py`
- Modify: `tests/test_upload_ingestion.py`

**Interfaces:**
- Consumes: Task 1's `NEIGHBORHOOD_DEPTH`.
- Produces: `split_ranked_hits(ranked, requested_k, neighborhood_depth) -> tuple[pd.DataFrame, pd.DataFrame]`, `search_hits_with_neighborhood(samples_df, sample_id, topk, *, neighborhood_depth=250, entrez_email=None, enable_biopython_metadata=True) -> tuple[pd.DataFrame, pd.DataFrame | None, str]`, and `run_uploaded_retrieval_with_neighborhood(counts_path, sample_column, topk, *, neighborhood_depth=250, entrez_email=None, enable_biopython_metadata=True) -> tuple[pd.DataFrame, pd.DataFrame]`.
- Preserves: existing `search_hits(samples_df, sample_id, topk, entrez_email=None, enable_biopython_metadata=True) -> tuple[pd.DataFrame, str]` and `run_uploaded_retrieval(counts_path, sample_column, topk, entrez_email=None, enable_biopython_metadata=True) -> pd.DataFrame` interfaces.

- [ ] **Step 1: Add failing one-scan and prefix tests**

Add to `tests/test_retrieval.py`:

```python
def test_sample_search_keeps_top_250_in_the_same_scan(monkeypatch, corpus):
    calls = []
    real = retrieval._topk_cosine_from_memmap

    def counted(index_vecs, q_vec, k):
        calls.append(k)
        return real(index_vecs, q_vec, k)

    monkeypatch.setattr(retrieval, "_topk_cosine_from_memmap", counted)
    samples = _samples_frame(corpus)
    hits, neighborhood, mode = retrieval.search_hits_with_neighborhood(
        samples, str(samples["sample_id"].iloc[0]), topk=5,
        neighborhood_depth=250, enable_biopython_metadata=False)

    assert mode == "cached"
    assert calls == [250]
    assert len(hits) == 5
    assert len(neighborhood) == 250
    assert hits["archs4_index"].tolist() == neighborhood.head(5)["archs4_index"].tolist()
    assert hits["score"].tolist() == neighborhood.head(5)["score"].tolist()


def test_existing_search_interface_still_scans_only_requested_depth(monkeypatch, corpus):
    calls = []
    real = retrieval._topk_cosine_from_memmap
    monkeypatch.setattr(
        retrieval, "_topk_cosine_from_memmap",
        lambda index_vecs, q_vec, k: (calls.append(k) or real(index_vecs, q_vec, k)))
    samples = _samples_frame(corpus)
    hits, mode = retrieval.search_hits(
        samples, str(samples["sample_id"].iloc[0]), topk=4,
        enable_biopython_metadata=False)
    assert mode == "cached" and len(hits) == 4
    assert calls == [4]


def test_only_requested_hits_are_sent_for_network_enrichment(monkeypatch, corpus):
    seen = []
    monkeypatch.setattr(
        retrieval, "_enrich_hits_from_ncbi_eutils",
        lambda frame, _email: (seen.append(len(frame)) or frame))
    samples = _samples_frame(corpus)
    hits, neighborhood, _ = retrieval.search_hits_with_neighborhood(
        samples, str(samples["sample_id"].iloc[0]), topk=5,
        neighborhood_depth=250, entrez_email="test@example.org",
        enable_biopython_metadata=True)
    assert seen == [5]
    assert len(hits) == 5 and len(neighborhood) == 250
```

Add an upload test that monkeypatches `embed_uploaded_counts` and counts one
call to `_topk_cosine_from_memmap`, asserting requested rows equal the first
five evidence rows.

- [ ] **Step 2: Run the focused tests and verify the missing-interface failures**

```bash
rtk pytest tests/test_retrieval.py tests/test_upload_ingestion.py -q
```

Expected: new tests fail because the two neighborhood-aware functions do not
exist.

- [ ] **Step 3: Add ranked splitting and sample retrieval wrapper**

In `bridge_rna/retrieval.py`, import `NEIGHBORHOOD_DEPTH` and add:

```python
def split_ranked_hits(ranked: pd.DataFrame, requested_k: int,
                      neighborhood_depth: int
                      ) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = ranked.sort_values("score", ascending=False).reset_index(drop=True)
    requested = ordered.head(max(1, int(requested_k))).copy().reset_index(drop=True)
    neighborhood = ordered.head(max(1, int(neighborhood_depth))).copy().reset_index(drop=True)
    return requested, neighborhood


def search_hits_with_neighborhood(
    samples_df: pd.DataFrame,
    sample_id: str,
    topk: int,
    *,
    neighborhood_depth: int = NEIGHBORHOOD_DEPTH,
    entrez_email: str | None = None,
    enable_biopython_metadata: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame | None, str]:
    row = samples_df.loc[samples_df["sample_id"] == sample_id]
    if row.empty:
        raise ValueError(f"Unknown sample_id: {sample_id}")
    scan_k = max(int(topk), int(neighborhood_depth))
    if cached_query_vector(sample_id) is not None:
        ranked = run_cached_query_retrieval(sample_id, scan_k)
        hits, neighborhood = split_ranked_hits(ranked, topk, neighborhood_depth)
        if enable_biopython_metadata and _safe_str(entrez_email):
            hits = _enrich_hits_from_ncbi_eutils(hits, _safe_str(entrez_email))
        return hits, neighborhood, "cached"
    # Keep the existing precomputed/demo selection. Return None when that path
    # cannot produce a complete, locatable ranked prefix without another scan.
    hits, mode = search_hits(
        samples_df, sample_id, topk, entrez_email,
        enable_biopython_metadata)
    return hits, None, mode
```

Do not implement `search_hits` in terms of the new wrapper; doing so would make
the compatibility path scan 250 rows. Keep the old function's requested-depth
behavior intact.

- [ ] **Step 4: Add upload retrieval wrapper without re-embedding or re-scanning**

Extract the upload body into a neighborhood-aware function:

```python
def run_uploaded_retrieval_with_neighborhood(
    counts_path: str | Path,
    sample_column: str | None,
    topk: int,
    *,
    neighborhood_depth: int = NEIGHBORHOOD_DEPTH,
    entrez_email: str | None = None,
    enable_biopython_metadata: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    q_vec = embed_uploaded_counts(counts_path, sample_column)
    index_vecs, _, _ = _load_archs4_index()
    scan_k = max(int(topk), int(neighborhood_depth))
    idx, score = _topk_cosine_from_memmap(index_vecs, q_vec, scan_k)
    ranked = _annotate_from_cache(idx, score)
    ranked["archs4_index"] = idx.astype(int)
    hits, neighborhood = split_ranked_hits(ranked, topk, neighborhood_depth)
    if enable_biopython_metadata and _safe_str(entrez_email):
        hits = _enrich_hits_from_ncbi_eutils(hits, _safe_str(entrez_email))
    return hits, neighborhood
```

Make existing `run_uploaded_retrieval` call this helper with
`neighborhood_depth=topk` and return only `hits`, so its public behavior and
cost stay unchanged.

- [ ] **Step 5: Run focused retrieval/upload tests**

```bash
rtk pytest tests/test_retrieval.py tests/test_upload_ingestion.py -q
```

Expected: all pass.

- [ ] **Step 6: Commit the one-scan sample/upload slice**

```bash
rtk git add bridge_rna/retrieval.py tests/test_retrieval.py tests/test_upload_ingestion.py
rtk git diff --cached --check
rtk git commit -m "feat: retain evidence neighbors in one retrieval scan"
```

---

### Task 3: Cohort/comparison evidence and cross-route payloads

**Files:**
- Modify: `bridge_rna/retrieval.py`
- Modify: `bridge_rna/callbacks.py`
- Modify: `tests/test_cohorts.py`
- Modify: `tests/test_app.py`

**Interfaces:**
- Consumes: Task 1's `build_payload`/`unavailable_payload`; Task 2's split and neighborhood-aware sample/upload helpers.
- Produces: `run_cohort_retrieval_with_neighborhood(sample_ids, topk, *, neighborhood_depth=250) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, StabilityMeasurement | None]`; `hits-store.neighborhood`; `hits-store.comparison.neighborhood_b`.
- Preserves: `run_cohort_retrieval(sample_ids, topk) -> tuple[pd.DataFrame, np.ndarray, StabilityMeasurement | None]`.

- [ ] **Step 1: Add failing cohort depth and stability tests**

Add to `tests/test_cohorts.py`:

```python
def test_cohort_evidence_is_250_but_stability_uses_requested_depth(monkeypatch, corpus):
    from bridge_rna import cohorts as C

    real_measure = C.measure_stability
    captured = {}

    def measure(**kwargs):
        captured.update(kwargs)
        return real_measure(**kwargs)

    monkeypatch.setattr(C, "measure_stability", measure)
    hits, neighborhood, rows, stability = retrieval.run_cohort_retrieval_with_neighborhood(
        _cohort_keys(corpus, 4), topk=6, neighborhood_depth=250)
    assert len(hits) == 6 and len(neighborhood) == 250
    assert hits["archs4_index"].tolist() == neighborhood.head(6)["archs4_index"].tolist()
    assert len(captured["pooled_top"]) == 6
    assert all(len(row) == 6 for row in captured["loo_tops"])
    assert stability.depth == 6
```

Also add a scan-count test by wrapping `_topk_cosine_matrix` and asserting one
call with `k=250` for one cohort invocation.

- [ ] **Step 2: Run cohort tests and confirm the missing-helper failure**

```bash
rtk pytest tests/test_cohorts.py -q
```

Expected: the new helper is absent.

- [ ] **Step 3: Implement pooled evidence while slicing stability inputs**

Extract the current cohort body into:

```python
from collections.abc import Sequence


def run_cohort_retrieval_with_neighborhood(
    sample_ids: Sequence[str],
    topk: int,
    *,
    neighborhood_depth: int = NEIGHBORHOOD_DEPTH,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, Any]:
    # Build rows, leave-one-out vectors, q_mat, and validate missing samples
    # exactly as run_cohort_retrieval does today.
    scan_k = max(int(topk), int(neighborhood_depth))
    idx, score = _topk_cosine_matrix(index_vecs=index_vecs, q_mat=q_mat, k=scan_k)
    ranked = _annotate_from_cache(idx[0], score[0])
    ranked["archs4_index"] = idx[0].astype(int)
    hits, neighborhood = split_ranked_hits(ranked, topk, neighborhood_depth)
    depth = int(topk)
    stability = measure_stability(
        members=ids,
        pooled_top=idx[0, :depth],
        loo_tops=idx[1:1 + len(loo), :depth],
        member_tops=idx[1 + len(loo):1 + len(loo) + len(rows), :depth],
        depth=depth,
    )
    return hits, neighborhood, rows, stability
```

Keep the current guards and vector-building code verbatim around the shown
scan/split. Make `run_cohort_retrieval` call the new helper with
`neighborhood_depth=topk` and discard the neighborhood.

- [ ] **Step 4: Add callback payload helpers and tests**

At module scope in `bridge_rna/callbacks.py`, add one serializer used by all
three search callbacks:

```python
def _evidence_payload(frame: pd.DataFrame | None, label: str = "") -> dict:
    if frame is None:
        return unavailable_payload(
            "Run this retrieval again to build its 250-sample evidence neighborhood.")
    return build_payload(frame, label=label)
```

Add tests in `tests/test_app.py` that call this helper with a two-row frame and
with `None`, asserting JSON-safe ranks, the fixed requested depth, and the rerun
reason.

- [ ] **Step 5: Wire every search callback to the new evidence result**

Make these exact payload additions:

```python
# Single sample
hits_df, neighborhood_df, mode = search_hits_with_neighborhood(
    samples_df=samples_df, sample_id=sample_id, topk=int(topk),
    entrez_email=email_value, enable_biopython_metadata=enable_biopython)
payload["neighborhood"] = _evidence_payload(
    neighborhood_df, label=_safe_str(q_row.get("sample_name")))

# Upload
hits_df, neighborhood_df = run_uploaded_retrieval_with_neighborhood(
    counts_path=upload_store["path"], sample_column=column, topk=int(topk),
    entrez_email=email_value, enable_biopython_metadata=enable_biopython)
payload["neighborhood"] = _evidence_payload(neighborhood_df, label=column)

# Cohort A
hits_df, neighborhood_df, rows, stability = \
    run_cohort_retrieval_with_neighborhood(members, topk=int(topk))
payload["neighborhood"] = _evidence_payload(
    neighborhood_df, label=cohort.label)

# Cohort B comparison
other_hits, other_neighborhood, other_rows, other_stability = \
    run_cohort_retrieval_with_neighborhood(list(other.members), topk=int(topk))
payload["comparison"]["neighborhood_b"] = _evidence_payload(
    other_neighborhood, label=other.label)
```

Only the requested `hits_df`/`other_hits` go to network drawing, NCBI
enrichment, overlap, stability panels, and AI input. The 250-row evidence frame
must not expand those existing behaviors.

- [ ] **Step 6: Run retrieval, cohort, upload, and callback tests**

```bash
rtk pytest tests/test_retrieval.py tests/test_upload_ingestion.py tests/test_cohorts.py tests/test_app.py -q
```

Expected: all pass.

- [ ] **Step 7: Commit the payload slice**

```bash
rtk git add bridge_rna/retrieval.py bridge_rna/callbacks.py tests/test_cohorts.py tests/test_app.py
rtk git diff --cached --check
rtk git commit -m "feat: carry evidence neighborhoods across views"
```

---

### Task 4: Uniform map evidence and focus traces

**Files:**
- Modify: `manifold/theme.py`
- Modify: `manifold/render.py`
- Modify: `manifold/layout.py`
- Modify: `manifold/callbacks.py`
- Modify: `tests/test_render.py`
- Modify: `tests/test_app.py`

**Interfaces:**
- Consumes: `hits-store.neighborhood`, optional `comparison.neighborhood_b`, active arm `"a" | "b"`, and focus `{kind: "study" | "sample", value: str} | None`.
- Produces: `_neighborhood_overlay(payload, arm, focus) -> dict | None`; `build_figure(method, dims, color_by, layers, budget, viewport, retrieval=None, neighborhood=None, found=None)`; a base evidence trace and optional outer focus trace.

- [ ] **Step 1: Write failing overlay normalization tests**

In `tests/test_app.py`, add:

```python
def _neighborhood_payload(label="A", offset=0):
    return {
        "available": True, "label": label, "depth_requested": 250,
        "depth_returned": 3, "metric": "cosine", "space": "embedding-512d",
        "hits": [
            {"rank": 1, "gsm": "GSM1", "gse": "GSE1", "tissue": "Eye",
             "species": "mouse", "score": .99, "archs4_index": offset + 1},
            {"rank": 2, "gsm": "GSM2", "gse": "GSE1", "tissue": "Eye",
             "species": "mouse", "score": .98, "archs4_index": offset + 2},
            {"rank": 3, "gsm": "GSM3", "gse": "GSE2", "tissue": "Brain",
             "species": "human", "score": .97, "archs4_index": None},
        ],
    }


def test_neighborhood_overlay_selects_an_arm_and_focus_points():
    payload = {"neighborhood": _neighborhood_payload("A"),
               "comparison": {"neighborhood_b": _neighborhood_payload("B", 10)}}
    overlay = callbacks._neighborhood_overlay(
        payload, "a", {"kind": "study", "value": "GSE1"})
    assert overlay["label"] == "A"
    assert overlay["points"] == [1, 2]
    assert overlay["focus_points"] == [1, 2]
    assert overlay["returned"] == 3 and overlay["locatable"] == 2
    assert callbacks._neighborhood_overlay(payload, "b", None)["points"] == [11, 12]


def test_unavailable_neighborhood_draws_nothing():
    assert callbacks._neighborhood_overlay(
        {"neighborhood": {"available": False, "hits": []}}, "a", None) is None
```

- [ ] **Step 2: Write failing render invariants**

In `tests/test_render.py`, create a 2-D/3-D parametrized test asserting:

```python
overlay = {
    "label": "OSD-100", "points": [1, 2, 3],
    "rows": [
        {"rank": 1, "gsm": "GSM1", "gse": "GSE1", "score": .99,
         "tissue": "Eye", "species": "mouse"},
        {"rank": 2, "gsm": "GSM2", "gse": "GSE1", "score": .98,
         "tissue": "Eye", "species": "mouse"},
        {"rank": 3, "gsm": "GSM3", "gse": "GSE2", "score": .97,
         "tissue": "Brain", "species": "human"},
    ],
    "focus_points": [2], "returned": 3, "locatable": 3,
}
base, focus = render._neighborhood_traces(coords, is_3d, overlay)
assert base.marker.size == theme.NEIGHBORHOOD_SIZE * (0.5 if is_3d else 1)
assert isinstance(base.marker.color, str)
assert "lines" not in (base.mode or "")
assert len(base.customdata) == 3
assert focus is not None and len(focus.x) == 1
```

Add a `build_figure` ordering assertion that the evidence trace precedes every
trace named `retrieved hit` and the outer focus trace follows it. Add an
out-of-range test showing invalid point indices are omitted without losing the
returned/locatable count.

- [ ] **Step 3: Run the focused tests and confirm missing helpers**

```bash
rtk pytest tests/test_render.py tests/test_app.py -q
```

Expected: failures name `_neighborhood_overlay`, `_neighborhood_traces`, and
the new `build_figure` parameter.

- [ ] **Step 4: Add theme constants and overlay normalization**

In `manifold/theme.py`, define one source of truth:

```python
NEIGHBORHOOD_COLOR = "#61ddd3"
NEIGHBORHOOD_SIZE = 7.0
NEIGHBORHOOD_OPACITY = 0.78
NEIGHBORHOOD_FOCUS_COLOR = "#d9791b"
NEIGHBORHOOD_FOCUS_SIZE = 15.0
NEIGHBORHOOD_FOCUS_LINE = 2.0
```

In `manifold/callbacks.py`, implement `_neighborhood_payload_for_arm` and
`_neighborhood_overlay`. Preserve row order, omit non-integer indices from map
arrays, retain `returned` from the payload, calculate `locatable`, and select
focus points by matching GSE for a study or GSM for a sample.

- [ ] **Step 5: Render evidence behind exact hits and focus around them**

Add `_neighborhood_traces(coords, is_3d, neighborhood)` using `go.Scattergl`
in 2-D and `go.Scatter3d` in 3-D. Its base trace uses `circle-open`, a scalar
color/size/opacity, and custom rows beginning with the literal
`"neighborhood"` so the existing OSDR click handler cannot mistake them for
sample keys. The hover template is:

```python
("<b>%{customdata[2]}</b> · %{customdata[3]}<br>"
 "512-d rank %{customdata[1]:,} · cosine %{customdata[4]:.4f}<br>"
 "%{customdata[5]} · %{customdata[6]}<extra></extra>")
```

Return `(base_trace, focus_trace_or_none)`. Extend `build_figure` to accept
`neighborhood=None`, add the base trace after corpus layers but before
`_retrieval_traces`, then add the larger open focus trace after retrieval so it
surrounds rather than replaces a white requested-hit ring. Add a badge such as
`Evidence neighborhood: <b>248</b> of 250 locatable` only when the counts differ;
otherwise use `Evidence neighborhood: <b>250</b>`.

- [ ] **Step 6: Add a keyed evidence-neighborhood row**

Extend `layout.retrieval_key_children` with an optional `neighborhood` argument
or add `layout.neighborhood_key_children`. The row uses a distinct open-dot CSS
shape, reads “512-D evidence neighbor,” and appears only while evidence marks
are drawn. Read its color from `theme.NEIGHBORHOOD_COLOR` inline, following the
existing no-mirrored-hues rule.

- [ ] **Step 7: Run map tests**

```bash
rtk pytest tests/test_render.py tests/test_app.py -q
```

Expected: all pass.

- [ ] **Step 8: Commit the map-rendering slice**

```bash
rtk git add manifold/theme.py manifold/render.py manifold/layout.py manifold/callbacks.py tests/test_render.py tests/test_app.py
rtk git diff --cached --check
rtk git commit -m "feat: draw exact evidence neighborhoods on the map"
```

---

### Task 5: Adaptive Overview / Studies / Samples drawer

**Files:**
- Modify: `manifold/layout.py`
- Modify: `manifold/callbacks.py`
- Modify: `assets/map.css`
- Modify: `tests/test_app.py`

**Interfaces:**
- Consumes: Task 1 aggregation functions and Task 4 overlay/focus shape.
- Produces: static ids `explore-neighborhood`, `neighborhood-drawer`, `neighborhood-close`, `neighborhood-heading`, `neighborhood-arm`, `neighborhood-tab`, `neighborhood-search`, `neighborhood-body`, `neighborhood-open-store`, `neighborhood-focus-store`, and `neighborhood-focus-sink`.

- [ ] **Step 1: Add failing static-layout and state-owner tests**

In `tests/test_app.py`, assert every id above is present in `layout.build_view()`.
Add callback-map assertions:

```python
def test_one_callback_owns_neighborhood_open_state(app):
    writers = [k for k in app.callback_map if "neighborhood-open-store.data" in k]
    assert len(writers) == 1
    inputs = {i["id"] for i in app.callback_map[writers[0]]["inputs"]}
    assert {"frame-retrieval", "explore-neighborhood", "neighborhood-close",
            "hits-store"} <= inputs


def test_explorer_does_not_make_the_viewport_multi_writer(app):
    writers = [k for k in app.callback_map if "viewport-store.data" in k]
    assert len(writers) == 1
```

Add pure callback tests for `next_neighborhood_open_state(trigger, previous,
has_retrieval)` and drawer builders for Overview, Studies, Samples, unavailable,
and comparison-arm states.

- [ ] **Step 2: Run focused app tests and confirm missing ids/functions**

```bash
rtk pytest tests/test_app.py -q
```

Expected: the new ids and state helper are absent.

- [ ] **Step 3: Build the static adaptive workspace and drawer**

Wrap the existing `bm-plot-wrap` and a new `html.Aside` in:

```python
html.Div(id="map-workspace", className="bm-map-workspace", children=[
    existing_plot_main,
    html.Aside(
        id="neighborhood-drawer",
        className="bm-neighborhood",
        style={"display": "none"},
        **{"aria-labelledby": "neighborhood-heading"},
        children=[
            html.Div(className="bm-neighborhood-head", children=[
                html.Div("Neighborhood explorer", className="bm-group-label"),
                html.Button("Close", id="neighborhood-close", n_clicks=0,
                            className="bm-neighborhood-close",
                            **{"aria-label": "Close neighborhood explorer"}),
                html.H2(id="neighborhood-heading", tabIndex=-1),
                html.Div(id="neighborhood-meta", className="bm-hint"),
                dcc.RadioItems(id="neighborhood-arm", value="a",
                               className="bm-seg bm-neighborhood-arm"),
                dcc.RadioItems(
                    id="neighborhood-tab", value="overview",
                    options=[{"label": "Overview", "value": "overview"},
                             {"label": "Studies", "value": "studies"},
                             {"label": "Samples", "value": "samples"}],
                    className="bm-seg bm-neighborhood-tabs"),
                html.Label("Search neighborhood samples", htmlFor="neighborhood-search",
                           className="visually-hidden"),
                dcc.Input(id="neighborhood-search", type="search",
                          placeholder="Search GSM, GSE, tissue…"),
            ]),
            html.Div(id="neighborhood-body", className="bm-neighborhood-body"),
            html.Div("Nearest in 512-D—not everything inside the visible frame.",
                     className="bm-neighborhood-foot"),
        ],
    ),
])
```

Add **Explore neighborhood** to the existing retrieval group and add the three
map-local stores after the existing stores. Keep components static so Dash can
validate every callback target at startup.

- [ ] **Step 4: Implement one owner for open/close state and accessible focus**

Use a pure helper:

```python
def next_neighborhood_open_state(trigger: str | None, previous: dict | None,
                                 has_retrieval: bool) -> dict:
    old = previous or {"open": False, "opener": "explore-neighborhood"}
    if trigger == "hits-store":
        return {"open": False, "opener": old["opener"]}
    if trigger == "neighborhood-close":
        return {"open": False, "opener": old["opener"]}
    if trigger in ("frame-retrieval", "explore-neighborhood") and has_retrieval:
        return {"open": True, "opener": trigger}
    return old
```

Register one server callback that feeds it from the three buttons and
`hits-store`. `has_retrieval` means that the active `hits-store` can produce the
existing retrieval overlay; it does not require `neighborhood.available`, so a
stale result can open the drawer and explain that it must be rerun. Extend the
existing retrieval-group visibility callback so **Explore neighborhood** is
hidden only when there is no retrieval and remains visible in 3-D. A second
read-only visibility callback sets the drawer style and `map-workspace` class.
Register a clientside callback to focus
`neighborhood-heading` on open and the stored opener on close, writing only to
`neighborhood-focus-sink.data`.

- [ ] **Step 5: Render header, arm options, tabs, and deterministic bodies**

Add four layout builders with these exact signatures:
`neighborhood_overview_children(summary: dict) -> list`,
`neighborhood_studies_children(groups: list[dict], focus: dict | None) -> list`,
`neighborhood_samples_children(rows: list[dict], focus: dict | None) -> list`,
and `neighborhood_unavailable_children(reason: str) -> list`.

Use this shared keyboard-native row shape for Studies and Samples:

```python
def _neighborhood_row(kind: str, value: str, primary: str,
                      secondary: str, trailing: str, selected: bool,
                      disabled: bool = False) -> html.Button:
    return html.Button(
        id={"type": f"neighborhood-{kind}", "value": value},
        n_clicks=0,
        disabled=disabled,
        className=("bm-neighborhood-row "
                   f"bm-neighborhood-{kind}"
                   + (" is-selected" if selected else "")),
        children=[
            html.Span(primary, className="bm-neighborhood-row-primary"),
            html.Span(secondary, className="bm-neighborhood-row-secondary"),
            html.Span(trailing, className="bm-neighborhood-row-trailing"),
        ],
    )
```

The Overview builder renders the summary sentence, coverage line, three metric
cells, tissue/species bars, and leading studies. The Studies builder emits
keyboard-native `html.Button` rows with ids
`{"type": "neighborhood-study", "value": display_gse}`. Wrap each study
button and its optional GEO `html.A` in a non-interactive row container; never
nest the link inside the button. Link real GSE accessions to
`https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=<GSE>`. The Samples builder emits every filtered row
as an `html.Button` with id
`{"type": "neighborhood-sample", "value": gsm}`. An unlocatable row can still
be selected to read its metadata; it simply produces no map focus point. When a
sample is selected, the Samples builder adds a compact detail block for that
GSM above the list and links it through the same GEO accession route. Each
builder has an explicit empty state.

One drawer callback consumes payload, arm, tab, search, and focus. Search is
shown only on Samples. Comparison arm options name the two cohort labels; the
single-arm control is hidden. Tabs include live counts: `Studies 41`,
`Samples 250`. A separate single-writer control callback resets tab to Overview,
arm to A, and search to empty whenever `hits-store` changes; the focus callback
in Step 6 resets focus to `None`. Ordinary open/close actions preserve the
active tab and search.

- [ ] **Step 6: Implement one owner for study/sample focus and wire the figure**

Use one callback with pattern-matching `ALL` inputs for study/sample row
buttons, `manifold-graph.clickData`, and `hits-store`. Reset on new retrieval;
otherwise accept only a changed click whose triggered id is one of the two
row families or a map customdata row beginning with `"neighborhood"`.
Study focus stores the displayed GSE value; sample focus stores its GSM, and a
map click reads the GSM from `customdata[2]`.

Extend the figure callback inputs with open state, active arm, and focus:

```python
state = open_state or {}
neighborhood = (_neighborhood_overlay(hits_payload, arm or "a", focus)
                if state.get("open") else None)
fig, legend_data, badges = render.build_figure(
    method, dims, color_by, layers or [], budget,
    vp if dims == "2d" else None,
    retrieval=retrieval, neighborhood=neighborhood, found=found)
```

Changing tabs or filtering the list must not be a figure input. Selecting a
study/sample is the only drawer content action that redraws the map.

- [ ] **Step 7: Style the desktop drawer and mobile bottom sheet**

In `assets/map.css` add:

```css
.bm-map-workspace { display: flex; flex: 1; min-width: 0; min-height: 0; }
.bm-map-workspace > .bm-plot-wrap { min-width: 0; }
.bm-neighborhood {
  width: 390px; min-width: 0; flex: none; display: flex; flex-direction: column;
  background: var(--bg-panel); border-left: var(--border); color: var(--text-primary);
}
.bm-neighborhood-body { flex: 1; min-height: 0; overflow-y: auto; padding: 0 14px 14px; }
.bm-neighborhood-row { width: 100%; border: 0; border-top: var(--border); background: transparent; text-align: left; }
.bm-neighborhood-row:hover,
.bm-neighborhood-row:focus-visible,
.bm-neighborhood-row.is-selected { background: var(--accent-soft); }
@media (max-width: 900px) {
  .bm-map-workspace { position: relative; flex: none; height: 68vh; min-height: 420px; }
  .bm-map-workspace > .bm-plot-wrap { height: 100%; min-height: 0; }
  .bm-neighborhood {
    position: absolute; z-index: 12; right: 0; bottom: 0; left: 0;
    width: auto; max-height: min(72%, 560px); border-left: 0; border-top: var(--border);
  }
}
```

Complete the header, tabs, coverage bars, study/sample grid, focus, footer, and
empty-state rules using only existing tokens and the 4/8/12/20/32 spacing
scale. Prevent overflow with `minmax(0, 1fr)`, `min-width: 0`, and wrapped or
ellipsized accession/title cells.

- [ ] **Step 8: Run unit and callback tests**

```bash
rtk pytest tests/test_neighborhoods.py tests/test_render.py tests/test_app.py -q
```

Expected: all pass.

- [ ] **Step 9: Commit the drawer slice**

```bash
rtk git add manifold/layout.py manifold/callbacks.py assets/map.css tests/test_app.py
rtk git diff --cached --check
rtk git commit -m "feat: add the map neighborhood explorer"
```

---

### Task 6: Browser verification, documentation, push, and local handoff

**Files:**
- Modify: `tests/e2e_check.py`
- Modify: `README.md`
- Modify: `docs/design-notes.md`
- Modify: `progress.md`
- Modify only if the real screenshot materially changes: `docs/bridge-rna-map.png`

**Interfaces:**
- Consumes: the complete feature from Tasks 1–5.
- Produces: repeatable browser coverage, updated user-facing explanation, pushed `origin/main`, and a tracked local server process.

- [ ] **Step 1: Add browser assertions before changing documentation**

Extend `tests/e2e_check.py` after its existing retrieval/frame checks. Reuse its
real OSDR query and check:

```python
page.click("#explore-neighborhood")
c.ok(page.locator("#neighborhood-drawer").is_visible(),
     "the evidence drawer opens from the map")
c.ok("250 nearest" in page.locator("#neighborhood-meta").inner_text(),
     "the drawer names its exact evidence depth")
evidence_traces = page.evaluate("""() => {
  const plot = document.querySelector('#manifold-graph .js-plotly-plot');
  return (plot && plot.data || []).filter(t => t.name === '512-D evidence neighbor').length;
}""")
c.ok(evidence_traces == 1, "the exact evidence neighborhood is drawn once")
page.get_by_text(re.compile(r"Studies\s+\d+")).click()
c.ok(page.locator(".bm-neighborhood-study").count() > 0,
     "every represented study is available")
page.get_by_text(re.compile(r"Samples\s+250")).click()
page.fill("#neighborhood-search", "GSM")
c.ok(page.locator(".bm-neighborhood-sample").count() > 0,
     "the complete sample list is searchable")
```

Also check: selecting a study produces a focus trace without rerunning search;
Close hides drawer/evidence but not white requested hits; Frame reopens in 2-D;
Explore remains visible and opens in 3-D; comparison arm switching changes the
drawer label; at 768 and 320 px the drawer is a bottom sheet within viewport;
Tab reaches close/tabs/search/rows; Escape is not claimed unless implemented;
browser console contains no errors.

- [ ] **Step 2: Run the browser suite and fix only reproduced failures**

```bash
rtk test .venv/bin/python tests/e2e_check.py
```

Expected: every existing and new browser check passes. If the harness requires
the app separately, start it with `.venv/bin/python app.py --port 8050` in a
tracked foreground execution session, then rerun the suite.

- [ ] **Step 3: Update the user and design documentation**

Add a concise README paragraph under the map section explaining that a
retrieval can open a top-250 exact cosine neighborhood with deterministic
tissue/study summaries and complete GSE/GSM lists. State explicitly that teal
evidence marks are nearest in 512-D, not everything inside the framed
projection.

Add a design-notes section recording the fixed depth, one-scan prefix
invariant, uniform mark decision, coverage denominators, comparison-arm
behavior, and why viewport census/hulls/AI copy were rejected. Append a dated
`progress.md` entry with changed files, test counts, browser observations, and
any measured first-open/render timing.

- [ ] **Step 4: Run the complete automated verification**

```bash
rtk pytest tests/ -q
rtk test .venv/bin/python tests/e2e_check.py
rtk git diff --check
```

Expected: all tests and browser checks pass, with no whitespace errors.

- [ ] **Step 5: Perform runtime accessibility and responsive checks**

Using the locally running app and Chrome DevTools:

- inspect console and network for errors;
- tab through Explore, Close, all tabs, search, a study row, and a sample row;
- inspect accessible names/roles for the drawer, tabs, search, close, and rows;
- capture 320, 768, 1024, and 1440 px screenshots;
- verify no horizontal overflow, clipped controls, obscured Plotly modebar, or
  unreadable drawer content;
- verify opening/closing and row focus do not initiate another retrieval or
  ARCHS4 memmap scan.

- [ ] **Step 6: Run the mandated review and simplification gates**

Use `agent-skills:code-review-and-quality`, then
`agent-skills:code-simplification`, then
`superpowers:verification-before-completion`. Apply only findings grounded in
the approved spec, rerunning the affected focused tests after each correction.

- [ ] **Step 7: Commit documentation and final verification record**

```bash
rtk git add tests/e2e_check.py README.md docs/design-notes.md progress.md
rtk git add docs/bridge-rna-map.png  # only when regenerated intentionally
rtk git diff --cached --check
rtk git commit -m "docs: explain exact retrieval neighborhoods"
```

- [ ] **Step 8: Confirm commit scope and push main**

```bash
rtk git status --short
rtk git log --oneline origin/main..main
rtk git diff --stat origin/main..main
rtk git push origin main
```

Confirm that `.env.example`, `app.py`, `requirements.txt`, `wsgi.py`, `.lavish/`,
and `.superpowers/` were not included unless separately authorized. Do not
force-push.

- [ ] **Step 9: Leave the verified app running locally**

Start the app in a tracked foreground execution session on the configured
loopback port. Verify the health response and open the exact local URL with the
browser controller. Report the URL, PID/session ownership, pushed commit, test
totals, and any intentionally uncommitted pre-existing files to the user.
