"""What a cohort is, and the geometry of pooling one.

A spaceflight study does not have one sample; it has an experimental group.
This module is the only place that knows how those groups are defined, how a
group becomes one query vector, and how much to trust the result.

It deliberately depends on **no embedding index and no memmap**. Cohort
membership is metadata grouping, and the estimator is 512-d arithmetic over
vectors the caller supplies, so everything here is testable against the fixture
corpus on a machine that has neither the 963 MB ARCHS4 memmap nor the model
checkpoint. `retrieval.run_cohort_retrieval` is the seam where the two meet.

The evidence behind every constant and every design decision is in
`docs/cohort_pooling.md` (measured) and `docs/cohort_retrieval.md` (built).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from .util import _safe_str


# --- The facets a cohort can be defined by -----------------------------------


@dataclass(frozen=True)
class Facet:
    """One column a cohort can be grouped by."""

    key: str            # column in cache/osdr_metadata.parquet
    label: str          # what the chip says
    default: bool       # ticked when the view opens
    pinned: bool = False  # cannot be unticked, and why
    reason: str = ""      # the hover text explaining a pinned facet


# Order is the order the chips appear in. Study first because it is pinned,
# then the two that complete OSDR's own curated ISA-Tab factor grouping.
#
# These three are the whole registry, and that is a deliberate narrowing. An
# earlier build offered six more columns from the same table - sex, strain,
# genotype, habitat, mission duration, diet - and every one of them made the
# cohort *smaller*. Size is the thing the measured stability curve is a
# function of (0.34 at k=2 against 0.86 at k>=15), so a finer definition trades
# away the exact quantity the feature exists to buy, and it does so for a
# contrast the two-arm comparison already answers more directly. The six are
# recorded in `docs/cohort_retrieval.md` rather than left on the rail; adding
# one back is one line here, and nothing else in the app hard-codes a facet.
FACETS: tuple[Facet, ...] = (
    Facet("study", "Study", default=True, pinned=True,
          reason="Pooling across studies would average across the strongest "
                 "batch boundary in the corpus. Random samples from one study "
                 "already reach 0.9805 mean pairwise cosine against 0.9933 for "
                 "a real cohort, so same-study membership supplies 84% of a "
                 "cohort's coherence on its own."),
    Facet("tissue", "Tissue", default=True),
    Facet("spaceflight", "Spaceflight arm", default=True),
)

FACETS_BY_KEY: dict[str, Facet] = {f.key: f for f in FACETS}
PINNED_FACETS: tuple[str, ...] = tuple(f.key for f in FACETS if f.pinned)
DEFAULT_FACETS: tuple[str, ...] = tuple(f.key for f in FACETS if f.default)


def normalize_facets(keys: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    """The facet set actually used, in registry order, with the pinned ones in.

    Callers get their selection from a checklist, which can arrive empty, in an
    arbitrary order, or carrying a key that no longer exists. All three would
    otherwise change what a cohort *is* without saying so, so the set is
    canonicalized here rather than trusted.
    """
    chosen = {k for k in (keys or ()) if k in FACETS_BY_KEY}
    chosen.update(PINNED_FACETS)
    return tuple(f.key for f in FACETS if f.key in chosen)


# --- Low N ------------------------------------------------------------------

# Two samples is the smallest thing that can be pooled at all; one is a sample
# wearing a cohort's name, and it is offered by the Sample mode already.
MIN_COHORT_SIZE = 2

# Set by measurement, not by taste. `precompute/validate_cohorts.py` scores
# leave-one-out top-5 stability against cohort size over all 212 cohorts on the
# real corpus, and k >= 5 is the first size bucket to reach 0.70. Below it a
# cohort is still searchable and still far better than one sample - it is
# flagged, never hidden or blocked.
#
# Measured 2026-08-05 over all 212 cohorts. Rerun with:
#   .venv/bin/python precompute/validate_cohorts.py --stability
LOW_N_THRESHOLD = 5

TIER_SINGLETON = "singleton"    # k < MIN_COHORT_SIZE: nothing to pool
TIER_LOW_N = "low_n"            # pooled, but below the measured stability knee
TIER_OK = "ok"


def size_tier(k: int) -> str:
    if k < MIN_COHORT_SIZE:
        return TIER_SINGLETON
    return TIER_LOW_N if k < LOW_N_THRESHOLD else TIER_OK


# Measured leave-one-out top-5 agreement of a pooled query, by cohort size, on
# the real corpus. This is what the confidence readout quotes, so a researcher
# sees how far to trust a list rather than the word "low".
#
# Keys are bucket *floors*: a size reads the largest key at or below it, so 7
# reads the 5-9 bucket. Buckets rather than exact sizes because the per-k
# figure is not stable enough to quote - the first sweep drew two cohorts per
# size and produced 0.38 at k=5 beside 0.90 at k=6, which is an artifact of
# which two were drawn. Each bucket below stands on at least 5 cohorts and the
# 5-9 entry is two adjacent buckets merged, because they inverted by 0.04
# against a within-bucket sd of 0.18 and a larger cohort must never be reported
# as less trustworthy than a smaller one.
#
# Measured 2026-08-05 over all 212 cohorts. Regenerate with:
#   .venv/bin/python precompute/validate_cohorts.py --stability
# which prints the dict in exactly this form.
STABILITY_BY_K: dict[int, float] = {
    2: 0.34,
    3: 0.51,
    4: 0.55,
    5: 0.72,
    10: 0.81,
    15: 0.86,
}

# The same statistic for a single sample, which is what pooling is measured
# against: two replicates of one cohort share 0.161 of their top-5 hits, so
# pooling is a 4.6x gain at the depth people actually read.
SINGLE_SAMPLE_STABILITY = 0.16


def expected_stability(k: int) -> float:
    """Measured top-5 leave-one-out agreement for a pooled query of size k."""
    if k < MIN_COHORT_SIZE:
        return SINGLE_SAMPLE_STABILITY
    applicable = [size for size in STABILITY_BY_K if size <= k]
    return STABILITY_BY_K[max(applicable)] if applicable else SINGLE_SAMPLE_STABILITY


# --- The estimator ----------------------------------------------------------


def cohort_query_vector(rows: np.ndarray) -> np.ndarray:
    """Spherical (von Mises-Fisher) mean of a cohort's cached embeddings.

    Each member is L2-normalized *before* averaging, so the pooled ranking is an
    unweighted mean of the members' own cosine scores: one animal, one vote.

    Averaging the raw vectors instead would weight each member by its L2 norm,
    and invariant 2 establishes that the norm is a transcriptome-*concentration*
    axis spanning 3.9x across the corpus rather than a nuisance scale. The most
    concentrated transcriptomes would therefore cast the loudest votes, for no
    stated reason. Within a real cohort the spread is only 1.09x and the two
    estimators agree to a median cosine of 0.9999994, so this changes almost
    nothing today; it is done because it is the maximum-likelihood estimate of
    a vMF mean direction, and because it stays correct if anyone unticks Tissue
    and pools across organs, where the 3.9x spread is real.
    """
    arr = np.asarray(rows, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0:
        raise ValueError("cohort_query_vector needs a (k, d) array with k >= 1.")
    u = _unit_rows(arr)
    m = u.mean(axis=0)
    norm = float(np.linalg.norm(m))
    if norm <= 0.0:
        # k vectors that cancel exactly. Physically impossible for real
        # embeddings, whose pairwise cosine within a cohort is 0.9933, but a
        # zero query vector would score every ARCHS4 sample identically and the
        # result would look like a ranking. Refuse instead.
        raise ValueError(
            "The pooled vector is zero: these members cancel out and have no "
            "mean direction.")
    return (m / norm).astype(np.float32)


def _unit_rows(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, 1e-12)


# A group-level tightness statistic used to live here: `R̄`, the vMF resultant
# length `|mean(unit vectors)|`, in [0, 1]. It was measured over all 212 real
# cohorts and it is essentially constant - median 0.9991, and no lower for a
# cohort of two than for one of thirty - so it never separated a group worth
# trusting from one that was not, while sitting on the card looking like a
# grade. It is deleted rather than hidden; `docs/cohort_pooling.md` keeps the
# measurement. What replaced it is nothing, because the honest confidence
# number was already beside it: `expected_stability(k)`.
#
# The per-member statistic below is deliberately *not* the same kind of thing
# and stays. It varies within a cohort and names an individual animal, which is
# something a user can act on.


def leave_one_out_cosines(rows: np.ndarray) -> np.ndarray:
    """Each member's cosine to the centroid of *the others*.

    The leave-one-out reference is the point. Scoring a member against the full
    centroid lets an outlier pull the reference towards itself and hide inside
    it, which is exactly the sample the statistic exists to surface.

    A two-member cohort degenerates to "each member's cosine to the other",
    which is the same number twice, and that is the correct answer rather than a
    special case: with two samples there is no majority to deviate from.
    """
    arr = np.asarray(rows, dtype=np.float32)
    k = arr.shape[0]
    if k < 2:
        return np.ones(k, dtype=np.float32)
    u = _unit_rows(arr)
    total = u.sum(axis=0)
    others = (total - u) / (k - 1)
    others = others / np.maximum(np.linalg.norm(others, axis=1, keepdims=True), 1e-12)
    return np.einsum("ij,ij->i", u, others).astype(np.float32)


# How far below its cohort a member has to sit before it is called out. Members
# of one cohort agree at a median pairwise cosine of 0.998, so a fixed absolute
# cutoff would either flag everything or nothing depending on the study. This
# is relative to the cohort's own spread: more than three median-absolute
# deviations below the median leave-one-out cosine, with a floor so a cohort
# that happens to be extraordinarily tight does not flag an ordinary member.
OUTLIER_MAD_MULTIPLE = 3.0
OUTLIER_MIN_GAP = 0.002


def outlier_flags(loo: np.ndarray) -> np.ndarray:
    """Which members sit far enough below the rest to be worth a second look.

    Flagged, never dropped. The app shows the member with its number and lets
    the user exclude it, which is the same treatment the sample picker gives an
    unretrievable sample: shown, with the reason, rather than silently hidden.
    """
    arr = np.asarray(loo, dtype=np.float64).reshape(-1)
    if arr.size < 3:
        # With two members the statistic is one number twice, so "below the
        # others" has no meaning and any flag would be an artifact.
        return np.zeros(arr.size, dtype=bool)
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    gap = max(OUTLIER_MAD_MULTIPLE * mad, OUTLIER_MIN_GAP)
    return arr < (median - gap)


# --- Cohorts over the OSDR metadata -----------------------------------------


UNKNOWN_VALUES = {"", "nan", "none", "na", "n/a", "unknown", "<na>"}


def _tidy(value: Any) -> str:
    """Unwrap ISA-Tab unit annotations, e.g. "37 {day}" -> "37 day"."""
    return re.sub(r"\s*\{([^}]*)\}", r" \1", _safe_str(value)).strip()


def facet_value(row: pd.Series, key: str) -> str:
    v = _tidy(row.get(key))
    return "Unknown" if v.lower() in UNKNOWN_VALUES else v


@dataclass(frozen=True)
class Cohort:
    """One experimental group, and everything the interface says about it."""

    cohort_id: str            # stable, encodes the facet values that define it
    study: str
    facets: tuple[str, ...]   # the definition that produced it
    values: dict[str, str]    # facet key -> value
    members: tuple[str, ...]  # sample_key / sample_id, in metadata order

    @property
    def size(self) -> int:
        return len(self.members)

    @property
    def tier(self) -> str:
        return size_tier(self.size)

    @property
    def label(self) -> str:
        """Human name: the facet values that are not the study, joined.

        The study is dropped because a cohort is only ever listed inside its own
        study's picker, so repeating it in every row is noise. When the study is
        the *only* facet, the whole study is the cohort and it says so.
        """
        parts = [self.values[k] for k in self.facets if k != "study"]
        return " · ".join(parts) if parts else "Whole study"

    def describe(self) -> str:
        """The one-line sentence the status banner and the query node use."""
        return f"{self.study} · {self.label}"


def _cohort_id(study: str, facets: tuple[str, ...], values: dict[str, str]) -> str:
    return "COHORT|" + "|".join(f"{k}={values[k]}" for k in facets)


@lru_cache(maxsize=1)
def cohort_metadata() -> pd.DataFrame:
    """The 2,108 OSDR samples that have a cached embedding, with their facets.

    This is `cache/osdr_metadata.parquet` rather than the OSDR TSV the sample
    picker reads, and the difference is deliberate. Only a sample with a cached
    vector can be pooled at all, and this table is exactly that set, so a cohort
    can never list a member it cannot pool. Its `sample_key` is the same string
    the retrieval calls `sample_id`, which is the join the whole app is built
    on. It also carries every column the map's color-by registry resolves, so a
    facet can be added back here without a new artifact.

    Returns an empty frame rather than raising when the cache is absent, which
    is the state a fresh clone starts in. Cohort mode then reports that it needs
    the cache instead of failing.
    """
    try:
        from manifold import data as mdata

        df = mdata.osdr_metadata()
    except Exception:
        return pd.DataFrame(columns=["sample_key", *FACETS_BY_KEY])
    if df is None or df.empty:
        return pd.DataFrame(columns=["sample_key", *FACETS_BY_KEY])
    return df.reset_index(drop=True)


def cohorts_available() -> bool:
    """Whether cohort mode can offer anything on this machine."""
    df = cohort_metadata()
    return not df.empty and "sample_key" in df.columns


def build_cohorts(facets: tuple[str, ...] | list[str] | None = None,
                  study: str | None = None,
                  metadata: pd.DataFrame | None = None) -> list[Cohort]:
    """Every cohort the given definition produces, largest first.

    Restricting to one `study` is a filter on the output rather than a different
    grouping: study is a pinned facet, so a cohort never spans two studies and
    filtering afterwards cannot merge or split anything.
    """
    keys = normalize_facets(facets if facets is not None else DEFAULT_FACETS)
    df = cohort_metadata() if metadata is None else metadata
    if df.empty or "sample_key" not in df.columns:
        return []

    missing = [k for k in keys if k not in df.columns]
    if missing:
        # A facet the shipped metadata does not carry would silently group every
        # sample under one value and quietly widen every cohort.
        keys = tuple(k for k in keys if k in df.columns)
        if not keys:
            return []

    if study is not None:
        df = df[df["study"].astype(str) == _safe_str(study)]
        if df.empty:
            return []

    buckets: dict[tuple[str, ...], list[str]] = {}
    bucket_values: dict[tuple[str, ...], dict[str, str]] = {}
    for _, row in df.iterrows():
        values = {k: facet_value(row, k) for k in keys}
        signature = tuple(values[k] for k in keys)
        buckets.setdefault(signature, []).append(_safe_str(row["sample_key"]))
        bucket_values.setdefault(signature, values)

    out = [
        Cohort(
            cohort_id=_cohort_id(bucket_values[sig].get("study", ""), keys,
                                 bucket_values[sig]),
            study=bucket_values[sig].get("study", ""),
            facets=keys,
            values=bucket_values[sig],
            members=tuple(members),
        )
        for sig, members in buckets.items()
    ]
    # Largest first, then alphabetically, so the picker opens on the cohort with
    # the most evidence behind it rather than on whichever row came first.
    out.sort(key=lambda c: (-c.size, c.label))
    return out


def find_cohort(cohort_id: str, facets: tuple[str, ...] | list[str] | None = None,
                metadata: pd.DataFrame | None = None) -> Cohort | None:
    """Re-resolve a cohort from its id, since the id encodes its definition.

    The store round-trips an id rather than a member list, so a cohort selected
    before a facet change cannot be searched under a definition that no longer
    produces it.
    """
    wanted = _safe_str(cohort_id)
    if not wanted:
        return None
    for c in build_cohorts(facets=facets, metadata=metadata):
        if c.cohort_id == wanted:
            return c
    return None


def sibling_cohorts(cohort: Cohort, metadata: pd.DataFrame | None = None) -> list[Cohort]:
    """The cohorts this one can be compared against: same study, one facet apart.

    "One facet apart" is what makes a comparison interpretable. The flight and
    ground arms of one study's liver differ in exactly the arm, so the overlap
    between their two hit sets is about the arm. Two cohorts differing in both
    tissue and arm would produce a number nobody can attribute.
    """
    others = build_cohorts(facets=cohort.facets, study=cohort.study,
                           metadata=metadata)
    out = []
    for other in others:
        if other.cohort_id == cohort.cohort_id or other.size < MIN_COHORT_SIZE:
            continue
        differing = [k for k in cohort.facets
                     if other.values.get(k) != cohort.values.get(k)]
        if len(differing) == 1:
            out.append(other)
    return out


def contrast_facet(a: Cohort, b: Cohort) -> str:
    """The one facet two sibling cohorts differ in, for labelling a comparison."""
    differing = [k for k in a.facets if a.values.get(k) != b.values.get(k)]
    return differing[0] if len(differing) == 1 else ""


# --- What the interface needs to know about a selected cohort ----------------


@dataclass(frozen=True)
class CohortGeometry:
    """The measured facts about a pooled cohort, ready to render."""

    members: tuple[str, ...]
    loo_cosines: tuple[float, ...]
    outliers: tuple[bool, ...]

    @property
    def size(self) -> int:
        return len(self.members)

    @property
    def tier(self) -> str:
        return size_tier(self.size)

    @property
    def stability(self) -> float:
        return expected_stability(self.size)


def cohort_geometry(members: list[str] | tuple[str, ...],
                    vectors: np.ndarray) -> CohortGeometry:
    """Bundle the per-cohort statistics from its members' cached vectors."""
    loo = leave_one_out_cosines(vectors)
    return CohortGeometry(
        members=tuple(members),
        loo_cosines=tuple(float(x) for x in loo),
        outliers=tuple(bool(x) for x in outlier_flags(loo)),
    )
