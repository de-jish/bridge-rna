"""The color-by registry: one place that knows what can be colored, and how much.

Before this module the renderer branched on the color-by key with a chain of
if/elif, and every branch decided for itself what to do about the corpus it did
not describe. The result was the failure this redesign exists to remove: pick
any OSDR field and 940,455 of 942,563 points - 99.8% of the map - became one
flat grey cloud, indistinguishable from "ARCHS4 has no structure here".

The fix is to make coverage a first-class, declared property. Every color-by is
a `ColorBy` that reports which corpora it can actually color *right now*, given
which artifacts exist on this machine. That single fact drives everything
downstream:

  * the menu groups whole-map fields above spaceflight-only ones,
  * an option with no data at all is offered but disabled, with the command to
    run attached, instead of silently missing or silently grey,
  * the control rail states the exact point count a field colors, and
  * the renderer never paints a uniform grey glyph cloud: a corpus a field does
    not describe is drawn as *context*, in one faint colour with no legend row,
    which shows the real manifold shape instead of impersonating data.

`labels()` returns one array over the full corpus in fixed global order, with
`NOT_COVERED` marking points the field does not describe. Everything else is a
plain categorical render.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import data, tissue

# Sentinel for a point this field says nothing about. Never rendered as a
# category and never given a legend row - it is the absence of a value, not a
# value, and giving it a swatch is what made the map read as grey data.
NOT_COVERED = "\x00not-covered"

# Categories that are real answers but carry no information, so they must never
# take a bright palette slot ahead of a category that does.
RESIDUAL = (tissue.OTHER, tissue.UNKNOWN, "Unknown", "unknown", "nan", "None", "")

ARCHS4, OSDR = "archs4", "osdr"


@dataclass(frozen=True)
class ColorBy:
    key: str
    label: str
    # Which corpora this field could describe if every artifact were present.
    scope: tuple[str, ...]
    # Per-point label resolver, returning an array over the full corpus.
    resolver: Callable[[], np.ndarray]
    hint: str = ""
    # (predicate, what to run when it returns False). The predicate is the same
    # function the data layer gates its loader on, deliberately: a registry that
    # decided availability by re-deriving the artifact path would be a second
    # source of truth, and the two could disagree about whether a field works.
    needs: tuple[Callable[[], bool], str] | None = None

    def covers(self) -> tuple[str, ...]:
        """Corpora this field can colour on this machine, right now."""
        if self.needs and not self.needs[0]():
            # The missing artifact only ever gates ARCHS4; OSDR metadata is
            # built by the same job as the embeddings and is never absent alone.
            return tuple(c for c in self.scope if c != ARCHS4)
        return self.scope

    def available(self) -> bool:
        return bool(self.covers())

    def missing_hint(self) -> str:
        return "" if self.needs is None or ARCHS4 in self.covers() else self.needs[1]


# --- Resolvers --------------------------------------------------------------
# Each returns one array over the full corpus in the fixed global order
# [ARCHS4 0..n_archs4-1, then OSDR]. Cheap enough to call per figure build;
# the underlying parquet loads are cached in data.py.

def _species() -> np.ndarray:
    sid = data.points_meta()["species_id"].to_numpy()
    return np.where(sid == 0, "Human", "Mouse")


def _tissue() -> np.ndarray:
    n_archs4, n_osdr, total = data.counts()
    out = np.full(total, NOT_COVERED, dtype=object)
    archs4 = data.archs4_tissue()
    if archs4 is not None:
        out[:n_archs4] = archs4
    out[n_archs4:] = data.osdr_tissue()
    return out


def _osdr_field(name: str) -> Callable[[], np.ndarray]:
    def resolve() -> np.ndarray:
        n_archs4, _, total = data.counts()
        out = np.full(total, NOT_COVERED, dtype=object)
        out[n_archs4:] = data.osdr_field_values(name).to_numpy()
        return out
    return resolve


_ARCHS4_META_NEEDS = (
    data.archs4_metadata_available,
    "ARCHS4 labels need the GEO metadata join - run "
    "precompute/fetch_archs4_meta.py (about 35 seconds, needs network).",
)

REGISTRY: tuple[ColorBy, ...] = (
    # --- Whole map: every point gets a real category. -----------------------
    ColorBy(
        key="tissue", label="Tissue", scope=(ARCHS4, OSDR), resolver=_tissue,
        needs=_ARCHS4_META_NEEDS,
        hint="Both corpora folded onto one anatomical vocabulary, so a liver in "
             "GEO and a NASA liver share a colour.",
    ),
    ColorBy(
        key="species", label="Species", scope=(ARCHS4, OSDR), resolver=_species,
        hint="The cleanest partition on the map, and the reference for what a "
             "working colour-by looks like. OSDR is entirely mouse.",
    ),
    # There is deliberately no unsupervised-cluster colour-by here. A k-means
    # partition of the same 512-d vectors the projection was fit on was built,
    # measured, and cut: 81.9% of its labels are recoverable from the 2-D
    # coordinates alone, so it mostly redraws the shape already on screen, and
    # a structure-free Voronoi null reproduced its spatial coherence to within
    # 1.5 points. It is also arbitrary - seed-to-seed agreement is only
    # ARI ~0.45 - and 81% species-pure. Painting an arbitrary partition on a
    # scientific instrument and numbering it "Cluster 1..24" invites exactly the
    # over-reading the rest of this file exists to prevent. See IMPLEMENTATION.md.
    # --- Spaceflight detail: defined for the 2,108 OSDR samples only. -------
    ColorBy(key="flight_status", label="Flight vs Ground", scope=(OSDR,),
            resolver=_osdr_field("flight_status"),
            hint="The one contrast the OSDR corpus is built around."),
    ColorBy(key="spaceflight", label="Spaceflight arm", scope=(OSDR,),
            resolver=_osdr_field("spaceflight"),
            hint="The seven raw control arms kept distinct - a basal animal and "
                 "a vivarium animal are different experiments."),
    ColorBy(key="strain", label="Strain", scope=(OSDR,), resolver=_osdr_field("strain")),
    ColorBy(key="sex", label="Sex", scope=(OSDR,), resolver=_osdr_field("sex")),
    ColorBy(key="genotype", label="Genotype", scope=(OSDR,), resolver=_osdr_field("genotype")),
    ColorBy(key="study", label="Study", scope=(OSDR,), resolver=_osdr_field("study"),
            hint="Each OSD study is one batch. Colour by this to see how much "
                 "apparent structure is study rather than biology."),
    ColorBy(key="habitat", label="Habitat", scope=(OSDR,), resolver=_osdr_field("habitat")),
    ColorBy(key="duration", label="Mission duration", scope=(OSDR,),
            resolver=_osdr_field("duration")),
    ColorBy(key="diet", label="Diet", scope=(OSDR,), resolver=_osdr_field("diet")),
)

_BY_KEY = {c.key: c for c in REGISTRY}

GROUP_WHOLE_MAP = "Whole map"
GROUP_OSDR = "Spaceflight detail (OSDR only)"


def get(key: str) -> ColorBy:
    """The spec for a key, falling back to the default rather than raising.

    A stale key can reach here from a browser that kept its old selection across
    a rebuild, and taking the app down for that would be a poor trade.
    """
    return _BY_KEY.get(key) or _BY_KEY[default_key()]


def default_key() -> str:
    """The best colour-by to open on: the first whole-map field that works."""
    for spec in REGISTRY:
        if len(spec.covers()) > 1:
            return spec.key
    return "species"


def group_of(spec: ColorBy) -> str:
    return GROUP_WHOLE_MAP if len(spec.covers()) > 1 else GROUP_OSDR


def labels(key: str) -> np.ndarray:
    """Per-point categories over the full corpus, NOT_COVERED where undefined."""
    return get(key).resolver()


def coverage(key: str) -> tuple[int, int]:
    """(points this field colours, total points)."""
    spec = get(key)
    n_archs4, n_osdr, total = data.counts()
    covered = 0
    if ARCHS4 in spec.covers():
        covered += n_archs4
    if OSDR in spec.covers():
        covered += n_osdr
    return covered, total


def covers_corpus(key: str, corpus: str) -> bool:
    return corpus in get(key).covers()


def scope_note(spec: ColorBy) -> str:
    """The short suffix that tells a user what a field will paint before they pick it."""
    if not spec.available():
        return "unavailable"
    return "whole map" if len(spec.covers()) > 1 else "OSDR only"


def menu_options() -> list[dict]:
    """Dropdown options, whole-map fields first, each annotated with its scope.

    Dash's dropdown has no option-group support, so the grouping is carried by
    ordering plus the scope suffix rather than by faked disabled header rows.
    The exact point count lives in the coverage line under the control, where it
    has room to be precise.

    An option that cannot colour anything is shown and disabled rather than
    hidden. Hiding it makes the app look like it never had the feature; showing
    it disabled, next to the command that enables it, says the feature exists
    and how to switch it on.
    """
    ordered = sorted(REGISTRY, key=lambda s: (group_of(s) != GROUP_WHOLE_MAP,))
    return [
        {
            "label": f"{spec.label}  ·  {scope_note(spec)}",
            "value": spec.key,
            "disabled": not spec.available(),
        }
        for spec in ordered
    ]


def is_residual(category: str) -> bool:
    """True for categories that must not claim a bright palette slot."""
    return category in RESIDUAL or category == NOT_COVERED
