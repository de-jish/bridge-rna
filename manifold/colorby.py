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

  * the menu groups whole-map fields above OSDR-only ones,
  * an option with no data at all is offered but disabled, with the command to
    run attached, instead of silently missing or silently grey,
  * the control rail states the exact point count a field colors, and
  * the renderer never paints a uniform grey glyph cloud: a corpus a field does
    not describe is drawn as *context*, in one faint color with no legend row,
    which shows the real manifold shape instead of impersonating data.

`labels()` returns one array over the full corpus in fixed global order, with
`NOT_COVERED` marking points the field does not describe. Everything else is a
plain categorical render.

**Two fields are registered, and the partial-coverage machinery is still the
point.** The nine OSDR-only fields this module was written around are gone
(see the note in `REGISTRY`), so with `cache/archs4_metadata.parquet` present
both Tissue and Species color all 942,563 points and none of the degraded paths
run. Without it, Tissue drops to the 2,108 OSDR points and every one of them
does: the menu labels it "OSDR only", the coverage bar goes amber and states the
count, the fix is named under the control, and the renderer draws ARCHS4 as a
faint context cloud rather than as a grey category. That is invariant 5, it is
the state a fresh clone starts in, and it is what these declarations are for.
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
        """Corpora this field can color on this machine, right now."""
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


#: The one sentence that says how to get the optional GEO join, so the two
#: controls that depend on it - the color-by and the find box - cannot end up
#: telling a user to run two different things.
ARCHS4_META_HINT = ("ARCHS4 labels need the GEO metadata join - run "
                    "precompute/fetch_archs4_meta.py (about 35 seconds, "
                    "needs network).")

_ARCHS4_META_NEEDS = (data.archs4_metadata_available, ARCHS4_META_HINT)

REGISTRY: tuple[ColorBy, ...] = (
    # --- Whole map: every point gets a real category. -----------------------
    # Categories are harmonized from metadata, not measured tissue annotations.
    ColorBy(
        key="tissue", label="Tissue", scope=(ARCHS4, OSDR), resolver=_tissue,
        hint="Tissue categories group OSDR labels and keyword matches from GEO metadata. "
             "Other: no category matched. Unknown: no tissue text.",
        needs=_ARCHS4_META_NEEDS,
    ),
    ColorBy(
        key="species", label="Species", scope=(ARCHS4, OSDR), resolver=_species,
        hint="OSDR samples are mouse; ARCHS4 includes mouse and human samples.",
    ),
    # There is deliberately no unsupervised-cluster color-by here. A k-means
    # partition of the same 512-d vectors the projection was fit on was built,
    # measured, and cut: 81.9% of its labels are recoverable from the 2-D
    # coordinates alone, so it mostly redraws the shape already on screen, and
    # a structure-free Voronoi null reproduced its spatial coherence to within
    # 1.5 points. It is also arbitrary - seed-to-seed agreement is only
    # ARI ~0.45 - and 81% species-pure. Painting an arbitrary partition on a
    # scientific instrument and numbering it "Cluster 1..24" invites exactly the
    # over-reading the rest of this file exists to prevent. See IMPLEMENTATION.md.
    #
    # There are also deliberately no OSDR-only fields. Nine of them shipped -
    # Flight vs Ground, Spaceflight arm, Strain, Sex, Genotype, Study, Habitat,
    # Mission duration, Diet - and every one was removed on 2026-08-11 because
    # none of them separated anything on this map. They were the field the
    # coverage machinery below was built for, and they are the reason it looks
    # over-engineered for two entries; it is not, because **Tissue becomes an
    # OSDR-only field on a machine that never fetched the GEO join**, which is
    # the state a fresh clone starts in. Coverage, the "OSDR only" scope note,
    # the amber partial bar and the renderer's context cloud are all still live
    # on that path and are all still load-bearing. See
    # docs/design-notes.md#osdr-only-color-bys for what was measured.
)

_BY_KEY = {c.key: c for c in REGISTRY}

GROUP_WHOLE_MAP = "Whole map"
#: A field that can only color the 2,108 OSDR points on this machine. No field
#: is registered that way any more; Tissue lands here when the optional GEO
#: join is absent, which is the only route into this group and the reason it
#: still exists.
GROUP_OSDR = "OSDR only"


def get(key: str) -> ColorBy:
    """The spec for a key, falling back to the default rather than raising.

    A stale key can reach here from a browser that kept its old selection across
    a rebuild, and taking the app down for that would be a poor trade.
    """
    return _BY_KEY.get(key) or _BY_KEY[default_key()]


def default_key() -> str:
    """The best color-by to open on: the first whole-map field that works."""
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
    """(points this field colors, total points)."""
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

    An option that cannot color anything is shown and disabled rather than
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
