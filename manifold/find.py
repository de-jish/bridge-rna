"""Resolve a study identifier to the points it names on the map.

The map draws 942,563 glyphs and, until this module existed, offered no way to
ask about a specific one. This is the whole of the answer: a string in, a list
of point indices out.

It opens no embedding, builds no figure and imports nothing from Dash, so it is
unit-testable against the synthetic fixture corpus on a machine with neither the
963 MB memmap nor the real cache - the same boundary `bridge_rna/cohorts.py`
draws for what a cohort is.

**Studies, not samples.** Two grammars resolve: a GEO series (`GSE…`) and an
OSDR study (`OSD-###`). That is a deliberate narrowing, made on 2026-08-13. The
box used to take an individual GEO sample (`GSM…`) and an OSDR sample by its
full `<study>|<name>` key or by name alone, which meant four grammars, two
suggestion rankers and a 39-character identifier nobody types from memory - and
the question a reader actually brings to a 942,563-point map is "where did this
experiment land", not "where is replicate 3". A study is also the unit that
draws a *shape* on the map: one glyph among a million is a dot, and its twelve
siblings are a neighbourhood. `describe()` still renders a single sample's
record, because a series of one resolves to one point.

Clicking an OSDR diamond still selects that one sample, through `osdr_sample()`
rather than through the typed grammar. The map's own marks may address a sample;
the box may not.

**Free text does not resolve, and that is a decision rather than an omission.**
`archs4_metadata.parquet` carries `title`, `source_name` and `characteristics`,
and a substring scan of 940,455 rows is affordable - but it would return
hundreds of rows for "liver" and read as a biological query when it is a string
match, on the one map whose whole design is that a field declares what it does
and does not describe. "Where is liver" is the Tissue color-by's question and
`manifold/colorby.py` already answers it across both corpora. A query that
matches no grammar is reported as `"shape"` rather than `"absent"` precisely so
the interface can say that.

**Two failures that look alike and are not.** A well-formed accession this
corpus does not contain is `"absent"`. A well-formed GEO series on a machine
where the optional GEO join was never fetched is `"no_geo_metadata"`, because
without `cache/archs4_metadata.parquet` 940,455 of the 942,563 points cannot be
addressed at all, and telling a user their accession does not exist when the
truth is that this machine cannot look it up is the same class of error
invariant 5 exists to prevent. `searchable()` is what the rail states up front,
and it reads `data.archs4_metadata_available` itself rather than re-deriving the
path - a second source of truth for that file was already a real bug once.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from . import data

#: A miss carries one of these, and each one gets its own sentence on the rail.
#: "" is a hit.
EMPTY = "empty"                    # nothing typed
SHAPE = "shape"                    # matches no grammar; probably free text
ABSENT = "absent"                  # well-formed, and this corpus lacks it
NO_GEO_METADATA = "no_geo_metadata"  # a GEO id, on a machine with no join

_GSE = re.compile(r"^GSE(\d+)$")
#: The hyphen is optional because OSD-100 and OSD100 are the same study to
#: everyone except a string compare, and whitespace creeps in from a paste.
_OSD = re.compile(r"^OSD[\s-]*(\d+)$")


@dataclass(frozen=True)
class _Archs4Index:
    """Sorted integer series keys into the ARCHS4 block, and the rows they name.

    A row **is** a point index: ARCHS4 occupies rows 0..n_archs4-1 of the global
    point order, and `global_index` is verified equal to row position in the
    parquet, so nothing here joins or offsets.

    int32 throughout, which is what makes this affordable. Measured on the real
    corpus when the index also carried per-sample GSM keys: four arrays retained
    **15.0 MB** and built in **460 ms**, against 96.8 MB retained for
    `pd.Index(accessions).get_loc`, on an app whose whole working set is
    80.8 MB. Dropping the GSM half with the sample grammar halves both figures.

    That build cost is one-time, on the first search of a session, and a warm
    lookup is then about 0.8 ms. It is dominated by materializing the string
    column, not by the integer parse: the parse alone is about 200 ms whether it
    is written as a regex, a slice, or a Python loop, all three measured within
    20 ms of each other, so there is nothing to win by making it cleverer.
    """
    gse_keys: np.ndarray
    gse_rows: np.ndarray


@lru_cache(maxsize=1)
def _archs4_index() -> _Archs4Index | None:
    """Build the series index, or None when the GEO join is absent.

    Built on first search rather than at import, so a session that never
    searches never pays for it.
    """
    meta = data.archs4_metadata()
    if meta is None:
        return None
    return _Archs4Index(*_sorted_keys(meta.get("series_id"), "GSE"))


def _sorted_keys(column, prefix: str) -> tuple[np.ndarray, np.ndarray]:
    """(keys, rows) sorted by key, skipping anything not `<prefix><digits>`.

    The filter is load-bearing, not defensive tidiness. 839 rows of the real
    join carry an **empty** series_id: samples present in the release-matched
    v2.5 metadata and absent from the v2.latest the API serves. Slicing the
    digits off one raises `ValueError` and takes the whole index build with it,
    and coercing it to 0 would file those 839 samples under a series that does
    not exist. They are skipped, so no query reaches them.
    """
    if column is None:
        return np.empty(0, np.int32), np.empty(0, np.int32)
    # A pandas extract rather than `np.char`, which has no loop for the object
    # dtype these columns arrive in and raises rather than falling back.
    digits = column.astype(str).str.extract(rf"^{prefix}(\d+)$", expand=False)
    rows = np.flatnonzero(digits.notna().to_numpy()).astype(np.int32)
    if rows.size == 0:
        return np.empty(0, np.int32), np.empty(0, np.int32)
    keys = digits.dropna().astype("int64").to_numpy()
    if keys.max() > np.iinfo(np.int32).max:  # pragma: no cover - GEO is far below
        raise RuntimeError(f"{prefix} accession exceeds int32; widen the index")
    keys = keys.astype(np.int32)
    order = np.argsort(keys, kind="stable")
    return keys[order], rows[order]


@lru_cache(maxsize=1)
def _osdr_index() -> tuple[dict, dict]:
    """(by full sample key, by study) -> point indices.

    2,108 rows, so plain dicts. OSDR occupies the block after ARCHS4, hence the
    `n_archs4` offset on every value.

    The sample keys are here for the map's own click path, which selects the
    diamond that was clicked. Nothing typed into the box reaches them.
    """
    n_archs4, _, _ = data.counts()
    meta = data.osdr_metadata()
    by_key: dict[str, int] = {}
    by_study: dict[str, list[int]] = {}
    keys = meta["sample_key"].astype(str).to_numpy() if "sample_key" in meta else []
    studies = (meta["study"].astype(str).to_numpy() if "study" in meta
               else np.array([""] * len(keys)))
    for row, key in enumerate(keys):
        by_key[key.casefold()] = n_archs4 + row
        by_study.setdefault(str(studies[row]).casefold(), []).append(n_archs4 + row)
    return by_key, by_study


def clear_caches() -> None:
    """Drop the memoized indexes. For tests that swap the artifacts underneath."""
    _archs4_index.cache_clear()
    _osdr_index.cache_clear()


def searchable() -> tuple[str, ...]:
    """Which corpora this machine can resolve an identifier in, right now.

    OSDR is always searchable: it needs only `osdr_metadata.parquet`, which the
    app cannot start without. ARCHS4 needs the optional GEO join. The rail
    states this under the control, with the command that fixes it, which is the
    same shape of answer `colorby.py` gives for a field it cannot color.
    """
    return ("osdr", "archs4") if data.archs4_metadata_available() else ("osdr",)


def _miss(query: str, reason: str) -> dict:
    return {"points": [], "kind": "", "label": query, "reason": reason}


def _hit(points, kind: str, label: str, query: str) -> dict:
    return {"points": sorted(int(p) for p in points),
            "kind": kind, "label": label, "query": query, "reason": ""}


def find(query: str | None) -> dict:
    """Resolve `query` to the points of the study it names.

    Returns `{"points": [int], "kind": str, "label": str, "reason": str}`, where
    `reason` is `""` on a hit and one of the module constants on a miss, and
    `label` is the *canonical* identifier rather than what was typed, so the
    interface echoes `GSE143281` back at someone who typed ` gse143281 `.

    `points` is sorted. Both consumers - the renderer indexing coordinates and
    `frame_points` taking a bounding box - are order-independent, but a stable
    order keeps the marks' draw order stable between identical searches.
    """
    raw = (query or "").strip()
    if not raw:
        return _miss(raw, EMPTY)

    token = " ".join(raw.split()).upper()

    m = _GSE.match(token)
    if m:
        return _series_lookup(m.group(1), f"GSE{int(m.group(1))}", raw)

    m = _OSD.match(token)
    if m:
        label = f"OSD-{int(m.group(1))}"
        points = _osdr_index()[1].get(label.casefold(), [])
        return _hit(points, "osd", label, raw) if points else _miss(label, ABSENT)

    return _miss(raw, SHAPE)


def osdr_sample(key: str | None) -> dict | None:
    """The point one OSDR `<study>|<name>` key names, or None.

    The map's click path, and nothing else. A diamond on screen is an
    unambiguous reference to one sample, so selecting it is not the free-text
    problem the typed grammar refuses - but it is also not something the box
    offers, which is why this is a separate entry point rather than a fifth
    branch of `find()`.
    """
    folded = str(key or "").strip().casefold()
    point = _osdr_index()[0].get(folded)
    if point is None:
        return None
    return _hit([point], "osdr_sample", str(key), str(key))


# --- Suggestions -------------------------------------------------------------
#
# What the box offers while a study identifier is being typed. Everything here
# is a *prefix* of the same two grammars `find()` resolves, and that is the
# whole design rule: a suggestion is a completion of what the user is typing,
# never a search for what they might have meant.
#
# **The free-text refusal survives, and it had to.** `archs4_metadata.parquet`
# carries `title`, `source_name` and `characteristics`, and substring-scanning
# them is affordable - it is the same 200 ms scan the module docstring above
# already costs out. Offering it here would have reintroduced, as a dropdown,
# exactly the feature that was rejected as a search box: "liver" would return
# hundreds of rows and read as a biological query on the one map whose whole
# design is that a field declares what it does and does not describe. So a query
# matching no grammar gets no suggestions and keeps its `shape` reason, which is
# what puts the sentence about the Tissue color-by under the box instead.

#: How many suggestions the box will offer. The cap is what bounds the payload;
#: the list's `max-height` is what keeps the control compact. Two separate
#: decisions, deliberately - ten rows in a five-row window is what makes the
#: list visibly scrollable rather than silently truncated.
SUGGESTION_LIMIT = 10


def _decade_slices(keys: np.ndarray, digits: str) -> list[tuple[int, int]]:
    """`(lo, hi)` slices of the sorted key array whose numbers start with `digits`.

    The index stores numbers, not strings - int32 keys against a pandas string
    index that measured 96.8 MB, which is what made the find box affordable at
    all. A numeric prefix is therefore not a string compare but a union of
    ranges: the integers beginning "42" are [42, 43), then [420, 430), then
    [4200, 4300), and so on. Nine `searchsorted` pairs covers every GEO
    accession ever issued, and each is a binary search over 940,455 int32s.

    Returned shortest-number-first, so typing "42" offers GSE42 above GSE4200.
    """
    if not digits or not digits.isdigit() or keys.size == 0:
        return []
    base = int(digits)
    ceiling = int(keys[-1])
    out: list[tuple[int, int]] = []
    lo_value, hi_value = base, base + 1
    while lo_value <= ceiling:
        lo = int(np.searchsorted(keys, lo_value, "left"))
        hi = int(np.searchsorted(keys, hi_value, "left"))
        if hi > lo:
            out.append((lo, hi))
        lo_value *= 10
        hi_value *= 10
    return out


def _suggestion(value: str, label: str, detail: str, kind: str) -> dict:
    """One row. `value` is what lands in the box, and `find()` must resolve it."""
    return {"value": value, "label": label, "detail": detail, "kind": kind}


def _samples(n: int) -> str:
    return f"{n:,} sample{'s' if n != 1 else ''}"


def _gse_suggestions(digits: str, limit: int) -> list[dict]:
    index = _archs4_index()
    if index is None or limit <= 0:
        return []
    out: list[dict] = []
    for lo, hi in _decade_slices(index.gse_keys, digits):
        # The slice is sorted, so a run of equal keys is one series and its
        # length is that series' sample count - no grouping pass needed.
        values, counts = np.unique(index.gse_keys[lo:hi], return_counts=True)
        for value, count in zip(values.tolist(), counts.tolist()):
            label = f"GSE{int(value)}"
            out.append(_suggestion(label, label, _samples(count), "gse"))
            if len(out) >= limit:
                return out
    return out


def _osd_suggestions(digits: str, limit: int) -> list[dict]:
    """OSDR studies whose accession starts with `digits`, lowest number first."""
    _, by_study = _osdr_index()
    matches = []
    for folded, points in by_study.items():
        number = folded[4:] if folded.startswith("osd-") else ""
        if number.isdigit() and number.startswith(digits):
            matches.append((int(number), len(points)))
    matches.sort()
    return [_suggestion(f"OSD-{n}", f"OSD-{n}", _samples(count), "osd")
            for n, count in matches[:limit]]


def suggest(query: str | None, limit: int = SUGGESTION_LIMIT) -> dict:
    """What `query` could be completed to.

    Returns ``{"items": [...], "reason": str}``. ``reason`` is ``""`` when there
    is something to offer and otherwise one of the module's miss constants, so
    the interface can tell "nothing matches GSE99999999" from "that is not an
    identifier at all" - the same distinction `find()` draws, drawn by the same
    constants, because answering both with an empty dropdown is how a user
    concludes the box is broken.

    Every ``value`` is an identifier `find()` resolves, so selecting a
    suggestion is indistinguishable from having typed it. That is what keeps one
    resolution path: this function ranks candidates and nothing else, and no
    part of the map is reachable through a suggestion that is not reachable by
    typing.
    """
    raw = (query or "").strip()
    if not raw:
        return {"items": [], "reason": EMPTY}
    limit = max(int(limit), 0)
    token = " ".join(raw.split()).upper()

    m = _GSE.match(token)
    if m:
        if _archs4_index() is None:
            return {"items": [], "reason": NO_GEO_METADATA}
        items = _gse_suggestions(m.group(1), limit)
        return {"items": items, "reason": "" if items else ABSENT}

    m = _OSD.match(token)
    if m:
        items = _osd_suggestions(m.group(1), limit)
        return {"items": items, "reason": "" if items else ABSENT}

    return {"items": [], "reason": SHAPE}


#: The canonical GEO record page. One accession, one URL, built in one place.
GEO_ACC_URL = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc="


def describe(found: dict | None) -> dict | None:
    """A display record for what was found: a heading, a few rows, and a link.

    Kept here rather than in `layout` because it is a metadata lookup and not a
    rendering decision, so it can be tested without Dash - and because the click
    path and the find path both want it, and the alternative was two functions
    reading the same two parquets.

    Returns None when there is nothing to show. A set gets its count rather than
    its members: the map is already drawing every one of them, and a list of
    8,764 accessions on a 268 px rail is not a panel.
    """
    points = (found or {}).get("points") or []
    if not points:
        return None
    n_archs4, _, _ = data.counts()
    label = str(found.get("label") or "")

    if len(points) > 1:
        # The identifier is the heading, so it is not also a row. A panel
        # reading "GSE143281 / Series: GSE143281" says it twice and buys
        # nothing; the count is the only thing here the heading does not carry.
        return {"kind": "set", "title": label,
                "rows": [("Samples on the map", f"{len(points):,}")],
                # An OSDR study has no GEO record, so it gets no link rather
                # than a link to a page that does not exist.
                "geo": label if found.get("kind") == "gse" else "",
                "sample_key": ""}

    point = int(points[0])
    if point >= n_archs4:
        return _osdr_record(point - n_archs4, label)
    return _archs4_record(point, label)


def _archs4_record(row: int, label: str) -> dict:
    meta = data.archs4_metadata()
    if meta is None or row >= len(meta):
        return {"kind": "archs4", "title": label, "rows": [], "geo": label,
                "sample_key": ""}
    r = meta.iloc[row]
    rows = [("Series", str(r.get("series_id") or "")),
            ("Tissue", str(r.get("tissue") or "")),
            ("Title", str(r.get("title") or ""))]
    return {"kind": "archs4", "title": str(r.get("geo_accession") or label),
            "rows": [(k, v) for k, v in rows if v],
            "geo": str(r.get("geo_accession") or label), "sample_key": ""}


def _osdr_record(row: int, label: str) -> dict:
    meta = data.osdr_metadata()
    if row >= len(meta):
        return {"kind": "osdr", "title": label, "rows": [], "geo": "",
                "sample_key": ""}
    r = meta.iloc[row]
    key = str(r.get("sample_key") or label)
    rows = [("Study", str(r.get("study") or "")),
            ("Tissue", str(r.get("tissue") or "")),
            ("Spaceflight", str(r.get("spaceflight") or ""))]
    return {"kind": "osdr", "title": key.split("|", 1)[-1],
            "rows": [(k, v) for k, v in rows if v],
            "geo": "", "sample_key": key}


def _series_lookup(digits: str, label: str, query: str) -> dict:
    index = _archs4_index()
    if index is None:
        return _miss(label, NO_GEO_METADATA)
    key = int(digits)
    if key > np.iinfo(np.int32).max:
        return _miss(label, ABSENT)
    lo = int(np.searchsorted(index.gse_keys, key, "left"))
    hi = int(np.searchsorted(index.gse_keys, key, "right"))
    if lo == hi:
        return _miss(label, ABSENT)
    return _hit(index.gse_rows[lo:hi], "gse", label, query)
