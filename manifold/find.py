"""Resolve an identifier to the points it names on the map.

The map draws 942,563 glyphs and, until this module existed, offered no way to
ask about a specific one. This is the whole of the answer: a string in, a list
of point indices out.

It opens no embedding, builds no figure and imports nothing from Dash, so it is
unit-testable against the synthetic fixture corpus on a machine with neither the
963 MB memmap nor the real cache - the same boundary `bridge_rna/cohorts.py`
draws for what a cohort is.

**Identity only.** Four grammars resolve: a GEO sample (`GSM…`), a GEO series
(`GSE…`), an OSDR study (`OSD-###`), and an OSDR sample by its full
`<study>|<name>` key or by the name alone. Free text does not, and that is a
decision rather than an omission. `archs4_metadata.parquet` carries `title`,
`source_name` and `characteristics`, and a substring scan of 940,455 rows is
affordable - but it would return hundreds of rows for "liver" and read as a
biological query when it is a string match, on the one map whose whole design is
that a field declares what it does and does not describe. "Where is liver" is
the Tissue color-by's question and `manifold/colorby.py` already answers it
across both corpora. A query that matches no grammar is reported as `"shape"`
rather than `"absent"` precisely so the interface can say that.

**Two failures that look alike and are not.** A well-formed accession this
corpus does not contain is `"absent"`. A well-formed accession on a machine
where the optional GEO join was never fetched is `"no_geo_metadata"`, because
without `cache/archs4_metadata.parquet` 940,455 of the 942,563 points cannot be
addressed at all, and telling a user their accession does not exist when the
truth is that this machine cannot look it up is the same class of error
invariant 5 exists to prevent. `searchable()` is what the rail states up front,
and it reads `data.archs4_metadata_available` itself rather than re-deriving the
path - a second source of truth for that file was already a real bug once.

**One distinction is deliberately not drawn.** 788 of the 2,896 OSDR samples the
retrieval picker lists were never embedded and have no position here, so
searching one of them returns `"absent"` along with every genuine typo. The map
cannot honestly do better: the catalog that knows those 788 exist lives on the
retrieval side, behind `BRIDGE_RNA_ROOT`, and the map view deliberately opens
nothing there. Reading it to sharpen one error message would give the map a
dependency on the Bridge RNA repository that it does not otherwise have. Only 1
of the 788 has a study that is itself on the map, so inferring it from map-side
data alone would sharpen the message for one sample in 788 and mislead about the
rest.
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

_GSM = re.compile(r"^GSM(\d+)$")
_GSE = re.compile(r"^GSE(\d+)$")
#: The hyphen is optional because OSD-100 and OSD100 are the same study to
#: everyone except a string compare, and whitespace creeps in from a paste.
_OSD = re.compile(r"^OSD[\s-]*(\d+)$")


@dataclass(frozen=True)
class _Archs4Index:
    """Sorted integer keys into the ARCHS4 block, and the rows they address.

    A row **is** a point index: ARCHS4 occupies rows 0..n_archs4-1 of the global
    point order, and `global_index` is verified equal to row position in the
    parquet, so nothing here joins or offsets.

    int32 throughout, which is what makes this affordable. Measured on the real
    corpus: these four arrays retain **15.0 MB** and build in **460 ms**,
    against 96.8 MB retained for `pd.Index(accessions).get_loc`, on an app whose
    whole working set is 80.8 MB. Accession numbers are far below int32's range
    and 940,455 rows are further still.

    Two notes on that 460 ms, because both figures it corrects were wrong in
    instructive ways. It is one-time, on the first search of a session, and a
    warm lookup is then about 0.8 ms. And it is dominated by materializing two
    940,455-row string columns, not by the integer parse: the parse alone is
    about 200 ms whether it is written as a regex, a slice, or a Python loop,
    all three measured within 20 ms of each other, so there is nothing to win by
    making it cleverer.
    """
    gsm_keys: np.ndarray
    gsm_rows: np.ndarray
    gse_keys: np.ndarray
    gse_rows: np.ndarray


@lru_cache(maxsize=1)
def _archs4_index() -> _Archs4Index | None:
    """Build the accession index, or None when the GEO join is absent.

    Built on first search rather than at import, so a session that never
    searches never pays for it.
    """
    meta = data.archs4_metadata()
    if meta is None:
        return None
    return _Archs4Index(*_sorted_keys(meta.get("geo_accession"), "GSM"),
                        *_sorted_keys(meta.get("series_id"), "GSE"))


def _sorted_keys(column, prefix: str) -> tuple[np.ndarray, np.ndarray]:
    """(keys, rows) sorted by key, skipping anything not `<prefix><digits>`.

    The filter is load-bearing, not defensive tidiness. 839 rows of the real
    join carry an **empty** series_id: samples present in the release-matched
    v2.5 metadata and absent from the v2.latest the API serves. Slicing the
    digits off one raises `ValueError` and takes the whole index build with it,
    and coercing it to 0 would file those 839 samples under a series that does
    not exist. They are skipped, so they are addressable by their own GSM and
    by nothing else.
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
def _osdr_index() -> tuple[dict, dict, dict]:
    """(by full key, by bare name, by study) -> point indices.

    2,108 rows, so plain dicts. OSDR occupies the block after ARCHS4, hence the
    `n_archs4` offset on every value.
    """
    n_archs4, _, _ = data.counts()
    meta = data.osdr_metadata()
    by_key: dict[str, int] = {}
    by_name: dict[str, int] = {}
    by_study: dict[str, list[int]] = {}
    keys = meta["sample_key"].astype(str).to_numpy() if "sample_key" in meta else []
    studies = (meta["study"].astype(str).to_numpy() if "study" in meta
               else np.array([""] * len(keys)))
    for row, key in enumerate(keys):
        point = n_archs4 + row
        by_key[key.casefold()] = point
        if "|" in key:
            by_name[key.split("|", 1)[1].casefold()] = point
        by_study.setdefault(str(studies[row]).casefold(), []).append(point)
    return by_key, by_name, by_study


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
    """Resolve `query` to the points it names.

    Returns `{"points": [int], "kind": str, "label": str, "reason": str}`, where
    `reason` is `""` on a hit and one of the module constants on a miss, and
    `label` is the *canonical* identifier rather than what was typed, so the
    interface echoes `GSM9000005` back at someone who typed ` gsm9000005 `.

    `points` is sorted. Both consumers - the renderer indexing coordinates and
    `frame_points` taking a bounding box - are order-independent, but a stable
    order keeps the marks' draw order stable between identical searches.
    """
    raw = (query or "").strip()
    if not raw:
        return _miss(raw, EMPTY)

    token = " ".join(raw.split()).upper()

    m = _GSM.match(token)
    if m:
        return _geo_lookup(m.group(1), "gsm", f"GSM{int(m.group(1))}", raw)
    m = _GSE.match(token)
    if m:
        return _geo_lookup(m.group(1), "gse", f"GSE{int(m.group(1))}", raw)

    m = _OSD.match(token)
    if m:
        label = f"OSD-{int(m.group(1))}"
        points = _osdr_index()[2].get(label.casefold(), [])
        return _hit(points, "osd", label, raw) if points else _miss(label, ABSENT)

    by_key, by_name, _ = _osdr_index()
    folded = raw.casefold()
    point = by_key.get(folded, by_name.get(folded))
    if point is not None:
        return _hit([point], "osdr_sample", raw, raw)

    # Nothing matched, and the two reasons that can be need different sentences.
    return _miss(raw, _shape_or_absent(raw))


def _shape_or_absent(raw: str) -> str:
    """Did this look like an identifier the map lacks, or like a word?

    `OSD-141|Mmus_C57-6J_SPL_cells_Rep1_SP1` is a perfectly well-formed OSDR
    sample key that this map does not carry - it is one of the 788 the retrieval
    catalog lists and the embedding never covered - and answering it with "that
    is not an identifier, try the Tissue color-by" would be wrong twice. A key
    is unambiguous: nothing else contains a pipe. An OSDR sample *name* is
    underscore-joined and unspaced, which is not proof but is a good deal more
    like an identifier than "mouse brain" is.
    """
    if "|" in raw:
        return ABSENT
    return ABSENT if ("_" in raw and not any(c.isspace() for c in raw)) else SHAPE


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


def _geo_lookup(digits: str, kind: str, label: str, query: str) -> dict:
    index = _archs4_index()
    if index is None:
        return _miss(label, NO_GEO_METADATA)
    key = int(digits)
    if key > np.iinfo(np.int32).max:
        return _miss(label, ABSENT)
    keys, rows = ((index.gsm_keys, index.gsm_rows) if kind == "gsm"
                  else (index.gse_keys, index.gse_rows))
    lo = int(np.searchsorted(keys, key, "left"))
    hi = int(np.searchsorted(keys, key, "right"))
    if lo == hi:
        return _miss(label, ABSENT)
    return _hit(rows[lo:hi], kind, label, query)
