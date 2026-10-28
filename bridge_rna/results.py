"""The rows a retrieval returned, and every way the interface narrows them.

This module is the only place that knows how a result row is built, filtered,
sorted, paged or exported. The table, the plots, the CSV and the AI
summary all call the same `apply_filters`, so the four can never disagree about
which rows are in view. That is the reason it exists as one module rather than
as logic spread through the callbacks.

It is pure Python over plain lists of dicts: it imports no Dash, opens no
memmap and imports nothing from `bridge_rna.retrieval`, so it is tested against
hand-built rows on a machine with neither artifact. The shapes it reads and
writes are fixed in the "Contracts" section of `docs/retrieval-results.md`.

Two rules from that contract shape most of the code below.

Rank is a fact about the reference collection, never about the table. A row's
rank is its position among every ARCHS4 sample for that query, so filtering
and sorting reorder or drop rows but never renumber them.

Identical profiles (one expression profile deposited under two accessions) are
marked on their rows and never merged. A "Merge identical profiles" checkbox
existed until 2026-10-22; Joshua found it unclear, and each copy now stays its
own row with a plain "Same data as GSM..." note.
"""

from __future__ import annotations

import csv
import io
import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

PAGE_SIZE = 50

RETRIEVED_BY = ("either", "both", "a", "b")
SORT_COLUMNS = ("rank_a", "score_a", "rank_b", "score_b", "gsm", "gse",
                "title", "tissue", "species")
_NUMERIC_COLUMNS = frozenset({"rank_a", "score_a", "rank_b", "score_b"})
_TEXT_SEARCH_FIELDS = ("gsm", "gse", "title", "source_name")

# What pandas and `astype(str)` leave behind for a missing value. The retrieval
# frames go through `astype(str)` before `fillna("")`, so a missing title
# arrives as the literal string "nan" rather than as NaN, and printing it would
# put the word "nan" in a GEO title column.
_MISSING_TEXT = frozenset({"nan", "none", "<na>", "nat"})

_COLUMN_NAMES = {
    "rank_a": "Rank", "score_a": "Score", "rank_b": "Rank B", "score_b": "Score B",
    "gsm": "GSM", "gse": "Series", "title": "Title",
    "tissue": "Tissue category", "species": "Species",
}


# --- coercion ---------------------------------------------------------------


# What a score is called, keyed by the scoring that produced it
# (`bridge_rna/scoring.py`). Here rather than there so this module stays pure
# stdlib; scoring imports the names from here. A result saved before whitening
# existed has no scoring recorded and was plain cosine.
SCORE_NAMES = {"whitened": "whitened similarity", "cosine": "cosine similarity"}


def score_name(scoring: Any) -> str:
    return SCORE_NAMES.get(str(scoring or "").strip().lower(), SCORE_NAMES["cosine"])


def _text(value: Any) -> str:
    """A display string, with every spelling of "missing" folded to ""."""
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in _MISSING_TEXT else text


def _number(value: Any) -> float | None:
    """A finite float, or None. A bool is not a score."""
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _index(value: Any) -> int | None:
    """A memmap row as an int, or None on the demo path that has none."""
    number = _number(value)
    return None if number is None else int(number)


def _one_line(text: str) -> str:
    """Collapse whitespace so a value cannot break out of a `# ` provenance line."""
    return re.sub(r"\s+", " ", text).strip()


# --- building the rows ------------------------------------------------------


def _label(query: Any) -> str:
    """The human name of a query or cohort.

    A cohort carries `cohort_label`, the short name every other surface prints
    beside its mark ("Retina · Space Flight"); its `sample_name` is the longer
    sentence with the study prefixed. A sample or an upload has only
    `sample_name`. The id is the last resort so a label is never blank.
    """
    if not isinstance(query, Mapping):
        return ""
    for key in ("cohort_label", "sample_name", "sample_id"):
        text = _text(query.get(key))
        if text:
            return text
    return ""


def _profile(values: Any) -> list[float]:
    if not isinstance(values, (list, tuple)):
        return []
    return [n for n in (_number(v) for v in values) if n is not None]


def _best_rank(row: Mapping[str, Any]) -> int | None:
    ranks = [r for r in (row.get("rank_a"), row.get("rank_b")) if r is not None]
    return min(ranks) if ranks else None


def _best_score(row: Mapping[str, Any]) -> float | None:
    scores = [s for s in (_number(row.get("score_a")), _number(row.get("score_b")))
              if s is not None]
    return max(scores) if scores else None


def _rank_key(row: Mapping[str, Any]) -> tuple:
    """Best rank first, then memmap row (None last), then accession.

    The two tiebreaks only matter in a comparison, where A's rank 3 and B's
    rank 3 are different samples that both claim the same best rank.
    """
    best = _best_rank(row)
    index = row.get("archs4_index")
    return (best is None, best if best is not None else 0,
            index is None, index if index is not None else 0,
            _text(row.get("gsm")))


def build_results(
    payload: Mapping[str, Any] | None,
    *,
    reference_count: int,
    identical_groups: Sequence[Sequence[str]] = (),
) -> dict[str, Any] | None:
    """A hits-store payload as the `results-store` dict, or None if it holds no hits.

    The long free-text fields (`characteristics`, GEO summaries) are dropped
    because the table never shows them and every filter, sort or page request
    uploads this store; the export reads them back from `hits-store`.
    """
    if not isinstance(payload, Mapping):
        return None
    comparison = payload.get("comparison")
    comparison = comparison if isinstance(comparison, Mapping) else None
    hits_a = list(payload.get("hits") or [])
    hits_b = list(comparison.get("hits_b") or []) if comparison else []
    if not hits_a and not hits_b:
        return None

    rows: dict[Any, dict[str, Any]] = {}
    for side, hits in (("a", hits_a), ("b", hits_b)):
        seen: set[str] = set()
        for position, hit in enumerate(hits):
            if not isinstance(hit, Mapping):
                continue
            gsm = _text(hit.get("gsm"))
            # A GSM is unique in ARCHS4 (checked on the real metadata), so a
            # repeat within one list can only be an upstream defect; keeping the
            # first occurrence keeps that sample's better rank.
            if gsm and gsm in seen:
                continue
            seen.add(gsm)
            # A hit with no accession cannot be joined across the two lists,
            # but it is still a result at its rank, so it gets its own row.
            key = gsm or (side, position)
            row = rows.get(key)
            if row is None:
                row = rows[key] = {
                    "gsm": gsm, "gse": "", "title": "", "source_name": "",
                    "tissue": "", "species": "", "archs4_index": None,
                    "rank_a": None, "score_a": None, "cosine_a": None,
                    "rank_b": None, "score_b": None, "cosine_b": None,
                    "identical": [],
                }
            # Metadata is a property of the sample, not of the query that found
            # it, so either side may fill a field the other left blank.
            for field in ("gse", "title", "source_name", "tissue", "species"):
                if not row[field]:
                    row[field] = _text(hit.get(field))
            if row["archs4_index"] is None:
                row["archs4_index"] = _index(hit.get("archs4_index"))
            # Hits arrive best first from one ranked scan of the whole index,
            # so list position is the rank among every reference sample.
            row[f"rank_{side}"] = position + 1
            row[f"score_{side}"] = _number(hit.get("score"))
            cosine = hit.get("cosine")
            row[f"cosine_{side}"] = _number(cosine if cosine is not None else hit.get("score"))

    ordered = sorted(rows.values(), key=_rank_key)

    present = {row["gsm"]: i for i, row in enumerate(ordered) if row["gsm"]}
    twins: dict[str, set[str]] = {}
    for group in identical_groups or ():
        members = {_text(g) for g in group} & present.keys()
        for gsm in members:
            twins.setdefault(gsm, set()).update(members - {gsm})
    for row in ordered:
        row["identical"] = sorted(twins.get(row["gsm"], ()), key=present.__getitem__)

    query_b = comparison.get("query_b") if comparison else None
    return {
        "mode": _text(payload.get("mode")),
        "scoring": _text(payload.get("scoring")) or "cosine",
        "k": max(len(hits_a), len(hits_b)),
        "comparison": comparison is not None,
        "label_a": _label(payload.get("query")) or _text(payload.get("sample_id")),
        "label_b": _label(query_b) if comparison else "",
        "query_id": _text(payload.get("sample_id")),
        "reference_count": int(reference_count),
        "profile_a": _profile(payload.get("score_profile")),
        "profile_b": _profile(comparison.get("score_profile_b")) if comparison else [],
        "rows": ordered,
    }


# --- filters ----------------------------------------------------------------


def default_filters() -> dict[str, Any]:
    return {
        "series": [],
        "tissue": [], "species": [],
        "min_score": None, "text": "",
        "retrieved_by": "either",
    }


def _string_list(value: Any) -> list[str]:
    """A multi-select's value as a clean list; Dash sends a bare string for one."""
    if value is None:
        return []
    items: Iterable[Any] = [value] if isinstance(value, str) else (
        value if isinstance(value, (list, tuple, set)) else [value])
    out: list[str] = []
    for item in items:
        text = _text(item)
        if text and text not in out:
            out.append(text)
    return out


def _flag(value: Any) -> bool:
    """A checkbox's value: a bool, or a Checklist list that is ticked when non-empty."""
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "on", "yes"}
    if isinstance(value, (list, tuple, set)):
        return len(value) > 0
    return bool(value) if isinstance(value, (bool, int)) else False


def normalize_filters(raw: Any) -> dict[str, Any]:
    """Any widget state, including None and wrong types, as a valid Filters dict."""
    raw = raw if isinstance(raw, Mapping) else {}
    retrieved_by = _text(raw.get("retrieved_by")).lower()
    text = raw.get("text")
    return {
        "series": _string_list(raw.get("series")),
        "tissue": _string_list(raw.get("tissue")),
        "species": _string_list(raw.get("species")),
        "min_score": _number(raw.get("min_score")),
        "text": text.strip() if isinstance(text, str) else "",
        "retrieved_by": retrieved_by if retrieved_by in RETRIEVED_BY else "either",
    }


def _is_comparison(results: Mapping[str, Any] | None) -> bool:
    # Without the results there is no way to know, and naming a retrieved-by
    # filter that turns out to be inert is better than hiding one that is not.
    return True if results is None else bool(results.get("comparison"))


def filters_active(filters: Any, results: Mapping[str, Any] | None = None) -> bool:
    """Whether any filter narrows the rows; drives "Clear filters".

    Defined as "describe_filters has something to say", so the control that
    clears filters and the summary that names them cannot disagree. Pass the
    results so a retrieved-by choice left over on a single query, where it is
    ignored, does not count.
    """
    return bool(describe_filters(filters, results))


def _passes(row: Mapping[str, Any], f: Mapping[str, Any], comparison: bool) -> bool:
    # A series filter shows only the chosen series. A Hide mode existed until
    # 2026-10-22 and was removed at Joshua's request.
    if f["series"] and row.get("gse", "") not in f["series"]:
        return False
    if f["tissue"] and row.get("tissue", "") not in f["tissue"]:
        return False
    if f["species"] and row.get("species", "") not in f["species"]:
        return False
    if f["min_score"] is not None:
        best = _best_score(row)
        if best is None or best < f["min_score"]:
            return False
    if f["text"]:
        needle = f["text"].casefold()
        if not any(needle in _text(row.get(k)).casefold() for k in _TEXT_SEARCH_FIELDS):
            return False
    if comparison and f["retrieved_by"] != "either":
        in_a, in_b = row.get("rank_a") is not None, row.get("rank_b") is not None
        wanted = {"both": in_a and in_b, "a": in_a and not in_b,
                  "b": in_b and not in_a}[f["retrieved_by"]]
        if not wanted:
            return False
    return True


def apply_filters(rows: Sequence[Mapping[str, Any]],
                  filters: Any) -> list[dict[str, Any]]:
    """The rows that pass, in input order."""
    f = normalize_filters(filters)
    # Every comparison row came from one of two top-k lists and B's list is
    # never empty, so a row set with no B rank is a single query.
    comparison = any(r.get("rank_b") is not None for r in rows)
    return [r for r in rows if _passes(r, f, comparison)]


# --- sort and page ----------------------------------------------------------


def _normalize_sort(sort: Any) -> tuple[str | None, bool]:
    sort = sort if isinstance(sort, Mapping) else {}
    column = _text(sort.get("column"))
    if column not in SORT_COLUMNS:
        return None, False
    return column, _flag(sort.get("descending"))


def sort_rows(rows: Sequence[Mapping[str, Any]], sort: Any) -> list[dict[str, Any]]:
    """Rows in the requested order; ranks are never renumbered.

    A row with no value in the column goes last in either direction, because
    a blank rank means "not in that cohort's top k", and sorting it first when
    descending would lead the table with rows that have nothing to compare.
    """
    by_rank = sorted(rows, key=_rank_key)
    column, descending = _normalize_sort(sort)
    if column is None:
        return by_rank

    def value(row: Mapping[str, Any]) -> Any:
        if column in _NUMERIC_COLUMNS:
            return _number(row.get(column))
        return _text(row.get(column)).casefold() or None

    present = [r for r in by_rank if value(r) is not None]
    missing = [r for r in by_rank if value(r) is None]
    # Python's sort is stable with reverse=True too, so equal values keep the
    # best-rank order the list already has in both directions.
    present.sort(key=value, reverse=descending)
    return present + missing


def page_rows(rows: Sequence[Mapping[str, Any]], page: Any) -> dict[str, Any]:
    """One page of rows and the numbers the pager prints ("Rows 51-100 of 412").

    An empty result still has one page, so page 0 is always a valid request
    and the pager never has to special-case "page 1 of 0".
    """
    total = len(rows)
    pages = max(1, math.ceil(total / PAGE_SIZE))
    number = _number(page)
    current = min(max(int(number) if number is not None else 0, 0), pages - 1)
    first = current * PAGE_SIZE
    chunk = list(rows[first:first + PAGE_SIZE])
    return {
        "rows": chunk,
        "page": current,
        "pages": pages,
        "start": first + 1 if chunk else 0,
        "end": first + len(chunk),
        "total": total,
    }


def filter_options(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[tuple[str, int]]]:
    """Each filterable value with its row count, most common first.

    Counted over the rows handed in, which is every retrieved row, so a choice
    in one filter does not make the options of another disappear under the
    user while they are still choosing.
    """
    out: dict[str, list[tuple[str, int]]] = {}
    for name, field in (("series", "gse"), ("tissue", "tissue"), ("species", "species")):
        counts = Counter(v for v in (_text(r.get(field)) for r in rows) if v)
        out[name] = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0].casefold(), kv[0]))
    return out


# --- describing and exporting -----------------------------------------------


def _side_name(results: Mapping[str, Any] | None, side: str) -> str:
    label = _text((results or {}).get(f"label_{side}"))
    return label or f"cohort {side.upper()}"


def describe_filters(filters: Any, results: Mapping[str, Any] | None) -> list[str]:
    """One plain-English line per active filter, in the order the panel shows them."""
    f = normalize_filters(filters)
    lines: list[str] = []
    if f["series"]:
        lines.append(f"Only series: {', '.join(f['series'])}")
    if f["tissue"]:
        lines.append(f"Tissue category: {', '.join(f['tissue'])}")
    if f["species"]:
        lines.append(f"Species: {', '.join(f['species'])}")
    if f["min_score"] is not None:
        lines.append(f"Minimum score {f['min_score']!r}")
    if f["text"]:
        lines.append(f"Text contains \"{f['text']}\"")
    if f["retrieved_by"] != "either" and _is_comparison(results):
        lines.append({
            "both": "Retrieved by both cohorts",
            "a": f"Retrieved only by {_side_name(results, 'a')}",
            "b": f"Retrieved only by {_side_name(results, 'b')}",
        }[f["retrieved_by"]])
    return lines


def describe_sort(sort: Any, results: Mapping[str, Any] | None = None) -> str:
    """The sort as a phrase, naming cohorts in a comparison."""
    column, descending = _normalize_sort(sort)
    if column is None:
        return "Best rank first"
    name = _COLUMN_NAMES[column]
    if results is not None and results.get("comparison") and column in _NUMERIC_COLUMNS:
        name = f"{name.split()[0]} ({_side_name(results, column[-1])})"
    return f"{name}, {'descending' if descending else 'ascending'}"


def _csv_score(value: Any) -> str:
    # repr is the shortest string that reads back as the same float64, so the
    # file carries full precision without trailing noise digits.
    number = _number(value)
    return "" if number is None else repr(number)


def _csv_int(value: Any) -> str:
    return "" if value is None else str(int(value))


def to_csv(
    rows: Sequence[Mapping[str, Any]],
    results: Mapping[str, Any],
    filters: Any,
    sort: Any,
    hits_lookup: Mapping[str, Mapping[str, Any]],
    exported_at: str,
) -> str:
    """The rows as CSV, in the order given, under `# ` provenance lines.

    `pandas.read_csv(..., comment="#")` must read it back, and pandas truncates
    an *unquoted* field at a `#` anywhere on a line - a GEO title such as
    "Sample #3" would lose its tail and shift no columns, which is the worst
    kind of silent damage. A quoted field is respected (checked against pandas
    3.0's C parser, including a quoted field that starts with `#` and one that
    spans lines with `#` after the break). So every data field is quoted, and
    the provenance lines are written raw with their whitespace collapsed, so no
    value can start a line that is not a comment.
    """
    comparison = bool(results.get("comparison"))
    reference = int(results.get("reference_count") or 0)
    provenance = ["Bridge RNA retrieval export"]
    if comparison:
        provenance += [f"Query A: {_text(results.get('label_a'))}",
                       f"Query B: {_text(results.get('label_b'))}"]
    else:
        provenance.append(f"Query: {_text(results.get('label_a'))}")
    whitened = _text(results.get("scoring")) == "whitened"
    provenance += [
        f"Query ID: {_text(results.get('query_id'))}",
        f"Mode: {_text(results.get('mode'))}",
        ("Scoring: whitened similarity (cosine after whitening the embedding); "
         "each match's plain cosine is in the cosine column" if whitened
         else "Scoring: cosine similarity"),
        f"Matches requested (k): {int(results.get('k') or 0)}"
        + (" per cohort" if comparison else ""),
        f"Reference collection: {reference:,} ARCHS4 samples",
    ]
    described = describe_filters(filters, results)
    provenance += [f"Filter: {line}" for line in described] or ["Filters: none"]
    provenance += [
        f"Rows: {len(rows)}",
        f"Sort: {describe_sort(sort, results)}",
        f"Exported at: {_text(exported_at)}",
        f"Ranks are positions among all {reference:,} reference samples for the "
        f"query, not row numbers in this file. {score_name(results.get('scoring')).capitalize()} "
        "between embeddings is not biological equivalence.",
    ]

    if comparison:
        rank_columns = (["rank_a", "score_a", "cosine_a", "rank_b", "score_b", "cosine_b", "gsm"]
                        if whitened else ["rank_a", "score_a", "rank_b", "score_b", "gsm"])
    else:
        rank_columns = ["rank", "gsm", "score"] + (["cosine"] if whitened else [])
    header = rank_columns + ["gse", "title", "source_name", "characteristics",
                             "tissue_category", "species", "identical_profile_gsms",
                             "archs4_index"]

    out = io.StringIO()
    for line in provenance:
        out.write(f"# {_one_line(line)}\n")
    writer = csv.writer(out, quoting=csv.QUOTE_ALL, lineterminator="\n")
    writer.writerow(header)
    for row in rows:
        gsm = _text(row.get("gsm"))
        hit = hits_lookup.get(gsm) if gsm else None
        if comparison and whitened:
            lead = [_csv_int(row.get("rank_a")), _csv_score(row.get("score_a")),
                    _csv_score(row.get("cosine_a")),
                    _csv_int(row.get("rank_b")), _csv_score(row.get("score_b")),
                    _csv_score(row.get("cosine_b")), gsm]
        elif comparison:
            lead = [_csv_int(row.get("rank_a")), _csv_score(row.get("score_a")),
                    _csv_int(row.get("rank_b")), _csv_score(row.get("score_b")), gsm]
        else:
            lead = [_csv_int(row.get("rank_a")), gsm, _csv_score(row.get("score_a"))]
            if whitened:
                lead.append(_csv_score(row.get("cosine_a")))
        writer.writerow(lead + [
            _text(row.get("gse")),
            _text(row.get("title")),
            _text(row.get("source_name")),
            _text((hit or {}).get("characteristics")),
            _text(row.get("tissue")),
            _text(row.get("species")),
            ";".join(row.get("identical") or ()),
            _csv_int(row.get("archs4_index")),
        ])
    return out.getvalue()
