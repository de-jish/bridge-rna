"""Shared terms: the words a set of matches has in common, counted by series.

A result set is described by the words its GEO submitters used, compared with
how often they appear across the ARCHS4 reference's GEO series. It is a string match over
metadata, never a claim about biology, and it is counted once per **series**
because one series often deposits dozens of samples with near-identical
titles. Design and measurements: docs/match-digest.md.

The words come from exactly the fields the table's "Text contains" filter
searches (`results._TEXT_SEARCH_FIELDS`, minus the two accessions), so pressing
a word always shows the rows it was counted in. The tokenizer here is the only
one: `precompute/build_term_index.py` imports it to build the corpus side, so
the two sides cannot split words differently.

This module opens no embedding and imports no Dash, so it is testable against a
plain list of dicts.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from .results import _text  # the table's own missing-value folding

# The fields a word may come from: the free-text ones the text filter searches.
TEXT_FIELDS = ("title", "source_name")

# A word must be used by at least this many of the matches' series...
MIN_SERIES = 2
# ...and be at least this many times more common there than across GEO.
MIN_RATIO = 3.0
# At most this many words are shown per result set.
MAX_TERMS = 8
# Words shorter than this are mostly unit and replicate noise ("rep", "wk").
MIN_LENGTH = 3

BUILD_COMMAND = "python precompute/build_term_index.py"

# Letters first, then letters, digits and inner hyphens: "cd4", "il-6",
# "c57bl", "retinal". Pure numbers and dates never form a word.
_TOKEN = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*")
_TRAILING_DIGITS = re.compile(r"\d+$")

# Words that describe how a sample was labelled rather than what it is. Most of
# them would fail the 3x ratio anyway; this list keeps the ones that sometimes
# pass by chance ("gender" is written by a minority of series) off the row.
STOPWORDS = frozenset("""
and the for with from into onto per via not non
sample samples rep replicate replicates biological technical
rna rnaseq seq total library libraries polya mrna
control controls ctrl ctl untreated treated treatment vehicle mock
group groups batch run lane time point timepoint day days week weeks
month months hour hours age aged old young
sex gender male female males females
mouse mice murine human homo sapiens mus musculus
tissue tissues cell cells sample_id subject patient donor individual
type wild wildtype genotype strain line clone
experiment exp set condition state biol bio whole
""".split())

# Above this a ratio is shown as ">100x": its expected count is a small fraction
# of one series, so its digits are noise. The hover text keeps the exact value.
RATIO_DISPLAY_CAP = 100.0


def words(text: Any) -> set[str]:
    """The distinct countable words in one string, lowercased.

    Missing values are folded exactly as the table folds them (`results._text`),
    so a title stored as "NaT" or "None" is no title here either: a word the
    table cannot show must never be offered as a filter.
    """
    text = _text(text)
    if not text:
        return set()
    out = set()
    for token in _TOKEN.findall(text.casefold()):
        if len(token) < MIN_LENGTH or token in STOPWORDS:
            continue
        # "mouse1", "rep2", "day14" are a stopword with a counter on it.
        stem = _TRAILING_DIGITS.sub("", token)
        if stem in STOPWORDS or len(stem) < 2:
            continue
        out.add(token)
    return out


def sample_words(row: Mapping[str, Any]) -> set[str]:
    """The words of one sample's title and source name."""
    found: set[str] = set()
    for field in TEXT_FIELDS:
        found |= words(row.get(field))
    return found


def series_key(row: Mapping[str, Any]) -> str:
    """The unit a word is counted in. A sample with no series is its own."""
    gse = _text(row.get("gse") if "gse" in row else row.get("series_id"))
    if gse:
        return gse
    gsm = _text(row.get("gsm") if "gsm" in row else row.get("geo_accession"))
    return f"sample:{gsm}" if gsm else ""


# --- the corpus side ---------------------------------------------------------


@dataclass(frozen=True)
class TermTable:
    """How many corpus series use each word, and how many series there are."""

    counts: Mapping[str, int]
    total_series: int


def build_term_table(metadata: pd.DataFrame) -> pd.DataFrame:
    """One row per word: `term`, `series` (corpus series using it).

    `metadata` has `series_id`, `geo_accession` and the `TEXT_FIELDS`, as
    `cache/archs4_metadata.parquet` does. The total series count is carried as
    the `__series__` row so the artifact is self-describing; words used by a
    single series are dropped, because no result set can show one.
    """
    by_series: dict[str, set[str]] = {}
    columns = ["series_id", "geo_accession", *TEXT_FIELDS]
    for row in metadata[columns].itertuples(index=False):
        record = dict(zip(columns, row))
        key = series_key(record)
        if not key:
            continue
        by_series.setdefault(key, set()).update(sample_words(record))
    counts: dict[str, int] = {}
    for found in by_series.values():
        for word in found:
            counts[word] = counts.get(word, 0) + 1
    kept = {w: c for w, c in counts.items() if c >= MIN_SERIES}
    frame = pd.DataFrame({"term": list(kept), "series": list(kept.values())})
    frame = frame.sort_values(["series", "term"], ascending=[False, True])
    total = pd.DataFrame({"term": ["__series__"], "series": [len(by_series)]})
    return pd.concat([total, frame], ignore_index=True).astype(
        {"term": "string", "series": "int32"})


def table_from_frame(frame: pd.DataFrame) -> TermTable | None:
    if frame is None or frame.empty or "__series__" not in set(frame["term"]):
        return None
    total = int(frame.loc[frame["term"] == "__series__", "series"].iloc[0])
    body = frame[frame["term"] != "__series__"]
    return TermTable(dict(zip(body["term"].astype(str), body["series"].astype(int))),
                     total)


def _terms_path() -> Path:
    from manifold import paths  # the one registry of artifact paths

    return paths.ARCHS4_TERMS_PARQUET


def available() -> bool:
    return _terms_path().exists()


@lru_cache(maxsize=1)
def load_term_table() -> TermTable | None:
    """The corpus table, or None when it has not been built on this machine."""
    path = _terms_path()
    if not path.exists():
        return None
    try:
        return table_from_frame(pd.read_parquet(path))
    except Exception:  # a truncated or foreign file is the same as none
        return None


# --- the result-set side -----------------------------------------------------


def shared_terms(hits: Sequence[Mapping[str, Any]] | None,
                 table: TermTable | None) -> dict[str, Any]:
    """Describe one result set. Always returns a dict the view can render.

    `status` is one of:

    - ``ok``: `terms` holds up to `MAX_TERMS` words, each with `series` (how
      many of the matches' series use it), `of` (the matches' series count),
      `corpus_series`, `expected` and `ratio`.
    - ``none``: no word passed both thresholds.
    - ``one_series``: every match is from one series, so "shared across
      series" has nothing to compare.
    - ``no_text``: the matches carry no title or source name.
    - ``unavailable``: the corpus table has not been built.
    - ``empty``: there are no matches.
    """
    rows = [h for h in (hits or []) if isinstance(h, Mapping)]
    base = {"terms": [], "series_count": 0, "match_count": len(rows),
            "corpus_series": table.total_series if table else 0}
    if not rows:
        return {**base, "status": "empty"}
    if table is None:
        return {**base, "status": "unavailable"}

    by_series: dict[str, set[str]] = {}
    any_text = False
    for row in rows:
        key = series_key(row)
        if not key:
            continue
        found = sample_words(row)
        any_text = any_text or bool(found) or any(
            _text(row.get(f)) for f in TEXT_FIELDS)
        by_series.setdefault(key, set()).update(found)
    m = len(by_series)
    base["series_count"] = m
    if not any_text:
        return {**base, "status": "no_text"}
    if m < MIN_SERIES:
        return {**base, "status": "one_series"}

    tally: dict[str, int] = {}
    for found in by_series.values():
        for word in found:
            tally[word] = tally.get(word, 0) + 1

    n = max(1, table.total_series)
    terms = []
    for word, c in tally.items():
        if c < MIN_SERIES:
            continue
        # A word in two matched series is in at least two corpus series, so a
        # table that lacks it was built from other metadata. Its rate was never
        # measured, and inventing one would print a ratio from nothing.
        s = table.counts.get(word)
        if not s:
            continue
        expected = m * s / n
        ratio = c / expected
        if ratio < MIN_RATIO:
            continue
        terms.append({"term": word, "series": c, "of": m, "corpus_series": s,
                      "expected": expected, "ratio": ratio})
    terms.sort(key=lambda t: (-t["series"], -t["ratio"], t["term"]))
    terms = terms[:MAX_TERMS]
    return {**base, "status": "ok" if terms else "none", "terms": terms}


def message(description: Mapping[str, Any]) -> str:
    """The sentence for a result set with no word to show; empty when there are words."""
    status = description.get("status")
    m = description.get("series_count") or 0
    if status == "none":
        return (f"No word is shared by two or more of these {m:,} series "
                f"at {MIN_RATIO:.0f}x its reference rate.")
    if status == "one_series":
        return "Every match is from one GEO series, so there is nothing to compare across series."
    if status == "no_text":
        return "These matches have no GEO title or source name."
    if status == "unavailable":
        return f"The GEO word counts are not built on this machine. Run {BUILD_COMMAND}."
    return ""


def prompt_text(description: Mapping[str, Any]) -> str:
    """The shared terms as plain lines for the AI summary prompt."""
    terms = description.get("terms") or []
    if not terms:
        return message(description) or "No matches."
    return "\n".join(
        f"- {t['term']}: used by {t['series']} of the hits' {t['of']} series, "
        f"{format_ratio(t['ratio'])} its share of the ARCHS4 reference's series"
        for t in terms)


def format_ratio(ratio: float, *, exact: bool = False) -> str:
    """`20x`, `7.4x`, `>100x`: never a false decimal.

    `exact=True` prints the full value, for the hover text that explains it.
    """
    if not math.isfinite(ratio):
        return ""
    if ratio > RATIO_DISPLAY_CAP and not exact:
        return f">{RATIO_DISPLAY_CAP:.0f}x"
    if ratio >= 10:
        return f"{ratio:,.0f}x"
    return f"{ratio:.1f}x"


def term_label(term: Mapping[str, Any]) -> str:
    """The spoken name of a word's button: the three visible parts, in words."""
    ratio = format_ratio(term["ratio"]).replace(">", "more than ").replace("x", " times")
    return (f"{term['term']}: {term['series']} of {term['of']} series, "
            f"{ratio} its reference rate. Filter the table to it.")


def term_title(term: Mapping[str, Any], corpus_series: int) -> str:
    """The hover text for one word: every number behind the chip."""
    share = term["corpus_series"] / max(1, corpus_series)
    return (f"\"{term['term']}\" is in the title or source name of matches from "
            f"{term['series']} of their {term['of']} series. Of the "
            f"{corpus_series:,} series in the ARCHS4 reference (a sample with no "
            f"series counts as its own), {term['corpus_series']:,} use it "
            f"({share:.2%}), so {term['expected']:.2f} would be expected: "
            f"{format_ratio(term['ratio'], exact=True)}. Press to filter the table to it.")



def hits_of(payload: Mapping[str, Any] | None) -> tuple[list[dict], list[dict]]:
    """The A and B hit lists of a hits-store payload (B empty unless a comparison)."""
    if not isinstance(payload, Mapping):
        return [], []
    comparison = payload.get("comparison")
    hits_b = (list(comparison.get("hits_b") or [])
              if isinstance(comparison, Mapping) else [])
    return list(payload.get("hits") or []), hits_b


def describe_all(payload: Mapping[str, Any] | None,
                 table: TermTable | None = None,
                 *, load: bool = True) -> list[dict[str, Any]]:
    """One description per result set in a hits-store payload."""
    if table is None and load:
        table = load_term_table()
    hits_a, hits_b = hits_of(payload)
    out = [shared_terms(hits_a, table)]
    if isinstance(payload, Mapping) and isinstance(payload.get("comparison"), Mapping):
        out.append(shared_terms(hits_b, table))
    return out


__all__: Iterable[str] = (
    "TEXT_FIELDS", "MIN_SERIES", "MIN_RATIO", "MAX_TERMS", "BUILD_COMMAND",
    "words", "sample_words", "series_key", "TermTable", "build_term_table",
    "table_from_frame", "available", "load_term_table", "shared_terms",
    "format_ratio", "term_title", "hits_of", "describe_all",
)
