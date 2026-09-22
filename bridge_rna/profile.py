"""The ranked similarity profile: how a query's cosine scores decay with rank.

The retrieval network answers "what are the top five" and the map answers
"where do they sit". Neither answers "how far down is this still worth
reading", because both draw a fixed slice and neither shows the decay.

This module is the shape of that decay, and nothing else. It opens no memmap,
imports no Dash and no Plotly, and takes a plain descending score array, so it
is testable against a fixture on a machine with neither artifact.

Two things measured on the real corpus set every decision here, and both are
recorded in `docs/ranked-similarity-profile.md`.

**There are no breakpoints in this corpus.** 120 random cached OSDR queries
were scored at top-100, top-250 and top-1,000: **0 of 120 cleared the test at
any range**, median prominence 1.32 to 1.41 against a floor of 3. The estimator
is not blind - a synthetic step of 30% of span at the corpus's own scale scores
7.69, one of 15% scores 4.29, and a profile with no step at all scores 0.80, so
the real profiles sit with the no-step control. A ranked similarity profile
here is a smooth decay, and an interface that marked a transition on it would
invent one in every sample. `largest_local_step` therefore exists to *report
that finding with its number*, and `Step.found` is expected to be False.

**The axis cannot be fixed.** Span over the top 250 runs from 0.00027 at the
5th percentile to 0.01503 at the 95th, a 55x spread, so one window would draw
most samples as a flat line at the top. The axis is fitted per sample and
`span` is published beside it, because the span is what stops a magnified
nothing from reading as a separation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# One scan serves every range, so the deepest one is what the scan asks for.
PROFILE_DEPTH = 1000
RANGE_CHOICES = (100, 250, 500, 1000)
DEFAULT_RANGE = 250

# A local step is the mean of the W scores at and before a rank minus the mean
# of the W after it. Five on each side keeps one noisy rank from winning.
STEP_WINDOW = 5
# The neighbourhood a candidate step has to stand out from, on each side.
STEP_NEIGHBOURHOOD = 25

# A step must beat the busier of its two sides by this much...
MIN_PROMINENCE = 3.0
# ...and be worth this share of the whole displayed fall. The second floor is
# relative rather than absolute because span varies 55x across the corpus: an
# absolute floor would fire on every wide sample and never on a tight one.
MIN_RELATIVE_STEP = 0.15


@dataclass(frozen=True)
class Step:
    """The largest local step in a profile, and whether it clears the floors."""

    rank: int
    delta: float
    prominence: float
    relative: float
    found: bool


@dataclass(frozen=True)
class Shape:
    """What the panel says about one profile at one displayed range."""

    n: int
    top: float
    last: float
    span: float
    half_span_rank: int
    step: Step | None
    headline: str
    detail: str


def _clean(scores) -> np.ndarray:
    """A finite, descending float array. Retrieval hands these back sorted."""
    arr = np.asarray(scores, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    return np.sort(arr)[::-1]


def local_steps(scores) -> np.ndarray:
    """Windowed drop at each rank, NaN where the window does not fit."""
    s = _clean(scores)
    n = s.size
    out = np.full(n, np.nan)
    if n < 2 * STEP_WINDOW:
        return out
    w = STEP_WINDOW
    # Cumulative sums rather than a Python loop: this runs on every control
    # change, and at n=1000 the loop was the only part of the panel anyone
    # could feel.
    c = np.concatenate([[0.0], np.cumsum(s)])
    i = np.arange(w - 1, n - w)
    before = (c[i + 1] - c[i + 1 - w]) / w
    after = (c[i + 1 + w] - c[i + 1]) / w
    out[i] = before - after
    return out


def largest_local_step(scores) -> Step | None:
    """The most prominent local step, and whether it is one at all.

    A candidate needs a **full neighbourhood on both sides**. Without that rule
    the answer is the first evaluable rank every time, because a decay is
    steepest where it starts - measured on this corpus, an earlier version of
    this function returned rank 5 for 36 of 40 samples and called a slope a
    step. Prominence is measured against the *busier* side, since a step is
    only a step if it stands out from what is on both sides of it.
    """
    s = _clean(scores)
    n = s.size
    d = local_steps(s)
    lo = STEP_WINDOW - 1 + STEP_NEIGHBOURHOOD
    hi = n - STEP_WINDOW - STEP_NEIGHBOURHOOD
    if hi <= lo:
        return None
    candidates = np.arange(lo, hi)
    values = d[candidates]
    if not np.isfinite(values).any():
        return None
    at = int(candidates[int(np.nanargmax(values))])

    left = d[at - STEP_NEIGHBOURHOOD:at - 1]
    right = d[at + 2:at + STEP_NEIGHBOURHOOD + 1]
    base = max(float(np.nanmedian(left)), float(np.nanmedian(right)))

    delta = float(d[at])
    total = span(s)
    prominence = delta / base if base > 0 else float("inf")
    relative = delta / total if total > 0 else 0.0
    return Step(
        rank=at + 1,
        delta=delta,
        prominence=prominence,
        relative=relative,
        found=bool(prominence >= MIN_PROMINENCE and relative >= MIN_RELATIVE_STEP),
    )


def span(scores) -> float:
    """How much cosine the displayed range covers, top match to last."""
    s = _clean(scores)
    if s.size < 2:
        return 0.0
    return float(s[0] - s[-1])


def half_span_rank(scores) -> int:
    """The rank by which the curve has given up half of its total fall.

    This is the shape statistic that replaced the breakpoint. It answers "are
    there 20 close analogs here or 200" without asserting a cutoff, and it
    discriminates: median 44 over 120 real queries, spread 10 to 136.
    """
    s = _clean(scores)
    if s.size < 2:
        return int(s.size)
    total = s[0] - s[-1]
    if total <= 0:
        return int(s.size)
    reached = np.nonzero((s[0] - s) >= total / 2.0)[0]
    return int(reached[0]) + 1 if reached.size else int(s.size)


def _fmt(value: float) -> str:
    """Cosine differences here are small; show enough digits to be a number."""
    if value >= 0.01:
        return f"{value:.3f}"
    if value >= 0.001:
        return f"{value:.4f}"
    return f"{value:.5f}"


def describe(scores, n: int | None = None) -> Shape:
    """Everything the panel states about one profile, as plain measurements."""
    s = _clean(scores)
    if n is not None:
        s = s[:max(2, int(n))]
    size = int(s.size)
    if size < 2:
        return Shape(n=size, top=float(s[0]) if size else 0.0,
                     last=float(s[-1]) if size else 0.0, span=0.0,
                     half_span_rank=size, step=None,
                     headline="Not enough matches to draw a profile.",
                     detail="A profile needs at least two ranked matches.")

    total = span(s)
    half = half_span_rank(s)
    step = largest_local_step(s)

    if total <= 0:
        headline = "Every match scores the same."
        detail = (f"All {size:,} displayed matches share one cosine value, "
                  f"{s[0]:.4f}. There is no decay to read.")
    elif step is not None and step.found:
        headline = f"A step down after rank {step.rank:,}."
        detail = (
            f"Across the top {size:,} the scores cover {_fmt(total)} of cosine, "
            f"from {s[0]:.4f} down to {s[-1]:.4f}. A local drop of {_fmt(step.delta)} "
            f"at rank {step.rank:,} stands {step.prominence:.1f} times above the drops "
            f"on either side of it, which is unusual for this corpus."
        )
    else:
        headline = "Similarity falls smoothly."
        detail = (
            f"Across the top {size:,} the scores cover {_fmt(total)} of cosine, "
            f"from {s[0]:.4f} down to {s[-1]:.4f}, and half of that fall has "
            f"happened by rank {half:,}. No rank separates a strong set from a "
            f"weak one."
        )

    return Shape(n=size, top=float(s[0]), last=float(s[-1]), span=total,
                 half_span_rank=half, step=step, headline=headline, detail=detail)
