"""The ranked similarity profile's shape statistics.

Every test here runs on synthetic arrays, because `bridge_rna.profile` opens no
artifact by design. The corpus measurements that set its constants are in
`docs/ranked-similarity-profile.md`; what is pinned here is that the estimator
still behaves the way those measurements say it does.
"""

from __future__ import annotations

import numpy as np
import pytest

from bridge_rna import profile as P


# --- fixtures, at the corpus's own scale ------------------------------------
# Real spans over the top 250 run 0.00027 to 0.01503, so a synthetic profile
# built at 0.9 to 0.5 would not exercise anything this module actually meets.

def smooth(n: int = 250, top: float = 0.9990, total: float = 0.0030) -> np.ndarray:
    ranks = np.arange(1, n + 1)
    curve = np.log1p(ranks / 40.0)
    curve = (curve - curve[0]) / (curve[-1] - curve[0])   # exactly 0 to 1
    return top - total * curve


def stepped(n: int = 250, at: int = 60, share: float = 0.30) -> np.ndarray:
    """A smooth decay with a genuine step worth `share` of the total fall."""
    base = smooth(n)
    total = base[0] - base[-1]
    out = base - (np.arange(1, n + 1) > at) * (total * share)
    return np.minimum.accumulate(out)


# --- the shape of the array -------------------------------------------------

def test_scores_are_sorted_descending_whatever_order_they_arrive_in():
    shuffled = np.array([0.991, 0.9995, 0.995, 0.9990])
    assert P.describe(shuffled).top == pytest.approx(0.9995)
    assert P.describe(shuffled).last == pytest.approx(0.991)


def test_non_finite_scores_are_dropped_rather_than_poisoning_the_span():
    with_nan = np.array([0.999, np.nan, 0.998, np.inf, 0.997])
    assert P.span(with_nan) == pytest.approx(0.002)


def test_a_profile_needs_two_matches_and_says_so_rather_than_raising():
    shape = P.describe([0.999])
    assert shape.step is None
    assert "at least two" in shape.detail


# --- span and half-span -----------------------------------------------------

def test_span_is_the_fall_from_the_top_match_to_the_last_displayed():
    assert P.span(smooth(250, top=0.999, total=0.003)) == pytest.approx(0.003, abs=1e-9)


def test_span_is_measured_over_the_displayed_range_not_the_whole_scan():
    full = smooth(1000)
    assert P.describe(full, n=100).span < P.describe(full, n=1000).span


def test_half_span_rank_lands_where_half_the_fall_has_happened():
    s = smooth(250)
    r = P.half_span_rank(s)
    fallen = s[0] - s[r - 1]
    assert fallen >= (s[0] - s[-1]) / 2.0
    assert s[0] - s[r - 2] < (s[0] - s[-1]) / 2.0


def test_half_span_rank_is_early_for_a_tight_core_and_late_for_an_even_decline():
    linear = np.linspace(0.999, 0.996, 250)          # perfectly even
    front_loaded = stepped(250, at=12, share=0.6)     # most of the fall by rank 12
    assert P.half_span_rank(front_loaded) < P.half_span_rank(linear)


def test_a_flat_profile_reports_no_span_and_says_there_is_no_decay():
    shape = P.describe(np.full(250, 0.9991))
    assert shape.span == 0.0
    assert "no decay" in shape.detail


# --- the step estimator -----------------------------------------------------

def test_a_real_step_is_found_at_the_rank_it_was_planted():
    step = P.largest_local_step(stepped(250, at=60, share=0.30))
    assert step is not None and step.found
    assert step.rank == 60
    assert step.prominence >= P.MIN_PROMINENCE
    assert step.relative >= P.MIN_RELATIVE_STEP


def test_a_smooth_decay_reports_a_candidate_but_never_clears_the_floors():
    """The corpus case. 0 of 120 real queries cleared this at any range."""
    step = P.largest_local_step(smooth(250))
    assert step is not None
    assert not step.found
    assert step.prominence < P.MIN_PROMINENCE


def test_a_step_too_small_to_matter_is_rejected_on_the_relative_floor():
    step = P.largest_local_step(stepped(250, at=60, share=0.05))
    assert step is not None and not step.found
    assert step.relative < P.MIN_RELATIVE_STEP


def test_the_leading_edge_can_never_be_reported_as_a_step():
    """A decay is steepest where it starts, and that is a slope, not a step.

    An earlier version without the two-sided neighbourhood rule returned rank 5
    for 36 of 40 real samples, which is this failure exactly.
    """
    floor = P.STEP_WINDOW - 1 + P.STEP_NEIGHBOURHOOD
    for n in (100, 250, 1000):
        step = P.largest_local_step(smooth(n))
        assert step is not None
        assert step.rank > floor, f"leading edge reported at n={n}"


def test_prominence_is_measured_against_the_busier_side():
    """A drop with a steep run on one side of it is not standing out from it."""
    n = 250
    ranks = np.arange(1, n + 1)
    # Flat, then a fast run, then a single drop sitting right at the run's end.
    s = np.where(ranks <= 120, 0.9990, 0.9990 - (ranks - 120) * 2e-5)
    s = np.minimum.accumulate(s - (ranks > 180) * 0.00012)
    step = P.largest_local_step(s)
    assert step is not None
    one_sided = step.delta / float(np.nanmedian(P.local_steps(s)[205:230]))
    assert step.prominence <= one_sided


def test_too_few_matches_for_a_two_sided_neighbourhood_returns_nothing():
    assert P.largest_local_step(smooth(40)) is None


def test_local_steps_leaves_the_unevaluable_ends_as_nan():
    d = P.local_steps(smooth(250))
    assert np.isnan(d[:P.STEP_WINDOW - 1]).all()
    assert np.isnan(d[-P.STEP_WINDOW:]).all()
    assert np.isfinite(d[P.STEP_WINDOW - 1:-P.STEP_WINDOW]).all()


# --- what the panel says ----------------------------------------------------

def test_the_smooth_headline_never_claims_a_transition():
    shape = P.describe(smooth(250))
    assert shape.headline == "Similarity falls smoothly."
    assert "No rank separates" in shape.detail
    for banned in ("cutoff", "threshold", "recommended"):
        assert banned not in shape.detail.lower()


def test_the_stepped_headline_names_the_rank_and_calls_it_unusual():
    shape = P.describe(stepped(250, at=60, share=0.30))
    assert "rank 60" in shape.headline
    assert "unusual for this corpus" in shape.detail


def test_the_detail_always_states_the_span_because_the_axis_is_fitted():
    """A fitted axis magnifies, so the number that stops a misread is the span."""
    for scores in (smooth(250), smooth(250, total=0.0003), stepped(250)):
        assert "of cosine" in P.describe(scores).detail


def test_describe_honours_the_requested_range():
    full = smooth(1000)
    assert P.describe(full, n=100).n == 100
    assert P.describe(full, n=1000).n == 1000


def test_the_default_range_is_one_of_the_offered_ranges():
    assert P.DEFAULT_RANGE in P.RANGE_CHOICES
    assert max(P.RANGE_CHOICES) == P.PROFILE_DEPTH
