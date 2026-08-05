"""What a cohort is, and whether pooling one means what the interface says.

Everything here runs against the synthetic fixture corpus and plain numpy, so
none of it needs the 963 MB memmap or the model checkpoint. The claims that do
need the real corpus - leave-one-out stability, the two nulls, the
stability-versus-k curve - are measured by `precompute/validate_cohorts.py`
instead, and the constants it produced are pinned here for shape rather than
re-measured.
"""

from __future__ import annotations

import numpy as np
import pytest

from bridge_rna import cohorts as C


# --- The facet registry ------------------------------------------------------


def test_study_is_pinned_and_cannot_be_removed():
    """Pooling across studies would average across the corpus's biggest batch
    boundary: same-study membership alone supplies 84% of a cohort's coherence.
    So the pinned facet has to survive every route into the definition, not just
    the one the UI takes."""
    assert C.PINNED_FACETS == ("study",)
    assert "study" in C.normalize_facets([])
    assert "study" in C.normalize_facets(["tissue"])
    assert "study" in C.normalize_facets(None)
    assert C.FACETS_BY_KEY["study"].reason, "a pinned facet must say why"


def test_normalize_facets_canonicalizes_order_and_drops_unknowns():
    got = C.normalize_facets(["spaceflight", "tissue", "not-a-column", "study"])
    assert got == ("study", "tissue", "spaceflight"), (
        "registry order, unknowns dropped")


def test_default_definition_is_the_curated_isatab_grouping():
    assert C.DEFAULT_FACETS == ("study", "tissue", "spaceflight")


def test_the_registry_is_exactly_the_curated_grouping_and_nothing_else():
    """Six further columns (sex, strain, genotype, habitat, duration, diet) were
    offered and removed. Every one of them could only make a cohort smaller, and
    size is what the measured stability curve is a function of, so a finer
    definition trades away the quantity the feature exists to buy. Pinned here
    so a facet cannot drift back onto the rail without this test being read."""
    assert tuple(f.key for f in C.FACETS) == ("study", "tissue", "spaceflight")
    assert C.DEFAULT_FACETS == tuple(f.key for f in C.FACETS), (
        "every remaining facet is on by default")


# --- Grouping ----------------------------------------------------------------


def test_every_cohort_lives_in_exactly_one_study():
    for c in C.build_cohorts():
        studies = {m.split("|", 1)[0] for m in c.members}
        assert studies == {c.study}, f"{c.cohort_id} spans {studies}"


def test_members_are_partitioned_without_overlap_or_loss():
    cohorts = C.build_cohorts()
    seen: list[str] = []
    for c in cohorts:
        seen.extend(c.members)
    assert len(seen) == len(set(seen)), "a sample landed in two cohorts"
    assert len(seen) == len(C.cohort_metadata()), "a sample landed in none"


def test_adding_a_facet_only_ever_splits(two_arm_study):
    """Narrowing the definition must refine the partition, never reshuffle it.

    Every cohort under the finer definition has to sit inside one cohort of the
    coarser one. If that ever fails, the grouping is not a facet intersection
    and the count under the chips is describing something else.

    It runs on `two_arm_study` rather than on the synthetic corpus, and the
    strictly-finer assertions are why. The synthetic corpus gives every study
    exactly one tissue and one arm, so *every* definition produces the same 12
    cohorts on it and the containment loop below passes without comparing
    anything. That was true of this test from the day it was written; it named a
    facet that has since been deleted, which would have been a second way to go
    vacuous, but it was already not testing the invariant it is named for.
    `two_arm_study` crosses two tissues with three arms, so each step of the
    chain genuinely divides.
    """
    chain = [["study"], ["study", "tissue"], ["study", "tissue", "spaceflight"]]
    partitions = [C.build_cohorts(keys, metadata=two_arm_study) for keys in chain]
    assert [len(p) for p in partitions] == [1, 2, 6], (
        "each step must actually divide, or the containment check is vacuous")

    for coarser, finer in zip(partitions, partitions[1:]):
        parent_of = {m: c.cohort_id for c in coarser for m in c.members}
        for fine in finer:
            parents = {parent_of[m] for m in fine.members}
            assert len(parents) == 1, f"{fine.cohort_id} straddles {parents}"


def test_every_sample_keeps_its_cohort_when_a_facet_is_dropped():
    """The corpus-wide half of the invariant above, on the real fixture corpus.

    Widening can only merge, so no sample may leave the company of a sample it
    was already grouped with. This is checkable even though the fixture corpus
    cannot express a split.
    """
    fine = {m: c.cohort_id for c in C.build_cohorts() for m in c.members}
    for wide in C.build_cohorts(["study"]):
        assert {fine[m] for m in wide.members}, wide.cohort_id
        assert all(m in fine for m in wide.members), (
            "widening dropped a sample that the default definition grouped")


def test_widening_to_study_alone_gives_one_cohort_per_study():
    cohorts = C.build_cohorts(["study"])
    studies = {c.study for c in cohorts}
    assert len(cohorts) == len(studies)
    assert all(c.label == "Whole study" for c in cohorts)


def test_cohorts_are_listed_largest_first():
    sizes = [c.size for c in C.build_cohorts()]
    assert sizes == sorted(sizes, reverse=True)


def test_find_cohort_round_trips_its_id():
    original = C.build_cohorts()[0]
    again = C.find_cohort(original.cohort_id)
    assert again is not None
    assert again.members == original.members


def test_a_cohort_id_from_another_definition_does_not_resolve():
    """The store carries an id, not a member list, so a cohort selected under
    one definition must not silently resolve under a different one."""
    wide = C.build_cohorts(["study"])[0]
    assert C.find_cohort(wide.cohort_id, facets=["study", "tissue"]) is None


def test_study_filter_matches_grouping_then_filtering():
    study = C.build_cohorts()[0].study
    filtered = C.build_cohorts(study=study)
    manual = [c for c in C.build_cohorts() if c.study == study]
    assert {c.cohort_id for c in filtered} == {c.cohort_id for c in manual}


# --- The estimator -----------------------------------------------------------


@pytest.fixture
def rng():
    return np.random.default_rng(20260805)


def test_pooled_vector_is_a_unit_vector(rng):
    rows = rng.normal(size=(7, 512)).astype(np.float32)
    assert np.linalg.norm(C.cohort_query_vector(rows)) == pytest.approx(1.0, abs=1e-6)


def test_pooling_one_sample_is_that_sample_normalized(rng):
    v = rng.normal(size=(1, 512)).astype(np.float32)
    expected = v[0] / np.linalg.norm(v[0])
    assert np.dot(C.cohort_query_vector(v), expected) == pytest.approx(1.0, abs=1e-6)


def test_pooling_does_not_depend_on_member_order(rng):
    rows = rng.normal(size=(6, 512)).astype(np.float32)
    shuffled = rows[rng.permutation(6)]
    assert np.dot(C.cohort_query_vector(rows),
                  C.cohort_query_vector(shuffled)) == pytest.approx(1.0, abs=1e-6)


def test_one_animal_one_vote_regardless_of_transcriptome_concentration(rng):
    """The reason each member is normalized before averaging.

    Raw embedding norms span 3.9x across the corpus and encode transcriptome
    concentration, not a nuisance scale (invariant 2). Averaging raw vectors
    would let the most concentrated member dominate. Scaling one member's norm
    by 10 must leave the pooled direction untouched.
    """
    rows = rng.normal(size=(5, 512)).astype(np.float32)
    loud = rows.copy()
    loud[0] *= 10.0
    assert np.dot(C.cohort_query_vector(rows),
                  C.cohort_query_vector(loud)) == pytest.approx(1.0, abs=1e-6)

    raw_mean = rows.mean(axis=0)
    loud_raw_mean = loud.mean(axis=0)
    tilted = float(np.dot(raw_mean / np.linalg.norm(raw_mean),
                          loud_raw_mean / np.linalg.norm(loud_raw_mean)))
    assert tilted < 0.99, ("the raw mean should have been dragged by the loud "
                           "member, or this test proves nothing")


def test_pooled_ranking_is_the_mean_of_the_members_own_cosines(rng):
    """The central claim of docs/cohort_pooling.md, checked directly.

    Ranking ARCHS4 by cosine to the spherical mean is identical to ranking by
    the unweighted average of the members' own cosine scores, because the
    pooled vector's norm does not depend on the sample being scored. If this
    ever breaks, "ask every animal, then average the votes" stops being what
    the feature does.
    """
    rows = rng.normal(size=(6, 512)).astype(np.float32)
    index = rng.normal(size=(400, 512)).astype(np.float32)
    index /= np.linalg.norm(index, axis=1, keepdims=True)

    pooled = index @ C.cohort_query_vector(rows)
    units = rows / np.linalg.norm(rows, axis=1, keepdims=True)
    averaged = (index @ units.T).mean(axis=1)

    assert np.array_equal(np.argsort(-pooled), np.argsort(-averaged))
    assert np.corrcoef(pooled, averaged)[0, 1] == pytest.approx(1.0, abs=1e-6)


def test_a_cohort_that_cancels_out_is_refused_rather_than_ranked(rng):
    """Two opposed vectors have no mean direction. A zero query vector would
    score every ARCHS4 sample identically and the result would still look like
    a ranking, which is the worst available failure mode."""
    v = rng.normal(size=512).astype(np.float32)
    with pytest.raises(ValueError, match="no mean direction"):
        C.cohort_query_vector(np.stack([v, -v]))


def test_empty_input_is_refused():
    with pytest.raises(ValueError):
        C.cohort_query_vector(np.zeros((0, 512), dtype=np.float32))


# --- Per-member outliers -----------------------------------------------------


def test_no_group_tightness_statistic_is_offered(rng):
    """`R̄`, the vMF resultant length, was measured over all 212 real cohorts and
    is near-constant at a median 0.9991 - no lower for a cohort of two than for
    one of thirty. It never separated a group worth trusting from one that was
    not, while sitting on the card looking like a grade, so it is gone. The
    per-member leave-one-out cosine below is a different kind of statistic and
    stays: it varies within a cohort and names an individual animal."""
    assert not hasattr(C, "resultant_length")
    g = C.cohort_geometry(["S|0", "S|1"], rng.normal(size=(2, 512)).astype(np.float32))
    assert not hasattr(g, "resultant")


def test_leave_one_out_scores_the_member_against_the_others(rng):
    """An outlier must score lowest, and it must be scored against a centroid it
    is not part of - otherwise it drags the reference towards itself and hides."""
    core = rng.normal(size=512).astype(np.float32)
    rows = np.stack([core + 0.01 * rng.normal(size=512) for _ in range(5)]
                    + [rng.normal(size=512)]).astype(np.float32)
    loo = C.leave_one_out_cosines(rows)
    assert len(loo) == 6
    assert int(np.argmin(loo)) == 5, "the planted outlier should score lowest"
    assert C.outlier_flags(loo)[5]
    assert not C.outlier_flags(loo)[:5].any()


def test_two_members_get_the_same_leave_one_out_score_twice(rng):
    rows = rng.normal(size=(2, 512)).astype(np.float32)
    loo = C.leave_one_out_cosines(rows)
    assert loo[0] == pytest.approx(loo[1], abs=1e-6)
    assert not C.outlier_flags(loo).any(), (
        "with two members there is no majority to deviate from, so an outlier "
        "flag would be an artifact")


def test_cohort_geometry_bundles_what_the_interface_reads(rng):
    rows = rng.normal(size=(4, 512)).astype(np.float32)
    members = [f"S|{i}" for i in range(4)]
    g = C.cohort_geometry(members, rows)
    assert g.size == 4
    assert g.members == tuple(members)
    assert len(g.loo_cosines) == 4 and len(g.outliers) == 4
    assert g.tier == C.size_tier(4)
    assert g.stability == C.expected_stability(4)


# --- Low N -------------------------------------------------------------------


def test_size_tiers():
    assert C.size_tier(1) == C.TIER_SINGLETON
    assert C.size_tier(C.LOW_N_THRESHOLD - 1) == C.TIER_LOW_N
    assert C.size_tier(C.LOW_N_THRESHOLD) == C.TIER_OK
    assert C.size_tier(38) == C.TIER_OK


def test_the_stability_curve_never_punishes_a_bigger_cohort():
    """A larger cohort must never be reported as less trustworthy than a smaller
    one. The measured per-size figures do invert - 5-6 scored 0.736 beside 7-9
    at 0.696 - which is why validate_cohorts.py merges adjacent buckets that
    invert before printing the curve. This pins the result of that."""
    sizes = sorted(C.STABILITY_BY_K)
    values = [C.STABILITY_BY_K[s] for s in sizes]
    assert values == sorted(values), f"non-monotone curve: {C.STABILITY_BY_K}"
    quoted = [C.expected_stability(k) for k in range(2, 45)]
    assert quoted == sorted(quoted)


def test_expected_stability_reads_the_bucket_floor():
    assert C.expected_stability(7) == C.STABILITY_BY_K[5]
    assert C.expected_stability(100) == C.STABILITY_BY_K[max(C.STABILITY_BY_K)]


def test_low_n_threshold_is_where_the_curve_reaches_its_claim():
    """The threshold is a measurement, not a preference: it is the first bucket
    whose measured stability reaches 0.70."""
    reached = min(k for k, v in C.STABILITY_BY_K.items() if v >= 0.70)
    assert C.LOW_N_THRESHOLD == reached


def test_pooling_beats_a_single_sample_at_every_offered_size():
    for k in sorted(C.STABILITY_BY_K):
        assert C.expected_stability(k) > C.SINGLE_SAMPLE_STABILITY


# --- Comparison --------------------------------------------------------------


@pytest.fixture
def two_arm_study():
    """One study shaped like the real OSD-137: two tissues, three arms.

    The synthetic corpus assigns tissue by cluster and arm by row index, so it
    happens to produce no two cohorts that are one facet apart - which left the
    whole comparison path untested. This frame is written to have them, because
    the real corpus does: OSD-137 alone carries Liver in Basal, Ground and
    Space Flight arms.
    """
    import pandas as pd

    rows = []
    for tissue in ("Liver", "Soleus"):
        for arm in ("Space Flight", "Ground Control", "Basal Control"):
            for rep in range(3):
                rows.append({
                    "sample_key": f"OSD-137|{tissue[:3]}_{arm[:3]}_{rep}",
                    "study": "OSD-137", "tissue": tissue, "spaceflight": arm,
                })
    return pd.DataFrame(rows)


def test_siblings_differ_in_exactly_one_facet_and_share_the_study(two_arm_study):
    cohorts = C.build_cohorts(metadata=two_arm_study)
    assert len(cohorts) == 6, "two tissues x three arms"
    tested = 0
    for cohort in cohorts:
        for sib in C.sibling_cohorts(cohort, metadata=two_arm_study):
            tested += 1
            assert sib.study == cohort.study
            differing = [k for k in cohort.facets
                         if sib.values[k] != cohort.values[k]]
            assert len(differing) == 1
            assert C.contrast_facet(cohort, sib) == differing[0]
            assert sib.size >= C.MIN_COHORT_SIZE
    assert tested, "no comparable pair was produced"


def test_siblings_are_offered_along_each_axis_but_never_along_two(two_arm_study):
    """What "one facet apart" buys, stated concretely.

    Liver/Space Flight can be compared against the other two Liver arms, where
    the contrast is the arm, and against Soleus/Space Flight, where the contrast
    is the tissue. Both are attributable, so both are offered and the UI names
    which facet differs. Soleus/Ground Control is not offered: it differs in
    tissue *and* arm, so its overlap number could not be attributed to either.
    """
    liver_flight = next(c for c in C.build_cohorts(metadata=two_arm_study)
                        if c.values["tissue"] == "Liver"
                        and c.values["spaceflight"] == "Space Flight")
    siblings = C.sibling_cohorts(liver_flight, metadata=two_arm_study)

    by_facet: dict[str, set] = {}
    for s in siblings:
        by_facet.setdefault(C.contrast_facet(liver_flight, s), set()).add(s.label)
    assert set(by_facet) == {"spaceflight", "tissue"}
    assert by_facet["spaceflight"] == {"Liver · Ground Control",
                                       "Liver · Basal Control"}
    assert by_facet["tissue"] == {"Soleus · Space Flight"}

    offered = {s.label for s in siblings}
    assert "Soleus · Ground Control" not in offered
    assert "Soleus · Basal Control" not in offered


def test_a_cohort_is_never_its_own_sibling(two_arm_study):
    for cohort in C.build_cohorts(metadata=two_arm_study):
        assert cohort.cohort_id not in {
            s.cohort_id for s in C.sibling_cohorts(cohort, metadata=two_arm_study)}


def test_sibling_relation_is_symmetric(two_arm_study):
    cohorts = C.build_cohorts(metadata=two_arm_study)
    by_id = {c.cohort_id: c for c in cohorts}
    for cohort in cohorts:
        for sib in C.sibling_cohorts(cohort, metadata=two_arm_study):
            back = {s.cohort_id for s in
                    C.sibling_cohorts(by_id[sib.cohort_id], metadata=two_arm_study)}
            assert cohort.cohort_id in back


def test_the_fixture_corpus_sibling_walk_stays_consistent():
    """Whatever the synthetic corpus does produce must still obey the rules,
    even if it produces nothing."""
    for cohort in C.build_cohorts():
        for sib in C.sibling_cohorts(cohort):
            differing = [k for k in cohort.facets
                         if sib.values[k] != cohort.values[k]]
            assert len(differing) == 1 and sib.study == cohort.study


# --- Labelling ---------------------------------------------------------------


def test_a_cohort_never_labels_itself_with_one_members_name():
    """The banner and the query node must describe the group. Announcing a
    pooled result under one animal's name is the same class of error as the
    status banner that announced cached results as subprocess output."""
    for c in C.build_cohorts():
        assert c.cohort_id not in c.members
        assert c.describe().startswith(c.study)
        for member in c.members:
            assert member.split("|", 1)[-1] not in c.label


def test_unknown_facet_values_are_named_rather_than_blank():
    assert C.facet_value({"tissue": ""}, "tissue") == "Unknown"
    assert C.facet_value({"tissue": "nan"}, "tissue") == "Unknown"
    assert C.facet_value({"duration": "37 {day}"}, "duration") == "37 day"
