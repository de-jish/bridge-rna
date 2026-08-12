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
    """The central claim of docs/design-notes.md#cohort-pooling, checked directly.

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
    assert not hasattr(g, "stability"), (
        "geometry describes the group, not the result. Stability is measured "
        "during the search now, by retrieval.run_cohort_retrieval.")


# --- Low N -------------------------------------------------------------------


def test_size_tiers():
    assert C.size_tier(1) == C.TIER_SINGLETON
    assert C.size_tier(C.LOW_N_THRESHOLD - 1) == C.TIER_LOW_N
    assert C.size_tier(C.LOW_N_THRESHOLD) == C.TIER_OK
    assert C.size_tier(38) == C.TIER_OK


def test_the_precomputed_stability_curve_is_gone():
    """`STABILITY_BY_K` was a population average printed beside one cohort's
    name, and it was read as a property of that cohort. The spread inside a
    bucket is most of the range: measured live, a cohort of 7 scored 0.316 and
    one of 6 scored 0.849, and both were quoted 0.72.

    It is deleted rather than left unused, the way `resultant_length` was.
    Reintroducing it needs an argument that answers docs/design-notes.md#live-stability,
    which is what this test exists to make someone read."""
    for name in ("STABILITY_BY_K", "expected_stability", "SINGLE_SAMPLE_STABILITY"):
        assert not hasattr(C, name), (
            f"{name} is back. The measured replacement is StabilityMeasurement; "
            "see docs/design-notes.md#live-stability section 5.")


def test_the_caution_floor_is_the_threshold_that_set_low_n():
    """0.70 picked LOW_N_THRESHOLD as the first size bucket to reach it. The
    floor applies that same threshold to the measurement itself rather than to
    size standing in for it, so the two must not drift apart."""
    assert C.STABILITY_FLOOR == 0.70
    assert 0.0 < C.STABILITY_FLOOR < 1.0


# --- Result stability, measured rather than predicted -------------------------
#
# The set arithmetic lives here because it is arithmetic; the scan that feeds it
# is exercised against the fixture memmap in test_retrieval.py, and the
# corpus-scale claim is `precompute/validate_cohorts.py` check 2.


def test_agreement_is_jaccard_and_agrees_with_the_validators_definition():
    """One statistic in both places it appears. The number the inspector prints
    after a search and the number validate_cohorts.py computes over all 212
    cohorts have to be the same thing, or neither can be read against the
    other."""
    assert C.top_k_agreement([1, 2, 3], [1, 2, 3]) == 1.0
    assert C.top_k_agreement([1, 2, 3], [4, 5, 6]) == 0.0
    # Two of three shared: intersection 2, union 4.
    assert C.top_k_agreement([1, 2, 3], [2, 3, 9]) == pytest.approx(0.5)
    assert C.top_k_agreement([1, 2], [2, 1]) == 1.0, "order is not part of it"
    assert C.top_k_agreement([], []) == 0.0


def test_agreement_is_symmetric_and_bounded(rng):
    for _ in range(20):
        a = rng.choice(50, size=8, replace=False)
        b = rng.choice(50, size=8, replace=False)
        got = C.top_k_agreement(a, b)
        assert 0.0 <= got <= 1.0
        assert got == pytest.approx(C.top_k_agreement(b, a))


def test_leave_one_out_vectors_are_the_pools_each_absence_would_produce(rng):
    rows = rng.normal(size=(5, 512)).astype(np.float32)
    loo = C.leave_one_out_vectors(rows)
    assert loo.shape == (5, 512)
    for i in range(5):
        expected = C.cohort_query_vector(np.delete(rows, i, axis=0))
        assert np.allclose(loo[i], expected, atol=1e-6)
    assert np.allclose(np.linalg.norm(loo, axis=1), 1.0, atol=1e-5), (
        "each one is a query vector, so each one is a unit direction")


def test_a_cohort_with_nothing_to_leave_out_produces_no_variants(rng):
    rows = rng.normal(size=(1, 512)).astype(np.float32)
    assert C.leave_one_out_vectors(rows).shape[0] == 0
    assert C.measure_stability(["only"], [1, 2], [], [[1, 2]], depth=2) is None


def test_measure_stability_reports_the_mean_over_droppable_members():
    """Three members. Dropping the first leaves the list alone; dropping the
    third replaces two of three hits."""
    m = C.measure_stability(
        members=["a", "b", "c"],
        pooled_top=[1, 2, 3],
        loo_tops=[[1, 2, 3], [1, 2, 9], [1, 8, 9]],
        member_tops=[[1, 2, 3], [1, 2, 3], [7, 8, 9]],
        depth=3,
    )
    assert m.size == 3 and m.depth == 3
    assert m.per_member == pytest.approx((1.0, 0.5, 0.2))
    assert m.pooled == pytest.approx((1.0 + 0.5 + 0.2) / 3)
    # Members a and b agree completely; neither agrees with c at all.
    assert m.single_sample == pytest.approx(1.0 / 3)
    assert m.gain == pytest.approx(m.pooled / m.single_sample)
    assert m.weakest_member == ("c", pytest.approx(0.2))


def test_stability_is_measured_at_the_depth_on_screen():
    """The offline curve was fixed at top-5 because it had to pick one. The list
    a reader is looking at is `topk` deep, and that is the list whose stability
    they are being told about."""
    shallow = C.measure_stability(
        members=["a", "b"], pooled_top=[1, 2, 3, 4], loo_tops=[[1, 2, 8, 9]] * 2,
        member_tops=[[1, 2, 3, 4], [1, 2, 3, 4]], depth=2)
    deep = C.measure_stability(
        members=["a", "b"], pooled_top=[1, 2, 3, 4], loo_tops=[[1, 2, 8, 9]] * 2,
        member_tops=[[1, 2, 3, 4], [1, 2, 3, 4]], depth=4)
    assert shallow.depth == 2 and shallow.pooled == pytest.approx(1.0), (
        "the top 2 are untouched")
    assert deep.depth == 4 and deep.pooled == pytest.approx(1.0 / 3)


def test_a_zero_baseline_is_reported_rather_than_divided_by():
    """One real cohort of four measured a single-sample baseline of exactly
    0.000: no two of its members share a hit. Dividing by it printed a
    nine-digit gain."""
    m = C.measure_stability(
        members=["a", "b"], pooled_top=[1, 2], loo_tops=[[1, 2], [1, 2]],
        member_tops=[[1, 2], [3, 4]], depth=2)
    assert m.single_sample == 0.0
    assert m.gain is None
    assert m.as_dict()["gain"] is None


def test_no_member_is_named_when_they_all_move_it_equally():
    m = C.measure_stability(
        members=["a", "b", "c"], pooled_top=[1, 2], loo_tops=[[1, 9]] * 3,
        member_tops=[[1, 2]] * 3, depth=2)
    assert m.weakest_member is None, (
        "naming one of three identical members would be arbitrary")


def test_the_caution_flag_follows_the_measurement_not_the_size():
    strong = C.measure_stability(
        members=["a", "b"], pooled_top=[1, 2], loo_tops=[[1, 2], [1, 2]],
        member_tops=[[1, 2], [1, 2]], depth=2)
    weak = C.measure_stability(
        members=[f"m{i}" for i in range(6)], pooled_top=[1, 2],
        loo_tops=[[8, 9]] * 6, member_tops=[[1, 2]] * 6, depth=2)
    assert strong.pooled == 1.0 and not strong.is_low, (
        "a pair whose result does not move is not flagged for being a pair")
    assert weak.pooled == 0.0 and weak.is_low, (
        "and six samples do not exempt a result that moves completely")


def test_the_measurement_survives_the_json_store():
    """It travels in `hits-store`, which is JSON, so every field has to be a
    plain type - and the panel is rebuilt from the dict after the router
    destroys the view."""
    import json

    m = C.measure_stability(
        members=["a", "b", "c"], pooled_top=[1, 2, 3],
        loo_tops=[[1, 2, 3], [1, 2, 9], [1, 8, 9]],
        member_tops=[[1, 2, 3], [1, 2, 3], [7, 8, 9]], depth=3)
    d = json.loads(json.dumps(m.as_dict()))
    assert d["size"] == 3 and d["depth"] == 3
    assert d["pooled"] == pytest.approx(m.pooled)
    assert d["single_sample"] == pytest.approx(m.single_sample)
    assert d["weakest_member"] == "c"
    assert d["weakest_value"] == pytest.approx(0.2)
    # 0.57 mean, under the floor, and it has to survive as a plain bool rather
    # than as numpy's - json.dumps refuses np.bool_ outright.
    assert d["is_low"] is True and d["is_low"] == m.is_low


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


# --- Two pooled queries, two descriptions ------------------------------------
#
# A comparison runs a second independent pooled query. Until 2026-08-06 the
# interface described the first and gave the second only a colour, in the
# network figure and on the map. `STABILITY_BY_K` is a function of size, so the
# arm whose size was off screen is exactly the one that decides how much of the
# reported overlap to believe.


def _series(component) -> str:
    """Flatten a Dash component tree to its visible text."""
    from dash.development.base_component import Component

    if isinstance(component, str):
        return component
    if isinstance(component, (list, tuple)):
        return " ".join(_series(c) for c in component)
    if isinstance(component, Component):
        children = getattr(component, "children", None)
        return _series(children) if children is not None else ""
    return ""


def _classes(component) -> set[str]:
    """Every CSS class anywhere in a Dash component tree.

    `_series` deliberately flattens to *visible text*, so asking it whether a
    meter was drawn always answers no. Encodings that are carried by class
    rather than by words - the meter, the amber flag, the role rules - have to
    be asserted on separately or the assertion passes vacuously.
    """
    from dash.development.base_component import Component

    out: set[str] = set()
    if isinstance(component, (list, tuple)):
        for c in component:
            out |= _classes(c)
    elif isinstance(component, Component):
        out.update(str(getattr(component, "className", "") or "").split())
        children = getattr(component, "children", None)
        if children is not None:
            out |= _classes(children)
    return out


def _one_cohort():
    """Any real cohort from the fixture corpus, with its geometry."""
    import numpy as np

    cohort = next(c for c in C.build_cohorts() if c.size >= C.MIN_COHORT_SIZE)
    rng = np.random.default_rng(0)
    rows = rng.normal(size=(cohort.size, 8)).astype("float32")
    return cohort, C.cohort_geometry(list(cohort.members), rows)


def test_a_lone_cohort_card_carries_no_role_letter():
    """With no second arm on screen there is nothing for a letter to tell it
    apart from, so adding one would be chrome for a distinction that does not
    exist yet."""
    from bridge_rna.panels import build_cohort_card

    cohort, geometry = _one_cohort()
    card = build_cohort_card(cohort, geometry)
    assert "Cohort A" not in _series(card)
    assert "is-a" not in str(card.className)


def test_each_arm_of_a_comparison_names_itself_and_its_hue():
    """The role line and the left rule are the only thing binding a card to the
    star it describes in the network figure and to the mark on the map."""
    from bridge_rna.panels import build_cohort_card

    cohort, geometry = _one_cohort()
    a = build_cohort_card(cohort, geometry, role="a")
    b = build_cohort_card(cohort, geometry, role="b",
                          contrast="spaceflight arm")

    assert "Cohort A" in _series(a) and "is-a" in a.className
    assert "Cohort B" in _series(b) and "is-b" in b.className
    # The facet belongs to the pair, so it is stated once, on the second card.
    assert "differs by spaceflight arm" in _series(b)
    assert "differs by" not in _series(a)
    # Both still answer the question the card exists for.
    for card in (a, b):
        assert "samples pooled into one query" in _series(card)


def test_the_rail_card_says_nothing_about_result_stability():
    """The rail speaks before the search, when the only thing known about the
    result is how many samples are going into it. It used to quote a stability
    figure looked up from a curve by cohort size, which is a population average
    printed beside one cohort's name; the number is measured during the search
    now and reported on the right afterwards."""
    from bridge_rna.panels import build_cohort_card

    cohort, geometry = _one_cohort()
    for card in (build_cohort_card(cohort, geometry),
                 build_cohort_card(cohort, geometry, role="a"),
                 build_cohort_card(cohort, geometry, role="b", contrast="tissue")):
        text, classes = _series(card), _classes(card)
        assert "stability" not in text.lower()
        assert not {"cohort-meter", "cohort-stat"} & classes, (
            "no meter and no stat block, because there is no number to show")
        assert "cohort-flag" not in classes
        assert "samples pooled into one query" in text


# --- The stability panel ------------------------------------------------------


def _measured(pooled=0.8, single=0.2, size=6, depth=10, weakest="OSD-1|m3"):
    return {"depth": depth, "size": size, "pooled": pooled,
            "single_sample": single, "gain": None if not single else pooled / single,
            "is_low": pooled < C.STABILITY_FLOOR,
            "weakest_member": weakest, "weakest_value": 0.4}


def test_no_stability_panel_until_a_cohort_has_been_searched():
    """A single sample and an uploaded file have nothing to leave out, so there
    is no leave-one-out stability for them and an empty panel would be the same
    empty promise the map link avoids before a search."""
    from bridge_rna.panels import build_stability_panel

    for payload in (None, {}, {"mode": "cached", "hits": [{"gsm": "GSM1"}]},
                    {"mode": "uploaded"}, {"mode": "cohort", "stability": None}):
        children, style = build_stability_panel(payload)
        assert children == []
        assert style == {"display": "none"}


def test_the_panel_reports_the_measured_number_and_its_scale():
    from bridge_rna.panels import build_stability_panel

    children, style = build_stability_panel({
        "mode": "cohort", "stability": _measured(),
        "query": {"cohort_label": "Liver · Space Flight"}})
    text = _series(children)
    assert style == {}
    assert "Result stability" in text
    assert "0.80" in text, "the headline is quoted, not reduced to a word"
    assert "Measured on this query" in text
    assert "10 hits" in text, "the depth measured at is stated, once"
    assert "of 10" in text, "and rides the number, so the share has a unit"
    assert "6 pooled" in text, "the block says how many went in"
    assert "4.0x" in text, "and what pooling bought, measured on this cohort"
    assert "m3" in text, "and names the member that moves it most"
    # The accession is dropped from that name: every member of a cohort shares
    # it, and carrying it wrapped the name onto a second line for nothing.
    assert "OSD-1|m3" not in text
    # "Result stability" is the panel's heading and appears exactly once. It
    # used to label every block as well, which pushed cohort B off screen.
    assert text.lower().count("result stability") == 1


def test_a_zero_baseline_is_described_rather_than_divided_by():
    from bridge_rna.panels import build_stability_panel

    children, _ = build_stability_panel({
        "mode": "cohort", "stability": _measured(single=0.0)})
    text = _series(children)
    assert "agree on a hit alone" in text
    assert "x gain" not in text and "inf" not in text.lower()


def test_a_baseline_too_small_for_two_decimals_is_not_printed_as_zero():
    """The baseline and the gain it produced share one sentence, so rounding the
    baseline to 0.00 beside a 340x ratio makes the sentence contradict itself.
    Reachable: a large cohort's members can agree on almost nothing alone, which
    is exactly the case where the gain is most worth stating."""
    from bridge_rna.panels import build_stability_panel

    children, _ = build_stability_panel({
        "mode": "cohort", "stability": _measured(pooled=0.85, single=0.0025)})
    text = _series(children)
    assert "overlaps another by 0.00," not in text
    assert "overlaps another by 0.003," in text
    assert "340.0x" in text


def test_the_caution_is_amber_and_fires_on_the_measurement():
    from bridge_rna.panels import build_stability_panel

    steady, _ = build_stability_panel({"mode": "cohort",
                                       "stability": _measured(pooled=0.9)})
    shaky, _ = build_stability_panel({"mode": "cohort",
                                      "stability": _measured(pooled=0.3)})
    assert "cohort-flag" not in _classes(steady)
    assert "is-low" not in _classes(steady)
    shaky_text, shaky_classes = _series(shaky), _classes(shaky)
    assert "cohort-flag" in shaky_classes
    assert "neighbourhood" in shaky_text
    # The sentence quotes the constant that triggered it, so the two cannot drift.
    assert f"{round(C.STABILITY_FLOOR * 100)}%" in shaky_text
    assert "is-low" in shaky_classes, "the meter goes amber with it"


def test_a_comparison_measures_both_arms_separately():
    """An overlap of 0.25 between two arms measuring 0.86 means something quite
    different from the same 0.25 between one at 0.86 and one at 0.31, and the
    number that decides which it is has to be on screen for both."""
    from bridge_rna.panels import build_stability_panel

    children, _ = build_stability_panel({
        "mode": "cohort",
        "stability": _measured(pooled=0.86, size=12, weakest="OSD-1|a2"),
        "query": {"cohort_label": "Liver · Space Flight"},
        "comparison": {
            "facet": "spaceflight arm",
            "query_b": {"cohort_label": "Liver · Ground Control"},
            "stability": _measured(pooled=0.31, size=2, weakest="OSD-1|b1"),
        },
    })
    text = _series(children)
    assert "Cohort A" in text and "Cohort B" in text
    assert "0.86" in text and "0.31" in text
    # Each arm is named beside its own mark, because cohort B's hex cannot agree
    # across the retrieval network and the map.
    assert "Liver · Space Flight" in text and "Liver · Ground Control" in text
    # The facet the two differ in is a fact about the pair, so it is stated with
    # the heading and the subtitle rather than hanging off cohort B's letter,
    # where it made B's name start a line below A's once the two went side by
    # side. Once, either way.
    assert "differ by spaceflight arm" in text
    assert text.count("spaceflight arm") == 1
    classes = _classes(children)
    assert "is-a" in classes and "is-b" in classes
    # Cohort B is the shakier arm here and says so on its own block.
    assert "cohort-flag" in classes


def _find(node, want: str):
    """Every node in the tree carrying `want` among its classes, in tree order.

    Tree order matters: these tests assert which arm carries which row, so a
    traversal that returned cohort B first would pass on the wrong evidence.
    """
    found = []

    def walk(n):
        if n is None:
            return
        if isinstance(n, (list, tuple)):
            for child in n:
                walk(child)
            return
        if want in (getattr(n, "className", "") or "").split():
            found.append(n)
        walk(getattr(n, "children", None))

    walk(node)
    return found


def test_two_arms_are_laid_out_as_an_even_pair_and_one_arm_is_not():
    """Stacked, the two arms got visibly unequal treatment: cohort A rendered
    complete and cohort B's last row was clipped by the panel's own fold at every
    viewport measured, 7.8px at 1680x1050 and 65.6px at 1280x800. `is-pair` is
    what makes them two even columns, and a lone cohort must not get it - a
    single block in a two-column grid is a half-empty table."""
    from bridge_rna.panels import build_stability_panel

    paired, _ = build_stability_panel({
        "mode": "cohort", "stability": _measured(),
        "query": {"cohort_label": "Liver · Space Flight"},
        "comparison": {"facet": "spaceflight arm",
                       "query_b": {"cohort_label": "Liver · Ground Control"},
                       "stability": _measured(pooled=0.7)}})
    pair = _find(paired, "stability-pair")
    assert len(pair) == 1, "both arms live in one grid, so their rows can align"
    assert "is-pair" in (pair[0].className or "")
    assert len(_find(pair[0], "stability-cohort")) == 2

    alone, _ = build_stability_panel({
        "mode": "cohort", "stability": _measured(),
        "query": {"cohort_label": "Liver · Space Flight"}})
    solo = _find(alone, "stability-pair")
    assert len(solo) == 1
    assert "is-pair" not in (solo[0].className or "")
    assert len(_find(solo[0], "stability-cohort")) == 1


def test_every_row_of_an_arm_is_addressable_so_the_columns_can_align():
    """`subgrid` aligns the two arms row by row, and the rows are assigned by
    class rather than by child order. Either of the last two can be missing from
    either arm - a cohort whose members all matter equally names none, and only a
    shaky arm is flagged - so counting children would let cohort B's flag land in
    the row holding cohort A's member name."""
    from bridge_rna.panels import build_stability_panel

    children, _ = build_stability_panel({
        "mode": "cohort",
        "stability": _measured(pooled=0.86, weakest="OSD-1|a2"),
        "query": {"cohort_label": "Liver · Space Flight"},
        "comparison": {
            "facet": "spaceflight arm",
            "query_b": {"cohort_label": "Liver · Ground Control"},
            # No weakest member, and low enough to be flagged: the mirror image
            # of arm A's rows.
            "stability": _measured(pooled=0.31, weakest=None)}})
    arms = _find(children, "stability-cohort")
    assert len(arms) == 2
    rows_a = {c for row in ("stability-name", "cohort-stat", "stability-weakest",
                            "cohort-flag") for c in [row] if _find(arms[0], row)}
    rows_b = {c for row in ("stability-name", "cohort-stat", "stability-weakest",
                            "cohort-flag") for c in [row] if _find(arms[1], row)}
    # The weakest row is on both arms - arm B's says there is no such member
    # rather than going missing. The flag is on the shaky arm only, and stays
    # that way: the counterpart badge an equalized flag would need is a pass mark
    # for a healthy cohort, which is the grade `R̄` was deleted for being.
    assert rows_a == {"stability-name", "cohort-stat", "stability-weakest"}
    assert rows_b == {"stability-name", "cohort-stat", "stability-weakest",
                      "cohort-flag"}
    # Each row is a direct child of its arm, which is what `grid-row` needs: a
    # row nested one level deeper would not be a grid item at all.
    for arm in arms:
        direct = {c for kid in (arm.children or [])
                  for c in (getattr(kid, "className", "") or "").split()}
        assert {"stability-name", "cohort-stat"} <= direct


def test_every_row_the_pair_grid_places_has_a_rule_that_places_it():
    """A fifth row added to an arm would shear the two columns, silently.

    `subgrid` aligns only the rows the parent declares and the child places. A
    direct child with no `grid-row` lands in an implicit track, which the other
    column does not have unless it emits the same row - so one arm's flag could
    end up beside the other arm's member name, which is the failure that
    addressing rows by class exists to prevent.

    Nothing else guards this: `test_app.py`'s stylesheet check filters to `bm-`
    and `app-` prefixes, so no `stability-*` class is covered by it.
    """
    import re
    from pathlib import Path

    from bridge_rna.panels import build_stability_panel

    css = (Path(__file__).resolve().parent.parent
           / "assets" / "retrieve.css").read_text()

    placed = {}
    for m in re.finditer(
            r"\.stability-pair\.is-pair\s*>\s*\.stability-cohort\s*>\s*"
            r"\.([a-z-]+)\s*\{[^}]*grid-row:\s*(\d+)", css):
        placed[m.group(1)] = int(m.group(2))
    assert placed, "no row is placed at all; the columns cannot align"

    # The mirror-image payload, so both optional rows are represented: arm A
    # names a member and arm B is flagged instead.
    children, _ = build_stability_panel({
        "mode": "cohort",
        "stability": _measured(pooled=0.86, weakest="OSD-1|a2"),
        "query": {"cohort_label": "Liver · Space Flight"},
        "comparison": {"facet": "spaceflight arm",
                       "query_b": {"cohort_label": "Liver · Ground Control"},
                       "stability": _measured(pooled=0.31, weakest=None)}})
    emitted = set()
    for arm in _find(children, "stability-cohort"):
        for kid in (arm.children or []):
            emitted.update(c for c in
                           (getattr(kid, "className", "") or "").split()
                           if c)

    missing = emitted - set(placed)
    assert not missing, (
        f"{sorted(missing)} are direct children of an arm with no `grid-row` "
        f"rule, so each lands in an implicit track the other column does not "
        f"have and the two arms shear apart")
    stale = set(placed) - emitted
    assert not stale, (
        f"{sorted(stale)} are placed in the pair grid but nothing emits them")

    tracks = re.search(r"\.stability-pair\.is-pair\s*\{[^}]*grid-template-rows:"
                       r"\s*([^;]+);", css)
    assert tracks, "the pair grid declares no rows for its arms to borrow"
    assert len(tracks.group(1).split()) >= max(placed.values()), (
        f"row {max(placed.values())} is placed but only "
        f"{len(tracks.group(1).split())} tracks are declared, so it falls into "
        f"an implicit one that `subgrid` does not align")


def test_the_member_that_moves_it_most_puts_its_name_on_its_own_line():
    """Label, score and a 27-character sample key shared one baseline row while
    the panel was a single 322px column. In a 155px one they cannot: the label
    and the value are fixed width, which left about 29px for the key and - since
    it wraps rather than truncates, deliberately - broke it one character per
    line into a 400px column."""
    from bridge_rna.panels import build_stability_panel

    children, _ = build_stability_panel({
        "mode": "cohort",
        "stability": _measured(weakest="OSD-137|Mmus_C57-6J_EYE_GC_Rep1_M33")})
    weakest = _find(children, "stability-weakest")
    assert len(weakest) == 1
    row = _find(weakest[0], "stability-weakest-row")
    assert len(row) == 1, "the label and the score share one line"
    assert _find(row[0], "stability-weakest-label")
    assert _find(row[0], "stability-weakest-value")
    # The name is a sibling of that row, not a third item inside it.
    assert not _find(row[0], "stability-weakest-name")
    assert _find(weakest[0], "stability-weakest-name")


def test_a_cohort_with_no_weakest_member_says_so_rather_than_dropping_the_row():
    """`cohorts.weakest_member` returns None when every member's absence moves
    the list equally far. That is an answer, not a gap - and an absent row and a
    clipped row look identical on screen, which is the exact ambiguity the even
    split was built to remove."""
    from bridge_rna.panels import build_stability_panel

    children, _ = build_stability_panel({
        "mode": "cohort", "stability": _measured(weakest=None)})
    weakest = _find(children, "stability-weakest")
    assert len(weakest) == 1, "the row is drawn even with nobody to name"
    text = _series(weakest[0])
    assert "Moves it most" in text
    assert "every member equally" in text
    # No score, because there is no member for one to belong to.
    assert not _find(weakest[0], "stability-weakest-value")
    # And it is not dressed as a sample key: it is prose, not an accession.
    assert "is-none" in _classes(weakest[0])


def test_the_inspector_names_which_arm_it_is_opening_only_in_a_comparison():
    """Opening on cohort A used to read as *the* pooled query rather than as one
    of two, with nothing saying the other star leads to its twin."""
    import pandas as pd

    from bridge_rna.panels import build_cohort_details

    query = pd.Series({"cohort_label": "Liver · Space Flight", "is_cohort": "1",
                       "study_id": "OSD-1", "members": "a|1\na|2",
                       "grouped_by": "Study, Tissue", "stability": "0.34 at k = 2"})
    assert "Cohort A" not in _series(build_cohort_details(query))
    assert "Cohort A" in _series(build_cohort_details(query, role="a"))
    assert "Cohort B" in _series(build_cohort_details(query, role="b"))
    # The letter qualifies the heading; it never replaces the cohort's own name.
    assert "Liver · Space Flight" in _series(build_cohort_details(query, role="b"))


def test_the_cards_dot_colours_are_the_figures_own():
    """Plotly cannot read a CSS variable, so `GRAPH_THEME` mirrors the tokens.
    A drifted pair would give a card and the star it describes two colours."""
    import re
    from pathlib import Path

    from bridge_rna.figures import GRAPH_THEME

    css = (Path(__file__).resolve().parent.parent / "assets" / "00-tokens.css").read_text()

    def token(name: str) -> str:
        m = re.search(rf"{name}:\s*(#[0-9a-fA-F]{{6}})", css)
        assert m, f"{name} is not defined"
        return m.group(1)

    assert token("--accent-teal") == GRAPH_THEME["cohort_a"]
    assert token("--accent-warm") == GRAPH_THEME["cohort_b"]
