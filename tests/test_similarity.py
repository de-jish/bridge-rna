"""Identical-profile groups and the clustering behind the similarity plot."""

from __future__ import annotations

import numpy as np
import pytest

from bridge_rna import similarity as S


def _unit(rows) -> np.ndarray:
    x = np.asarray(rows, dtype=np.float64)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_identical_profiles_group_and_replicates_do_not():
    rng = np.random.default_rng(3)
    base = rng.normal(size=(4, 512))
    rows = np.stack([
        base[0],
        base[1],
        base[0] * 1.7,                 # same direction, different depth: identical
        base[2],
        base[1] + 1e-9,                # float noise: identical
        base[0] + base[3] * 0.003,     # a close replicate, not the same profile
    ])
    unit = _unit(rows)
    cos = S.cosine_matrix(unit)
    assert 1 - cos[0, 5] > S.IDENTICAL_PROFILE_DISTANCE, "fixture must keep the replicate apart"
    assert S.identical_groups(unit) == [[0, 2], [1, 4]]


def test_identical_groups_are_transitive_components():
    v = np.ones(8)
    unit = _unit([v, v, np.arange(1, 9), v])
    assert S.identical_groups(unit) == [[0, 1, 3]]


def test_no_groups_for_tiny_or_distinct_inputs():
    assert S.identical_groups(np.zeros((0, 0))) == []
    assert S.identical_groups(_unit([[1.0, 0.0]])) == []
    assert S.identical_groups(_unit([[1.0, 0.0], [0.0, 1.0]])) == []


def test_cosine_matrix_is_symmetric_with_unit_diagonal():
    unit = _unit(np.random.default_rng(1).normal(size=(6, 5)))
    cos = S.cosine_matrix(unit)
    assert np.array_equal(cos, cos.T)
    assert np.all(np.diag(cos) == 1.0)


@pytest.mark.parametrize("n,seed", [(2, 0), (5, 1), (12, 2), (40, 3)])
def test_average_linkage_matches_scipy(n, seed):
    hierarchy = pytest.importorskip("scipy.cluster.hierarchy")
    distance = pytest.importorskip("scipy.spatial.distance")
    pts = np.random.default_rng(seed).normal(size=(n, 7))
    d = distance.squareform(distance.pdist(pts))
    ours = S.average_linkage(d)
    ref = hierarchy.linkage(distance.squareform(d, checks=False), method="average")
    np.testing.assert_allclose([m.height for m in ours], ref[:, 2], rtol=1e-12)
    assert [m.size for m in ours] == ref[:, 3].astype(int).tolist()
    # Leaf membership per merge must agree, whatever ids each side assigns.
    def members(z, i, n, left, right):
        out = set()
        for c in (left(z, i), right(z, i)):
            out |= {c} if c < n else members(z, c - n, n, left, right)
        return out
    for i in range(n - 1):
        mine = members(ours, i, n, lambda z, k: z[k].left, lambda z, k: z[k].right)
        theirs = members(ref, i, n, lambda z, k: int(z[k, 0]), lambda z, k: int(z[k, 1]))
        assert mine == theirs


def test_leaf_order_is_a_permutation_that_keeps_clusters_contiguous():
    pts = np.array([[0.0], [10.0], [0.2], [10.1], [5.0]])
    d = np.abs(pts - pts.T)
    merges = S.average_linkage(d)
    order = S.leaf_order(merges, 5)
    assert sorted(order) == list(range(5))
    pos = {leaf: i for i, leaf in enumerate(order)}
    assert abs(pos[0] - pos[2]) == 1
    assert abs(pos[1] - pos[3]) == 1


def test_dendrogram_segments_draw_one_bracket_per_merge():
    d = np.array([[0, 1, 4], [1, 0, 4], [4, 4, 0]], dtype=float)
    merges = S.average_linkage(d)
    segs = S.dendrogram_segments(merges, 3)
    assert len(segs) == 3 * len(merges)
    heights = sorted({round(s[3], 9) for s in segs if s[1] == s[3]})
    assert heights == [1.0, 4.0]
    xs = [v for s in segs for v in (s[0], s[2])]
    assert min(xs) == 0.0 and max(xs) == 2.0


def test_linkage_of_fewer_than_two_points_is_empty():
    assert S.average_linkage(np.zeros((1, 1))) == []
    assert S.leaf_order([], 1) == [0]
    assert S.dendrogram_segments([], 1) == []
