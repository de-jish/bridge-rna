"""Identical-profile groups among retrieved hits."""

from __future__ import annotations

import numpy as np

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
