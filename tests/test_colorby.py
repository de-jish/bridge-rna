"""The colour-by registry: coverage, availability, and the degraded state.

These are the tests for the defect this redesign removed. Choosing an OSDR-only
field used to paint 99.8% of the map one flat grey, which is indistinguishable
from "ARCHS4 was measured and has no structure". Coverage is now a declared
property, so it can be asserted rather than eyeballed.
"""

from __future__ import annotations

import numpy as np

from manifold import colorby, data


def test_every_registered_field_resolves_to_full_length_labels(corpus):
    """A resolver returning the wrong length would silently shift every label."""
    total = corpus["total"]
    for spec in colorby.REGISTRY:
        values = colorby.labels(spec.key)
        assert len(values) == total, f"{spec.key} returned {len(values)} of {total}"


def test_coverage_matches_the_labels_actually_produced(corpus):
    """The advertised coverage must equal what the resolver really fills in.

    A drift between these two is the exact failure the readout exists to
    prevent: the control would promise a whole-map colouring and the renderer
    would deliver a partial one.
    """
    for spec in colorby.REGISTRY:
        covered, total = colorby.coverage(spec.key)
        values = colorby.labels(spec.key)
        actual = int((values != colorby.NOT_COVERED).sum())
        assert actual == covered, (
            f"{spec.key} advertises {covered} covered points but labelled {actual}")
        assert total == corpus["total"]


def test_whole_map_fields_cover_every_point(corpus):
    whole = [s for s in colorby.REGISTRY
             if colorby.group_of(s) == colorby.GROUP_WHOLE_MAP]
    assert whole, "no field colours the whole map"
    for spec in whole:
        covered, total = colorby.coverage(spec.key)
        assert covered == total, f"{spec.key} is grouped as whole-map but covers {covered}"


def test_osdr_only_fields_leave_archs4_uncovered(corpus):
    n_archs4 = corpus["n_archs4"]
    values = colorby.labels("flight_status")
    assert (values[:n_archs4] == colorby.NOT_COVERED).all()
    assert (values[n_archs4:] != colorby.NOT_COVERED).all()


def test_the_default_field_colours_the_whole_map(corpus):
    covered, total = colorby.coverage(colorby.default_key())
    assert covered == total, "the app opens on a field that leaves the map partly blank"


def test_menu_lists_whole_map_fields_before_osdr_only(corpus):
    options = colorby.menu_options()
    groups = [colorby.group_of(colorby.get(o["value"])) for o in options]
    first_osdr = groups.index(colorby.GROUP_OSDR)
    assert colorby.GROUP_WHOLE_MAP not in groups[first_osdr:], (
        "a whole-map field is buried below the OSDR-only ones")


def test_every_menu_option_declares_its_scope(corpus):
    for option in colorby.menu_options():
        assert "whole map" in option["label"] or "OSDR only" in option["label"] \
            or "unavailable" in option["label"], option["label"]


def test_no_enabled_option_is_undrawable(corpus):
    """Anything the menu lets a user pick must actually produce a figure."""
    from manifold import render

    for option in colorby.menu_options():
        if option["disabled"]:
            continue
        fig, legend, _ = render.build_figure(
            "pca", "2d", option["value"], ["archs4", "osdr", "density"], 1000, None)
        assert len(fig.data) > 0, f"{option['value']} drew nothing"
        assert legend["items"], f"{option['value']} produced no legend"


# --- The degraded state: no GEO metadata on this machine --------------------

def test_tissue_falls_back_to_osdr_only_without_the_join(corpus, without_archs4_metadata):
    """Tissue must degrade to a partial field, not vanish and not lie."""
    assert not data.archs4_metadata_available()
    spec = colorby.get("tissue")
    assert spec.available(), "tissue disappeared entirely; OSDR still has tissue"
    assert colorby.ARCHS4 not in spec.covers()
    covered, total = colorby.coverage("tissue")
    assert covered == corpus["n_osdr"] < total


def test_tissue_moves_out_of_the_whole_map_group_without_the_join(
        corpus, without_archs4_metadata):
    assert colorby.group_of(colorby.get("tissue")) == colorby.GROUP_OSDR


def test_a_missing_join_names_the_command_that_fixes_it(corpus, without_archs4_metadata):
    """A dead end is a bug; a dead end with the fix attached is a feature."""
    hint = colorby.get("tissue").missing_hint()
    assert "fetch_archs4_meta" in hint


def test_no_missing_hint_when_the_join_is_present(corpus):
    assert colorby.get("tissue").missing_hint() == ""


def test_species_still_covers_the_whole_map_without_any_join(
        corpus, without_archs4_metadata):
    """Species comes from the identity table, so it must survive a bare cache."""
    covered, total = colorby.coverage("species")
    assert covered == total


def test_unknown_key_falls_back_instead_of_raising(corpus):
    """A browser holding a stale selection across a rebuild must not 500."""
    spec = colorby.get("no-such-field-was-ever-registered")
    assert spec.key == colorby.default_key()


# --- Residual categories ----------------------------------------------------

def test_not_covered_is_never_treated_as_a_category():
    assert colorby.is_residual(colorby.NOT_COVERED)


def test_unknown_and_other_are_both_residual():
    assert colorby.is_residual("Unknown")
    assert colorby.is_residual("Other")
    assert not colorby.is_residual("Liver")


def test_tissue_shares_one_vocabulary_across_both_corpora(corpus):
    """The point of the shared vocabulary: the two corpora must overlap.

    If ARCHS4 and OSDR tissues landed in disjoint bucket sets, one "Tissue"
    colour-by would be two colour-bys wearing one name, and the legend would
    imply a comparison the data does not support.
    """
    n_archs4 = corpus["n_archs4"]
    values = colorby.labels("tissue")
    real = {c for c in np.unique(values) if not colorby.is_residual(c)}
    archs4_side = {c for c in np.unique(values[:n_archs4]) if c in real}
    osdr_side = {c for c in np.unique(values[n_archs4:]) if c in real}
    assert archs4_side & osdr_side, (
        f"no shared tissue bucket: ARCHS4 {sorted(archs4_side)} vs OSDR {sorted(osdr_side)}")
