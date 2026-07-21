"""The shared tissue vocabulary.

Folding OSDR's curated anatomy and ARCHS4's GEO free text onto one bucket list
is what lets a single "Tissue" colour-by paint the whole map. The mapping is
ordered keyword matching, so the risks are ordering collisions ("bone marrow"
losing to "bone") and word-boundary mistakes ("renal" firing inside "adrenal").
Both are cheap to pin and expensive to notice by eye on a 940k-point scatter.
"""

from __future__ import annotations

import pytest

from manifold import tissue


@pytest.mark.parametrize("raw,expected", [
    # OSDR's register: anatomically precise and hyper-specific.
    ("Liver", "Liver"),
    ("Left Lobe of the Liver", "Liver"),
    ("Right extensor digitorum longus", "Skeletal muscle"),
    ("Left gastrocnemius", "Skeletal muscle"),
    ("dorsal skin", "Skin"),
    ("Right hippocampus", "Brain / CNS"),
    ("right hemisphere of cerebellum", "Brain / CNS"),
    ("Right retina", "Eye"),
    ("Temporal Bone", "Bone / cartilage"),
    ("white adipose tissue", "Adipose"),
    ("descending colon", "Intestine"),
    ("Spleen-distal", "Spleen"),
    # ARCHS4's register: GEO free text, lowercase, cell lines and abbreviations.
    ("whole blood", "Blood / immune"),
    ("PBMC", "Blood / immune"),
    ("bone marrow", "Bone marrow"),
    ("HeLa", "Cell line"),
    ("293S cells", "Cell line"),
    ("GM12878", "Blood / immune"),
    ("zygote", "Embryo / stem cell"),
    ("iPS", "Embryo / stem cell"),
    ("Universal Human Reference RNA (UHRR) from Stratagene", "Reference RNA"),
])
def test_known_values_map_to_the_expected_bucket(raw, expected):
    assert tissue.canonical_tissue(raw) == expected


@pytest.mark.parametrize("raw,expected", [
    # Ordering: the longer, more specific rule must be tested first.
    ("bone marrow", "Bone marrow"),
    ("femur bone", "Bone / cartilage"),
    # Word boundaries: "renal" must not fire inside "adrenal".
    ("adrenal gland", "Adrenal gland"),
    ("renal cortex", "Kidney"),
    # Smooth muscle is vascular, and must not be claimed by the "muscle" stem.
    ("smooth muscle cell", "Vasculature"),
    ("skeletal muscle", "Skeletal muscle"),
    # Heart is tested before muscle, so cardiac tissue does not become skeletal.
    ("cardiac muscle", "Heart"),
    # Tumour morphemes are substring rules because GEO glues them onto a stem.
    ("Osteosarcoma", "Tumor / cancer"),
    ("neuroblastoma", "Tumor / cancer"),
    # ...but a tissue stem still wins when the organ is named outright.
    ("hepatocarcinoma", "Liver"),
    # A developmental qualifier must not outrank the organ it qualifies.
    ("fetal liver", "Liver"),
])
def test_ordering_and_boundary_collisions_resolve_correctly(raw, expected):
    assert tissue.canonical_tissue(raw) == expected


@pytest.mark.parametrize("raw", ["", "   ", "NA", "n/a", "none", "unknown", None])
def test_placeholders_read_as_unknown(raw):
    assert tissue.canonical_tissue(raw) == tissue.UNKNOWN


def test_unplaceable_text_is_other_not_unknown():
    """The two must stay distinct: nothing recorded, versus recorded but unplaceable."""
    assert tissue.canonical_tissue("qwertyuiop widget 42") == tissue.OTHER
    assert tissue.canonical_tissue("") == tissue.UNKNOWN


def test_coalesce_prefers_a_real_organ_over_an_earlier_weak_hit():
    """A later field naming an organ must beat an earlier one saying only "cells"."""
    assert tissue.coalesce_tissue("cells", "liver") == "Liver"
    assert tissue.coalesce_tissue("qwertyuiop", "spleen") == "Spleen"


def test_coalesce_keeps_the_most_informative_weak_answer():
    """An early unplaceable value must not pin the result to Other.

    This was a real defect: "Other" is weak but non-empty, so it used to block a
    later field that did identify the sample, and HeLa samples read as Other.
    """
    assert tissue.coalesce_tissue("qwertyuiop widget", "HeLa") == "Cell line"
    assert tissue.coalesce_tissue("HeLa", "qwertyuiop widget") == "Cell line"


def test_coalesce_of_nothing_is_unknown():
    assert tissue.coalesce_tissue("", None, "n/a") == tissue.UNKNOWN


def test_every_osdr_value_in_the_fixture_lands_in_an_anatomical_bucket(corpus):
    """OSDR is curated, so an unplaceable value there means the map has a gap."""
    from manifold import data

    unplaced = sorted({
        v for v in data.osdr_field_values("tissue").unique()
        if tissue.canonical_tissue(v) in (tissue.OTHER, tissue.UNKNOWN)
    })
    assert not unplaced, f"OSDR tissues with no bucket: {unplaced}"


def test_buckets_are_unique_and_include_the_residuals():
    assert len(tissue.BUCKETS) == len(set(tissue.BUCKETS))
    assert tissue.OTHER in tissue.BUCKETS and tissue.UNKNOWN in tissue.BUCKETS


def test_case_and_whitespace_do_not_change_the_answer():
    for variant in ("LIVER", "liver", "  Liver  ", "Liver\t"):
        assert tissue.canonical_tissue(variant) == "Liver"
