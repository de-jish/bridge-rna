"""The shared tissue vocabulary.

Folding OSDR's curated anatomy and ARCHS4's GEO free text onto one bucket list
is what lets a single "Tissue" colour-by paint the whole map. The mapping is
ordered keyword matching, so the risks are ordering collisions ("bone marrow"
losing to "bone") and word-boundary mistakes ("renal" firing inside "adrenal").
Both are cheap to pin and expensive to notice by eye on a 940k-point scatter.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from manifold import tissue  # noqa: E402
from precompute import fetch_archs4_meta as fam  # noqa: E402


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


@pytest.mark.parametrize("raw", [
    # The reported defect: GSE210492 writes its eyes as "posterior eyecup", and
    # "\beye\b" needs a boundary after "eye" that "eyecup" does not have. Every
    # field of those samples fell to Other, 7 of one retrieval's top 30.
    "posterior eyecup",
    "Eyecup",
    "eyeball organization from C57/B6 mouse",
    "case_eyeball_RNA-seq",
    "Eyes",
    # Eye anatomy the rule never named. Each was measured on the real ARCHS4
    # corpus landing in Other, Cultured cells or Cell line.
    "Sclera",
    "Peripapillary Sclera",
    "Iris",
    "Human iris pigment epithelial cell",
    "Conjunctiva",
    "Conjunctival epithelium",
    "Ciliary body",
    "Non-Pigment Ciliary Epithelial Cell",
    "Human trabecular meshwork explants",
    "SV40 immortalized trabecular meshwork cells",
    "Limbal Rim",
    "limbus",
    "Macular pucker",
    "macula lutea",
    "Hyalocytes_Macular_hole",
    "vitreous humor",
    "pterygium",
    "ARPE-19",
    "arpe19",
    "Isolated rods",
    "Cones",
    # A tumour of the eye joins the eye, the rule the organs already follow
    # ("hepatocarcinoma" is Liver): uveal melanoma was 460 Tumor samples.
    "uveal melanoma cell line",
    "Ocular melanoma cell line",
    "Conjunctival melanoma",
])
def test_eye_anatomy_maps_to_eye(raw):
    assert tissue.canonical_tissue(raw) == "Eye"


@pytest.mark.parametrize("raw,expected", [
    # The choroid plexus is the CSF-secreting tissue of the brain's ventricles,
    # not the eye's choroid. The bare "choroid" stem filed 396 ARCHS4 samples
    # under Eye, so it is named in the Brain rule, which runs first.
    ("Choroid Plexus", "Brain / CNS"),
    ("choroid plexus, lateral", "Brain / CNS"),
    ("Choroid plexus endothelial cell", "Brain / CNS"),
    ("choroid plexus and ventricular wall", "Brain / CNS"),
    ("hiPSC-derived choroid plexus organoids (CPOs)", "Brain / CNS"),
    # ...while the eye's own choroid stays Eye.
    ("RPE-choroid eyecups", "Eye"),
    ("Macula Choroid", "Eye"),
    ("Ocular Choroid Fibroblast", "Eye"),
    # "conjunctiva", not "conjunctiv": GEO has a mistranslated "conjunctive
    # tissue" that means connective tissue (the 10T1/2 fibroblast line).
    ("C3H/An mouse conjunctive tissue (fibroblasts)", "Cultured cells"),
    # "sclera" is anchored at both ends, so sclerosis and scleroderma (1,073
    # Blood samples, almost all multiple sclerosis) are not read as eye.
    ("multiple sclerosis", tissue.OTHER),
    ("scleroderma", tissue.OTHER),
    # "iris" only as a whole word.
    ("irisin", tissue.OTHER),
    # "limbal" is the corneal limbus, not the limbic system.
    ("limbic system", tissue.OTHER),
    # Photoreceptors are named by plural; a neuron's growth cone is not one.
    ("growth cones", tissue.OTHER),
    ("ectoplacental cone", tissue.OTHER),
    # "retinal" with a leading boundary does not fire inside "retinoic".
    ("retinoic acid treated fibroblasts", "Cultured cells"),
    # Extraocular muscles are skeletal muscle, and stay so.
    ("Extra ocular muscles", "Skeletal muscle"),
])
def test_eye_terms_do_not_claim_what_is_not_the_eye(raw, expected):
    assert tissue.canonical_tissue(raw) == expected


def test_case_and_whitespace_do_not_change_the_answer():
    for variant in ("LIVER", "liver", "  Liver  ", "Liver\t"):
        assert tissue.canonical_tissue(variant) == "Liver"


def test_derive_tissue_places_the_reported_eyecup_samples_in_eye():
    """GSM6431263 exactly as the live metadata parquet carries it.

    Its `cell line:` key and its source_name both say "posterior eyecup" and its
    `cell type:` says "female", so every field used to fall to Other.
    """
    row = pd.DataFrame({
        "source_name": ["posterior eyecup"],
        "characteristics": ["cell line: posterior eyecup,cell type: female,"
                            "genotype: Wt,treatment: none,time: 17 months"],
    })
    assert fam.derive_tissue(row).tolist() == ["Eye"]


def test_retissue_rederives_only_the_tissue_column(tmp_path):
    """`fetch_archs4_meta.py --retissue` re-buckets offline and touches nothing else.

    A vocabulary change must not need the network fetch, and it must not be able
    to disturb the positional join: every other column, and the row order, has
    to come back exactly as it went in.
    """
    src = tmp_path / "archs4_metadata.parquet"
    stale = pd.DataFrame({
        "global_index": pd.array([0, 1, 2], dtype="int32"),
        "geo_accession": ["GSM6431263", "GSM1", "GSM2"],
        "series_id": ["GSE210492", "GSE1", ""],
        "title": ["posterior eyecup, Wt, 17 months, 7-2", "liver 1", ""],
        "source_name": ["posterior eyecup", "liver", ""],
        "characteristics": ["cell line: posterior eyecup,cell type: female",
                            "tissue: liver", ""],
        "tissue": ["Other", "Liver", "Unknown"],
    })
    stale.to_parquet(src, index=False)
    dst = tmp_path / "candidate" / "archs4_metadata.parquet"

    transitions = fam.retissue(src, dst)

    out = pd.read_parquet(dst)
    assert list(out.columns) == list(stale.columns)
    pd.testing.assert_frame_equal(out.drop(columns="tissue"),
                                  stale.drop(columns="tissue"))
    assert out["tissue"].tolist() == ["Eye", "Liver", "Unknown"]
    assert transitions.to_dict("records") == [
        {"from": "Other", "to": "Eye", "samples": 1}]
    # Naming a separate output reads the source and never rewrites it.
    assert pd.read_parquet(src)["tissue"].tolist() == ["Other", "Liver", "Unknown"]
