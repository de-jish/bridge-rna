"""One tissue vocabulary shared by both corpora.

OSDR and ARCHS4 name tissues in completely different registers. OSDR is curated
and anatomical but hyper-specific ("Right extensor digitorum longus", "Left Lobe
of the Liver", 48 distinct values over 2,108 samples). ARCHS4 has no curated
tissue column at all: the signal lives in GEO's free-text ``characteristics_ch1``
and ``source_name_ch1``, which yields 42,754 distinct lowercased strings over
940,455 samples.

Left alone those are two disjoint vocabularies, and "Tissue" has to be two
separate color-bys that each leave the other corpus grey. Folding both onto one
canonical bucket list is what makes a single "Tissue" color-by paint the whole
map - the entire point of the redesign.

The mapping is deliberately keyword-based and ordered, not learned. Every rule
is readable, auditable, and fails towards "Other" rather than towards a
confident guess: on a plot people read biology off, an honestly empty label
beats a wrong one. ``canonical_tissue`` is the only entry point, and it is used
by ``precompute/fetch_archs4_meta.py`` for ARCHS4 and by ``manifold/data.py``
for OSDR, so the two corpora can never drift apart.
"""

from __future__ import annotations

import re

UNKNOWN = "Unknown"
OTHER = "Other"

# Ordered (bucket, patterns) rules. FIRST MATCH WINS, so the order is
# load-bearing: "bone marrow" must be tested before "bone", "cardiac muscle"
# before "muscle", "adrenal" before "renal".
#
# A pattern is matched against a lowercased, whitespace-collapsed string with a
# leading \b word boundary applied, so "renal" cannot fire inside "adrenal".
# Prefix a pattern with "~" to drop that boundary and match as a bare substring
# - needed for the morphemes GEO glues onto a stem, where "sarcoma" has to match
# inside "osteosarcoma" and "carcinoma" inside "hepatocarcinoma".
_RULES: list[tuple[str, tuple[str, ...]]] = [
    # --- Haematopoietic. Tested first: "bone marrow" contains "bone", and
    # blood-cell names are specific enough to be unambiguous. ---
    ("Bone marrow", ("bone marrow", "marrow", "haematopoietic", "hematopoietic",
                     "hspc", "cd34\\+")),
    ("Blood / immune", ("blood", "pbmc", "buffy coat", "serum", "plasma",
                        "leukocyte", "leucocyte", "lymphocyte", "monocyte",
                        "macrophage", "neutrophil", "granulocyte", "eosinophil",
                        "basophil", "erythrocyte", "erythroid", "platelet",
                        "megakaryocyte", "dendritic cell", "nk cell",
                        "natural killer", "t cell", "t-cell", "b cell", "b-cell",
                        "cd4", "cd8", "treg", "thymocyte", "myeloid", "lymphoid",
                        "lymphoblastoid", "\\blcl\\b", "gm12878", "gm18",
                        "jurkat", "k562", "hl-60", "thp-1")),
    # "cortex" is not only a brain word - kidneys and adrenal glands have one
    # too, and GEO writes all three. These two rules run ahead of Brain / CNS so
    # the organ named in the string wins over the region word. Same
    # first-match-wins mechanism as the smooth-muscle case below.
    ("Adrenal gland", ("adrenal cortex", "adrenal medulla")),
    ("Kidney", ("renal cortex", "renal medulla", "kidney cortex")),
    # --- Nervous system. "spinal cord" and peripheral nerve are split out of
    # Brain because they are anatomically distinct and both are well populated. ---
    ("Brain / CNS", ("brain", "cerebell", "cerebrum", "cerebral", "cortex",
                     "hippocamp", "hypothalam", "thalamus", "striatum",
                     "midbrain", "forebrain", "hindbrain", "amygdala",
                     "substantia nigra", "neuron", "neural", "glia", "glial",
                     "astrocyte", "oligodendrocyte", "microglia", "olfactory bulb",
                     "prefrontal", "frontal lobe", "temporal lobe", "occipital",
                     "putamen", "cerebellar")),
    ("Nerve / spinal cord", ("spinal cord", "sciatic", "dorsal root", "drg",
                             "peripheral nerve", "schwann")),
    # --- Solid organs. ---
    ("Liver", ("liver", "hepatic", "hepatocyte", "hepg2", "hepato")),
    ("Lung", ("lung", "pulmonary", "alveol", "bronch", "airway", "a549",
              "respiratory epithel")),
    ("Adrenal gland", ("adrenal",)),
    ("Kidney", ("kidney", "renal", "nephron", "podocyte", "glomerul", "hek293",
                "hek 293")),
    ("Heart", ("heart", "cardiac", "cardiomyocyte", "myocardi", "ventricle",
               "ventricular", "atrium", "atrial", "aortic valve")),
    # Smooth muscle is not skeletal muscle, and the bare "muscle" stem below
    # would otherwise claim it. Placed here rather than merged into the
    # Vasculature rule further down because first-match-wins ordering is what
    # resolves the collision; a bucket may legitimately appear in two rules.
    ("Vasculature", ("smooth muscle", "vascular smooth")),
    ("Skeletal muscle", ("skeletal muscle", "muscle", "soleus", "gastrocnemius",
                         "quadriceps", "tibialis", "extensor digitorum",
                         "myotube", "myoblast", "myofib", "sarcopenia", "c2c12")),
    ("Skin", ("skin", "epiderm", "dermis", "dermal", "keratinocyte", "melanocyte",
              "hair follicle", "sebaceous")),
    ("Intestine", ("intestin", "colon", "colorectal", "ileum", "ileal", "jejunum",
                   "duodenum", "rectum", "rectal", "cecum", "caecum", "bowel",
                   "gut", "enteric", "caco-2", "crypt")),
    ("Stomach", ("stomach", "gastric", "antrum", "pylor")),
    ("Esophagus", ("esophag", "oesophag")),
    ("Pancreas", ("pancrea", "islet", "beta cell", "acinar")),
    ("Spleen", ("spleen", "splenic", "splenocyte")),
    ("Thymus", ("thymus", "thymic")),
    ("Lymph node", ("lymph node", "lymphatic", "tonsil", "peyer")),
    ("Thyroid", ("thyroid",)),
    ("Pituitary", ("pituitary", "hypophys")),
    # --- Reproductive and developmental. ---
    ("Breast / mammary", ("breast", "mammary", "mcf-7", "mcf7", "mda-mb")),
    ("Prostate", ("prostate", "prostatic", "lncap")),
    ("Testis", ("testis", "testes", "testicular", "sperm", "spermato", "sertoli",
                "leydig")),
    ("Ovary", ("ovary", "ovarian", "oocyte", "granulosa", "follicular fluid")),
    ("Uterus", ("uterus", "uterine", "endometri", "myometri", "fallopian",
                "oviduct")),
    ("Placenta", ("placenta", "trophoblast", "chorion", "decidua", "amnion")),
    ("Embryo / stem cell", ("embryo", "embryonic stem", "blastocyst", "ipsc",
                            "ips cell", "\\bips\\b", "induced pluripotent",
                            "pluripotent", "\\besc\\b", "\\bes cell",
                            "germ layer", "gastrul", "organoid", "stem cell",
                            "zygote", "morula", "pronucle", "oocyte",
                            "blastomere", "\\bicm\\b", "fetal", "foetal")),
    # --- Connective, vascular, other structural. ---
    ("Adipose", ("adipose", "adipocyte", "fat pad", "\\bfat\\b", "omental",
                 "subcutaneous fat", "visceral fat", "3t3-l1")),
    ("Bone / cartilage", ("bone", "osteoblast", "osteoclast", "osteocyte",
                          "cartilage", "chondrocyte", "mandible", "femur",
                          "femoral bone", "calvaria", "vertebra", "skeletal tissue")),
    ("Vasculature", ("endothelial", "huvec", "umbilical vein", "aorta", "aortic",
                     "artery", "arterial", "\\bvein\\b", "venous", "vascular",
                     "vasculature", "pericyte", "smooth muscle cell")),
    ("Eye", ("retina", "retinal", "\\beye\\b", "cornea", "\\blens\\b", "choroid",
             "photoreceptor", "rpe\\b", "optic")),
    ("Bladder", ("bladder", "urothel", "urinary")),
    ("Salivary gland", ("salivary", "parotid", "submandibular gland")),
    # --- Last resorts. These are not tissues, but they name a large and
    # genuinely distinct slice of GEO, and burying them in "Other" would hide
    # real structure the map does show. Tumour morphemes are substring rules
    # ("~sarcoma") because GEO glues them onto a tissue stem: "osteosarcoma",
    # "hepatocarcinoma", "neuroblastoma" would all miss a word-anchored match. ---
    ("Tumor / cancer", ("tumor", "tumour", "~carcinoma", "~sarcoma", "~melanoma",
                        "~lymphoma", "~leukemia", "~leukaemia", "~myeloma",
                        "~glioma", "~blastoma", "~cytoma", "cancer", "malignan",
                        "metasta", "neoplas", "xenograft", "\\bpdx\\b")),
    ("Reference RNA", ("universal human reference", "\\buhrr\\b", "\\bhbrr\\b",
                       "reference rna", "\\bermcc\\b", "spike-in control",
                       "\\bseqc\\b", "\\bmaqc\\b")),
    ("Cell line", ("cell line", "cell-line", "hela", "\\bu2os\\b", "\\bsh-sy5y\\b",
                   "\\bcos-7\\b", "immortalized", "immortalised", "\\bcho\\b",
                   "\\b293", "\\bhek\\b", "\\bhct116\\b", "\\bntera\\b")),
    ("Cultured cells", ("\\bcells\\b", "\\bcell\\b", "primary culture",
                        "\\bculture\\b", "fibroblast", "passage")),
]

def _compile(pattern: str) -> str:
    """One rule pattern -> one regex alternative.

    Default is a leading word boundary, which is what keeps short stems safe:
    \brenal cannot fire inside "adrenal". A "~" prefix drops the boundary for
    the morphemes GEO concatenates, and a pattern already containing a backslash
    is passed through as hand-written regex.
    """
    if pattern.startswith("~"):
        return re.escape(pattern[1:])
    if "\\" in pattern:
        return pattern
    return r"\b" + re.escape(pattern)


_COMPILED: list[tuple[str, re.Pattern[str]]] = [
    (bucket, re.compile("|".join(_compile(p) for p in patterns)))
    for bucket, patterns in _RULES
]

_MISSING = {"", "na", "n/a", "none", "null", "nan", "unknown", "not specified",
            "not applicable", "not available", "-", "--", "?"}

# Buckets that carry no anatomy. A value that is only "cells" or "HeLa" is not
# evidence of a tissue, so it must not outrank a real organ named in a later
# field. Ranked, because they are not equally uninformative: naming a cell line
# is more than naming nothing, and both beat text we could not place at all.
# Without this ranking an early unplaceable field would pin the answer to
# "Other" and block a later field that did identify the sample.
_WEAK_RANK = {"Cell line": 3, "Reference RNA": 3, "Cultured cells": 2,
              OTHER: 1, UNKNOWN: 0}

# A bucket may be named by more than one rule (see the smooth-muscle case), so
# dedupe while preserving rule order.
BUCKETS: tuple[str, ...] = tuple(
    dict.fromkeys([b for b, _ in _RULES] + [OTHER, UNKNOWN])
)


def _normalize(value: object) -> str:
    """Lowercase, collapse whitespace, and strip GEO's punctuation noise."""
    s = str(value or "").strip().lower()
    if not s or s in _MISSING:
        return ""
    s = s.replace("_", " ").replace("/", " / ")
    return re.sub(r"\s+", " ", s)


def canonical_tissue(value: object) -> str:
    """Map one free-text tissue/source string onto the shared bucket vocabulary.

    Returns ``Unknown`` for an empty or placeholder value and ``Other`` for a
    real value that matches no rule. Those two are kept distinct on purpose:
    "we were never told" and "we were told something we cannot place" are
    different facts, and collapsing them would overstate coverage.
    """
    s = _normalize(value)
    if not s:
        return UNKNOWN
    for bucket, pattern in _COMPILED:
        if pattern.search(s):
            return bucket
    return OTHER


def is_weak(bucket: str) -> bool:
    """True if a bucket carries no anatomical information.

    Lets a caller prefer a real tissue found in a lower-priority field over a
    bare "cells" found in a higher-priority one.
    """
    return bucket in _WEAK_RANK


def coalesce_tissue(*values: object) -> str:
    """Best canonical tissue across several candidate fields, best-evidence first.

    Fields are tried in the order given and the first anatomical hit wins. A
    weak hit does not stop the search: a later field naming a real organ beats
    an earlier one saying only "cells". Among weak results the most informative
    is kept rather than the first, so a sample whose `tissue:` key was
    unplaceable but whose source_name says "HeLa" reads as a cell line instead
    of as "Other".
    """
    best = UNKNOWN
    for value in values:
        bucket = canonical_tissue(value)
        if not is_weak(bucket):
            return bucket
        if _WEAK_RANK[bucket] > _WEAK_RANK[best]:
            best = bucket
    return best
