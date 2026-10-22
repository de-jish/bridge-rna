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
    # An abbreviation lands where its expansion does. "BMDM" is a bone-marrow-
    # derived macrophage and "BM-MSC" a bone-marrow mesenchymal stromal cell,
    # and both expansions are claimed here by "bone marrow" before Blood /
    # immune sees "macrophage". Measured: 2,059 spelled-out BMDM samples were
    # here while 1,221 abbreviated ones read Other. LT-/ST-HSC and LSK
    # (Lin-Sca1+Kit+) are haematopoietic stem and progenitor gates, but a bare
    # "HSC" is not named: GEO uses it as often for the liver's hepatic stellate
    # cells, and nothing in the string says which.
    ("Bone marrow", ("bone marrow", "marrow", "haematopoietic", "hematopoietic",
                     "hspc", "cd34\\+", "\\bbmdms?\\b", "\\bbm-?mscs?\\b",
                     "\\bbmscs?\\b", "\\b[ls]t-hscs?\\b", "\\blsks?\\b")),
    # "cd4" and "cd8" are refused a following digit: as bare stems they fired
    # inside CD44, CD45, CD43, CD48, CD49f and CD80-88, and so filed CD45-negative
    # tumour cells, CD49f+ mammary basal cells and "CD45low retinal microglia"
    # here. CD45 is the pan-leukocyte marker, so it is named in its own right,
    # but only when it is positive (or a CD45RA/RB/RO T-cell isoform): CD45- is
    # the gate that *excludes* immune cells, and CD45low/int is how microglia
    # are sorted.
    #
    # The haematological-cancer abbreviations land where their expansions do:
    # "acute myeloid leukemia" is claimed here by "myeloid", "diffuse large
    # B-cell lymphoma" by "b-cell" and "T-cell acute lymphoblastic leukemia" by
    # "t-cell", ahead of Tumor / cancer, as the leukaemia lines K562, HL-60 and
    # THP-1 already are. AML12 is a mouse hepatocyte line, not a leukaemia, and
    # is refused so the Liver rule can claim it; AML-MSCs are a patient's marrow
    # stroma, not the leukaemia.
    ("Blood / immune", ("blood", "pbmc", "buffy coat", "serum", "plasma",
                        "leukocyte", "leucocyte", "lymphocyte", "monocyte",
                        "macrophage", "neutrophil", "granulocyte", "eosinophil",
                        "basophil", "erythrocyte", "erythroid", "~erythroblast",
                        "platelet", "thrombocyte", "megakaryocyte",
                        "dendritic cell", "(?<!non-)(?<!non )\\bimmune cell", "nk cell",
                        "natural killer", "t cell", "t-cell", "b cell",
                        "b-cell", "\\bcd4(?!\\d)", "\\bcd8(?!\\d)",
                        "\\bcd45(?:\\.[12])?\\+",
                        "\\bcd45 ?(?:pos|positive|hi|high|enriched)\\b",
                        "\\bcd45r[abo]", "treg", "thymocyte", "myeloid", "lymphoid",
                        "lymphoblastoid", "\\blcl\\b", "gm12878", "gm18",
                        "jurkat", "k562", "hl-60", "thp-1",
                        "\\baml\\b(?![- ]?(?:12|mscs?)\\b)", "\\bdlbcl\\b",
                        "\\b[bt]-all\\b")),
    # "cortex" is not only a brain word - kidneys and adrenal glands have one
    # too, and GEO writes all three. These two rules run ahead of Brain / CNS so
    # the organ named in the string wins over the region word. Same
    # first-match-wins mechanism as the smooth-muscle case below.
    ("Adrenal gland", ("adrenal cortex", "adrenal medulla")),
    ("Kidney", ("renal cortex", "renal medulla", "kidney cortex")),
    # The retina is CNS tissue, so GEO describes it in Brain's words: "neural
    # retina", "retinal microglia", "retina astrocytes". The organ named in the
    # string wins here too, as with cortex above. Only "retina" is guarded: the
    # rest of the Eye rule stays below Brain because "pre-optic area" is
    # hypothalamus and \boptic would claim it. Measured: 248 samples, every one
    # naming the retina, moved from Brain / CNS.
    ("Eye", ("retina", "retinal")),
    # --- Nervous system. "spinal cord" and peripheral nerve are split out of
    # Brain because they are anatomically distinct and both are well populated. ---
    ("Brain / CNS", ("brain", "cerebell", "cerebrum", "cerebral", "cortex",
                     "hippocamp", "hypothalam", "thalamus", "striatum",
                     "midbrain", "forebrain", "hindbrain", "amygdala",
                     "substantia nigra", "neuron", "neural", "glia", "glial",
                     "astrocyte", "oligodendrocyte", "microglia", "olfactory bulb",
                     "prefrontal", "frontal lobe", "temporal lobe", "occipital",
                     "putamen", "cerebellar",
                     # Regions the rule never named, each measured landing in
                     # Other on the real corpus. "neocort" because \bcortex
                     # cannot fire inside "neocortex"; "gyrus" rather than
                     # "fusiform", which also names a bile-duct cyst; the raphe
                     # only as a brainstem nucleus, since the perineum and the
                     # palate have one too.
                     "neocort", "isocort", "nucleus accumbens", "suprachiasmatic",
                     "dentate", "gyrus", "gyri", "striatal", "cingulate",
                     "habenul", "\\bpons\\b", "tegmental", "dorsal raphe",
                     "median raphe", "raphe nucle", "globus pallidus",
                     "locus coeruleus", "claustrum",
                     # The ventricles' CSF-secreting tissue, not the eye's
                     # choroid: 396 ARCHS4 samples were filed under Eye by the
                     # bare "choroid" stem before this was named here.
                     "choroid plexus",
                     # Hypothalamus, named here because the Eye rule's \boptic
                     # fires after the hyphen in "pre-optic"; unhyphenated, 24
                     # samples' "preoptic area" matched nothing and read Other.
                     "preoptic", "pre-optic")),
    ("Nerve / spinal cord", ("spinal cord", "sciatic", "dorsal root", "drg",
                             "peripheral nerve", "schwann")),
    # --- Solid organs. ---
    ("Liver", ("liver", "hepatic", "hepatocyte", "hepg2", "hepato",
               "\\baml[- ]?12\\b")),
    ("Lung", ("lung", "pulmonary", "alveol", "bronch", "airway", "a549",
              "respiratory epithel")),
    ("Adrenal gland", ("adrenal",)),
    ("Kidney", ("kidney", "renal", "nephron", "podocyte", "glomerul", "hek293",
                "hek 293")),
    # "hiPSC-CM" is a stem-cell-derived cardiomyocyte, and the organ it was
    # derived into wins, as "iPSC-derived cardiomyocytes" already did here. The
    # CM is only read after a stem-cell prefix: a bare "CM" is as often a
    # central-memory T cell or conditioned medium.
    ("Heart", ("heart", "cardiac", "cardiomyocyte", "myocardi", "ventricle",
               "ventricular", "atrium", "atrial", "aortic valve",
               "(?:psc|ips|esc)s?[- ](?:derived )?cms?\\b")),
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
    ("Pancreas", ("pancrea", "islet", "beta cell", "acinar", "\\bpdac\\b")),
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
    # T HESC is a telomerase-immortalized endometrial stromal line, not a human
    # ESC; named here, ahead of Embryo / stem cell, so the ESC plurals below
    # cannot claim it.
    ("Uterus", ("uterus", "uterine", "endometri", "myometri", "fallopian",
                "oviduct", "\\bt[- ]?hescs?\\b")),
    ("Placenta", ("placenta", "trophoblast", "chorion", "decidua", "amnion")),
    # Eye anatomy, measured against the real corpus: before "eyecup", "eyeball",
    # sclera, iris, conjunctiva, ciliary body, trabecular meshwork, limbus,
    # macula, vitreous and photoreceptor names were missing, 560 ARCHS4 samples
    # naming one of them read as Other, Cultured cells or Cell line. Each short
    # stem is anchored where a neighbour word would otherwise fire: "\bsclera\b"
    # not sclerosis, "\biris\b" not irisin, "conjunctiva" not GEO's
    # mistranslated "conjunctive tissue", "limbal" not limbic, and photoreceptors
    # by plural only, never a neuron's growth cone. "choroid" here is the eye's
    # vascular layer; the choroid plexus is a brain tissue and is claimed by the
    # Brain / CNS rule above. Ocular adnexa (lacrimal, meibomian and Harderian
    # glands, eyelid) are deliberately not named: they are exocrine glands and
    # skin beside the eye rather than the eye, and remain Other.
    #
    # The eye is an organ and sits with the organs, ahead of Embryo / stem cell
    # and the structural rules, for the reason "fetal liver" is Liver: a
    # developmental or cell-type qualifier must not outrank the organ it
    # qualifies. It used to sit after them, which filed "fetal retina", retinal
    # organoids, limbal stem cells and iPSC-derived RPE under Embryo / stem cell
    # (1,034 samples) and corneal and retinal endothelium under Vasculature
    # (178) - corneal endothelium is not vascular at all - while liver
    # organoids and lung endothelium already went to their organ.
    ("Eye", ("retina", "retinal", "\\beye\\b", "\\beyes\\b", "eyecup", "eyeball",
             "cornea", "\\blens\\b", "choroid", "photoreceptor", "rpe\\b",
             "\\barpe-?19\\b", "optic", "\\bsclera\\b", "scleral", "\\biris\\b",
             "conjunctiva", "uvea", "ocular melanoma", "ciliary body",
             "ciliary epithel", "trabecular meshwork", "limbal", "\\blimbus\\b",
             "macula", "vitreous", "hyalocyte", "pterygi",
             "\\brods\\b", "(?<!growth )\\bcones\\b")),
    ("Embryo / stem cell", ("embryo", "embryonic stem", "blastocyst", "ipsc",
                            "ips cell", "\\bips\\b", "induced pluripotent",
                            "pluripotent", "\\b[hm]?escs?\\b", "\\bes cell",
                            "germ layer", "gastrul", "organoid", "stem cell",
                            "zygote", "morula", "pronucle", "oocyte",
                            "blastomere", "\\bicm\\b", "fetal", "foetal")),
    # --- Connective, vascular, other structural. ---
    # White and brown adipose tissue by their abbreviations, with the depot
    # prefixes GEO writes onto them: inguinal, epididymal, gonadal, perigonadal,
    # visceral, subcutaneous, retroperitoneal, interscapular. Every "BAT" in the
    # corpus is brown fat; none is the animal.
    ("Adipose", ("adipose", "adipocyte", "fat pad", "\\bfat\\b", "omental",
                 "subcutaneous fat", "visceral fat", "3t3-l1",
                 "\\b(?:[egiprsv]|sc|ing|pg|ps|vs)?wat\\b", "\\b(?:i|sc|f)?bat\\b")),
    ("Bone / cartilage", ("bone", "osteoblast", "osteoclast", "osteocyte",
                          "cartilage", "chondrocyte", "mandible", "femur",
                          "femoral bone", "calvaria", "vertebra", "skeletal tissue")),
    ("Vasculature", ("endothelial", "huvec", "umbilical vein", "aorta", "aortic",
                     "artery", "arterial", "\\bvein\\b", "venous", "vascular",
                     "vasculature", "pericyte", "smooth muscle cell")),
    ("Bladder", ("bladder", "urothel", "urinary")),
    ("Salivary gland", ("salivary", "parotid", "submandibular gland")),
    # Brain-region abbreviations, and "caudate", which is also a lobe of the
    # liver. They are weaker evidence than a written-out organ, so they run
    # after every organ rule: "NAC resistant breast cancer" (neoadjuvant
    # chemotherapy) stays Breast and "caudate liver lobe" stays Liver, while a
    # bare "NAc" or "PFC" with nothing else beside it is still placed.
    ("Brain / CNS", ("\\bnac\\b", "\\bpfc\\b", "\\bscn\\b", "\\bvta\\b",
                     "caudate")),
    # --- Last resorts. These are not tissues, but they name a large and
    # genuinely distinct slice of GEO, and burying them in "Other" would hide
    # real structure the map does show. Tumour morphemes are substring rules
    # ("~sarcoma") because GEO glues them onto a tissue stem: "osteosarcoma",
    # "hepatocarcinoma", "neuroblastoma" would all miss a word-anchored match. ---
    ("Tumor / cancer", ("tumor", "tumour", "~carcinoma", "~sarcoma", "~melanoma",
                        "~lymphoma", "~leukemia", "~leukaemia", "~myeloma",
                        "~glioma", "~blastoma", "~cytoma", "meningioma", "cancer",
                        "malignan", "metasta", "neoplas", "xenograft",
                        "\\bpdx\\b")),
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
