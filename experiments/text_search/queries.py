"""The fixed free-text query set, written 2026-11-05 12:05 EST before any result was seen.

Each query is (text, relevant tissue buckets, species or None). A held-out
sample is relevant when its canonical tissue bucket is in the set and, if a
species is named, its species matches. The labels are the canonical tissue
buckets of `manifold/tissue.py` and the species of `points_meta`; nothing
finer exists for held-out samples, so a query like "activated T cells" is
scored at the bucket level. That makes the free-text score a test of whether
the method maps paraphrase and context onto the right tissue, not of whether
it resolves the condition. Do not edit after the baseline round.
"""

HUMAN, MOUSE = 0, 1

FREE_TEXT_QUERIES = [
    ("mouse liver", {"Liver"}, MOUSE),
    ("human liver biopsy", {"Liver"}, HUMAN),
    ("hepatocytes", {"Liver"}, None),
    ("human T cells activated", {"Blood / immune"}, HUMAN),
    ("peripheral blood mononuclear cells", {"Blood / immune"}, None),
    ("mouse bone marrow macrophages", {"Bone marrow", "Blood / immune"}, MOUSE),
    ("skeletal muscle after unloading", {"Skeletal muscle"}, None),
    ("mouse soleus muscle", {"Skeletal muscle"}, MOUSE),
    ("gastrocnemius", {"Skeletal muscle"}, None),
    ("mouse hippocampus", {"Brain / CNS"}, MOUSE),
    ("human prefrontal cortex postmortem", {"Brain / CNS"}, HUMAN),
    ("retina", {"Eye"}, None),
    ("mouse heart left ventricle", {"Heart"}, MOUSE),
    ("cardiomyocytes", {"Heart"}, None),
    ("human lung tissue", {"Lung"}, HUMAN),
    ("airway epithelial cells", {"Lung"}, None),
    ("kidney cortex", {"Kidney"}, None),
    ("mouse spleen", {"Spleen"}, MOUSE),
    ("thymus", {"Thymus"}, None),
    ("small intestine epithelium", {"Intestine"}, None),
    ("colon", {"Intestine"}, None),
    ("pancreatic islets", {"Pancreas"}, None),
    ("white adipose tissue", {"Adipose"}, None),
    ("human skin fibroblasts", {"Skin"}, HUMAN),
    ("mouse skin", {"Skin"}, MOUSE),
    ("testis", {"Testis"}, None),
    ("mammary gland", {"Breast / mammary"}, None),
    ("breast cancer tumor", {"Breast / mammary", "Tumor / cancer"}, HUMAN),
    ("embryonic stem cells", {"Embryo / stem cell"}, None),
    ("femur bone", {"Bone / cartilage"}, None),
    ("adrenal gland", {"Adrenal gland"}, None),
    ("mouse thyroid", {"Thyroid"}, MOUSE),
]

# Tissue buckets used as text queries and as classifier labels. "Other" and
# "Unknown" are not descriptions; every other bucket is used verbatim.
EXCLUDED_BUCKETS = {"Other", "Unknown"}


def bucket_query_text(bucket: str) -> str:
    """The text a bucket is queried with: its name, slash-joined parts spelled out."""
    return bucket.replace(" / ", " or ")
