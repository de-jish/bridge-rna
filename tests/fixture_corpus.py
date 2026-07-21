"""Build a tiny, fully synthetic Bridge Manifold corpus on disk.

The real corpus is a 963 MB memmap plus a multi-hour embedding job, so nothing
in the test suite may depend on it. This module manufactures a miniature but
*structurally identical* stand-in: the same file names, the same dtypes, the
same fixed global point order ([ARCHS4 ..., then OSDR ...]), and the same
sidecar schemas. Code that works against this fixture works against the real
artifacts, because the only difference is scale.

The synthetic embeddings are deliberately not random noise. Points are drawn
around a small number of latent cluster centers in 512-d, and the metadata
labels are generated *from* those clusters. That gives the tests real ground
truth: a selection that follows one cluster must read as coherent and must
enrich for that cluster's tissue, and a selection scattered across clusters must
read as incoherent. A uniform-noise fixture could not distinguish a correct
coherence implementation from one that always says "yes".

Used by ``tests/conftest.py`` for the automated suite and by
``tests/build_dev_corpus.py`` to produce a browsable corpus for the running app.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

EMB_DIM = 512

# Category vocabularies, chosen so the fixture exercises both the low-cardinality
# color-bys (sex, spaceflight) and the Top-N-plus-Other legend path (tissue).
TISSUES = ["Liver", "Left kidney", "dorsal skin", "Soleus", "Quadriceps",
           "Eye", "Spleen", "Thymus", "Gastrocnemius", "Heart",
           "Adrenal gland", "Tibia", "Brain"]
STRAINS = ["C57BL/6J", "C57BL/6NTac", "C57BL/6NCrl", "BALB/c"]
SEXES = ["Female", "Male"]
GENOTYPES = ["Wild Type", "Nrf2KO", "p21-null"]
HABITATS = ["Rodent Habitat", "Habitat Cage Unit (HCU)", "Vivarium Cage"]
DIETS = ["Nutrient Upgraded Rodent Food Bar (NuRFB)", "Charles River Formula 1 (CRF-1)"]
SPACEFLIGHT = ["Space Flight", "Ground Control", "Basal Control"]

# ARCHS4's register is GEO free text, not a controlled vocabulary - which is the
# whole reason manifold/tissue.py exists. These are written raw and mapped
# through the real canonicalizer, so the fixture tests the mapping rather than
# assuming it. Chosen to overlap the OSDR vocabulary above, since a shared
# bucket for both corpora is what the "Tissue" color-by is for.
ARCHS4_SOURCES = ["liver", "whole blood", "Brain cortex", "HeLa",
                  "left kidney", "skeletal muscle biopsy"]


def _cluster_vectors(
    rng: np.random.Generator,
    n: int,
    n_clusters: int,
    spread: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (vectors, cluster_id): ``n`` points around ``n_clusters`` centers.

    ``spread`` sets within-cluster scatter relative to the between-cluster
    distance, so a caller can dial coherence up or down. Vectors are returned
    un-normalized, matching the real ARCHS4 memmap, which stores raw encoder
    output with L2 norms spanning roughly 6.7 to 25.5.
    """
    centers = rng.normal(size=(n_clusters, EMB_DIM)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    cluster_id = rng.integers(0, n_clusters, size=n)
    vecs = centers[cluster_id] + rng.normal(scale=spread, size=(n, EMB_DIM)).astype(np.float32)
    # Reproduce the real magnitude spread so any code that forgets to
    # L2-normalize before reducing is caught by the fixture rather than in prod.
    scale = rng.uniform(6.7, 25.5, size=(n, 1)).astype(np.float32)
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True) * scale
    return vecs.astype(np.float32), cluster_id


def build_bridge_rna_stub(root: Path, n_archs4: int,
                          seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Write a miniature stand-in for the Bridge RNA ARCHS4 artifacts.

    Produces the three files ``build_projections`` opens: the float16 embedding
    memmap, ``sample_locations.parquet``, and ``embedding_manifest.json``.
    Returns ``(vectors, cluster_id)`` so a caller can derive both ground-truth
    coordinates and cluster-correlated metadata from the same numbers.
    """
    rng = np.random.default_rng(seed)
    d = root / "archs4_sample_embeddings_full"
    d.mkdir(parents=True, exist_ok=True)

    vecs, cluster_id = _cluster_vectors(rng, n_archs4, n_clusters=6, spread=0.35)
    mm = np.memmap(d / "sample_embeddings.float16.mmap", dtype=np.float16,
                   mode="w+", shape=(n_archs4, EMB_DIM))
    mm[:] = vecs.astype(np.float16)
    mm.flush()
    del mm

    # Species tracks cluster identity so a species color-by shows real structure.
    species = (cluster_id % 2).astype(np.int16)
    pd.DataFrame({
        "global_index": np.arange(n_archs4, dtype=np.int64),
        "shard_idx": np.zeros(n_archs4, dtype=np.int32),
        "shard_file": ["shard_000.npy"] * n_archs4,
        "row_in_shard": np.arange(n_archs4, dtype=np.int32),
        "geo_accession": [f"GSM{9000000 + i}" for i in range(n_archs4)],
        "species_id": species,
    }).to_parquet(d / "sample_locations.parquet", index=False)

    (d / "embedding_manifest.json").write_text(json.dumps({
        "total_samples": int(n_archs4),
        "embedding_dim": EMB_DIM,
        "embedding_dtype": "float16",
        "normalization": "log1p_tpm",
        "feature_type": "flash",
        "l2_normalized": False,
    }, indent=2))

    # The precompute preflight also checks these; empty stand-ins are enough
    # because no test runs the model.
    for rel in [
        "checkpoints_performer/r7hnr92k/best_model.pt",
        "data/osdr/metadata/selected_sample_metadata.tsv",
        "data/ensembl/orthologs_one2one.txt",
        "data/gencode/gencode_v49_mouse_gene_exon_lengths.csv",
        "data/archs4/train_orthologs/canonical_genes.csv",
    ]:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            p.write_bytes(b"stub" * 2000)  # > 4096 B so the LFS guard passes
    return vecs, cluster_id


def _finish_cache(cache_dir, archs4_vecs, osdr_vecs, osdr_cluster, meta,
                  with_umap) -> dict:
    """Write the coordinate, identity, and density artifacts.

    Coordinates are a genuine PCA of the L2-normalized joint corpus rather than
    random numbers, so the 2D layout actually reflects the 512-d structure and
    what the tests measure about the picture is a property of the data.
    """
    from sklearn.decomposition import PCA

    (cache_dir / "density").mkdir(parents=True, exist_ok=True)
    n_archs4, n_osdr = len(archs4_vecs), len(osdr_vecs)
    total = n_archs4 + n_osdr

    def l2(x):
        n = np.linalg.norm(x, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return (x / n).astype(np.float32)

    joint = np.vstack([l2(archs4_vecs), l2(osdr_vecs)])

    # Species: ARCHS4 alternates by construction in the stub; OSDR is all mouse.
    archs4_species = np.load(cache_dir / "_archs4_species.npy") \
        if (cache_dir / "_archs4_species.npy").exists() else None
    if archs4_species is None:
        archs4_species = np.zeros(n_archs4, dtype=np.int8)
    species_full = np.concatenate([archs4_species,
                                   np.ones(n_osdr, dtype=np.int8)]).astype(np.int8)

    pd.DataFrame({
        "dataset": np.concatenate([np.zeros(n_archs4, np.int8), np.ones(n_osdr, np.int8)]),
        "src_index": np.concatenate([np.arange(n_archs4, dtype=np.int32),
                                     np.arange(n_osdr, dtype=np.int32)]),
        "species_id": species_full,
    }).to_parquet(cache_dir / "points_meta.parquet", index=False)

    pd.DataFrame({
        "geo_accession": [f"GSM{9000000 + i}" for i in range(n_archs4)]
    }).to_parquet(cache_dir / "archs4_geo.parquet", index=False)

    # --- Coordinates: a real PCA of the joint normalized corpus ------------
    stats: dict = {"n_archs4": n_archs4, "n_osdr": n_osdr, "total": total}
    pca = PCA(n_components=3, random_state=0).fit(joint)
    coords3 = pca.transform(joint).astype(np.float32)
    stats["pca_explained_variance_ratio"] = pca.explained_variance_ratio_.tolist()
    stats["pca_pc1_pct"] = float(pca.explained_variance_ratio_[0] * 100)
    stats["pca_cum_pct"] = float(pca.explained_variance_ratio_.sum() * 100)

    _write_coords(coords3[:, :2], cache_dir / "coords_pca2.parquet")
    _write_coords(coords3[:, :3], cache_dir / "coords_pca3.parquet")
    stats["density_pca2"] = _write_density(coords3[:, :2], cache_dir / "density" / "pca2.png")

    if with_umap:
        # A cheap stand-in for UMAP: an isotropic rotation of the PCA coords with
        # mild local jitter. The test suite only needs a second, differently
        # shaped coordinate set to prove the method toggle wires through.
        rot = np.array([[0.8, -0.6], [0.6, 0.8]], dtype=np.float32)
        u2 = coords3[:, :2] @ rot
        u3 = np.column_stack([u2, coords3[:, 2]]).astype(np.float32)
        _write_coords(u2, cache_dir / "coords_umap2.parquet")
        _write_coords(u3, cache_dir / "coords_umap3.parquet")
        stats["density_umap2"] = _write_density(u2, cache_dir / "density" / "umap2.png")

    (cache_dir / "projection_stats.json").write_text(json.dumps(stats, indent=2))
    return {"n_archs4": n_archs4, "n_osdr": n_osdr, "total": total,
            "osdr_cluster": osdr_cluster, "coords": coords3, "stats": stats}


def _write_coords(coords: np.ndarray, path: Path) -> None:
    cols = ["x", "y", "z"][: coords.shape[1]]
    pd.DataFrame({c: coords[:, i].astype(np.float32) for i, c in enumerate(cols)}) \
        .to_parquet(path, index=False)


def _write_density(coords2d: np.ndarray, path: Path, res: int = 128) -> dict:
    from PIL import Image

    x, y = coords2d[:, 0], coords2d[:, 1]
    xlo, xhi = float(x.min()), float(x.max())
    ylo, yhi = float(y.min()), float(y.max())
    H, _, _ = np.histogram2d(x, y, bins=res, range=[[xlo, xhi], [ylo, yhi]])
    dens = np.log1p(H.T)
    if dens.max() > 0:
        dens = dens / dens.max()
    rgba = np.zeros((res, res, 4), dtype=np.uint8)
    rgba[..., 0] = 34
    rgba[..., 1] = 90
    rgba[..., 2] = 140
    rgba[..., 3] = (np.clip(dens * 2.0, 0, 1) * 220).astype(np.uint8)
    Image.fromarray(rgba[::-1], mode="RGBA").save(path)
    return {"x0": xlo, "x1": xhi, "y0": ylo, "y1": yhi}


def _write_archs4_metadata(cache_dir: Path, cluster_id: np.ndarray) -> None:
    """A stand-in for the GEO metadata join, keyed to the latent clusters.

    Tissue tracks cluster identity so the ARCHS4 half of the shared "Tissue"
    color-by has real structure to find, exactly as species does. The raw values
    are written in ARCHS4's own free-text register ("liver", "whole blood") and
    canonicalized through ``manifold.tissue`` here, so the fixture exercises the
    real mapping rather than pre-baking the answer.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from manifold import tissue as tissue_map

    n = len(cluster_id)
    raw = [ARCHS4_SOURCES[int(c) % len(ARCHS4_SOURCES)] for c in cluster_id]
    # A slice with no usable label at all, so the Unknown path is always covered.
    for i in range(0, n, 37):
        raw[i] = ""
    pd.DataFrame({
        "global_index": np.arange(n, dtype=np.int32),
        "geo_accession": [f"GSM{9000000 + i}" for i in range(n)],
        "series_id": [f"GSE{5000 + int(c)}" for c in cluster_id],
        "title": [f"sample {i}" for i in range(n)],
        "source_name": raw,
        "characteristics": [f"tissue: {r}" if r else "" for r in raw],
        "tissue": [tissue_map.canonical_tissue(r) for r in raw],
    }).to_parquet(cache_dir / "archs4_metadata.parquet", index=False)


def build_all(root: Path, n_archs4: int = 4000, n_osdr: int = 300,
              seed: int = 7, with_umap: bool = True,
              with_archs4_meta: bool = True) -> dict:
    """Build both halves of the fixture under ``root`` and return a descriptor.

    ``root/bridge_rna`` stands in for the Bridge RNA repository and
    ``root/cache`` for Bridge Manifold's cache. Point the two environment
    variables ``BRIDGE_RNA_ROOT`` and ``MANIFOLD_CACHE_DIR`` at them before
    importing ``manifold.paths``.
    """
    rna_root = root / "bridge_rna"
    cache_dir = root / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    archs4_vecs, archs4_cluster = build_bridge_rna_stub(rna_root, n_archs4, seed=seed)
    loc = pd.read_parquet(rna_root / "archs4_sample_embeddings_full" / "sample_locations.parquet")
    np.save(cache_dir / "_archs4_species.npy",
            loc["species_id"].to_numpy().astype(np.int8))
    if with_archs4_meta:
        _write_archs4_metadata(cache_dir, archs4_cluster)

    rng = np.random.default_rng(seed + 1)
    osdr_vecs, osdr_cluster = _cluster_vectors(rng, n_osdr, n_clusters=4, spread=0.28)
    np.save(cache_dir / "osdr_sample_embeddings.float32.npy", osdr_vecs)

    studies = [f"OSD-{100 + (int(c) * 7) + (i % 3)}" for i, c in enumerate(osdr_cluster)]
    meta = pd.DataFrame({
        "sample_key": [f"{studies[i]}|SAMPLE_{i:04d}" for i in range(n_osdr)],
        "tissue": [TISSUES[int(c) % len(TISSUES)] for c in osdr_cluster],
        "spaceflight": [SPACEFLIGHT[i % len(SPACEFLIGHT)] for i in range(n_osdr)],
        "strain": [STRAINS[int(c) % len(STRAINS)] for c in osdr_cluster],
        "sex": [SEXES[i % len(SEXES)] for i in range(n_osdr)],
        "genotype": [GENOTYPES[i % len(GENOTYPES)] for i in range(n_osdr)],
        "study": studies,
        "habitat": [HABITATS[i % len(HABITATS)] for i in range(n_osdr)],
        "duration": [f"{30 + (int(c) * 3)} day" for c in osdr_cluster],
        "diet": [DIETS[i % len(DIETS)] for i in range(n_osdr)],
    })
    meta.to_parquet(cache_dir / "osdr_metadata.parquet", index=False)

    desc = _finish_cache(cache_dir, archs4_vecs, osdr_vecs, osdr_cluster, meta,
                         with_umap)
    (cache_dir / "_archs4_species.npy").unlink(missing_ok=True)
    desc["bridge_rna_root"] = rna_root
    desc["cache_dir"] = cache_dir
    desc["osdr_metadata"] = meta
    desc["archs4_cluster"] = archs4_cluster
    return desc
