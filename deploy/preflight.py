#!/usr/bin/env python3
"""Read-only scientific artifact checks and application smoke checks.

No rebuilding, downloads, inferred gene files, schema changes or migrations.
Uploads used by the smoke check go only to a temporary directory.
"""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["BRIDGE_RNA_ROOT"] = str(ROOT)
os.environ["MANIFOLD_CACHE_DIR"] = str(ROOT / "cache")
sys.dont_write_bytecode = True


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate():
    import numpy as np
    import pandas as pd
    from deploy.common import sha256
    from manifold import paths
    from manifold.bridge_rna import load_bridge_rna_symbols
    from precompute.embed_upload import _assert_gene_digest, load_model, preprocess_counts, embed
    from demo_osdr_top5 import _resolve_counts_path
    import torch

    required = [paths.CHECKPOINT, paths.ARCHS4_MMAP, paths.ARCHS4_LOCATIONS, paths.ARCHS4_MANIFEST,
                paths.OSDR_METADATA_TSV, paths.ORTHOLOGS_TXT, paths.MOUSE_EXON_LENGTHS_CSV,
                paths.CANONICAL_GENES_CSV, paths.POINTS_META_PARQUET, paths.OSDR_METADATA_PARQUET,
                paths.OSDR_EMBEDDINGS_NPY, paths.ARCHS4_GEO_PARQUET, paths.ARCHS4_METADATA_PARQUET,
                paths.COORDS_PCA2, paths.COORDS_PCA3, paths.COORDS_UMAP2, paths.COORDS_UMAP3,
                paths.PROJECTION_STATS_JSON]
    required += [p for p in (paths.COORDS_TSNE2, paths.COORDS_TSNE3) if p.exists()]
    for path in required:
        require(path.is_file() and os.access(path, os.R_OK), f"required readable artifact: {path}")
        with path.open("rb") as stream:
            require(not stream.read(40).startswith(b"version https://git-lfs"), f"LFS pointer: {path}")
    manifest = json.loads(paths.ARCHS4_MANIFEST.read_text())
    n, dim = int(manifest["total_samples"]), int(manifest["embedding_dim"])
    require(n > 0 and dim == 512 and manifest["embedding_dtype"] == "float16", "expected nonempty float16 N x 512 index")
    require(manifest.get("normalization") == "log1p_tpm" and manifest.get("feature_type") == "flash", "incompatible embedding preprocessing/model")
    require(paths.ARCHS4_MMAP.stat().st_size == n * dim * 2, "memmap byte size does not match manifest")
    loc = pd.read_parquet(paths.ARCHS4_LOCATIONS)
    require(len(loc) == n and np.array_equal(loc.global_index, np.arange(n)), "locations not in memmap row order")
    require({"geo_accession", "species_id"}.issubset(loc.columns), "locations schema")
    points = pd.read_parquet(paths.POINTS_META_PARQUET)
    osdr = pd.read_parquet(paths.OSDR_METADATA_PARQUET)
    m = len(osdr)
    require(m > 1 and {"sample_key", "study", "tissue", "spaceflight"}.issubset(osdr.columns), "OSDR cache schema")
    require(osdr.sample_key.is_unique and osdr.sample_key.notna().all(), "duplicate/missing OSDR identity")
    require(len(points) == n + m, "map/index row count mismatch")
    require(np.array_equal(points.dataset, np.r_[np.zeros(n), np.ones(m)]), "map corpus order mismatch")
    require(np.array_equal(points.src_index, np.r_[np.arange(n), np.arange(m)]), "map source indices mismatch")
    require(np.array_equal(points.species_id[:n], loc.species_id), "map/index species order mismatch")
    vectors = np.load(paths.OSDR_EMBEDDINGS_NPY, allow_pickle=False)
    require(vectors.shape == (m, 512) and vectors.dtype == np.float32, "OSDR embeddings must be float32 M x 512")
    require(np.isfinite(vectors).all() and (np.linalg.norm(vectors, axis=1) > 0).all(), "invalid OSDR embeddings")
    mmap = np.memmap(paths.ARCHS4_MMAP, dtype=np.float16, mode="r", shape=(n, dim))
    for start in range(0, n, 20000):
        block = np.asarray(mmap[start:start + 20000], dtype=np.float32)
        require(np.isfinite(block).all() and (np.linalg.norm(block, axis=1) > 0).all(), "invalid index vectors")
    geo = pd.read_parquet(paths.ARCHS4_GEO_PARQUET)
    rich = pd.read_parquet(paths.ARCHS4_METADATA_PARQUET)
    require(np.array_equal(geo.geo_accession.astype(str), loc.geo_accession.astype(str)), "map accession order mismatch")
    require(len(rich) == n and np.array_equal(rich.global_index, np.arange(n)), "GEO metadata row order mismatch")
    require(np.array_equal(rich.geo_accession.astype(str), loc.geo_accession.astype(str)), "GEO identity mismatch")
    require({"series_id", "title", "source_name", "characteristics", "tissue"}.issubset(rich.columns), "GEO metadata schema")
    for path in required:
        if path.name.startswith("coords_"):
            coords = pd.read_parquet(path)
            dims = int(path.stem[-1])
            require(coords.shape == (n + m, dims) and np.isfinite(coords.to_numpy()).all(), f"incompatible coordinates: {path}")
    require(isinstance(json.loads(paths.PROJECTION_STATS_JSON.read_text()), dict), "projection build record invalid")
    catalog = pd.read_csv(paths.OSDR_METADATA_TSV, sep="\t")
    require({"id.accession", "id.sample name", "counts_path", "study.factor value.spaceflight"}.issubset(catalog.columns), "catalog schema")
    catalog["sample_key"] = catalog["id.accession"].astype(str) + "|" + catalog["id.sample name"].astype(str)
    require(set(osdr.sample_key) <= set(catalog.sample_key), "cached identities absent from catalog")
    symbols = load_bridge_rna_symbols()
    genes = pd.read_csv(paths.CANONICAL_GENES_CSV).gene_symbol.astype(str).tolist()
    _assert_gene_digest(genes, symbols)
    # Exercise real counts resolution and the uploaded inference path against
    # its precomputed reference. No source metadata or artifacts are rewritten.
    counts = {}
    for raw in catalog.counts_path.dropna().unique():
        path = _resolve_counts_path(str(raw), paths.OSDR_DATA_DIR)
        require(path.is_file() and os.access(path, os.R_OK), f"unresolved counts file: {path}")
        counts[str(raw)] = path
    sample = next((row for _, row in catalog.iterrows()
                   if row.sample_key in set(osdr.sample_key) and str(row.counts_path) in counts), None)
    require(sample is not None, "no cached sample with readable source counts")
    x, _, mapped = preprocess_counts(counts[str(sample.counts_path)], str(sample["id.sample name"]),
                                     paths.ORTHOLOGS_TXT, paths.MOUSE_EXON_LENGTHS_CSV, genes, symbols)
    require(mapped > 0, "no mapped genes")
    model = load_model(paths.CHECKPOINT, torch.device("cpu"), symbols)
    live = embed(x, model, torch.device("cpu"))
    cached = vectors[osdr.sample_key.tolist().index(sample.sample_key)]
    cosine = float(live @ cached / (np.linalg.norm(live) * np.linalg.norm(cached)))
    require(cosine > 0.9999, f"checkpoint/counts vs cached vector mismatch: {cosine}")
    from bridge_rna.retrieval import run_cached_query_retrieval, run_cohort_retrieval
    hits = run_cached_query_retrieval(sample.sample_key, 5)
    require(len(hits) == 5 and "archs4_index" in hits, "cached retrieval failed")
    cohort_hits, _ = run_cohort_retrieval(osdr.sample_key.iloc[:2].tolist(), 5)
    require(len(cohort_hits) == 5, "pooled retrieval failed")
    from wsgi import application, RELEASE_ID
    with application.test_client() as client:
        for path in ("/", "/map", "/_dash-layout", "/_dash-dependencies", "/assets/00-tokens.css", "/assets/find-suggest.js", "/__release"):
            response = client.get(path)
            require(response.status_code == 200, f"startup failed: {path}")
            require(response.headers.get("X-Bridge-Release") == RELEASE_ID, "wrong startup release identity")
        # Prove the clean payload serves every stylesheet, font and brand asset.
        manifest = json.loads((ROOT / "ship.json").read_text())
        for name in manifest["files"]:
            if name.startswith("assets/"):
                response = client.get("/" + name)
                require(response.status_code == 200 and response.data == (ROOT / name).read_bytes(),
                        f"missing or incorrect runtime asset: {name}")
    return {"archs4_samples": n, "osdr_samples": m, "upload_reference_cosine": cosine,
            "artifacts": {os.path.relpath(p, ROOT): {"path": str(p.resolve()), "sha256": sha256(p)} for p in required},
            "counts_files_checked": len(counts), "tsne": paths.COORDS_TSNE2.exists()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = validate()
    if args.report:
        from deploy.common import atomic_json
        atomic_json(args.report, result)
    print(f"Preflight passed: {result['archs4_samples']} ARCHS4, {result['osdr_samples']} OSDR; "
          f"upload/reference cosine {result['upload_reference_cosine']:.8f}")


if __name__ == "__main__":
    main()
