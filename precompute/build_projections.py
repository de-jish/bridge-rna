#!/usr/bin/env python3
"""Phase 2+4: build the joint projection artifacts the serving app loads.

Produces, for the union of ARCHS4 (940,455) and OSDR (2,108) embeddings:
  - cache/coords_pca2.parquet, coords_pca3.parquet   (L2 -> IncrementalPCA-50)
  - cache/coords_umap2.parquet, coords_umap3.parquet  (landmark UMAP on PCA-50)
  - cache/density/*.png                               (numpy density rasters)
  - cache/projection_stats.json                       (variance, extents, timings)

Global point order is fixed as [all ARCHS4 in global_index order, then all
OSDR in row order]; every artifact shares this order so the app can index
positionally. Row i < N_ARCHS4 is ARCHS4 memmap[i]; row i >= N_ARCHS4 is OSDR
npy[i - N_ARCHS4].

Design notes:
  * L2-normalize before any reduction. Raw ARCHS4 norms span 6.7-25.5 and PC1
    would otherwise be a magnitude axis (REFERENCE.md section 4).
  * UMAP is a landmark fit on a stratified subsample (all OSDR + an ARCHS4
    sample) then .transform() of the rest, because a direct 940k fit is hours.
  * Density is a plain numpy 2D histogram, not datashader - fewer fragile deps,
    full control, and trivially fast at this scale.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from manifold import paths, preflight  # noqa: E402

EMB_DIM = 512

# Density raster ramp. Occupancy is heavy-tailed, so the colour scale is
# normalized against this percentile of the OCCUPIED bins rather than the
# global max; see render_density for the measurement that motivated it.
DENSITY_CLIP_PCT = 99.5
# Faintest visible alpha for a bin that has any points at all, so sparse
# structure (thin filaments, small islands) does not vanish into the canvas.
DENSITY_ALPHA_FLOOR = 0.22


def log(msg: str) -> None:
    print(f"[proj] {msg}", flush=True)


def open_archs4() -> tuple[np.memmap, pd.DataFrame]:
    manifest = json.loads(paths.ARCHS4_MANIFEST.read_text())
    n = int(manifest["total_samples"])
    d = int(manifest["embedding_dim"])
    assert d == EMB_DIM, f"unexpected embedding dim {d}"
    mm = np.memmap(paths.ARCHS4_MMAP, dtype=np.float16, mode="r", shape=(n, d))
    loc = pd.read_parquet(paths.ARCHS4_LOCATIONS)
    loc = loc.sort_values("global_index").reset_index(drop=True)
    assert len(loc) == n, "sample_locations length mismatch"
    return mm, loc


def l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (x / n).astype(np.float32)


def chunks(n: int, size: int):
    for s in range(0, n, size):
        yield s, min(s + size, n)


def fit_incremental_pca(mm, n_archs4, osdr_norm, sample_idx, n_components, batch):
    from sklearn.decomposition import IncrementalPCA

    ipca = IncrementalPCA(n_components=n_components)
    # Fit in mini-batches over a shuffled ARCHS4 subsample plus all OSDR.
    log(f"fitting IncrementalPCA-{n_components} on {len(sample_idx)} ARCHS4 + "
        f"{len(osdr_norm)} OSDR samples")
    t0 = time.time()
    buf = []
    buf_n = 0
    for s, e in chunks(len(sample_idx), batch):
        idx = np.sort(sample_idx[s:e])
        block = l2_normalize(np.asarray(mm[idx], dtype=np.float32))
        buf.append(block)
        buf_n += len(block)
        if buf_n >= batch:
            ipca.partial_fit(np.concatenate(buf))
            buf, buf_n = [], 0
    # Fold OSDR into the fit so its manifold is represented.
    tail = ([np.concatenate(buf)] if buf else []) + [osdr_norm]
    ipca.partial_fit(np.concatenate(tail))
    log(f"PCA fit done in {time.time() - t0:.0f}s; "
        f"PC1 var {ipca.explained_variance_ratio_[0]*100:.1f}%, "
        f"cum50 {ipca.explained_variance_ratio_.sum()*100:.1f}%")
    return ipca


def transform_all_pca(ipca, mm, n_archs4, osdr_norm, batch):
    """Stream the full corpus through the fitted PCA into a (N, 50) float32 array."""
    total = n_archs4 + len(osdr_norm)
    out = np.empty((total, ipca.n_components_), dtype=np.float32)
    t0 = time.time()
    for s, e in chunks(n_archs4, batch):
        block = l2_normalize(np.asarray(mm[s:e], dtype=np.float32))
        out[s:e] = ipca.transform(block)
        if (s // batch) % 5 == 0:
            log(f"  pca transform {e}/{n_archs4} ({time.time()-t0:.0f}s)")
    out[n_archs4:] = ipca.transform(osdr_norm)
    log(f"PCA transform of {total} points done in {time.time()-t0:.0f}s")
    return out


def stratified_fit_index(n_archs4, species, n_sample, seed):
    """Indices into the ARCHS4 block for the UMAP landmark fit, balanced by species."""
    rng = np.random.default_rng(seed)
    human = np.where(species == 0)[0]
    mouse = np.where(species == 1)[0]
    per = n_sample // 2
    pick = np.concatenate([
        rng.choice(human, size=min(per, len(human)), replace=False),
        rng.choice(mouse, size=min(per, len(mouse)), replace=False),
    ])
    return np.sort(pick)


def run_umap(pca_all, n_archs4, species, n_components, fit_sample, seed, transform_batch):
    import umap

    total = pca_all.shape[0]
    fit_idx = stratified_fit_index(n_archs4, species, fit_sample, seed)
    # Always include all OSDR points (the tail) in the landmark fit.
    osdr_idx = np.arange(n_archs4, total)
    landmark_idx = np.concatenate([fit_idx, osdr_idx])
    log(f"UMAP-{n_components}d landmark fit on {len(landmark_idx)} points "
        f"({len(fit_idx)} ARCHS4 + {len(osdr_idx)} OSDR)")
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=30,
        min_dist=0.1,
        metric="euclidean",  # PCA-50 space is already L2-normalized + linear
        random_state=seed,
        verbose=True,
    )
    t0 = time.time()
    emb_landmark = reducer.fit_transform(pca_all[landmark_idx])
    log(f"UMAP fit done in {time.time()-t0:.0f}s; transforming remaining points")

    out = np.empty((total, n_components), dtype=np.float32)
    out[landmark_idx] = emb_landmark.astype(np.float32)
    remaining = np.setdiff1d(np.arange(total), landmark_idx, assume_unique=False)
    t1 = time.time()
    for s, e in chunks(len(remaining), transform_batch):
        idx = remaining[s:e]
        out[idx] = reducer.transform(pca_all[idx]).astype(np.float32)
        log(f"  umap transform {e}/{len(remaining)} ({time.time()-t1:.0f}s)")
    log(f"UMAP transform done in {time.time()-t1:.0f}s")
    return out


def write_coords(coords: np.ndarray, path: Path) -> None:
    cols = ["x", "y", "z"][: coords.shape[1]]
    df = pd.DataFrame({c: coords[:, i].astype(np.float32) for i, c in enumerate(cols)})
    df.to_parquet(path, index=False)
    log(f"wrote {path.name} {coords.shape}")


def render_density(coords2d: np.ndarray, name: str, res: int = 2048) -> dict:
    """Log-scaled 2D histogram -> navy-to-teal PNG underlay. Returns placement extent."""
    from PIL import Image

    x, y = coords2d[:, 0], coords2d[:, 1]
    # Robust extent (1st-99th percentile) so a few outliers don't shrink the map.
    xlo, xhi = np.percentile(x, [0.2, 99.8])
    ylo, yhi = np.percentile(y, [0.2, 99.8])
    H, _, _ = np.histogram2d(
        x, y, bins=res, range=[[xlo, xhi], [ylo, yhi]]
    )
    H = H.T  # histogram2d returns [x, y]; image wants [row=y, col=x]
    dens = np.log1p(H)
    # Normalize against a high percentile of the OCCUPIED bins, not the global
    # max. Bin occupancy is heavy-tailed (median 2 points, max 638 on the real
    # corpus), so dividing by the max crushes the whole ramp into its bottom
    # fraction: measured, only 0.78% of occupied bins cleared the 0.5 threshold
    # where the teal half begins, leaving the high end of the ramp unused and
    # the map a flat two-tone wash. Clipping at a percentile spends the ramp on
    # the range that actually has pixels in it.
    occupied = dens > 0
    if occupied.any():
        ref = float(np.percentile(dens[occupied], DENSITY_CLIP_PCT))
        dens = np.clip(dens / max(ref, 1e-9), 0.0, 1.0)
    # Map density -> RGBA: transparent where empty, navy->teal->white where dense.
    rgba = np.zeros((res, res, 4), dtype=np.uint8)
    # Colour ramp anchored to the plot navy.
    c0 = np.array([14, 29, 52])      # PLOT_BG navy (low)
    c1 = np.array([34, 90, 140])     # mid blue
    c2 = np.array([34, 199, 189])    # teal (header line) high
    t = dens[..., None]
    lo = c0 + (c1 - c0) * np.clip(t * 2, 0, 1)
    hi = c1 + (c2 - c1) * np.clip((t - 0.5) * 2, 0, 1)
    col = np.where(t < 0.5, lo, hi).astype(np.uint8)
    rgba[..., :3] = col
    # Alpha grows with density; empty cells stay fully transparent. The old
    # 2.2x slope saturated alpha at dens 0.4545, i.e. *before* the colour ramp
    # reached its teal half at 0.5, so the densest cores were indistinguishable
    # from merely-busy regions. Ramp alpha across the same 0..1 span the colour
    # uses, with a floor so sparse-but-occupied bins still read.
    alpha = np.where(occupied, DENSITY_ALPHA_FLOOR
                     + (1.0 - DENSITY_ALPHA_FLOOR) * dens, 0.0)
    rgba[..., 3] = (alpha * 235).astype(np.uint8)
    # Flip vertically because image row 0 is the top but y increases upward.
    img = Image.fromarray(rgba[::-1], mode="RGBA")
    out = paths.DENSITY_DIR / f"{name}.png"
    img.save(out)
    log(f"wrote density {out.name}")
    return {"x0": float(xlo), "x1": float(xhi), "y0": float(ylo), "y1": float(yhi)}


def rerender_density_only() -> None:
    """Redraw the density rasters from cached coordinates and refresh their extents.

    The rasters are a pure function of the 2-d coordinates, so tuning the colour
    ramp does not require repeating PCA, UMAP, or the index build.
    """
    if not paths.PROJECTION_STATS_JSON.exists():
        raise SystemExit("ABORT: no projection_stats.json; run a full build first.")
    stats = json.loads(paths.PROJECTION_STATS_JSON.read_text())
    todo = [("pca2", paths.COORDS_PCA2), ("umap2", paths.COORDS_UMAP2)]
    rendered = 0
    for name, path in todo:
        if not path.exists():
            log(f"skipping {name}: {path.name} not present")
            continue
        coords = pd.read_parquet(path).to_numpy(dtype=np.float64)[:, :2]
        stats[f"density_{name}"] = render_density(coords, name)
        rendered += 1
    if not rendered:
        raise SystemExit("ABORT: no coordinate parquets found to render from.")
    paths.PROJECTION_STATS_JSON.write_text(json.dumps(stats, indent=2))
    log(f"re-rendered {rendered} density raster(s); extents refreshed in "
        f"{paths.PROJECTION_STATS_JSON.name}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build joint PCA/UMAP/density artifacts.")
    ap.add_argument("--pca-fit-sample", type=int, default=60000)
    ap.add_argument("--umap-fit-sample", type=int, default=120000)
    ap.add_argument("--pca-components", type=int, default=50)
    ap.add_argument("--batch", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--archs4-limit", type=int, default=0, help="Debug: cap ARCHS4 rows.")
    ap.add_argument("--skip-umap", action="store_true")
    ap.add_argument("--density-only", action="store_true",
                    help="Re-render the density rasters from the cached coordinate "
                         "parquets and update their extents, then exit. For tuning "
                         "the colour ramp without repeating the projection build.")
    args = ap.parse_args()

    paths.ensure_cache_dirs()

    if args.density_only:
        rerender_density_only()
        return
    preflight.require(
        [("ARCHS4 memmap", paths.ARCHS4_MMAP), ("ARCHS4 locations", paths.ARCHS4_LOCATIONS),
         ("OSDR embeddings", paths.OSDR_EMBEDDINGS_NPY)],
        "projection build",
    )

    mm, loc = open_archs4()
    n_archs4 = args.archs4_limit or len(loc)
    species = loc["species_id"].to_numpy()[:n_archs4].astype(np.int8)
    log(f"ARCHS4 rows: {n_archs4} (human {int((species==0).sum())}, mouse {int((species==1).sum())})")

    osdr_emb = np.load(paths.OSDR_EMBEDDINGS_NPY).astype(np.float32)
    osdr_norm = l2_normalize(osdr_emb)
    log(f"OSDR embeddings: {osdr_emb.shape}")

    # Gate: the embeddings and their metadata are indexed positionally by every
    # downstream artifact, so a length mismatch does not fail - it silently
    # attributes each OSDR point to a different sample's metadata.
    if paths.OSDR_METADATA_PARQUET.exists():
        n_meta = len(pd.read_parquet(paths.OSDR_METADATA_PARQUET))
        if n_meta != len(osdr_emb):
            raise SystemExit(
                f"ABORT: OSDR embeddings ({len(osdr_emb)}) and metadata ({n_meta}) "
                "disagree on row count. They are joined positionally, so every "
                "OSDR point would carry the wrong labels. Re-run "
                "precompute/embed_osdr.py."
            )

    stats: dict = {"n_archs4": int(n_archs4), "n_osdr": int(len(osdr_norm)),
                   "total": int(n_archs4 + len(osdr_norm))}

    def save_stats() -> None:
        """Persist after every stage, so a later failure does not lose the earlier work."""
        paths.PROJECTION_STATS_JSON.write_text(json.dumps(stats, indent=2))


    # --- Identity table (dataset + src_index), fixed global order ----------
    dataset = np.concatenate([np.zeros(n_archs4, np.int8), np.ones(len(osdr_norm), np.int8)])
    src_index = np.concatenate([np.arange(n_archs4, dtype=np.int32),
                                np.arange(len(osdr_norm), dtype=np.int32)])
    species_full = np.concatenate([species, np.ones(len(osdr_norm), np.int8)])  # OSDR all mouse
    geo = loc["geo_accession"].astype(str).to_numpy()[:n_archs4]
    meta_df = pd.DataFrame({
        "dataset": dataset, "src_index": src_index, "species_id": species_full,
    })
    meta_df.to_parquet(paths.POINTS_META_PARQUET, index=False)
    # geo accessions for archs4 (for batch/study context), stored compactly.
    pd.DataFrame({"geo_accession": geo}).to_parquet(
        paths.ARCHS4_GEO_PARQUET, index=False)
    log("wrote points_meta.parquet + archs4_geo.parquet")

    # --- PCA ---------------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    pca_sample = rng.choice(n_archs4, size=min(args.pca_fit_sample, n_archs4), replace=False)
    ipca = fit_incremental_pca(mm, n_archs4, osdr_norm, pca_sample, args.pca_components, args.batch)
    stats["pca_explained_variance_ratio"] = ipca.explained_variance_ratio_.tolist()
    stats["pca_pc1_pct"] = float(ipca.explained_variance_ratio_[0] * 100)
    stats["pca_cum_pct"] = float(ipca.explained_variance_ratio_.sum() * 100)

    pca_all = transform_all_pca(ipca, mm, n_archs4, osdr_norm, args.batch)
    write_coords(pca_all[:, :2], paths.COORDS_PCA2)
    write_coords(pca_all[:, :3], paths.COORDS_PCA3)
    stats["density_pca2"] = render_density(pca_all[:, :2], "pca2")
    save_stats()

    # --- UMAP --------------------------------------------------------------
    if not args.skip_umap:
        umap2 = run_umap(pca_all, n_archs4, species, 2, args.umap_fit_sample, args.seed, args.batch)
        write_coords(umap2, paths.COORDS_UMAP2)
        stats["density_umap2"] = render_density(umap2, "umap2")
        save_stats()
        umap3 = run_umap(pca_all, n_archs4, species, 3, args.umap_fit_sample, args.seed, args.batch)
        write_coords(umap3, paths.COORDS_UMAP3)
        save_stats()

    save_stats()
    log(f"ALL DONE. stats -> {paths.PROJECTION_STATS_JSON.name}")


if __name__ == "__main__":
    main()
