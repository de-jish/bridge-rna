"""Fixed data preparation for the text-search experiment.

Writes exp_data/split_s{seed}/:
  meta.parquet      every ARCHS4 row: global_index, series_id, species, tissue,
                    text (title. source. characteristics), split (train|test)
  gallery.npy       global indices of the held-out evaluation pool (labelled
                    tissue only, de-duplicated against train)
  captions.npy      global indices of the caption-retrieval pool (subset of
                    gallery, one sample per series, unique text)
  leak.json         the de-duplication record

The split is by GEO series_id, never by sample: a series is held out when the
md5 of f"{seed}:{series_id}" falls in the lowest 15%. Rows with no series_id
are training-only. A held-out sample whose plain cosine distance to any
training sample is below 1e-6 (an identical profile re-deposited under another
series, bridge_rna/similarity.py) is dropped from the gallery.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import os

# The repository root that holds the LFS embedding index. On Josh's Mac the
# worktree's index is not checked out, so it defaults to the main checkout;
# in a fresh clone (the cloud) set BRIDGE_RNA_ROOT to the clone itself.
ROOT = Path(os.environ.get("BRIDGE_RNA_ROOT", "/Users/josh/Bridge-RNA"))
MMAP = ROOT / "archs4_sample_embeddings_full/sample_embeddings.float16.mmap"
LOCATIONS = ROOT / "archs4_sample_embeddings_full/sample_locations.parquet"
# Snapshots of the gitignored cache files this experiment reads, committed to
# the experiment branch on 2026-11-05 with Josh's approval so a cloud session
# can run it. Copied byte for byte from /Users/josh/Bridge-RNA/cache/.
DATA = Path(__file__).resolve().parent / "data"
N, D = 940_455, 512
EXP = Path(__file__).resolve().parents[2] / "exp_data"
TEST_PCT = 15
GALLERY_SIZE = 40_000
CAPTION_POOL = 5_000
IDENTICAL = 1e-6


def expression() -> np.memmap:
    return np.memmap(MMAP, dtype=np.float16, mode="r", shape=(N, D))


def unit_rows(idx: np.ndarray | None = None, block: int = 100_000) -> np.ndarray:
    """L2-normalised float32 rows of the memmap (invariant 2)."""
    mm = expression()
    idx = np.arange(N) if idx is None else np.asarray(idx)
    out = np.empty((len(idx), D), dtype=np.float32)
    for a in range(0, len(idx), block):
        x = np.asarray(mm[np.sort(idx[a:a + block])], dtype=np.float32)
        order = np.argsort(np.argsort(idx[a:a + block]))
        x = x[order]
        out[a:a + block] = x / np.linalg.norm(x, axis=1, keepdims=True)
    return out


def sample_text(df: pd.DataFrame) -> pd.Series:
    parts = [df[c].fillna("").astype(str).str.strip() for c in ("title", "source_name", "characteristics")]
    return (parts[0] + ". " + parts[1] + ". " + parts[2]).str.replace(r"\s+", " ", regex=True)


def held_out(series: pd.Series, seed: int) -> np.ndarray:
    def h(s: str) -> bool:
        if not s:
            return False
        return int(hashlib.md5(f"{seed}:{s}".encode()).hexdigest()[:8], 16) % 100 < TEST_PCT
    uniq = {s: h(s) for s in series.unique()}
    return series.map(uniq).to_numpy(bool)


def max_cos_to_train(q: np.ndarray, train_idx: np.ndarray) -> np.ndarray:
    """Max plain cosine of each query row to any training row, exact near the top.

    One float32 GPU pass (CUDA, else MPS, else CPU) over all training rows finds
    the best candidate per query; candidates within 1e-4 are recomputed in float64.
    """
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available()
                       else "mps" if torch.backends.mps.is_available() else "cpu")
    qt = torch.from_numpy(q).to(dev)
    best = torch.full((len(q),), -2.0, device=dev)
    arg = torch.zeros((len(q),), dtype=torch.long, device=dev)
    mm = expression()
    for a in range(0, len(train_idx), 200_000):
        ti = train_idx[a:a + 200_000]
        x = torch.from_numpy(np.asarray(mm[ti], dtype=np.float32)).to(dev)
        x = x / x.norm(dim=1, keepdim=True)
        for b in range(0, len(q), 8192):
            s = qt[b:b + 8192] @ x.T
            v, i = s.max(dim=1)
            upd = v > best[b:b + 8192]
            best[b:b + 8192] = torch.where(upd, v, best[b:b + 8192])
            arg[b:b + 8192] = torch.where(upd, torch.from_numpy(ti).to(dev)[i], arg[b:b + 8192])
    best = best.cpu().numpy().astype(np.float64)
    arg = arg.cpu().numpy()
    near = np.nonzero(1 - best < 1e-4)[0]
    if len(near):
        x = np.asarray(mm[np.sort(arg[near])], dtype=np.float64)
        x = x[np.argsort(np.argsort(arg[near]))]
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        qq = q[near].astype(np.float64)
        qq /= np.linalg.norm(qq, axis=1, keepdims=True)
        best[near] = np.sum(x * qq, axis=1)
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    out = EXP / f"split_s{args.seed}"
    out.mkdir(parents=True, exist_ok=True)

    m = pd.read_parquet(DATA / "archs4_metadata.parquet")
    loc = pd.read_parquet(LOCATIONS)
    assert (loc.global_index.to_numpy() == np.arange(N)).all()
    assert (m.global_index.to_numpy() == np.arange(N)).all()
    assert (loc.geo_accession.to_numpy() == m.geo_accession.to_numpy()).all()
    meta = pd.DataFrame({
        "global_index": np.arange(N, dtype=np.int32),
        "series_id": m.series_id.fillna("").astype(str),
        "species": loc.species_id.to_numpy().astype(np.int8),  # 0 human, 1 mouse
        "tissue": m.tissue.astype(str),
        "text": sample_text(m),
    })
    test = held_out(meta.series_id, args.seed)
    meta["split"] = np.where(test, "test", "train")
    meta.to_parquet(out / "meta.parquet")
    print(f"split: {test.sum():,} test rows in {meta.series_id[test].nunique():,} series; "
          f"{(~test).sum():,} train", flush=True)

    rng = np.random.default_rng(args.seed)
    labelled = test & ~meta.tissue.isin(["Other", "Unknown"]).to_numpy()
    cand = np.sort(rng.choice(np.nonzero(labelled)[0], size=GALLERY_SIZE, replace=False))
    q = unit_rows(cand)
    best = max_cos_to_train(q, np.nonzero(~test)[0])
    dup = (1 - best) < IDENTICAL
    gallery = cand[~dup]
    np.save(out / "gallery.npy", gallery)

    # OSDR against ARCHS4 train: report, and drop exact duplicates from the OSDR test.
    osdr = np.load(DATA / "osdr_sample_embeddings.float32.npy").astype(np.float32)
    osdr /= np.linalg.norm(osdr, axis=1, keepdims=True)
    obest = max_cos_to_train(osdr, np.nonzero(~test)[0])
    np.save(out / "osdr_max_cos_train.npy", obest)

    g = meta.iloc[gallery]
    g = g[~g.text.duplicated(keep=False)]
    one = g.groupby("series_id").sample(n=1, random_state=args.seed)
    caps = np.sort(rng.choice(one.global_index.to_numpy(), size=min(CAPTION_POOL, len(one)), replace=False))
    np.save(out / "captions.npy", caps)

    rec = {
        "seed": args.seed, "test_rows": int(test.sum()),
        "test_series": int(meta.series_id[test].nunique()),
        "gallery_candidates": int(len(cand)), "gallery_dropped_identical": int(dup.sum()),
        "gallery": int(len(gallery)), "caption_pool": int(len(caps)),
        "gallery_max_cos_quantiles": np.quantile(best, [0.5, 0.9, 0.99, 1.0]).round(6).tolist(),
        "osdr_identical_to_train": int(((1 - obest) < IDENTICAL).sum()),
        "osdr_max_cos_quantiles": np.quantile(obest, [0.5, 0.9, 0.99, 1.0]).round(6).tolist(),
        "osdr_above_0.999": int((obest > 0.999).sum()),
        "seconds": round(time.time() - t0, 1),
    }
    (out / "leak.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
