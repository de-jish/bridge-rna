"""One-time read of CellWhisperer's curated captions from its archs4_geo h5ad over HTTP ranges.

Reads only obs/geo_id, obs/organism and obs/natural_language_annotation (about
a few hundred MB of a 163 GB file) and writes exp_data/cw_captions.parquet.
"""
import sys, time
import h5py, numpy as np, pandas as pd
sys.path.insert(0, "experiments/text_search")
from httpfile import HTTPRangeFile

URL = "https://medical-epigenomics.org/papers/schaefer2025cellwhisperer/data/datasets/archs4_geo/full_data.h5ad"
t = time.time()
f = HTTPRangeFile(URL, block=1 << 23)
h = h5py.File(f, "r")
obs = h["obs"]

def cat(name):
    g = obs[name]
    cats = g["categories"][:]
    codes = g["codes"][:]
    cats = np.array([c.decode() if isinstance(c, bytes) else c for c in cats], dtype=object)
    out = np.where(codes >= 0, cats[np.clip(codes, 0, None)], "")
    print(name, "done", round(time.time() - t), "s", f.fetched // 2**20, "MB", flush=True)
    return out

geo = obs["geo_id"][:]
geo = np.array([g.decode() if isinstance(g, bytes) else g for g in geo], dtype=object)
print("geo_id", len(geo), geo[:3], flush=True)
org = cat("organism")
cap = cat("natural_language_annotation")
pd.DataFrame({"geo_accession": geo, "organism": org, "caption": cap}).to_parquet("exp_data/cw_captions.parquet")
print("wrote", len(geo), "elapsed", round(time.time() - t), "fetched MB", f.fetched // 2**20)
