"""Probe CellWhisperer's archs4_geo full_data.h5ad over HTTP range requests: list obs columns without downloading."""
import sys, time, h5py
sys.path.insert(0, "experiments/text_search"); from httpfile import HTTPRangeFile
URL = "https://medical-epigenomics.org/papers/schaefer2025cellwhisperer/data/datasets/archs4_geo/full_data.h5ad"
t = time.time()
f = HTTPRangeFile(URL)
h = h5py.File(f, "r")
print("top", list(h.keys()), flush=True)
obs = h["obs"]
print("obs attrs", dict(obs.attrs), flush=True)
for k in obs.keys():
    o = obs[k]
    if isinstance(o, h5py.Dataset):
        print(k, "dataset", o.shape, o.dtype, o.chunks, o.compression, flush=True)
    else:
        print(k, "group", list(o.keys()), {kk: (o[kk].shape, o[kk].dtype) for kk in o.keys()}, flush=True)
print("elapsed", time.time() - t)
