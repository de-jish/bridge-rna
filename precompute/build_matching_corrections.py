"""Build optional correction geometry; never overwrite source embeddings."""
import argparse
import json
from bridge_rna.matching import build_artifact, FIT_SIZE, SEED
from bridge_rna.retrieval import _load_archs4_index
from bridge_rna.config import ROOT, EMBEDDING_DIR


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit-size', type=int, default=FIT_SIZE)
    parser.add_argument('--seed', type=int, default=SEED)
    args = parser.parse_args()
    index, _, _ = _load_archs4_index()
    print(f'Fitting corrections and measuring norms for {len(index):,} reference rows...', flush=True)
    metadata = build_artifact(index, index.filename, EMBEDDING_DIR/'sample_locations.parquet',
                              ROOT/'cache'/'matching_corrections.npz',
                              fit_size=args.fit_size, seed=args.seed)
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
