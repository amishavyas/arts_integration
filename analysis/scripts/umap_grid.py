"""10x10 grid search over UMAP n_neighbors x min_dist on the snippet embeddings.

Each cell is an independent 2D UMAP fit of emb_0 .. emb_4095 from
aggregated_data_with_embeddings.csv. Fits run in parallel across cells (UMAP itself
is single-threaded when random_state is set). Rows of every embedding follow the CSV's
row order, so they line up with the dataframe the notebook loads.

Run in the `stats` env. Usage:
    python analysis/scripts/umap_grid.py                       # -> umap_grid.npz
    python analysis/scripts/umap_grid.py --metric cosine --n-jobs 16

Output npz keys:
    embeddings   float32 (n_nn, n_md, n_rows, 2); embeddings[i, j] is the fit with
                 n_neighbors[i], min_dist[j]
    n_neighbors  int   (n_nn,)
    min_dist     float (n_md,)
    metric       str
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from paths import DATA_ROOT

N_NEIGHBORS = [5, 10, 15, 20, 30, 50, 75, 100, 150, 200]
MIN_DIST = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.99]


def fit_one(X, n_neighbors, min_dist, metric, seed):
    from umap import UMAP

    t0 = time.time()
    emb = UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist,
               metric=metric, random_state=seed).fit_transform(X)
    print(f"n_neighbors={n_neighbors:>3} min_dist={min_dist:.2f} "
          f"({time.time() - t0:.1f}s)", flush=True)
    return emb.astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", type=Path, default=DATA_ROOT / "aggregated_data_with_embeddings.csv")
    ap.add_argument("--output", type=Path, default=DATA_ROOT / "umap_grid.npz")
    ap.add_argument("--metric", default="cosine")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-jobs", type=int, default=16)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    embcols = [c for c in df.columns if c.startswith("emb_")]
    X = df[embcols].to_numpy(np.float32)
    print(f"{X.shape[0]} rows x {X.shape[1]} dims, metric={args.metric}", flush=True)

    grid = [(nn, md) for nn in N_NEIGHBORS for md in MIN_DIST]
    results = Parallel(n_jobs=args.n_jobs)(
        delayed(fit_one)(X, nn, md, args.metric, args.seed) for nn, md in grid)

    embeddings = np.stack(results).reshape(len(N_NEIGHBORS), len(MIN_DIST), X.shape[0], 2)
    np.savez_compressed(args.output, embeddings=embeddings,
                        n_neighbors=np.array(N_NEIGHBORS), min_dist=np.array(MIN_DIST),
                        metric=args.metric)
    print(f"saved {embeddings.shape} -> {args.output}")


if __name__ == "__main__":
    main()
