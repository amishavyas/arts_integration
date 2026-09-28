"""Re-embed the installation's playback bank with the current TextEmbedder
model/backend (see text_embeddings.py).

data/utterance_library/aggregated_segments_with_embeddings.csv is a fixed
asset of this repo (the installation's playback library), not part of the
per-session recorded data under ARTS_DATA_ROOT - so it's addressed relative
to the repo root, not paths.DATA_ROOT.

Whenever text_embeddings.DEFAULT_MODEL or its backend changes, the bank has
to be re-embedded with it: the realtime pipeline embeds live queries with
that same module, and cosine similarity between vectors from two different
embedding spaces doesn't error, it just silently returns nonsense matches.

Backs up the existing embeddings once (skipped if a backup already exists),
drops the old emb_* columns, and re-embeds the `text` column in place.

Run in the `artsinteg` env - ~4400 rows, one at a time via mlx (see the
module docstring in text_embeddings.py for why), expect on the order of an
hour. This is a "run it yourself" job per analysis/CLAUDE.md:
    python analysis/scripts/reembed_utterance_bank.py
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd

from paths import REPO
from text_embeddings import DEFAULT_MODEL, TextEmbedder, add_embeddings


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    bank_dir = REPO / "data" / "utterance_library"
    parser.add_argument("--csv", type=Path, default=bank_dir / "aggregated_segments_with_embeddings.csv")
    parser.add_argument("--backend", choices=["mlx", "transformers"], default="mlx")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                         help="HF repo id or local mlx-converted path (e.g. a quantized model)")
    args = parser.parse_args()

    backup = args.csv.with_name(args.csv.stem + ".olmo3-7b.bak.csv")
    if backup.exists():
        print(f"backup already exists at {backup}, not overwriting")
    else:
        print(f"backing up current embeddings -> {backup}")
        shutil.copy2(args.csv, backup)

    df = pd.read_csv(args.csv)
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    print(f"{len(df)} rows, dropping {len(emb_cols)} old embedding columns")
    df = df.drop(columns=emb_cols)

    print(f"embedding with {args.model} (backend={args.backend})")
    embedder = TextEmbedder(model_name=args.model, backend=args.backend)
    df = add_embeddings(df, embedder=embedder)
    embedder.unload()

    new_emb_cols = [c for c in df.columns if c.startswith("emb_")]
    print(f"embedded {len(new_emb_cols)} dims")
    df.to_csv(args.csv, index=False)
    print(f"saved {df.shape} -> {args.csv}")


if __name__ == "__main__":
    main()
