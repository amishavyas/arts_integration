"""Aggregate segments.csv across sessions with full permissions and embed each segment.

The segment-level counterpart of aggregate_data.py: same session filter (both
participants gave Level 1 permission in permissions.csv) and the same minimum
length, but reads data/{session}/segments.csv (build_segments.py) and writes
aggregated_segments_with_embeddings.csv, leaving the utterance-level
aggregated_data_with_embeddings.csv alone. Embeddings are OLMo, emb_0 .. emb_4095,
from text_embeddings.add_embeddings.

Run in the `fusion` env:
    /safestore/users/landry/miniconda3/envs/fusion/bin/python analysis/scripts/aggregate_segments.py
"""

import argparse
from pathlib import Path

import pandas as pd

from paths import DATA_DIR, DATA_ROOT
from text_embeddings import add_embeddings


def full_permission_pairs(permissions_csv: Path) -> list[int]:
    """Pairs in which both participants selected Level 1."""
    p = pd.read_csv(permissions_csv)
    p = p[p["Level 1"] == "X"]
    counts = p.groupby("Pair ID").size()
    return sorted(counts[counts == 2].index)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output", type=Path, default=DATA_ROOT / "aggregated_segments_with_embeddings.csv")
    parser.add_argument("--min-words", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    pairs = full_permission_pairs(DATA_ROOT / "permissions.csv")
    csvs = [args.data_dir / f"{pair:03d}" / "segments.csv" for pair in pairs]
    missing = [c.parent.name for c in csvs if not c.exists()]
    if missing:
        print(f"no segments.csv for permitted sessions {', '.join(missing)}; skipping them")
    df = pd.concat([pd.read_csv(c) for c in csvs if c.exists()], ignore_index=True)
    print(f"{len(df)} segments from {df['pairID'].nunique()} sessions")

    df = df[df["text"].str.split().str.len() >= args.min_words].reset_index(drop=True)
    print(f"{len(df)} segments with at least {args.min_words} words")

    df = add_embeddings(df, batch_size=args.batch_size)
    df.to_csv(args.output, index=False)
    print(f"saved {df.shape} -> {args.output}")


if __name__ == "__main__":
    main()
