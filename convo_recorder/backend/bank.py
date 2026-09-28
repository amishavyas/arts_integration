"""In-memory utterance bank for the realtime intervention pipeline.

Loads data/utterance_library/aggregated_segments_with_embeddings.csv once and
answers "closest same-image, different-pair utterance" queries - same
candidate logic as choose_random_utterance in
analysis/notebooks/explore.ipynb (same imgID, different pairID; that also
excludes the querying pair's own rows, not just the exact source row). A
single masked dot product per query isn't a latency concern at ~4400 rows;
transcription and embedding are where the pipeline's time goes.

The bank is a fixed asset of this repo (data/utterance_library/), not part
of the per-session recorded data under ARTS_DATA_ROOT - paths here are
relative to this file, not analysis/scripts/paths.py's DATA_ROOT.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DEFAULT_BANK_CSV = REPO / "data" / "utterance_library" / "aggregated_segments_with_embeddings.csv"
DEV_BANK_CSV = REPO / "data" / "utterance_library" / "aggregated_segments_with_embeddings.dev.csv"


def resolve_bank_csv(devdata: bool = False) -> Path:
    """Real and devdata sessions search/accumulate into separate databases,
    so test utterances never end up as match candidates for real installation
    visitors. The dev database is seeded from a copy of the real one the
    first time it's needed, so dev testing still has something realistic to
    match against instead of starting empty."""
    if not devdata:
        return DEFAULT_BANK_CSV
    if not DEV_BANK_CSV.exists():
        import shutil
        shutil.copy2(DEFAULT_BANK_CSV, DEV_BANK_CSV)
        print(f"Seeded dev bank from {DEFAULT_BANK_CSV} -> {DEV_BANK_CSV}")
    return DEV_BANK_CSV


def load_segment_audio(audio_dir: Path, row: pd.Series, pad: float = 0.1) -> tuple[np.ndarray, int]:
    """Read one segment from its source track -> (int16 samples, sample rate).

    Same logic as explore.ipynb's load_segment_audio: pad adds a little
    either side since segment boundaries can clip the first/last word.
    """
    import wave

    with wave.open(str(audio_dir / row["audio_path"])) as w:
        rate = w.getframerate()
        a = max(0, int((row["start"] - pad) * rate))
        b = min(w.getnframes(), int((row["end"] + pad) * rate))
        w.setpos(a)
        return np.frombuffer(w.readframes(b - a), dtype=np.int16), rate


class UtteranceBank:
    def __init__(self, csv_path: Path = DEFAULT_BANK_CSV):
        self.csv_path = Path(csv_path)
        self.audio_dir = self.csv_path.parent

        df = pd.read_csv(self.csv_path)
        emb_cols = [c for c in df.columns if c.startswith("emb_")]
        self.meta = df.drop(columns=emb_cols).reset_index(drop=True)

        X = df[emb_cols].to_numpy(np.float32)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        self.embeddings = X / norms
        print(f"UtteranceBank: {len(self.meta)} rows, {X.shape[1]}-dim, from {self.csv_path}")

    def find_match(self, query_embedding: np.ndarray, img_id: str, exclude_pair_id) -> pd.Series | None:
        """Best cosine match for query_embedding among rows with the same img_id and a
        different pairID than exclude_pair_id. None if there's no candidate row."""
        q = np.asarray(query_embedding, dtype=np.float32)
        q = q / np.linalg.norm(q)

        candidate = (self.meta["imgID"] == img_id) & (self.meta["pairID"] != exclude_pair_id)
        if not candidate.any():
            return None

        sims = self.embeddings @ q
        sims = np.where(candidate.to_numpy(), sims, -np.inf)
        best = int(np.argmax(sims))

        row = self.meta.iloc[best].copy()
        row["similarity"] = float(sims[best])
        return row

    def load_audio(self, row: pd.Series, pad: float = 0.1) -> tuple[np.ndarray, int]:
        return load_segment_audio(self.audio_dir, row, pad=pad)

    def add_utterance(self, embedding: np.ndarray, row: dict) -> None:
        """Append a newly transcribed utterance to the bank, in memory and on
        disk, so later sessions can match against it - this is how the
        corpus grows over time instead of staying frozen at whatever was
        embedded offline. row needs at least pairID/subID/imgID/text/start/
        end/audio_path; audio_path may be an absolute path (e.g. a session's
        continuous recording) rather than one relative to self.audio_dir -
        load_segment_audio resolves either correctly. Missing meta columns
        (fields only the offline WhisperX pipeline produces, like
        word_score) are left blank for live-added rows.
        """
        q = np.asarray(embedding, dtype=np.float32)
        q = q / np.linalg.norm(q)
        self.embeddings = np.vstack([self.embeddings, q[None, :]])

        full_meta_row = {col: row.get(col) for col in self.meta.columns}
        self.meta = pd.concat([self.meta, pd.DataFrame([full_meta_row])], ignore_index=True)

        emb_cols = [f"emb_{i}" for i in range(len(q))]
        full_row = {**full_meta_row, **dict(zip(emb_cols, q.tolist()))}
        pd.DataFrame([full_row], columns=list(self.meta.columns) + emb_cols).to_csv(
            self.csv_path, mode="a", header=False, index=False)
