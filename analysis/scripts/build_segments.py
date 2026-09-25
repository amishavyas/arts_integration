"""Turn each session's WhisperX transcripts into one CSV row per segment.

Reads data/{session}/{speaker}_transcript.json (transcribe_isolated.py) and writes
data/{session}/segments.csv with one row per aligned WhisperX segment (roughly one
sentence):

    pairID, subID       session and speaker
    imgID               image on screen, from the recorder's data.csv row for the
                        utterance WAV this segment's audio came from (see below)
    text
    start, end          seconds from the start of the isolated track
    timestamp           UTC start, t0 + start -- same clock as data.csv's timestamp
    duration, n_words
    word_score          mean WhisperX alignment score of the segment's words
    partner_live_frac   fraction of the segment during which the partner's mic was
                        also recording, i.e. where voicolate could remove bleed;
                        the rest passed through un-isolated
    source_wav          the speaker's utterance WAV overlapping the segment most
    audio_path          the isolated track, relative to DATA_ROOT (scripts/paths.py);
                        slice it with start/end

Every stretch of a concat track is some utterance WAV placed at its data.csv
timestamp, so a segment's source WAV is the same speaker's WAV it overlaps most,
and that WAV's data.csv row gives the image. A segment WhisperX placed just outside
every WAV (alignment drift) takes the nearest one.

Run from the repo root in the `base` env (conda activate base):
    python analysis/scripts/build_segments.py
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from concat_speaker_audio import utterance_layout  # noqa: E402
from paths import DATA_DIR, DATA_ROOT  # noqa: E402

SPEAKERS = ["0", "1"]
GRID_HZ = 100  # resolution of the partner-live computation


def image_by_wav(session_dir: Path) -> dict:
    """WAV filename -> imgID, from data.csv (last row wins, as in concat_speaker_audio)."""
    df = pd.read_csv(session_dir / "data.csv")
    df["filename"] = df["audio_path"].map(lambda p: Path(p).name)
    df = df.drop_duplicates("filename", keep="last")
    return dict(zip(df["filename"], df["imgID"]))


def live_grid(intervals, n):
    on = np.zeros(n, dtype=bool)
    for a, b in intervals:
        on[int(a * GRID_HZ):int(np.ceil(b * GRID_HZ))] = True
    return on


def build_session(session_dir: Path) -> pd.DataFrame | None:
    transcripts = {s: session_dir / f"{s}_transcript.json" for s in SPEAKERS}
    if not all(p.exists() for p in transcripts.values()):
        return None

    layout, samplerate, _, t0 = utterance_layout(session_dir, verbose=False)
    layout["a"] = layout["offset"] / samplerate
    layout["b"] = (layout["offset"] + layout["n_frames"]) / samplerate
    images = image_by_wav(session_dir)
    n_grid = int(np.ceil(layout["b"].max() * GRID_HZ)) + 1
    live = {s: live_grid(zip(g["a"], g["b"]), n_grid) for s, g in layout.groupby("speaker")}

    rows = []
    for speaker, path in transcripts.items():
        partner = live[SPEAKERS[1 - SPEAKERS.index(speaker)]]
        utts = layout[layout["speaker"] == speaker]
        for seg in json.loads(path.read_text())["segments"]:
            text = seg["text"].strip()
            if not text:
                continue
            start, end = seg["start"], seg["end"]
            # overlap in seconds; negative is the gap, so the max is also the nearest WAV
            overlap = np.minimum(utts["b"], end) - np.maximum(utts["a"], start)
            src = utts.loc[overlap.idxmax()]
            g0, g1 = int(start * GRID_HZ), max(int(np.ceil(end * GRID_HZ)), int(start * GRID_HZ) + 1)
            scores = [w["score"] for w in seg.get("words", []) if "score" in w]
            rows.append({
                "pairID": int(session_dir.name),
                "subID": int(speaker),
                "imgID": images.get(src["wav"].name),
                "text": text,
                "start": start,
                "end": end,
                "timestamp": t0 + start,
                "duration": end - start,
                "n_words": len(text.split()),
                "word_score": np.mean(scores) if scores else np.nan,
                "partner_live_frac": partner[g0:g1].mean(),
                "source_wav": src["wav"].name,
                "audio_path": os.path.relpath(session_dir / f"{speaker}_isolated.wav", DATA_ROOT),
            })

    df = pd.DataFrame(rows).sort_values("start", ignore_index=True)
    if df["imgID"].isna().any():
        # a WAV with no data.csv row: take the image of the nearest row in time
        csv = pd.read_csv(session_dir / "data.csv").sort_values("timestamp")
        missing = df["imgID"].isna()
        nearest = pd.merge_asof(df.loc[missing, ["timestamp"]].reset_index().sort_values("timestamp"),
                                csv[["timestamp", "imgID"]], on="timestamp", direction="nearest")
        df.loc[nearest["index"], "imgID"] = nearest["imgID"].values
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--sessions", nargs="+", help="session folder names (default: all)")
    args = parser.parse_args()

    session_dirs = sorted(p for p in args.data_dir.iterdir() if (p / "audio").is_dir())
    if args.sessions:
        session_dirs = [p for p in session_dirs if p.name in args.sessions]

    for session_dir in session_dirs:
        df = build_session(session_dir)
        if df is None:
            print(f"{session_dir.name}: no transcripts, skipping")
            continue
        df.to_csv(session_dir / "segments.csv", index=False)
        print(f"{session_dir.name}: {len(df)} segments, {df['n_words'].sum()} words, "
              f"median {df['duration'].median():.1f} s, "
              f"{(df['partner_live_frac'] > 0).mean():.0%} at least partly isolated")


if __name__ == "__main__":
    main()
