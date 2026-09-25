"""Concatenate each speaker's utterances into one time-aligned file per session.

For every session folder in data/, places each audio/utterance_{speaker}_{utc}.wav
at its original time, filling the gaps with silence, and writes to the session
folder:
    0_concat.wav, 1_concat.wav   each speaker's track
    mix_concat.wav               both tracks summed

Timing: the recorder (convo_recorder/backend/audio_processor.py) logs each
utterance's start as `timestamp` in data.csv, taken at the first audio block of
the utterance. The filename's {utc} is only when the WAV was written, after
queued transcription, so it lags the audio by a variable amount; it's used
(as utc - duration) only for the rare file without a CSV row.

All tracks share the session's earliest utterance start as t=0 and have the same
length, so they line up sample-for-sample. Overlapping audio is summed.
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from paths import DATA_DIR

UTTERANCE_RE = re.compile(r"^utterance_(\d+)_(\d+)\.wav$")


def load_start_times(session_dir: Path) -> dict:
    """Map WAV filename -> utterance start time (UTC seconds) from data.csv."""
    csv = session_dir / "data.csv"
    if not csv.exists():
        return {}
    df = pd.read_csv(csv)
    df["filename"] = df["audio_path"].map(lambda p: Path(p).name)
    # Two utterances saved in the same second share a filename, and the later
    # write overwrote the earlier one; its CSV row was appended last.
    df = df.drop_duplicates("filename", keep="last")
    return dict(zip(df["filename"], df["timestamp"]))


def write_track(path: Path, track: np.ndarray, samplerate: int, subtype: str) -> None:
    n_clipped = np.count_nonzero((track > 32767) | (track < -32768))
    if n_clipped:
        print(f"  warning: {path.name} clipped {n_clipped} samples")
    sf.write(path, np.clip(track, -32768, 32767).astype(np.int16), samplerate, subtype=subtype)


def utterance_layout(session_dir: Path, verbose: bool = True):
    """Where each utterance WAV sits in the session's concat tracks.

    Returns (layout, samplerate, subtype, t0): layout has one row per WAV with
    speaker, wav, start (UTC seconds), offset (first sample in the concat track)
    and n_frames; t0 is the UTC time of sample 0. None if the session has no audio.
    """
    start_times = load_start_times(session_dir)
    utterances = []
    samplerate = subtype = None
    for wav in sorted((session_dir / "audio").glob("*.wav")):
        m = UTTERANCE_RE.match(wav.name)
        if not m:
            continue
        info = sf.info(wav)
        if samplerate is None:
            samplerate, subtype = info.samplerate, info.subtype
        elif info.samplerate != samplerate:
            raise ValueError(f"{wav}: samplerate {info.samplerate} != {samplerate}")
        start = start_times.get(wav.name)
        if start is None:
            start = int(m.group(2)) - info.duration
            if verbose:
                print(f"  warning: {wav.name} has no data.csv row; placing it at filename stamp - duration")
        utterances.append({"speaker": m.group(1), "wav": wav, "start": start, "n_frames": info.frames})

    if not utterances:
        return None

    layout = pd.DataFrame(utterances)
    t0 = layout["start"].min()
    layout["offset"] = ((layout["start"] - t0) * samplerate).round().astype(int)
    return layout, samplerate, subtype, t0


def concat_session(session_dir: Path) -> None:
    result = utterance_layout(session_dir)
    if result is None:
        return
    layout, samplerate, subtype, _ = result
    n_total = int((layout["offset"] + layout["n_frames"]).max())

    tracks = {}
    for u in layout.itertuples():
        track = tracks.setdefault(u.speaker, np.zeros(n_total, dtype=np.int32))
        audio, _ = sf.read(u.wav, dtype="int16")
        track[u.offset:u.offset + len(audio)] += audio

    for speaker, track in sorted(tracks.items()):
        n_utts = (layout["speaker"] == speaker).sum()
        write_track(session_dir / f"{speaker}_concat.wav", track, samplerate, subtype)
        print(f"{session_dir / f'{speaker}_concat.wav'} ({n_utts} utterances, {n_total / samplerate:.1f} s)")
    write_track(session_dir / "mix_concat.wav", sum(tracks.values()), samplerate, subtype)
    print(f"{session_dir / 'mix_concat.wav'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()

    for session_dir in sorted(p for p in args.data_dir.iterdir() if (p / "audio").is_dir()):
        concat_session(session_dir)


if __name__ == "__main__":
    main()
