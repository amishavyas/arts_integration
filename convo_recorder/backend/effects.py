"""Playback effects for the installation:

- A continuous, low-volume background murmur, looped for the whole session.
  The preflight dialog lets the RA pick which file from data/effects/murmurs/
  (or none at all) - see list_murmur_files() and load_murmur().
- Pre-generated "interference" (solo_fx) versions of bank utterances,
  played instead of the raw clip whenever one exists for the matched row -
  data/effects/solo_fx/index.csv maps a row's (audio_path, start, end) to
  its fx_file. Rows added live during a session (no pre-generated fx
  version) fall back to the raw clip.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
EFFECTS_DIR = REPO / "data" / "effects"
MURMURS_DIR = EFFECTS_DIR / "murmurs"
DEFAULT_MURMUR_FILENAME = "gallery.wav"
MURMUR_MAX_SECONDS = 180  # loop a bounded prefix rather than holding the whole file in RAM
SOLO_FX_DIR = EFFECTS_DIR / "solo_fx"
SOLO_FX_INDEX_CSV = SOLO_FX_DIR / "index.csv"


def list_murmur_files() -> list[str]:
    """Filenames available in data/effects/murmurs/, for the preflight
    dialog's dropdown."""
    return sorted(p.name for p in MURMURS_DIR.glob("*.wav"))


def load_audio_file(path: Path, max_seconds: float | None = None) -> tuple[np.ndarray, int]:
    """Read a WAV file (or just its first max_seconds, if given) -> (int16
    mono samples, sample rate). Downmixes to mono (channel average) if the
    file isn't already mono - the installation's output is a single mono
    channel, and effects files aren't guaranteed to be recorded/generated
    mono. max_seconds limits how many frames are ever read off disk, so a
    long file's full length never gets materialized in memory just to be
    truncated afterwards."""
    with wave.open(str(path)) as w:
        rate = w.getframerate()
        channels = w.getnchannels()
        n_frames = w.getnframes()
        if max_seconds is not None:
            n_frames = min(n_frames, int(max_seconds * rate))
        data = np.frombuffer(w.readframes(n_frames), dtype=np.int16)
        if channels > 1:
            data = data.reshape(-1, channels).mean(axis=1).astype(np.int16)
    return data, rate


def load_murmur(filename: str) -> tuple[np.ndarray, int]:
    """Load (up to MURMUR_MAX_SECONDS of) the named murmur file from
    data/effects/murmurs/ -> (int16 mono samples, sample rate). Loaded once
    per session and looped - capped rather than loading the whole file,
    since this machine's memory budget is tight enough that a large in-RAM
    buffer has previously contributed to instability elsewhere in this
    pipeline (see CLAUDE.md)."""
    path = MURMURS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Murmur file not found: {path}")
    data, rate = load_audio_file(path, max_seconds=MURMUR_MAX_SECONDS)
    print(f"[effects] background murmur for this session: {path.name} "
          f"({len(data) / rate:.0f}s loaded, loops)")
    return data, rate


class SoloFxIndex:
    """Maps a bank row's (audio_path, start, end) to its pre-generated fx
    file, if one exists. Built once from data/effects/solo_fx/index.csv."""

    def __init__(self, index_csv: Path = SOLO_FX_INDEX_CSV):
        self.dir = index_csv.parent
        self._lookup = {}
        if index_csv.exists():
            df = pd.read_csv(index_csv)
            for row in df.itertuples(index=False):
                key = (row.audio_path, round(float(row.start), 3), round(float(row.end), 3))
                self._lookup[key] = row.fx_file
        print(f"[effects] solo_fx index: {len(self._lookup)} entries from {index_csv}")

    def path_for(self, audio_path: str, start: float, end: float) -> Path | None:
        fx_file = self._lookup.get((audio_path, round(float(start), 3), round(float(end), 3)))
        return (self.dir / fx_file) if fx_file else None
