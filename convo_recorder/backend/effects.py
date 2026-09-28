"""Playback effects for the installation:

- A continuous, low-volume background murmur (MURMUR_FILE), looped for the
  whole session.
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
MURMUR_FILE = MURMURS_DIR / "layered_murmur_10layers.wav"
SOLO_FX_DIR = EFFECTS_DIR / "solo_fx"
SOLO_FX_INDEX_CSV = SOLO_FX_DIR / "index.csv"


def load_audio_file(path: Path) -> tuple[np.ndarray, int]:
    """Read a whole mono WAV file -> (int16 samples, sample rate)."""
    with wave.open(str(path)) as w:
        rate = w.getframerate()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return data, rate


def load_murmur() -> tuple[np.ndarray, int]:
    """Load the background murmur -> (int16 mono samples, sample rate).
    Loaded once per session and looped."""
    if not MURMUR_FILE.exists():
        raise FileNotFoundError(f"Murmur file not found: {MURMUR_FILE}")
    data, rate = load_audio_file(MURMUR_FILE)
    print(f"[effects] background murmur for this session: {MURMUR_FILE.name} "
          f"({len(data) / rate:.0f}s, loops)")
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
