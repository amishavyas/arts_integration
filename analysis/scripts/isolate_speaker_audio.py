"""Remove the partner's bleed from each speaker's concat track with voicolate v3.

Input is each session's {speaker}_concat.wav from concat_speaker_audio.py; output,
next to it, is {speaker}_isolated.wav (same length and t=0, so sample-aligned with
the concat tracks) plus isolation_params.json.

The concat tracks are not continuous recordings: each speaker's recorder only ran
while their own VAD was triggered, and the gaps are digital silence. voicolate's
calibration and per-track noise floor assume a live microphone throughout, and
cross-microphone dominance can only judge a frame when both mics were recording.
So voicolate is run on just the stretches where both speakers were recording
(about half of each person's audio), cut out and joined end to end. Stretches
where only one mic was recording pass through untouched apart from the same
calibration and output gain, so levels match across the splice; there is no
second mic there to separate against.

Run from the repo root in the `base` env (conda activate base):
    python analysis/scripts/isolate_speaker_audio.py
    ... --sessions 020 021 --force
"""

import argparse
import json
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from concat_speaker_audio import utterance_layout  # noqa: E402
from paths import DATA_DIR, VOICOLATE_DIR  # noqa: E402

# voicolate isn't pip-installed; import it from its repo, as its own scripts do
sys.path.insert(0, str(VOICOLATE_DIR))

SPEAKERS = ["0", "1"]


def live_masks(layout, n_total):
    """Per-speaker boolean mask of the samples that speaker's recorder was running."""
    masks = {s: np.zeros(n_total, dtype=bool) for s in SPEAKERS}
    for u in layout.itertuples():
        masks[u.speaker][u.offset:u.offset + u.n_frames] = True
    return masks


def to_int16(x):
    return (np.clip(x, -1.0, 1.0) * 32767).astype(np.int16)


def isolate_session(session_dir: Path, force: bool = False) -> bool:
    concat = [session_dir / f"{s}_concat.wav" for s in SPEAKERS]
    outputs = [session_dir / f"{s}_isolated.wav" for s in SPEAKERS]
    if not all(p.exists() for p in concat):
        print(f"{session_dir.name}: no concat tracks, skipping")
        return True
    if all(p.exists() for p in outputs) and not force:
        print(f"{session_dir.name}: already isolated, skipping (use --force)")
        return True

    from voicolate import IsolationConfig, isolate_v3

    layout, samplerate, _, t0 = utterance_layout(session_dir, verbose=False)
    tracks = [sf.read(p, dtype="float32")[0] for p in concat]
    n_total = len(tracks[0])
    masks = live_masks(layout, n_total)
    both = masks["0"] & masks["1"]
    print(f"{session_dir.name}: {n_total / samplerate / 60:.1f} min, both mics live for "
          f"{both.sum() / samplerate / 60:.1f} min "
          f"({both.sum() / masks['0'].sum():.0%} of 0's audio, {both.sum() / masks['1'].sum():.0%} of 1's)")

    # voicolate reads files, so hand it the both-live stretches as temporary wavs
    with tempfile.TemporaryDirectory() as tmp:
        paths = [str(Path(tmp) / f"{s}_both_live.wav") for s in SPEAKERS]
        for path, track in zip(paths, tracks):
            sf.write(path, track[both], samplerate, subtype="PCM_16")
        result = isolate_v3(paths, config=IsolationConfig())

    # pass-through gain: the calibration gain voicolate gave this track, then its output gain
    gains = [10 ** ((g + result["output_gain_db"]) / 20) for g in result["calibration"]["gains_db"]]
    for s, track, isolated, gain, out in zip(SPEAKERS, tracks, result["audio"], gains, outputs):
        full = track * gain
        full[both] = isolated[:both.sum()]
        n_clipped = np.count_nonzero(np.abs(full) > 1.0)
        if n_clipped:
            print(f"  warning: {out.name} clipped {n_clipped} samples")
        sf.write(out, to_int16(full), samplerate, subtype="PCM_16")
        print(f"  wrote {out}")

    params = {k: v for k, v in result.items() if k not in ("audio", "speech_probability")}
    params.update(t0=t0, both_live_seconds=both.sum() / samplerate,
                  live_seconds={s: m.sum() / samplerate for s, m in masks.items()},
                  passthrough_gain_db=[round(20 * np.log10(g), 2) for g in gains])
    (session_dir / "isolation_params.json").write_text(json.dumps(params, indent=2, default=float))
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--sessions", nargs="+", help="session folder names (default: all)")
    parser.add_argument("--force", action="store_true", help="re-isolate sessions that already have output")
    args = parser.parse_args()

    session_dirs = sorted(p for p in args.data_dir.iterdir() if (p / "audio").is_dir())
    if args.sessions:
        session_dirs = [p for p in session_dirs if p.name in args.sessions]

    failed = []
    for session_dir in session_dirs:
        try:
            isolate_session(session_dir, force=args.force)
        except Exception as e:
            traceback.print_exc()
            print(f"{session_dir.name}: FAILED: {e}")
            failed.append(session_dir.name)
    if failed:
        print(f"failed: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
