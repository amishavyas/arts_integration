"""Transcribe each speaker's whole isolated track with WhisperX large-v3 + alignment.

For every session folder in data/ with {speaker}_isolated.wav (from
isolate_speaker_audio.py), writes {speaker}_transcript.json: WhisperX's aligned
output, i.e. sentence-level `segments` (start, end, text, words) and a flat
`word_segments` list. Times are seconds from the start of the track, which is
the concat tracks' t=0 (t0 in isolation_params.json).

Run from the repo root in the `whisperx` env (conda activate whisperx):
    python analysis/scripts/transcribe_isolated.py
    ... --sessions 020 --overwrite

Tracks that already have a transcript are skipped unless --overwrite.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# A broken TensorFlow in ~/.local leaks into this env; see transcribeV3.py.
os.environ["USE_TF"] = "0"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import DATA_DIR  # noqa: E402
from transcribeV3 import preload_cudnn  # noqa: E402

SAMPLE_RATE = 16000  # Whisper's input rate


def transcribe_track(model, align_model, align_metadata, wav: Path, batch_size: int) -> dict:
    import librosa
    import whisperx
    audio, _ = librosa.load(str(wav), sr=SAMPLE_RATE)
    result = model.transcribe(audio, batch_size=batch_size, language="en")
    aligned = whisperx.align(result["segments"], align_model, align_metadata, audio, "cuda",
                             return_char_alignments=False)
    return {"audio": wav.name, "language": "en", **aligned}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--sessions", nargs="+", help="session folder names (default: all)")
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    wavs = sorted(args.data_dir.glob("*/*_isolated.wav"))
    if args.sessions:
        wavs = [w for w in wavs if w.parent.name in args.sessions]
    todo = [(w, w.with_name(w.name.replace("_isolated.wav", "_transcript.json"))) for w in wavs]
    if not args.overwrite:
        todo = [(w, out) for w, out in todo if not out.exists()]
    if not todo:
        print("Nothing to do.")
        return

    preload_cudnn()
    import whisperx
    model = whisperx.load_model(args.model, "cuda", compute_type="float16", language="en")
    align_model, align_metadata = whisperx.load_align_model(language_code="en", device="cuda")

    for i, (wav, out) in enumerate(todo, 1):
        t = time.time()
        result = transcribe_track(model, align_model, align_metadata, wav, args.batch_size)
        out.write_text(json.dumps(result, indent=1, default=float))
        print(f"[{i}/{len(todo)}] {wav.parent.name}/{out.name}: {len(result['segments'])} segments "
              f"in {time.time() - t:.0f} s", flush=True)


if __name__ == "__main__":
    main()
