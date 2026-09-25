"""Re-transcribe every session's utterances with Whisper large-v3 (WhisperX).

For each session folder in data/, copies data.csv to data_v3.csv with the same
columns and rows, replacing only `text` with a large-v3 transcript of the
row's audio/utterance_*.wav. (The recorder transcribed with Whisper tiny.)

Run with the whisperx env's interpreter:
    /safestore/users/landry/miniconda3/envs/whisperx/bin/python analysis/scripts/transcribeV3.py

Sessions that already have data_v3.csv are skipped unless --overwrite.
"""

import argparse
import difflib
import os
from pathlib import Path

# A broken TensorFlow in ~/.local leaks into this env; without this, transformers
# tries to import it and whisperx fails with "Could not import module 'Pipeline'".
# Must be set before transformers is first imported.
os.environ["USE_TF"] = "0"

import pandas as pd
from tqdm import tqdm

from paths import DATA_DIR

SAMPLE_RATE = 16000  # Whisper's input rate


def preload_cudnn():
    """Load pip-installed cuDNN with RTLD_GLOBAL so CTranslate2 can find it.

    Without this, CTranslate2 fails to dlopen libcudnn_cnn.so.9 and the process
    core-dumps on the first convolution. Must run before importing whisperx.
    See groupconv/extract/transcription.py for the full story.
    """
    import ctypes
    try:
        import nvidia
    except ImportError:
        return
    root = Path(nvidia.__file__).parent
    for name in ("libcudnn_graph.so.9", "libcudnn_ops.so.9",
                 "libcudnn_cnn.so.9", "libcudnn.so.9"):
        for lib in root.rglob(name):
            try:
                ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
                break
            except OSError:
                continue


def transcribe(model, wav: Path, batch_size: int) -> str:
    import librosa
    audio, _ = librosa.load(str(wav), sr=SAMPLE_RATE)
    result = model.transcribe(audio, batch_size=batch_size, language="en")
    return " ".join(seg["text"].strip() for seg in result["segments"]).strip()


def assign_duplicate(rows: pd.DataFrame, new_text: str):
    """Pick which of several rows sharing one filename the surviving audio belongs to.

    Two utterances written in the same second got the same filename, so only the
    last-written audio survives. Match on the old (tiny) transcript rather than
    trusting row order, since two transcription threads wrote the CSV.
    """
    scores = [difflib.SequenceMatcher(None, str(old).lower(), new_text.lower()).ratio()
              for old in rows["text"]]
    return rows.index[max(range(len(scores)), key=scores.__getitem__)]


def transcribe_session(model, session_dir: Path, batch_size: int) -> None:
    df = pd.read_csv(session_dir / "data.csv")
    out = df.copy()
    out["text"] = pd.NA
    filenames = df["audio_path"].map(lambda p: Path(p).name)

    n_missing = n_lost = n_empty = 0
    for filename, rows in tqdm(df.groupby(filenames, sort=False), desc=session_dir.name, leave=False):
        wav = session_dir / "audio" / filename
        if not wav.exists():
            n_missing += len(rows)
            continue
        text = transcribe(model, wav, batch_size)
        idx = rows.index[0] if len(rows) == 1 else assign_duplicate(rows, text)
        out.loc[idx, "text"] = text
        n_lost += len(rows) - 1
        n_empty += not text

    out.to_csv(session_dir / "data_v3.csv", index=False)
    print(f"{session_dir.name}: {len(out)} rows"
          + (f", {n_missing} missing audio" if n_missing else "")
          + (f", {n_lost} overwritten audio" if n_lost else "")
          + (f", {n_empty} with no speech found" if n_empty else ""))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--sessions", nargs="+", help="session folder names (default: all)")
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    session_dirs = sorted(p.parent for p in args.data_dir.glob("*/data.csv"))
    if args.sessions:
        session_dirs = [p for p in session_dirs if p.name in args.sessions]
    if not args.overwrite:
        session_dirs = [p for p in session_dirs if not (p / "data_v3.csv").exists()]
    if not session_dirs:
        print("Nothing to do.")
        return

    preload_cudnn()
    import whisperx
    model = whisperx.load_model(args.model, "cuda", compute_type="float16", language="en")

    for session_dir in session_dirs:
        transcribe_session(model, session_dir, args.batch_size)


if __name__ == "__main__":
    main()
