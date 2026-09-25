# Analysis

Offline analysis of the conversations recorded by `convo_recorder/`: isolating each
speaker's audio, transcribing it, and embedding what was said.

## Data stays out of git

Audio, transcripts, and anything built from them (CSVs, embeddings, UMAPs) are
participant data and are never committed: `.gitignore` blocks those file types
repo-wide. The code reads them from `ARTS_DATA_ROOT`, which defaults to the folder
the repo is cloned into:

    <ARTS_DATA_ROOT>/
        data/<session>/     per-session audio, data.csv, transcripts, segments.csv
        permissions.csv
        aggregated_*.csv
        arts_integration/   this repo

Elsewhere, point it at your copy: `export ARTS_DATA_ROOT=/path/to/data_root`.

## One-time setup per clone

Notebook outputs contain transcripts and participant audio, so they are stripped
before commit. After cloning, run once from the repo root:

    pip install nbstripout pre-commit
    nbstripout --install     # strips outputs on `git add`; your local copy keeps them
    pre-commit install       # blocks commits of notebooks with outputs and files > 1 MB

## Pipeline

Run from the repo root, in order. Each skips work already done (`--force` /
`--overwrite` to redo) and takes `--sessions 020 021 ...`.

| step | script | env |
|---|---|---|
| per-speaker tracks from utterance WAVs | `analysis/scripts/concat_speaker_audio.py` | base |
| remove partner bleed (voicolate) | `analysis/scripts/isolate_speaker_audio.py` | base |
| WhisperX transcripts | `analysis/scripts/transcribe_isolated.py` | whisperx |
| one row per segment | `analysis/scripts/build_segments.py` | base |
| aggregate + OLMo embeddings | `analysis/scripts/aggregate_segments.py` | fusion |

`legacy/` holds earlier one-off scripts, kept for reference.
