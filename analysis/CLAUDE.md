# CLAUDE.md

## Running scripts

- Only run short scripts yourself: quick checks and small tests that finish in about a minute.
- For anything longer (model inference, embedding extraction, UMAP/grid searches, jobs over the full dataset), don't run it. Write the script, then give me the exact command, including which conda env to use. I'll run it myself in a tmux session.

## Data and privacy

- Participant data (audio, transcripts, CSVs, embeddings) lives outside this repo, under `ARTS_DATA_ROOT` (default: the folder the repo is cloned into; see `scripts/paths.py`). Never copy it into the repo or commit it.
- Notebooks are committed with outputs stripped; don't work around `.gitattributes` or the pre-commit hooks.
