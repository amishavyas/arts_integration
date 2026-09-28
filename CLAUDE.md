# CLAUDE.md

## What this project is

Started as a study of in-person conversation (pairs of participants discuss
images together, audio recorded per-speaker). It is becoming an **art
installation**: while a new pair talks in front of an image, their speech is
transcribed and embedded in real time, and the system immediately plays back
the most semantically similar utterance ever said by a *different* pair about
the *same* image. The effect depends on this happening at conversational
speed — every stage of the pipeline is a latency budget, not just a
correctness problem.

## Repo layout

- `convo_recorder/` — Flask + React app that both collects study data *and*
  runs the realtime installation (same app, mode-switchable — see below).
  - `preflight_dialog.py` — native tkinter dialog `run.py` shows before
    starting the backend/frontend (skipped by `run.py --test`). RA sees live
    per-channel input waveforms and the next pair ID, and chooses
    real-vs-devdata and full-intervention-vs-data-collection-only.
  - `backend/audio_devices.py` — `find_scarlett_device()`, split into its
    own module so the preflight dialog doesn't have to import the ML stack
    just to find the input device.
  - `backend/audio_processor.py` — the realtime pipeline: mic capture
    (`sounddevice`) → Silero VAD (per-channel, with pre-roll so onsets
    aren't clipped) → transcription (`mlx-whisper`) → [if intervention
    enabled] embed (`text_embeddings.TextEmbedder`, imported directly from
    `analysis/scripts/`) → bank lookup → playback (`sounddevice.play`).
    Capture → transcribe → CSV-write always runs (that's the data-collection
    path); embed → match → play only runs in intervention mode. Playback
    always jumps to the newest match and interrupts whatever's currently
    playing rather than queuing a backlog — see "Known follow-ups" for why
    that matters.
  - `backend/bank.py` — loads
    `data/utterance_library/aggregated_segments_with_embeddings.csv` once at
    startup into a normalized in-memory matrix; `find_match()` is a single
    masked dot product (same-`imgID`, different-`pairID`, same candidate
    logic as `explore.ipynb`'s `choose_random_utterance`).
  - `backend/session_manager.py` — three session-directory modes: real
    (`data/NNN`, numbered), devdata (`devdata/NNN`, numbered, own counter —
    the preflight dialog's "test/dev data" choice), and `--test` (single
    reused `data/test/`, CLI-only fast path that skips the dialog).
- `analysis/` — **offline** pipeline that turns recorded sessions into the
  utterance bank: isolate each speaker's audio (voicolate) → WhisperX
  transcription → segment → aggregate + embed
  (`analysis/scripts/text_embeddings.py`). See `analysis/CLAUDE.md` for the
  rules that apply there (data privacy, how to run long scripts) and
  `analysis/README.md` for the step-by-step pipeline / conda envs per step.
  `analysis/scripts/reembed_utterance_bank.py` re-embeds the bank in place
  whenever the embedding model changes (keeps a one-time backup of whatever
  it overwrites).
- `data/utterance_library/` — the **playback bank** for the installation:
  per-participant isolated `.wav` segments (`pairXXX_subN_isolated.wav`) plus
  `aggregated_segments_with_embeddings.csv` (one row per utterance segment,
  embedding columns `emb_0..emb_2047`, plus `pairID`, `imgID`, `text`,
  `start`/`end`).
- `tools/` — repo-hygiene scripts (notebook-output checks for the pre-commit
  hook).

## Models and where they live

- **Embedding**: `allenai/OLMo-2-0425-1B`, 2048-dim, run via `mlx-lm`
  (`text_embeddings.TextEmbedder`, `backend="mlx"` default). The realtime
  pipeline and the bank re-embed script both import this one module, so
  query and bank vectors can never drift into different spaces.
  **Quantized to 4-bit** at `~/models/olmo2-1b-4bit` (via `mlx_lm.convert
  --hf-path allenai/OLMo-2-0425-1B --mlx-path ~/models/olmo2-1b-4bit -q
  --q-bits 4 --q-group-size 64`) — this is a **local path, not in git**; a
  fresh checkout of this repo needs that conversion re-run once. Both
  `AudioConfig.embed_model` and `reembed_utterance_bank.py --model` default
  to `DEFAULT_EMBED_MODEL` in `text_embeddings.py` / `audio_processor.py`,
  which resolves to this path via `Path.home()`.
- **Transcription**: `mlx-community/whisper-base.en-mlx` via `mlx-whisper`
  (`AudioConfig.whisper_model`). Not benchmarked against other sizes yet —
  worth revisiting once the full show latency budget is measured.
- **VAD**: Silero (`silero-vad` pip package — ships its ONNX/jit weights in
  the wheel, no network needed at runtime, which matters given the gallery's
  network is unreliable). One model instance per input channel; a shared
  instance would corrupt the other channel's recurrent state.

## The machine constraint that shaped all of this

The installation machine (this one) is an **Apple M2 with 8GB unified RAM**,
no discrete GPU. That ruled out the original offline embedding model
(`allenai/Olmo-3-1025-7B`, fp16 ≈ 14GB) for realtime use, and even the fp16
1B model caused severe memory thrashing under this machine's normal
background load (Chrome, mysqld, CrowdStrike, etc. — often 7+GB already in
use before anything ML-related loads). Quantizing to 4-bit
(~600-900MB) fixed it. **Lesson for future tuning on this machine**: before
assuming a latency number is a code/model problem, check `top`/`vm_stat` for
compressed memory and swap — a `python` process showing multiple GB
"CMPRS" is why something that should take 50ms takes 20s, not a bug.
Closing Chrome/mysqld before a show or a benchmark run measurably helps.

`mlx-lm` also compiles a distinct kernel graph per unique input *shape*;
embedding thousands of different-length utterances one at a time without
bucketing caused both the slowdown above and unbounded memory growth.
`text_embeddings.py`'s `MLX_LENGTH_BUCKETS` pads every input up to the next
bucket so the shape count — and therefore the compiled-graph count — stays
small regardless of how many utterances get embedded.

## Known follow-ups (not yet done)

- Some bank utterances are quite long (25-30s) — a data-quality issue
  already noted in `explore.ipynb`, to be fixed by re-segmenting the source
  recordings into shorter/sentence-level clips. Until then, a single long
  match can still play in full; it just can no longer back up a queue of
  stale reactions (playback interrupts on every new match).
- Whisper model size and VAD thresholds (`vad_threshold`,
  `vad_min_silence_ms`, `vad_speech_pad_ms` in `AudioConfig`) are reasonable
  defaults, not tuned against real gallery-floor noise conditions.
- Playback routing assumes the installation's audio output is physically
  isolated from the input mics (confirmed for the current setup) — the
  brief input-mute during playback (`playback_mute_seconds`) is a
  belt-and-suspenders safeguard, not the primary defense.
- `preflight_dialog.py`'s pair-ID display is a preview (counts existing
  session folders); the real number is assigned when the backend actually
  creates the session directory a few seconds later.

## Data and privacy

Same rule repo-wide as `analysis/CLAUDE.md`: participant data (audio,
transcripts, CSVs, embeddings) is never committed and lives outside the repo
under `ARTS_DATA_ROOT` / `analysis/scripts/paths.py`. `.gitignore` blocks
these file types at the root, including `/data/` and `/devdata/`.
`data/utterance_library/` is the one place raw participant audio +
embeddings are read from directly by the installation — still don't add
exceptions to `.gitignore` for it.

## Conda envs

- `artsinteg` — the env `launch_experiment.sh` actually launches with, and
  now the one with everything the realtime pipeline needs: `mlx`, `mlx-lm`,
  `mlx-whisper`, `silero-vad`, `transformers`, `torch`, `sounddevice`,
  `flask`. See `convo_recorder/backend/requirements.txt`.
- `convo_art` — older env with `faster-whisper`/`whisperx`/CPU
  `openai-whisper`; not used by the current pipeline, kept for reference.
- `whisperx` — offline transcription env used by
  `analysis/scripts/transcribe_isolated.py` (per `analysis/README.md`).
- No env here is named `fusion` (the env `analysis/README.md` historically
  says to embed in) — the bank re-embed now runs fine in `artsinteg` on this
  machine (a few minutes with the quantized model, not the multi-hour job
  the unquantized fp16 model would have been).

## Running scripts

Same as `analysis/CLAUDE.md`: short checks (~1 min) are fine to run directly.
Anything longer (model loads, batch embedding, full-session realtime tests) —
write the script and hand over the exact command + conda env, don't run it
yourself.
