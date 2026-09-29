import math
import sounddevice as sd
import numpy as np
import sys
import threading
import queue
import time as tm
import os
import warnings
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import mlx_whisper
import pandas as pd
import torch
from scipy.io.wavfile import write as wav_write
from scipy import signal
from silero_vad import VADIterator, load_silero_vad
import librosa

from audio_devices import find_scarlett_device  # noqa: F401  (re-exported for audio_endpoints.py)
from bank import DEFAULT_BANK_CSV, UtteranceBank
from effects import SoloFxIndex, load_audio_file, load_murmur

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "analysis" / "scripts"))
from text_embeddings import TextEmbedder  # noqa: E402  (needs the sys.path insert above)

# Silero VAD runs as a PyTorch TorchScript model, called continuously from a
# background thread. A real crash was traced (via a macOS diagnostic report)
# directly into libtorch_cpu/libtorch_python with this unconstrained -
# PyTorch spinning up its own multi-threaded worker pool per inference call
# under concurrent load is a known instability source. Pin it before any
# torch-backed model loads.
torch.set_num_threads(1)

# Suppress the FP16 warning some downstream libs still emit
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

"""
The individual channels for the interface should be set to max volume. Set the master volume to the halfway point.

INCREMENTAL REBUILD - STEP 4 of 4 (final): playback added (mlx-whisper from
step 1, Silero VAD from step 2, embedding/bank growth from step 3). This is
the piece most likely to have been unstable - a second concurrent native
audio stream - so it reuses lessons paid for earlier today rather than the
original playback design:
  - One persistent sd.OutputStream, opened once per session, never
    torn down and recreated per clip. The original approach used
    sd.play()/sd.stop() per clip, which repeatedly tears down and rebuilds
    a native PortAudio stream - under concurrent MLX load this produced an
    actual macOS crash report (malloc: "pointer being freed was not
    allocated"). Interrupting a clip here is just swapping a buffer and an
    index under a lock; the stream itself never closes until stop_session().
  - playback_mute_seconds is a short, fixed-length input mute, deliberately
    NOT scaled by clip duration (bank clips run up to ~30s; muting for a
    whole clip's duration made the system deaf to speech for that whole
    window after every single playback).
  - Skips restarting playback if the newest match is literally the same
    clip already playing.
See CLAUDE.md for why this file is being rebuilt one piece at a time
instead of all at once.

VAD inference is real work (a neural net forward pass, ~every 32ms per
channel) - too expensive to run safely inside PortAudio's real-time
callback, which has a hard per-buffer deadline. So the callback here does
only the minimum (copy the buffer onto a queue); _process_capture does the
actual VAD/state-machine work on an ordinary thread with no such deadline.
"""

DEFAULT_WHISPER_MODEL = "mlx-community/whisper-base.en-mlx"
DEFAULT_EMBED_MODEL = str(Path.home() / "models" / "olmo2-1b-4bit")


@dataclass
class AudioConfig:
    channels: int = 2
    device_sample_rate: int = 44100
    target_sample_rate: int = 16000  # Rate for Whisper
    vad_sample_rate: int = 16000  # rate silero-vad expects (8000 or 16000 only)
    blocksize: int = 1024  # Increased from 512 for more stable timing
    min_utterance_seconds: float = 0.5
    device_index: Optional[int] = None
    preroll_seconds: float = 0.5  # raw audio kept before a detected speech onset
    # VAD (silero). threshold is the speech-probability cutoff; min_silence_ms
    # is how long below-threshold audio has to persist before an utterance is
    # considered finished (silero's own hangover, no wall-clock gap timer
    # needed); speech_pad_ms pads each side of the detected span.
    vad_threshold: float = 0.5
    vad_min_silence_ms: int = 600
    vad_speech_pad_ms: int = 30
    whisper_model: str = DEFAULT_WHISPER_MODEL
    # Embedding + bank growth/search. embedder/bank are loaded if either
    # add_to_database (grow the corpus) or intervention_enabled (search +
    # play a match) is on - they share the same embedding step.
    add_to_database: bool = True
    intervention_enabled: bool = True
    embed_model: str = DEFAULT_EMBED_MODEL
    bank_csv: Optional[Path] = None
    output_device_index: Optional[int] = None
    playback_mute_seconds: float = 0.3  # fixed-length input-mute after playback starts
    max_playback_seconds: float = 7.0  # bank clips longer than this are never played back
    murmur_volume: float = 3.0  # 0-1 scale factor for the continuous background murmur


def debug_print_audio_stats(stage: str, data: np.ndarray, sample_rate: int):
    """Helper function to print audio statistics at various stages."""
    duration = len(data) / sample_rate
    if stage == "Before WAV Write":
        print(f"\n=== Processing Audio ===")
        print(f"Duration: {duration:.3f} seconds")
    else:
        print(f"\n=== Audio Stats at {stage} ===")
        print(f"Data shape: {data.shape}")
        print(f"Duration: {duration:.3f} seconds")
        print(f"Sample rate: {sample_rate} Hz")
        print(f"Number of samples: {len(data)}")
        print(f"Min value: {np.min(data):.3f}, Max value: {np.max(data):.3f}")
        print("================================\n")

class AudioProcessor:
    _instance = None
    _lock = threading.Lock()
    _stream = None
    _stream_lock = threading.Lock()  # Add dedicated stream lock
    _active_stream_thread = None  # Track which thread owns the stream

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance

    def __init__(self, session_dir, audio_dir, config: Optional[AudioConfig] = None):
        # Only initialize once
        if hasattr(self, '_initialized'):
            return

        self._initialized = True
        self.session_dir = session_dir
        self.audio_dir = audio_dir
        self.csv_path = f"{session_dir}/data.csv"
        self.config = config or AudioConfig()
        self.output_dir = Path(audio_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Extract pair ID from session directory
        try:
            self.pair_id = int(Path(session_dir).name)
        except ValueError:
            print(f"Warning: Could not extract pair ID from session directory {session_dir}, using default value 1")
            self.pair_id = 1

        self.preroll_samples = int(self.config.device_sample_rate * self.config.preroll_seconds)

        # Exact-ratio resampler (device_sample_rate -> vad_sample_rate), e.g.
        # 44100 -> 16000 reduces to 160/441. resample_poly is used per-block,
        # so an exact small ratio matters for cost.
        g = math.gcd(self.config.vad_sample_rate, self.config.device_sample_rate)
        self._resample_up = self.config.vad_sample_rate // g
        self._resample_down = self.config.device_sample_rate // g

        self.recording_state = {c: self._new_channel_state() for c in range(self.config.channels)}

        # Queues: raw_audio_queue is the handoff from the real-time callback
        # to _process_capture (see both) - bounded so a processing backlog
        # drops old blocks instead of growing memory without limit.
        self.raw_audio_queue = queue.Queue(maxsize=200)
        self.transcription_queue = queue.Queue()
        self.csv_queue = queue.Queue()
        self.embedding_queue = queue.Queue()

        self.current_image = None
        self.session_active = False

        self._setup_audio_device()

        # mlx-whisper caches its loaded model in a class-level attribute
        # with no locking of its own, so concurrent calls from multiple
        # threads aren't safe - this lock serializes them (paired with
        # running a single transcription thread below). openai-whisper
        # (pure PyTorch CPU) didn't have this constraint, which is why the
        # original version of this file ran two transcription threads.
        self.mlx_lock = threading.Lock()
        print(f"Warming up whisper model: {self.config.whisper_model}")
        with self.mlx_lock:
            mlx_whisper.transcribe(np.zeros(self.config.target_sample_rate, dtype=np.float32),
                                    path_or_hf_repo=self.config.whisper_model)

        self.embedder = None
        self.bank = None
        if self.config.add_to_database or self.config.intervention_enabled:
            print(f"Loading embedding model: {self.config.embed_model}")
            self.embedder = TextEmbedder(model_name=self.config.embed_model)
            self.bank = UtteranceBank(self.config.bank_csv or DEFAULT_BANK_CSV)

        self.playback_queue = queue.Queue()
        self.playback_active_until = 0.0  # input is ignored until this monotonic time
        self._playback_lock = threading.Lock()
        self._playback_buffer = np.zeros(0, dtype=np.int16)
        self._playback_pos = 0
        self._playback_key = None

        self.solo_fx = None
        self._murmur_buffer = np.zeros(0, dtype=np.int16)
        self._murmur_pos = 0
        if self.config.intervention_enabled:
            self.solo_fx = SoloFxIndex()
            murmur, murmur_rate = load_murmur()
            if murmur_rate != self.config.device_sample_rate:
                print(f"[effects] WARNING: murmur rate {murmur_rate} != device rate "
                      f"{self.config.device_sample_rate}, will play pitched/sped up")
            self._murmur_buffer = murmur

        # Initialize output file path
        self.output_file = Path(self.csv_path)
        self.output_lock = threading.Lock()

        # Create initial CSV if it doesn't exist
        if not self.output_file.exists():
            self._create_initial_csv()
        elif self.output_file.is_file():
            if os.stat(self.output_file).st_size == 0:
                print("Existing CSV file is empty. Creating new CSV file...")
                self._create_initial_csv()
            else:
                df = pd.read_csv(self.output_file)
                # check if it is empty or has no columns
                if df.empty or len(df.columns) != 6:
                    print("Existing CSV file has incorrect columns. Creating new CSV file...")
                    self._create_initial_csv()

    def _create_initial_csv(self):
        """Create the initial CSV file with headers."""
        df = pd.DataFrame(columns=[
            "pairID", "subID", "imgID", "audio_path", "text", "timestamp"
        ])
        df.to_csv(self.output_file, index=False)

    def _setup_audio_device(self):
        """Find and set up the Scarlett audio interface."""
        if self.config.device_index is None:
            devices = sd.query_devices()
            print("\nAvailable audio devices:")
            for i, device in enumerate(devices):
                print(f"{i}: {device['name']} (in: {device['max_input_channels']}, out: {device['max_output_channels']})")

            device_index, device_info = find_scarlett_device()
            if device_index is not None:
                self.config.device_index = device_index
                print(f"\nSelected Scarlett device: {device_info['name']}")
                print(f"Device details: {device_info}")

        if self.config.device_index is None:
            raise RuntimeError("Could not find Scarlett audio interface")

        # Verify the selected device
        device_info = sd.query_devices(self.config.device_index)
        print(f"\nUsing audio device: {device_info['name']}")
        print(f"Default samplerate: {device_info['default_samplerate']}")
        print(f"Input channels: {device_info['max_input_channels']}")
        if 'default_low_input_latency' in device_info:
            print(f"Default low input latency: {device_info['default_low_input_latency']}")
        if 'default_high_input_latency' in device_info:
            print(f"Default high input latency: {device_info['default_high_input_latency']}")

    def _new_channel_state(self):
        return {
            "recording": False,
            "data": None,
            "start_time": None,
            "preroll": np.zeros(self.preroll_samples, dtype=np.int32),
            "vad_resid": np.zeros(0, dtype=np.float32),
            "vad": VADIterator(
                load_silero_vad(),  # own model instance per channel - it carries
                                    # recurrent state across calls, so two channels
                                    # sharing one instance would corrupt each other
                threshold=self.config.vad_threshold,
                sampling_rate=self.config.vad_sample_rate,
                min_silence_duration_ms=self.config.vad_min_silence_ms,
                speech_pad_ms=self.config.vad_speech_pad_ms,
            ),
        }

    def _next_murmur_chunk(self, frames):
        """Next `frames` samples of the looping background murmur, wrapping
        at the end. Runs on the real-time callback thread - plain array
        slicing only, no lock needed (only that callback ever touches
        _murmur_pos)."""
        buf = self._murmur_buffer
        n = len(buf)
        if n == 0:
            return np.zeros(frames, dtype=np.int16)
        pos = self._murmur_pos
        end = pos + frames
        if end <= n:
            chunk = buf[pos:end]
            self._murmur_pos = end % n
        else:
            first = buf[pos:n]
            remaining = frames - len(first)
            second = buf[0:remaining]
            chunk = np.concatenate([first, second])
            self._murmur_pos = remaining
        return chunk

    def _record_audio(self):
        """Capture (and, if intervention is enabled, play back) audio via a
        single full-duplex sd.Stream.

        Input and output both default to the same physical Scarlett device.
        An earlier version of this opened two separate streams (InputStream
        + OutputStream) against that one device concurrently - each got its
        own CoreAudio I/O thread, and running both under load produced a
        real SIGSEGV (confirmed via macOS diagnostic reports, deep in the
        sounddevice/PortAudio callback bridge) regardless of how carefully
        either stream's lifecycle was managed. A single duplex stream is
        PortAudio's standard pattern for simultaneous I/O on one device and
        avoids that dual-stream contention entirely.
        """
        current_thread = threading.current_thread()

        with AudioProcessor._stream_lock:
            if AudioProcessor._stream is not None:
                if AudioProcessor._active_stream_thread == current_thread:
                    return
                else:
                    return

            AudioProcessor._active_stream_thread = current_thread

        def callback(indata, outdata, frames, time, status):
            if not self.session_active:
                raise sd.CallbackStop()
            if status:
                print(f"Status: {status}")
            # Deliberately minimal: this runs on PortAudio's real-time
            # callback thread, which has a hard per-call deadline
            # (~blocksize/samplerate). VAD inference is too slow/GIL-heavy
            # to do safely here - see _process_capture, which does the real
            # work on an ordinary thread. indata is a PortAudio-owned buffer
            # reused right after this returns, so it must be copied here,
            # not just referenced.
            try:
                self.raw_audio_queue.put_nowait(indata.copy())
            except queue.Full:
                pass  # drop a block under extreme backlog rather than block the callback

            if self.config.intervention_enabled:
                # Continuous background murmur (looping, scaled down) mixed
                # with whatever intervention clip is currently playing, if
                # any - the murmur plays the whole session regardless of
                # whether a match is active.
                mixed = (self._next_murmur_chunk(frames).astype(np.int32)
                         * self.config.murmur_volume).astype(np.int32)

                with self._playback_lock:
                    buf = self._playback_buffer
                    pos = self._playback_pos
                    n = min(frames, max(0, len(buf) - pos))
                    if n > 0:
                        mixed[:n] += buf[pos:pos + n].astype(np.int32)
                        self._playback_pos = pos + n

                np.clip(mixed, -32768, 32767, out=mixed)
                outdata[:, 0] = mixed.astype(np.int16)
            else:
                outdata.fill(0)

        try:
            with AudioProcessor._stream_lock:
                if AudioProcessor._stream is not None:
                    return

                out_device = self.config.output_device_index or self.config.device_index
                AudioProcessor._stream = sd.Stream(
                    device=(self.config.device_index, out_device),
                    channels=(self.config.channels, 1),
                    samplerate=self.config.device_sample_rate,
                    blocksize=self.config.blocksize,
                    dtype=('int32', 'int16'),
                    latency='high',
                    callback=callback,
                )

            with AudioProcessor._stream:
                while self.session_active:
                    tm.sleep(0.1)
        finally:
            with AudioProcessor._stream_lock:
                if AudioProcessor._active_stream_thread == current_thread:
                    AudioProcessor._stream = None
                    AudioProcessor._active_stream_thread = None

    def _process_capture(self):
        """Consumes raw blocks off raw_audio_queue (queued by the real-time
        callback in _record_audio, which stays minimal on purpose) and does
        the actual VAD/state-machine work here, on an ordinary thread with
        no hard real-time deadline to miss."""
        while self.session_active or not self.raw_audio_queue.empty():
            try:
                indata = self.raw_audio_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if tm.time() >= self.playback_active_until:
                for channel in range(self.config.channels):
                    self._process_channel_block(channel, indata[:, channel])

            self.raw_audio_queue.task_done()

    def _process_channel_block(self, channel, channel_data):
        """Feed one block's worth of raw samples for one channel through
        VAD, updating that channel's recording state."""
        state = self.recording_state[channel]

        # Pre-roll: raw audio from just before speech is detected, so the
        # segment we transcribe doesn't clip the first word. Grab it *before*
        # appending the current block, so it never double-counts the block
        # we're about to process.
        preroll_before = state["preroll"]
        state["preroll"] = np.concatenate([state["preroll"], channel_data])[-self.preroll_samples:]

        float_block = channel_data.astype(np.float32) / (2 ** 31)
        resampled = signal.resample_poly(float_block, self._resample_up, self._resample_down).astype(np.float32)
        buf = np.concatenate([state["vad_resid"], resampled])
        n_chunks = len(buf) // 512
        events = [state["vad"](buf[i * 512:(i + 1) * 512]) for i in range(n_chunks)]
        state["vad_resid"] = buf[n_chunks * 512:]

        for event in events:
            if event is None:
                continue
            if "start" in event and not state["recording"]:
                state["recording"] = True
                state["start_time"] = tm.time()
                state["data"] = preroll_before.copy()
                print(f"\nChannel {channel} starting recording")
            elif "end" in event and state["recording"]:
                self._finish_utterance(channel, state)

        if state["recording"]:
            state["data"] = np.concatenate([state["data"], channel_data])

    def _finish_utterance(self, channel, state):
        """VAD signaled the end of a speech span on this channel: package the
        accumulated audio for transcription if it's long enough, then reset."""
        if tm.time() - state["start_time"] > self.config.min_utterance_seconds:
            audio_data = state["data"]
            print(f"\nChannel {channel} finished recording")

            float_data = audio_data.astype(np.float32) / (2 ** 31)
            audio_int16 = (float_data * 32767).astype(np.int16)

            packet = {
                "channel": channel,
                "data": audio_int16,
                "sample_rate": self.config.device_sample_rate,
                "start_time": state["start_time"],
                "end_time": tm.time(),
                "image_id": self.current_image,
            }
            self.transcription_queue.put(packet)

        state["recording"] = False
        state["data"] = None
        state["start_time"] = None

    def _transcribe_audio(self):
        """Transcribe audio from the queue."""
        while self.session_active or not self.transcription_queue.empty():
            try:
                packet = self.transcription_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                # Resample for transcription first
                resampled_data = librosa.resample(
                    y=packet['data'].astype(np.float32) / 32767.0,  # Convert back to float32
                    orig_sr=packet['sample_rate'],
                    target_sr=self.config.target_sample_rate
                )

                # Attempt transcription before saving WAV
                with self.mlx_lock:
                    result = mlx_whisper.transcribe(resampled_data, path_or_hf_repo=self.config.whisper_model)
                transcribed_text = result['text'].strip()

                # Only save WAV and create CSV entry if transcription produced text
                if transcribed_text:
                    # Save audio to WAV file
                    timestamp = int(tm.time())
                    filename = f"utterance_{packet['channel']}_{timestamp}.wav"
                    filepath = self.output_dir / filename

                    # Write the original high-quality audio
                    wav_write(str(filepath), packet['sample_rate'], packet['data'])

                    output_row = {
                        "pairID": self.pair_id,  # Use extracted pair ID
                        "subID": packet["channel"],
                        "imgID": packet["image_id"],
                        "audio_path": str(filepath),
                        "text": transcribed_text,
                        "timestamp": packet["start_time"]
                    }

                    # Add to CSV queue
                    self.csv_queue.put(output_row)
                    print(f"Channel {packet['channel']} transcribed: {transcribed_text}")

                    if (self.config.add_to_database or self.config.intervention_enabled) and packet["image_id"]:
                        self.embedding_queue.put({
                            "text": transcribed_text,
                            "image_id": packet["image_id"],
                            "channel": packet["channel"],
                            "start_time": packet["start_time"],
                            "audio_path": str(filepath),
                            "duration": len(packet["data"]) / packet["sample_rate"],
                        })
                else:
                    print(f"Channel {packet['channel']}: No speech detected in audio segment")

            except Exception as e:
                print(f"Error processing audio: {e}")

            finally:
                self.transcription_queue.task_done()

    def _embed_and_add(self):
        """Embed each transcript, optionally add it to the bank (so later
        sessions can match against it), and optionally look up the closest
        same-image, different-pair utterance to play back. Independent:
        add_to_database controls whether the corpus grows; intervention_enabled
        controls whether we search it."""
        while self.session_active or not self.embedding_queue.empty():
            try:
                item = self.embedding_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                with self.mlx_lock:
                    vec = self.embedder.embed([item["text"]])[0]

                if self.config.add_to_database:
                    self.bank.add_utterance(vec, {
                        "pairID": self.pair_id,
                        "subID": item["channel"],
                        "imgID": item["image_id"],
                        "text": item["text"],
                        "start": 0.0,
                        "end": item["duration"],
                        "timestamp": item["start_time"],
                        "audio_path": item["audio_path"],
                    })
                    print(f"[bank] added utterance for {item['image_id']}: \"{item['text'][:60]}\"")

                if self.config.intervention_enabled:
                    match = self.bank.find_match(vec, item["image_id"], exclude_pair_id=self.pair_id,
                                                  max_duration_seconds=self.config.max_playback_seconds)
                    if match is not None:
                        print(f"[intervention] match (sim={match['similarity']:.2f}): \"{match['text'][:60]}\"")
                        self.playback_queue.put({"match": match})
                    else:
                        print(f"[intervention] no cross-pair match yet for {item['image_id']}")

            except Exception as e:
                print(f"Error embedding/matching: {e}")

            finally:
                self.embedding_queue.task_done()

    def _playback(self):
        """Play back matched utterances by swapping the buffer that the
        duplex stream's callback (in _record_audio) reads from - no
        separate output stream to manage here. Always plays the newest match and
        interrupts whatever's currently playing rather than queuing a
        backlog - a reaction that plays 20s late (behind one long clip)
        breaks the "immediate" effect far worse than skipping it would.
        Skips restarting if the newest match is literally the same clip
        already playing."""
        while self.session_active or not self.playback_queue.empty():
            try:
                item = self.playback_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            dropped = 0
            while True:
                try:
                    newer = self.playback_queue.get_nowait()
                except queue.Empty:
                    break
                self.playback_queue.task_done()
                item = newer
                dropped += 1
            if dropped:
                print(f"[playback] dropped {dropped} stale match(es), playing the newest")

            try:
                match = item["match"]
                key = (match["audio_path"], match["start"], match["end"])
                if key == self._playback_key:
                    print("[playback] newest match is the clip already playing - not restarting")
                else:
                    fx_path = self.solo_fx.path_for(match["audio_path"], match["start"], match["end"]) \
                        if self.solo_fx else None
                    if fx_path is not None and fx_path.exists():
                        audio, rate = load_audio_file(fx_path)
                        print(f"[playback] using solo_fx: {fx_path.name}")
                    else:
                        audio, rate = self.bank.load_audio(match)
                    if rate != self.config.device_sample_rate:
                        print(f"[playback] WARNING: clip rate {rate} != device rate "
                              f"{self.config.device_sample_rate}, will play pitched/sped up")
                    duration = len(audio) / rate

                    with self._playback_lock:
                        self._playback_buffer = audio
                        self._playback_pos = 0
                        self._playback_key = key
                    self.playback_active_until = tm.time() + self.config.playback_mute_seconds

                    print(f"[playback] playing {duration:.1f}s clip")

            except Exception as e:
                print(f"Error playing back: {e}")

            finally:
                self.playback_queue.task_done()

    def _csv_writer(self):
        """Write transcribed data to CSV."""
        while self.session_active or not self.csv_queue.empty():
            try:
                row = self.csv_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                # Only write to CSV if there's actual text content
                if row['text'] and row['text'].strip():  # Check if text exists and isn't just whitespace
                    with self.output_lock:
                        # Read existing CSV
                        if self.output_file.exists():
                            df = pd.read_csv(self.output_file)
                        else:
                            df = pd.DataFrame(columns=[
                                "pairID", "subID", "imgID", "audio_path", "text", "timestamp"
                            ])

                        # Append new row
                        df = pd.concat([
                            df,
                            pd.DataFrame([row])
                        ], ignore_index=True)

                        # Save back to CSV
                        df.to_csv(self.output_file, index=False)
                        print(f"Saved new row to CSV: {row}")
                else:
                    print("Skipping empty transcription")

            except Exception as e:
                print(f"Error writing to CSV: {e}")

            finally:
                self.csv_queue.task_done()

    def start_session(self):
        """Start recording and transcription threads."""
        self.recording_state = {c: self._new_channel_state() for c in range(self.config.channels)}
        self.session_active = True

        # Start recording thread (only one)
        self.record_thread = threading.Thread(target=self._record_audio, name="RecordingThread")
        self.record_thread.start()

        self.capture_thread = threading.Thread(target=self._process_capture, name="CaptureProcessThread")
        self.capture_thread.daemon = True
        self.capture_thread.start()

        # One transcription thread, not two - see the mlx_lock comment in
        # __init__ for why (mlx-whisper's model cache isn't safe for
        # concurrent calls from multiple threads).
        self.transcribe_thread = threading.Thread(target=self._transcribe_audio, name="TranscriptionThread")
        self.transcribe_thread.daemon = True
        self.transcribe_thread.start()

        # Start CSV writer thread
        self.csv_thread = threading.Thread(target=self._csv_writer, name="CSVWriterThread")
        self.csv_thread.daemon = True
        self.csv_thread.start()

        self.embed_thread = None
        if self.config.add_to_database or self.config.intervention_enabled:
            self.embed_thread = threading.Thread(target=self._embed_and_add, name="EmbedThread")
            self.embed_thread.daemon = True
            self.embed_thread.start()

        self.playback_thread = None
        if self.config.intervention_enabled:
            self.playback_thread = threading.Thread(target=self._playback, name="PlaybackThread")
            self.playback_thread.daemon = True
            self.playback_thread.start()

    def stop_session(self):
        """Stop all threads and cleanup."""
        self.session_active = False

        # Deliberately NOT closing AudioProcessor._stream here. _record_audio
        # holds it in a `with AudioProcessor._stream:` block, whose __exit__
        # already closes it (and its own finally clause clears the class
        # attributes) once session_active goes False and that loop notices.
        # Closing it here too would be a double-close on the same native
        # PortAudio stream object.

        # Wait for recording to finish
        if hasattr(self, 'record_thread'):
            self.record_thread.join()

        self.raw_audio_queue.join()
        # Wait for transcription queue to empty
        self.transcription_queue.join()

        if hasattr(self, 'capture_thread'):
            self.capture_thread.join(timeout=3.0)
        if hasattr(self, 'transcribe_thread'):
            self.transcribe_thread.join(timeout=3.0)

        # Wait for CSV queue to empty
        self.csv_queue.join()

        self.embedding_queue.join()
        if getattr(self, 'embed_thread', None) is not None:
            self.embed_thread.join(timeout=3.0)

        self.playback_queue.join()
        if getattr(self, 'playback_thread', None) is not None:
            self.playback_thread.join(timeout=3.0)

        # Clean up orphaned audio files
        self.cleanup_orphaned_audio()

    def update_current_image(self, image_id: str):
        """Update the current image ID."""
        self.current_image = image_id
        print(f"Updated current image to: {image_id}")

    def cleanup_orphaned_audio(self):
        """Delete audio files that don't have corresponding entries in the CSV."""
        try:
            # Read the CSV file
            df = pd.read_csv(self.csv_path)

            # Get all audio paths from CSV
            valid_audio_paths = set(df['audio_path'].values)

            # Get all wav files in the audio directory
            audio_dir = Path(self.audio_dir)
            all_audio_files = set(str(f) for f in audio_dir.glob('*.wav'))

            # Find orphaned files (files that exist but aren't in CSV)
            orphaned_files = all_audio_files - valid_audio_paths

            # Delete orphaned files
            for file_path in orphaned_files:
                try:
                    os.remove(file_path)
                    print(f"Deleted orphaned audio file: {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")

            if orphaned_files:
                print(f"Cleaned up {len(orphaned_files)} orphaned audio files")
            else:
                print("No orphaned audio files found")

        except Exception as e:
            print(f"Error during cleanup: {e}")

if __name__ == "__main__":
    print("Starting audio processor...")
    processor = AudioProcessor("session_dir", "audio_outputs2")

    processor.start_session()
    print("Session started")

    tm.sleep(30)

    processor.stop_session()
    print("Session stopped")
