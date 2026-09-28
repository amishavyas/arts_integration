"""Preflight dialog shown before each experiment run, launched from run.py
before the backend/frontend start. Lets the RA confirm the mic input is
live (per-channel waveforms) and choose real-vs-devdata and
intervention-vs-data-collection-only before the browser experiment opens.

Plain tkinter (stdlib) + sounddevice for a short-lived input stream - no new
dependencies, and deliberately doesn't import audio_processor.py (which
pulls in mlx-whisper/silero-vad/torch) since all this needs is
find_scarlett_device(), now split out into audio_devices.py for exactly this
reason.

Closes its own input stream before returning, so the backend can claim the
Scarlett device exclusively once it starts.

show_preflight_dialog() returns a dict of choices, or None if the RA
cancelled - run.py should not start the experiment in that case.
"""

import sys
from pathlib import Path

import numpy as np
import sounddevice as sd
import tkinter as tk

sys.path.insert(0, str(Path(__file__).parent / "backend"))
from audio_devices import find_scarlett_device
from session_manager import next_session_number

WINDOW_W = 760
WAVE_H = 200
HISTORY_SECONDS = 3.0
FPS = 20


class PreflightDialog:
    def __init__(self):
        self.result = None  # set on Start; stays None on cancel/close
        self.stream = None
        self._wave_bufs = None

        self.root = tk.Tk()
        self.root.title("Arts Integration - Preflight")
        self.root.resizable(False, False)

        self.data_mode = tk.StringVar(value="real")
        self.intervention_mode = tk.StringVar(value="full")
        self.add_to_database = tk.BooleanVar(value=True)

        self._build_ui()

        self.device_index, self.device_info = find_scarlett_device()
        if self.device_index is not None:
            self._start_stream()
        else:
            self.status_label.config(
                text="No Scarlett interface detected - waveforms unavailable.\n"
                     "You can still start; the backend will show the same warning."
            )

        self._update_pair_id_preview()
        self.root.protocol("WM_DELETE_WINDOW", self._on_cancel)
        self._tick()

    # -- UI -------------------------------------------------------------
    def _build_ui(self):
        pad = {"padx": 16, "pady": 8}

        top = tk.Frame(self.root)
        top.pack(fill="x", **pad)
        tk.Label(top, text="Pair ID:", font=("Helvetica", 13, "bold")).pack(side="left")
        self.pair_id_label = tk.Label(top, text="--", font=("Helvetica", 13))
        self.pair_id_label.pack(side="left", padx=(6, 0))

        wave_frame = tk.Frame(self.root)
        wave_frame.pack(fill="x", **pad)
        tk.Label(wave_frame, text="Channel 0").pack(anchor="w")
        self.canvas0 = tk.Canvas(wave_frame, width=WINDOW_W - 32, height=WAVE_H, bg="black")
        self.canvas0.pack()
        tk.Label(wave_frame, text="Channel 1").pack(anchor="w")
        self.canvas1 = tk.Canvas(wave_frame, width=WINDOW_W - 32, height=WAVE_H, bg="black")
        self.canvas1.pack()

        self.status_label = tk.Label(self.root, text="", fg="#aa0000",
                                      wraplength=WINDOW_W - 32, justify="left")
        self.status_label.pack(fill="x", **pad)

        mode_frame = tk.LabelFrame(self.root, text="Session type")
        mode_frame.pack(fill="x", **pad)
        tk.Radiobutton(mode_frame, text="Real participant", variable=self.data_mode,
                        value="real", command=self._update_pair_id_preview).pack(anchor="w")
        tk.Radiobutton(mode_frame, text="Test / dev data", variable=self.data_mode,
                        value="devdata", command=self._update_pair_id_preview).pack(anchor="w")

        interv_frame = tk.LabelFrame(self.root, text="Mode")
        interv_frame.pack(fill="x", **pad)
        tk.Radiobutton(interv_frame, text="Full utterance intervention", variable=self.intervention_mode,
                        value="full").pack(anchor="w")
        tk.Radiobutton(interv_frame, text="Data collection only", variable=self.intervention_mode,
                        value="collection_only").pack(anchor="w")
        tk.Checkbutton(interv_frame, text="Add utterances to database (so future sessions can match against them)",
                       variable=self.add_to_database).pack(anchor="w", pady=(4, 0))

        btn_frame = tk.Frame(self.root)
        btn_frame.pack(fill="x", **pad)
        tk.Button(btn_frame, text="Start", command=self._on_start, width=12,
                  font=("Helvetica", 12, "bold")).pack(side="right")
        tk.Button(btn_frame, text="Cancel", command=self._on_cancel, width=12).pack(side="right", padx=(0, 8))

    def _update_pair_id_preview(self):
        devdata = self.data_mode.get() == "devdata"
        n = next_session_number(devdata=devdata)
        folder = "devdata" if devdata else "data"
        self.pair_id_label.config(text=f"{n:03d}   ({folder}/{n:03d})")

    # -- audio ------------------------------------------------------------
    def _start_stream(self):
        rate = int(self.device_info["default_samplerate"])
        channels = self.device_info["max_input_channels"]
        history_len = int(rate * HISTORY_SECONDS)
        self._wave_bufs = [np.zeros(history_len, dtype=np.float32) for _ in range(channels)]

        def callback(indata, frames, time_info, status):
            for ch in range(min(channels, indata.shape[1])):
                block = indata[:, ch].astype(np.float32) / (2 ** 31)
                buf = np.concatenate([self._wave_bufs[ch], block])
                self._wave_bufs[ch] = buf[-history_len:]

        try:
            self.stream = sd.InputStream(
                device=self.device_index, channels=channels, samplerate=rate,
                blocksize=1024, dtype=np.int32, callback=callback,
            )
            self.stream.start()
        except Exception as e:
            self.status_label.config(text=f"Could not open input stream for waveform preview: {e}")
            self.stream = None

    def _draw_waveform(self, canvas, buf):
        canvas.delete("all")
        w = int(canvas["width"])
        h = int(canvas["height"])
        mid = h / 2
        canvas.create_line(0, mid, w, mid, fill="#333333")
        if buf is None:
            return
        step = max(1, len(buf) // w)
        points = []
        for x in range(w):
            chunk = buf[x * step:(x + 1) * step]
            amp = float(np.max(np.abs(chunk))) if len(chunk) else 0.0
            points.extend([x, mid - min(1.0, amp) * mid])
        if len(points) >= 4:
            canvas.create_line(*points, fill="#4CAF50", width=1)

    def _tick(self):
        if self._wave_bufs is not None:
            self._draw_waveform(self.canvas0, self._wave_bufs[0] if len(self._wave_bufs) > 0 else None)
            self._draw_waveform(self.canvas1, self._wave_bufs[1] if len(self._wave_bufs) > 1 else None)
        self.root.after(int(1000 / FPS), self._tick)

    # -- actions ------------------------------------------------------------
    def _close_stream(self):
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

    def _on_start(self):
        self._close_stream()
        self.result = {
            "devdata": self.data_mode.get() == "devdata",
            "intervention_enabled": self.intervention_mode.get() == "full",
            "add_to_database": self.add_to_database.get(),
        }
        self.root.destroy()

    def _on_cancel(self):
        self._close_stream()
        self.result = None
        self.root.destroy()

    def run(self):
        self.root.mainloop()
        return self.result


def show_preflight_dialog():
    return PreflightDialog().run()


if __name__ == "__main__":
    print(show_preflight_dialog())
