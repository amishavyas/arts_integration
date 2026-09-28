"""Audio device lookup shared by audio_processor.py (the realtime pipeline,
heavy ML deps) and preflight_dialog.py (a lightweight standalone window) -
kept in its own module so the dialog doesn't have to import mlx/whisper/
silero/torch just to find the input device."""

import sounddevice as sd


def find_scarlett_device():
    """Look for a connected Focusrite Scarlett USB audio interface.

    Returns (device_index, device_info) if found, otherwise (None, None).
    Never raises - callers decide how to handle an absent device.
    """
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if ("Scarlett" in device["name"] and
                "USB" in device["name"] and
                device["max_input_channels"] > 0 and
                "virtual" not in device["name"].lower()):
            return i, device
    return None, None
