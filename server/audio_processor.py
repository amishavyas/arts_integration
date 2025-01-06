import sounddevice as sd
import numpy as np
import threading
import queue
import time as tm
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import whisper
import pandas as pd
from scipy.io.wavfile import write as wav_write
import torch

"""
The individual channels for the interface should be set to max volume. Set the master volume to the halfway point. 
"""

# Set max number of threads for PyTorch
torch.set_num_threads(1)

@dataclass
class AudioConfig:
    channels: int = 2
    sample_rate: int = 16000
    blocksize: int = 400
    threshold: float = 0.05
    gap_seconds: float = 2.0
    min_utterance_seconds: float = 1.0
    device_index: Optional[int] = None

class AudioProcessor:
    def __init__(self, session_dir, audio_dir, config: Optional[AudioConfig] = None):
        self.session_dir = session_dir
        self.audio_dir = audio_dir
        self.csv_path = f"{session_dir}/data.csv"
        self.config = config or AudioConfig()
        self.output_dir = Path(audio_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate buffer size for 500ms of audio
        self.buffer_samples = int(self.config.sample_rate * 0.5)  # 500ms buffer
        
        # Initialize recording state with buffers
        self.recording_state = {
            0: {
                "recording": False, 
                "data": None, 
                "last_active": 0, 
                "start_time": None,
                "buffer": np.zeros(self.buffer_samples)
            },
            1: {
                "recording": False, 
                "data": None, 
                "last_active": 0, 
                "start_time": None,
                "buffer": np.zeros(self.buffer_samples)
            }
        }
        
        # Initialize queues for transcription and CSV writing
        self.transcription_queue = queue.Queue()
        self.csv_queue = queue.Queue()
        
        self.current_image = None
        self.session_active = False
        
        self._setup_audio_device()
        self.model = whisper.load_model("tiny", device="cpu")
        
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
            for i, device in enumerate(devices):
                if "Scarlett" in device["name"] and device["max_input_channels"] > 0:
                    self.config.device_index = i
                    break
        
        if self.config.device_index is None:
            raise RuntimeError("Could not find Scarlett audio interface")

    def _calculate_rms(self, data: np.ndarray) -> float:
        """Calculate Root Mean Square of audio data."""
        return np.sqrt(np.mean(np.square(data)))

    def _record_audio(self):
        """Record audio from both channels."""
        def callback(indata, frames, time, status):
            if status:
                print(f"Status: {status}")
            
            # Process each channel separately
            for channel in range(self.config.channels):
                channel_data = indata[:, channel]
                rms = self._calculate_rms(channel_data)
                
                state = self.recording_state[channel]
                
                # Update circular buffer when not recording
                if not state["recording"]:
                    # Roll the buffer and add new data
                    state["buffer"] = np.roll(state["buffer"], -len(channel_data))
                    state["buffer"][-len(channel_data):] = channel_data
                
                if rms > self.config.threshold:
                    if not state["recording"]:
                        state["recording"] = True
                        print(f'Channel {channel} starting recording. RMS: {rms:.3f}')
                        state["start_time"] = tm.time() - 0.5  # Adjust start time to account for buffer
                        # Initialize recording with buffer contents followed by current data
                        state["data"] = np.concatenate([state["buffer"], channel_data])
                    else:
                        if state["data"] is not None:
                            state["data"] = np.concatenate([state["data"], channel_data])
                    state["last_active"] = tm.time()
                    
                elif state["recording"]:
                    if state["data"] is not None:
                        state["data"] = np.concatenate([state["data"], channel_data])
                    
                    if tm.time() - state["last_active"] > self.config.gap_seconds:
                        if tm.time() - state["start_time"] > self.config.min_utterance_seconds:
                            # Scale float32 [-1.0, 1.0] to int16 range
                            audio_data = (state["data"] * 32767).astype(np.int16)
                            
                            packet = {
                                "channel": channel,
                                "data": audio_data,
                                "start_time": state["start_time"],
                                "end_time": tm.time(),
                                "image_id": self.current_image
                            }
                            
                            self.transcription_queue.put(packet)
                            print(f"Added packet to transcription queue for channel {channel}")
                            
                        # Reset state
                        state["recording"] = False
                        state["data"] = None
                        state["start_time"] = None

        with sd.InputStream(
            device=self.config.device_index,
            channels=self.config.channels,
            samplerate=self.config.sample_rate,
            blocksize=self.config.blocksize,
            callback=callback
        ):
            print("Recording stream started")
            while self.session_active:
                tm.sleep(0.1)

    def _transcribe_audio(self):
        """Transcribe audio from the queue."""
        while self.session_active or not self.transcription_queue.empty():
            try:
                packet = self.transcription_queue.get(timeout=1.0)
            except queue.Empty:
                continue
                
            try:
                # Save audio to WAV file
                timestamp = int(tm.time())
                filename = f"utterance_{packet['channel']}_{timestamp}.wav"
                filepath = self.output_dir / filename
                
                # Write the already-scaled int16 data
                wav_write(str(filepath), self.config.sample_rate, packet['data'])
                
                # Transcribe audio
                print(f"Transcribing audio from channel {packet['channel']}")
                result = self.model.transcribe(str(filepath))
                
                output_row = {
                    "pairID": 1,
                    "subID": packet["channel"],
                    "imgID": packet["image_id"],
                    "audio_path": str(filepath),
                    "text": result['text'].strip(),
                    "timestamp": packet["start_time"]
                }
                
                # Add to CSV queue
                self.csv_queue.put(output_row)
                print(f"Channel {packet['channel']} transcribed: {result['text']}")
                    
            except Exception as e:
                print(f"Error processing audio: {e}")
                
            finally:
                self.transcription_queue.task_done()

    def _csv_writer(self):
        """Write transcribed data to CSV."""
        while self.session_active or not self.csv_queue.empty():
            try:
                row = self.csv_queue.get(timeout=1.0)
            except queue.Empty:
                continue
                
            try:
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
            
            except Exception as e:
                print(f"Error writing to CSV: {e}")
            
            finally:
                self.csv_queue.task_done()

    def start_session(self):
        """Start recording and transcription threads."""
        self.session_active = True
        
        # Start recording thread
        self.record_thread = threading.Thread(target=self._record_audio)
        self.record_thread.start()
        
        # Start transcription threads
        self.transcribe_threads = []
        for _ in range(2):  # Two transcription threads
            thread = threading.Thread(target=self._transcribe_audio)
            thread.daemon = True
            thread.start()
            self.transcribe_threads.append(thread)
        
        # Start CSV writer thread
        self.csv_thread = threading.Thread(target=self._csv_writer)
        self.csv_thread.daemon = True
        self.csv_thread.start()
            
    def stop_session(self):
        """Stop all threads and cleanup."""
        print("Stopping session...")
        self.session_active = False
        
        # Wait for recording to finish
        self.record_thread.join()
        print("Recording thread stopped")
        
        # Wait for transcription queue to empty
        self.transcription_queue.join()
        print("Transcription queue emptied")
        
        # Wait for CSV queue to empty
        self.csv_queue.join()
        print("CSV queue emptied")
        
    def update_current_image(self, image_id: str):
        """Update the current image ID."""
        self.current_image = image_id
        print(f"Updated current image to: {image_id}")

if __name__ == "__main__":
    print("Starting audio processor...")
    processor = AudioProcessor("session_dir", "audio_outputs2")
    
    processor.start_session()
    print("Session started")
    
    # Test with a few images
    for img in ["img_01", "img_02", "img_03"]:
        processor.update_current_image(img)
        print(f"Processing image: {img}")
        tm.sleep(20)  # Record for 20 seconds per image
    
    processor.stop_session()
    print("Session stopped")