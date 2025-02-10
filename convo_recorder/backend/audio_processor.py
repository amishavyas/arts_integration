import sounddevice as sd
import numpy as np
import threading
import queue
import time as tm
import os
import warnings
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import whisper
import pandas as pd
from scipy.io.wavfile import write as wav_write
import torch
from scipy import signal
import librosa

# Suppress the FP16 warning from Whisper
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

"""
The individual channels for the interface should be set to max volume. Set the master volume to the halfway point. 
"""

# Set max number of threads for PyTorch
torch.set_num_threads(1)

@dataclass
class AudioConfig:
    channels: int = 2
    device_sample_rate: int = 44100
    target_sample_rate: int = 16000  # Rate for Whisper
    blocksize: int = 1024  # Increased from 512 for more stable timing
    threshold: float = 0.02  
    gap_seconds: float = 1.5
    min_utterance_seconds: float = 0.5
    device_index: Optional[int] = None
    buffer_size: int = 20  # Number of blocks to buffer

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
        
        self.buffer_samples = self.config.blocksize
        
        # Initialize recording state with buffers
        self.recording_state = {
            0: {
                "recording": False, 
                "data": None, 
                "last_active": 0, 
                "start_time": None,
                "buffer": np.zeros(self.buffer_samples),
                "timing_buffer": []  # Add timing buffer
            },
            1: {
                "recording": False, 
                "data": None, 
                "last_active": 0, 
                "start_time": None,
                "buffer": np.zeros(self.buffer_samples),
                "timing_buffer": []  # Add timing buffer
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
            print("\nAvailable audio devices:")
            for i, device in enumerate(devices):
                print(f"{i}: {device['name']} (in: {device['max_input_channels']}, out: {device['max_output_channels']})")
                
            for i, device in enumerate(devices):
                # Look specifically for USB Scarlett and verify it's not a virtual device
                if ("Scarlett" in device["name"] and 
                    "USB" in device["name"] and 
                    device["max_input_channels"] > 0 and
                    "virtual" not in device["name"].lower()):
                    self.config.device_index = i
                    print(f"\nSelected Scarlett device: {device['name']}")
                    print(f"Device details: {device}")
                    break
        
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

    def _calculate_rms(self, data: np.ndarray) -> float:
        """Calculate Root Mean Square of audio data."""
        # Convert int32 to float32 for RMS calculation
        float_data = data.astype(np.float32) / (2**31)  # Normalize by full 32-bit range
        return np.sqrt(np.mean(np.square(float_data)))

    def _record_audio(self):
        """Record audio from both channels."""
        current_thread = threading.current_thread()
        
        with AudioProcessor._stream_lock:
            if AudioProcessor._stream is not None:
                if AudioProcessor._active_stream_thread == current_thread:
                    return
                else:
                    return
            
            AudioProcessor._active_stream_thread = current_thread
        
        last_time = [tm.time()]
        expected_interval = self.config.blocksize / self.config.device_sample_rate
        first_callback = [True]
        
        def callback(indata, frames, time, status):
            if not self.session_active:
                raise sd.CallbackStop()
            
            current_time = tm.time()
            interval = current_time - last_time[0]
            
            # Update timing buffer and calculate average interval
            for channel in range(self.config.channels):
                state = self.recording_state[channel]
                state["timing_buffer"].append(interval)
                if len(state["timing_buffer"]) > self.config.buffer_size:
                    state["timing_buffer"].pop(0)
            
            last_time[0] = current_time
            
            if status:
                print(f"Status: {status}")
            
            for channel in range(self.config.channels):
                channel_data = indata[:, channel]
                rms = self._calculate_rms(channel_data)
                state = self.recording_state[channel]
                
                # Start recording if above threshold
                if rms > self.config.threshold and not state["recording"]:
                    state["recording"] = True
                    print(f'\nChannel {channel} starting recording')
                    
                    state["start_time"] = tm.time()
                    state["data"] = channel_data.copy()
                    state["last_active"] = tm.time()
                
                # If recording, append data
                elif state["recording"]:
                    state["data"] = np.concatenate([state["data"], channel_data])
                    
                    # Update last_active only if above threshold
                    if rms > self.config.threshold:
                        state["last_active"] = tm.time()
                    
                    # Check if we should stop recording
                    if tm.time() - state["last_active"] > self.config.gap_seconds:
                        # check if the recording has been going on for at least the minimum utterance time
                        if tm.time() - state["start_time"] > self.config.min_utterance_seconds:
                            audio_data = state["data"]
                            
                            print(f"\nChannel {channel} finished recording")
                            
                            # Convert to float32 and scale for int16
                            float_data = audio_data.astype(np.float32) / (2**31)
                            audio_int16 = (float_data * 32767).astype(np.int16)
                            
                            packet = {
                                "channel": channel,
                                "data": audio_int16,
                                "sample_rate": self.config.device_sample_rate,
                                "start_time": state["start_time"],
                                "end_time": tm.time(),
                                "image_id": self.current_image
                            }
                            
                            self.transcription_queue.put(packet)

                        # Reset state
                        state["recording"] = False
                        state["data"] = None
                        state["start_time"] = None
                        state["buffer"] = np.zeros(self.buffer_samples)
                else:
                    # Update the rolling buffer
                    state["buffer"] = channel_data
        
        try:
            with AudioProcessor._stream_lock:
                if AudioProcessor._stream is not None:
                    return
                    
                AudioProcessor._stream = sd.InputStream(
                    device=self.config.device_index,
                    channels=self.config.channels,
                    samplerate=self.config.device_sample_rate,
                    blocksize=self.config.blocksize,
                    dtype=np.int32,
                    latency='high',
                    callback=callback,
                    prime_output_buffers_using_stream_callback=True
                )
                
            with AudioProcessor._stream:
                while self.session_active:
                    tm.sleep(0.1)
        finally:
            with AudioProcessor._stream_lock:
                if AudioProcessor._active_stream_thread == current_thread:
                    AudioProcessor._stream = None
                    AudioProcessor._active_stream_thread = None

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
                result = self.model.transcribe(resampled_data)
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
                else:
                    print(f"Channel {packet['channel']}: No speech detected in audio segment")
                    
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
        # Initialize/reset recording state
        for channel in range(self.config.channels):
            self.recording_state[channel] = {
                "recording": False,
                "data": None,
                "last_active": 0,
                "start_time": None,
                "buffer": np.zeros(self.buffer_samples),
                "timing_buffer": []
            }
        
        self.session_active = True
        
        # Start recording thread (only one)
        self.record_thread = threading.Thread(target=self._record_audio, name="RecordingThread")
        self.record_thread.start()
        
        # Start transcription threads (these should only transcribe, not record)
        self.transcribe_threads = []
        for i in range(2):  # Two transcription threads
            thread = threading.Thread(target=self._transcribe_audio, name=f"TranscriptionThread-{i}")
            thread.daemon = True
            thread.start()
            self.transcribe_threads.append(thread)
        
        # Start CSV writer thread
        self.csv_thread = threading.Thread(target=self._csv_writer, name="CSVWriterThread")
        self.csv_thread.daemon = True
        self.csv_thread.start()
            
    def stop_session(self):
        """Stop all threads and cleanup."""
        self.session_active = False
        
        # Close the stream if it exists
        with AudioProcessor._stream_lock:
            if AudioProcessor._stream is not None:
                try:
                    AudioProcessor._stream.close()
                except:
                    pass
                AudioProcessor._stream = None
                AudioProcessor._active_stream_thread = None
        
        # Wait for recording to finish
        if hasattr(self, 'record_thread'):
            self.record_thread.join()
        
        # Wait for transcription queue to empty
        self.transcription_queue.join()
        
        # Wait for CSV queue to empty
        self.csv_queue.join()
        
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