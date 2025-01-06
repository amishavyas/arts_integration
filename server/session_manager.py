import os
from pathlib import Path

def setup_session():
    # Get base data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Count existing sessions
    existing_sessions = [d for d in data_dir.iterdir() if d.is_dir()]
    session_num = len(existing_sessions)
    
    # Create new session directory with padded number
    session_dir = data_dir / f"{session_num:03d}"
    session_dir.mkdir(exist_ok=True)
    
    # Create audio subdirectory
    audio_dir = session_dir / "audio"
    audio_dir.mkdir(exist_ok=True)
    
    return str(session_dir), str(audio_dir)
