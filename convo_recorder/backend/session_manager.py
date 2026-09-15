import os
from pathlib import Path

def setup_session(test_mode=False):
    # Get absolute path for data directory
    base_dir = Path(__file__).parent.parent.absolute()
    data_dir = base_dir / "data"
    data_dir.mkdir(exist_ok=True)

    if test_mode:
        # Reuse a single fixed folder instead of creating a new numbered
        # session each run, so test runs don't pollute real session numbering.
        session_dir = data_dir / "test"
        session_dir.mkdir(exist_ok=True)
        audio_dir = session_dir / "audio"
        audio_dir.mkdir(exist_ok=True)
        print(f"[TEST MODE] Reusing test session at: {session_dir.absolute()}")
        print(f"[TEST MODE] Audio directory at: {audio_dir.absolute()}")
        return str(session_dir.absolute()), str(audio_dir.absolute())

    # Count existing sessions
    existing_sessions = sorted([
        d for d in data_dir.iterdir() 
        if d.is_dir() and d.name.isdigit()
    ])
    
    # Create new session number
    session_num = len(existing_sessions)
    
    # Create new session directory with padded number
    session_dir = data_dir / f"{session_num:03d}"
    session_dir.mkdir(exist_ok=True)
    
    # Create audio subdirectory
    audio_dir = session_dir / "audio"
    audio_dir.mkdir(exist_ok=True)
    
    print(f"Created new session at: {session_dir.absolute()}")
    print(f"Audio directory at: {audio_dir.absolute()}")
    
    return str(session_dir.absolute()), str(audio_dir.absolute())
