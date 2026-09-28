from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.absolute()


def _count_existing(base_dir: Path) -> int:
    if not base_dir.exists():
        return 0
    return len([d for d in base_dir.iterdir() if d.is_dir() and d.name.isdigit()])


def next_session_number(devdata: bool = False) -> int:
    """Read-only preview of the session number the next setup_session() call
    would create, for the preflight dialog's pair-ID display. Creates
    nothing."""
    base_dir = REPO_ROOT / ("devdata" if devdata else "data")
    return _count_existing(base_dir)


def _numbered_session(base_dir: Path, label: str):
    base_dir.mkdir(parents=True, exist_ok=True)
    session_num = _count_existing(base_dir)
    session_dir = base_dir / f"{session_num:03d}"
    session_dir.mkdir(exist_ok=True)
    audio_dir = session_dir / "audio"
    audio_dir.mkdir(exist_ok=True)
    print(f"[{label}] Created new session at: {session_dir.absolute()}")
    print(f"[{label}] Audio directory at: {audio_dir.absolute()}")
    return str(session_dir.absolute()), str(audio_dir.absolute())


def setup_session(test_mode: bool = False, devdata: bool = False):
    """Set up (and return) the (session_dir, audio_dir) for a run.

    test_mode: reuse a single fixed data/test/ folder every run (the CLI
        --test flag on run.py - quick manual dev runs without the preflight
        dialog. Never numbered, so it doesn't pollute real or devdata
        session numbering, and each run overwrites the last).
    devdata: number sessions under devdata/ instead of data/, with its own
        counter (the preflight dialog's "test / dev data" choice - full
        session records are kept, one folder per run, just segregated from
        real participant data).
    Neither flag set (the default): real sessions, numbered under data/.
    """
    if test_mode:
        session_dir = REPO_ROOT / "data" / "test"
        session_dir.mkdir(parents=True, exist_ok=True)
        audio_dir = session_dir / "audio"
        audio_dir.mkdir(exist_ok=True)
        print(f"[TEST MODE] Reusing test session at: {session_dir.absolute()}")
        print(f"[TEST MODE] Audio directory at: {audio_dir.absolute()}")
        return str(session_dir.absolute()), str(audio_dir.absolute())

    base_dir = REPO_ROOT / ("devdata" if devdata else "data")
    return _numbered_session(base_dir, "DEVDATA" if devdata else "DATA")
