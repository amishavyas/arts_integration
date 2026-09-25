"""Where the participant data lives. It is never in the repo (see .gitignore).

DATA_ROOT holds data/ (one folder per session), permissions.csv and the aggregated
CSVs. Set ARTS_DATA_ROOT to point at it; by default it is the folder the repo is
cloned into, which is how the lab server is laid out:

    projects/arts_integration/          <- DATA_ROOT
        data/  permissions.csv  aggregated_*.csv
        arts_integration/               <- this repo
"""

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("ARTS_DATA_ROOT", REPO.parent)).expanduser().resolve()
DATA_DIR = DATA_ROOT / "data"
