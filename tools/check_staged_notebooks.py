"""Pre-commit hook: refuse to commit a notebook that still has outputs, or is large.

Notebook outputs in this project hold transcripts and embedded participant audio,
so they must never reach git. The nbstripout filter (.gitattributes) strips them
on `git add`; this checks the version that is actually staged, which catches a
clone where the filter was never installed. The working-tree notebook is not
looked at, so local outputs are fine.

Usage (from .pre-commit-config.yaml): python tools/check_staged_notebooks.py NB.ipynb ...
"""

import json
import subprocess
import sys

MAX_KB = 500


def main(paths):
    bad = []
    for path in paths:
        staged = subprocess.run(["git", "show", f":{path}"], capture_output=True).stdout
        if not staged:  # deleted, or not staged
            continue
        problems = []
        if len(staged) > MAX_KB * 1024:
            problems.append(f"{len(staged) // 1024} KB staged (limit {MAX_KB} KB)")
        try:
            cells = json.loads(staged).get("cells", [])
        except json.JSONDecodeError:
            problems.append("staged version is not valid notebook JSON")
            cells = []
        n_out = sum(bool(c.get("outputs")) for c in cells)
        if n_out:
            problems.append(f"{n_out} cells with outputs")
        if problems:
            bad.append(f"  {path}: {'; '.join(problems)}")

    if bad:
        print("Notebooks must be committed without outputs (they contain participant data):")
        print("\n".join(bad))
        print("\nIf nbstripout is not set up in this clone, run once:\n"
              "    pip install nbstripout && nbstripout --install\n"
              "then `git add` the notebooks again.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
