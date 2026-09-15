#!/bin/bash
# Launches the latest dev version of the conversation-recorder experiment.
# This is the script the Desktop icon (Run Experiment.command) calls into.
# Keeping the real logic here (in the repo) means it updates itself along
# with everything else on `git pull` - the Desktop icon never needs to change.

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="/Users/experimenter/miniconda3/envs/artsinteg/bin/python"

fail() {
    echo ""
    echo "ERROR: $1"
    echo "Please contact the researcher instead of trying to fix this yourself."
    read -r -p "Press Enter to close this window..."
    exit 1
}

cd "$REPO_DIR" || fail "Could not find the project folder at $REPO_DIR"

echo "Checking for the latest version..."
git fetch origin dev || fail "Could not reach GitHub to check for updates (check your internet connection)."
git checkout dev || fail "Could not switch to the dev branch."

# --ff-only refuses to do anything if local changes would conflict with the
# update, rather than silently overwriting or merging - safe to run even
# while the researcher has work in progress in this same folder.
git merge --ff-only origin/dev || fail "Could not update to the latest version - there may be unsaved local changes in this folder."

if [ ! -x "$PYTHON_BIN" ]; then
    fail "Expected Python environment not found at $PYTHON_BIN"
fi

echo "Starting the experiment..."
cd "$REPO_DIR/convo_recorder" || fail "Could not find the convo_recorder folder."
"$PYTHON_BIN" run.py

echo ""
echo "Experiment session ended."
read -r -p "Press Enter to close this window..."
