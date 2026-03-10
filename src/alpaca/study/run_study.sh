#!/bin/bash

# Study runner script for the Alpaca optimization framework.

# This script sets up the environment and runs the study pipeline.

# Ignore SIGHUP to prevent termination when the shell closes

trap '' HUP

# Get the directory of the script

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Project root is two levels up from data/study/

PROJECT_ROOT=$(dirname "$(dirname "$SCRIPT_DIR")")

# Add src to PYTHONPATH

export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"

# Run the study module with all passed arguments

python3 -m alpaca.study "$@"
