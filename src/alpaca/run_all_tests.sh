#!/bin/bash

# This script finds all .osil test files and runs them in parallel
# using the 'run_instance.py' script. It is designed to be run from
# the 'alpaca/src/alpaca' directory. It assigns 2 cores per job,
# running up to 14 jobs simultaneously on 28 cores.

# Get the directory of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# The project root is two levels up from the script's location
# SCRIPT_DIR is .../alpaca/src/alpaca
# dirname "$SCRIPT_DIR" is .../alpaca/src
# dirname $(dirname "$SCRIPT_DIR") is .../alpaca
PROJECT_ROOT=$(dirname $(dirname "$SCRIPT_DIR"))

# Paths to the test instances and export directory, derived from the project root
TEST_PATH="$PROJECT_ROOT/data/import/test_instances"
EXPORT_PATH="$PROJECT_ROOT/data/export"

# Create a timestamped results file
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
RESULTS_FILE="$EXPORT_PATH/results_${TIMESTAMP}.csv"
echo "osil_file_name,runtime" > "$RESULTS_FILE"

# Find all .osil files and store their full paths in an array
mapfile -t files < <(find "$TEST_PATH" -name "*.osil")
NUM_FILES=${#files[@]}

# Define core configuration
NUM_CORES=28
CORES_PER_JOB=2
MAX_PARALLEL_JOBS=$((NUM_CORES / CORES_PER_JOB))

echo "Found $NUM_FILES test instances."
echo "Running up to $MAX_PARALLEL_JOBS jobs in parallel, using $CORES_PER_JOB cores each."
echo "Results will be saved to $RESULTS_FILE"

# Loop through the files, running jobs in batches
for ((i=0; i<NUM_FILES; i+=MAX_PARALLEL_JOBS)); do
    echo "Starting batch $((i / MAX_PARALLEL_JOBS + 1))..."
    for ((j=0; j<MAX_PARALLEL_JOBS && (i+j)<NUM_FILES; j++)); do
        FILE="${files[$((i+j))]}"
        
        # Calculate the core range for this job
        CORE_START=$((j * CORES_PER_JOB))
        CORE_END=$((CORE_START + CORES_PER_JOB - 1))
        CORES="$CORE_START-$CORE_END"
        
        echo "  - Submitting job for $(basename "$FILE") on cores $CORES"
        
        # Run the Python script in the background with taskset
        # Set PYTHONPATH to include the project's src directory for the import to work
        PYTHONPATH="$PROJECT_ROOT/src" taskset -c "$CORES" python3 "$SCRIPT_DIR/run_instance.py" "$FILE" >> "$RESULTS_FILE" &
    done

    # Wait for all background jobs in the current batch to complete
    wait
    echo "Batch completed. Moving to the next one."
done

echo "All optimization runs completed. Results saved to $RESULTS_FILE"
