#!/bin/bash

# This script finds all .osil test files and runs them in parallel
# using the 'run_instance.py' script. It assigns 2 cores per job,
# running up to 14 jobs simultaneously on 28 cores.

# Get the directory of the script and set up paths based on your settings.py
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT=$(dirname $(dirname "$SCRIPT_DIR"))

# Paths to the test instances and export directory, derived from settings.py
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

# Loop through the files, running jobs in batches
for ((i=0; i<NUM_FILES; i+=MAX_PARALLEL_JOBS)); do
    echo "Starting batch $((i / MAX_PARALLEL_JOBS + 1))..."
    for ((j=0; j<MAX_PARALLEL_JOBS && (i+j)<NUM_FILES; j++)); do
        FILE="${files[$((i+j))]}"

        # Calculate the core range for this job
        CORE_START=$((j * CORES_PER_JOB))
        CORE_END=$((CORE_START + CORES_PER_JOB - 1))
        CORES="$CORE_START-$CORE_END"

        echo "  - Submitting job for $FILE on cores $CORES"

        # Run the Python script in the background with taskset
        taskset -c "$CORES" python3 "$SCRIPT_DIR/run_instance.py" "$FILE" >> "$RESULTS_FILE" &
    done

    # Wait for all background jobs in the current batch to complete
    wait
    echo "Batch completed. Moving to the next one."
done

echo "All optimization runs completed. Results saved to $RESULTS_FILE"
