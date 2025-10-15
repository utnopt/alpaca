#!/bin/bash

# Ignore SIGHUP to prevent termination when the shell closes
trap '' HUP

# This script finds all 'pooling' .osil test files,
# runs the stair_locatelli test case on them in parallel,
# and collects the results.

# --- Configuration ---

# Get the directory of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# The project root is four levels up from the script's location
PROJECT_ROOT=$(dirname "$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")")

# Paths to the test instances and export directory
IMPORT_PATH="$PROJECT_ROOT/data/import/instances"
EXPORT_PATH="$PROJECT_ROOT/data/export"

# Core configuration for parallel execution
# Each job is single-threaded
NUM_CORES=28
CORES_PER_JOB=4
MAX_PARALLEL_JOBS=$((NUM_CORES / CORES_PER_JOB))

# --- Setup ---

# Create export directory if it doesn't exist
mkdir -p "$EXPORT_PATH"

# Create a timestamped results file and write the header
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
RESULTS_FILE="$EXPORT_PATH/stair_locatelli_results_${TIMESTAMP}.csv"
echo "instance_name,obj_without,obj_locatelli,obj_stair_locatelli" > "$RESULTS_FILE"

# --- Job Definition ---

# Find all 'pooling' .osil files and prepare the list of jobs.
declare -a jobs
while IFS= read -r file; do
    jobs+=("$file")
done < <(find "$IMPORT_PATH" -name "*pooling*.osil")

NUM_JOBS=${#jobs[@]}
echo "Found $NUM_JOBS total jobs to run for the stair_locatelli test."
echo "Running up to $MAX_PARALLEL_JOBS jobs in parallel."
echo "Results will be saved to $RESULTS_FILE"

# --- Parallel Execution Engine ---

# Create core sets for taskset
core_sets=()
for i in $(seq 0 $((MAX_PARALLEL_JOBS - 1))); do
    start=$((i * CORES_PER_JOB))
    end=$((start + CORES_PER_JOB - 1))
    core_sets+=("$start-$end")
done

# Create a named pipe for job control (semaphore)
PIPE=$(mktemp -u)
mkfifo "$PIPE"
exec 3<>"$PIPE"
rm -f "$PIPE"

# Initialize the semaphore with available job slots
for i in $(seq 0 $((MAX_PARALLEL_JOBS - 1))); do
    echo "$i" >&3
done

# Function to run a single job
run_job() {
    local file="$1"
    local core_set="$2"
    local slot="$3"

    # Execute the Python script; its output is appended to the results file
    PYTHONPATH="$PROJECT_ROOT/src" taskset -c "$core_set" \
    python3 "$SCRIPT_DIR/run_test.py" \
        --file "$file" >> "$RESULTS_FILE" 2>&1

    # Return the slot to the semaphore, making it available for the next job
    echo "$slot" >&3
}

# --- Job Dispatching ---

# Process all jobs using the semaphore for parallel control
for job_file in "${jobs[@]}"; do
    # Wait for an available slot from the semaphore
    read -r -u 3 slot

    # Assign a core set to the job
    core_set="${core_sets[$slot]}"

    # Start the job in the background
    echo "Starting job for $(basename "$job_file") on cores $core_set (slot $slot)"
    run_job "$job_file" "$core_set" "$slot" &
done

# Wait for all background jobs to complete
wait

# Close the file descriptor for the pipe
exec 3>&-

echo "All optimization runs completed. Results saved to $RESULTS_FILE"
