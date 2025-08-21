#!/bin/bash

# Ignore SIGHUP to prevent termination when the shell closes
trap '' HUP

# This script finds all .osil test files across multiple directories,
# runs them with different settings in parallel, and collects the results.

# --- Configuration ---

# Get the directory of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# The project root is two levels up from the script's location
PROJECT_ROOT=$(dirname "$(dirname "$SCRIPT_DIR")")

# Paths to the test instances and export directory
IMPORT_PATH="$PROJECT_ROOT/data/import"
EXPORT_PATH="$PROJECT_ROOT/data/export"

# Core configuration for parallel execution
NUM_CORES=28
CORES_PER_JOB=4
MAX_PARALLEL_JOBS=$((NUM_CORES / CORES_PER_JOB))

# --- Setup ---

# Create export directory if it doesn't exist
mkdir -p "$EXPORT_PATH"

# Create a timestamped results file and write the header
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
RESULTS_FILE="$EXPORT_PATH/results_${TIMESTAMP}.csv"
echo "number_of_breakpoints,test_case,osil_file_name,runtime,gap,nodes,cuts" > "$RESULTS_FILE"

# --- Job Definition ---

# Find all test instance directories (e.g., test_instances_5, test_instances_10)
# and prepare the list of jobs to be executed.
declare -a jobs
for dir in "$IMPORT_PATH"/test_instances_*; do
    # Extract the number of breakpoints from the directory name
    num_breakpoints=$(basename "$dir" | grep -o '[0-9]*$')
    if [ -z "$num_breakpoints" ]; then
        echo "Warning: Could not determine number of breakpoints from directory name: $dir"
        continue
    fi

    # Find all .osil files in the directory
    while IFS= read -r file; do
        # For each file, create two jobs: one with MPIP on, one with it off.
        jobs+=("$file $num_breakpoints MPIP 1")
        jobs+=("$file $num_breakpoints Standard 0")
    done < <(find "$dir" -name "*.osil")
done

NUM_JOBS=${#jobs[@]}
echo "Found $NUM_JOBS total jobs to run across all test configurations."
echo "Running up to $MAX_PARALLEL_JOBS jobs in parallel, using $CORES_PER_JOB cores each."
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
    local breakpoints="$2"
    local test_case="$3"
    local stripe_flag="$4"
    local core_set="$5"
    local slot="$6"

    # Execute the Python script with all required arguments
    # The output is directly appended to the results file
    PYTHONPATH="$PROJECT_ROOT/src" taskset -c "$core_set" \
    python3 "$SCRIPT_DIR/run_instance.py" \
        --file "$file" \
        --breakpoints "$breakpoints" \
        --test_case "$test_case" \
        --mpip_stripe "$stripe_flag" >> "$RESULTS_FILE" 2>&1

    # Return the slot to the semaphore, making it available for the next job
    echo "$slot" >&3
}

# --- Job Dispatching ---

# Process all jobs using the semaphore for parallel control
for job_params in "${jobs[@]}"; do
    # Wait for an available slot from the semaphore
    read -r -u 3 slot

    # Assign a core set to the job
    core_set="${core_sets[$slot]}"

    # Start the job in the background
    # The job_params are split into individual arguments for run_job
    echo "Starting job for $(basename $job_params) on cores $core_set (slot $slot)"
    run_job $job_params "$core_set" "$slot" &
done

# Wait for all background jobs to complete
wait

# Close the file descriptor for the pipe
exec 3>&-

echo "All optimization runs completed. Results saved to $RESULTS_FILE"
