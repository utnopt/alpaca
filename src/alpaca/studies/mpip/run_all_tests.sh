#!/bin/bash

# Ignore SIGHUP to prevent termination when the shell closes
trap '' HUP

# This script finds all .osil test files across multiple directories,
# runs them with different settings and random seeds in parallel,
# and collects the results.

# --- Configuration ---

# Get the directory of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# The project root is four levels up from the script's location
PROJECT_ROOT=$(dirname "$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")")

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
RESULTS_FILE="$EXPORT_PATH/mpip_results_${TIMESTAMP}.csv"
# Added 'seed_value' to the CSV header
echo "test_case,pwl_method,nr_of_breakpoints,seed,osil_file_name,runtime,mip_gap,mpip_cuts,mpip_cuts_applied,mpip_ratio" > "$RESULTS_FILE"

# --- Job Definition ---

# Find all test instance directories (e.g., test_instances_5, test_instances_10)
# and prepare the list of jobs to be executed.
declare -a jobs
while IFS= read -r file; do
    # For each file and each test case, create 5 jobs with different seeds.
    for num_breakpoints in 5 10 15 20; do
      for seed in $(seq 42 47); do
        for pwl_method in "multiple_choice" "delta"; do
          # Each job is defined by: file, number_of_breakpoints, test_case, mpip_stripe_flag, seed_value
          # Added seed_value to the job parameters
          jobs+=("$file $num_breakpoints MPIP $seed $pwl_method")
          jobs+=("$file $num_breakpoints Standard $seed $pwl_method")
        done
      done
    done
done < <(find "$IMPORT_PATH"/instances -name "*.osil")

NUM_JOBS=${#jobs[@]}
echo "Found $NUM_JOBS total jobs to run across all test configurations and seeds."
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
    local seed_value="$4" # Added seed_value parameter
    local pwl_method="$5"
    local core_set="$6"
    local slot="$7"

    # Execute the Python script with all required arguments, including the seed value
    # The output is directly appended to the results file
    PYTHONPATH="$PROJECT_ROOT/src" taskset -c "$core_set" \
    python3 "$SCRIPT_DIR/run_test.py" \
        --file "$file" \
        --breakpoints "$breakpoints" \
        --test_case "$test_case" \
        --pwl_method "$pwl_method" \
        --seed_value "$seed_value" >> "$RESULTS_FILE" 2>/dev/null

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