#!/bin/bash

# Ignore SIGHUP to prevent termination when the shell closes
trap '' HUP

# This script finds all .osil test files and runs them in parallel
# using the 'run_instance.py' script with continuous job submission.
# It assigns 2 cores per job, running up to 14 jobs simultaneously on 28 cores.

# Get the directory of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# The project root is two levels up from the script's location
PROJECT_ROOT=$(dirname "$(dirname "$SCRIPT_DIR")")

# Paths to the test instances and export directory
TEST_PATH="$PROJECT_ROOT/data/import/test_instances"
EXPORT_PATH="$PROJECT_ROOT/data/export"

# Create export directory if it doesn't exist
mkdir -p "$EXPORT_PATH"

# Create a timestamped results file
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
RESULTS_FILE="$EXPORT_PATH/results_${TIMESTAMP}.csv"
echo "osil_file_name,runtime,gap" > "$RESULTS_FILE"

# Find all .osil files and store their full paths in an array
mapfile -t files < <(find "$TEST_PATH" -name "*.osil")
NUM_FILES=${#files[@]}

# Define core configuration
NUM_CORES=28
CORES_PER_JOB=4
MAX_PARALLEL_JOBS=$((NUM_CORES / CORES_PER_JOB))

# Create core sets (14 slots of 2 contiguous cores each)
core_sets=()
for i in $(seq 0 $((MAX_PARALLEL_JOBS - 1))); do
    start=$((i * CORES_PER_JOB))
    end=$((start + CORES_PER_JOB - 1))
    core_sets+=("$start-$end")
done

echo "Found $NUM_FILES test instances."
echo "Running up to $MAX_PARALLEL_JOBS jobs in parallel, using $CORES_PER_JOB cores each."
echo "Results will be saved to $RESULTS_FILE"

# Create a named pipe for job control
PIPE=$(mktemp -u)
mkfifo "$PIPE"
exec 3<>"$PIPE"
rm -f "$PIPE"

# Initialize the semaphore with core set indices
for i in $(seq 0 $((MAX_PARALLEL_JOBS - 1))); do
    echo "$i" >&3
done

# Function to run a job
run_job() {
    local file="$1"
    local core_set="$2"
    local slot="$3"
    
    # Run the Python script with taskset
    PYTHONPATH="$PROJECT_ROOT/src" taskset -c "$core_set" \
    python3 "$SCRIPT_DIR/run_instance.py" "$file" >> "$RESULTS_FILE" 2>&1
    
    # Return the slot to the semaphore
    echo "$slot" >&3
}

# Counter for files processed
processed=0

# Start initial batch of jobs
for ((i = 0; i < NUM_FILES && i < MAX_PARALLEL_JOBS; i++)); do
    file="${files[$i]}"
    read -u 3 -r slot
    core_set="${core_sets[$slot]}"
    echo "Starting job for $(basename "$file") on cores $core_set (slot $slot)"
    run_job "$file" "$core_set" "$slot" &
    ((processed++))
done

# Start remaining jobs as slots become available
for ((i = processed; i < NUM_FILES; i++)); do
    file="${files[$i]}"
    # Wait for an available slot
    read -u 3 -r slot
    core_set="${core_sets[$slot]}"
    echo "Starting job for $(basename "$file") on cores $core_set (slot $slot)"
    run_job "$file" "$core_set" "$slot" &
    ((processed++))
done

# Wait for all background jobs to complete
wait

# Close file descriptor
exec 3>&-

echo "All optimization runs completed. Results saved to $RESULTS_FILE"