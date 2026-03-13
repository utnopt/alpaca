#!/bin/bash

# Study runner script with shell-based job coordination.

#

# Features:

#   - One visible subprocess per instance+config (check with: ps aux | grep run_single)

#   - CPU affinity via taskset

#   - Immediate CSV writes on job completion

#   - Skip remaining configs when an instance fails

#

# Usage:

#   ./run_study.sh [OPTIONS]

#

# Options:

#   -i, --instances DIR     Directory with .osil files

#   -c, --configs DIR       Directory with .json config files

#   -r, --results DIR       Output directory for results

#   -l, --logs DIR          Log directory (use 'none' to disable)

#   -w, --workers N         Max parallel jobs

#   -t, --threads N         Threads per job (for CPU affinity)

#

# To run in background and detach:

#   nohup ./run_study.sh &> run_study.out &

# Ignore SIGHUP to survive shell closure

trap '' HUP

set -o pipefail

# --- Get script location ---

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT=$(dirname "$(dirname "$SCRIPT_DIR")")

# --- Default configuration ---

INSTANCES_DIR="$SCRIPT_DIR/test_instances"
CONFIGS_DIR="$SCRIPT_DIR/test_configs"
RESULTS_DIR="$SCRIPT_DIR/results"
LOG_DIR="$SCRIPT_DIR/results/logs"

NUM_CORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
THREADS_PER_JOB=4
MAX_PARALLEL_JOBS=""

# --- Parse command line arguments ---

while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--instances)
            INSTANCES_DIR="$2"
            shift 2
            ;;
        -c|--configs)
            CONFIGS_DIR="$2"
            shift 2
            ;;
        -r|--results)
            RESULTS_DIR="$2"
            shift 2
            ;;
        -l|--logs)
            LOG_DIR="$2"
            shift 2
            ;;
        -w|--workers)
            MAX_PARALLEL_JOBS="$2"
            shift 2
            ;;
        -t|--threads)
            THREADS_PER_JOB="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -i, --instances DIR   Directory with .osil files"
            echo "  -c, --configs DIR     Directory with .json config files"
            echo "  -r, --results DIR     Output directory"
            echo "  -l, --logs DIR        Log directory ('none' to disable)"
            echo "  -w, --workers N       Max parallel jobs"
            echo "  -t, --threads N       Threads per job"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information."
            exit 1
            ;;
    esac
done

# Calculate max workers if not specified

if [[ -z "$MAX_PARALLEL_JOBS" ]]; then
    MAX_PARALLEL_JOBS=$((NUM_CORES / THREADS_PER_JOB))
    if [[ $MAX_PARALLEL_JOBS -lt 1 ]]; then
        MAX_PARALLEL_JOBS=1
    fi
fi

# --- Setup directories ---

RAW_DIR="$RESULTS_DIR/raw"
mkdir -p "$RAW_DIR"

if [[ "$LOG_DIR" != "none" ]]; then
    mkdir -p "$LOG_DIR"
fi

# --- Create output files with timestamp ---

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
RESULTS_CSV="$RAW_DIR/study_results_${TIMESTAMP}.csv"
FAILED_FILE="$RAW_DIR/failed_instances_${TIMESTAMP}.txt"
ERROR_LOG="$RAW_DIR/errors_${TIMESTAMP}.log"

# Setup Python path

export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"

# --- Write CSV header ---

python3 -c "from alpaca.utils.localized_string_factory import LocalizedStringFactory as lsf; print(lsf.stats_column_names())" > "$RESULTS_CSV"

if [[ $? -ne 0 ]]; then
    echo "ERROR: Failed to write CSV header. Check PYTHONPATH and alpaca installation."
    exit 1
fi

touch "$FAILED_FILE"
touch "$ERROR_LOG"

# --- Discover instances and configs ---

mapfile -t INSTANCES < <(find "$INSTANCES_DIR" -maxdepth 1 -name "*.osil" -type f | sort)
mapfile -t CONFIGS < <(find "$CONFIGS_DIR" -maxdepth 1 -name "*.json" -type f | sort)

NUM_INSTANCES=${#INSTANCES[@]}
NUM_CONFIGS=${#CONFIGS[@]}
NUM_JOBS=$((NUM_INSTANCES * NUM_CONFIGS))

# --- Print configuration ---

echo "========================================================"
echo "Study Configuration"
echo "========================================================"
echo "  Instances:         $INSTANCES_DIR ($NUM_INSTANCES files)"
echo "  Configs:           $CONFIGS_DIR ($NUM_CONFIGS files)"
echo "  Results:           $RESULTS_DIR"
echo "  Logs:              $LOG_DIR"
echo "  Total jobs:        $NUM_JOBS"
echo "  Parallel workers:  $MAX_PARALLEL_JOBS"
echo "  Threads per job:   $THREADS_PER_JOB"
echo "  CPU cores:         $NUM_CORES"
echo "========================================================"
echo "  Results CSV:       $RESULTS_CSV"
echo "  Failed instances:  $FAILED_FILE"
echo "  Error log:         $ERROR_LOG"
echo "========================================================"

if [[ $NUM_JOBS -eq 0 ]]; then
    echo "ERROR: No jobs to run. Check instances and configs directories."
    exit 1
fi

# --- Create CPU core sets for taskset ---

declare -a CORE_SETS
for (( i=0; i<MAX_PARALLEL_JOBS; i++ )); do
    start=$((i * THREADS_PER_JOB))
    end=$((start + THREADS_PER_JOB - 1))
    # Wrap around if we exceed available cores
    if [[ $end -ge $NUM_CORES ]]; then
        end=$((NUM_CORES - 1))
    fi
    if [[ $start -ge $NUM_CORES ]]; then
        start=$((i % NUM_CORES))
        end=$(( (start + THREADS_PER_JOB - 1) % NUM_CORES ))
        if [[ $end -lt $start ]]; then
            end=$((NUM_CORES - 1))
        fi
    fi
    CORE_SETS+=("$start-$end")
done

# --- Semaphore for parallel job control ---

PIPE=$(mktemp -u)
mkfifo "$PIPE"
exec 3<>"$PIPE"
rm -f "$PIPE"

# Initialize semaphore with available slots

for (( i=0; i<MAX_PARALLEL_JOBS; i++ )); do
    echo "$i" >&3
done

# --- Job execution function ---

run_job() {
    local instance="$1"
    local config="$2"
    local core_set="$3"
    local slot="$4"

    local log_arg="$LOG_DIR"
    if [[ "$LOG_DIR" == "none" ]]; then
        log_arg="none"
    fi

    # Run with CPU affinity
    taskset -c "$core_set" python3 -m alpaca.study.run_single \
        --instance "$instance" \
        --config "$config" \
        --failed-file "$FAILED_FILE" \
        --log-dir "$log_arg" \
        --threads "$THREADS_PER_JOB" \
        >> "$RESULTS_CSV" 2>> "$ERROR_LOG"

    # Return slot to semaphore
    echo "$slot" >&3
}

# --- Dispatch all jobs ---

echo ""
echo "Starting job execution..."
echo ""

job_count=0
for instance in "${INSTANCES[@]}"; do
    for config in "${CONFIGS[@]}"; do
        # Wait for an available slot
        read -r -u 3 slot

        core_set="${CORE_SETS[$slot]}"
        instance_name=$(basename "${instance%.*}")
        config_name=$(basename "${config%.*}")

        job_count=$((job_count + 1))
        echo "[$job_count/$NUM_JOBS] Starting: $instance_name + $config_name (cores $core_set, slot $slot)"

        # Launch job in background
        run_job "$instance" "$config" "$core_set" "$slot" &
    done
done

# Wait for all background jobs to complete

echo ""
echo "All jobs dispatched. Waiting for completion..."
wait

# Close semaphore file descriptor

exec 3>&-

# --- Summary ---

echo ""
echo "========================================================"
echo "Study completed!"
echo "========================================================"

# Count results

if [[ -f "$RESULTS_CSV" ]]; then
    result_count=$(($(wc -l < "$RESULTS_CSV") - 1))
    echo "  Successful runs:   $result_count"
fi

if [[ -f "$FAILED_FILE" ]]; then
    failed_count=$(wc -l < "$FAILED_FILE" | tr -d ' ')
    if [[ $failed_count -gt 0 ]]; then
        echo "  Failed instances:  $failed_count"
        echo "  Failed names:      $(cat "$FAILED_FILE" | tr '\n' ' ')"
    fi
fi

echo ""
echo "  Results CSV:       $RESULTS_CSV"
echo "  Error log:         $ERROR_LOG"
echo "========================================================"
