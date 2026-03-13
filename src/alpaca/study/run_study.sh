#!/bin/bash

# -*- coding: utf-8 -*-

# Shell-based coordinator for computational studies.

# Submits one job per instance+config combination for visibility and control.

trap '' HUP

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Default values

INSTANCES_DIR=""
CONFIGS_DIR=""
RESULTS_DIR=""
LOG_DIR=""
CORES_PER_JOB=4
MAX_PARALLEL_JOBS=""
SKIP_EVALUATE=0

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -i, --instances DIR    Directory containing .osil instance files"
    echo "  -c, --configs DIR      Directory containing .json configuration files"
    echo "  -r, --results DIR      Directory for output files"
    echo "  -l, --logs DIR         Directory for log files (optional)"
    echo "  -w, --workers N        Maximum parallel jobs (default: auto)"
    echo "  -t, --threads N        Threads per job for CPU affinity (default: 4)"
    echo "  --no-evaluate          Skip evaluation after run"
    echo "  -h, --help             Show this help message"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--instances) INSTANCES_DIR="$2"; shift 2 ;;
        -c|--configs) CONFIGS_DIR="$2"; shift 2 ;;
        -r|--results) RESULTS_DIR="$2"; shift 2 ;;
        -l|--logs) LOG_DIR="$2"; shift 2 ;;
        -w|--workers) MAX_PARALLEL_JOBS="$2"; shift 2 ;;
        -t|--threads) CORES_PER_JOB="$2"; shift 2 ;;
        --no-evaluate) SKIP_EVALUATE=1; shift ;;
        -h|--help) print_usage; exit 0 ;;
        *) echo "Unknown option: $1"; print_usage; exit 1 ;;
    esac
done

if [[ -z "$INSTANCES_DIR" ]] || [[ -z "$CONFIGS_DIR" ]] || [[ -z "$RESULTS_DIR" ]]; then
    echo "Error: --instances, --configs, and --results are required."
    print_usage
    exit 1
fi

if [[ ! -d "$INSTANCES_DIR" ]]; then
    echo "Error: Instances directory does not exist: $INSTANCES_DIR"
    exit 1
fi

if [[ ! -d "$CONFIGS_DIR" ]]; then
    echo "Error: Configs directory does not exist: $CONFIGS_DIR"
    exit 1
fi

NUM_CORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
if [[ -z "$MAX_PARALLEL_JOBS" ]]; then
    MAX_PARALLEL_JOBS=$((NUM_CORES / CORES_PER_JOB))
    if [[ $MAX_PARALLEL_JOBS -lt 1 ]]; then
        MAX_PARALLEL_JOBS=1
    fi
fi

mkdir -p "$RESULTS_DIR/raw"
if [[ -n "$LOG_DIR" ]]; then
    mkdir -p "$LOG_DIR"
fi

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
RESULTS_FILE="$RESULTS_DIR/raw/study_results_${TIMESTAMP}.csv"
FAILED_FILE="$RESULTS_DIR/raw/.failed_instances_${TIMESTAMP}"
ERROR_LOG="$RESULTS_DIR/raw/errors_${TIMESTAMP}.log"

touch "$FAILED_FILE"
touch "$ERROR_LOG"

# Write CSV header using Python to get correct column names

python3 -c "
from alpaca.utils.localized_string_factory import LocalizedStringFactory as lsf
print(lsf.stats_column_names())
" > "$RESULTS_FILE"

mapfile -t INSTANCES < <(find "$INSTANCES_DIR" -name "*.osil" | sort)
mapfile -t CONFIGS < <(find "$CONFIGS_DIR" -name "*.json" | sort)

NUM_INSTANCES=${#INSTANCES[@]}
NUM_CONFIGS=${#CONFIGS[@]}
TOTAL_JOBS=$((NUM_INSTANCES * NUM_CONFIGS))

echo "========================================"
echo "Study Configuration"
echo "========================================"
echo "  Instances directory: $INSTANCES_DIR"
echo "  Configs directory:   $CONFIGS_DIR"
echo "  Results directory:   $RESULTS_DIR"
echo "  Log directory:       ${LOG_DIR:-disabled}"
echo "  Instances found:     $NUM_INSTANCES"
echo "  Configs found:       $NUM_CONFIGS"
echo "  Total combinations:  $TOTAL_JOBS"
echo "  Parallel jobs:       $MAX_PARALLEL_JOBS"
echo "  Threads per job:     $CORES_PER_JOB"
echo "  Results file:        $RESULTS_FILE"
echo "========================================"

if [[ $TOTAL_JOBS -eq 0 ]]; then
    echo "No jobs to run. Check your instance and config directories."
    exit 1
fi

declare -a JOBS
for instance in "${INSTANCES[@]}"; do
    for config in "${CONFIGS[@]}"; do
        JOBS+=("$instance|$config")
    done
done

core_sets=()
for i in $(seq 0 $((MAX_PARALLEL_JOBS - 1))); do
    start=$((i * CORES_PER_JOB))
    end=$((start + CORES_PER_JOB - 1))
    if [[ $end -ge $NUM_CORES ]]; then
        end=$((NUM_CORES - 1))
    fi
    core_sets+=("$start-$end")
done

PIPE=$(mktemp -u)
mkfifo "$PIPE"
exec 3<>"$PIPE"
rm -f "$PIPE"

for i in $(seq 0 $((MAX_PARALLEL_JOBS - 1))); do
    echo "$i" >&3
done

COMPLETED=0
SUCCESSFUL=0
FAILED=0
SKIPPED=0

run_job() {
    local instance_path="$1"
    local config_path="$2"
    local core_set="$3"
    local slot="$4"
    local job_num="$5"
    local total="$6"

    local instance_name
    instance_name=$(basename "$instance_path" .osil)
    local config_name
    config_name=$(basename "$config_path" .json)

    if grep -q "^${instance_name}$" "$FAILED_FILE" 2>/dev/null; then
        echo "[$job_num/$total] SKIPPED: $instance_name + $config_name (prior failure)"
        echo "$slot" >&3
        return 2
    fi

    echo "[$job_num/$total] RUNNING: $instance_name + $config_name (cores $core_set)"

    local log_arg=""
    if [[ -n "$LOG_DIR" ]]; then
        log_arg="--log-dir $LOG_DIR"
    fi

    local result
    result=$(taskset -c "$core_set" python3 -m alpaca.study.run_job \
        --instance "$instance_path" \
        --config "$config_path" \
        --threads "$CORES_PER_JOB" \
        --failed-file "$FAILED_FILE" \
        $log_arg 2>> "$ERROR_LOG")

    local exit_code=$?

    if [[ $exit_code -eq 0 ]] && [[ -n "$result" ]]; then
        echo "$result" >> "$RESULTS_FILE"
        echo "[$job_num/$total] SUCCESS: $instance_name + $config_name"
        echo "$slot" >&3
        return 0
    else
        echo "$instance_name" >> "$FAILED_FILE"
        echo "[$job_num/$total] FAILED: $instance_name + $config_name"
        echo "$slot" >&3
        return 1
    fi
}

job_num=0
for job in "${JOBS[@]}"; do
    IFS='|' read -r instance config <<< "$job"

    read -r -u 3 slot
    core_set="${core_sets[$slot]}"

    ((job_num++))

    run_job "$instance" "$config" "$core_set" "$slot" "$job_num" "$TOTAL_JOBS" &
done

wait

exec 3>&-

SUCCESSFUL=$(wc -l < "$RESULTS_FILE")
SUCCESSFUL=$((SUCCESSFUL - 1))

FAILED_COUNT=0
if [[ -f "$FAILED_FILE" ]]; then
    FAILED_COUNT=$(sort -u "$FAILED_FILE" | wc -l)
fi

echo ""
echo "========================================"
echo "Study Completed"
echo "========================================"
echo "  Total combinations:  $TOTAL_JOBS"
echo "  Successful runs:     $SUCCESSFUL"
echo "  Failed instances:    $FAILED_COUNT"
echo "  Results file:        $RESULTS_FILE"
echo "  Error log:           $ERROR_LOG"
echo "========================================"

if [[ -f "$FAILED_FILE" ]] && [[ -s "$FAILED_FILE" ]]; then
    echo ""
    echo "Failed instances:"
    sort -u "$FAILED_FILE" | while read -r name; do
        echo "  - $name"
    done
fi

rm -f "$FAILED_FILE"

if [[ $SKIP_EVALUATE -eq 0 ]] && [[ $SUCCESSFUL -gt 0 ]]; then
    echo ""
    echo "Running evaluation..."
    python3 -m alpaca.study evaluate --csv "$RESULTS_FILE" --results "$RESULTS_DIR"
fi

if [[ $FAILED_COUNT -gt 0 ]]; then
    exit 1
fi
exit 0