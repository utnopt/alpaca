#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""
Single job runner for study coordination.

Runs one instance+config combination. Designed to be spawned as a subprocess
so each job is visible in `ps aux | grep run_single`.

Usage:
    python -m alpaca.study.run_single \
        --instance path/to/instance.osil \
        --config path/to/config.json \
        --failed-file path/to/failed.txt \
        --log-dir path/to/logs \
        --threads 4

Exit codes:
    0 - Success (CSV row printed to stdout)
    1 - Error (instance marked as failed)
    2 - Skipped (instance previously failed)
"""

import argparse
import os
import sys
import traceback

import alpaca as alp


def extract_name_from_path(path: str) -> str:
    """Extracts the base name from a file path without extension."""
    return os.path.splitext(os.path.basename(path))[0]


def is_instance_failed(failed_file: str, instance_name: str) -> bool:
    """Checks if an instance is marked as failed."""
    if not os.path.exists(failed_file):
        return False
    try:
        with open(failed_file, "r", encoding="utf-8") as f:
            failed_instances = {line.strip() for line in f if line.strip()}
        return instance_name in failed_instances
    except OSError:
        return False


def mark_instance_failed(failed_file: str, instance_name: str) -> None:
    """Marks an instance as failed by appending to the failed file."""
    try:
        with open(failed_file, "a", encoding="utf-8") as f:
            f.write(instance_name + "\n")
    except OSError as e:
        print(f"Warning: Could not write to failed file: {e}", file=sys.stderr)


def run_single(
    instance_path: str,
    config_path: str,
    log_dir: str | None,
    threads: int,
) -> str:
    """Runs a single instance+config and returns the CSV row."""
    instance_name = extract_name_from_path(instance_path)
    config_name = extract_name_from_path(config_path)

    # Suppress stdout during execution
    original_stdout_fd = os.dup(sys.stdout.fileno())
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, sys.stdout.fileno())

    try:
        alpaca_model = alp.read_model_from_osil(instance_path)

        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, f"{instance_name}_{config_name}.log")
            alpaca_model.configure_logging(log_path, level="INFO")

        alpaca_model.customize_settings(config_path)
        alpaca_model.user_settings.solver_thread_limit = min(8, threads)
        alpaca_model.build_pwl_relaxation_solver()
        alpaca_model.solve()

        return alpaca_model.statistics.result_row_print
    finally:
        os.dup2(original_stdout_fd, sys.stdout.fileno())
        os.close(original_stdout_fd)
        os.close(devnull_fd)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run a single instance+config.")
    parser.add_argument("--instance", "-i", required=True, help="Path to .osil file.")
    parser.add_argument("--config", "-c", required=True, help="Path to .json config.")
    parser.add_argument(
        "--failed-file", "-f", required=True, help="Failed instances file."
    )
    parser.add_argument(
        "--log-dir", "-l", default=None, help="Log directory or 'none'."
    )
    parser.add_argument(
        "--threads", "-t", type=int, default=4, help="Threads for solver."
    )

    args = parser.parse_args()

    log_dir = (
        None if args.log_dir is None or args.log_dir.lower() == "none" else args.log_dir
    )
    instance_name = extract_name_from_path(args.instance)
    config_name = extract_name_from_path(args.config)

    # Check if instance already failed
    if is_instance_failed(args.failed_file, instance_name):
        print(f"SKIPPED: {instance_name} + {config_name}", file=sys.stderr)
        return 2

    try:
        csv_row = run_single(args.instance, args.config, log_dir, args.threads)
        print(csv_row)  # stdout -> captured and appended to CSV
        print(f"OK: {instance_name} + {config_name}", file=sys.stderr)
        return 0

    except Exception as exc:  # pylint: disable=broad-exception-caught
        print(f"ERROR: {instance_name} + {config_name}: {exc}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        mark_instance_failed(args.failed_file, instance_name)
        return 1


if __name__ == "__main__":
    sys.exit(main())
