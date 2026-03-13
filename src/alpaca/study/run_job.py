# -*- coding: utf-8 -*-

"""
Single job runner for computational studies.

This module executes a single instance-config combination and outputs
the result as a CSV row to stdout. Designed to be called as a subprocess.

Usage:
    python -m alpaca.study.run_job --instance <path> --config <path> [OPTIONS]
"""
import argparse
import os
import sys
import traceback

import alpaca as alp


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Run a single instance-config combination."
    )
    parser.add_argument(
        "--instance",
        type=str,
        required=True,
        help="Path to the .osil instance file.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the .json configuration file.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of threads for the solver.",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Directory for log files.",
    )
    return parser.parse_args()


def extract_name(path: str) -> str:
    """Extracts the base name without extension from a path.

    Args:
        path: Full file path.

    Returns:
        Base name without extension.
    """
    return os.path.splitext(os.path.basename(path))[0]


def run_single_job(args: argparse.Namespace) -> int:
    """Runs a single instance-config combination.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    instance_name = extract_name(args.instance)
    config_name = extract_name(args.config)

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            sys.stdout = devnull
            sys.stderr = devnull

            alpaca = alp.read_model_from_osil(args.instance)

            if args.log_dir is not None:
                os.makedirs(args.log_dir, exist_ok=True)
                log_path = os.path.join(
                    args.log_dir, f"{instance_name}_{config_name}.log"
                )
                alpaca.configure_logging(log_path, level="INFO")

            alpaca.customize_settings(args.config)
            alpaca.user_settings.solver_thread_limit = min(8, args.threads)
            alpaca.build_pwl_relaxation_solver()
            alpaca.solve()

            csv_row = alpaca.statistics.result_row_print

        sys.stdout = original_stdout
        sys.stderr = original_stderr

        print(csv_row)
        return 0

    except Exception as exc:  # pylint: disable=broad-exception-caught
        sys.stdout = original_stdout
        sys.stderr = original_stderr

        error_msg = f"{instance_name} + {config_name}: {type(exc).__name__}: {exc}"
        print(error_msg, file=sys.stderr)
        traceback.print_exc(file=sys.stderr)

        return 1


def main() -> int:
    """Main entry point.

    Returns:
        Exit code.
    """
    args = parse_arguments()
    return run_single_job(args)


if __name__ == "__main__":
    sys.exit(main())
