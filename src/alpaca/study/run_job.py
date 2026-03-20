# -*- coding: utf-8 -*-
"""
@authors: kuen,
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
        "--configs",
        type=str,
        required=True,
        help="Paths to the .json configuration files.",
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

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            sys.stdout = devnull
            sys.stderr = devnull

            csv_rows = []
            obbt_variable_bounds = {}
            locatelli_vertices = {}
            for config_path in args.configs.split(","):
                config_name = extract_name(config_path)

                alpaca = alp.read_model_from_osil(args.instance)
                alpaca.obbt_variable_bounds = obbt_variable_bounds
                alpaca.locatelli_vertices = locatelli_vertices

                if args.log_dir is not None:
                    os.makedirs(args.log_dir, exist_ok=True)
                    log_path = os.path.join(
                        args.log_dir, f"{instance_name}_{config_name}.log"
                    )
                    alpaca.configure_logging(log_path, level="INFO")

                alpaca.customize_settings(config_path)
                alpaca.user_settings.solver_thread_limit = min(8, args.threads)
                alpaca.build_pwl_relaxation_solver()
                alpaca.solve()

                obbt_variable_bounds = alpaca.obbt_variable_bounds
                locatelli_vertices = alpaca.locatelli_vertices

                csv_rows.append(alpaca.statistics.result_row_print)

        sys.stdout = original_stdout
        sys.stderr = original_stderr

        for csv_row in csv_rows:
            print(csv_row)
        return 0

    except Exception as exc:  # pylint: disable=broad-exception-caught
        sys.stdout = original_stdout
        sys.stderr = original_stderr

        error_msg = f"{instance_name}: {type(exc).__name__}: {exc}"
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
