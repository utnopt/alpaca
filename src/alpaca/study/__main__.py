# -*- coding: utf-8 -*-

"""
Command-line interface for running computational studies.

This module provides a CLI entry point for the study pipeline, including
execution, evaluation, and LaTeX generation.

Usage:
    python -m alpaca.study run [OPTIONS]
    python -m alpaca.study evaluate --csv <path> [OPTIONS]
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

from alpaca.study.evaluator import StudyEvaluator
from alpaca.study.latex_generator import LaTeXGenerator, LaTeXConfig
from alpaca.utils.logger import logger
from alpaca.utils import inout as ut_io


def get_default_paths() -> dict[str, str]:
    """Returns default paths relative to the current working directory.

    Returns:
        Dictionary with default paths for instances, configs, results, and logs.
    """
    cwd = Path.cwd()
    data_study_dir = cwd / "data" / "study"

    return {
        "instances_dir": str(data_study_dir / "instances"),
        "configs_dir": str(data_study_dir / "configs"),
        "results_dir": str(data_study_dir / "results"),
        "log_dir": str(data_study_dir / "results" / "logs"),
    }


def add_common_run_arguments(parser: argparse.ArgumentParser, defaults: dict) -> None:
    """Adds common arguments for run commands.

    Args:
        parser: The argument parser to add arguments to.
        defaults: Dictionary with default path values.
    """
    parser.add_argument(
        "-i",
        "--instances",
        type=str,
        default=defaults["instances_dir"],
        help="Directory containing .osil instance files.",
    )
    parser.add_argument(
        "-c",
        "--configs",
        type=str,
        default=defaults["configs_dir"],
        help="Directory containing .json configuration files.",
    )
    parser.add_argument(
        "-r",
        "--results",
        type=str,
        default=defaults["results_dir"],
        help="Directory for output files (CSV, tables, plots).",
    )
    parser.add_argument(
        "-l",
        "--logs",
        type=str,
        default=defaults["log_dir"],
        help="Directory for log files. Use 'none' to disable logging.",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=None,
        help="Maximum number of parallel workers. Default: auto.",
    )
    parser.add_argument(
        "-t",
        "--threads-per-job",
        type=int,
        default=4,
        help="Number of threads each solver job uses internally.",
    )


def create_parser() -> argparse.ArgumentParser:
    """Creates the argument parser with subcommands.

    Returns:
        Configured ArgumentParser.
    """
    defaults = get_default_paths()

    parser = argparse.ArgumentParser(
        description="Run computational studies and generate evaluation outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output."
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    run_parser = subparsers.add_parser(
        "run",
        help="Run all instance-config combinations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_run_arguments(run_parser, defaults)
    run_parser.add_argument(
        "--no-evaluate",
        action="store_true",
        help="Skip evaluation after run.",
    )

    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate existing CSV results and generate LaTeX outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    eval_parser.add_argument(
        "--csv", type=str, required=True, help="Path to the CSV results file."
    )
    eval_parser.add_argument(
        "-r",
        "--results",
        type=str,
        default=defaults["results_dir"],
        help="Directory for output files (tables, plots).",
    )
    eval_parser.add_argument(
        "--standalone",
        action="store_true",
        help="Generate standalone LaTeX documents.",
    )

    return parser


def get_shell_script_path() -> str:
    """Returns the path to the run_study.sh script.

    Returns:
        Absolute path to run_study.sh.
    """
    module_dir = Path(__file__).parent
    return str(module_dir / "run_study.sh")


def run_command(args: argparse.Namespace) -> int:
    """Executes the 'run' command via shell script.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    script_path = get_shell_script_path()

    if not os.path.exists(script_path):
        logger.error("Shell script not found: %s", script_path)
        return 1

    cmd = [
        "bash",
        script_path,
        "--instances",
        args.instances,
        "--configs",
        args.configs,
        "--results",
        args.results,
    ]

    if args.logs.lower() != "none":
        cmd.extend(["--logs", args.logs])

    if args.workers is not None:
        cmd.extend(["--workers", str(args.workers)])

    cmd.extend(["--threads", str(args.threads_per_job)])

    if args.no_evaluate:
        cmd.append("--no-evaluate")

    logger.info("Starting study via shell coordinator...")
    logger.info("Command: %s", " ".join(cmd))

    result = subprocess.run(cmd, check=False)
    return result.returncode


def evaluate_command(args: argparse.Namespace) -> int:
    """Executes the 'evaluate' command.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    logger.info("Evaluating CSV: %s", args.csv)
    evaluate_csv(args.csv, args.results, standalone=args.standalone)
    return 0


def evaluate_csv(csv_path: str, results_dir: str, standalone: bool = False) -> None:
    """Evaluates a CSV file and generates LaTeX outputs.

    Args:
        csv_path: Path to the CSV results file.
        results_dir: Base directory for output.
        standalone: Whether to generate standalone LaTeX documents.
    """
    logger.info("Starting evaluation...")

    evaluator = StudyEvaluator(csv_path)

    latex_config = LaTeXConfig(
        tables_dir=str(Path(results_dir) / "tables"),
        plots_dir=str(Path(results_dir) / "plots"),
        standalone=standalone,
    )

    generator = LaTeXGenerator(evaluator, latex_config)
    outputs = generator.generate_all()

    logger.info("Generated %d tables:", len(outputs["tables"]))
    for table_path in outputs["tables"]:
        logger.info("  - %s", table_path)

    logger.info("Generated %d plots:", len(outputs["plots"]))
    for plot_path in outputs["plots"]:
        logger.info("  - %s", plot_path)


def main() -> int:
    """Main entry point for the study CLI.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    parser = create_parser()
    args = parser.parse_args()

    log_level = "DEBUG" if args.verbose else "INFO"
    ut_io.config_console_logger(log_level)

    if args.command is None:
        parser.print_help()
        return 0

    try:
        if args.command == "run":
            return run_command(args)
        if args.command == "evaluate":
            return evaluate_command(args)

        parser.print_help()
        return 1

    except FileNotFoundError as exc:
        logger.error("File/Directory not found: %s", exc)
        return 1
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Unexpected error: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
