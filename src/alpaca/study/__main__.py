# -*- coding: utf-8 -*-

"""
Command-line interface for running computational studies.

Usage:
    python -m alpaca.study run [OPTIONS]
    python -m alpaca.study evaluate --csv <path> [OPTIONS]

To run in background (survives shell close):
    nohup python -m alpaca.study run [OPTIONS] > study.log 2>&1 &
"""
import argparse
import sys
from pathlib import Path

from alpaca.study.pipeline import run_study, StudyResults
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
    return {
        "instances_dir": str(cwd / "instances"),
        "configs_dir": str(cwd / "configs"),
        "results_dir": str(cwd / "results"),
        "log_dir": str(cwd / "results" / "logs"),
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
        epilog="To run in background: nohup python -m alpaca.study run [OPTIONS] &",
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


def run_command(args: argparse.Namespace) -> int:
    """Executes the 'run' command.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    log_dir = None if args.logs.lower() == "none" else args.logs

    logger.info("Starting study pipeline...")
    logger.info("  Instances: %s", args.instances)
    logger.info("  Configs: %s", args.configs)
    logger.info("  Results: %s", args.results)
    logger.info("  Logs: %s", log_dir or "disabled")
    logger.info("  Workers: %s", args.workers or "auto")
    logger.info("  Threads per job: %d", args.threads_per_job)

    results: StudyResults = run_study(
        instances_dir=args.instances,
        configs_dir=args.configs,
        results_dir=args.results,
        log_dir=log_dir,
        max_workers=args.workers,
        threads_per_job=args.threads_per_job,
    )

    if not args.no_evaluate and results.successful_runs > 0:
        evaluate_csv(results.csv_path, args.results)

    if results.failed_instances:
        return 1
    return 0


def evaluate_command(args: argparse.Namespace) -> int:
    """Executes the 'evaluate' command.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    logger.info("Evaluating CSV: %s", args.csv)
    evaluate_csv(args.csv, args.results)
    return 0


def evaluate_csv(csv_path: str, results_dir: str) -> None:
    """Evaluates a CSV file and generates LaTeX outputs.

    Args:
        csv_path: Path to the CSV results file.
        results_dir: Base directory for output.
    """
    logger.info("Starting evaluation...")

    evaluator = StudyEvaluator(csv_path)

    latex_config = LaTeXConfig(
        tables_dir=str(Path(results_dir) / "tables"),
        plots_dir=str(Path(results_dir) / "plots"),
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
