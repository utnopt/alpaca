# -*- coding: utf-8 -*-

"""
CLI for running and evaluating studies.

Usage:
    python -m alpaca.study run -i ./instances -c ./configs -r ./results
    python -m alpaca.study evaluate --csv ./results/raw/study_results_*.csv
"""
import argparse
import sys
from pathlib import Path

from alpaca.study.pipeline import run_study, StudyResults
from alpaca.study.evaluator import StudyEvaluator
from alpaca.study.latex_generator import LaTeXGenerator, LaTeXConfig
from alpaca.utils.logger import logger
from alpaca.utils import inout as ut_io


def run_command(args: argparse.Namespace) -> int:
    """Executes the 'run' command."""
    log_dir = None if args.logs.lower() == "none" else args.logs

    logger.info("Starting study...")
    logger.info("  Instances: %s", args.instances)
    logger.info("  Configs:   %s", args.configs)
    logger.info("  Results:   %s", args.results)
    logger.info("  Logs:      %s", log_dir or "disabled")
    logger.info("  Workers:   %s", args.workers or "auto")
    logger.info("  Threads:   %d", args.threads)

    results: StudyResults = run_study(
        instances_dir=args.instances,
        configs_dir=args.configs,
        results_dir=args.results,
        log_dir=log_dir,
        max_workers=args.workers,
        threads_per_job=args.threads,
    )

    return 1 if results.failed_instances else 0


def evaluate_command(args: argparse.Namespace) -> int:
    """Executes the 'evaluate' command."""
    logger.info("Evaluating: %s", args.csv)

    evaluator = StudyEvaluator(args.csv)
    latex_config = LaTeXConfig(
        tables_dir=str(Path(args.results) / "tables"),
        plots_dir=str(Path(args.results) / "plots"),
        standalone=args.standalone,
    )

    generator = LaTeXGenerator(evaluator, latex_config)
    outputs = generator.generate_all()

    logger.info(
        "Generated %d tables, %d plots",
        len(outputs["tables"]),
        len(outputs["plots"]),
    )
    return 0


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run computational studies and evaluate results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output.")

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # RUN command
    run_parser = subparsers.add_parser("run", help="Run study.")
    run_parser.add_argument(
        "-i", "--instances", required=True, help="Directory with .osil files."
    )
    run_parser.add_argument(
        "-c", "--configs", required=True, help="Directory with .json configs."
    )
    run_parser.add_argument("-r", "--results", required=True, help="Output directory.")
    run_parser.add_argument(
        "-l", "--logs", default="none", help="Log directory or 'none'."
    )
    run_parser.add_argument(
        "-w", "--workers", type=int, default=None, help="Max parallel workers."
    )
    run_parser.add_argument(
        "-t", "--threads", type=int, default=4, help="Threads per job."
    )

    # EVALUATE command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate CSV results.")
    eval_parser.add_argument("--csv", required=True, help="Path to CSV file.")
    eval_parser.add_argument(
        "-r", "--results", default="./results", help="Output directory."
    )
    eval_parser.add_argument(
        "--standalone", action="store_true", help="Standalone LaTeX."
    )

    args = parser.parse_args()

    log_level = "DEBUG" if args.verbose else "INFO"
    ut_io.config_console_logger(log_level)

    if args.command == "run":
        return run_command(args)
    if args.command == "evaluate":
        return evaluate_command(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
