# -*- coding: utf-8 -*-

"""
Pipeline module for orchestrating study execution.

This module discovers instances and configs, runs all combinations in parallel,
and writes results to a CSV file. If any configuration fails for an instance,
remaining configurations for that instance are skipped.
"""
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from multiprocessing import Manager
from pathlib import Path

from alpaca.study.runner import RunResult, RunStatus, run_single_combination
from alpaca.utils.localized_string_factory import LocalizedStringFactory as lsf
from alpaca.utils.logger import logger


@dataclass
class StudyConfig:
    """Configuration for a study run.

    Attributes:
        instances_dir: Directory containing .osil instance files.
        configs_dir: Directory containing .json configuration files.
        results_dir: Base directory for output (raw/, tables/, plots/).
        log_dir: Directory for log files. None disables file logging.
        max_workers: Maximum number of parallel workers. None calculates automatically.
        threads_per_job: Number of threads each solver job uses internally.
        suppress_output: If True, suppresses stdout during runs.
    """

    instances_dir: str
    configs_dir: str
    results_dir: str
    log_dir: str | None = None
    max_workers: int | None = None
    threads_per_job: int = 1
    suppress_output: bool = True

    def get_effective_max_workers(self) -> int:
        """Calculates the effective number of parallel workers.

        If max_workers is explicitly set, that value is used.
        Otherwise, calculates based on available CPU cores divided by
        threads_per_job to avoid oversubscription.

        Returns:
            Number of parallel workers to use.
        """
        if self.max_workers is not None:
            return self.max_workers

        available_cores = os.cpu_count() or 1
        return max(1, available_cores // self.threads_per_job)


@dataclass
class StudyResults:
    """Container for study execution results.

    Attributes:
        csv_path: Path to the generated CSV file.
        successful_runs: Number of successful runs written to CSV.
        failed_instances: Set of instance names that had at least one failure.
        skipped_runs: Number of runs skipped due to prior failures.
        total_combinations: Total number of instance-config combinations.
        execution_time: Total execution time in seconds.
    """

    csv_path: str
    successful_runs: int = 0
    failed_instances: set[str] = field(default_factory=set)
    skipped_runs: int = 0
    total_combinations: int = 0
    execution_time: float = 0.0


def discover_files(directory: str, extension: str) -> list[str]:
    """Discovers all files with a given extension in a directory.

    Args:
        directory: Directory to search in.
        extension: File extension to filter by (e.g., '.osil').

    Returns:
        Sorted list of full paths to matching files.

    Raises:
        FileNotFoundError: If the directory does not exist.
    """
    path = Path(directory)
    if not path.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory}")

    files = sorted([str(f) for f in path.glob(f"*{extension}")])
    return files


class StudyPipeline:
    """Orchestrates the execution of a computational study.

    This class manages the discovery of instances and configs, parallel
    execution of all combinations, and collection of results into a CSV file.
    If a configuration fails for an instance, remaining configurations for
    that instance are automatically skipped.
    """

    def __init__(self, config: StudyConfig) -> None:
        """Initializes the pipeline with the given configuration.

        Args:
            config: StudyConfig object with paths and settings.
        """
        self.config = config
        self.instances: list[str] = []
        self.configs: list[str] = []

    def discover(self) -> None:
        """Discovers all instances and configurations in the configured directories."""
        self.instances = discover_files(self.config.instances_dir, ".osil")
        self.configs = discover_files(self.config.configs_dir, ".json")

        logger.info(
            "Discovered %d instances and %d configs (%d total combinations)",
            len(self.instances),
            len(self.configs),
            len(self.instances) * len(self.configs),
        )

    def run(self) -> StudyResults:
        """Executes all instance-config combinations and writes results to CSV.

        All combinations are submitted for parallel execution. A shared state
        tracks failed instances, allowing workers to skip remaining configs
        for instances that have already failed.

        Returns:
            StudyResults with execution statistics.
        """
        start_time = time.time()

        # Ensure discovery has been run
        if not self.instances or not self.configs:
            self.discover()

        # Create output directories
        raw_dir = os.path.join(self.config.results_dir, "raw")
        os.makedirs(raw_dir, exist_ok=True)

        # Create CSV file with timestamp
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        csv_path = os.path.join(raw_dir, f"study_results_{timestamp}.csv")

        results = StudyResults(
            csv_path=csv_path,
            total_combinations=len(self.instances) * len(self.configs),
        )

        effective_workers = self.config.get_effective_max_workers()
        logger.info(
            "Starting parallel execution with %d workers (%d threads per job)...",
            effective_workers,
            self.config.threads_per_job,
        )

        # Execute all jobs in parallel with shared failed-instance tracking
        all_results = self._execute_jobs_parallel(effective_workers, self.config.threads_per_job)

        # Process results
        results_by_instance = self._group_results_by_instance(all_results)

        # Identify failed and skipped instances
        for instance_name, instance_results in results_by_instance.items():
            has_failure = any(r.status == RunStatus.ERROR for r in instance_results)
            if has_failure:
                results.failed_instances.add(instance_name)

            for result in instance_results:
                if result.status == RunStatus.SKIPPED:
                    results.skipped_runs += 1

        # Write successful results to CSV (only for instances without failures)
        self._write_results_to_csv(
            csv_path, results_by_instance, results.failed_instances
        )

        # Count successful runs
        for instance_name, instance_results in results_by_instance.items():
            if instance_name not in results.failed_instances:
                results.successful_runs += sum(
                    1 for r in instance_results if r.status == RunStatus.SUCCESS
                )

        results.execution_time = time.time() - start_time

        self._log_summary(results)

        return results

    def _execute_jobs_parallel(self, max_workers: int, nr_of_threads: int) -> list[RunResult]:
        """Executes all jobs in parallel using a process pool with shared state.

        Uses a multiprocessing Manager to share a dictionary of failed instances
        across all worker processes. When a worker encounters an error, it marks
        the instance as failed, and subsequent workers for the same instance
        will skip execution.

        Args:
            max_workers: Number of parallel workers to use.

        Returns:
            List of RunResult objects from all executions.
        """
        all_results: list[RunResult] = []

        # Create a shared dictionary for tracking failed instances
        with Manager() as manager:
            failed_instances = manager.dict()

            # Build job list with shared state reference
            jobs = [
                (
                    instance_path,
                    config_path,
                    self.config.log_dir,
                    self.config.suppress_output,
                    failed_instances,
                    nr_of_threads
                )
                for instance_path in self.instances
                for config_path in self.configs
            ]

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(run_single_combination, job): job for job in jobs
                }

                completed = 0
                total = len(futures)

                for future in as_completed(futures):
                    completed += 1
                    result = future.result()
                    all_results.append(result)

                    self._log_job_completion(completed, total, result)

        return all_results

    def _log_job_completion(
        self, completed: int, total: int, result: RunResult
    ) -> None:
        """Logs the completion of a single job.

        Args:
            completed: Number of completed jobs.
            total: Total number of jobs.
            result: The RunResult from the completed job.
        """
        status_map = {
            RunStatus.SUCCESS: "OK",
            RunStatus.ERROR: "FAILED",
            RunStatus.SKIPPED: "SKIPPED",
        }
        status_str = status_map.get(result.status, "UNKNOWN")

        logger.info(
            "[%d/%d] %s + %s: %s",
            completed,
            total,
            result.instance_name,
            result.config_name,
            status_str,
        )

        if result.status == RunStatus.ERROR:
            logger.error("  Error: %s", result.error_message)
        elif result.status == RunStatus.SKIPPED:
            logger.warning("  Skipped due to prior failure")

    def _group_results_by_instance(
        self, all_results: list[RunResult]
    ) -> dict[str, list[RunResult]]:
        """Groups run results by instance name.

        Args:
            all_results: List of all RunResult objects.

        Returns:
            Dictionary mapping instance names to lists of their results.
        """
        results_by_instance: dict[str, list[RunResult]] = {}

        for result in all_results:
            if result.instance_name not in results_by_instance:
                results_by_instance[result.instance_name] = []
            results_by_instance[result.instance_name].append(result)

        return results_by_instance

    def _write_results_to_csv(
        self,
        csv_path: str,
        results_by_instance: dict[str, list[RunResult]],
        failed_instances: set[str],
    ) -> None:
        """Writes successful results to the CSV file.

        Args:
            csv_path: Path to the output CSV file.
            results_by_instance: Dictionary of results grouped by instance.
            failed_instances: Set of instance names to exclude.
        """
        with open(csv_path, "w", encoding="utf-8") as csv_file:
            # Write header
            csv_file.write(lsf.stats_column_names() + "\n")

            # Write data rows for non-failed instances
            for instance_name in sorted(results_by_instance.keys()):
                if instance_name in failed_instances:
                    continue

                for result in results_by_instance[instance_name]:
                    if result.status == RunStatus.SUCCESS and result.csv_row:
                        csv_file.write(result.csv_row + "\n")

    def _log_summary(self, results: StudyResults) -> None:
        """Logs a summary of the study execution.

        Args:
            results: The StudyResults object with execution statistics.
        """
        logger.info("=" * 60)
        logger.info("Study completed!")
        logger.info("  Total combinations: %d", results.total_combinations)
        logger.info("  Successful runs: %d", results.successful_runs)
        logger.info("  Skipped runs: %d", results.skipped_runs)
        logger.info("  Failed instances: %d", len(results.failed_instances))
        if results.failed_instances:
            logger.info(
                "  Failed instance names: %s",
                ", ".join(sorted(results.failed_instances)),
            )
        logger.info("  Execution time: %.2f seconds", results.execution_time)
        logger.info("  Results written to: %s", results.csv_path)
        logger.info("=" * 60)


def run_study(  # pylint: disable=too-many-arguments, too-many-positional-arguments
    instances_dir: str,
    configs_dir: str,
    results_dir: str,
    log_dir: str | None = None,
    max_workers: int | None = None,
    threads_per_job: int = 1,
) -> StudyResults:
    """Convenience function to run a complete study.

    Args:
        instances_dir: Directory containing .osil files.
        configs_dir: Directory containing .json configuration files.
        results_dir: Directory for output files.
        log_dir: Directory for log files (optional).
        max_workers: Maximum parallel workers (optional, auto-calculated if None).
        threads_per_job: Number of threads each solver uses internally.

    Returns:
        StudyResults with execution statistics.
    """
    config = StudyConfig(
        instances_dir=instances_dir,
        configs_dir=configs_dir,
        results_dir=results_dir,
        log_dir=log_dir,
        max_workers=max_workers,
        threads_per_job=threads_per_job,
    )

    pipeline = StudyPipeline(config)
    return pipeline.run()
