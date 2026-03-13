# -*- coding: utf-8 -*-
# pylint: disable=too-many-locals

"""
Pipeline module using subprocess spawning.

Each job runs as a separate process visible in `ps aux | grep run_single`.
Results are written to CSV immediately when each job finishes.
"""
import os
import platform
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock, Semaphore, Thread

from alpaca.utils.localized_string_factory import LocalizedStringFactory as lsf
from alpaca.utils.logger import logger


@dataclass
class StudyConfig:
    """Configuration for a study run."""

    instances_dir: str
    configs_dir: str
    results_dir: str
    log_dir: str | None = None
    max_workers: int | None = None
    threads_per_job: int = 4

    def get_effective_max_workers(self) -> int:
        """Calculates number of parallel workers."""
        if self.max_workers is not None:
            return self.max_workers
        available_cores = os.cpu_count() or 1
        return max(1, available_cores // self.threads_per_job)


@dataclass
class StudyResults:
    """Container for study execution results."""

    csv_path: str
    successful_runs: int = 0
    failed_instances: set[str] = field(default_factory=set)
    skipped_runs: int = 0
    total_combinations: int = 0
    execution_time: float = 0.0


def discover_files(directory: str, extension: str) -> list[str]:
    """Discovers all files with a given extension in a directory."""
    path = Path(directory)
    if not path.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory}")
    return sorted([str(f) for f in path.glob(f"*{extension}")])


class StudyPipeline:
    """Orchestrates study execution by spawning visible subprocesses."""

    def __init__(self, config: StudyConfig) -> None:
        self.config = config
        self.instances: list[str] = []
        self.configs: list[str] = []
        self._csv_lock = Lock()
        self._results = StudyResults(csv_path="")

    def discover(self) -> None:
        """Discovers all instances and configurations."""
        self.instances = discover_files(self.config.instances_dir, ".osil")
        self.configs = discover_files(self.config.configs_dir, ".json")
        logger.info(
            "Discovered %d instances and %d configs (%d combinations)",
            len(self.instances),
            len(self.configs),
            len(self.instances) * len(self.configs),
        )

    def run(self) -> StudyResults:
        """Executes all combinations using subprocess spawning."""
        if hasattr(signal, "SIGHUP"):
            signal.signal(signal.SIGHUP, signal.SIG_IGN)

        start_time = time.time()

        if not self.instances or not self.configs:
            self.discover()

        # Setup directories and files
        raw_dir = os.path.join(self.config.results_dir, "raw")
        os.makedirs(raw_dir, exist_ok=True)

        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        csv_path = os.path.join(raw_dir, f"study_results_{timestamp}.csv")
        failed_file = os.path.join(raw_dir, f"failed_instances_{timestamp}.txt")
        error_log = os.path.join(raw_dir, f"errors_{timestamp}.log")

        self._results = StudyResults(
            csv_path=csv_path,
            total_combinations=len(self.instances) * len(self.configs),
        )

        # Write CSV header
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(lsf.stats_column_names() + "\n")

        # Create empty failed file
        Path(failed_file).touch()

        effective_workers = self.config.get_effective_max_workers()
        core_sets = self._create_core_sets(effective_workers)

        logger.info(
            "Starting %d workers, %d threads each",
            effective_workers,
            self.config.threads_per_job,
        )
        logger.info("Results: %s", csv_path)
        logger.info("Errors:  %s", error_log)

        # Semaphore to limit concurrent jobs
        semaphore = Semaphore(effective_workers)
        threads: list[Thread] = []

        job_num = 0
        total_jobs = len(self.instances) * len(self.configs)

        for instance in self.instances:
            for config in self.configs:
                job_num += 1
                slot = (job_num - 1) % effective_workers
                core_set = core_sets[slot]

                t = Thread(
                    target=self._run_job,
                    args=(
                        instance,
                        config,
                        core_set,
                        csv_path,
                        failed_file,
                        error_log,
                        semaphore,
                        job_num,
                        total_jobs,
                    ),
                )
                t.start()
                threads.append(t)

        # Wait for all threads
        for t in threads:
            t.join()

        # Count failed instances from file
        if os.path.exists(failed_file):
            with open(failed_file, "r", encoding="utf-8") as f:
                self._results.failed_instances = {
                    line.strip() for line in f if line.strip()
                }

        # Count successful runs from CSV
        if os.path.exists(csv_path):
            with open(csv_path, "r", encoding="utf-8") as f:
                self._results.successful_runs = sum(1 for _ in f) - 1  # minus header

        self._results.execution_time = time.time() - start_time
        self._log_summary()

        return self._results

    def _create_core_sets(self, num_workers: int) -> list[str]:
        """Creates CPU core sets for taskset."""
        num_cores = os.cpu_count() or 1
        core_sets = []
        for i in range(num_workers):
            start = (i * self.config.threads_per_job) % num_cores
            end = min(start + self.config.threads_per_job - 1, num_cores - 1)
            core_sets.append(f"{start}-{end}")
        return core_sets

    def _run_job(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        instance_path: str,
        config_path: str,
        core_set: str,
        csv_path: str,
        failed_file: str,
        error_log: str,
        semaphore: Semaphore,
        job_num: int,
        total_jobs: int,
    ) -> None:
        """Runs a single job as a subprocess."""
        semaphore.acquire()
        try:
            instance_name = os.path.splitext(os.path.basename(instance_path))[0]
            config_name = os.path.splitext(os.path.basename(config_path))[0]

            logger.info(
                "[%d/%d] Starting: %s + %s (cores %s)",
                job_num,
                total_jobs,
                instance_name,
                config_name,
                core_set,
            )

            log_dir = self.config.log_dir if self.config.log_dir else "none"

            # Build command
            cmd = [
                sys.executable,
                "-m",
                "alpaca.study.run_single",
                "--instance",
                instance_path,
                "--config",
                config_path,
                "--failed-file",
                failed_file,
                "--log-dir",
                log_dir,
                "--threads",
                str(self.config.threads_per_job),
            ]

            # Add taskset on Linux
            if platform.system() == "Linux":
                cmd = ["taskset", "-c", core_set] + cmd

            # Run subprocess
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)

            # Append stdout (CSV row) to results file immediately
            if result.returncode == 0 and result.stdout.strip():
                with self._csv_lock:
                    with open(csv_path, "a", encoding="utf-8") as f:
                        f.write(result.stdout.strip() + "\n")

            # Append stderr to error log
            if result.stderr.strip():
                with open(error_log, "a", encoding="utf-8") as f:
                    f.write(result.stderr)

            if result.returncode == 2:
                self._results.skipped_runs += 1

        finally:
            semaphore.release()

    def _log_summary(self) -> None:
        """Logs execution summary."""
        logger.info("=" * 60)
        logger.info("Study completed!")
        logger.info("  Total combinations: %d", self._results.total_combinations)
        logger.info("  Successful runs: %d", self._results.successful_runs)
        logger.info("  Skipped runs: %d", self._results.skipped_runs)
        logger.info("  Failed instances: %d", len(self._results.failed_instances))
        if self._results.failed_instances:
            logger.info(
                "  Failed: %s", ", ".join(sorted(self._results.failed_instances))
            )
        logger.info("  Time: %.2f seconds", self._results.execution_time)
        logger.info("  Results: %s", self._results.csv_path)
        logger.info("=" * 60)


def run_study(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    instances_dir: str,
    configs_dir: str,
    results_dir: str,
    log_dir: str | None = None,
    max_workers: int | None = None,
    threads_per_job: int = 4,
) -> StudyResults:
    """Convenience function to run a complete study."""
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
