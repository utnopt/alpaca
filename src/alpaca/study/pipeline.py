# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

from alpaca.utils.lsf.localized_string_factory import LocalizedStringFactory as lsf
from alpaca.utils.logger import logger


@dataclass
class StudyConfig:
    """Configuration for a study run.

    Attributes:
        instances_dir: Directory containing .osil instance files.
        configs_dir: Directory containing .json configuration files.
        results_dir: Base directory for output (raw/, tables/, plots/).
        log_dir: Directory for log files. None disables file logging.
        max_workers: Maximum number of parallel workers.
        threads_per_job: Number of threads each solver job uses internally.
    """

    instances_dir: str
    configs_dir: str
    results_dir: str
    log_dir: str | None = None
    max_workers: int | None = None
    threads_per_job: int = 4

    def get_effective_max_workers(self) -> int:
        """Calculates the effective number of parallel workers.

        Returns:
            Number of parallel workers to use.
        """
        available_cores = os.cpu_count() or 1
        if self.max_workers is not None:
            return min(self.max_workers, available_cores)

        return max(1, available_cores // self.threads_per_job)


@dataclass
class StudyResults:
    """Container for study execution results.

    Attributes:
        csv_path: Path to the generated CSV file.
        successful_runs: Number of successful runs written to CSV.
        failed_instances: Set of instance names that had at least one failure.
        skipped_runs: Number of runs skipped due to prior failures.
        total_instances: Total number of instances.
        execution_time: Total execution time in seconds.
    """

    csv_path: str
    successful_runs: int = 0
    failed_instances: set[str] = field(default_factory=set)
    skipped_runs: int = 0
    total_instances: int = 0
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


def extract_name(path: str) -> str:
    """Extracts the base name without extension from a path.

    Args:
        path: Full file path.

    Returns:
        Base name without extension.
    """
    return os.path.splitext(os.path.basename(path))[0]


class StudyPipeline:
    """Orchestrates the execution of a computational study.

    This class manages the discovery of instances and configs, spawns
    separate subprocesses for each combination (visible in ps aux),
    and writes results immediately to CSV.
    """

    def __init__(self, config: StudyConfig) -> None:
        """Initializes the pipeline with the given configuration.

        Args:
            config: StudyConfig object with paths and settings.
        """
        self.config = config
        self.instances: list[str] = []
        self.configs: list[str] = []
        self._failed_instances: set[str] = set()
        self._failed_file: str = ""

    def discover(self) -> None:
        """Discovers all instances and configurations."""
        self.instances = discover_files(self.config.instances_dir, ".osil")
        self.configs = discover_files(self.config.configs_dir, ".json")

        logger.info(
            "Discovered %d instances and %d configs (%d total combinations)",
            len(self.instances),
            len(self.configs),
            len(self.instances) * len(self.configs),
        )

    def run(self) -> StudyResults:
        """Executes all instance-config combinations as separate subprocesses.

        Each job runs as a separate Python process, visible in ps aux.
        Results are written immediately to CSV when each job completes.

        Returns:
            StudyResults with execution statistics.
        """
        start_time = time.time()

        if not self.instances or not self.configs:
            self.discover()

        raw_dir = os.path.join(self.config.results_dir, "raw")
        os.makedirs(raw_dir, exist_ok=True)

        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        csv_path = os.path.join(raw_dir, f"study_results_{timestamp}.csv")
        error_log_path = os.path.join(raw_dir, f"errors_{timestamp}.log")
        self._failed_file = os.path.join(raw_dir, f".failed_instances_{timestamp}")

        results = StudyResults(
            csv_path=csv_path,
            total_instances=len(self.instances),
        )

        effective_workers = self.config.get_effective_max_workers()
        logger.info(
            "Starting execution with %d parallel workers (%d threads per job)...",
            effective_workers,
            self.config.threads_per_job,
        )
        logger.info("Results file: %s", csv_path)
        logger.info("Error log: %s", error_log_path)
        logger.info("")
        logger.info("Monitor with: tail -f %s", csv_path)
        logger.info("See jobs with: ps aux | grep alpaca.study.run_job")
        logger.info("")

        with open(csv_path, "w", encoding="utf-8") as csv_file:
            csv_file.write(lsf.stats_column_names() + "\n")
            csv_file.flush()

            with open(error_log_path, "w", encoding="utf-8") as error_file:
                self._run_all_jobs(csv_file, error_file, results, effective_workers)

        if os.path.exists(self._failed_file):
            os.remove(self._failed_file)

        results.execution_time = time.time() - start_time
        self._log_summary(results)

        return results

    def _run_all_jobs(  # pylint: disable=too-many-locals
        self,
        csv_file: TextIO,
        error_file: TextIO,
        results: StudyResults,
        max_workers: int,
    ) -> None:
        """Runs all jobs with controlled parallelism.

        Args:
            csv_file: Open file handle for CSV output.
            error_file: Open file handle for error log.
            results: StudyResults to update.
            max_workers: Maximum parallel processes.
        """
        jobs = self.instances

        num_cores = os.cpu_count() or 1
        core_sets = self._compute_core_sets(max_workers, num_cores)

        available_slots = list(range(max_workers))

        running: dict[subprocess.Popen, tuple[str, int, int]] = {}
        job_index = 0
        completed = 0
        total = len(jobs)

        while completed < total:
            # Only start a new job if there's a slot explicitly available
            while available_slots and job_index < total:
                instance_path = jobs[job_index]
                instance_name = extract_name(instance_path)

                # Take the next available slot identifier (e.g., 0, 1, or 2)
                slot = available_slots.pop(0)
                core_set = core_sets[slot % len(core_sets)]

                proc = self._start_job(instance_path, core_set)

                # Store the slot alongside the process so we can return it when done
                running[proc] = (instance_path, job_index, slot)

                logger.info(
                    "[%d/%d] RUNNING: %s (cores %s, PID %d)",
                    job_index + 1,
                    total,
                    instance_name,
                    core_set,
                    proc.pid,
                )
                job_index += 1

            if running:
                finished_procs = []
                for proc in list(running.keys()):
                    return_code = proc.poll()
                    if return_code is not None:
                        finished_procs.append(proc)

                if not finished_procs:
                    time.sleep(0.1)
                    continue

                for proc in finished_procs:
                    # Retrieve the specific slot that this process was occupying
                    instance_path, _, slot = running.pop(proc)

                    # Return the slot back to the available pool
                    available_slots.append(slot)

                    instance_name = extract_name(instance_path)
                    completed += 1

                    stdout, stderr = proc.communicate()

                    if proc.returncode == 0 and stdout.strip():
                        csv_file.write(stdout.strip() + "\n")
                        csv_file.flush()
                        results.successful_runs += 1
                        logger.info(
                            "[%d/%d] SUCCESS: %s",
                            completed,
                            total,
                            instance_name,
                        )
                    else:
                        self._failed_instances.add(instance_name)
                        results.failed_instances.add(instance_name)
                        error_file.write(f"=== {instance_name} ===\n")
                        error_file.write(stderr)
                        error_file.write("\n\n")
                        error_file.flush()
                        logger.error(
                            "[%d/%d] FAILED: %s",
                            completed,
                            total,
                            instance_name,
                        )

    def _start_job(self, instance_path: str, core_set: str) -> subprocess.Popen:
        """Starts a single job as a subprocess.

        Args:
            instance_path: Path to the instance file.
            core_set: CPU cores to use (e.g., "0-3").

        Returns:
            Popen object for the started process.
        """
        cmd = [
            sys.executable,
            "-m",
            "alpaca.study.run_job",
            "--instance",
            instance_path,
            "--configs",
            ",".join(self.configs),
            "--threads",
            str(self.config.threads_per_job),
        ]

        if self.config.log_dir is not None:
            cmd.extend(["--log-dir", self.config.log_dir])

        use_taskset = self._taskset_available()
        if use_taskset:
            cmd = ["taskset", "-c", core_set] + cmd

        return subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    @staticmethod
    def _taskset_available() -> bool:
        """Checks if taskset command is available.

        Returns:
            True if taskset is available, False otherwise.
        """
        try:
            subprocess.run(
                ["taskset", "--version"],
                capture_output=True,
                check=False,
            )
            return True
        except FileNotFoundError:
            return False

    @staticmethod
    def _compute_core_sets(max_workers: int, num_cores: int) -> list[str]:
        """Computes CPU core sets for each worker slot.

        Args:
            max_workers: Number of parallel workers.
            num_cores: Total number of CPU cores.

        Returns:
            List of core set strings (e.g., ["0-3", "4-7"]).
        """
        cores_per_worker = max(1, num_cores // max_workers)
        core_sets = []

        for i in range(max_workers):
            start = (i * cores_per_worker) % num_cores
            end = min(start + cores_per_worker - 1, num_cores - 1)
            core_sets.append(f"{start}-{end}")

        return core_sets

    @staticmethod
    def _log_summary(results: StudyResults) -> None:
        """Logs a summary of the study execution.

        Args:
            results: The StudyResults object with execution statistics.
        """
        logger.info("=" * 60)
        logger.info("Study completed!")
        logger.info("  Total instances: %d", results.total_instances)
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
    threads_per_job: int = 4,
) -> StudyResults:
    """Convenience function to run a complete study.

    Args:
        instances_dir: Directory containing .osil files.
        configs_dir: Directory containing .json configuration files.
        results_dir: Directory for output files.
        log_dir: Directory for log files (optional).
        max_workers: Maximum parallel workers (optional).
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
