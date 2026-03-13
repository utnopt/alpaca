# -*- coding: utf-8 -*-

"""
Study module for computational experiments.

Usage:
    # Run a study (each job visible in ps aux)
    nohup python -m alpaca.study run -i ./instances -c ./configs -r ./results &

    # Check running jobs
    ps aux | grep run_single

    # Evaluate results
    python -m alpaca.study evaluate --csv ./results/raw/study_results_*.csv
"""

from alpaca.study.pipeline import (
    StudyConfig,
    StudyResults,
    StudyPipeline,
    run_study,
    discover_files,
)
from alpaca.study.evaluator import (
    StudyEvaluator,
    StudyData,
    ColumnStats,
    ConfigComparison,
)
from alpaca.study.latex_generator import LaTeXGenerator, LaTeXConfig

__all__ = [
    "StudyConfig",
    "StudyResults",
    "StudyPipeline",
    "run_study",
    "discover_files",
    "StudyEvaluator",
    "StudyData",
    "ColumnStats",
    "ConfigComparison",
    "LaTeXGenerator",
    "LaTeXConfig",
]
