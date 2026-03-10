# -*- coding: utf-8 -*-

"""
Study module for computational experiments.

This module provides functionality to:
- Run optimization instances with different configurations in parallel
- Collect and evaluate results
- Generate LaTeX tables and TikZ plots

Usage:
    # From command line
    python -m alpaca.study run
    python -m alpaca.study evaluate --csv path/to/results.csv
    python -m alpaca.study full

    # From Python
    from alpaca.study import run_study, StudyEvaluator, LaTeXGenerator
"""

from alpaca.study.pipeline import (
    StudyConfig,
    StudyResults,
    StudyPipeline,
    run_study,
    discover_files,
)
from alpaca.study.runner import (
    RunStatus,
    RunResult,
    run_single_combination,
    extract_name_from_path,
)
from alpaca.study.evaluator import (
    StudyEvaluator,
    StudyData,
    ColumnStats,
    ConfigComparison,
)
from alpaca.study.latex_generator import LaTeXGenerator, LaTeXConfig

__all__ = [
    # Pipeline
    "StudyConfig",
    "StudyResults",
    "StudyPipeline",
    "run_study",
    "discover_files",
    # Runner
    "RunStatus",
    "RunResult",
    "run_single_combination",
    "extract_name_from_path",
    # Evaluator
    "StudyEvaluator",
    "StudyData",
    "ColumnStats",
    "ConfigComparison",
    # LaTeX
    "LaTeXGenerator",
    "LaTeXConfig",
]
