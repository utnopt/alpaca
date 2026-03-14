# -*- coding: utf-8 -*-

"""
Study module for computational experiments.

This module provides functionality to:
- Run optimization instances with different configurations in parallel
- Collect and evaluate results
- Generate LaTeX tables and TikZ plots

Usage:
    # From command line (run in background)
    nohup python -m alpaca.study run -i instances/ -c configs/ -r results/ &

    # Monitor progress
    tail -f results/raw/study_results_*.csv
    ps aux | grep alpaca.study.run_job

    # Evaluate results
    python -m alpaca.study evaluate --csv results/raw/study_results_*.csv
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
    "LaTeXGenerator",
    "LaTeXConfig",
]
