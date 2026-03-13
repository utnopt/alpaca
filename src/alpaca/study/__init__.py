# -*- coding: utf-8 -*-

"""
Study module for computational experiments.

This module provides functionality to:
- Run optimization instances with different configurations in parallel
- Collect and evaluate results
- Generate LaTeX tables and TikZ plots

Usage:
    # From command line (recommended)
    bash run_study.sh -i instances/ -c configs/ -r results/

    # From Python CLI
    python -m alpaca.study run -i instances/ -c configs/ -r results/
    python -m alpaca.study evaluate --csv results/raw/study_results.csv

    # From Python
    from alpaca.study import StudyEvaluator, LaTeXGenerator
"""

from alpaca.study.evaluator import (
    StudyEvaluator,
    StudyData,
    ColumnStats,
    ConfigComparison,
)
from alpaca.study.latex_generator import LaTeXGenerator, LaTeXConfig

__all__ = [
    "StudyEvaluator",
    "StudyData",
    "ColumnStats",
    "ConfigComparison",
    "LaTeXGenerator",
    "LaTeXConfig",
]
