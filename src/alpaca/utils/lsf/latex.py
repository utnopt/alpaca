# -*- coding: utf-8 -*-
# pylint: disable=too-many-public-methods, missing-function-docstring

"""
LaTeX-related strings for table generation.

@authors: kuen,
"""


class Latex:
    """LaTeX-related strings for table generation."""

    @classmethod
    def begin_table(cls) -> str:
        return r"\begin{table}[htbp]"

    @classmethod
    def end_table(cls) -> str:
        return r"\end{table}"

    @classmethod
    def centering(cls) -> str:
        return r"\centering"

    @classmethod
    def begin_tabular(cls, alignment: str) -> str:
        return f"\\begin{{tabular}}{{{alignment.strip()}}}"

    @classmethod
    def end_tabular(cls) -> str:
        return r"\end{tabular}"

    @classmethod
    def toprule(cls) -> str:
        return r"\toprule"

    @classmethod
    def midrule(cls) -> str:
        return r"\midrule"

    @classmethod
    def bottomrule(cls) -> str:
        return r"\bottomrule"

    @classmethod
    def caption(cls, text: str) -> str:
        return f"\\caption{{{text}}}"

    @classmethod
    def label(cls, text: str) -> str:
        return f"\\label{{{text}}}"

    @classmethod
    def row_end(cls) -> str:
        return r" \\"

    @classmethod
    def column_separator(cls) -> str:
        return " & "

    @classmethod
    def placeholder(cls) -> str:
        return "--"

    @classmethod
    def begin_itemize(cls) -> str:
        return r"\begin{itemize}"

    @classmethod
    def end_itemize(cls) -> str:
        return r"\end{itemize}"

    @classmethod
    def item(cls, short: str, full: str) -> str:
        return f"  \\item {short}: {full}"

    @classmethod
    def underscore(cls) -> str:
        return "_"

    @classmethod
    def escape_underscore(cls, text: str) -> str:
        return text.replace("_", r"\\_")

    @classmethod
    def math_mode(cls, value: str) -> str:
        return f"${value}$"

    @classmethod
    def percentage_suffix(cls) -> str:
        return r"\%"

    @classmethod
    def mean_label(cls) -> str:
        return "Mean"

    @classmethod
    def shifted_geometric_mean_label(cls) -> str:
        return "Shifted Geometric Mean"

    @classmethod
    def default_alignment(cls, num_columns: int) -> str:
        return "l " + "r " * (num_columns - 1)
