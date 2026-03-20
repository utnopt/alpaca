# -*- coding: utf-8 -*-
# pylint: disable=missing-docstring, too-many-public-methods
"""
@authors: kuen,
"""


class Statistics:
    """Statistics format and field names."""

    @classmethod
    def stats_format_gurobi(cls) -> str:
        return "stats_gurobi"

    @classmethod
    def stats_format_scip(cls) -> str:
        return "stats_scip"

    @classmethod
    def stats_format_latex(cls) -> str:
        return "stats_latex"

    @classmethod
    def stats_solving_time(cls, f="") -> str:
        if f == cls.stats_format_gurobi():
            return "Runtime"
        if f == cls.stats_format_scip():
            return r"Solving Time \(sec\)\s*:\s*([\d.]+)"
        if f == cls.stats_format_latex():
            return r"Time (s)"
        return "solving_time"

    @classmethod
    def stats_nr_nodes(cls, f="") -> str:
        if f == cls.stats_format_gurobi():
            return "NodeCount"
        if f == cls.stats_format_scip():
            return r"Solving Nodes\s*:\s*(\d+)"
        if f == cls.stats_format_latex():
            return r"\# Nodes"
        return "nr_nodes"

    @classmethod
    def stats_solution_value(cls, f="") -> str:
        if f == cls.stats_format_gurobi():
            return "ObjVal"
        if f == cls.stats_format_scip():
            return r"Primal Bound\s*:\s*([+-]?[\d.eE+\-]+)(?:\s*\(\d+\s+solutions?\))?"
        if f == cls.stats_format_latex():
            return "Obj"
        return "solution_value"

    @classmethod
    def stats_mip_gap(cls, f="") -> str:
        if f == cls.stats_format_gurobi():
            return "MIPGap"
        if f == cls.stats_format_scip():
            return r"^Gap\s*:\s*([\d.]+)\s*%"
        if f == cls.stats_format_latex():
            return r"GAP (\%)"
        return "mip_gap"

    @classmethod
    def stats_scip_model_size_section(cls) -> str:
        return r"Original Problem\s*:(.+?)(?=Presolved Problem\s*:|$)"

    @classmethod
    def stats_final_nr_vars(cls, f="") -> str:
        if f == cls.stats_format_gurobi():
            return "NumVars"
        if f == cls.stats_format_scip():
            return r"Variables\s*:\s*(\d+)"
        if f == cls.stats_format_latex():
            return r"FIN \#Vars"
        return "final_nr_vars"

    @classmethod
    def stats_final_nr_constraints(cls, f="") -> str:
        if f == cls.stats_format_gurobi():
            return "NumConstrs"
        if f == cls.stats_format_scip():
            return r"Constraints\s*:\s*(\d+)\s+initial"
        if f == cls.stats_format_latex():
            return r"FIN \# Con"
        return "final_nr_constraints"

    @classmethod
    def stats_scip_presolve_section(cls) -> str:
        return r"Presolved Problem\s*:(.+?)(?=Presolvers\s*:|$)"

    @classmethod
    def stats_presolved_nr_vars(cls, f="") -> str:
        if f == cls.stats_format_gurobi():
            return "PresolvedNumVars"
        if f == cls.stats_format_scip():
            return r"Variables\s*:\s*(\d+)"
        if f == cls.stats_format_latex():
            return r"PRE \# Var"
        return "presolved_nr_vars"

    @classmethod
    def stats_presolved_nr_constraints(cls, f="") -> str:
        if f == cls.stats_format_gurobi():
            return "PresolvedNumConstrs"
        if f == cls.stats_format_scip():
            return r"Constraints\s*:\s*(\d+)\s+initial"
        if f == cls.stats_format_latex():
            return r"PRE \# Con"
        return "presolved_nr_constraints"

    @classmethod
    def stats_presolved_nr_nonzeros(cls, f="") -> str:
        if f == cls.stats_format_gurobi():
            return "PresolvedNumNZs"
        if f == cls.stats_format_scip():
            return r"Nonzeros\s*:\s*(\d+)\s+constraint"
        if f == cls.stats_format_latex():
            return r"PRE \# NZ"
        return "presolved_nr_nonzeros"

    @classmethod
    def stats_root_solution_value(cls, f="") -> str:
        if f == cls.stats_format_gurobi():
            return "RelaxObj"
        if f == cls.stats_format_scip():
            return r"First LP value\s*:\s*([+-]?[\d.eE+\-]+)"
        if f == cls.stats_format_latex():
            return "Root Value"
        return "root_solution_value"

    @classmethod
    def stats_root_solving_time(cls, f="") -> str:
        if f == cls.stats_format_gurobi():
            return "RelaxTime"
        if f == cls.stats_format_scip():
            return r"First LP Time\s*:\s*([+-]?[\d.eE+\-]+)"
        if f == cls.stats_format_latex():
            return "Root Time (s)"
        return "root_solving_time"

    @classmethod
    def stats_build_time(cls, f="") -> str:
        if f == cls.stats_format_latex():
            return "Model Building Time (s)"
        return "build_time"

    @classmethod
    def stats_original_nr_variables(cls, f="") -> str:
        if f == cls.stats_format_latex():
            return r"OG \# Var"
        return "original_nr_variables"

    @classmethod
    def stats_original_nr_constraints(cls, f="") -> str:
        if f == cls.stats_format_latex():
            return r"OG \# Con"
        return "original_nr_constraints"

    @classmethod
    def stats_original_nr_bilinear_expressions(cls, f="") -> str:
        if f == cls.stats_format_latex():
            return r"OG \# BL Expr"
        return "original_nr_bilinear_expressions"

    @classmethod
    def stats_original_nr_bilinear_binary_expressions(cls, f="") -> str:
        if f == cls.stats_format_latex():
            return r"OG \# BLB Expr"
        return "original_nr_bilinear_binary_expressions"

    @classmethod
    def stats_original_nr_mixed_binary_expressions(cls, f="") -> str:
        if f == cls.stats_format_latex():
            return r"OG \# MB Expr"
        return "original_nr_mixed_binary_expressions"

    @classmethod
    def stats_original_nr_multilinear_expressions(cls, f="") -> str:
        if f == cls.stats_format_latex():
            return r"OG \# nD Expr"
        return "original_nr_multilinear_expressions"

    @classmethod
    def stats_original_nr_one_dim_expressions(cls, f="") -> str:
        if f == cls.stats_format_latex():
            return r"OG \# 1D Expr"
        return "original_nr_one_dim_expressions"

    @classmethod
    def stats_pwl_nr_variables(cls, f="") -> str:
        if f == cls.stats_format_latex():
            return r"PWL \# Var"
        return "pwl_nr_variables"

    @classmethod
    def stats_pwl_nr_constraints(cls, f="") -> str:
        if f == cls.stats_format_latex():
            return r"PWL \# Con"
        return "pwl_nr_constraints"

    @classmethod
    def stats_pwl_nr_bilinear_expressions(cls, f="") -> str:
        if f == cls.stats_format_latex():
            return r"PWL \# BL Expr"
        return "pwl_nr_bilinear_expressions"

    @classmethod
    def stats_pwl_nr_bilinear_binary_expressions(cls, f="") -> str:
        if f == cls.stats_format_latex():
            return r"PWL \# BLB Expr"
        return "pwl_nr_bilinear_binary_expressions"

    @classmethod
    def stats_pwl_nr_mixed_binary_expressions(cls, f="") -> str:
        if f == cls.stats_format_latex():
            return r"PWL \# MB Expr"
        return "pwl_nr_mixed_binary_expressions"

    @classmethod
    def stats_pwl_nr_multilinear_expressions(cls, f="") -> str:
        if f == cls.stats_format_latex():
            return r"PWL \# nD Expr"
        return "pwl_nr_multilinear_expressions"

    @classmethod
    def stats_pwl_nr_one_dim_expressions(cls, f="") -> str:
        if f == cls.stats_format_latex():
            return r"PWL \# 1D Expr"
        return "pwl_nr_one_dim_expressions"

    @classmethod
    def stats_locatelli_domain_volume_polygon(cls, f="") -> str:
        if f == cls.stats_format_latex():
            return "LOC PG"
        return "locatelli_domain_volume_polygon"

    @classmethod
    def stats_locatelli_domain_volume_polytope(cls, f="") -> str:
        if f == cls.stats_format_latex():
            return "LOC PT"
        return "locatelli_domain_volume_polytope"

    @classmethod
    def stats_locatelli_nr_cuts(cls, f="") -> str:
        if f == cls.stats_format_latex():
            return r"\# Cuts Locatelli"
        return "locatelli_nr_cuts"

    @classmethod
    def stats_mpip_nr_instances(cls, f="") -> str:
        if f == cls.stats_format_latex():
            return r"\# Instances MPIP"
        return "mpip_nr_instances"

    @classmethod
    def stats_mpip_ratio(cls, f="") -> str:
        if f == cls.stats_format_latex():
            return "MPIP Ratio"
        return "mpip_ratio"

    @classmethod
    def stats_instance_name(cls, f="") -> str:
        if f == cls.stats_format_latex():
            return "Instance Name"
        return "instance_name"

    @classmethod
    def stats_config_name(cls, f="") -> str:
        if f == cls.stats_format_latex():
            return "Configuration Name"
        return "config_name"

    @classmethod
    def stats_column_names(cls) -> str:
        return (
            f"{cls.stats_instance_name()},"
            f"{cls.stats_config_name()},"
            f"{cls.stats_solving_time()},"
            f"{cls.stats_nr_nodes()},"
            f"{cls.stats_solution_value()},"
            f"{cls.stats_mip_gap()},"
            f"{cls.stats_final_nr_vars()},"
            f"{cls.stats_final_nr_constraints()},"
            f"{cls.stats_presolved_nr_vars()},"
            f"{cls.stats_presolved_nr_constraints()},"
            f"{cls.stats_presolved_nr_nonzeros()},"
            f"{cls.stats_root_solution_value()},"
            f"{cls.stats_root_solving_time()},"
            f"{cls.stats_build_time()},"
            f"{cls.stats_original_nr_variables()},"
            f"{cls.stats_original_nr_constraints()},"
            f"{cls.stats_original_nr_bilinear_expressions()},"
            f"{cls.stats_original_nr_bilinear_binary_expressions()},"
            f"{cls.stats_original_nr_mixed_binary_expressions()},"
            f"{cls.stats_original_nr_multilinear_expressions()},"
            f"{cls.stats_original_nr_one_dim_expressions()},"
            f"{cls.stats_pwl_nr_variables()},"
            f"{cls.stats_pwl_nr_constraints()},"
            f"{cls.stats_pwl_nr_bilinear_expressions()},"
            f"{cls.stats_pwl_nr_bilinear_binary_expressions()},"
            f"{cls.stats_pwl_nr_mixed_binary_expressions()},"
            f"{cls.stats_pwl_nr_multilinear_expressions()},"
            f"{cls.stats_pwl_nr_one_dim_expressions()},"
            f"{cls.stats_locatelli_domain_volume_polygon()},"
            f"{cls.stats_locatelli_domain_volume_polytope()},"
            f"{cls.stats_locatelli_nr_cuts()},"
            f"{cls.stats_mpip_nr_instances()},"
            f"{cls.stats_mpip_ratio()}"
        )
