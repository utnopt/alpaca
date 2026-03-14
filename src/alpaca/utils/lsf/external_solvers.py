# -*- coding: utf-8 -*-
# pylint: disable=missing-docstring
"""
@authors: kuen,
"""


class ExternalSolvers:
    """Solver names and solver-specific parameters."""

    # --- Solver Parameters ---

    @classmethod
    def mip_solver_parameter_time_limit(cls, solver_name: str) -> str:
        if solver_name == cls.solver_name_scip():
            return "limits/time"
        if solver_name == cls.solver_name_gurobi():
            return "TimeLimit"
        return "time_limit"

    @classmethod
    def mip_solver_parameter_presolve(cls, solver_name: str) -> str:
        if solver_name == cls.solver_name_scip():
            return "presolving/maxrounds"
        if solver_name == cls.solver_name_gurobi():
            return "Presolve"
        return "Presolve"

    @classmethod
    def mip_solver_parameter_thread_limit(cls, solver_name: str) -> str:
        if solver_name == cls.solver_name_scip():
            return "parallel/maxnthreads"
        if solver_name == cls.solver_name_gurobi():
            return "Threads"
        return "thread_limit"

    @classmethod
    def mip_solver_parameter_seed(cls, solver_name: str) -> str:
        if solver_name == cls.solver_name_scip():
            return "randomization/randomseedshift"
        if solver_name == cls.solver_name_gurobi():
            return "Seed"
        return "seed"

    # --- SCIP Specific ---

    @classmethod
    def solver_name_scip(cls) -> str:
        return "scip"

    @classmethod
    def scip_result_tag(cls) -> str:
        return "result"

    @classmethod
    def scip_status_infeasible(cls) -> str:
        return "infeasible"

    @classmethod
    def scip_status_unbounded(cls) -> str:
        return "unbounded"

    @classmethod
    def scip_status_optimal(cls) -> str:
        return "optimal"

    @classmethod
    def scip_parameter_reoptimization(cls) -> str:
        return "reoptimization/enable"

    @classmethod
    def scip_separator_name_mpip(cls) -> str:
        return "mpip"

    @classmethod
    def scip_separator_description_mpip(cls) -> str:
        return "Multipartite Implication Polytope Separator"

    # --- Gurobi Specific ---

    @classmethod
    def solver_name_gurobi(cls) -> str:
        return "gurobi"

    @classmethod
    def gurobi_parameter_output_flag(cls) -> str:
        return "OutputFlag"

    @classmethod
    def gurobi_parameter_logfile(cls) -> str:
        return "LogFile"

    @classmethod
    def gurobi_parameter_heuristics(cls) -> str:
        return "Heuristics"

    @classmethod
    def gurobi_model_attribute_sense(cls) -> str:
        return "ModelSense"
