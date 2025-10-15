# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
# pylint: disable=too-many-public-methods
from typing import Any
import gurobipy as gp
from gurobipy import nlfunc
import pyscipopt as scip
from pyscipopt import SCIP_RESULT

from alpaca.utils.localized_string_factory import LocalizedStringFactory as lsf


class SolverWrapper:
    """A wrapper class for solver-agnostic MIP model handling."""

    def __init__(self, mip_solver: str):
        self.mip_solver = mip_solver.lower()
        if self.mip_solver == lsf.solver_name_gurobi():
            self.model: gp.Model | scip.Model | scip.Eventhdlr = gp.Model()
        elif self.mip_solver == lsf.solver_name_scip():
            self.model: gp.Model | scip.Model | scip.Eventhdlr = scip.Model()
        else:
            raise ValueError(
                lsf.error_input_choice_invalid(
                    mip_solver, [lsf.solver_name_gurobi(), lsf.solver_name_scip()]
                )
            )

    def optimize(self, callback_function=None) -> None:
        """Optimize the model."""
        if self.mip_solver == lsf.solver_name_gurobi():
            self.model.optimize(callback_function)
        else:  # scip
            self.model.optimize()
            self.model.printStatistics()

    def add_constraint(self, expression: Any, name: str = "") -> Any:
        """Add a constraint to the model."""
        if self.mip_solver == lsf.solver_name_gurobi():
            return self.model.addConstr(expression, name=name)
        return self.model.addCons(expression, name=name)

    def add_nonlinear_constraint(
        self, res_var: Any, expression: Any, name: str = ""
    ) -> Any:
        """Add a nonlinear constraint to the model."""
        if self.mip_solver == lsf.solver_name_gurobi():
            return self.model.addGenConstrNL(res_var, expression, name)
        return self.model.addCons(res_var == expression, name=name)

    def add_variable(  # pylint: disable=too-many-arguments, too-many-positional-arguments
        self,
        name: str = lsf.empty_string(),
        lb: float = 0.0,
        ub: float = float(lsf.numpy_infinity()),
        obj: float = 0.0,
        vtype: str = lsf.var_type_continuous(),
    ) -> Any:
        """Add a variable to the model."""
        return self.model.addVar(name=name, vtype=vtype, lb=lb, ub=ub, obj=obj)

    def get_val(self, var: Any) -> float:
        """Get the value of a variable after solving."""
        if self.mip_solver == lsf.solver_name_gurobi():
            return var.X
        return self.model.getVal(var)

    def get_mip_gap(self) -> float:
        """Get the MIP gap of the current solution."""
        if self.mip_solver == lsf.solver_name_gurobi():
            return self.model.MIPGap
        return self.model.getGap()

    def get_nr_of_applied_cuts(self) -> int:
        """Get the number of applied cuts."""
        if self.mip_solver == lsf.solver_name_gurobi():
            return self.model.NumUserCuts
        return self.model.getNCutsApplied()

    def get_val_callback(self, var: Any) -> float:
        """Get the value of a variable in a callback."""
        if self.mip_solver == lsf.solver_name_gurobi():
            return self.model.cbGetNodeRel(var)
        return self.model.getVal(var)

    def set_objective_sense(self, sense: str) -> None:
        """Set the optimization sense (maximize or minimize)."""
        if sense == lsf.objective_sense_maximize():
            if self.mip_solver == lsf.solver_name_gurobi():
                self.model.setAttr(lsf.gurobi_model_attribute_sense(), gp.GRB.MAXIMIZE)
            else:  # scip
                self.model.setMaximize()
        elif sense == lsf.objective_sense_minimize():
            if self.mip_solver == lsf.solver_name_gurobi():
                self.model.setAttr(lsf.gurobi_model_attribute_sense(), gp.GRB.MINIMIZE)
            else:  # scip
                self.model.setMinimize()
        else:
            raise ValueError(
                lsf.error_input_choice_invalid(
                    sense,
                    [lsf.objective_sense_minimize(), lsf.objective_sense_maximize()],
                )
            )

    def update_objective(
        self, new_objective: Any, sense=lsf.objective_sense_minimize()
    ) -> None:
        """Update the model's objective function, for re-optimization."""
        if self.mip_solver == lsf.solver_name_gurobi():
            self.model.setObjective(new_objective)
            self.set_objective_sense(sense)
        else:  # scip
            self.model.freeReoptSolve()
            self.model.chgReoptObjective(new_objective, sense=sense)

    def get_new_expression(self) -> Any:
        """Get a new, empty expression object for building objectives/constraints."""
        if self.mip_solver == lsf.solver_name_gurobi():
            return gp.LinExpr()
        return scip.Expr()

    def hide_output(self) -> None:
        """Hide solver output."""
        if self.mip_solver == lsf.solver_name_gurobi():
            self.model.setParam(lsf.gurobi_parameter_output_flag(), 0)
        else:  # scip
            self.model.hideOutput()

    def enable_reoptimization(self):
        """Enable re-optimization features if supported."""
        if self.mip_solver == lsf.solver_name_scip():
            self.model.setParam(lsf.scip_parameter_reoptimization(), True)

    def set_seed(self, seed: int) -> None:
        """Set the random seed for the solver."""
        self.model.setParam(lsf.mip_solver_parameter_seed(self.mip_solver), seed)

    def create_cut(  # pylint: disable=too-many-arguments, too-many-positional-arguments
        self, separation_handler: Any, name: str, lhs=None, rhs=None, local=False
    ) -> Any:
        """Create a cut object."""
        if self.mip_solver == lsf.solver_name_gurobi():
            return GurobiCut(separation_handler, name, lhs=lhs, rhs=rhs, local=local)
        return self.model.createEmptyRowSepa(
            separation_handler,
            name,
            lhs=lhs,
            rhs=rhs,
            local=local,
        )

    def add_var_to_cut(self, cut: Any, var: Any, coeff: float) -> None:
        """Add a variable with coefficient to a cut."""
        if self.mip_solver == lsf.solver_name_gurobi():
            cut.lhs += coeff * var
        else:  # scip
            self.model.addVarToRow(
                cut,
                var,
                coeff,
            )

    def add_cut(self, cut: Any) -> None:
        """Add a cut to the model."""
        if self.mip_solver == lsf.solver_name_gurobi():
            self.model.cbCut(cut.lhs <= cut.rhs)
        else:  # scip
            self.model.addCut(cut, forcecut=True)
            self.model.releaseRow(cut)

    def set_objective(
        self, expression: Any, sense=lsf.objective_sense_minimize()
    ) -> None:
        """Set the model's objective function."""
        if self.mip_solver == lsf.solver_name_scip():
            self.model.freeTransform()
        self.model.setObjective(expression)
        self.set_objective_sense(sense)

    def set_time_limit(self, time_limit: int) -> None:
        """Set a time limit for the solver."""
        self.model.setParam(
            lsf.mip_solver_parameter_time_limit(self.mip_solver), time_limit
        )

    def turn_off_presolve(self) -> None:
        """Turn off presolve for the solver."""
        self.model.setParam(lsf.mip_solver_parameter_presolve(self.mip_solver), 0)

    def set_thread_limit(self, thread_limit: int) -> None:
        """Set a thread limit for the solver."""
        self.model.setParam(
            lsf.mip_solver_parameter_thread_limit(self.mip_solver), thread_limit
        )

    def get_nonlinear_function(self, nonlinearity_type: str) -> Any:
        """Get nonlinear function based on solver type."""
        if self.mip_solver == lsf.solver_name_gurobi():
            return get_nonlinear_function_gurobi(nonlinearity_type)
        return get_nonlinear_function_scip(nonlinearity_type)

    @staticmethod
    def set_variable_lb(variable: Any, lb: float) -> None:
        """Set the lower bound of a variable."""
        variable.LB = lb

    @staticmethod
    def set_variable_ub(variable: Any, ub: float) -> None:
        """Set the upper bound of a variable."""
        variable.UB = ub

    def get_objective_value(self) -> float:
        """Get the objective value of the solution."""
        if self.mip_solver == lsf.solver_name_gurobi():
            return self.model.ObjVal
        return self.model.getObjVal()

    def is_infeasible(self) -> bool:
        """Check if the model has a feasible solution."""
        if self.mip_solver == lsf.solver_name_gurobi():
            return self.model.Status == gp.GRB.INFEASIBLE
        return self.model.getStatus() == lsf.scip_status_infeasible()

    def is_optimal(self) -> bool:
        """Check if the model has been solved to optimality."""
        if self.mip_solver == lsf.solver_name_gurobi():
            return self.model.Status == gp.GRB.OPTIMAL
        return self.model.getStatus() == lsf.scip_status_optimal()


class GurobiCut:
    """A class representing a cut in Gurobi."""

    def __init__(  # pylint: disable=too-many-arguments, too-many-positional-arguments
        self,
        separation_handler: Any,
        name: str,
        lhs: gp.LinExpr | None = None,
        rhs: float | None = None,
        local=False,
    ) -> None:
        self.separation_handler = separation_handler
        self.name = name
        self.lhs = lhs if lhs is not None else gp.LinExpr()
        self.rhs = rhs if rhs is not None else 0.0
        self.local = local


class ScipSeparation(scip.Sepa):
    """Wrapper for SCIP separation handler."""

    def __init__(self, mpip_separation_handler: Any) -> None:
        self.mpip_separation_handler = mpip_separation_handler
        scip.Eventhdlr.__init__(mpip_separation_handler.opt_model.model)

    def sepaexeclp(self):
        """Run callback event."""
        self.mpip_separation_handler.opt_model = self.model
        return (
            {lsf.scip_result_tag(): SCIP_RESULT.SEPARATED}
            if self.mpip_separation_handler.separate_solution()
            else {lsf.scip_result_tag(): SCIP_RESULT.DIDNOTFIND}
        )


def gurobi_separation_callback(grb_model, where):
    """Callback for mpip separation"""
    if where == gp.GRB.Callback.MIPNODE:
        if grb_model.cbGet(gp.GRB.Callback.MIPNODE_STATUS) == gp.GRB.Status.OPTIMAL:
            grb_model._mpip_separation_handler.separate_solution()  # pylint: disable=protected-access


def get_nonlinear_function_scip(  # pylint: disable=too-many-return-statements
    nonlinearity_type: str,
) -> Any:
    """Get scip nonlinear function."""

    if nonlinearity_type == lsf.nonlinearity_type_square():
        return lambda x: x**2
    if nonlinearity_type == lsf.nonlinearity_type_exp():
        return scip.exp
    if nonlinearity_type == lsf.nonlinearity_type_ln():
        return scip.log
    if nonlinearity_type == lsf.nonlinearity_type_sqrt():
        return scip.sqrt
    if nonlinearity_type == lsf.nonlinearity_type_sin():
        return scip.sin
    if nonlinearity_type == lsf.nonlinearity_type_cos():
        return scip.cos
    if nonlinearity_type == lsf.nonlinearity_type_log10():
        return lambda x: scip.log(x) / scip.log(10)
    if nonlinearity_type == lsf.nonlinearity_type_tanh():
        return lambda x: (1 - scip.exp(-2 * x)) / (1 + scip.exp(-2 * x))
    if nonlinearity_type == lsf.nonlinearity_type_inverse():
        return lambda x: x**-1
    if nonlinearity_type == lsf.nonlinearity_type_xabsx():
        return lambda x: x * abs(x)
    if nonlinearity_type == lsf.nonlinearity_type_negate():
        return lambda x: -x

    raise NotImplementedError(lsf.error_nonlinearity_not_implemented(nonlinearity_type))


def get_nonlinear_function_gurobi(  # pylint: disable=too-many-return-statements
    nonlinearity_type: str,
) -> Any:
    """Get gurobi nonlinear function."""
    if nonlinearity_type == lsf.nonlinearity_type_square():
        return nlfunc.square
    if nonlinearity_type == lsf.nonlinearity_type_exp():
        return nlfunc.exp
    if nonlinearity_type == lsf.nonlinearity_type_ln():
        return nlfunc.log
    if nonlinearity_type == lsf.nonlinearity_type_sqrt():
        return nlfunc.sqrt
    if nonlinearity_type == lsf.nonlinearity_type_sin():
        return nlfunc.sin
    if nonlinearity_type == lsf.nonlinearity_type_cos():
        return nlfunc.cos
    if nonlinearity_type == lsf.nonlinearity_type_log10():
        return nlfunc.log10
    if nonlinearity_type == lsf.nonlinearity_type_tanh():
        return lambda x: (1 - nlfunc.exp(-2 * x)) / (1 + nlfunc.exp(-2 * x))
    if nonlinearity_type == lsf.nonlinearity_type_inverse():
        return lambda x: 1 / x
    if nonlinearity_type == lsf.nonlinearity_type_xabsx():
        return lambda x: x * gp.abs_(x)
    if nonlinearity_type == lsf.nonlinearity_type_negate():
        return lambda x: -x

    raise NotImplementedError(lsf.error_nonlinearity_not_implemented(nonlinearity_type))
