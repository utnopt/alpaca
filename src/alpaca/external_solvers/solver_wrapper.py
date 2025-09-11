# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
from typing import Any
import gurobipy as gp
import pyscipopt as scip
from pyscipopt import SCIP_RESULT


class SolverWrapper:
    """A wrapper class for solver-agnostic MIP model handling."""

    def __init__(self, mip_solver: str):
        self.mip_solver = mip_solver.lower()
        if self.mip_solver == "gurobi":
            self.model: gp.Model | scip.Model | scip.Eventhdlr = gp.Model()
        elif self.mip_solver == "scip":
            self.model: gp.Model | scip.Model | scip.Eventhdlr = scip.Model()
        else:
            raise ValueError("mip_solver must be 'gurobi' or 'scip'")

    def optimize(self, callback_function=None) -> None:
        """Optimize the model."""
        if self.mip_solver == "gurobi":
            self.model.optimize(callback_function)
        else:  # scip
            self.model.optimize()

    def add_constraint(self, expression: Any, name: str = "") -> Any:
        """Add a constraint to the model."""
        if self.mip_solver == "gurobi":
            return self.model.addConstr(expression, name=name)
        return self.model.addCons(expression, name=name)

    def add_variable(  # pylint: disable=too-many-arguments, too-many-positional-arguments
        self,
        name: str = "",
        lb: float = 0.0,
        ub: float = float("inf"),
        obj: float = 0.0,
        vtype: str = "C",
    ) -> Any:
        """Add a variable to the model."""
        return self.model.addVar(name=name, vtype=vtype, lb=lb, ub=ub, obj=obj)

    def get_val(self, var: Any) -> float:
        """Get the value of a variable after solving."""
        if self.mip_solver == "gurobi":
            return var.X
        return self.model.getVal(var)

    def get_val_callback(self, var: Any) -> float:
        """Get the value of a variable in a callback."""
        if self.mip_solver == "gurobi":
            return self.model.cbGetNodeRel(var)
        return self.model.getVal(var)

    def set_objective_sense(self, sense: str) -> None:
        """Set the optimization sense (maximize or minimize)."""
        if sense == "maximize":
            if self.mip_solver == "gurobi":
                self.model.setAttr("ModelSense", gp.GRB.MAXIMIZE)
            else:  # scip
                self.model.setMaximize()
        elif sense == "minimize":
            if self.mip_solver == "gurobi":
                self.model.setAttr("ModelSense", gp.GRB.MINIMIZE)
            else:  # scip
                self.model.setMinimize()
        else:
            raise ValueError("sense must be 'maximize' or 'minimize'")

    def update_objective(self, new_objective: Any, sense="minimize") -> None:
        """Update the model's objective function, for re-optimization."""
        if self.mip_solver == "gurobi":
            self.model.setObjective(new_objective)
            self.set_objective_sense(sense)
        else:  # scip
            self.model.freeReoptSolve()
            self.model.chgReoptObjective(new_objective, sense=sense)

    def get_new_expression(self) -> Any:
        """Get a new, empty expression object for building objectives/constraints."""
        if self.mip_solver == "gurobi":
            return gp.LinExpr()
        return scip.Expr()

    def hide_output(self) -> None:
        """Hide solver output."""
        if self.mip_solver == "gurobi":
            self.model.setParam("OutputFlag", 0)
        else:  # scip
            self.model.hideOutput()

    def enable_reoptimization(self):
        """Enable re-optimization features if supported."""
        if self.mip_solver == "scip":
            self.model.setParam("reoptimization/enable", True)

    def set_seed(self, seed: int) -> None:
        """Set the random seed for the solver."""
        if self.mip_solver == "gurobi":
            self.model.setParam("Seed", seed)
        else:  # scip
            self.model.setParam("randomization/randomseedshift", seed)

    def create_cut(  # pylint: disable=too-many-arguments, too-many-positional-arguments
        self, separation_handler: Any, name: str, lhs=None, rhs=None, local=False
    ) -> Any:
        """Create a cut object."""
        if self.mip_solver == "gurobi":
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
        if self.mip_solver == "gurobi":
            cut.lhs += coeff * var
        else:  # scip
            self.model.addVarToRow(
                cut,
                var,
                coeff,
            )

    def add_cut(self, cut: Any) -> None:
        """Add a cut to the model."""
        if self.mip_solver == "gurobi":
            self.model.cbCut(cut.lhs <= cut.rhs)
        else:  # scip
            self.model.addCut(cut, forcecut=True)
            self.model.releaseRow(cut)

    def set_objective(self, expression: Any, sense="minimize") -> None:
        """Set the model's objective function."""
        self.model.setObjective(expression)
        self.set_objective_sense(sense)

    def set_time_limit(self, time_limit: int) -> None:
        """Set a time limit for the solver."""
        if self.mip_solver == "gurobi":
            self.model.setParam("TimeLimit", time_limit)
        else:  # scip
            self.model.setRealParam("limits/time", time_limit)

    def set_thread_limit(self, thread_limit: int) -> None:
        """Set a thread limit for the solver."""
        if self.mip_solver == "gurobi":
            self.model.setParam("Threads", thread_limit)
        else:  # scip
            self.model.setParam("parallel/maxnthreads", thread_limit)


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
        if self.model.getDepth() == 0:
            return {"result": SCIP_RESULT.DIDNOTFIND}
        self.mpip_separation_handler.opt_model = self.model
        return (
            {"result": SCIP_RESULT.SEPARATED}
            if self.mpip_separation_handler.separate_solution()
            else {"result": SCIP_RESULT.DIDNOTFIND}
        )


def gurobi_separation_callback(grb_model, where):
    """Callback for mpip separation"""
    if where == gp.GRB.Callback.MIPNODE:
        if grb_model.cbGet(gp.GRB.Callback.MIPNODE_STATUS) == gp.GRB.Status.OPTIMAL:
            grb_model._mpip_separation_handler.separate_solution()  # pylint: disable=protected-access
