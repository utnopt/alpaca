# -*- coding: utf-8 -*-
# pylint: disable=duplicate-code
"""
@authors: kuen,
"""
import unittest

import alpaca.settings as s
from alpaca.external_solvers import model_scip as msc, model_gurobi as mgu
import alpaca.model_data.model_data as mda
import alpaca.solver.solver as slv
from alpaca.mpip import mpiphandler as mph
from alpaca.mpip.separation import separationhandler_scip as mps


class TestModelData(unittest.TestCase):
    """Unit test for the model data buildup."""

    def setUp(self):
        """Common setup for all test methods."""

    @staticmethod
    def _run_model_test(instance_name):
        """Helper method to test model creation for a given instance."""
        config_dict = {"osil_file_name": instance_name}
        user_settings = s.UserSettings(config_dict)

        model_data = mda.ModelData(user_settings)
        scip_model = msc.ModelScip(model_data, user_settings)
        solver = slv.Solver(scip_model, user_settings)

        mpip_handler = mph.MPIPHandler(
            [
                model_data.expressions.nonlinear_expressions[expr_key]
                for expr_key in model_data.expressions.first_level_nonlinear_expression_keys
            ],
            model_data.expressions.bilinear_expressions,
            model_data.expressions.multilinear_expressions,
        )

        mpip_separation_handler = mps.SeparationHandler(
            mpip_handler, scip_model.opt_model
        )
        solver.mpip_separation_handler = mpip_separation_handler

        mgu.ModelGurobi(model_data, user_settings)

    def test_alkyl_model_data_creation(self):
        """Test model data creation for alkyl instance."""
        self._run_model_test("alkyl")

    def test_least_model_data_creation(self):
        """Test model data creation for least instance."""
        self._run_model_test("least")

    def test_st_e41_model_data_creation(self):
        """Test model data creation for st_e41 instance."""
        self._run_model_test("st_e41")

    def test_chance_model_data_creation(self):
        """Test model data creation for chance instance."""
        self._run_model_test("chance")

    def test_chem_model_data_creation(self):
        """Test model data creation for chem instance."""
        self._run_model_test("chem")

    def test_st_glmp_kk92_model_data_creation(self):
        """Test model data creation for st_glmp_kk92 instance."""
        self._run_model_test("st_glmp_kk92")


if __name__ == "__main__":
    unittest.main()
