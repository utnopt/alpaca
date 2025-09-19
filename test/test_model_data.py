# -*- coding: utf-8 -*-
# pylint: disable=duplicate-code
"""
@authors: kuen,
"""
import unittest

import alpaca.model_data.model_data as mda
from alpaca.external_solvers import mip_model as mm
import alpaca.solver.solver as slv
import alpaca.mpip.mpip_handler as mph
import alpaca.mpip.separation.mpip_separationhandler as msh
import alpaca.settings as s


class TestModelData(unittest.TestCase):
    """Unit test for the model data buildup."""

    def setUp(self):
        """Common setup for all test methods."""

    @staticmethod
    def _run_model_test(instance_name, approximation, reformulate_multilinear):
        """Helper method to test model creation for a given instance."""
        config_dict = {
            "osil_file_name": instance_name,
            "approximation": approximation,
            "reformulate_multilinear": reformulate_multilinear,
        }
        user_settings = s.UserSettings(config_dict)

        model_data = mda.ModelData(user_settings)

        external_solver = mm.MIPModel(
            model_data, user_settings, user_settings.external_solver
        )
        solver = slv.Solver(external_solver, user_settings)
        mpip_handler = mph.MPIPHandler(model_data)
        mpip_separation_handler = msh.MPIPSeparationHandler(
            mpip_handler, external_solver.opt_model
        )
        solver.mpip_separation_handler = mpip_separation_handler

    def test_alkyl_model_data_creation(self):
        """Test model data creation for alkyl instance."""
        for approx in [0, 1]:
            for reformulate in [0, 1]:
                with self.subTest(
                    approximation=approx, reformulate_multilinear=reformulate
                ):
                    self._run_model_test("alkyl", approx, reformulate)

    def test_least_model_data_creation(self):
        """Test model data creation for least instance."""
        for approx in [0, 1]:
            for reformulate in [0, 1]:
                with self.subTest(
                    approximation=approx, reformulate_multilinear=reformulate
                ):
                    self._run_model_test("least", approx, reformulate)

    def test_st_e41_model_data_creation(self):
        """Test model data creation for st_e41 instance."""
        for approx in [0, 1]:
            for reformulate in [0, 1]:
                with self.subTest(
                    approximation=approx, reformulate_multilinear=reformulate
                ):
                    self._run_model_test("st_e41", approx, reformulate)

    def test_chance_model_data_creation(self):
        """Test model data creation for chance instance."""
        for approx in [0, 1]:
            for reformulate in [0, 1]:
                with self.subTest(
                    approximation=approx, reformulate_multilinear=reformulate
                ):
                    self._run_model_test("chance", approx, reformulate)

    def test_chem_model_data_creation(self):
        """Test model data creation for chem instance."""
        for approx in [0, 1]:
            for reformulate in [0, 1]:
                with self.subTest(
                    approximation=approx, reformulate_multilinear=reformulate
                ):
                    self._run_model_test("chem", approx, reformulate)

    def test_st_glmp_kk92_model_data_creation(self):
        """Test model data creation for st_glmp_kk92 instance."""
        for approx in [0, 1]:
            for reformulate in [0, 1]:
                with self.subTest(
                    approximation=approx, reformulate_multilinear=reformulate
                ):
                    self._run_model_test("st_glmp_kk92", approx, reformulate)


if __name__ == "__main__":
    unittest.main()
