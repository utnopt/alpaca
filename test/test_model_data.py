# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
import unittest

import alpaca.settings as s
import alpaca.model_scip.model_scip as msc
import alpaca.model_data.model_data as mda
import alpaca.solver.solver as slv
from alpaca.mpip import mpiphandler as mph, separationhandler as mps


class TestModelData(unittest.TestCase):
    """Unit test for the model data buildup."""

    def setUp(self):
        """
        Function that creates a problem instance and executes the unittests on it.
        """
        self.test_model_data_creation()

    def test_model_data_creation(self):
        """
        Function that executes the unittest for model data creation.
        """
        for test_instance in ["alkyl", "ann_compressor_exp",
                              "least", "st_e41", "chance", "chem"]:
            print(test_instance)
            config_dict = {"osil_file_name": test_instance}
            user_settings = s.UserSettings(config_dict)

            model_data = mda.ModelData(user_settings)

            scip_model = msc.ModelScip(model_data, user_settings)

            solver = slv.Solver(scip_model, user_settings)

            mpip_handler = mph.MPIPHandler(model_data.first_level_nonlinear_expressions)

            mpip_separation_handler = mps.SeparationHandler(mpip_handler, scip_model.opt_model)
            solver.mpip_separation_handler = mpip_separation_handler

            solver.solve_instance()


if __name__ == "__main__":
    unittest.main()
