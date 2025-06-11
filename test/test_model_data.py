# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
import unittest

import alpaca.settings as s
import alpaca.model_data.model_data as mda


class TestModelData(unittest.TestCase):
    """unit test for the model data buildup."""

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
                              "least", "st_e41", "chance"]:
            print(test_instance)
            config_dict = {"osil_file_name": test_instance}
            user_settings = s.UserSettings(config_dict)

            model_data = mda.ModelData(user_settings)
            model_data.build_model_from_osil_data()


if __name__ == "__main__":
    unittest.main()
