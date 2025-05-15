# -*- coding: utf-8 -*-
"""
Created on dd.mm.yyyy

@author: [first_name] [last_name]
"""
import unittest
from my_project import run_scenario


class TestExample(unittest.TestCase):
    """unit test for a function or feature"""

    def setUp(self):
        """
        function that creates a problem instance and executes the unittests on it
        """
        print("Create a examplary problem instance")
        self.problem_instance = (0, 1)

        print("Execute the unittest for feature x on the examplary problem instance")
        self.test_feature_x()

        print("Test if the run-code terminates without an error ")
        self.test_executable()

    def test_feature_x(self):
        """
        function that executes the unittest for a certain feature
        """
        a = run_scenario.run_optimization(
            # self.problem_instance[0]
        )
        b = run_scenario.run_optimization(
            # self.problem_instance[0]
        )
        # TODO: read relevant properties from both results
        self.assertEqual(a.solution, b.solution)

    def test_executable(self):
        """
        function that tests if the run-program terminates
        without an error
        """


if __name__ == "__main__":
    unittest.main()
