import unittest
import numpy as np
import hmip.other_solvers as other_solvers


class TestCvxpy(unittest.TestCase):
    def setUp(self):
        self.H = np.array([[2, 0], [0, 1]])
        self.q = np.array([-2.7, -1.8])
        self.binary_indicator = np.array([0, 1])
        self.ub = np.array([1, 1])
        self.lb = np.array([0, 0])

    def test_cxvpy_solver_default(self):
        x_solution = other_solvers.cvxpy_solver(self.H, self.q, self.lb, self.ub, self.binary_indicator)
        print(x_solution)
