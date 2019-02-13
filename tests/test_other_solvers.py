import unittest
import numpy as np
import hmip.other_solvers as other_solvers
import hmip.hopfield as hop


class TestCvxpy(unittest.TestCase):
    def setUp(self):
        self.H = np.array([[2, 0], [0, 1]])
        self.q = np.array([-2.7, -1.8])
        self.binary_indicator = np.array([0, 1])
        self.ub = np.array([1, 1])
        self.lb = np.array([0, 0])
        self.k_max = 200

    def test_cxvpy_solver_default(self):
        x_solution = other_solvers.cvxpy_solver(self.H, self.q, self.lb, self.ub, self.binary_indicator)
        self.assertTrue(np.allclose(x_solution, np.array([1, 1]), rtol=0.1), 2)

    def test_csvxpy_solver_default_comparison(self):
        x_cvxpy = other_solvers.cvxpy_solver(self.H, self.q, self.lb, self.ub, self.binary_indicator)
        x_hopfield, _, _, _ = hop.hopfield(self.H, self.q, self.lb, self.ub, self.binary_indicator,
                                                     k_max=self.k_max)
        self.assertTrue(np.allclose(x_cvxpy, x_hopfield[:, -1], rtol=0.1), 2)
