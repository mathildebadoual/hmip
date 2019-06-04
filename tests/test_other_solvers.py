import unittest
import numpy as np
import hmip.other_solvers as other_solvers
from hmip.hopfield import HopfieldSolver
import hmip.utils as utils


class TestCvxpy(unittest.TestCase):
    def setUp(self):
        self.H = np.array([[2, 0], [0, 1]])
        self.q = np.array([-2.7, -1.8])
        self.binary_indicator = np.array([0, 1])
        self.ub = np.array([1, 1])
        self.lb = np.array([0, 0])
        self.k_max = 200
        self.objective_function = lambda x: 1 / 2 * np.dot(
            np.dot(x.T, self.H), x) + np.dot(self.q.T, x)
        self.gradient = lambda x: np.dot(self.H, x) + self.q
        self.smoothness_coefficient = utils.smoothness_coefficient(self.H)

    def test_cxvpy_solver_default(self):
        x_solution = other_solvers.cvxpy_solver(self.H, self.q, self.lb,
                                                self.ub, self.binary_indicator)
        self.assertTrue(np.allclose(x_solution, np.array([1, 1]), rtol=0.1), 2)

    def test_cvxpy_solver_default_comparison(self):
        x_cvxpy = other_solvers.cvxpy_solver(self.H, self.q, self.lb, self.ub,
                                             self.binary_indicator)
        solver = HopfieldSolver(max_iterations=self.k_max)
        problem = solver.setup_optimization_problem(
            self.objective_function,
            self.gradient,
            self.lb,
            self.ub,
            self.binary_indicator,
            smoothness_coef=self.smoothness_coefficient)
        x_hopfield, _, _, _, _ = solver.solve(problem)
        self.assertTrue(np.allclose(x_cvxpy, x_hopfield[:, -1], rtol=0.1), 2)
