import unittest
import numpy as np
from hmip.hopfield import HopfieldSolver
import hmip.utils as utils


class TestHopfield(unittest.TestCase):
    def setUp(self):
        self.H = np.array([[2, 0], [0, 1]])
        self.q = np.array([-2.7, -1.8])
        self.k_max = 20
        self.binary_indicator = np.array([0, 1])
        self.beta = 1
        self.ub = np.array([1, 1])
        self.lb = np.array([0, 0])
        self.A = np.array([[0, 0], [0, 0]])
        self.b = np.array([0, 0])
        self.absorption = 1
        self.step_type = 'classic'
        self.objective_function = lambda x: 1 / 2 * np.dot(
            np.dot(x.T, self.H), x) + np.dot(self.q.T, x)
        self.gradient = lambda x: np.dot(self.H, x) + self.q
        self.smoothness_coefficient = utils.smoothness_coefficient(self.H)
        self.solver = HopfieldSolver(max_iterations=self.k_max)

    def test_hopfield_default(self):
        solver = HopfieldSolver(max_iterations=self.k_max)
        problem = solver.setup_optimization_problem(
            self.objective_function,
            self.gradient,
            self.lb,
            self.ub,
            self.binary_indicator,
            smoothness_coef=self.smoothness_coefficient)
        x, x_h, f_val_hist, step_size, _ = solver.solve(
            problem)
        self.assertEqual(x.shape[0], self.q.shape[0])
        self.assertEqual(x.shape[1], self.k_max)

    def test_hopfield_step_type_classic(self):
        solver = HopfieldSolver(max_iterations=self.k_max, step_type='classic')
        problem = solver.setup_optimization_problem(
            self.objective_function,
            self.gradient,
            self.lb,
            self.ub,
            self.binary_indicator,
            smoothness_coef=self.smoothness_coefficient)
        x, x_h, f_val_hist, step_size, _ = solver.solve(
            problem)
        self.assertEqual(x.shape[0], self.q.shape[0])
        self.assertEqual(x.shape[1], self.k_max)

    def test_hopfield_step_type_armijo(self):
        solver = HopfieldSolver(max_iterations=self.k_max, step_type='armijo')
        problem = solver.setup_optimization_problem(
            self.objective_function,
            self.gradient,
            self.lb,
            self.ub,
            self.binary_indicator,
            smoothness_coef=self.smoothness_coefficient)
        x, x_h, f_val_hist, step_size, _ = solver.solve(
            problem)
        self.assertEqual(x.shape[0], self.q.shape[0])
        self.assertEqual(x.shape[1], self.k_max)

    def test_hopfield_with_absorption(self):
        solver = HopfieldSolver(max_iterations=self.k_max,
                                step_type='classic',
                                absorption_criterion=True)
        problem = solver.setup_optimization_problem(
            self.objective_function,
            self.gradient,
            self.lb,
            self.ub,
            self.binary_indicator,
            smoothness_coef=self.smoothness_coefficient)
        x, x_h, f_val_hist, step_siz, _ = solver.solve(
            problem)
        self.assertEqual(x.shape[0], self.q.shape[0])
        self.assertEqual(x.shape[1], self.k_max)


class TestOthers(unittest.TestCase):
    def setUp(self):
        self.H = np.array([[2, 0], [0, 1]])
        self.q = np.array([-2.7, -1.8])
        self.k_max = 20
        self.binary_indicator = np.array([0, 1])
        self.beta = 1
        self.ub = np.array([1, 1])
        self.lb = np.array([0, 0])
        self.A = np.array([[0, 0], [0, 0]])
        self.b = np.array([0, 0])
        self.absorption = 1
        self.step_type = 'classic'
        self.x_0 = self.lb + (self.ub - self.lb) / 2
        self.objective_function = lambda x: 1 / 2 * np.dot(
            np.dot(x.T, self.H), x) + np.dot(self.q.T, x)
        self.gradient = lambda x: np.dot(self.H, x) + self.q
        self.smoothness_coefficient = utils.smoothness_coefficient(self.H)
        self.n = 2
        self.x = np.ones((self.n, self.k_max))

        self.ascent_stop_criterion = 0.1

    # TODO(Mathilde): For all the following -> need to find better tests

    def test_create_initial_ascent_ascent(self):
        solver = HopfieldSolver(max_iterations=self.k_max)
        problem = solver.setup_optimization_problem(
            self.objective_function,
            self.gradient,
            self.lb,
            self.ub,
            self.binary_indicator,
            smoothness_coef=self.smoothness_coefficient)
        x_init = solver._compute_x_0(problem)
        self.assertTrue(np.array_equal(self.x_0.shape, x_init.shape))

    def test_create_initial_ascent_binary_neutral_ascent(self):
        solver = HopfieldSolver(max_iterations=self.k_max,
                                initial_ascent_type='binary_neutral_ascent')
        problem = solver.setup_optimization_problem(
            self.objective_function,
            self.gradient,
            self.lb,
            self.ub,
            self.binary_indicator,
            smoothness_coef=self.smoothness_coefficient)
        x_init = solver._compute_x_0(problem)
        self.assertTrue(np.array_equal(self.x_0.shape, x_init.shape))

    def test_activation_function(self):
        x_0 = self.lb + (self.ub - self.lb) / 2
        beta = 0.5 * x_0
        activation_types = ['pwl', 'exp', 'sin', 'identity', 'tanh']

        for activation_type in activation_types:
            solver = HopfieldSolver(max_iterations=self.k_max,
                                    activation_type=activation_type,
                                    beta=beta)
            self.assertTrue(
                np.array_equal(x_0, solver._activation(x_0, self.lb, self.ub)))
