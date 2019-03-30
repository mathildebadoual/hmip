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
        self.objective_function = lambda x: 1 / 2 * np.dot(np.dot(x.T, self.H), x) + np.dot(self.q.T, x)
        self.gradient = lambda x: np.dot(self.H, x) + self.q
        self.smoothness_coefficient = utils.smoothness_coefficient(self.H)
        self.solver = HopfieldSolver(max_iterations=self.k_max)

    def test_raise_error_no_problem_set(self):
        with self.assertRaises(Exception) as context:
            self.solver.solve_optimization_problem()

        self.assertTrue('Problem is not set' in str(context.exception))

    def test_hopfield_default(self):
        solver = HopfieldSolver(max_iterations=self.k_max)
        solver.setup_optimization_problem(self.objective_function, self.gradient, self.lb, self.ub,
                                          self.A, self.b, self.binary_indicator,
                                          smoothness_coef=self.smoothness_coefficient)
        x, x_h, f_val_hist, step_size = solver.solve_optimization_problem()
        self.assertEqual(x.shape[0], self.q.shape[0])
        self.assertEqual(x.shape[1], self.k_max)

    def test_hopfield_step_type_classic(self):
        solver = HopfieldSolver(max_iterations=self.k_max, step_type='classic')
        solver.setup_optimization_problem(self.objective_function, self.gradient, self.lb, self.ub,
                                          self.A, self.b, self.binary_indicator,
                                          smoothness_coef=self.smoothness_coefficient)
        x, x_h, f_val_hist, step_size = solver.solve_optimization_problem()
        self.assertEqual(x.shape[0], self.q.shape[0])
        self.assertEqual(x.shape[1], self.k_max)

    def test_hopfield_step_type_armijo(self):
        solver = HopfieldSolver(max_iterations=self.k_max, step_type='armijo')
        solver.setup_optimization_problem(self.objective_function, self.gradient, self.lb, self.ub,
                                          self.A, self.b, self.binary_indicator,
                                          smoothness_coef=self.smoothness_coefficient)
        x, x_h, f_val_hist, step_size = solver.solve_optimization_problem()
        self.assertEqual(x.shape[0], self.q.shape[0])
        self.assertEqual(x.shape[1], self.k_max)

    def test_hopfield_with_absorption(self):
        solver = HopfieldSolver(max_iterations=self.k_max, step_type='classic', absorption_criterion=True)
        solver.setup_optimization_problem(self.objective_function, self.gradient, self.lb, self.ub,
                                          self.A, self.b, self.binary_indicator,
                                          smoothness_coef=self.smoothness_coefficient)
        x, x_h, f_val_hist, step_size = solver.solve_optimization_problem()
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
        self.objective_function = lambda x: 1 / 2 * np.dot(np.dot(x.T, self.H), x) + np.dot(self.q.T, x)
        self.gradient = lambda x: np.dot(self.H, x) + self.q
        self.smoothness_coefficient = utils.smoothness_coefficient(self.H)
        self.n = 2
        self.x = np.ones((self.n, self.k_max))

        self.ascent_stop_criterion = 0.1

    def test_create_initial_ascent_ascent(self):
        solver = HopfieldSolver(max_iterations=self.k_max)
        solver.setup_optimization_problem(self.objective_function, self.gradient, self.lb, self.ub,
                                          self.A, self.b, self.binary_indicator,
                                          smoothness_coef=self.smoothness_coefficient)
        self.assertTrue(np.array_equal(self.x_0.shape, solver._compute_x_0(self.x_0).shape))

    def test_create_initial_ascent_binary_neutral_ascent(self):
        solver = HopfieldSolver(max_iterations=self.k_max, initial_ascent_type='binary_neutral_ascent')
        solver.setup_optimization_problem(self.objective_function, self.gradient, self.lb, self.ub,
                                          self.A, self.b, self.binary_indicator,
                                          smoothness_coef=self.smoothness_coefficient)
        self.assertTrue(np.array_equal(self.x_0.shape, solver._compute_x_0(self.x_0).shape))

    def test_activation_function(self):
        x_0 = self.lb + (self.ub - self.lb) / 2
        beta = 0.5 * x_0
        activation_types = ['pwl', 'exp', 'sin', 'identity', 'tanh']

        for activation_type in activation_types:
            solver = HopfieldSolver(max_iterations=self.k_max, activation_type=activation_type)
            solver.setup_optimization_problem(self.objective_function, self.gradient, self.lb, self.ub,
                                              self.A, self.b, self.binary_indicator, x_0=x_0, beta=beta,
                                              smoothness_coef=self.smoothness_coefficient)
            self.assertTrue(np.array_equal(x_0, solver._activation(x_0, self.lb, self.ub)))

    def test_solve_dual_gradient_ascent(self):
        solver = HopfieldSolver(max_iterations=self.k_max)
        solver.setup_optimization_problem(self.objective_function, self.gradient, self.lb, self.ub,
                                          self.A, self.b, self.binary_indicator,
                                          smoothness_coef=self.smoothness_coefficient)
        dual_variable_init = np.ones((3 * self.n, 1))
        solver._solve_dual_gradient_ascent(dual_variable_init)


#
#     def test_compute_binary_absorption_mask(self):
#         x_0 = 0.5 * np.ones(self.n)
#         sol = hop.compute_binary_absorption_mask(x_0, self.lb, self.ub, self.binary_indicator)
#         self.assertTrue(np.array_equal(sol, 2 * x_0))
#
#     def test_smoothness_coefficient(self):
#         output = hop.smoothness_coefficient(self.H)
#         max_eigen_values = 2
#         self.assertEqual(output, max_eigen_values)
#
#     def test_alpha_hop(self):
#         grad_f = np.ones(2)
#         direction = np.ones(2)
#         k = 1
#         smoothness_coef = 1
#         direction_type = 'binary'
#         activation_type = 'sin'
#         output = hop.alpha_hop(self.x[:, k], grad_f, direction, k, self.lb, self.ub, smoothness_coef, self.beta,
#                                direction_type, activation_type)
#         self.assertEqual(output, 0)
#
#
# class TestHopfieldUpdate(unittest.TestCase):
#     def setUp(self):
#         self.n = 2
#         self.alpha = 1
#         self.direction = 1
#         self.beta = 0.5 * np.ones(2)
#         self.ub = np.array([1, 1])
#         self.lb = np.array([0, 0])
#
#     def test_hopfield_update(self):
#         activation_type = ['pwl', 'exp', 'sin', 'identity', 'tanh']
#         solution = [[1, 1], [0.88843492, 0.88843492], [0.99874749, 0.99874749], [1., 1.], [0.95257413, 0.95257413]]
#
#         x_h = np.ones(self.n)
#
#         for i in range(len(activation_type)):
#             self.assertTrue(np.array_equal(
#                 np.round(hop.hopfield_update(x_h, self.lb, self.ub, self.alpha, self.direction, self.beta,
#                                              activation_type[i])[0], decimals=8),
#                 solution[i]))
#
#
# class TestFindDirection(unittest.TestCase):
#     def setUp(self):
#         self.n = 2
#         self.H = np.array([[2, 0], [0, 1]])
#         self.q = np.array([-2.7, -1.8])
#         self.k_max = 20
#         self.binary_indicator = np.array([0, 1])
#         self.beta = np.ones(2)
#         self.ub = np.array([1, 1])
#         self.lb = np.array([0, 0])
#         self.x = np.ones((self.n, self.k_max))
#         self.x_0 = self.lb + (self.ub - self.lb) / 2
#         self.smoothness_coef = np.max(np.linalg.eigvals(self.H))
#         # self.grad_f = np.dot(self.H, self.x) + self.q
#
#     # def test_find_direction_type_classic(self):
#     #     activation_type = 'pwl'
#     #     direction_type = 'classic'
#     #     hop.find_direction(self.x, self.grad_f, self.lb, self.ub, self.binary_indicator, self.beta, direction_type, self.absorption, self.gamma, self.theta,
#     #                    activation_type)
#     #     pass
#     #
#     # def test_find_direction_type_stochastic(self):
#     #     activation_type = 'pwl'
#     #     direction_type = 'stochastic'
#     #     hop.find_direction(self.x, self.grad_f, self.lb, self.ub, self.binary_indicator, self.beta, direction_type,
#     #                        self.absorption, self.gamma, self.theta,
#     #                        activation_type)
#     #     pass
#     #
#     # def test_find_direction_type_binary(self):
#     #     activation_type = 'pwl'
#     #     direction_type = 'binary'
#     #     hop.find_direction(self.x, self.grad_f, self.lb, self.ub, self.binary_indicator, self.beta, direction_type,
#     #                        self.absorption, self.gamma, self.theta,
#     #                        activation_type)
#     #     pass
#     #
#     # def test_find_direction_type_soft_binary(self):
#     #     activation_type = 'pwl'
#     #     direction_type = 'soft binary'
#     #     hop.find_direction(self.x, self.grad_f, self.lb, self.ub, self.binary_indicator, self.beta, direction_type,
#     #                            self.absorption, self.gamma, self.theta,
#     #                            activation_type)
#     #     pass
#
#
# class TestStoppingCriterion(unittest.TestCase):
#     def setUp(self):
#         self.n = 2
#         self.H = np.array([[2, 0], [0, 1]])
#         self.q = np.array([-2.7, -1.8])
#         self.k_max = 20
#         self.binary_indicator = np.array([0, 1])
#         self.beta = np.ones(2)
#         self.ub = np.array([1, 1])
#         self.lb = np.array([0, 0])
#         self.x = np.ones((self.n, self.k_max))
#         self.x_0 = self.lb + (self.ub - self.lb) / 2
#         self.activation_type_list = ['pwl', 'exp', 'sin', 'identity', 'tanh']
#         self.stopping_criterion_type = 'gradient'
#         self.precision_stopping_criterion = 10 ^ -6
#
#     def test_stopping_criterion_met(self):
#         # TODO(Mathilde): Find a better solution to test all of that
#         grad_f = np.dot(self.H, self.x[:, 0]) + self.q
#         for activation_type in self.activation_type_list:
#             hop.stopping_criterion_met(self.x[:, 0], self.lb, self.ub, self.beta, activation_type, grad_f, 0,
#                                        self.k_max,
#                                        self.stopping_criterion_type, self.precision_stopping_criterion)
