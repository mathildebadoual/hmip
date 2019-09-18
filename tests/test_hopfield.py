import unittest
import numpy as np
import cvxpy as cvx

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from hmip.hopfield import HopfieldSolver
import hmip.utils as utils


class TestHopfield(unittest.TestCase):
    def setUp(self):
        self.H = np.array([[1, 1], [1, 10]])
        self.q = np.array([-1, -6])
        self.k_max = 20
        self.binary_indicator = np.array([1, 1])
        self.beta = 1
        self.ub = np.array([1, 1])
        self.lb = np.array([0, 0])
        self.A = np.array([[1, 2]])
        self.b = np.array([0.5])
        self.absorption = 1
        self.step_type = 'classic'
        self.penalty = 10
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

    def test_get_dual_variables_cvxpy(self):
        # test of the equality
        solver = HopfieldSolver()
        problem = solver.setup_optimization_problem(
            self.objective_function,
            self.gradient,
            self.lb,
            self.ub,
            self.binary_indicator,
            A_eq=self.A,
            b_eq=self.b,
            smoothness_coef=self.smoothness_coefficient,
            penalty_eq=self.penalty,
            penalty_ineq=self.penalty)
        dual_variables_eq, dual_variables_ineq = HopfieldSolver._get_dual_variables(solver, problem)
        dual_variables_eq_cvxpy = get_dual_variables_cvxpy_solver(self.H, self.q, self.lb, self.ub,
                A_eq=self.A, b_eq=self.b)
        self.assertTrue(abs(dual_variables_eq[0] - dual_variables_eq_cvxpy[0]) <= 0.1)

        # test of the inequality
        solver = HopfieldSolver()
        problem = solver.setup_optimization_problem(
            self.objective_function,
            self.gradient,
            self.lb,
            self.ub,
            self.binary_indicator,
            A_ineq=self.A,
            b_ineq=self.b,
            smoothness_coef=self.smoothness_coefficient,
            penalty_eq=self.penalty,
            penalty_ineq=self.penalty)
        dual_variables_eq, dual_variables_ineq = HopfieldSolver._get_dual_variables(solver, problem)
        dual_variables_ineq_cvxpy = get_dual_variables_cvxpy_solver(self.H, self.q, self.lb, self.ub,
                A_ineq=self.A, b_ineq=self.b)
        self.assertTrue(abs(dual_variables_ineq[0] - dual_variables_ineq_cvxpy[0]) <= 0.1)

        # test of all
        solver = HopfieldSolver()
        problem = solver.setup_optimization_problem(
            self.objective_function,
            self.gradient,
            self.lb,
            self.ub,
            self.binary_indicator,
            A_ineq=self.A,
            b_ineq=self.b,
            A_eq=self.A,
            b_eq=self.b,
            smoothness_coef=self.smoothness_coefficient,
            penalty_eq=self.penalty,
            penalty_ineq=self.penalty)
        dual_variables_eq, dual_variables_ineq = HopfieldSolver._get_dual_variables(solver, problem)
        dual_variables_eq_cvxpy, dual_variables_ineq_cvxpy = get_dual_variables_cvxpy_solver(self.H, self.q, self.lb, self.ub,
               A_ineq=self.A, b_ineq=self.b, A_eq=self.A, b_eq=self.b)

        self.assertTrue(abs(dual_variables_ineq[0] - dual_variables_ineq_cvxpy[0]) <= 0.1)
        self.assertTrue(abs(dual_variables_eq[0] - dual_variables_eq_cvxpy[0]) <= 0.2)


class TestOthers(unittest.TestCase):
    def setUp(self):
        self.H = np.array([[2, 0], [0, 1]])
        self.q = np.array([-2.7, -1.8])
        self.k_max = 20
        self.binary_indicator = np.array([0, 1])
        self.beta = 1
        self.ub = np.array([1, 1])
        self.lb = np.array([0, 0])
        self.A = np.array([[-1, 2], [0, 0]])
        self.b = np.array([-3, 0])
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


def get_dual_variables_cvxpy_solver(H, q, lb, ub,
        A_eq=None, A_ineq=None, b_eq=None, b_ineq=None, ):
    """
    Solves the same problem as hopfiel with cvxpy
    :param H:
    :param q:
    :param lb:
    :param ub:
    :param binary_indicator:
    :param solver: cvxpy solver
    :return:
    """
    n = q.shape[0]
    x = cvx.Variable(n)

    constraints = [lb <= x, x <= ub]

    if A_eq is not None and b_eq is not None:
        constraints += [A_eq * x - b_eq == 0]
    if A_ineq is not None and b_ineq is not None:
        n_ineq = len(b_ineq)
        s = cvx.Variable(n_ineq)
        constraints += [A_ineq * x - b_ineq - s == 0, s <= 0]

    objective = 1 / 2 * cvx.quad_form(x, H) + q.T * x
    objective = cvx.Minimize(objective)
    problem = cvx.Problem(objective, constraints)

    problem.solve()

    if A_eq is not None and b_eq is not None and A_ineq is not None and b_ineq is not None:
        return constraints[2].dual_value, constraints[3].dual_value
    elif (A_ineq is not None and b_ineq is not None) or (A_eq is not None and b_eq is not None):
        return constraints[2].dual_value
    else:
        return None


if __name__ == '__main__':
    unittest.main()
