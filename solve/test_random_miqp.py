import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
import pandas as pd
import time
import math
import gurobipy
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import hmip

TESTS = {
        1: {
            'activation_type': 'pwl',
            'absorption_criterion': None,
            'max_iterations': 500,
            'stopping_criterion_type': 'gradient',
            'direction_type': 'classic',
            'step_type': 'classic',
            'initial_ascent_type': 'binary_neutral_ascent',
            'precision_stopping_criterion': 10**(-6),
            'beta': 100,
            },
        2: {
            'activation_type': 'tanh',
            'absorption_criterion': None,
            'max_iterations': 500,
            'stopping_criterion_type': 'gradient',
            'direction_type': 'classic',
            'step_type': 'classic',
            'initial_ascent_type': 'binary_neutral_ascent',
            'precision_stopping_criterion': 10**(-6),
            'beta': 100,
            },
        3: {
            'activation_type': 'pwl',
            'absorption_criterion': None,
            'max_iterations': 500,
            'stopping_criterion_type': 'gradient',
            'direction_type': 'binary',
            'step_type': 'classic',
            'initial_ascent_type': 'binary_neutral_ascent',
            'precision_stopping_criterion': 10**(-6),
            'beta': 100,
            },
        4: {
            'activation_type': 'sin',
            'absorption_criterion': None,
            'max_iterations': 500,
            'stopping_criterion_type': 'gradient',
            'direction_type': 'binary',
            'step_type': 'classic',
            'initial_ascent_type': 'binary_neutral_ascent',
            'precision_stopping_criterion': 10**(-6),
            'beta': 100,
            },
        }


def test_without_constraints():
    num_tests = 100
    index = 0
    num_vars = [2, 4, 10, 100, 500, 1000, 5000, 10000]
    for test_info in TESTS.values():
        for num_var in num_vars:
            for i in range(num_tests):
                solver = hmip.HopfieldSolver(
                    activation_type=test_info['activation_type'],
                    absorption_criterion=test_info['absorption_criterion'],
                    stopping_criterion_type=test_info[
                        'stopping_criterion_type'],
                    direction_type=test_info['direction_type'],
                    step_type=test_info['step_type'],
                    initial_ascent_type=test_info['initial_ascent_type'],
                    precision_stopping_criterion=test_info[
                        'precision_stopping_criterion'],
                    beta=test_info['beta'],
                )

                dual_eq = None
                dual_ineq = None

                # solve with cplex
                problem, H, q = generate_problem(solver,
                                                 constraints=False,
                                                 num_variables=num_var)

                if num_var <= 100:
                    t = time.perf_counter()
                    x_cplex, f_cplex, dual_eq, dual_ineq = hmip.other_solvers.cvxpy_solver(
                        H,
                        q,
                        problem['lb'],
                        problem['ub'],
                        problem['binary_indicator'],
                        problem['A_eq'],
                        problem['b_eq'],
                        problem['A_ineq'],
                        problem['b_ineq'],
                        solver='CPLEX',
                        verbose=False)
                    t_cplex = time.perf_counter() - t
                else:
                    f_cplex = None
                    t_cplex = None

                t = time.perf_counter()
                x_cplex_relax, f_cplex_relax, _, _ = hmip.other_solvers.cvxpy_solver(
                    H,
                    q,
                    problem['lb'],
                    problem['ub'],
                    np.zeros(len(problem['binary_indicator'])),
                    problem['A_eq'],
                    problem['b_eq'],
                    problem['A_ineq'],
                    problem['b_ineq'],
                    solver='CPLEX',
                    verbose=False)
                t_cplex_relax = time.perf_counter() - t

                # solve with Hmip
                t = time.perf_counter()
                x, x_h, f_val_hist, step_size, other_dict = solver.solve(
                    problem)
                t_hmip = time.perf_counter() - t

                # save stats
                print('--- save stats ---')
                save_stats(solver, problem, x, x_h, f_val_hist, step_size,
                           t_hmip, other_dict, index, t_cplex, f_cplex,
                           t_cplex_relax, f_cplex_relax,
                           'stats_without_constraints')
                index += 1


def test_with_constraints():
    num_tests = 100
    index = 0
    num_vars = [2, 4, 10, 100]
    for test_info in TESTS.values():
        for num_var in num_vars:
            for i in range(num_tests):
                solver = hmip.HopfieldSolver(
                    activation_type=test_info['activation_type'],
                    absorption_criterion=test_info['absorption_criterion'],
                    stopping_criterion_type=test_info[
                        'stopping_criterion_type'],
                    direction_type=test_info['direction_type'],
                    step_type=test_info['step_type'],
                    initial_ascent_type=test_info['initial_ascent_type'],
                    precision_stopping_criterion=test_info[
                        'precision_stopping_criterion'],
                    beta=test_info['beta'],
                )

                dual_eq = None
                dual_ineq = None

                # solve with cplex
                while (dual_eq is None or dual_ineq is None):
                    print('TRY')
                    # chose a random problem
                    problem, H, q = generate_problem(solver,
                                                     constraints=True,
                                                     num_variables=num_var)

                    t = time.perf_counter()
                    x_cplex, f_cplex, dual_eq, dual_ineq = hmip.other_solvers.cvxpy_solver(
                        H,
                        q,
                        problem['lb'],
                        problem['ub'],
                        problem['binary_indicator'],
                        problem['A_eq'],
                        problem['b_eq'],
                        problem['A_ineq'],
                        problem['b_ineq'],
                        solver='CPLEX',
                        verbose=False)
                    t_cplex = time.perf_counter() - t
                    dual_eq = 1
                    dual_ineq = 1

                print('Found a feasible problem')

                t = time.perf_counter()
                x_cplex_relax, f_cplex_relax, _, _ = hmip.other_solvers.cvxpy_solver(
                    H,
                    q,
                    problem['lb'],
                    problem['ub'],
                    np.zeros(len(problem['binary_indicator'])),
                    problem['A_eq'],
                    problem['b_eq'],
                    problem['A_ineq'],
                    problem['b_ineq'],
                    solver='CPLEX',
                    verbose=False)
                t_cplex_relax = time.perf_counter() - t

                problem['dual_eq'] = dual_eq
                problem['dual_ineq'] = dual_ineq

                # solve with Hmip
                t = time.perf_counter()
                x, x_h, f_val_hist, step_size, other_dict = solver.solve(
                    problem)
                t_hmip = time.perf_counter() - t

                # save stats
                save_stats(solver, problem, x, x_h, f_val_hist, step_size,
                           t_hmip, other_dict, index, t_cplex, f_cplex,
                           t_cplex_relax, f_cplex_relax,
                           'stats_with_constraints')
                index += 1


def save_stats(solver, problem, x, x_h, f_val_hist, step_size, t_hmip,
               other_dict, index, t_cplex, f_cplex, t_cplex_relax,
               f_cplex_relax, csv_name):
    stop_index = x.shape[1]
    for i in range(x.shape[1]):
        if np.isnan(x[:, i]).any():
            stop_index = i
            break
    x_refactor = x[:, :stop_index]
    A_eq = problem['A_eq']
    b_eq = problem['b_eq']
    A_ineq = problem['A_ineq']
    b_ineq = problem['b_ineq']

    d = {}

    d['num_variables'] = problem['dim_problem']

    d['activation_function'] = solver.activation_type
    d['absorption_criterion'] = solver.absorption_criterion
    d['ascent_stop_criterion'] = solver.ascent_stop_criterion
    d['stopping_criterion_type'] = solver.stopping_criterion_type
    d['step_type'] = solver.step_type
    d['max_iterations'] = solver.max_iterations
    d['direction_type'] = solver.direction_type
    d['initial_ascent_type'] = solver.initial_ascent_type
    d['gamma'] = solver.gamma
    d['theta'] = solver.theta

    d['f_value'] = f_val_hist[x_refactor.shape[1] - 1]
    if csv_name == 'stats_with_constraints':
        d['norm_eq'] = np.linalg.norm(np.dot(A_eq, x_refactor[:, -1]) - b_eq,
                                      ord=2)
        d['norm_ineq'] = np.max(
            (0,
             np.linalg.norm(np.dot(A_ineq, x_refactor[:, -1]) - b_ineq,
                            ord=2)))
    d['binary'] = np.sum(
        np.multiply(x_refactor[:, -1], (1 - x_refactor[:, -1])))
    d['t_hmip'] = t_hmip
    d['t_cplex'] = t_cplex
    d['f_cplex'] = f_cplex

    d['t_cplex_relax'] = t_cplex_relax
    d['f_cplex_relax'] = f_cplex_relax

    df = pd.DataFrame(data=d, index=[index])

    with open(csv_name + '.csv', 'a') as f:
        if index == 0:
            df.to_csv(f, header=True)
        else:
            df.to_csv(f, header=False)


def generate_problem(solver, constraints=False, num_variables=2, beta=0.7, sparsity=0.6):
    """Generate random problems

    Args:
      num_variables: int
      beta: float in [0, 1] that is the percentage of binary variables
    """

    # binary indicator
    binary_indicator = np.zeros(num_variables)
    for i in range(num_variables):
        z = np.random.uniform()
        if z >= beta:
            binary_indicator[i] = 1

    # objective function
    A = scipy.sparse.random(num_variables, num_variables, density=sparsity).todense()
    V, _ = np.linalg.qr(A)
    d = np.random.uniform(0, 1, num_variables)
    D = np.diag(d)
    H = np.array(V.T @ D @ V)
    H = 0.5 * (H.T + H)

    B = scipy.sparse.random(num_variables, num_variables, density=sparsity).todense()
    S = B.T @ B
    M, v = np.linalg.eig(S)
    L = np.min(v)
    S = S - L * np.identity(num_variables)
    q = np.random.multivariate_normal(np.zeros(num_variables), S)

    gamma = 0.8
    Z = np.random.uniform(0, 1)
    num_eq = math.ceil(gamma * Z * num_variables)
    num_ineq = math.ceil(gamma * Z * num_variables)

    A_eq = np.random.uniform(0, 1, (num_eq, num_variables))
    A_ineq = np.random.uniform(0, 1, (num_ineq, num_variables))

    z = np.zeros(num_variables)
    for i, x_i in enumerate(binary_indicator):
        if x_i == 1:
            z[i] = np.random.binomial(1, 0.5)
        else:
            z[i] = np.random.uniform(0, 1)

    eps = np.random.uniform(0, 1, num_ineq) * 0.005

    b_eq = A_eq @ z
    b_ineq = A_ineq @ z + eps

    def objective_function(x):
        return 1 / 2 * x.T @ H @ x + q.T @ x

    def gradient(x):
        return np.dot(H, x).reshape((len(x),)) + q

    penalty_eq = 10
    penalty_ineq = 10

    ub = np.ones(num_variables)
    lb = np.zeros(num_variables)

    if not constraints:
        A_eq = None
        A_ineq = None
        b_eq = None
        b_ineq = None

    problem = solver.setup_optimization_problem(
            objective_function,
            gradient,
            lb,
            ub,
            binary_indicator,
            A_eq=A_eq,
            b_eq=b_eq,
            A_ineq=A_ineq,
            b_ineq=b_ineq,
            penalty_eq=penalty_eq,
            penalty_ineq=penalty_ineq,
            )

    return problem, H, q


if __name__ == '__main__':
    print('--- test without constraints ---')
    test_without_constraints()
    print('--- test with constraints ---')
    test_with_constraints()
