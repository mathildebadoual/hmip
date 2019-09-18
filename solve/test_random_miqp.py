import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
import pandas as pd
import time

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import hmip


def main():
    num_tests = 10
    for i in range(num_tests):
        index = i
        name = 'problem' + str(i)
        solver = hmip.HopfieldSolver()
        problem, H, q = generate_problem(solver)
        t = time.perf_counter()
        x, x_h, f_val_hist, step_size, other_dict = solver.solve(problem)
        t = time.perf_counter() - t
        save_stats(problem, x, x_h, f_val_hist, step_size, t, other_dict, index)
        save_plot(name, x, problem, H, q)


def save_stats(problem, x, x_h, f_val_hist, step_size, time, other_dict, index):
    stop_index = x.shape[1]
    for i in range(x.shape[1]):
        if np.isnan(x[:, i]).any():
            stop_index = i
            break
    x_refactor = x[:, :stop_index]
    print(x_refactor[:, -1])

    A_eq = problem['A_eq']
    b_eq = problem['b_eq']
    A_ineq = problem['A_ineq']
    b_ineq = problem['b_ineq']

    d = {}

    d['f_value'] = f_val_hist[x_refactor.shape[1]-1]
    #norm_eq = np.linalg.norm(np.dot(A_eq, x[-1]) - b_eq, ord=2)
    #norm_ineq = np.linalg.norm(np.max(0,
    #                                  np.dot(A_ineq, x[-1]) - b_ineq),
    #                           ord=2)
    d['norm_eq'] = 0
    d['norm_ineq'] = 0
    d['binary'] = np.sum(np.multiply(x_refactor[:, -1], (1 - x_refactor[:, -1])))
    d['time'] = time
    df = pd.DataFrame(data=d, index=[index])

    with open('stats.csv', 'a') as f:
        if index == 0:
            df.to_csv(f, header=True)
        else:
            df.to_csv(f, header=False)


def save_plot(name_fig, x, problem, H, q):
    ub = problem['ub']
    lb = problem['lb']
    A_eq = problem['A_eq']
    A_ineq = problem['A_ineq']
    b_eq = problem['b_eq']
    b_ineq = problem['b_ineq']

    def objective_function_2d(x_1, x_2):
        return 1 / 2 * (H[0, 0] * x_1**2 + H[1, 1] * x_2**2 +
                        2 * H[0, 1] * x_1 * x_2) + q[0] * x_1 + q[1] * x_2

    x_1 = np.linspace(-0.1 + lb[0], ub[0] + 0.1, num=500).reshape((1, -1))
    x_2 = np.linspace(-0.1 + lb[1], ub[1] + 0.1, num=500).reshape((1, -1))
    x_meshgrid_1, x_meshgrid_2 = np.meshgrid(x_1, x_2)

    objective = objective_function_2d(x_meshgrid_1, x_meshgrid_2)

    plt.figure(figsize=(10, 8))
    plt.contourf(x_meshgrid_1, x_meshgrid_2, objective, 50, cmap='plasma')
    plt.colorbar()

    if A_eq is not None and b_eq is not None:
        t_eq = np.linspace(-0.1 + lb[0], ub[0] + 0.1, num=500)
        plt.plot(t_eq, ((b_eq[0] - t_eq * A_eq[0, 0]) / A_eq[0, 1]), 'white')

    if A_ineq is not None and b_ineq is not None:
        t_ineq = np.linspace(-0.1 + lb[0], ub[0] + 0.1, num=500)
        plt.plot(t_ineq, ((b_ineq[0] - t_ineq * A_ineq[0, 0]) / A_ineq[0, 1]), 'yellow')

    plt.plot(x[0, :], x[1, :], 'black')
    plt.plot(x[0, 0], x[1, 0], 'bo', markersize=6, color='white')
    plt.plot(x[0, - 1], x[1, - 1],
             'x', markersize=10, color='white')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(name_fig)
    plt.xlim((lb[0] - 0.1, ub[0] + 0.1))
    plt.ylim((lb[1] - 0.1, ub[1] + 0.1))
    fig_path = 'plots/' + name_fig
    plt.savefig(fig_path)


def problem_0(solver):
    H = np.array([
    [1, 1],
    [1, 10]
    ])
    q = np.array([-1, -6])

    def objective_function(x):
        return 1 / 2 * x.T @ H @ x + q.T @ x

    def gradient(x):
        return H @ x + q

    binary_indicator = np.array([0, 0])
    ub = np.array([1, 1])
    lb = np.array([0, 0])

    A_eq = np.array([[-3, 2]])
    b_eq = np.array([0.5])

    A_ineq = np.array([[-3, 2]])
    b_ineq = np.array([0.5])

    penalty_eq = 10
    penalty_ineq = 10

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
            penalty_ineq=penalty_ineq)

    return problem, H, q


def generate_problem(solver, num_variables=2, beta=0.7, sparsity=0.6):
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

    B = scipy.sparse.random(num_variables, num_variables, density=sparsity).todense()
    S = B.T @ B
    M, v = np.linalg.eig(S)
    L = np.min(v)
    S = S - L * np.identity(num_variables)
    q = np.random.multivariate_normal(np.zeros(num_variables), S)

    def objective_function(x):
        return 1 / 2 * x.T @ H @ x + q.T @ x

    def gradient(x):
        return np.dot(H, x).reshape((len(x),)) + q

    penalty_eq = 10
    penalty_ineq = 10


    ub = np.ones(num_variables)
    lb = np.zeros(num_variables)

    problem = solver.setup_optimization_problem(
            objective_function,
            gradient,
            lb,
            ub,
            binary_indicator,
            A_eq=None,
            b_eq=None,
            A_ineq=None,
            b_ineq=None,
            penalty_eq=penalty_eq,
            penalty_ineq=penalty_ineq)

    return problem, H, q


if __name__ == '__main__':
    main()
