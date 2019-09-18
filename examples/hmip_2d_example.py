import numpy as np
import matplotlib.pyplot as plt

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import hmip

solver = hmip.HopfieldSolver()

H = np.array([
    [1, 1],
    [1, 10]
])
q = np.array([-1, -6])

binary_indicator = np.array([1, 0])

ub = np.array([1, 1])
lb = np.array([0, 0])

A_eq = np.array([[3, -2]])
b_eq = np.array([-0.5])

A_ineq = np.array([[3, -2]])
b_ineq = np.array([-0.5])

penalty_eq = 1
penalty_ineq = 1


def objective_function(x):
    return 1 / 2 * x.T @ H @ x + q.T @ x


def gradient(x):
    return H @ x + q


H = hmip.utils.make_symmetric(H)
smoothness_coefficient = hmip.utils.smoothness_coefficient(H)
print('true smooth coef', smoothness_coefficient)

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
    smoothness_coef=None,
    penalty_eq=penalty_eq,
    penalty_ineq=penalty_ineq)

x, x_h, f_val_hist, step_size, other_dict = solver.solve(problem)


def plot_2d(H, q, x, lb, ub, A_eq=None, b_eq=None, A_ineq=None, b_ineq=None):
    def objective_function_2d(x_1, x_2, H, q):
        return 1 / 2 * (H[0, 0] * x_1 ** 2 + H[1, 1] * x_2 ** 2 + 2 * H[0, 1] * x_1 * x_2) + q[0] * x_1 + q[1] * x_2

    x_1 = np.linspace(-0.1 + lb[0], ub[0] + 0.1, num=500).reshape((1, -1))
    x_2 = np.linspace(-0.1 + lb[1], ub[1] + 0.1, num=500).reshape((1, -1))
    x_meshgrid_1, x_meshgrid_2 = np.meshgrid(x_1, x_2)

    objective = objective_function_2d(x_meshgrid_1, x_meshgrid_2, H, q)

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
    plt.xlim((lb[0] - 0.1, ub[0] + 0.1))
    plt.ylim((lb[1] - 0.1, ub[1] + 0.1))
    plt.show()

plot_2d(H, q, x, lb, ub, A_eq=A_eq, b_eq=b_eq, A_ineq=None, b_ineq=None)

