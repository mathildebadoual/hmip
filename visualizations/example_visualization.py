import autograd.numpy as npgrad  # Thinly-wrapped version of Numpy
from autograd import grad
import numpy as np
from hmip.hopfield import HopfieldSolver
import visualizations.utils_visuals as visuals
import hmip.utils as utils

H = np.array([[10, 1], [1, 40]])
# H = np.array([[0, 0], [0, 0]])
q = np.array([1, 6])
k_max = 1000
binary_indicator = np.array([0, 0])
ub = np.array([1, 1])
lb = np.array([0, 0])
H = utils.make_symmetric(H)
mu = 0.001 * np.ones(2)
smoothness_coefficient = utils.smoothness_coefficient(H)
smoothness_coefficient = 0.001
beta = np.array([10.0, 1.0])
A = np.array([[-1, -1], [0, 0]])
b = np.array([-0.5, 0])
Aeq = np.array([[1, 1], [0, 0]])
beq = np.array([0.5, 0])
rho = 15


# def objective_function(x):
#     return 1 / 2 * npgrad.dot(npgrad.dot(x.T, H), x) + npgrad.dot(q.T, x) + npgrad.dot(mu.T,
#                                                                                        npgrad.log(b - npgrad.dot(A, x)))


def objective_function(x):
    return 1 / 2 * np.dot(np.dot(x.T, H), x) + np.dot(q.T, x) - rho / 2 * (
                (b[0] - np.dot(A[0, :], x)).clip(0) + (b[1] - np.dot(A[1, :], x)).clip(0))


# gradient = grad(objective_function)


# def gradient(x):
#     return np.dot(x, H) + q - A[0, :] / (b[0] - np.dot(A[0, :], x)) - A[1, :] / (b[1] - np.dot(A[1, :], x))

def gradient(x):
    return np.dot(x, H) + q + rho / 2 * (A[0, :] + A[1, :]).clip(0)


activation_types = ['sin', 'tanh', 'exp', 'pwl', 'identity']

for activation_type in activation_types:
    solver = HopfieldSolver(max_iterations=k_max, activation_type=activation_type)
    solver.setup_optimization_problem(objective_function, gradient, lb, ub, A, b, binary_indicator,
                                      smoothness_coef=smoothness_coefficient, beta=beta)
    x, x_h, f_val_hist, step_size = solver.solve_optimization_problem()

    visuals.plot_evolution_objective_function_2d(H, q, x, lb, ub, k_max, 'objective_function_2d.png', A=A, b=b, rho=rho)
    # visuals.plot_value_function(f_val_hist, 'value_function.png')
    # visuals.plot_step_size(step_size, 'step_size.png')
