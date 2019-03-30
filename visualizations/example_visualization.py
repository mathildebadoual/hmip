import numpy as np
from hmip.hopfield import HopfieldSolver
import visualizations.utils_visuals as visuals
import hmip.utils as utils
import matplotlib.pyplot as plt

H = np.array([[1, 1], [1, 10]])
# H = np.array([[0, 0], [0, 0]])
q = np.array([1, 6])
# q = np.array([0, 0])
k_max = 1000
binary_indicator = np.array([1, 0])
ub = np.array([1, 1])
lb = np.array([0, 0])
H = utils.make_symmetric(H)
smoothness_coefficient = utils.smoothness_coefficient(H)
beta = np.array([10.0, 1.0])
A = np.array([[1, 0.2], [0, 0]])
b = np.array([0.5, 0])
n = 2


def objective_function(x):
    return 1 / 2 * np.dot(np.dot(x.T, H), x) + np.dot(q.T, x)


def gradient(x):
    return np.dot(x, H) + q


activation_type = 'sin'

solver_dual = HopfieldSolver(max_iterations=k_max, activation_type=activation_type)
solver_dual.setup_optimization_problem(objective_function, gradient, lb, ub, A, b, binary_indicator,
                                       smoothness_coef=smoothness_coefficient, beta=beta)
dual_variable, penalty, list_dual = solver_dual._solve_dual_gradient_ascent(np.zeros(2), 1)
plt.plot(list_dual)
print(dual_variable, penalty)

# dual_variable = 100 * np.ones(2)


def inequality_constraint(z):
    return np.dot(A, z) - b


def dual_function(x, dual_variable, penalty):
    return objective_function(x) + np.dot(dual_variable.T, inequality_constraint(x)) + penalty / 2 * np.linalg.norm(
        inequality_constraint(x), 2)


def new_objective_function(x):
    return dual_function(x, dual_variable, penalty)


def new_gradient(x):
    return gradient(x) + np.dot(A.T, dual_variable) + penalty * np.dot(A.T, inequality_constraint(x))


solver = HopfieldSolver(max_iterations=k_max, activation_type=activation_type)
solver.setup_optimization_problem(new_objective_function, new_gradient, lb, ub, A, b, binary_indicator,
                                  smoothness_coef=smoothness_coefficient, beta=beta)

x, x_h, f_val_hist, step_size = solver.solve_optimization_problem()

visuals.plot_evolution_objective_function_2d(H, q, x, lb, ub, k_max, 'objective_function_2d.png', A=A, b=b,
                                             dual=dual_variable, penalty=penalty)

for i in range(x.shape[1]):
    if np.isnan(x[:, i]).all():
        print(x[:, i-1])
        break

visuals.plot_value_function(f_val_hist, 'value_function.png')
visuals.plot_step_size(step_size, 'step_size.png')
