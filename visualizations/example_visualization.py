import numpy as np
from hmip.hopfield import HopfieldSolver
import visualizations.utils_visuals as visuals
import hmip.utils as utils
import matplotlib.pyplot as plt
import cvxpy as cvx

H = np.array([[1, 1], [1, 10]])
# H = np.array([[0, 0], [0, 0]])
q = np.array([1, 6])
# q = np.array([0, 0])
k_max = 1000
binary_indicator = np.array([0, 0])
ub = np.array([1, 1])
lb = np.array([0, 0])
H = utils.make_symmetric(H)
smoothness_coefficient = utils.smoothness_coefficient(H)
beta = np.array([10.0, 1.0])
A_ineq = None
b_ineq = None
A_eq = None
b_eq = None
A_eq = np.array([[3, 1], [0, 0]])
b_eq = np.array([0.7, 0])
n = 2
activation_type = 'sin'
penalty_eq = 10
penalty_ineq = 10

# with cvxpy
x = cvx.Variable(n)
constraint = [A_eq * x - b_eq == 0, x <= 1, x >= 0]
objective = cvx.Minimize(cvx.quad_form(x, H) + q.T * x + penalty_eq * cvx.norm(A_eq * x - b_eq, 2))
problem = cvx.Problem(objective, constraint)
problem.solve()
print(constraint[0].dual_value)
print(x.value)


def objective_function(x):
    return 1 / 2 * np.dot(np.dot(x.T, H), x) + np.dot(q.T, x)


def gradient(x):
    return np.dot(x, H) + q


solver_dual = HopfieldSolver(max_iterations=k_max, activation_type=activation_type)

solver = HopfieldSolver(max_iterations=k_max, activation_type=activation_type)
solver.setup_optimization_problem(objective_function, gradient, lb, ub, binary_indicator, A_eq=A_eq, b_eq=b_eq,
                                  A_ineq=A_ineq, b_ineq=b_ineq,
                                  smoothness_coef=smoothness_coefficient, beta=beta, penalty_eq=penalty_eq,
                                  penalty_ineq=penalty_ineq)

x, x_h, f_val_hist, step_size, other_dict = solver.solve_optimization_problem()

visuals.plot_evolution_objective_function_2d(H, q, x, lb, ub, k_max, 'objective_function_2d.png', A=A_eq, b=b_eq)

for i in range(x.shape[1]):
    if np.isnan(x[:, i]).all():
        print(x[:, i - 1])
        print('equality:', np.dot(A_eq, x[:, i - 1]) - b_eq)
        # print('inequality:', np.dot(A_ineq, x[:, i - 1]) - b_ineq)
        break
    if i == 999:
        print(x[:, -1])
        print('equality:', np.dot(A_eq, x[:, -1]) - b_eq)
        break

visuals.plot_value_function(f_val_hist, 'value_function.png')
visuals.plot_step_size(step_size, 'step_size.png')
