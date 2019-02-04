import matplotlib.pyplot as plt
import numpy as np
import hmip.hopfield as hop
from visualizations.utils_visuals import plot_evolution_objective_function_2d


H = np.array([[1, 1], [1, 1]])
q = np.array([-1, -1])
k_max = 100
binary_indicator = np.array([0, 1])
ub = np.array([1, 1])
lb = np.array([0, 0])

x, x_h, f_val_hist, step_size = hop.hopfield(H, q, lb, ub, binary_indicator, k_max=k_max)

plot_evolution_objective_function_2d(H, q, x, 'plots/objective_function_2d.png')
