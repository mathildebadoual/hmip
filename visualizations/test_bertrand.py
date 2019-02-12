import matplotlib.pyplot as plt
import numpy as np
import hmip.hopfield as hop
from visualizations.utils_visuals import plot_evolution_objective_function_2d


H = np.array([[2.5, 2.1], [1, 2]])
q = np.array([-1.4, -0.5])
k_max = 300
binary_indicator = np.array([1, 0])
ub = np.array([1, 1])
lb = np.array([0, 0])

x, x_h, f_val_hist, step_size = hop.hopfield(H, q, lb, ub, binary_indicator, k_max=k_max, absorption=0.1, initial_ascent_type='binary_neutral_ascent')

plot_evolution_objective_function_2d(H, q, x, k_max, 'plots/objective_function_2d.png')
