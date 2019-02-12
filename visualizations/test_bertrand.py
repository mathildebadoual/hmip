import matplotlib.pyplot as plt
import numpy as np
import hmip.hopfield as hop
import visualizations.utils_visuals as visuals


H = np.array([[3.1, 1.9], [1, 1]])
q = np.array([-1, -0.5])
k_max = 500
binary_indicator = np.array([0, 1])
ub = np.array([1, 1])
lb = np.array([0, 0])

x, x_h, f_val_hist, step_size = hop.hopfield(H, q, lb, ub, binary_indicator, k_max=k_max, initial_ascent_type='binary_neutral_ascent')

visuals.plot_evolution_objective_function_2d(H, q, x, k_max, 'objective_function_2d.png')

visuals.plot_value_function(f_val_hist, 'value_function.png')