import matplotlib.pyplot as plt
import numpy as np
import hmip.hopfield as hop
import visualizations.utils_visuals as visuals


H = np.array([[2, 0.4], [0, 2]])
q = np.array([-1, -1])
k_max = 10000
binary_indicator = np.array([1, 0])
ub = np.array([1, 2])
lb = np.array([0.5, 0.2])

x, x_h, f_val_hist, step_size = hop.hopfield(H, q, lb, ub, binary_indicator, k_max=k_max)

visuals.plot_evolution_objective_function_2d(H, q, x, lb, ub, k_max, 'objective_function_2d.png')

visuals.plot_value_function(f_val_hist, 'value_function.png')

visuals.plot_step_size(step_size,  'step_size.png')
