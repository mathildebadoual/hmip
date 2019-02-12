import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def plot_evolution_objective_function_2d(H, q, x, k_max, figure_path):
    """
    Plot the evolution of the objective function for 2d variables at the different steps of the execution of hopfield()
    :param H: (np.array) size (n, n) from the problem formulation - quadratic parameter
    :param q: (np.array) size (n,) from the problem formulation - linear parameter
    :param x: (np.array) size (n, k_max) solution of the problem at each step of the gradient descent
    :param figure_path: (string) path where to solve the function
    :return: plot a function and save it in the path
    """
    x_1 = np.linspace(-0.1, 1.1, num=500).reshape((1, -1))
    x_2 = np.linspace(-0.1, 1.1, num=500).reshape((1, -1))
    x_meshgrid_1, x_meshgrid_2 = np.meshgrid(x_1, x_2)

    objective = objective_function_2d(x_meshgrid_1, x_meshgrid_2, H, q)
    plt.figure(figsize=(7, 5))
    plt.contourf(x_meshgrid_1, x_meshgrid_2, objective, 50, cmap='plasma')
    plt.colorbar()
    plt.plot(x[0, :], x[1, :], 'black')
    plt.plot(x[0, 0], x[1, 0],  'bo', markersize=6, color='white')
    plt.plot(x[0, k_max-1], x[1, k_max-1], 'x',  markersize=10, color='white')
    plt.plot([0, 0], [1, 0], 'white')
    plt.plot([0, 1], [0, 0], 'white')
    plt.plot([0, 1], [1, 1], 'white')
    plt.plot([1, 1], [1, 0], 'white')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.savefig('plots/' + figure_path)
    plt.show()


def objective_function_2d(x_1, x_2, H, q):
    """
    This function is the same cost function as in hopfield method but in a more practical form for meshgrid plots
    :param x_1:
    :param x_2:
    :param H:
    :param q:
    :return:
    """
    return 1 / 2 * (H[0, 0] * x_1 ** 2 + H[1, 1] * x_2 ** 2 + 2 * H[0, 1] * x_1 * x_2) + q[0] * x_1 + q[1] * x_2


def plot_value_function(f_val_hist, figure_path):
    plt.plot(f_val_hist)
    plt.xscale('log')
    plt.title('Value function for each iteration')
    plt.xlabel('Iterations (log)')
    plt.ylabel('Value function')
    plt.savefig('plots/' + figure_path)
    plt.grid()
    plt.show()
