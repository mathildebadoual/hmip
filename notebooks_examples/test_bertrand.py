import matplotlib.pyplot as plt
import numpy as np
import hmip.hopfield as hop


def objective_function(x_1, x_2, H, q):
    return 1 / 2 * (H[0, 0] * x_1 ** 2 + H[1, 1] * x_2 ** 2 + 2 * H[0, 1] * x_1 * x_2) + q[0] * x_1 + q[1] * x_2

H = np.array([[2, 0.1], [0, 1]])
q = np.array([-2.7, -1.8])
k_max = 100
binary_indicator = np.array([0, 1])
ub = np.array([1, 1])
lb = np.array([0, 0])

x, x_h, f_val_hist, step_size = hop.hopfield(H, q, lb, ub, binary_indicator, k_max=k_max)

x_1 = np.linspace(0, 1, num=500)
x_2 = np.linspace(0, 1, num=500)
X_1, X_2 = np.meshgrid(x_1, x_2)
Z = objective_function(X_1, X_2, H, q)

plt.figure(figsize=(7, 5))
plt.contourf(X_1, X_2, Z, 50, cmap='plasma')
plt.plot(x[0, :], x[1, :],'black')
plt.colorbar()
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()