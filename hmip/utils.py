import numpy as np


def proxy_distance_vector_tanh(x, beta):
    tanh = 4 * np.multiply(np.multiply(beta, x), (1 - x))
    return tanh


def proxy_distance_vector_pwl(x, beta):
    pwl = np.zeros(len(x))
    for i in range(len(x)):
        if 0 < x[i] < 1:
            pwl[i] = beta[i]
    return pwl


def proxy_distance_vector_sin(x, beta):
    sin = 2 * np.multiply(np.multiply(beta, np.sqrt(x)), np.sqrt(1 - x))
    return sin


def proxy_distance_vector_exp(x, beta):
    exp = np.multiply(beta, np.minimum(1 - x, x))
    return exp


def proxy_distance_vector_identity(x, beta=None):
    id = np.zeros(len(x))
    return id

# TODO(Mathilde): find a librairy with the activation functions -> keras?


def activation_tanh(x, beta):
    tanh = 1 / 2 * (np.tanh(2 * np.multiply(beta, (x - 1 /2))) + 1)
    return tanh


def activation_pwl(x, beta):
    pwl = np.maximum(np.zeros(len(x)), np.minimum(np.ones(len(x)), np.multiply(beta, (x - 1 / 2)) + 1 / 2 * np.ones(len(x))))
    return pwl


def activation_sin(x, beta):
    sin = np.zeros(len(x))
    for i in range(len(x)):
        if x[i] > 1 / 2 + np.pi / (4 * beta[i]):
            sin[i] = 1
        elif x[i] < 1 / 2 - np.pi / (4 * beta[i]):
            sin[i] = 0
        else:
            sin[i] = 1 / 2 * np.sin(2 * beta[i] * (x[i] - 1 / 2)) + 1 / 2
    return sin


def activation_exp(x, beta):
    exp = np.zeros(len(x))
    for i in range(len(x)):
        if x[i] > 1 / 2:
            exp[i] = 1 - np.exp(2 * beta[i] * (0.5 - x[i]) - np.log(2))
        else:
            exp[i] = np.exp(2 * beta[i] * (x[i] - 1 / 2) - np.log(2))
    return exp


def activation_identity(x, beta=None):
    return x


def normalize_array(array):
    norm = np.linalg.norm(array)
    if norm != 0:
        array = array / norm
    return array