import numpy as np


def is_in_box(x, ub, lb):
    if np.all(np.greater(ub, x)) and np.all(np.greater(x, lb)):
        return True
    else:
        return False


def proxy_distance_vector_tanh(x, beta):
    """

    :param x: (np.array) variable
    :param beta: (np.array) size of x, parameter of the function
    :return:
    """
    tanh = 4 * np.multiply(np.multiply(beta, x), (1 - x))
    return tanh


def proxy_distance_vector_pwl(x, beta):
    """

    :param x: (np.array) variable
    :param beta: (np.array) size of x, parameter of the function
    :return:
    """
    pwl = np.zeros(len(x))
    for i in range(len(x)):
        if 0 < x[i] < 1:
            pwl[i] = beta[i]
    return pwl


def proxy_distance_vector_sin(x, beta):
    """

    :param x: (np.array) variable
    :param beta: (np.array) size of x, parameter of the function
    :return:
    """
    sin = 2 * np.multiply(np.multiply(beta, np.sqrt(x)), np.sqrt(1 - x))
    return sin


def proxy_distance_vector_exp(x, beta):
    """

    :param x: (np.array) variable
    :param beta: (np.array) size of x, parameter of the function
    :return:
    """
    exp = np.multiply(beta, np.minimum(1 - x, x))
    return exp


def proxy_distance_vector_identity(x, beta=None):
    """

    :param x: (np.array) variable
    :param beta: (np.array) size of x, parameter of the function
    :return:
    """
    id = np.zeros(len(x))
    return id


# TODO(Mathilde): find a librairy with the activation functions -> keras?


def activation_tanh(x, beta):
    """

    :param x: (np.array) variable
    :param beta: (np.array) size of x, parameter of the function
    :return:
    """
    tanh = 1 / 2 * (np.tanh(2 * np.multiply(beta, (x - 1 / 2))) + 1)
    return tanh


def activation_pwl(x, beta):
    """

    :param x: (np.array) variable
    :param beta: (np.array) size of x, parameter of the function
    :return:
    """
    pwl = np.maximum(np.zeros(len(x)),
                     np.minimum(np.ones(len(x)), np.multiply(beta, (x - 1 / 2)) + 1 / 2 * np.ones(len(x))))
    return pwl


def activation_sin(x, beta):
    """

    :param x: (np.array) variable
    :param beta: (np.array) size of x, parameter of the function
    :return:
    """
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
    """

    :param x: (np.array) variable
    :param beta: (np.array) size of x, parameter of the function
    :return:
    """
    exp = np.zeros(len(x))
    for i in range(len(x)):
        if x[i] > 1 / 2:
            exp[i] = 1 - np.exp(2 * beta[i] * (0.5 - x[i]) - np.log(2))
        else:
            exp[i] = np.exp(2 * beta[i] * (x[i] - 1 / 2) - np.log(2))
    return exp


def activation_identity(x, beta=None):
    """

    :param x: (np.array) variable
    :param beta: (np.array) size of x, parameter of the function
    :return:
    """
    return x


def normalize_array(array):
    """

    :param array:
    :return:
    """
    norm = np.linalg.norm(array)
    if norm != 0:
        array = array / norm
    return array


def inverse_activation_pwl(x, beta):
    """

    :param x: (np.array) variable
    :param beta: (np.array) size of x, parameter of the function
    :return:
    """
    pwl = np.ones(len(x))
    for i in range(len(x)):
        if 0 <= x[i] <= 1:
            pwl[i] = (beta[i]) ** (-1) * (x[i] - 1 / 2) + 1 / 2
        elif x[i] < 0:
            pwl[i] = 0
    return pwl


def inverse_activation_tanh(x, beta):
    """

    :param x: (np.array) variable
    :param beta: (np.array) size of x, parameter of the function
    :return:
    """
    tanh = 2 * np.multiply(np.linalg.inv(beta), np.arctanh(2 * x - 1)) + 1 / 2
    return tanh


def inverse_activation_sin(x, beta):
    """
    compute the inverse of the activation function sin
    :param x: (np.array) variable
    :param beta: (np.array) size of x, parameter of the function
    :return: inverse of the activation function sin
    """
    sin = np.ones(len(x))
    for i in range(len(x)):
        if 0 <= x[i] <= 1:
            sin[i] = (1 / (beta[i] * 2)) * np.arcsin(2 * x[i] - 1) + 1 / 2
        elif x[i] < 0:
            sin[i] = 0
    return sin


def inverse_activation_exp(x, beta):
    """

    :param x: (np.array) variable
    :param beta: (np.array) size of x, parameter of the function
    :return:
    """
    exp = np.ones(len(x))
    for i in range(len(x)):
        if 0 <= x[i] < 1 / 2:
            exp[i] = 1 / 2 + (1 / (2 * beta[i])) * np.log(2 * x[i])
        elif x[i] <= 0:
            exp[i] = 0
        else:
            exp[i] = 1 / 2 - (1 / (2 * beta[i])) * np.log(2 * (1 - x[i]))
    return exp


def inverse_activation_identity(x, beta=None):
    """

    :param x: (np.array) variable
    :param beta: (np.array) size of x, parameter of the function
    :return:
    """
    return x


def check_type(n, H=None, q=None, lb=None, ub=None, binary_indicator=None, L=None, k_max=None,
               absorption_val=None, gamma=None, theta=None, initial_state=None,
               beta=None, absorption=None,
               step_type=None, direction_type=None,
               activation_type=None, initial_ascent_type=None):
    """

    :param n: (integer)
    :param H: (numpy.ndarray)
    :param q: (numpy.ndarray)
    :param lb: (numpy.ndarray)
    :param ub: (numpy.ndarray)
    :param binary_indicator: (numpy.ndarray)
    :param L: (numpy.ndarray)
    :param k_max: (integer)
    :param absorption_val: (float)
    :param gamma: (float)
    :param theta: (float)
    :param initial_state: (numpy.ndarray)
    :param beta: (numpy.ndarray)
    :param absorption: (boolean)
    :param step_type: (string)
    :param direction_type: (string)
    :param activation_type: (string)
    :param initial_ascent_type: (string)
    :return:
    """

    # TODO(Mathilde): Print messages if not condition?

    if initial_state is not None:
        if isinstance(initial_state, np.ndarray):
            return len(initial_state) == n
        else:
            return False

    if H is not None:
        if isinstance(H, np.ndarray):
            return H.shape == (n, n)
        else:
            return False

    if q is not None:
        if isinstance(q, np.ndarray):
            return len(q) == n
        else:
            return False

    if lb is not None:
        if isinstance(lb, np.ndarray):
            return len(lb) == n
        else:
            return False

    if ub is not None:
        if isinstance(ub, np.ndarray):
            return len(ub) == n
        else:
            return False

    if binary_indicator is not None:
        if isinstance(binary_indicator, np.ndarray):
            for i in binary_indicator:
                if i != 0 and i != 1:
                    return False
            return len(binary_indicator) == n
        else:
            return False

    # TODO(Mathilde): check this one with Bertrand
    if L is not None:
        if isinstance(L, float):
            return True
        else:
            return False

    if absorption_val is not None:
        return isinstance(absorption_val, float)

    if gamma is not None:
        return isinstance(gamma, float)

    if theta is not None:
        return isinstance(theta, float)

    if beta is not None:
        if isinstance(beta, np.ndarray):
            return len(beta) == n
        else:
            return False

    if k_max is not None:
        return isinstance(k_max, float)

    if absorption is not None:
        return isinstance(ub, bool)

    # TODO(Mathilde): Maybe check if they are in possible options?
    if step_type is not None:
        return isinstance(step_type, str)

    if direction_type is not None:
        return isinstance(direction_type, str)

    if activation_type is not None:
        return isinstance(activation_type, str)

    if initial_ascent_type is not None:
        return isinstance(initial_ascent_type, str)

    print('Add a variable to check')
    return None


def check_symmetric(H):
    if not np.allclose(H, H.T, atol=0):
        H_new = 0.5 * (H + H.T)
        print('Specified matrix H was not symmetric, matrix H has been replaced by 0.5(H+H.transpose) ')
        return H_new
    else:
        return H


def check_ascent_stop(ascent_stop, absorption):
    if absorption is not None and ascent_stop is not None and ascent_stop <= absorption:
        ascent_stop = absorption * 2
        print('Choice of initial ascent stopping criterion was smaller than the '
              'chosen absorption value, ascent_stop was taken to be absorption * 2')
    return ascent_stop
