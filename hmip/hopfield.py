import numpy as np
import hmip.utils as utils
import math

DEFAULT_ACTIVATION_TYPE = 'pwl'


# TODO(Mathilde): find another name fot L -> should be lower case
def hopfield(H, q, lb, ub, binary_indicator, L, k_max=0, beta=None, step_type='classic', initial_ascent=False,
             absorption=False,
             absorption_val=1, direction_type='classic', activation_type=DEFAULT_ACTIVATION_TYPE):
    """Solves the following optimization problem by computing the Hopfield method
        min f(x) = 1/2 x^T * H * x + q^T * x
            st lb <= x <= ub
            x_i \in {0, 1}^n if binary_indicator_i == 1

        inputs:
        H, q, lb, ub, binary_indicator, L, k_max=0, step_type='classic', initial_ascent=False, absorption=False,
        absorption_val=1

        H: (size(x), size(x)) matrix of the optimization problem
        q: (size(x), 1) matrix of the optimization problem
        lb: (size(x), 1) matrix of the optimization problem
        ub: (size(x), 1) matrix of the optimization problem
        binary_indicator: (size(x), 1) vector of value 1 if the corresponding x is in {0, 1} and 0 otherwise.
        L:
        k_max: (default = 0) integer
        step_size: (default = 'classic') string
        initial_ascent: (default = False) boolean
        absorption: (default = False) boolean
        absorption_val: (default = 1) float

        outputs:
        x, x_h, f_val_hist, step_size

        x: (n, size(x)) vector of the optimal solution at each update
        x_h: (n, size(x)) vector of the hidden variable at each update
        f_val_hist: (n, 1) vector of the function f at each update
        step_size: (n, 1) vector of the step size at each update
    """

    n = np.size(q)

    # initialization of the hopfield hidden vectors
    x = np.ones(n, k_max)
    x_h = np.ones(n, k_max)
    f_val_hist = np.ones(k_max)
    step_size = np.ones(k_max)

    if initial_ascent:
        x0 = create_initial_ascent(H, q, lb, ub, binary_indicator, L)
    else:
        x0 = 0

    x[:, 0] = x0
    x_h[:, 0] = inverse_activation(x0, lb, ub, beta=beta, activation_type=activation_type)
    f_val_hist[0] = 0.5 * np.dot(np.dot(x[:, 0].T, q), x[:, 0]) + np.dot(q.T, x[:, 0])

    for k in range(k_max):
        # gradient
        grad_f = np.dot(H, x[:, k]) + q
        # direction
        direction = find_direction(x[:, k], grad_f, lb, ub, binary_indicator, direction_type, absorption)

        # update hidden values
        # TODO(Mathilde): remove the for loop (more efficient matrix product)
        if step_type == 'classic':
            alpha = (x[:, k], grad_f, direction, k, lb, ub, L)
            x[:, k + 1], x_h[:, k + 1] = hopfield_update(x_h[:, k], lb, ub, alpha, direction, beta=beta,
                                                         activation_type=activation_type)
            step_size[k] = alpha
            f_val_hist[k + 1] = 0.5 * np.dot(np.dot(x[:, k + 1].T, H), x[:, k + 1]) + np.dot(h.T, x[:, k + 1])

        elif step_type == 'armijo':
            alpha = np.linalg.norm(grad_f) / L
            f_val_hist[k + 1] = f_val_hist[k] + 1
            prox_dist = proxy_distance_vector(x[:, k], lb, ub, beta=beta, activation_type=activation_type)
            while f_val_hist[k + 1] > f_val_hist[k] + alpha * np.dot(np.multiply(prox_dist, grad_f).T, direction):
                x[:, k + 1], x_h[:, k + 1] = hopfield_update(x_h[:, k], lb, ub, alpha, direction, beta, activation_type)
                f_val_hist[k + 1] = 0.5 * np.dot(np.dot(x[:, k + 1].T, H), x[:, k + 1]) + np.dot(q.T, x[:, k + 1])
                alpha = alpha / 2
        else:
            print('step_size of the wrong type')
            # raise error about the step_type

        step_size[k] = 2 * alpha

        # absorption
        if absorption:
            for i in range(n):
                if min(x[i, k + 1] - lb[i], ub[i] - x[i, k + 1]) < absorption_val:
                    if x[i, k + 1] + 0.5 * (lb[i] - ub[i]) < 0:
                        x[i, k + 1] = lb[i]
                    else:
                        x[i, k + 1] = ub[i]

    return x, x_h, f_val_hist, step_size


def create_initial_ascent(H, q, lb, ub, binary_indicator, L):
    pass


def find_direction(x, grad_f, lb, ub, binary_indicator, direction_type='classic', absorption=False, gamma=0, theta=0,
                   beta=None, activation_type=DEFAULT_ACTIVATION_TYPE):
    """

    :param x:
    :param grad_f:
    :param lb:
    :param ub:
    :param binary_indicator:
    :param direction_type:
    :param absorption:
    :param gamma:
    :param theta:
    :param beta:
    :param activation_type:
    :return:
    """
    n = np.size(x)
    binary_absorption_mask = compute_binary_absorption_mask(x, lb, ub, binary_indicator)

    # classic gradient
    if direction_type == 'classic' or direction_type is 'stochastic':
        if absorption:
            direction = - grad_f
        else:
            direction = - np.multiply(binary_absorption_mask, grad_f)

        if direction_type is 'stochastic':
            # TODO(Mathilde): make 0.3 as a parameter
            direction = - np.multiply(direction, (np.random.uniform(0, 1, n) - 0.3))

    # binary gradient related direction methods
    # TODO(Mathilde): check with paper + Bertrand - change name of variables
    elif direction_type == 'binary' or direction_type == 'soft binary':
        if direction_type == 'soft binary':
            b = np.multiply(activation(x, lb, ub, activation_type=activation_type) + 0.5 * (lb - ub), binary_indicator)
            h = - grad_f
        elif direction_type == 'soft binary':
            b = np.multiply(np.sign(x + 0.5 * (lb - ub)), binary_indicator)
            h = - grad_f

        g = - np.multiply(proxy_distance_vector(x, lb, ub, beta, activation_type=activation_type), grad_f)

        if absorption:
            b = np.multiply(binary_absorption_mask, b)
            h = np.multiply(binary_absorption_mask, h)

        b = utils.normalize_array(b)
        h = utils.normalize_array(h)
        g = utils.normalize_array(g)

        w = gamma * b + (1 - gamma) * h
        y = np.max(0, - np.dot(g.T, w) + math.atan(theta) * np.sqrt(np.linalg.norm(w) ** 2 - np.dot(g.T, w) ** 2))

        direction = w + y * g

    else:
        print('direction type does not exist -> automatically set direction type to pwl')
        direction = find_direction(x, grad_f, lb, ub, binary_indicator, direction_type=direction_type,
                                   absorption=absorption, gamma=gamma, theta=theta, beta=beta,
                                   activation_type=DEFAULT_ACTIVATION_TYPE)

    # normalize
    direction = utils.normalize_array(direction)

    return direction


def proxy_distance_vector(x, lb, ub, beta=None, activation_type=DEFAULT_ACTIVATION_TYPE):
    """

    :param x:
    :param lb:
    :param ub:
    :param beta:
    :param activation_type:
    :return:
    """
    if beta is None:
        beta = np.ones(len(x))
    z = np.divide((x - lb), (ub - lb))
    if activation_type == 'pwl':
        return utils.proxy_distance_vector_pwl(z, beta)
    if activation_type == 'exp':
        return utils.proxy_distance_vector_exp(z, beta)
    if activation_type == 'sin':
        return utils.proxy_distance_vector_sin(z, beta)
    if activation_type == 'identity':
        return utils.proxy_distance_vector_pwl(z, beta)
    if activation_type == 'tanh':
        return utils.proxy_distance_vector_tanh(z, beta)


# TODO(Mathilde): remove the integers without justification
def compute_alpha_hop(x, lb, ub, grad_f, direction, k, L, beta, direction_type):
    sigma = proxy_distance_vector(x, lb, ub)
    scale = L * np.norm(np.multiply(beta, direction)) ** 2 + 12 * np.power(np.multiply(beta, direction),
                                                                           2).T * np.absolute(grad_f)
    alpha = - np.multiply(sigma, grad_f).T * direction / scale

    if direction_type == 'stochastic':
        alpha = (1 - 1 / np.sqrt(k)) * alpha + 1 / (L * np.sqrt(k))

    return alpha


def hopfield_update(x_h, lb, ub, alpha, direction, beta=None, activation_type=DEFAULT_ACTIVATION_TYPE):
    """

    :param x_h:
    :param lb:
    :param ub:
    :param alpha:
    :param direction:
    :param beta:
    :param activation_type:
    :return:
    """
    # update of the hidden states
    x_h = x_h + alpha * direction
    # update of the state
    x = activation(x_h, lb, ub, beta=beta, activation_type=activation_type)
    return x, x_h


# TODO(Mathilde): decide of which default activation
def activation(x, lb, ub, beta=None, activation_type=DEFAULT_ACTIVATION_TYPE):
    """

    :param x:
    :param lb:
    :param ub:
    :param beta:
    :param activation_type:
    :return:
    """
    if beta is None:
        beta = np.ones(len(x))
    z = np.divide((x - lb), (ub - lb))
    if activation_type == 'pwl':
        return utils.activation_pwl(z, beta)
    if activation_type == 'exp':
        return utils.activation_exp(z, beta)
    if activation_type == 'sin':
        return utils.activation_sin(z, beta)
    if activation_type == 'identity':
        return utils.activation_pwl(z, beta)
    if activation_type == 'tanh':
        return utils.activation_tanh(z, beta)


def inverse_activation(x, lb, ub, beta=None, activation_type=DEFAULT_ACTIVATION_TYPE):
    """

    :param x:
    :param lb:
    :param ub:
    :param beta:
    :param activation_type:
    :return:
    """
    if beta is None:
        beta = np.ones(len(x))
    z = np.divide((x - lb), (ub - lb))
    if activation_type == 'pwl':
        return utils.inverse_activation_pwl(z, beta)
    if activation_type == 'exp':
        return utils.inverse_activation_exp(z, beta)
    if activation_type == 'sin':
        return utils.inverse_activation_sin(z, beta)
    if activation_type == 'identity':
        return utils.inverse_activation_pwl(z, beta)
    if activation_type == 'tanh':
        return utils.inverse_activation_tanh(z, beta)


def compute_binary_absorption_mask(x, lb, ub, binary_indicator):
    """

    :param x:
    :param lb:
    :param ub:
    :param binary_indicator:
    :return:
    """
    n = np.size(x)
    binary_absorption_mask = np.ones(n, 1)
    for i in range(n):
        if binary_indicator[i]:
            if x[i] == ub[i] or x[i] == lb[i]:
                binary_absorption_mask[i] = 0
    return binary_absorption_mask
