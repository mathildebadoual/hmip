import numpy as np
import hmip.utils as utils
import math

DEFAULT_ACTIVATION_TYPE = 'pwl'
DEFAULT_INITIAL_ASCENT_TYPE = 'classic'
DEFAULT_STEP_TYPE = 'classic'
DEFAULT_DIRECTION_TYPE = 'classic'


# TODO(Mathilde): find another name fot L and H -> should be lower case
def hopfield(H, q, lb, ub, binary_indicator, L,
             k_max=0, absorption_val=1, beta=None,
             initial_ascent=False, absorption=False,
             step_type=DEFAULT_STEP_TYPE, direction_type=DEFAULT_DIRECTION_TYPE,
             activation_type=DEFAULT_ACTIVATION_TYPE):
    """

    Solves the following optimization problem by computing the Hopfield method
        min f(x) = 1/2 x^T * H * x + q^T * x
            st lb <= x <= ub
            x_i \in {0, 1}^n if binary_indicator_i == 1

    :param H: (size(x), size(x)) matrix of the optimization problem
    :param q: (size(x), 1) matrix of the optimization problem
    :param lb: (size(x), 1) matrix of the optimization problem
    :param ub: (size(x), 1) matrix of the optimization problem
    :param binary_indicator: (size(x), 1) vector of value 1 if the corresponding x is in {0, 1} and 0 otherwise.
    :param L:
    :param k_max: (default = 0) integer
    :param beta: (default = None) array of size x, if it is None, it will be ones(size(x))
    :param initial_ascent: (default = False) boolean
    :param absorption: (default = False) boolean
    :param absorption_val: (default = 1) float
    :param step_type: (string)
    :param direction_type: (string)
    :param activation_type: (string)

    :return: x, x_h, f_val_hist, step_size

        x: (n, size(x)) vector of the optimal solution at each update
        x_h: (n, size(x)) vector of the hidden variable at each update
        f_val_hist: (n, 1) vector of the function f at each update
        step_size: (n, 1) vector of the step size at each update
    """

    n = np.size(q)

    # initialization of the hopfield vectors
    x = np.ones((n, k_max))
    x_h = np.ones((n, k_max))
    f_val_hist = np.ones(k_max)
    step_size = np.ones(k_max)

    if initial_ascent:
        x0 = initial_ascent(H, q, lb, ub, binary_indicator, L)
    else:
        x0 = np.zeros(n)

    x[:, 0] = x0
    x_h[:, 0] = inverse_activation(x0, lb, ub, beta=beta, activation_type=activation_type)
    f_val_hist[0] = objective_function(x[:, 0], H, q)

    for k in range(k_max - 1):
        # gradient
        grad_f = np.dot(H, x[:, k]) + q
        # direction
        direction = find_direction(x[:, k], grad_f, lb, ub, binary_indicator, direction_type, absorption)

        # update hidden values
        # TODO(Mathilde): remove the for loop (more efficient matrix product)
        if step_type == 'armijo':
            alpha = np.linalg.norm(grad_f) / L
            f_val_hist[k + 1] = f_val_hist[k] + 1
            prox_dist = proxy_distance_vector(x[:, k], lb, ub, beta=beta, activation_type=activation_type)
            while f_val_hist[k + 1] > f_val_hist[k] + alpha * np.dot(np.multiply(prox_dist, grad_f).T, direction):
                x[:, k + 1], x_h[:, k + 1] = hopfield_update(x_h[:, k], lb, ub, alpha, direction, beta, activation_type)
                f_val_hist[k + 1] = objective_function(x[:, k + 1], H, q)
                alpha = alpha / 2
            # why this?
            step_size[k] = 2 * alpha

        else:
            # alpha = alpha_hop(x[:, k], grad_f, direction, k, lb, ub, L)
            alpha = 0.1
            x[:, k + 1], x_h[:, k + 1] = hopfield_update(x_h[:, k], lb, ub, alpha, direction, beta=beta,
                                                         activation_type=activation_type)
            step_size[k] = alpha
            f_val_hist[k + 1] = objective_function(x[:, k + 1], H, q)

        if absorption:
            for i in range(n):
                if min(x[i, k + 1] - lb[i], ub[i] - x[i, k + 1]) < absorption_val:
                    if x[i, k + 1] + 1 / 2 * (lb[i] - ub[i]) < 0:
                        x[i, k + 1] = lb[i]
                    else:
                        x[i, k + 1] = ub[i]

    return x, x_h, f_val_hist, step_size


def objective_function(x, H, q):
    """
    Compute the value of the objective function

        f(x) = 1/2 x^T * H * x + q^T * x

    :param x:
    :param H:
    :param q:
    :return:
    """
    return 1 / 2 * np.dot(np.dot(x.T, H), x) + np.dot(q.T, x)


# def initial_ascent(H, q, lb, ub, binary_indicator, L, initial_ascent_type=DEFAULT_INITIAL_ASCENT_TYPE):
#     n = len(q)
#     ascent = True
#
#     if len(binary_indicator) < n:
#         while ascent:
#             grad_f = H * x0 + f;
#             if strcmp(options.initial_ascent, 'ascent') and ascent:
#                 if norm(gradf) == 0
#                     gradf = L * rand(n, 1) / 10;
#                 end
#                 x0 = segment_projection(x0 + gradf / L, lb + options.ascent_boundary, ub - options.ascent_boundary);
#                 if min(x0 - lb) == options.ascent_boundary | | min(ub - x0) == options.ascent_boundary
#                     ascent = 'no';
#                 end
#
#             elseif
#             strcmp(options.initial_ascent, 'binary_neutral_ascent') & & strcmp(ascent, 'yes')
#             if norm((1 - binary_indicator). * gradf) == 0
#                 gradf = L * rand(n, 1) / 10;
#             end
#             x0 = segment_projection(x0 + (1 - binary_indicator). * gradf / L, lb + options.ascent_boundary,
#                                     ub - options.ascent_boundary);
#             if min(x0 - lb) == options.ascent_boundary | | min(ub - x0) == options.ascent_boundary
#                 ascent = 'no';
#
#     return x0


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
        if not absorption:
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
    Compute the binary proxy distance sigma'(sigma^(-1)(x))

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
def alpha_hop(x, lb, ub, grad_f, direction, k, L, beta, direction_type):
    sigma = proxy_distance_vector(x, lb, ub)
    scale = L * np.norm(np.multiply(beta, direction)) ** 2 + 12 * np.power(np.multiply(beta, direction),
                                                                           2).T * np.absolute(grad_f)
    alpha = - np.multiply(sigma, grad_f).T * direction / scale

    if direction_type == 'stochastic':
        alpha = (1 - 1 / np.sqrt(k)) * alpha + 1 / (L * np.sqrt(k))

    return alpha


def hopfield_update(x_h, lb, ub, alpha, direction, beta=None, activation_type=DEFAULT_ACTIVATION_TYPE):
    """
    Compute the explicit discretization of HNN for one step.

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
    binary_absorption_mask = np.ones(n)
    for i in range(n):
        if binary_indicator[i]:
            if x[i] == ub[i] or x[i] == lb[i]:
                binary_absorption_mask[i] = 0
    return binary_absorption_mask
