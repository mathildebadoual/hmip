import numpy as np
import hmip.utils as utils
import math


# TODO(Mathilde): find another name fot L and H -> should be lower case
def hopfield(H, q, lb, ub, binary_indicator,
             k_max=None, absorption_criterion=None, gamma=0.9, theta=0.1, ascent_stop_criterion=0.1,
             precision_stopping_criterion=10 ^ -6,
             x_0=None, beta=None, alpha=None,
             step_type='classic', direction_type='binary',
             activation_type='sin', initial_ascent_type='binary_neutral_ascent',
             stopping_criterion_type='gradient'):
    """

    Solves the following optimization problem by computing the Hopfield method
        min f(x) = 1/2 x^T * H * x + q^T * x
            st lb <= x <= ub
            x_i \in {lb_i, ub_i} if binary_indicator_i == 1

    :param H: (size(x), size(x)) matrix of the optimization problem
    :param q: (size(x), 1) matrix of the optimization problem
    :param lb: (size(x), 1) matrix of the optimization problem
    :param ub: (size(x), 1) matrix of the optimization problem
    :param binary_indicator: (size(x), 1) vector of value 1 if the corresponding x is in {0, 1} and 0 otherwise.
    :param k_max: (default = 0) integer
    :param absorption: (default = None) float
    :param gamma:
    :param theta:
    :param x_0: initial state
    :param beta: (default = None) array of size x, if it is None, it will be ones(size(x))
    :param step_type: (string)
    :param direction_type: (string)
    :param activation_type: (string)
    :param initial_ascent_type (string):
            - 'no_ascent'               (takes default value at the center of the box [lb,ub] if no initial type is given)
            - 'ascent'                  (classic gradient ascent with step size 1/L)
            - 'binary_neutral_ascent'   (gradient ascent for non binary components, binary states are taken at the center)

    :return: x, x_h, f_val_hist, step_size

        x: (n, size(x)) vector of the optimal solution at each update
        x_h: (n, size(x)) vector of the hidden variable at each update
        f_val_hist: (n, 1) vector of the function f at each update
        step_size: (n, 1) vector of the step size at each update
    """

    n = np.size(q)

    # initialization of the Hopfield vector
    x = np.ones((n, k_max))
    x_h = np.ones((n, k_max))
    f_val_hist = np.ones(k_max)
    step_size = np.ones(k_max)

    # check if matrix is symmetric
    H = utils.make_symmetric(H)

    # Assess convexity of matrix H
    convexity = utils.assess_convexity_of_objective(H)

    # check if absorption value is strictly larger than ascent stop
    ascent_stop_criterion = utils.adapt_ascent_stop_criterion(ascent_stop_criterion, absorption_criterion)

    if beta is None:
        beta = np.ones(n)

    smoothness_coef = smoothness_coefficient(H)

    x0 = initial_state(H, q, lb, ub, binary_indicator, k_max, smoothness_coef, x_0, initial_ascent_type,
                       ascent_stop_criterion)

    x[:, 0] = x0
    x_h[:, 0] = inverse_activation(x0, lb, ub, beta, activation_type)
    f_val_hist[0] = objective_function(x[:, 0], H, q)
    k = 0
    grad_f = np.dot(H, x[:, k]) + q

    while not stopping_criterion_met(x[:, k], lb, ub, beta, activation_type, grad_f, k, k_max, stopping_criterion_type,
                                     precision_stopping_criterion):
        # gradient
        grad_f = np.dot(H, x[:, k]) + q

        # Make sure initial point is not stuck at gradient=0 at the initial point
        if k is 0 and np.linalg.norm(grad_f) == 0:
            grad_f = (smoothness_coef/10)*(np.random.rand(n)-0.5)
            print(grad_f)

        # direction
        direction = find_direction(x[:, k], grad_f, lb, ub, binary_indicator, beta, direction_type,
                                   absorption_criterion, gamma,
                                   theta, activation_type)
        # update hidden values
        # TODO(Mathilde): remove the for loop (more efficient matrix product)
        if step_type == 'armijo':
            alpha = np.divide(np.linalg.norm(grad_f), smoothness_coef)
            f_val_hist[k + 1] = f_val_hist[k] + 1
            prox_dist = proxy_distance_vector(x[:, k], lb, ub, beta, activation_type)
            while f_val_hist[k + 1] > f_val_hist[k] + alpha * np.dot(np.multiply(prox_dist, grad_f).T, direction):
                x[:, k + 1], x_h[:, k + 1] = hopfield_update(x_h[:, k], lb, ub, alpha, direction, beta, activation_type)
                f_val_hist[k + 1] = objective_function(x[:, k + 1], H, q)
                alpha = alpha / 2
            step_size[k] = 2 * alpha

        else:
            if alpha is None:
                alpha = alpha_hop(x[:, k], grad_f, direction, k, lb, ub, smoothness_coef, beta, direction_type,
                                  activation_type)
            x[:, k + 1], x_h[:, k + 1] = hopfield_update(x_h[:, k], lb, ub, alpha, direction, beta, activation_type)
            step_size[k] = alpha
            f_val_hist[k + 1] = objective_function(x[:, k + 1], H, q)

        if absorption_criterion is not None:
            x[:, k + 1] = absorb_solution_to_limits(x[:, k + 1], ub, lb, absorption_criterion)

        k += 1

    print('Candidate solution found with %s number of iterations.' % k)
    return x, x_h, f_val_hist, step_size


def smoothness_coefficient(H):
    """
    Compute the soothness coefficient with max(eig(H))
    :param H: (np.array) matrix of size (n, n), quadratic term of the problem
    :return: (np.float) scalar, smoothness coefficient
    """
    return np.absolute(np.max(np.linalg.eigvals(H)))


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


def absorb_solution_to_limits(x, ub, lb, absorption_criterion):
    for i in range(len(x)):
        if min(x[i] - lb[i], ub[i] - x[i]) < absorption_criterion:
            if x[i] + 1 / 2 * (lb[i] - ub[i]) < 0:
                x[i] = lb[i]
            else:
                x[i] = ub[i]
    return x


def initial_state(H, q, lb, ub, binary_indicator, k_max, smoothness_coeff, x_0,
                  initial_ascent_type, ascent_stop_criterion):
    if x_0 is None or not utils.is_in_box(x_0, ub, lb):
        # x_0 = lb + (ub - lb) / 2
        x_0 = lb + (ub - lb) / 2
        grad_f = 1;

    if initial_ascent_type is 'ascent':
        k = 0
        while k < k_max and utils.is_in_box(x_0, ub - ascent_stop_criterion, lb + ascent_stop_criterion) \
                and np.linalg.norm(grad_f) > 10 ^ -6:
            grad_f = np.dot(H, x_0) + q
            x_0 = x_0 + (1 / smoothness_coeff) * grad_f
            k = k + 1
        x_0 = activation(x_0, lb + ascent_stop_criterion, ub - ascent_stop_criterion, np.ones(len(x_0)),
                         activation_type='pwl')


    elif initial_ascent_type is 'binary_neutral_ascent':
        k = 0
        while k < k_max and utils.is_in_box(x_0, ub - ascent_stop_criterion, lb + ascent_stop_criterion) and np.linalg.norm(grad_f) > 10 ^ -6:
            grad_f = np.dot(H, x_0) + q
            x_0 = x_0 + (1 / smoothness_coeff) * np.multiply(grad_f, np.ones(len(x_0)) - binary_indicator)
            k = k + 1
        x_0 = activation(x_0, lb + ascent_stop_criterion, ub - ascent_stop_criterion, np.ones(len(x_0)),
                         activation_type='pwl')

    return x_0


def find_direction(x, grad_f, lb, ub, binary_indicator, beta, direction_type, absorption_criterion, gamma, theta,
                   activation_type):
    """

    :param x:
    :param grad_f:
    :param lb:
    :param ub:
    :param binary_indicator:
    :param direction_type:
    :param absorption_criterion:
    :param gamma:
    :param theta:
    :param beta:
    :param activation_type:
    :return:
    """
    n = np.size(x)
    binary_absorption_mask = compute_binary_absorption_mask(x, lb, ub, binary_indicator)

    # classic gradient
    if direction_type is 'classic' or direction_type is 'stochastic':
        if not absorption_criterion:
            direction = - grad_f
        else:
            direction = - np.multiply(binary_absorption_mask, grad_f)

        if direction_type is 'stochastic':
            # TODO(Mathilde): make 0.3 as a parameter
            direction = - np.multiply(direction, (np.random.uniform(0, 1, n) - 0.3))

    # binary gradient related direction methods
    # TODO(Mathilde): check with paper + Bertrand - change name of variables
    elif direction_type is 'binary' or direction_type is 'soft_binary':
        if direction_type is 'soft_binary':
            b = np.multiply(activation(x, lb, ub, beta, activation_type=activation_type) + 1 / 2 * (lb - ub),
                            binary_indicator)
            h = - grad_f
        elif direction_type is 'binary':
            b = np.multiply(np.sign(x + 1 / 2 * (lb - ub)), binary_indicator)
            h = - grad_f

        g = - np.multiply(proxy_distance_vector(x, lb, ub, beta, activation_type=activation_type), grad_f)

        if absorption_criterion:
            b = np.multiply(binary_absorption_mask, b)
            h = np.multiply(binary_absorption_mask, h)

        b = utils.normalize_array(b)
        h = utils.normalize_array(h)
        g = utils.normalize_array(g)

        w = gamma * b + (1 - gamma) * h
        y = max(0, - np.dot(g.T, w) + math.atan(theta) * np.sqrt(np.linalg.norm(w) ** 2 - np.dot(g.T, w) ** 2))

        direction = w + y * g

    else:
        print('direction type does not exist -> automatically set direction type to pwl')
        direction = find_direction(x, grad_f, lb, ub, binary_indicator, beta, direction_type='classic',
                                   absorption_criterion=absorption_criterion, gamma=gamma, theta=theta,
                                   activation_type=activation_type)

    # normalize
    direction = utils.normalize_array(direction)

    return direction


def proxy_distance_vector(x, lb, ub, beta, activation_type):
    """
    Compute the binary proxy distance sigma'(sigma^(-1)(x))

    :param x:
    :param lb:
    :param ub:
    :param beta:
    :param activation_type:
    :return:
    """

    # TODO(BERTRAND): Check and rectify that function
    # z = np.divide((x - lb), (ub - lb))
    z = x
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

# TODO(Bertrand): not a priority, but the coefficient 12 in the denominator could be improved, by choosing the best
#  constant for each activation function type

def alpha_hop(x, grad_f, direction, k, lb, ub, smoothness_coef, beta, direction_type, activation_type):
    sigma = proxy_distance_vector(x, lb, ub, beta, activation_type)

    denominator = smoothness_coef * np.linalg.norm(np.multiply(beta, direction)) ** 2 + 12 * np.dot(np.power(
        np.multiply(beta, direction), 2), np.absolute(grad_f))
    numerator = - np.dot(np.multiply(sigma, grad_f), direction)

    alpha = np.divide(numerator, denominator)

    if direction_type is 'stochastic':
        alpha = (1 - 1 / np.sqrt(k)) * alpha + 1 / (smoothness_coef * np.sqrt(k))

    return alpha


def hopfield_update(x_h, lb, ub, alpha, direction, beta, activation_type):
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
    x = activation(x_h, lb, ub, beta, activation_type)
    return x, x_h


def activation(x, lb, ub, beta, activation_type):
    """

    :param x:
    :param lb:
    :param ub:
    :param beta:
    :param activation_type:
    :return:
    """
    z = np.divide((x - lb), (ub - lb))
    if activation_type is 'pwl':
        return lb + np.multiply(ub - lb, utils.activation_pwl(z, beta))
    if activation_type is 'exp':
        return lb + np.multiply(ub - lb, utils.activation_exp(z, beta))
    if activation_type is 'sin':
        return lb + np.multiply(ub - lb, utils.activation_sin(z, beta))
    if activation_type is 'identity':
        return lb + np.multiply(ub - lb, utils.activation_pwl(z, beta))
    if activation_type is 'tanh':
        return lb + np.multiply(ub - lb, utils.activation_tanh(z, beta))


def inverse_activation(x, lb, ub, beta, activation_type):
    """

    :param x:
    :param lb:
    :param ub:
    :param beta:
    :param activation_type:
    :return:
    """
    z = np.divide((x - lb), (ub - lb))
    if activation_type is 'pwl':
        return utils.inverse_activation_pwl(z, beta)
    if activation_type is 'exp':
        return utils.inverse_activation_exp(z, beta)
    if activation_type is 'sin':
        return utils.inverse_activation_sin(z, beta)
    if activation_type is 'identity':
        return utils.inverse_activation_pwl(z, beta)
    if activation_type is 'tanh':
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


def stopping_criterion_met(x, lb, ub, beta, activation_type, gradf, k, kmax, stopping_criterion_type,
                           precision_stopping_criterion):
    if k >= kmax - 1:
        return True
    else:
        if stopping_criterion_type is 'gradient' and np.linalg.norm(
                np.multiply(gradf, proxy_distance_vector(x, lb, ub, beta,
                                                         activation_type=activation_type))) < precision_stopping_criterion:
            return True
        else:
            return False
