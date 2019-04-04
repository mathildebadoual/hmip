import numpy as np
import hmip.utils as utils
import math
from scipy.optimize import minimize, Bounds
import cvxpy as cvx
from scipy.optimize import LinearConstraint


class HopfieldSolver():
    def __init__(self, activation_type='sin',
                 gamma=0.95, theta=0.05, ascent_stop_criterion=0.06, absorption_criterion=None, max_iterations=100,
                 stopping_criterion_type='gradient', direction_type='classic', step_type='classic',
                 initial_ascent_type='binary_neutral_ascent', precision_stopping_criterion=10 ** -6):

        self.activation_function = getattr(utils, 'activation_' + activation_type)
        self.inverse_activation_function = getattr(utils, 'inverse_activation_' + activation_type)
        self.proxy_distance_vector = getattr(utils, 'proxy_distance_vector_' + activation_type)
        self.ascent_stop_criterion = utils.adapt_ascent_stop_criterion(ascent_stop_criterion, absorption_criterion)
        self.stopping_criterion_type = stopping_criterion_type
        self.absorption_criterion = absorption_criterion
        self.initial_ascent_type = initial_ascent_type
        self.step_type = step_type
        self.direction_type = direction_type
        self.max_iterations = max_iterations
        self.precision_stopping_criterion = precision_stopping_criterion
        self.gamma = gamma
        self.theta = theta
        self.problem = None
        self.beta = None

    def setup_optimization_problem(self, objective_function, gradient, lb, ub,
                                   binary_indicator, A_eq=None, b_eq=None, A_ineq=None, b_ineq=None, x_0=None,
                                   smoothness_coef=None, beta=None, penalty_eq=0, penalty_ineq=0):
        # TODO: Find how to chose smoothness_coef when using barrier method
        self.problem = dict({'objective_function': objective_function, 'gradient': gradient, 'lb': lb, 'ub': ub,
                             'A_eq': A_eq, 'b_eq': b_eq, 'A_ineq': A_ineq, 'b_ineq': b_ineq,
                             'binary_indicator': binary_indicator,
                             'smoothness_coef': smoothness_coef, 'x_0': x_0, 'dim_problem': len(binary_indicator),
                             'penalty_eq': penalty_eq, 'penalty_ineq': penalty_ineq})
        if type(beta) == int:
            self.beta = beta * np.ones(self.problem['dim_problem'])
        elif beta is None:
            self.beta = np.ones(self.problem['dim_problem'])
        else:
            self.beta = beta
        self.problem['x_0'] = self._compute_x_0(x_0)

    def solve_optimization_problem(self):
        if self.problem is None:
            raise Exception('Problem is not set')

        x = np.nan * np.ones((self.problem['dim_problem'], self.max_iterations))
        x_h = np.nan * np.ones((self.problem['dim_problem'], self.max_iterations))
        f_val_hist = np.nan * np.ones(self.max_iterations)
        step_size = np.nan * np.ones(self.max_iterations)

        def gradient_wrt_slack_variable(variables):
            return None

        dual_variable_eq, dual_variable_ineq = self._solve_dual_gradient_ascent()

        # if the problem has constraints:
        if self.problem['A_ineq'] is not None and self.problem['b_ineq'] is not None and self.problem[
            'A_eq'] is not None and self.problem['b_eq'] is not None:

            def inequality_constraint(optimization_variable, slack_variable):
                return np.dot(self.problem['A_ineq'], optimization_variable) - self.problem['b_ineq'] - slack_variable

            def equality_constraint(optimization_variable):
                return np.dot(self.problem['A_eq'], optimization_variable) - self.problem['b_eq']

            def objective_function(variables):
                optimization_variable, slack_variable = variables
                main_function = self.problem['objective_function'](optimization_variable)
                inequality_term = np.dot(dual_variable_ineq.T,
                                         inequality_constraint(optimization_variable, slack_variable)) + self.problem[
                                      'penalty_ineq'] / 2 * np.linalg.norm(
                    inequality_constraint(optimization_variable, slack_variable), 2)
                equality_term = np.dot(dual_variable_eq.T, equality_constraint(optimization_variable)) + self.problem[
                    'penalty_eq'] / 2 * np.linalg.norm(equality_constraint(optimization_variable), 2)
                return main_function + inequality_term + equality_term

            def gradient(variables):
                optimization_variable, slack_variable = variables
                main_function = self.problem['gradient'](optimization_variable)
                equality_term = np.dot(self.problem['A_eq'].T, dual_variable_eq) + self.problem[
                    'penalty_eq'] * np.dot(self.problem['A_eq'].T, equality_constraint(optimization_variable))
                inequality_term = np.dot(self.problem['A_ineq'].T, dual_variable_ineq) + self.problem[
                    'penalty_ineq'] * np.dot(self.problem['A_ineq'].T,
                                             inequality_constraint(optimization_variable, slack_variable))
                return main_function + equality_term + inequality_term

            def gradient_wrt_slack_variable(variables):
                optimization_variable, slack_variable = variables
                return - self.problem['penalty_ineq'] * inequality_constraint(optimization_variable,
                                                                              slack_variable) - dual_variable_ineq

        elif self.problem['A_ineq'] is not None and self.problem['b_ineq'] is not None and (
                self.problem['A_eq'] is None or self.problem['b_eq'] is None):

            def inequality_constraint(optimization_variable, slack_variable):
                return np.dot(self.problem['A_ineq'], optimization_variable) - self.problem['b_ineq'] - slack_variable

            def objective_function(variables):
                optimization_variable, slack_variable = variables
                main_function = self.problem['objective_function'](optimization_variable)
                inequality_term = np.dot(dual_variable_ineq.T,
                                         inequality_constraint(optimization_variable, slack_variable)) + self.problem[
                                      'penalty_ineq'] / 2 * np.linalg.norm(
                    inequality_constraint(optimization_variable, slack_variable), 2)
                return main_function + inequality_term

            def gradient(variables):
                optimization_variable, slack_variable = variables
                main_function = self.problem['gradient'](optimization_variable)
                inequality_term = np.dot(self.problem['A_ineq'].T, dual_variable_ineq) + self.problem[
                    'penalty_ineq'] * np.dot(self.problem['A_ineq'].T,
                                             inequality_constraint(optimization_variable, slack_variable))
                return main_function + inequality_term

            def gradient_wrt_slack_variable(variables):
                optimization_variable, slack_variable = variables
                return - self.problem['penalty_ineq'] * inequality_constraint(optimization_variable,
                                                                              slack_variable) - dual_variable_ineq

        elif self.problem['A_eq'] is not None and self.problem['b_eq'] is not None and (
                self.problem['A_ineq'] is None or self.problem['b_ineq'] is None):

            def equality_constraint(optimization_variable):
                return np.dot(self.problem['A_eq'], optimization_variable) - self.problem['b_eq']

            def objective_function(variable):
                main_function = self.problem['objective_function'](variable)
                equality_term = np.dot(dual_variable_eq.T, equality_constraint(variable)) + self.problem[
                    'penalty_eq'] / 2 * np.linalg.norm(equality_constraint(variable), 2)
                return main_function + equality_term

            def gradient(variable):
                main_function = self.problem['gradient'](variable)
                equality_term = np.dot(self.problem['A_eq'].T, dual_variable_eq) + self.problem[
                    'penalty_eq'] * np.dot(self.problem['A_eq'].T, equality_constraint(variable))
                return main_function + equality_term

        else:
            def objective_function(variable):
                return self.problem['objective_function'](variable)

            def gradient(variable):
                return self.problem['gradient'](variable)

        x[:, 0] = self.problem['x_0']
        x_h[:, 0] = self._inverse_activation(self.problem['x_0'], self.problem['lb'], self.problem['ub'])
        if self.problem['A_ineq'] is not None and self.problem['b_ineq'] is not None:
            s = np.nan * np.ones((self.problem['dim_problem'], self.max_iterations))
            s[:, 0] = 0 * self.problem['x_0']
            f_val_hist[0] = objective_function((x[:, 0], s[:, 0]))
            grad_f = gradient((x[:, 0], s[:, 0]))
        else:
            grad_f = gradient(x[:, 0])
        if np.linalg.norm(grad_f) == 0:
            grad_f = (self.problem['smoothness_coef'] / 10) * (np.random.rand(self.problem['dim_problem']) - 0.5)
        k = 0

        while not self._stopping_criterion_met(x[:, k], grad_f, k):

            direction = self._find_direction(x[:, k], grad_f)
            # TODO(Mathilde): remove the for loop (more efficient matrix product)

            if self.step_type == 'armijo':
                alpha = np.divide(np.linalg.norm(grad_f), self.problem['smoothness_coef'])
                f_val_hist[k + 1] = f_val_hist[k] + 1
                prox_dist = self._proxy_distance_vector(x[:, k])
                while f_val_hist[k + 1] > f_val_hist[k] + alpha * np.dot(np.multiply(prox_dist, grad_f).T, direction):
                    x[:, k + 1], x_h[:, k + 1] = self._hopfield_update(x_h[:, k], alpha, direction)
                    if self.problem['A_ineq'] is not None and self.problem['b_ineq'] is not None:
                        s[:, k + 1] = np.min(np.zeros(len(s[:, k + 1])), s[:, k] - 1 / self.problem[
                            'penalty_ineq'] * gradient_wrt_slack_variable((x[:, k + 1], s[:, k])))
                        f_val_hist[k + 1] = objective_function((x[:, k + 1], s[:, k + 1]))
                        grad_f = gradient((x[:, k + 1], s[:, k + 1]))
                    else:
                        f_val_hist[k + 1] = objective_function(x[:, k + 1])
                        grad_f = gradient(x[:, k + 1])
                    alpha = alpha / 2
                step_size[k] = 2 * alpha

            else:
                alpha = self._alpha_hop(x[:, k], grad_f, k, direction)
                x[:, k + 1], x_h[:, k + 1] = self._hopfield_update(x_h[:, k], alpha, direction)
                if self.problem['A_ineq'] is not None and self.problem['b_ineq'] is not None:
                    s[:, k + 1] = s[:, k + 1] = np.min(np.zeros(len(s[:, k + 1])), s[:, k] - 1 / self.problem[
                        'penalty_ineq'] * gradient_wrt_slack_variable((x[:, k + 1], s[:, k])))
                    f_val_hist[k + 1] = objective_function((x[:, k + 1], s[:, k + 1]))
                    grad_f = gradient((x[:, k + 1], s[:, k + 1]))
                else:
                    f_val_hist[k + 1] = objective_function(x[:, k + 1])
                    grad_f = gradient(x[:, k + 1])
                    step_size[k] = alpha

            if self.absorption_criterion is not None:
                x[:, k + 1] = self._absorb_solution_to_limits(x[:, k + 1])

            k += 1

        print('Candidate solution found with %s number of iterations.' % k)
        if self.problem['A_ineq'] is not None and self.problem['b_ineq'] is not None:
            return x, x_h, f_val_hist, step_size, dict(
                {'slack_variable': s, 'dual_variable_eq': dual_variable_eq, 'dual_variable_ineq': dual_variable_eq})
        else:
            return x, x_h, f_val_hist, step_size, dict()

    def _solve_dual_gradient_ascent(self):
        n = self.problem['dim_problem']

        # create the dual function:
        if self.problem['A_ineq'] is not None and self.problem['b_ineq'] is not None and self.problem[
            'A_eq'] is not None and self.problem['b_eq'] is not None:
            lb = np.concatenate((self.problem['lb'].copy(), - np.inf * np.ones((n,))), axis=0)
            ub = np.concatenate((self.problem['ub'].copy(), np.zeros((n,))), axis=0)
            bounds = Bounds(lb, ub)

            def inequality_constraint(variables):
                return np.dot(self.problem['A_ineq'], variables[:n]) - self.problem['b_ineq'] - variables[n:]

            def equality_constraint(variables):
                return np.dot(self.problem['A_eq'], variables[:n]) - self.problem['b_eq']

            def dual_function(variables):
                return self.problem['objective_function'](variables[:n]) + self.problem['penalty_eq'] / 2 * np.linalg.norm(
                    equality_constraint(variables), 2) + self.problem['penalty_ineq'] / 2 * np.linalg.norm(
                    inequality_constraint(variables),
                    2)

            s_0 = np.zeros(n)
            x0 = np.concatenate((self.problem['x_0'], s_0))
            result = minimize(dual_function, x0, method='trust-constr', constraints=(
                {"fun": inequality_constraint, "type": "eq"}, {"fun": equality_constraint, "type": "eq"}),
                              options={'verbose': 1}, bounds=bounds)
            dual_variable_ineq = result.v[0]
            dual_variable_eq = result.v[1]

            print('dual_ineq = ', dual_variable_ineq)
            print('dual_eq = ', dual_variable_eq)

            print(result.x[n:])

            return dual_variable_eq, dual_variable_ineq

        elif self.problem['A_ineq'] is not None and self.problem['b_ineq'] is not None and (
                self.problem['A_eq'] is None or self.problem['b_eq'] is None):

            lb = np.concatenate((self.problem['lb'].copy(), - np.inf * np.ones((n,))), axis=0)
            ub = np.concatenate((self.problem['ub'].copy(), np.zeros((n,))), axis=0)
            bounds = Bounds(lb, ub)

            def inequality_constraint(variables):
                return np.dot(self.problem['A_ineq'], variables[:n]) - self.problem['b_ineq'] - variables[n:]

            def dual_function(variables):
                return self.problem['objective_function'](variables[:n]) + self.problem['penalty_ineq'] / 2 * np.linalg.norm(
                    inequality_constraint(variables),
                    2)

            s_0 = np.zeros(n)
            x0 = np.concatenate((self.problem['x_0'], s_0))
            result = minimize(dual_function, x0, method='trust-constr',
                              constraints=({"fun": inequality_constraint, "type": "eq"}),
                              options={'verbose': 1}, bounds=bounds)
            dual_variable_ineq = result.v[0]

            print('dual_ineq = ', dual_variable_ineq)

            print('s = ', result.x[n:])

            return None, dual_variable_ineq

        elif self.problem['A_eq'] is not None and self.problem['b_eq'] is not None and (
                self.problem['A_ineq'] is None or self.problem['b_ineq'] is None):
            bounds = Bounds(self.problem['lb'], self.problem['ub'])

            def equality_constraint(variable):
                return np.dot(self.problem['A_eq'], variable) - self.problem['b_eq']

            def dual_function(variable):
                return self.problem['objective_function'](variable) + self.problem['penalty_eq'] / 2 * np.linalg.norm(
                    equality_constraint(variable), 2)

            x0 = self.problem['x_0']
            result = minimize(dual_function, x0, method='trust-constr',
                              constraints=({"fun": equality_constraint, "type": "eq"}),
                              options={'verbose': 1, 'factorization_method':'SVDFactorization'}, bounds=bounds)
            dual_variable_eq = result.v[0] / 5

            # dual_variable_eq = np.array([-2, 0])

            print(result)

            print('dual_eq = ', result.v[0])

            return dual_variable_eq, None

        else:
            return None, None

    def _hopfield_update(self, x_h, alpha, direction):
        if self.problem is None:
            raise Exception('Problem is not set')
        x_h = x_h + alpha * direction
        x = self._activation(x_h, self.problem['lb'], self.problem['ub'])
        return x, x_h

    def _alpha_hop(self, x, grad_f, k, direction):
        if self.problem is None:
            raise Exception('Problem is not set')
        sigma = self._proxy_distance_vector(x)
        denominator = self.problem['smoothness_coef'] * np.linalg.norm(
            np.multiply(self.beta, direction)) ** 2 + 12 * np.dot(np.power(
            np.multiply(self.beta, direction), 2), np.absolute(grad_f))
        numerator = - np.dot(np.multiply(sigma, grad_f), direction)
        alpha = np.divide(numerator, denominator)

        if self.direction_type is 'stochastic':
            alpha = (1 - 1 / np.sqrt(k)) * alpha + 1 / (self.problem['smoothness_coef'] * np.sqrt(k))

        return alpha

    def _compute_x_0(self, x_0):
        if self.problem is None:
            raise Exception('Problem is not set')
        if x_0 is None or not utils.is_in_box(x_0, self.problem['ub'], self.problem['lb']):
            x_0 = self.problem['lb'] + (self.problem['ub'] - self.problem['lb']) / 2
        n = len(x_0)
        iterations = 0
        max_iterations = 10
        grad_f = self.problem['gradient'](x_0)
        if np.linalg.norm(grad_f) == 0:
            grad_f = (self.problem['smoothness_coef'] / 10) * (np.random.rand(n) - 0.5)

        while iterations < max_iterations and utils.is_in_box(x_0, self.problem['ub'] - self.ascent_stop_criterion,
                                                              self.problem['lb'] + self.ascent_stop_criterion) \
                and np.linalg.norm(grad_f) > (10 ** -6) * (1 / n):
            if self.initial_ascent_type is 'ascent':
                x_0 = x_0 + (1 / self.problem['smoothness_coef']) * grad_f
            elif self.initial_ascent_type is 'binary_neutral_ascent':
                x_0 = x_0 + (1 / self.problem['smoothness_coef']) * np.multiply(grad_f,
                                                                                np.ones(n) - self.problem[
                                                                                    'binary_indicator'])
            iterations += 1

        return self._activation(x_0, self.problem['ub'] - self.ascent_stop_criterion,
                                self.problem['lb'] + self.ascent_stop_criterion)

    def _stopping_criterion_met(self, x, grad_f, iterations):
        if self.problem is None:
            raise Exception('Problem is not set')
        if iterations >= self.max_iterations - 1:
            return True
        else:
            # TODO(Mathilde): here there is not other option for the stopping criterion!!
            if self.stopping_criterion_type is 'gradient' and np.linalg.norm(
                    np.multiply(grad_f, self._proxy_distance_vector(x))) < self.precision_stopping_criterion:
                return True
            else:
                return False

    def _find_direction(self, x, grad_f):
        if self.problem is None:
            raise Exception('Problem is not set')
        # TODO(Mathilde): Here sometimes there is no solution
        n = np.size(x)
        binary_absorption_mask = self._compute_binary_absorption_mask(x)

        # classic gradient
        if self.direction_type is 'classic' or self.direction_type is 'stochastic':
            if self.absorption_criterion is not None:
                direction = - grad_f
            else:
                direction = - np.multiply(binary_absorption_mask, grad_f)

            if self.direction_type is 'stochastic':
                # TODO(Mathilde): make 0.3 as a parameter
                direction = - np.multiply(direction, (np.random.uniform(0, 1, n) - 0.3))

        elif (self.direction_type is 'binary' or self.direction_type is 'soft_binary'):
            if self.direction_type is 'soft_binary':
                b = np.multiply(
                    self._activation(x, self.problem['ub'], self.problem['lb']) + 1 / 2 * (
                            self.problem['lb'] - self.problem['ub']),
                    self.problem['binary_indicator'])
                h = - grad_f
            elif self.direction_type is 'binary':
                b = np.multiply(np.sign(x + 1 / 2 * (self.problem['lb'] - self.problem['ub'])),
                                self.problem['binary_indicator'])
                h = - grad_f

            g = - np.multiply(self._proxy_distance_vector(x), grad_f)

            if self.absorption_criterion is not None:
                b = np.multiply(binary_absorption_mask, b)
                h = np.multiply(binary_absorption_mask, h)

            b = utils.normalize_array(b)
            h = utils.normalize_array(h)
            g = utils.normalize_array(g)
            w = self.gamma * b + (1 - self.gamma) * h
            y = max(0, - np.dot(g.T, w) + math.atan(self.theta) * np.sqrt(np.linalg.norm(w) ** 2 - np.dot(g.T, w) ** 2))
            direction = np.multiply(w + y * g, binary_absorption_mask)

        else:
            raise Exception('Direction Type does not exist!')

        direction = utils.normalize_array(direction)

        return direction

    def _compute_binary_absorption_mask(self, x):
        if self.problem is None:
            raise Exception('Problem is not set')
        n = np.size(x)
        binary_absorption_mask = np.ones(n)
        for i in range(n):
            if self.problem['binary_indicator'][i]:
                if x[i] == self.problem['ub'][i] or x[i] == self.problem['lb'][i]:
                    binary_absorption_mask[i] = 0
        return binary_absorption_mask

    def _absorb_solution_to_limits(self, x):
        if self.problem is None:
            raise Exception('Problem is not set')
        for i in range(len(x)):
            if min(x[i] - self.problem['lb'][i], self.problem['ub'][i] - x[i]) < self.absorption_criterion:
                if x[i] + 1 / 2 * (self.problem['lb'][i] - self.problem['ub'][i]) < 0:
                    x[i] = self.problem['lb'][i]
                else:
                    x[i] = self.problem['ub'][i]
        return x

    def _inverse_activation(self, x, ub, lb):
        z = np.divide((x - lb), (ub - lb))
        return lb + np.multiply(ub - lb, self.inverse_activation_function(z, self.beta))

    def _activation(self, x, ub, lb):
        z = np.divide((x - lb), (ub - lb))
        return lb + np.multiply(ub - lb, self.activation_function(z, self.beta))

    def _proxy_distance_vector(self, x):
        if self.problem is None:
            raise Exception('Problem is not set')
        # TODO(BERTRAND): Check and rectify that function
        z = np.divide((x - self.problem['lb']), (self.problem['ub'] - self.problem['lb']))
        return self.proxy_distance_vector(z, self.beta)
