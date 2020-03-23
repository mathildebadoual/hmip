import math

import hmip.utils as utils
import numpy as np


class HopfieldSolver():
    def __init__(self,
                 activation_type='sin',
                 gamma=0.90,
                 theta=0.01,
                 ascent_stop_criterion=0.01,
                 absorption_criterion=None,
                 max_iterations=500,
                 stopping_criterion_type='gradient',
                 direction_type='binary',
                 step_type='classic',
                 initial_ascent_type='binary_neutral_ascent',
                 precision_stopping_criterion=10**-6,
                 beta=None):

        self.activation_type = activation_type
        self.activation_function = getattr(utils,
                                           'activation_' + activation_type)
        self.inverse_activation_function = getattr(
            utils, 'inverse_activation_' + activation_type)
        self.proxy_distance_vector = getattr(
            utils, 'proxy_distance_vector_' + activation_type)
        self.ascent_stop_criterion = utils.adapt_ascent_stop_criterion(
            ascent_stop_criterion, absorption_criterion)
        self.stopping_criterion_type = stopping_criterion_type
        self.absorption_criterion = absorption_criterion
        self.initial_ascent_type = initial_ascent_type
        self.step_type = step_type
        self.direction_type = direction_type
        self.max_iterations = max_iterations
        self.precision_stopping_criterion = precision_stopping_criterion
        self.gamma = gamma
        self.theta = theta
        self.beta = beta

    def setup_optimization_problem(self,
                                   objective_function,
                                   gradient,
                                   lb,
                                   ub,
                                   binary_indicator,
                                   A_eq=None,
                                   b_eq=None,
                                   A_ineq=None,
                                   b_ineq=None,
                                   x_0=None,
                                   smoothness_coef=None,
                                   penalty_eq=0,
                                   penalty_ineq=0,
                                   dual_eq=None,
                                   dual_ineq=None):
        print('Set up optimization problem ....')
        utils.check_type(len(binary_indicator),
                         lb=lb,
                         ub=ub,
                         binary_indicator=binary_indicator)

        if not smoothness_coef:
            print('compute smoothness coef')
            smoothness_coef = utils.compute_approximate_smoothness_coef(
                gradient, lb, ub)

        if A_eq is not None and len(A_eq.shape) == 1:
            A_eq = A_eq.reshape((1, -1))

        if A_ineq is not None and len(A_ineq.shape) == 1:
            A_ineq = A_ineq.reshape((1, -1))

        problem = dict({
            'objective_function': objective_function,
            'gradient': gradient,
            'lb': lb,
            'ub': ub,
            'A_eq': A_eq,
            'b_eq': b_eq,
            'A_ineq': A_ineq,
            'b_ineq': b_ineq,
            'binary_indicator': binary_indicator,
            'smoothness_coef': smoothness_coef,
            'x_0': x_0,
            'dim_problem': len(binary_indicator),
            'penalty_eq': penalty_eq,
            'penalty_ineq': penalty_ineq,
            'dual_eq': dual_eq,
            'dual_ineq': dual_ineq,
        })

        if type(self.beta) == int:
            self.beta = self.beta * binary_indicator - binary_indicator + np.ones(problem['dim_problem'])
        elif self.beta is None:
            self.beta = np.ones(problem['dim_problem'])

        problem['x_0'] = self._compute_x_0(problem)

        print('.... Optimization problem set up.')

        return problem

    def solve(self, problem):
        print('Solving optimization problem ....')

        x = np.nan * np.ones((problem['dim_problem'], self.max_iterations))
        x_h = np.nan * np.ones((problem['dim_problem'], self.max_iterations))
        f_val_hist = np.nan * np.ones(self.max_iterations)
        step_size = np.nan * np.ones(self.max_iterations)

        A_ineq = problem['A_ineq']
        A_eq = problem['A_eq']
        b_ineq = problem['b_ineq']
        b_eq = problem['b_eq']
        dual_eq = problem['dual_eq']
        dual_ineq = problem['dual_ineq']

        if (A_eq is not None and b_eq is not None and dual_eq is None) or \
                (A_ineq is not None and b_ineq is not None and dual_ineq is None):
            print('Computing the dual variable ....')
            dual_variables_eq, dual_variables_ineq = self._get_dual_variables(
                problem)
            print('.... Dual variable computed.')
        else:
            print('Dual known')
            dual_variables_eq, dual_variables_ineq = dual_eq, dual_ineq

        if A_ineq is not None and b_ineq is not None and \
                A_eq is not None and b_eq is not None:
            objective_function, gradient, gradient_wrt_slack_variable = \
                self._all_constraints_problem(
                    problem, dual_variables_eq, dual_variables_ineq)

        elif A_ineq is not None and b_ineq is not None and (A_eq is None
                                                            or b_eq is None):
            objective_function, gradient, gradient_wrt_slack_variable = \
                self._inequality_constraints_problem(
                    problem, dual_variables_ineq)

        elif A_eq is not None and b_eq is not None and (A_ineq is None
                                                        or b_ineq is None):
            objective_function, gradient = \
                self._equality_constraints_problem(
                    problem, dual_variables_eq)

        else:
            objective_function, gradient = \
                self._no_constraints_problem(problem)

        x[:, 0] = problem['x_0']
        x_h[:, 0] = self._inverse_activation(problem['x_0'], problem['lb'],
                                             problem['ub'])
        if A_ineq is not None and b_ineq is not None:
            s = np.nan * np.ones((len(b_ineq), self.max_iterations))
            s[:, 0] = 0 * problem['b_ineq']
            f_val_hist[0] = objective_function((x[:, 0], s[:, 0]))
            grad_f = gradient((x[:, 0], s[:, 0]))
        else:
            grad_f = gradient(x[:, 0])
        if np.linalg.norm(grad_f) == 0:
            grad_f = (problem['smoothness_coef'] / 10) * \
                (np.random.rand(problem['dim_problem']) - 0.5)
        k = 0

        while not self._stopping_criterion_met(x[:, k], grad_f, k, problem):

            direction = self._find_direction(x[:, k], grad_f, problem)

            if self.step_type == 'armijo':
                alpha = np.divide(np.linalg.norm(grad_f),
                                  problem['smoothness_coef'])
                f_val_hist[k + 1] = f_val_hist[k] + 1
                prox_dist = self._proxy_distance_vector(
                    x[:, k], problem['ub'], problem['lb'])
                while f_val_hist[k + 1] > f_val_hist[k] + alpha * np.dot(
                        np.multiply(prox_dist, grad_f).T, direction):
                    x[:, k + 1], x_h[:, k + 1] = self._hopfield_update(
                        x_h[:, k], alpha, direction, problem)
                    if A_ineq is not None and b_ineq is not None:
                        s[:, k + 1] = np.minimum(
                            np.zeros(len(s[:, k + 1])),
                            s[:, k] - 1 / problem['penalty_ineq'] *
                            gradient_wrt_slack_variable(
                                (x[:, k + 1], s[:, k])))
                        f_val_hist[k + 1] = objective_function(
                            (x[:, k + 1], s[:, k + 1]))
                        grad_f = gradient((x[:, k + 1], s[:, k + 1]))
                    else:
                        f_val_hist[k + 1] = objective_function(x[:, k + 1])
                        grad_f = gradient(x[:, k + 1])
                    alpha = alpha / 2
                step_size[k] = 2 * alpha

            else:
                alpha = self._alpha_hop(x[:, k], grad_f, k, direction, problem)
                x[:, k + 1], x_h[:, k + 1] = self._hopfield_update(
                    x_h[:, k], alpha, direction, problem)
                if A_ineq is not None and b_ineq is not None:
                    s[:, k + 1] = s[:, k + 1] = np.minimum(
                        np.zeros(len(s[:, k + 1])), s[:, k] - 1 /
                        problem['penalty_ineq'] * gradient_wrt_slack_variable(
                            (x[:, k + 1], s[:, k])))
                    f_val_hist[k + 1] = objective_function(
                        (x[:, k + 1], s[:, k + 1]))
                    grad_f = gradient((x[:, k + 1], s[:, k + 1]))
                else:
                    f_val_hist[k + 1] = objective_function(x[:, k + 1])
                    grad_f = gradient(x[:, k + 1])
                    step_size[k] = alpha

            if self.absorption_criterion is not None:
                x[:, k + 1] = self._absorb_solution_to_limits(
                    x[:, k + 1], problem)

            k += 1

        x = utils.remove_nan_results(x)

        print('Candidate solution found with %s number of iterations.' % k)
        if A_ineq is not None and b_ineq is not None:
            return x, x_h, f_val_hist, step_size, dict(
                {'slack_variable': s,
                 'dual_variable_eq': dual_variables_eq,
                 'dual_variable_ineq': dual_variables_eq})
        else:
            return x, x_h, f_val_hist, step_size, dict()


    def _get_dual_variables(self, problem):
        n = problem['dim_problem']
        A_ineq = problem['A_ineq']
        A_eq = problem['A_eq']
        b_ineq = problem['b_ineq']
        b_eq = problem['b_eq']
        lb = problem['lb']
        ub = problem['ub']
        penalty_ineq = problem['penalty_ineq']
        penalty_eq = problem['penalty_eq']

        # TODO(Mathilde): add as a class variable
        precision = 10e-4
        if A_eq is not None:
            n_eq = A_eq.shape[0]
        if A_ineq is not None:
            n_ineq = A_ineq.shape[0]

        def inequality_constraint(variables):
            return np.dot(A_ineq, variables[:n]) - b_ineq - variables[n:]

        def equality_constraint(variables):
            return np.dot(A_eq, variables[:n]) - b_eq

        if A_ineq is not None and b_ineq is not None and \
                A_eq is not None and b_eq is not None:

            rate = 1 / (problem['smoothness_coef'] + penalty_eq * max(
                np.linalg.eigvals(np.dot(A_eq.T, A_eq))) + penalty_ineq *
                        max(np.linalg.eigvals(np.dot(A_ineq.T, A_ineq))))

            def gradient_augmented_lagrangian(variables, dual_variables_eq,
                                              dual_variables_ineq):
                gradient_x_ineq = np.dot(
                    A_ineq.T, dual_variables_ineq) + penalty_ineq * np.dot(
                        A_ineq.T, inequality_constraint(variables))
                gradient_x_eq = np.dot(
                    A_eq.T, dual_variables_eq) + penalty_eq * np.dot(
                        A_eq.T, equality_constraint(variables))
                gradient_x = problem['gradient'](
                    variables[:n]) + gradient_x_ineq + gradient_x_eq
                gradient_s = -penalty_ineq * inequality_constraint(
                    variables) - dual_variables_ineq
                return np.concatenate((gradient_x, gradient_s))

            dual_variables_eq = np.ones(n_eq)
            next_dual_variables_eq = 0.1 * np.ones(n_eq)
            dual_variables_ineq = np.ones(n_ineq)
            next_dual_variables_ineq = 0.1 * np.ones(n_ineq)
            c_k = 0.1
            d_k = 0.1
            iterations = 0

            while np.linalg.norm(next_dual_variables_eq - dual_variables_eq) > precision and \
                    np.linalg.norm(next_dual_variables_ineq - dual_variables_ineq) > precision:
                iterations += 1

                s_0 = np.zeros(n_ineq)
                next_x = np.concatenate((problem['x_0'], s_0))
                x = np.ones(next_x.shape)

                dual_variables_eq = next_dual_variables_eq.copy()
                dual_variables_ineq = next_dual_variables_ineq.copy()
                prev_x = np.copy(x)

                while np.linalg.norm(next_x - x) > precision:
                    x = next_x
                    next_x = utils.projection(
                        x - rate * gradient_augmented_lagrangian(
                            x, dual_variables_eq, dual_variables_ineq), n, lb,
                        ub)
                print('precision: %s' % np.linalg.norm(next_x - x))
                next_x = x

                alpha = 0.9
                c_k = alpha**iterations * np.linalg.norm(
                    equality_constraint(prev_x)) / np.linalg.norm(
                        equality_constraint(x))
                beta = 0.9
                d_k = beta**iterations * np.linalg.norm(
                    inequality_constraint(prev_x)) / np.linalg.norm(
                        inequality_constraint(x))

                next_dual_variables_eq = dual_variables_eq + c_k * equality_constraint(
                    x)
                next_dual_variables_ineq = dual_variables_ineq + d_k * inequality_constraint(
                    x)

                print(
                    'precision dual: %s' %
                    np.linalg.norm(next_dual_variables_eq - dual_variables_eq))

            return dual_variables_eq, dual_variables_ineq

        elif A_ineq is not None and b_ineq is not None \
                and (A_eq or b_eq) is None:

            rate = 1 / (problem['smoothness_coef'] + penalty_ineq *
                        max(np.linalg.eigvals(np.dot(A_ineq.T, A_ineq))))

            def gradient_augmented_lagrangian(variables, dual_variables):
                gradient_x = problem['gradient'](variables[:n]) + np.dot(
                    A_ineq.T, dual_variables) + penalty_ineq * np.dot(
                        A_ineq.T, inequality_constraint(variables))
                gradient_s = -penalty_eq * inequality_constraint(
                    variables) - dual_variables
                return np.concatenate((gradient_x, gradient_s))

            dual_variables = np.ones(n_ineq)
            next_dual_variables = 0.9 * np.ones(n_ineq)
            d_k = 0.1
            iterations = 0

            while np.linalg.norm(next_dual_variables -
                                 dual_variables) > precision:
                iterations += 1
                s_0 = np.zeros(n_ineq)
                next_x = np.concatenate((problem['x_0'], s_0))
                x = np.ones(next_x.shape)

                dual_variables = next_dual_variables.copy()
                prev_x = np.copy(x)

                while np.linalg.norm(next_x - x) > precision:
                    x = next_x
                    next_x = utils.projection(
                        x - rate *
                        gradient_augmented_lagrangian(x, dual_variables), n,
                        lb, ub)
                next_x = x

                beta = 0.9
                d_k = beta**iterations * np.linalg.norm(
                    inequality_constraint(prev_x)) / np.linalg.norm(
                        inequality_constraint(x))
                next_dual_variables = np.maximum(
                    np.zeros(n_ineq),
                    dual_variables + d_k * inequality_constraint(next_x))

            return None, next_dual_variables

        elif A_eq is not None and b_eq is not None and (A_ineq is None
                                                        or b_ineq is None):

            rate = 1 / (problem['smoothness_coef'] + penalty_eq *
                        max(np.linalg.eigvals(np.dot(A_eq.T, A_eq))))

            def gradient_augmented_lagrangian(variables, dual_variables):
                gradient_x_eq = np.dot(
                    A_eq.T, dual_variables) + penalty_eq * np.dot(
                        A_eq.T, equality_constraint(variables))
                gradient_x = problem['gradient'](variables[:n]) + gradient_x_eq
                return gradient_x

            dual_variables = np.ones(n_eq)
            next_dual_variables = 0.9 * np.ones(n_eq)
            d_k = 0.1
            iterations = 0

            while np.linalg.norm(next_dual_variables -
                                 dual_variables) > precision:
                iterations += 1
                next_x = problem['x_0']
                x = np.ones(next_x.shape)
                dual_variables = next_dual_variables.copy()
                prev_x = np.copy(x)

                while np.linalg.norm(next_x - x) > precision:
                    x = next_x
                    next_x = utils.projection(
                        x - rate *
                        gradient_augmented_lagrangian(x, dual_variables), n,
                        lb, ub)

                alpha = 0.9
                c_k = alpha**iterations * np.linalg.norm(
                    equality_constraint(prev_x)) / np.linalg.norm(
                        equality_constraint(x))
                next_dual_variables = np.maximum(
                    np.zeros(n_eq),
                    dual_variables + c_k * equality_constraint(next_x))

            return next_dual_variables, None


    def _hopfield_update(self, x_h, alpha, direction, problem):
        x_h = x_h + alpha * direction
        x = self._activation(x_h, problem['lb'], problem['ub'])
        return x, x_h


    def _alpha_hop(self, x, grad_f, k, direction, problem):
        sigma = self._proxy_distance_vector(x, problem['ub'],
                                            problem['lb'])
        denominator = problem['smoothness_coef'] * np.linalg.norm(
            np.multiply(self.beta, direction))**2 + 12 * np.dot(
                np.power(np.multiply(self.beta, direction), 2),
                np.absolute(grad_f))
        numerator = -np.dot(np.multiply(sigma, grad_f), direction)
        alpha = np.divide(numerator, denominator)

        if self.direction_type == 'stochastic':
            alpha = (1 - 1 / np.sqrt(k)) * alpha + 1 / (
                problem['smoothness_coef'] * np.sqrt(k))

        return alpha


    def _compute_x_0(self, problem):
        x_0 = np.copy(problem['x_0'])
        if x_0.all() is None or not utils.is_in_box(x_0, problem['ub'],
                                                    problem['lb']):
            x_0 = (problem['ub'] + problem['lb']) / 2

        n = len(x_0)
        iterations = 0
        max_iterations = 10**3
        grad_f = problem['gradient'](x_0)
        if np.linalg.norm(grad_f) == 0:
            grad_f = (problem['smoothness_coef'] / 10) *\
                (np.random.rand(n) - 0.5)

        while iterations < max_iterations and utils.is_in_box(
            x_0, problem['ub'] - self.ascent_stop_criterion,
            problem['lb'] + self.ascent_stop_criterion) \
                and np.linalg.norm(grad_f) > (10 ** -6) * (1 / n):
            if self.initial_ascent_type == 'ascent':
                x_0 = x_0 + (1 / problem['smoothness_coef']) * grad_f
            elif self.initial_ascent_type == 'binary_neutral_ascent':
                x_0 = x_0 + (1 / problem['smoothness_coef']) * np.multiply(
                    grad_f,
                    np.ones((n, )) - problem['binary_indicator'])
            iterations += 1
        return np.minimum(
            np.maximum(x_0, problem['lb'] + self.ascent_stop_criterion),
            problem['ub'] - self.ascent_stop_criterion)


    def _stopping_criterion_met(self, x, grad_f, iterations, problem):
        if iterations >= self.max_iterations - 1:
            return True
        else:
            # TODO(Mathilde): here there is not other option for the stopping
            # criterion!!
            precision = np.linalg.norm(
                np.multiply(
                    grad_f,
                    self._proxy_distance_vector(x, problem['ub'],
                                                problem['lb'])))
            if self.stopping_criterion_type == 'gradient' and \
                    precision < self.precision_stopping_criterion:
                return True
            else:
                return False


    def _compute_binary_absorption_mask(self, x, problem):
        n = np.size(x)
        binary_absorption_mask = np.ones(n)
        for i in range(n):
            if problem['binary_indicator'][i]:
                if x[i] == problem['ub'][i] or x[i] == problem['lb'][i]:
                    binary_absorption_mask[i] = 0
        return binary_absorption_mask


    def _find_direction(self, x, grad_f, problem):
        # TODO(Mathilde): Here sometimes there is no solution
        n = np.size(x)
        binary_absorption_mask = self._compute_binary_absorption_mask(x,
                                                                      problem)

        # classic gradient
        if (self.direction_type == 'classic') \
                or (self.direction_type == 'stochastic'):
            if self.absorption_criterion is not None:
                direction = - grad_f
            else:
                direction = - np.multiply(binary_absorption_mask, grad_f)

            if self.direction_type == 'stochastic':
                # TODO(Mathilde): make 0.3 as a parameter
                direction = - np.multiply(direction,
                                          (np.random.uniform(0, 1, n) - 0.3))

        elif self.direction_type == 'binary' \
                or self.direction_type == 'soft_binary':
            if self.direction_type == 'soft_binary':
                # TODO check that : definition of d looks weird
                b = np.multiply(
                    self._activation(x, problem['ub'], problem['lb']) + 1 / 2
                    * (problem['lb'] - problem['ub']),
                    problem['binary_indicator'])
                h = - grad_f
            elif self.direction_type == 'binary':
                b = np.multiply(
                    np.sign(x + 1 / 2 * (problem['lb'] - problem['ub'])),
                    problem['binary_indicator'])
                h = -grad_f

            g = -np.multiply(
                self._proxy_distance_vector(x, problem['ub'], problem['lb']),
                grad_f)
            # TODO check that next part
            if self.absorption_criterion is not None:
                b = np.multiply(binary_absorption_mask, b)
                h = np.multiply(binary_absorption_mask, h)

            b = utils.normalize_array(b)
            h = utils.normalize_array(h)
            g = utils.normalize_array(g)
            w = self.gamma * b + (1 - self.gamma) * h
            y = max(0, - np.dot(g.T, w) + math.atan(self.theta) * np.sqrt(
                np.linalg.norm(w) ** 2 - np.dot(g.T, w) ** 2))
            direction = np.multiply(w + y * g, binary_absorption_mask)

        else:
            raise Exception('Direction Type does not exist!')

        direction = utils.normalize_array(direction)

        return direction


    def _absorb_solution_to_limits(self, x, problem):
        for i in range(len(x)):
            if min(x[i] - problem['lb'][i], problem['ub'][i] - x[i]) \
                    < self.absorption_criterion:
                if x[i] + 1 / 2 * (problem['lb'][i] - problem['ub'][i]) < 0:
                    x[i] = problem['lb'][i]
                else:
                    x[i] = problem['ub'][i]
        return x


    def _inverse_activation(self, x, ub, lb):
        z = np.divide((x - lb), (ub - lb))
        return lb + np.multiply(ub - lb,
                                self.inverse_activation_function(z, self.beta))


    def _activation(self, x, ub, lb):
        z = np.divide((x - lb), (ub - lb))
        return lb + np.multiply(ub - lb, self.activation_function(
            z, self.beta))


    def _proxy_distance_vector(self, x, ub, lb):
        z = np.divide((x - lb), (ub - lb))
        return self.proxy_distance_vector(z, self.beta)


    def _inequality_constraints_problem(self, problem, dual_variable_ineq):
        A_ineq = problem['A_ineq']
        b_ineq = problem['b_ineq']

        def inequality_constraint(optimization_variable, slack_variable):
            return np.dot(
                A_ineq,
                optimization_variable) - b_ineq - slack_variable

        def objective_function(variables):
            optimization_variable, slack_variable = variables
            main_function = problem['objective_function'](
                optimization_variable)
            inequality_term = np.dot(dual_variable_ineq.T,
                                     inequality_constraint(
                                         optimization_variable,
                                         slack_variable)) + problem[
                'penalty_ineq'] / 2 * np.linalg.norm(
                inequality_constraint(optimization_variable,
                                      slack_variable), 2)
            return main_function + inequality_term

        def gradient(variables):
            optimization_variable, slack_variable = variables
            main_function = problem['gradient'](optimization_variable)
            inequality_term = np.dot(A_ineq.T,
                                     dual_variable_ineq) + problem[
                'penalty_ineq'] * \
                np.dot(A_ineq.T,
                       inequality_constraint(
                           optimization_variable, slack_variable))
            return main_function + inequality_term

        def gradient_wrt_slack_variable(variables):
            optimization_variable, slack_variable = variables
            ineq_cst = inequality_constraint(
                optimization_variable, slack_variable)
            return - problem['penalty_ineq'] * ineq_cst
            - dual_variable_ineq

        return objective_function, gradient, gradient_wrt_slack_variable

    def _equality_constraints_problem(self, problem, dual_variable_eq):
        A_eq = problem['A_eq']
        b_eq = problem['b_eq']

        def equality_constraint(optimization_variable):
            return np.dot(A_eq, optimization_variable)
            - b_eq

        def objective_function(variable):
            main_function = problem['objective_function'](variable)
            equality_term = np.dot(dual_variable_eq.T,
                                   equality_constraint(
                                       variable)) + problem[
                'penalty_eq'] / 2 * np.linalg.norm(
                    equality_constraint(variable), 2)
            return main_function + equality_term

        def gradient(variable):
            main_function = problem['gradient'](variable)
            equality_term = np.dot(A_eq.T, dual_variable_eq)
            + problem['penalty_eq'] * np.dot(A_eq.T,
                                             equality_constraint(variable))
            return main_function + equality_term

        return objective_function, gradient


    def _all_constraints_problem(self, problem, dual_variable_eq,
                                 dual_variable_ineq):
        A_ineq = problem['A_ineq']
        b_ineq = problem['b_ineq']
        A_eq = problem['A_eq']
        b_eq = problem['b_eq']

        def inequality_constraint(optimization_variable, slack_variable):
            return np.dot(A_ineq,
                          optimization_variable)
            - b_ineq - slack_variable

        def equality_constraint(optimization_variable):
            return np.dot(A_eq,
                          optimization_variable) - b_eq

        def objective_function(variables):
            optimization_variable, slack_variable = variables
            main_function = problem['objective_function'](
                optimization_variable)
            ineq_cst = inequality_constraint(
                optimization_variable,
                slack_variable)
            equ_cst = equality_constraint(
                optimization_variable)
            inequality_term = np.dot(
                dual_variable_ineq.T, ineq_cst)
            + problem['penalty_ineq'] / 2 * np.linalg.norm(
                ineq_cst, 2)
            equality_term = np.dot(
                dual_variable_eq.T, equ_cst)
            + problem['penalty_eq'] / 2 * np.linalg.norm(
                equ_cst, 2)
            return main_function + inequality_term + equality_term

        def gradient(variables):
            optimization_variable, slack_variable = variables
            main_function = problem['gradient'](optimization_variable)
            equ_cst = equality_constraint(
                optimization_variable)
            ineq_cst = inequality_constraint(
                optimization_variable,
                slack_variable)
            equality_term = np.dot(
                A_eq.T,
                dual_variable_eq) + problem['penalty_eq'] * np.dot(
                    A_eq.T, equ_cst)
            inequality_term = np.dot(
                A_ineq.T,
                dual_variable_ineq) + problem[
                    'penalty_ineq'] * np.dot(
                A_ineq.T, ineq_cst)
            return main_function + equality_term + inequality_term

        def gradient_wrt_slack_variable(variables):
            optimization_variable, slack_variable = variables
            return - problem['penalty_ineq'] * inequality_constraint(
                optimization_variable, slack_variable) - dual_variable_ineq

        return objective_function, gradient, gradient_wrt_slack_variable


    def _no_constraints_problem(self, problem):
        def objective_function(variable):
            return problem['objective_function'](variable)

        def gradient(variable):
            return problem['gradient'](variable)

        return objective_function, gradient
