import unittest
import numpy as np
import hmip.hopfield as hop


class TestHopfield(unittest.TestCase):
    def setUp(self):
        self.H = np.array([[2, 0], [0, 1]])
        self.q = np.array([-2.7, -1.8])
        self.k_max = 20
        self.binary_indicator = np.array([0, 1])
        self.beta = 1
        self.ub = np.array([1, 1])
        self.lb = np.array([0, 0])
        self.absorption = 1
        self.step_type = 'classic'

    def test_hopfield_default(self):
        x, x_h, f_val_hist, step_size = hop.hopfield(self.H, self.q, self.lb, self.ub, self.binary_indicator,
                                                     k_max=self.k_max)
        self.assertEqual(x.shape[0], self.q.shape[0])
        self.assertEqual(x.shape[1], self.k_max)

    def test_hopfield_step_type_classic(self):
        x, x_h, f_val_hist, step_size = hop.hopfield(self.H, self.q, self.lb, self.ub, self.binary_indicator,
                                                     k_max=self.k_max,
                                                     step_type='classic')

        self.assertEqual(x.shape[0], self.q.shape[0])
        self.assertEqual(x.shape[1], self.k_max)

    def test_hopfield_step_type_armijo(self):
        x, x_h, f_val_hist, step_size = hop.hopfield(self.H, self.q, self.lb, self.ub, self.binary_indicator,
                                                     k_max=self.k_max,
                                                     step_type='armijo')

        self.assertEqual(x.shape[0], self.q.shape[0])
        self.assertEqual(x.shape[1], self.k_max)

    def test_hopfield_with_absorption(self):
        x, x_h, f_val_hist, step_size = hop.hopfield(self.H, self.q, self.lb, self.ub, self.binary_indicator,
                                                     k_max=self.k_max,
                                                     step_type='armijo',
                                                     absorption=1)

        self.assertEqual(x.shape[0], self.q.shape[0])
        self.assertEqual(x.shape[1], self.k_max)

    def test_hopfield_step_type_wrong(self):

            return (type(self.step_type) is str)


class TestOthers(unittest.TestCase):
    def setUp(self):
        self.n = 2
        self.H = np.array([[2, 0], [0, 1]])
        self.q = np.array([-2.7, -1.8])
        self.k_max = 20
        self.binary_indicator = np.array([0, 1])
        self.beta = np.ones(2)
        self.ub = np.array([1, 1])
        self.lb = np.array([0, 0])
        self.x = np.ones((self.n, self.k_max))
        self.x_0 = self.lb + (self.ub - self.lb) / 2
        self.smoothness_coef = np.max(np.linalg.eigvals(self.H))


    def test_create_initial_ascent_ascent(self):
        self.assertEqual(self.x_0.shape[0], hop.initial_state(self.H, self.q, self.lb, self.ub, self.binary_indicator,
                                                         self. k_max, self.smoothness_coef,self.x_0,
                                                         initial_ascent_type='ascent').shape[0])


    def test_create_initial_ascent_binary_neutral_ascent(self):
        self.assertEqual(self.x_0.shape[0], hop.initial_state(self.H, self.q, self.lb, self.ub, self.binary_indicator,
                                                              self.k_max, self.smoothness_coef, self.x_0,
                                                              initial_ascent_type='binary_neutral_ascent').shape[0])


    def test_compute_inverse_activation(self):
        activation_type = ['pwl', 'exp', 'sin', 'identity', 'tanh']

        x_0 = self.lb + (self.ub - self.lb) / 2
        beta = 0.5 * x_0

        for i in activation_type:
            if i is 'tanh':
                beta = [[1, 0], [0, 1]]
                self.assertTrue(np.array_equal(x_0, hop.inverse_activation(x_0, self.lb, self.ub, beta, i)[0]))
            else:
                self.assertTrue(np.array_equal(x_0, hop.inverse_activation(x_0, self.lb, self.ub, beta, i)))

    def test_compute_activation(self):
        activation_type = ['pwl', 'exp', 'sin', 'identity', 'tanh']
        solution = [[1, 1], [0.81606028, 0.81606028], [0.92073549, 0.92073549], [1., 1.], [0.88079708, 0.88079708]]

        x_0 = np.ones(self.n)

        for i in range(len(activation_type)):
            self.assertTrue(np.array_equal(np.round(hop.activation(x_0, self.lb, self.ub, self.beta,activation_type[i]),decimals=8),
                             solution[i]))



    def test_compute_binary_absorption_mask(self):
        x_0 = 0.5 * np.ones(self.n)
        sol = hop.compute_binary_absorption_mask(x_0, self.lb, self.ub, self.binary_indicator)
        self.assertTrue(np.array_equal(sol, 2*x_0))

    def test_smoothness_coefficient(self):
        output = hop.smoothness_coefficient(self.H)
        max_eigen_values = 2
        self.assertEqual(output, max_eigen_values)

    def test_alpha_hop(self):
        grad_f = np.ones(2)
        direction = np.ones(2)
        k = 1
        smoothness_coef = 1
        direction_type = hop.DEFAULT_DIRECTION_TYPE
        activation_type = hop.DEFAULT_ACTIVATION_TYPE
        output = hop.alpha_hop(self.x[:, k], grad_f, direction, k, self.lb, self.ub, smoothness_coef, self.beta,
                               direction_type, activation_type)
        self.assertEqual(output, 0)


class TestHopfieldUpdate(unittest.TestCase):
    def setUp(self):
        self.n = 2
        self.alpha = 1
        self.direction =1
        self.beta = 0.5*np.ones(2)
        self.ub = np.array([1, 1])
        self.lb = np.array([0, 0])

    def test_hopfield_update(self):
        activation_type = ['pwl', 'exp', 'sin', 'identity', 'tanh']
        solution = [[1, 1], [0.88843492, 0.88843492], [0.99874749, 0.99874749], [1., 1.], [0.95257413, 0.95257413]]

        x_h = np.ones(self.n)

        for i in range(len(activation_type)):
            self.assertTrue(np.array_equal(np.round(hop.hopfield_update(x_h, self.lb, self.ub, self.alpha, self.direction, self.beta,
                                                                   activation_type[i])[0], decimals=8),
                             solution[i]))


class TestFindDirection(unittest.TestCase):
    def setUp(self):
        self.n = 2
        self.H = np.array([[2, 0], [0, 1]])
        self.q = np.array([-2.7, -1.8])
        self.k_max = 20
        self.binary_indicator = np.array([0, 1])
        self.beta = np.ones(2)
        self.ub = np.array([1, 1])
        self.lb = np.array([0, 0])
        self.x = np.ones((self.n, self.k_max))
        self.x_0 = self.lb + (self.ub - self.lb) / 2
        self.smoothness_coef = np.max(np.linalg.eigvals(self.H))
        #self.grad_f = np.dot(self.H, self.x) + self.q

    # def test_find_direction_type_classic(self):
    #     activation_type = 'pwl'
    #     direction_type = 'classic'
    #     hop.find_direction(self.x, self.grad_f, self.lb, self.ub, self.binary_indicator, self.beta, direction_type, self.absorption, self.gamma, self.theta,
    #                    activation_type)
    #     pass
    #
    # def test_find_direction_type_stochastic(self):
    #     activation_type = 'pwl'
    #     direction_type = 'stochastic'
    #     hop.find_direction(self.x, self.grad_f, self.lb, self.ub, self.binary_indicator, self.beta, direction_type,
    #                        self.absorption, self.gamma, self.theta,
    #                        activation_type)
    #     pass
    #
    # def test_find_direction_type_binary(self):
    #     activation_type = 'pwl'
    #     direction_type = 'binary'
    #     hop.find_direction(self.x, self.grad_f, self.lb, self.ub, self.binary_indicator, self.beta, direction_type,
    #                        self.absorption, self.gamma, self.theta,
    #                        activation_type)
    #     pass
    #
    # def test_find_direction_type_soft_binary(self):
    #     activation_type = 'pwl'
    #     direction_type = 'soft binary'
    #     hop.find_direction(self.x, self.grad_f, self.lb, self.ub, self.binary_indicator, self.beta, direction_type,
    #                            self.absorption, self.gamma, self.theta,
    #                            activation_type)
    #     pass
