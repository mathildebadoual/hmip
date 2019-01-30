import unittest
import numpy as np
import hmip.utils as utils

class TestIsInBox(unittest.TestCase):
    def setUp(self):
        pass

    def test_is_in_box(self):
        output = utils.is_in_box(np.array([0.5, 0.5]), np.array([1, 1]), np.array([0, 0]))
        self.assertTrue(output)


class TestProxyDistanceVector(unittest.TestCase):
    def setUp(self):
        self.dim = 10

    def test_tanh(self):
        x = 1/2 * np.ones(self.dim)
        beta = np.ones(self.dim)
        output = utils.proxy_distance_vector_tanh(x, beta)
        self.assertTrue(np.array_equal(output, np.ones(self.dim)))

    def test_pwl(self):
        x = 1 / 2 * np.ones(self.dim)
        beta = np.ones(self.dim)
        output = utils.proxy_distance_vector_pwl(x, beta)
        self.assertTrue(np.array_equal(output, beta))

        x = 2 * np.ones(self.dim)
        beta = np.ones(self.dim)
        output = utils.proxy_distance_vector_pwl(x, beta)
        self.assertTrue(np.array_equal(output, np.zeros(self.dim)))

    def test_sin(self):
        x = np.ones(self.dim)
        beta = np.ones(self.dim)
        output = utils.proxy_distance_vector_sin(x, beta)
        self.assertTrue(np.array_equal(output, np.zeros(self.dim)))

    def test_exp(self):
        x = 1 / 2 * np.ones(self.dim)
        beta = np.ones(self.dim)
        output = utils.proxy_distance_vector_exp(x, beta)
        self.assertTrue(np.array_equal(output, 1 / 2 * np.ones(self.dim)))

    def test_identity(self):
        x = 1 / 2 * np.ones(self.dim)
        beta = np.ones(self.dim)
        output = utils.proxy_distance_vector_identity(x, beta)
        self.assertTrue(np.array_equal(output, np.zeros(self.dim)))


class TestActivationFunction(unittest.TestCase):
    def setUp(self):
        self.dim = 10

    def test_tanh(self):
        x = 1/2 * np.ones(self.dim)
        beta = np.ones(self.dim)
        output = utils.activation_tanh(x, beta)
        self.assertTrue(np.array_equal(output, x))

    def test_pwl(self):
        x = 1 / 2 * np.ones(self.dim)
        beta = np.ones(self.dim)
        output = utils.activation_pwl(x, beta)
        self.assertTrue(np.array_equal(output, x))

        x = 2 * np.ones(self.dim)
        beta = np.ones(self.dim)
        output = utils.activation_pwl(x, beta)
        self.assertTrue(np.array_equal(output, beta))

    def test_sin(self):
        x = np.ones(self.dim)
        beta = np.ones(self.dim)
        output = utils.activation_sin(x, beta)
        self.assertTrue(np.array_equal(np.round_(output, decimals=8), 0.92073549 * x))

    def test_exp(self):
        x = 1 / 2 * np.ones(self.dim)
        beta = np.ones(self.dim)
        output = utils.activation_exp(x, beta)
        self.assertTrue(np.array_equal(output, 1 / 2 * np.ones(self.dim)))

    def test_identity(self):
        x = 1 / 2 * np.ones(self.dim)
        beta = np.ones(self.dim)
        output = utils.activation_identity(x, beta)
        self.assertTrue(np.array_equal(output, x))


class TestCheckType(unittest.TestCase):
    def setUp(self):
        self.n = 10

    def test_check_initial_state(self):
        output = utils.check_type(self.n, initial_state=None)
        self.assertEqual(output, None)

        initial_state = np.ones(self.n)
        output = utils.check_type(self.n, initial_state=initial_state)
        self.assertEqual(output, True)

        initial_state = np.ones(self.n - 2)
        output = utils.check_type(self.n, initial_state=initial_state)
        self.assertEqual(output, False)

        initial_state = 1
        output = utils.check_type(self.n, initial_state=initial_state)
        self.assertEqual(output, False)