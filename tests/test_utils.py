import unittest
import numpy as np
import hmip.utils as utils


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