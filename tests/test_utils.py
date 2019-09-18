import unittest
import numpy as np

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import hmip.utils as utils


class TestIsInBox(unittest.TestCase):
    def setUp(self):
        pass

    def test_is_in_box(self):
        output = utils.is_in_box(np.array([0.5, 0.5]), np.array([1, 1]),
                                 np.array([0, 0]))
        self.assertTrue(output)

        output = utils.is_in_box(None, np.array([1, 1]), np.array([0, 0]))
        self.assertFalse(output)

        output = utils.is_in_box(np.array([2, 0.5]), np.array([1, 1]),
                                 np.array([0, 0]))
        self.assertFalse(output)


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
        self.assertTrue(np.array_equal(output, np.ones(self.dim)))

        x = 2 * np.ones(self.dim)
        beta = np.ones(self.dim)
        output = utils.proxy_distance_vector_pwl(x, beta)
        self.assertTrue(np.array_equal(output, np.zeros(self.dim)))

    def test_sin(self):
        x = np.ones(self.dim)
        beta = np.ones(self.dim)
        output = utils.proxy_distance_vector_sin(x, beta)
        self.assertTrue(np.allclose(output, np.zeros(self.dim), atol=1e-2))

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
        self.assertTrue(np.allclose(np.round_(output, decimals=8), 0.92073549 * np.ones(self.dim), rtol=0.01))

    def test_exp(self):
        x = 1 / 2 * np.ones(self.dim)
        beta = np.ones(self.dim)
        output = utils.activation_exp(x, beta)
        self.assertTrue(np.array_equal(output, 0.5 * np.ones(self.dim)))

    def test_identity(self):
        x = 1 / 2 * np.ones(self.dim)
        beta = np.ones(self.dim)
        output = utils.activation_identity(x, beta)
        self.assertTrue(np.array_equal(output, x))

class TestInverseActivationFunction(unittest.TestCase):
    def setUp(self):
        self.dim = 10

    def test_tanh(self):
        x = 0.1 *  np.ones(self.dim)
        beta = np.ones(self.dim)
        output = utils.inverse_activation_tanh(x, beta)
        self.assertTrue(np.allclose(output, -0.04930614 * np.ones(self.dim), rtol=0.01))

    def test_pwl(self):
        x = 1 / 2 * np.ones(self.dim)
        beta = np.ones(self.dim)
        output = utils.inverse_activation_pwl(x, beta)
        self.assertTrue(np.array_equal(output, x))

        x = 2 * np.ones(self.dim)
        beta = np.ones(self.dim)
        output = utils.inverse_activation_pwl(x, beta)
        self.assertTrue(np.array_equal(output, beta))

    def test_sin(self):
        x = np.ones(self.dim)
        beta = np.ones(self.dim)
        output = utils.inverse_activation_sin(x, beta)
        self.assertTrue(np.array_equal(np.round_(output), x))

    def test_exp(self):
        x = 1 / 2 * np.ones(self.dim)
        beta = np.ones(self.dim)
        output = utils.inverse_activation_exp(x, beta)
        self.assertTrue(np.array_equal(output, x))

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


class TestCheck(unittest.TestCase):
    def setUp(self):
        pass

    def test_check_ascent_stop(self):
        pass

    def test_check_symmetric(self):
        non_symmetric_matrix = np.array([[1, 0], [0, 1]])
        self.assertTrue(np.array_equal(utils.make_symmetric(non_symmetric_matrix),
                                       0.5 * (non_symmetric_matrix + non_symmetric_matrix.T)))


if __name__ == '__main__':
    unittest.main()
