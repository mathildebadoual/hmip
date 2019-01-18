from hmip.hopfield import hopfield
from scipy import sparse
import numpy as np


class HMIP():
    def __init__(self):
        pass

    def setup(self):
        pass

    def solve(self, P, q, A, l, u, binary_indicator, verbose=False, **solver_opts):
        """
        Setup HMIP solver problem of the form
                minimize     1/2 x' * P * x + q' * x
                subject to   l <= A * x <= u
                             x * binary_indicator are integers
        solver settings can be specified as additional keyword arguments

        :param P: (numpy.ndarray)
        :param q: (numpy.ndarray)
        :param A: (numpy.ndarray)
        :param l: (numpy.ndarray)
        :param u: (numpy.ndarray)
        :param binary_indicator: (numpy.ndarray)
        :param verbose: (boolean) to print optimization status or not
        :param solver_opts:
        :return:
        """
        #
        # #
        # # Get problem dimensions
        # #
        #
        # if P is None:
        #     if q is not None:
        #         n = len(q)
        #     elif A is not None:
        #         n = A.shape[1]
        #     else:
        #         raise ValueError("The problem does not have any variables")
        # else:
        #     n = P.shape[0]
        # if A is None:
        #     m = 0
        # else:
        #     m = A.shape[0]
        #
        # #
        # # Create parameters if they are None
        # #
        #
        # if (A is None and (l is not None or u is not None)) or \
        #         (A is not None and (l is None and u is None)):
        #     raise ValueError("A must be supplied together " +
        #                      "with at least one bound l or u")
        #
        # # Add infinity bounds in case they are not specified
        # if A is not None and l is None:
        #     l = -np.inf * np.ones(A.shape[0])
        # if A is not None and u is None:
        #     u = np.inf * np.ones(A.shape[0])
        #
        # # Create elements if they are not specified
        # if P is None:
        #     P = sparse.csc_matrix((np.zeros((0,), dtype=np.double),
        #                            np.zeros((0,), dtype=np.int),
        #                            np.zeros((n + 1,), dtype=np.int)),
        #                           shape=(n, n))
        # if q is None:
        #     q = np.zeros(n)
        #
        # if A is None:
        #     A = sparse.csc_matrix((np.zeros((0,), dtype=np.double),
        #                            np.zeros((0,), dtype=np.int),
        #                            np.zeros((n + 1,), dtype=np.int)),
        #                           shape=(m, n))
        #     l = np.zeros(A.shape[0])
        #     u = np.zeros(A.shape[0])
        #
        # #
        # # Check vector dimensions (not checked from C solver)
        # #
        #
        # # Check if second dimension of A is correct
        # # if A.shape[1] != n:
        # #     raise ValueError("Dimension n in A and P does not match")
        # if len(q) != n:
        #     raise ValueError("Incorrect dimension of q")
        # if len(l) != m:
        #     raise ValueError("Incorrect dimension of l")
        # if len(u) != m:
        #     raise ValueError("Incorrect dimension of u")
        #
        # #
        # # Check or Sparsify Matrices
        # #
        # if not sparse.issparse(P) and isinstance(P, np.ndarray) and \
        #         len(P.shape) == 2:
        #     raise TypeError("P is required to be a sparse matrix")
        # if not sparse.issparse(A) and isinstance(A, np.ndarray) and \
        #         len(A.shape) == 2:
        #     raise TypeError("A is required to be a sparse matrix")
