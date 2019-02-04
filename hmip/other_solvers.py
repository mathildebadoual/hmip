import numpy as np
import hmip.utils as utils
import cvxpy as cvx
from hmip.hopfield import objective_function


def cvxpy_solver(H, q, lb, ub, binary_indicator, solver=None):
    # create the vector for boolean option in cvxpy - list of tuples with index of the value that is an integer and
    n = q.shape[0]
    x = cvx.Variable(n, integer=binary_indicator)
    constraints = [lb <= x, x <= ub]
    objective = cvx.Minimize(objective_function(x, H, q))
    problem = cvx.Problem(objective, constraints)

    if solver is None:
        problem.solve()
    else:
        problem.solve(solver=solver)

    return x.values
