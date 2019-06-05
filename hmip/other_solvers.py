import cvxpy as cvx
import cplex


def cvxpy_solver(H, q, lb, ub, binary_indicator, solver=None):
    """
    Solves the same problem as hopfiel with cvxpy
    :param H:
    :param q:
    :param lb:
    :param ub:
    :param binary_indicator:
    :param solver: cvxpy solver
    :return:
    """
    n = q.shape[0]

    # creates the cvxpy type binary vector
    index_binary = []
    for i in range(len(binary_indicator)):
        if binary_indicator[i]:
            index_binary.append((i,))

    x = cvx.Variable(n, integer=index_binary)
    constraints = [lb <= x, x <= ub]
    objective = 1 / 2 * cvx.quad_form(x, H) + q.T * x
    objective = cvx.Minimize(objective)
    problem = cvx.Problem(objective, constraints)

    if solver is None:
        problem.solve()
    else:
        problem.solve(solver=solver)

    return x.value


def cplex_solver():
    """
    Solves the same problem as hopfield with CPLEX solver
    (requires CPLEX)
    :param H:
    :param q:
    :param lb:
    :param ub:
    :param binary_indicator:
    :param solver: cvxpy solver
    :return: result of the optimization
    """
    pass
