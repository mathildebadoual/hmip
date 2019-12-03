import cvxpy as cvx


def cvxpy_solver(H,
                 q,
                 lb,
                 ub,
                 binary_indicator,
                 A_eq,
                 b_eq,
                 A_ineq,
                 b_ineq,
                 solver=None,
                 verbose=False,
                 dual=False):
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
            index_binary.append((i, ))

    x = cvx.Variable(n, integer=index_binary)
    constraints = [lb <= x, x <= ub]
    if A_eq is not None and A_ineq is not None:
        constraints += [A_eq * x == b_eq, A_ineq * x <= b_ineq]
    objective = 1 / 2 * cvx.quad_form(x, H) + q.T * x
    objective = cvx.Minimize(objective)
    problem = cvx.Problem(objective, constraints)

    if solver is None:
        problem.solve(verbose=verbose)
    else:
        problem.solve(verbose=verbose, solver=solver)

    if dual and A_eq is not None and A_ineq is not None:
        return x.value, objective.value, constraints[2].dual_value, constraints[3].dual_value
    else:
        return x.value, objective.value


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
