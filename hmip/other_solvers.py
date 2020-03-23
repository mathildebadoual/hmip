import cvxpy as cvx
import time


def cvxpy_solver(H,
                 q,
                 lb,
                 ub,
                 binary_indicator,
                 A_eq=None,
                 b_eq=None,
                 A_ineq=None,
                 b_ineq=None,
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

    t = time.perf_counter()
    if solver is None:
        sol = problem.solve(verbose=verbose)
    else:
        sol = problem.solve(verbose=verbose, solver=solver)
    t_total = time.perf_counter() - t

    print('PROBLEM STATUS: %s' % problem.status)

    if dual:
        return x.value, objective.value, constraints[2].dual_value, constraints[3].dual_value, t_total
    else:
        return x.value, objective.value, t_total

