from __future__ import print_function

import sys

import argparse
import cplex
import numpy as np
from cplex.exceptions import CplexSolverError


def adapt_lp_file_to_numpy(filename):
    c = cplex.Cplex(filename)

    n = len(c.variables.get_names())
    binary_indicator = np.zeros(n)
    A_ineq = np.zeros((n, n))
    b_ineq = np.zeros(n)
    A_eq = np.zeros((n, n))
    b_eq = np.zeros(n)
    objective_function = 0
    gradient = 0
    lower_bounds = c.variables.get_lower_bounds()
    upper_bounds = c.variables.get_upper_bounds()

    for i, variable_type in enumerate(c.variables.get_types()):
        if variable_type == 'B':
            binary_indicator[i] = 1

    for i, constraint_type in enumerate(c.linear_constraints.get_senses()):
        if constraint_type == 'L':
            for k, j in enumerate(c.linear_constraints.get_rows(i).ind):
                A_ineq[i, j] = c.linear_constraints.get_rows(i).val[k]
            b_ineq[i] = c.linear_constraints.get_rhs(i)
        if constraint_type == 'E':
            for k, j in enumerate(c.linear_constraints.get_rows(i).ind):
                A_eq[i, j] = c.linear_constraints.get_rows(i).val[k]
            b_eq[i] = c.linear_constraints.get_rhs(i)

    print(A_ineq, A_eq, b_ineq, b_eq)


    print(binary_indicator)

    return objective_function, gradient, lower_bounds, upper_bounds, binary_indicator, A_eq, b_eq, A_ineq, b_ineq


def solve_with_cplex(filename):
    c = cplex.Cplex(filename)

    alg = c.parameters.lpmethod.values
    c.parameters.lpmethod.set(alg.auto)

    try:
        c.solve()
    except CplexSolverError:
        print("Exception raised during solve")
        return

    # solution.get_status() returns an integer code
    status = c.solution.get_status()
    print(c.solution.status[status])
    if status == c.solution.status.unbounded:
        print("Model is unbounded")
        return
    if status == c.solution.status.infeasible:
        print("Model is infeasible")
        return
    if status == c.solution.status.infeasible_or_unbounded:
        print("Model is infeasible or unbounded")
        return

    s_method = c.solution.get_method()
    s_type = c.solution.get_solution_type()

    print("Solution status = ", status, ":", end=' ')
    # the following line prints the status as a string
    print(c.solution.status[status])
    print("Solution method = ", s_method, ":", end=' ')
    print(c.solution.method[s_method])

    if s_type == c.solution.type.none:
        print("No solution available")
        return
    print("Objective value = ", c.solution.get_objective_value())

    if s_type == c.solution.type.basic:
        basis = c.solution.basis.get_basis()[0]
    else:
        basis = None

    print()

    x = c.solution.get_values(0, c.variables.get_num() - 1)
    # because we're querying the entire solution vector,
    # x = c.solution.get_values()
    # would have the same effect
    for j in range(c.variables.get_num()):
        print("Column %d: Value = %17.10g" % (j, x[j]))
        if basis is not None:
            if basis[j] == c.solution.basis.status.at_lower_bound:
                print("  Nonbasic at lower bound")
            elif basis[j] == c.solution.basis.status.basic:
                print("  Basic")
            elif basis[j] == c.solution.basis.status.at_upper_bound:
                print("  Nonbasic at upper bound")
            elif basis[j] == c.solution.basis.status.free_nonbasic:
                print("  Superbasic, or free variable at zero")
            else:
                print("  Bad basis status")

    infeas = c.solution.get_float_quality(
        c.solution.quality_metric.max_primal_infeasibility)
    print("Maximum bound violation = ", infeas)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename',
                        '-f',
                        default='toroidal2g20_5555.lp',
                        help='.lp file path',
                        type=str)
    args = parser.parse_args()

    adapt_lp_file_to_numpy(args.filename)
