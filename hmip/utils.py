import numpy as np


def smoothness_coefficient(H):
    """
    Compute the soothness coefficient with max(eig(H))
    :param H: (np.array) matrix of size (n, n), quadratic term of the problem
    :return: (np.float) scalar, smoothness coefficient
    """
    return np.absolute(np.max(np.linalg.eigvals(H)))


def projection(z, n, lb, ub):
    z[:n] = np.maximum(z[:n], lb)
    z[:n] = np.minimum(z[:n], ub)
    if len(z) > n:
        z[n:] = np.minimum(z[n:], np.zeros(len(z) - n))
    return z


def compute_approximate_smoothness_coef(gradient, lb, ub):
    n = len(lb)
    n_rand =  n
    smoothness_val_list = []
    for n_rand_trials in range(n_rand):
        point_1 = np.multiply(np.random.rand(n), ub - lb) + lb
        point_2 = np.multiply(np.random.rand(n), ub - lb) + lb
        distance = np.linalg.norm(point_1 - point_2)
        smoothness_val_list.append(np.linalg.norm(gradient(point_1) - gradient(point_2)) / distance)
    return np.max(smoothness_val_list)


def is_in_box(x, ub, lb):
    if x is None:
        return False
    elif np.all(np.greater(ub, x)) and np.all(np.greater(x, lb)):
        return True
    else:
        return False


def proxy_distance_vector_tanh(x, beta):
    """

    :param x: (np.array) variable
    :param beta: (np.array) size of x, parameter of the function
    :return:
    """
    tanh = 4 * np.multiply(np.multiply(beta, x), (1 - x))
    return tanh


def proxy_distance_vector_pwl(x, beta):
    """

    :param x: (np.array) variable
    :param beta: (np.array) size of x, parameter of the function
    :return:
    """
    pwl = np.zeros(len(x))
    for i in range(len(x)):
        if 0 < x[i] < 1:
            pwl[i] = beta[i]
    return pwl


def proxy_distance_vector_sin(x, beta):
    """

    :param x: (np.array) variable
    :param beta: (np.array) size of x, parameter of the function
    :return:
    """
    # TODO(Bertrand): check
    if np.isnan(x).any():
        return x
    if np.less_equal((1 - x), 0).any():
        x = x - 0.00001 * np.ones(len(x))
    x = np.maximum(np.zeros(len(x)), x)

    sin = 2 * np.multiply(np.multiply(beta, np.sqrt(x)), np.sqrt(1 - x))
    return sin


def proxy_distance_vector_exp(x, beta):
    """

    :param x: (np.array) variable
    :param beta: (np.array) size of x, parameter of the function
    :return:
    """
    exp = np.multiply(beta, np.minimum(1 - x, x))
    return exp


def proxy_distance_vector_identity(x, beta=None):
    """

    :param x: (np.array) variable
    :param beta: (np.array) size of x, parameter of the function
    :return:
    """
    id = np.zeros(len(x))
    return id


def activation_tanh(x, beta):
    """

    :param x: (np.array) variable
    :param beta: (np.array) size of x, parameter of the function
    :return:
    """
    tanh = 1 / 2 * (np.tanh(2 * np.multiply(beta, (x - 1 / 2))) + 1)
    return tanh


def activation_pwl(x, beta):
    """

    :param x: (np.array) variable
    :param beta: (np.array) size of x, parameter of the function
    :return:
    """
    pwl = np.maximum(
        np.zeros(len(x)),
        np.minimum(np.ones(len(x)),
                   np.multiply(beta, (x - 1 / 2)) + 1 / 2 * np.ones(len(x))))
    return pwl


def activation_sin(x, beta):
    """

    :param x: (np.array) variable
    :param beta: (np.array) size of x, parameter of the function
    :return:
    """
    sin = np.zeros(len(x))
    for i in range(len(x)):
        if x[i] > 1 / 2 + np.pi / (4 * beta[i]):
            sin[i] = 1
        elif x[i] < 1 / 2 - np.pi / (4 * beta[i]):
            sin[i] = 0
        else:
            sin[i] = 1 / 2 * np.sin(2 * beta[i] * (x[i] - 1 / 2)) + 1 / 2
    return sin


def activation_exp(x, beta):
    """

    :param x: (np.array) variable
    :param beta: (np.array) size of x, parameter of the function
    :return:
    """
    exp = np.zeros(len(x))
    for i in range(len(x)):
        if x[i] > 1 / 2:
            exp[i] = 1 - np.exp(2 * beta[i] * (0.5 - x[i]) - np.log(2))
        else:
            exp[i] = np.exp(2 * beta[i] * (x[i] - 1 / 2) - np.log(2))
    return exp


def activation_identity(x, beta=None):
    """

    :param x: (np.array) variable
    :param beta: (np.array) size of x, parameter of the function
    :return:
    """
    return x


def normalize_array(array):
    """

    :param array:
    :return:
    """
    norm = np.linalg.norm(array)
    if norm != 0:
        array = array / norm
    return array


def inverse_activation_pwl(x, beta):
    """

    :param x: (np.array) variable
    :param beta: (np.array) size of x, parameter of the function
    :return:
    """
    pwl = np.ones(len(x))
    for i in range(len(x)):
        if 0 <= x[i] <= 1:
            pwl[i] = (beta[i]) ** (-1) * (x[i] - 1 / 2) + 1 / 2
        elif x[i] < 0:
            pwl[i] = 0
    return pwl


def inverse_activation_tanh(x, beta):
    """

    :param x: (np.array) variable
    :param beta: (np.array) size of x, parameter of the function
    :return:
    """
    tanh = np.divide(np.arctanh(2 * x - 1), 2 * beta) + 1 / 2
    return tanh


def inverse_activation_sin(x, beta):
    """
    compute the inverse of the activation function sin
    :param x: (np.array) variable
    :param beta: (np.array) size of x, parameter of the function
    :return: inverse of the activation function sin
    """
    sin = np.ones(len(x))
    for i in range(len(x)):
        if 0 <= x[i] <= 1:
            sin[i] = (1 / (beta[i] * 2)) * np.arcsin(2 * x[i] - 1) + 1 / 2
        elif x[i] < 0:
            sin[i] = 0
    return sin


def inverse_activation_exp(x, beta):
    """

    :param x: (np.array) variable
    :param beta: (np.array) size of x, parameter of the function
    :return:
    """
    exp = np.ones(len(x))
    for i in range(len(x)):
        if 0 <= x[i] < 1 / 2:
            exp[i] = 1 / 2 + (1 / (2 * beta[i])) * np.log(2 * x[i])
        elif x[i] <= 0:
            exp[i] = 0
        else:
            exp[i] = 1 / 2 - (1 / (2 * beta[i])) * np.log(2 * (1 - x[i]))
    return exp


def inverse_activation_identity(x, beta=None):
    """

    :param x: (np.array) variable
    :param beta: (np.array) size of x, parameter of the function
    :return:
    """
    return x


def check_type(n,
               lb=None,
               ub=None,
               binary_indicator=None,
               L=None,
               k_max=None,
               absorption_val=None,
               gamma=None,
               theta=None,
               initial_state=None,
               beta=None,
               absorption=None,
               step_type=None,
               direction_type=None,
               activation_type=None,
               initial_ascent_type=None):
    """

    :param n: (integer)
    :param lb: (numpy.ndarray)
    :param ub: (numpy.ndarray)
    :param binary_indicator: (numpy.ndarray)
    :param L: (numpy.ndarray)
    :param k_max: (integer)
    :param absorption_val: (float)
    :param gamma: (float)
    :param theta: (float)
    :param initial_state: (numpy.ndarray)
    :param beta: (numpy.ndarray)
    :param absorption: (boolean)
    :param step_type: (string)
    :param direction_type: (string)
    :param activation_type: (string)
    :param initial_ascent_type: (string)
    :return:
    """

    if initial_state is not None:
        if isinstance(initial_state, np.ndarray):
            return len(initial_state) == n
        else:
            print('initial_state is not of the correct type or dim')
            return False

    if lb is not None:
        if isinstance(lb, np.ndarray):
            return len(lb) == n
        else:
            print('lb is not of the correct type or dim')
            return False

    if ub is not None:
        if isinstance(ub, np.ndarray):
            return len(ub) == n
        else:
            return False

    if binary_indicator is not None:
        if isinstance(binary_indicator, np.ndarray):
            for i in binary_indicator:
                if i != 0 and i != 1:
                    return False
            return len(binary_indicator) == n
        else:
            return False

    # TODO(Mathilde): check this one with Bertrand
    if L is not None:
        if isinstance(L, float):
            return True
        else:
            return False

    if absorption_val is not None:
        return isinstance(absorption_val, float)

    if gamma is not None:
        return isinstance(gamma, float)

    if theta is not None:
        return isinstance(theta, float)

    if beta is not None:
        if isinstance(beta, np.ndarray):
            return len(beta) == n
        else:
            return False

    if k_max is not None:
        return isinstance(k_max, float)

    if absorption is not None:
        return isinstance(ub, bool)

    # TODO(Mathilde): Maybe check if they are in possible options?
    if step_type is not None:
        return isinstance(step_type, str)

    if direction_type is not None:
        return isinstance(direction_type, str)

    if activation_type is not None:
        return isinstance(activation_type, str)

    if initial_ascent_type is not None:
        return isinstance(initial_ascent_type, str)

    print('Add a variable to check')
    return None


def make_symmetric(matrix):
    """
    Check if the matrix is symmetric, if no it returns a new symmetric matrix
    :param matrix: (np.array) size (n, n)
    :return: (np.array) size (n, n) symmetric
    """
    if not np.allclose(matrix, matrix.T, atol=0):
        matrix = 1 / 2 * (matrix + matrix.T)
        print(
            'Specified matrix H was not symmetric, matrix H has been replaced by 1/2 * (matrix + matrix.transpose)'
        )
    return matrix


def adapt_ascent_stop_criterion(ascent_stop, absorption):
    """
    :param ascent_stop:
    :param absorption:
    :return:
    """
    if absorption is not None and ascent_stop is not None and ascent_stop <= absorption:
        ascent_stop = absorption * 2
        print(
            'Choice of initial ascent stopping criterion was smaller than the '
            'chosen absorption value, ascent_stop was taken to be absorption * 2'
        )
    return ascent_stop


def assess_convexity_of_objective(H):
    """
    :return: False if not convex, true if convex
    """
    m = min(np.linalg.eigvals(H))
    if m < 0:
        return False
    else:
        return True


def parser_mps_file(file_path):
    fixed = True
    sections = dict({
        "NAME": False,
        "ROWS": False,
        "COLUMNS": False,
        "RHS": False,
        "BOUNDS": False,
        "RANGES": False,
        "SOS": False,
        "ENDATA": False
    })
    if not fixed:
        sections["OBJSENSE"] = False
        sections["OBJNAME"] = False

    var_names = dict()
    con_names = dict()
    var_types = []  # true : float   False : int
    con_types = []  # 1 : <=, 2 : ==, 3 : >=

    # Default values
    name = ""
    objective_name = ""
    objsense = 'min'
    n_var = 0
    n_con = 0
    A = None
    b = None
    c = None
    c0 = 0.0
    bounds = []

    f = open(file_path, 'r')
    while not sections["ENDATA"]:
        line = f.readline()
        # Read the section name and avoid empty lines and comments
        while line.strip() == "" or line[1] == '*':
            line = f.readline()
        words = line.split()
        section = words[0]

        # Errors in sections names and order
        if not section in sections.keys():
            raise Exception("Section ", section, " is not a valid section")
        elif sections[section]:
            raise Exception("Section ", section, " appears twice")
        elif section == "COLUMNS" and not sections["ROWS"]:
            raise Exception("ROWS must come before COLUMNS")
        elif section == "RHS" and (not sections["COLUMNS"]
                                   or not sections["NAME"]):
            raise Exception("NAME and COLUMNS must come before RHS")
        elif section == "BOUNDS" and not sections["COLUMNS"]:
            raise Exception("COLUMNS must come before BOUNDS")
        elif section == "RANGES" and not sections["RHS"]:
            raise Exception("RHS must come before RANGES")

        # Read info
        if section == "NAME":
            name = words[1]
        elif section == "OBJSENSE":
            objsense = read_objsense(f)
        elif section == "OBJNAME":
            objective_name = read_objname(f)
        elif section == "ROWS":
            objective_name, con_names, con_types = read_rows(f, objective_name, fixed)
            n_con = len(con_types)
        elif section == "COLUMNS":
            var_names, c, A, var_types = read_columns(f, objective_name, con_names, fixed)
        elif section == "RHS":
            b, c0 = read_rhs(f, objective_name, con_names, fixed)
        elif section == "RANGES":
            A, b, con_types = read_ranges(f, A, b, con_names, con_types, fixed)
            n_con = len(con_types)
        elif section == "BOUNDS":
            bounds, var_types = read_bounds(f, var_names, var_types, fixed)
        elif section == "SOS":
            raise Exception("SOS section reader not ready yet")

        sections[section] = True

    if not sections["ENDATA"]:
        raise Exception("No ENDATA section in the file")

    return var_types, bounds, objsense, np.array(c), np.array(c0), np.array(A), np.array(b), con_types


def read_objsense(f):
    objective_sense = ["MIN", "MAX"]
    line = f.readline()
    while line.strip() == "" or line[0] == '*':
        line = f.readline()
    if not (line.strip() in objective_sense):
        raise Exception("No valid information in OBJSENSE section")
    f.seek(f.tell())
    if line.strip() == "MIN":
        return 'min'
    else:
        return 'max'


def read_objname(f):
    line = f.readline()
    while line.strip() == "" or line[0] == '*':
        line = f.readline()
    f.seek(f.tell())
    return line.strip()


def read_rows(f, objective_name, fixed):
    pos = f.tell()
    objective = False
    con_names = dict()
    con_types = []  # 1 : <=, 2 : ==, 3 : >=
    n_con = 0

    line = f.readline()
    while line.strip() == "" or line[1] == '*':
        line = f.readline()
    while line[0] == ' ':
        pos = f.tell()
        if fixed:
            type_tmp = line[1:2].strip()
            name = line[4:min(len(line), 11)].strip()
        else:
            type_tmp, name = line.split()
            type_tmp = type_tmp.upper()

        if type_tmp == "N":
            if objective:
                raise Exception("MultiObjectives")
            else:
                objective = True
                objective_name = name
        else:
            if type_tmp == "G":
                con_types.append(2)
            elif type_tmp == "L":
                con_types.append(0)
            elif type_tmp == "E":
                con_types.append(1)
            else:
                raise Exception("Error in type of the row")
            con_names[name] = n_con
            n_con += 1

        line = f.readline()
        while line.strip() == "" or line[0] == '*':
            pos = f.tell()
            line = f.readline()

    f.seek(pos)
    return objective_name, con_names, con_types


def read_columns(f, objective_name, con_names, fixed):
    pos = f.tell()
    var_names = dict()
    var_types = []  # true : float   false : int
    n_con = len(con_names.keys())
    A = None
    init_var = np.zeros((n_con, 1))
    c = []
    n_var = 0
    is_float = True

    line = f.readline()
    while line.strip() == "" or line[0] == '*':
        line = f.readline()
    while line[0] == ' ':
        pos = f.tell()
        words = line.split()
        len_words = len(words)

        # MARKER for INT
        if words[1] == "'MARKER'":
            ind = 2
            if fixed:
                ind = 3
            if words[ind] == "'INTORG'":
                is_float = False
            elif words[ind] == "'INTEND'":
                is_float = True
            else:
                raise Exception("Unknown MARKER in MPS file")
        else:
            name = words[0]
            if name not in var_names.keys():
                var_names[name] = n_var
                n_var += 1
                if A is None:
                    A = np.copy(init_var)
                else:
                    A = np.concatenate((A, init_var), axis=1)
                var_types.append(is_float)
                c.append(0.0)

        for i in range(1, len_words-1, 2):
            if words[i] == objective_name:
                # Case add to objective
                c[var_names[name]] = float(words[i + 1])
            else:
                # Case add to A
                A[con_names[words[i]], var_names[name]] = float(words[i + 1])

        line = f.readline()
        while line.strip() == "" or line[1] == '*':
            pos = f.tell()
            line = f.readline()

    f.seek(pos)
    return var_names, c, A, var_types


def read_rhs(f, objective_name, con_names, fixed):
    pos = f.tell()
    b = np.zeros(len(con_names.keys()))
    c0 = 0.0

    line = f.readline()
    while line.strip() == "" or line[0] == '*':
        line = f.readline()
    while line[0] == ' ':
        pos = f.tell()
        words = line.split()
        len_words = len(words)

        for i in range(1, len_words-1, 2):
            name = words[i]
            if name == objective_name:
                if c0 != 0:
                    raise Exception("Multiple presence of ", objective_name, " in RHS")
                else:
                    c0 = - float(words[i + 1])
            elif name != "":
                if b[con_names[name]] != 0:
                    raise Exception("Multiple presence of ", name, " in RHS")
                else:
                    b[con_names[name]] = float(words[i + 1])

        line = f.readline()
        while line.strip() == "" or line[0] == '*':
            pos = f.tell()
            line = f.readline()

    f.seek(pos)
    return b, c0


def read_ranges(f, A_init, b_init, con_names, con_types_init, fixed):
    A = np.copy(A_init)
    b = np.copy(b_init)
    con_types = np.copy(con_types_init)
    pos = f.tell()

    line = f.readline()
    while line.strip() == "" or line[0] == '*':
        line = f.readline()
    while line[0] == ' ':
        pos = f.tell()
        words = line.split()
        len_words = len(words)

        for i in range(1, len_words-1, 2):
            name = words[i]
            if name != "":
                if con_types[con_names[name]] == 2:
                    raise Exception("Range on an equality constraint")
                elif con_types[con_names[name]] == 1:
                    con_types.append(3)
                    b0 = -abs(float(words[i + 1]))
                elif con_types[con_names[name]] == 3:
                    con_types.append(1)
                    b0 = abs(float(words[i + 1]))
                else:
                    raise Exception("Should never happen")
    A = np.vstack((A, A[con_names[name], :]))
    b.append(b0)

    line = f.readline()

    while line.strip() == "" or line[0] == '*':
        pos = f.tell()
        line = f.readline()

    f.seek(pos)
    return A, b, con_types

def read_bounds(f, var_names, var_typesinit, fixed):
    var_types = np.copy(var_typesinit)
    pos = f.tell()
    bounds = [(-np.inf, np.inf) for i in range(len(var_names.keys()))]

    line = f.readline()
    while line.strip() == "" or line[0] == '*':
        line = f.readline()
    while line[0] == ' ':
        pos = f.tell()
        words = line.split()

        bound_type = words[0]
        if not fixed:
            bound_type = bound_type.upper()

        name = words[2]

        if bound_type == "FR":
            bounds[var_names[name]] = (-np.inf, np.inf)
        elif bound_type == "MI":
            # Lower bound -inf
            if np.isfinite(bounds[var_names[name]][1]):
                bounds[var_names[name]] = (-np.inf, bounds[var_names[name]][1])
            else:
                bounds[var_names[name]] = (-np.inf, 0.0)
        elif bound_type == "PL":
            # Default value
            bounds[var_names[name]] = (bounds[var_names[name]][0], np.inf)
        elif bound_type == "BV":
            # Binary variable
            bounds[var_names[name]] = (0.0, 1.0)
            var_types[var_names[name]] = False
        elif bound_type == "SC":
            # Semi-continuous variable
            raise Exception("SC bound not ready yet")
        elif bound_type == "LO":
            # Lower bound
            bound_val = float(words[3])
            bounds[var_names[name]] = (bound_val, bounds[var_names[name]][1])
        elif bound_type == "UP":
            # Upper bound
            bound_val = float(words[3])
            bounds[var_names[name]] = (bounds[var_names[name]][0], bound_val)
        elif bound_type == "FX":
            # Fixed variable
            bound_val = float(words[3])
            bounds[var_names[name]] = (bound_val, bound_val)
        elif bound_type == "LI":
            # Integer variable / Lower bound
            bound_val = float(words[3])
            bounds[var_names[name]] = (bound_val, bounds[var_names[name]][1])
            var_types[var_names[name]] = False
        elif bound_type == "UI":
            # Integer variable / Upper bound
            bound_val = float(words[3])
            bounds[var_names[name]] = (bounds[var_names[name]][0], bound_val)
            var_types[var_names[name]] = False
        else:
            raise Exception("Unknown bound type ", bound_type)

        line = f.readline()
        while line.strip() == "" or line[0] == '*':
            pos = f.tell()
            line = f.readline()

    f.seek(pos)
    return bounds, var_types


def remove_nan_results(x):
    """
    remove nan values at the end of x
    :param x: (np.array) variable dimension n
    :return: (np.array) dimension <= n
    """
    stop_index = len(x)
    for i in range(x.shape[1]):
        if np.isnan(x[:, i]).any():
            stop_index = i
            break
    x_refactor = x[:, :stop_index]
    return x_refactor
