import numpy as np


def class_init_args2class_args(string):
    arguments = string.split(',')
    out_string = ''

    for argument in arguments:
        arg = argument.strip().rsplit('=', 1)[0]
        out_string += f"self.{arg} = {arg}\n"

    return out_string


def class_init_args2dict(string):
    arguments = string.split(',')
    out_string = '{\n'

    for argument in arguments:
        arg = argument.strip().rsplit('=', 1)[0]
        out_string += f"    '{arg}': ,\n"

    out_string += '}'

    return out_string


def calculate_uncertainty(derivatives, uncertainties, covariance_matrix=None):
    """
    Calculate the uncertainty of a function f(x, y, ...) with uncertainties on x, y, ... and Pearson's correlation
    coefficients between x, y, ...
    The function must be (approximately) linear with its variables within the uncertainties of said variables.
    For independent variables, set the covariance matrix to identity.
    Uncertainties can be asymmetric, in that case for N variables, use a (N, 2) array for the uncertainties.
    Asymmetric uncertainties are handled **the wrong way** (see source 2), but it is better than nothing.

    Sources:
        - https://en.wikipedia.org/wiki/Propagation_of_uncertainty
        - https://phas.ubc.ca/~oser/p509/Lec_10.pdf
        - http://math.jacobs-university.de/oliver/teaching/jacobs/fall2015/esm106/handouts/error-propagation.pdf
    Args:
        derivatives: partial derivatives of the function with respect to each variables (df/dx, df/dy, ...)
        uncertainties: uncertainties of each variables (either a 1D-array or a 2D-array containing - and + unc.)
        covariance_matrix: covariance matrix between the variables, by default set to the identity matrix

    Returns:
        A size-2 array containing the - and + uncertainties of the function
    """
    if covariance_matrix is None:
        covariance_matrix = np.identity(np.size(derivatives))

    if np.ndim(uncertainties) == 1:
        sigmas = derivatives * uncertainties

        return np.matmul(sigmas, np.matmul(covariance_matrix, np.transpose(sigmas)))
    elif np.ndim(uncertainties) == 2:
        sigma_less = derivatives * uncertainties[:, 0]
        sigma_more = derivatives * uncertainties[:, 1]

        return np.sqrt(np.array([  # beware, this is not strictly correct
            np.matmul(sigma_less, np.matmul(covariance_matrix, np.transpose(sigma_less))),
            np.matmul(sigma_more, np.matmul(covariance_matrix, np.transpose(sigma_more)))
        ]))
