import os

import numpy as np

module_dir = os.path.dirname(__file__)  # TODO find a cleaner way to do this?


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
        1. https://en.wikipedia.org/wiki/Propagation_of_uncertainty
        2. https://phas.ubc.ca/~oser/p509/Lec_10.pdf
        3. http://math.jacobs-university.de/oliver/teaching/jacobs/fall2015/esm106/handouts/error-propagation.pdf
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

        return np.sqrt(np.matmul(sigmas, np.matmul(covariance_matrix, np.transpose(sigmas))))
    elif np.ndim(uncertainties) == 2:
        sigma_less = derivatives * uncertainties[:, 0]
        sigma_more = derivatives * uncertainties[:, 1]

        return np.sqrt(np.array([  # beware, this is not strictly correct
            np.matmul(sigma_less, np.matmul(covariance_matrix, np.transpose(sigma_less))),
            np.matmul(sigma_more, np.matmul(covariance_matrix, np.transpose(sigma_more)))
        ]))


def calculate_chi2(data, model, uncertainties):
    return np.sum(((data - model) / uncertainties) ** 2)


def calculate_reduced_chi2(data, model, uncertainties, degrees_of_freedom=0):
    return calculate_chi2(data, model, uncertainties) / (np.size(data) - degrees_of_freedom - 1)


def mean_uncertainty(uncertainties):
    """Calculate the uncertainty of the mean of an array.

    Args:
        uncertainties: individual uncertainties of the averaged array

    Returns:
        The uncertainty of the mean of the array
    """
    return np.sqrt(np.sum(uncertainties ** 2)) / np.size(uncertainties)


def median_uncertainties(uncertainties):
    """Calculate the uncertainty of the median of an array.

    Demonstration:
        uncertainty ~ standard deviation = sqrt(variance) = sqrt(V)
        V_mean / V_median = 2 * (N - 1) / (pi * N); (see source)
        => V_median = V_mean * pi * N / (2 * (N - 1))
        => uncertainty_median = uncertainty_mean * sqrt(pi * N / (2 * (N - 1)))

    Source:
        https://mathworld.wolfram.com/StatisticalMedian.html

    Args:
        uncertainties: individual uncertainties of the median of the array

    Returns:
        The uncertainty of the median of the array
    """
    return mean_uncertainty(uncertainties) \
        * np.sqrt(np.pi * np.size(uncertainties) / (2 * (np.size(uncertainties) - 1)))
