"""Stores useful generic functions.
"""

import numpy as np


def box_car_conv(array, points):
    res = np.zeros_like(array)
    len_arr = len(array)

    for i in range(len(array)):
        if (i - points / 2 >= 0) and (i + points / 2 <= len_arr + 1):
            smooth_val = array[i - points / 2:i + points / 2]
            res[i] = np.sum(smooth_val) / len(smooth_val)
        elif i + points / 2 > len_arr + 1:
            len_use = len_arr + 1 - i
            smooth_val = array[i - len_use:i + len_use]
            res[i] = np.sum(smooth_val) / len(smooth_val)
        elif i - points / 2 < 0:
            smooth_val = array[:max(2 * i, 1)]
            res[i] = np.sum(smooth_val) / len(smooth_val)
    return res


logs_g = np.array([12., 10.93])

logs_met = np.array([1.05, 1.38, 2.7, 8.43, 7.83, 8.69, 4.56, 7.93, 6.24, 7.6, 6.45, 7.51, 5.41,
                     7.12, 5.5, 6.4, 5.03, 6.34, 3.15, 4.95, 3.93, 5.64, 5.43, 7.5,
                     4.99, 6.22, 4.19, 4.56, 3.04, 3.65, 3.25, 2.52, 2.87, 2.21, 2.58,
                     1.46, 1.88])


def calc_met(f):
    return np.log10((f / (np.sum(1e1 ** logs_g) + f * np.sum(1e1 ** logs_met)))
                    / (1. / (np.sum(1e1 ** logs_g) + np.sum(1e1 ** logs_met))))


def read_abunds(path):
    f = open(path)
    header = f.readlines()[0][:-1]
    f.close()
    ret = {}

    dat = np.genfromtxt(path)
    ret['P'] = dat[:, 0]
    ret['T'] = dat[:, 1]
    ret['rho'] = dat[:, 2]

    for i in range(int((len(header) - 21) / 22)):
        if i % 2 == 0:
            name = header[21 + i * 22:21 + (i + 1) * 22][3:].replace(' ', '')
            number = int(header[21 + i * 22:21 + (i + 1) * 22][0:3])
            # print(name)
            ret['m' + name] = dat[:, number]
        else:
            name = header[21 + i * 22:21 + (i + 1) * 22][3:].replace(' ', '')
            number = int(header[21 + i * 22:21 + (i + 1) * 22][0:3])
            # print(name)
            ret['n' + name] = dat[:, number]

    return ret
