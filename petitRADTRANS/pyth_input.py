from __future__ import print_function

import numpy as np
from . import nat_cst as nc


def sigma_hm_ff(lambda_angstroem, temp, P_e):
    """
    Returns the H- free-free cross-section in units of cm^2
    per H per e- pressure (in cgs), as defined on page 156 of
    "The Observation and Analysis of Stellar Photospheres"
    by David F. Gray
    """

    index = (lambda_angstroem >= 2600.) & (lambda_angstroem <= 113900.)
    lamb_use = lambda_angstroem[index]

    if temp >= 2500.:
        # Convert to Angstrom (from cgs)
        theta = 5040. / temp

        f0 = -2.2763 - 1.6850 * np.log10(lamb_use) + \
          0.76661*np.log10(lamb_use)**2. \
          - 0.053346*np.log10(lamb_use)**3.
        f1 = 15.2827 - 9.2846 * np.log10(lamb_use) + \
          1.99381*np.log10(lamb_use)**2. \
          - 0.142631*np.log10(lamb_use)**3.
        f2 = -197.789 + 190.266 * np.log10(lamb_use) - 67.9775*np.log10(lamb_use)**2. \
                 + 10.6913*np.log10(lamb_use)**3. - 0.625151*np.log10(lamb_use)**4.

        ret_val = np.zeros_like(lambda_angstroem)
        ret_val[index] = 1e-26 * P_e * 1e1 ** (
                f0 + f1 * np.log10(theta) + f2 * np.log10(theta) ** 2.)
        return ret_val

    else:

        return np.zeros_like(lambda_angstroem)


def sigma_bf_mean(border_lambda_angstroem):
    """
    Returns the H- bound-free cross-section in units of cm^2 \
    per H-, as defined on page 155 of
    "The Observation and Analysis of Stellar Photospheres"
    by David F. Gray
    """

    left = border_lambda_angstroem[:-1]
    right = border_lambda_angstroem[1:]
    diff = np.diff(border_lambda_angstroem)

    a = [
        1.99654,
        -1.18267e-5,
        2.64243e-6,
        -4.40524e-10,
        3.23992e-14,
        -1.39568e-18,
        2.78701e-23
    ]

    ret_val = np.zeros_like(border_lambda_angstroem[1:])

    index = right <= 1.64e4

    for i_a in range(len(a)):
        ret_val[index] += a[i_a] * (
                right[index] ** (i_a + 1) - left[index] ** (i_a + 1)
        ) / (i_a + 1)

    index_bracket = (left < 1.64e4) & (right > 1.64e4)
    for i_a in range(len(a)):
        ret_val[index_bracket] += a[i_a] * (1.64e4 ** (i_a + 1) -
                                            left[index_bracket] ** (i_a + 1)) / (i_a + 1)

    index = (left + right) / 2. > 1.64e4
    ret_val[index] = 0.
    index = ret_val < 0.
    ret_val[index] = 0.

    return ret_val * 1e-18 / diff


def hminus_opacity(lambda_angstroem, border_lambda_angstroem,
                   temp, press, mmw, abundances):
    """ Calc the H- opacity."""

    ret_val = np.array(np.zeros(len(lambda_angstroem) * len(press)).reshape(
        len(lambda_angstroem),
        len(press)), dtype='d', order='F')

    # Calc. electron number fraction
    # e- mass in amu:
    m_e = 5.485799e-4
    n_e = mmw / m_e * abundances['e-']

    # Calc. e- partial pressure
    p_e = press * n_e

    kappa_hminus_bf = sigma_bf_mean(border_lambda_angstroem) / nc.amu

    for i_struct in range(len(n_e)):
        kappa_hminus_ff = sigma_hm_ff(lambda_angstroem, temp[i_struct],
                                      p_e[i_struct]) / nc.amu * abundances['H'][i_struct]

        ret_val[:, i_struct] = kappa_hminus_bf * abundances['H-'][i_struct] \
            + kappa_hminus_ff

    return ret_val


# Functions to read custom PT grids
# Function to sort custom (potentially randomly sorted) PT grid of opacities
def sort_opa_PTgrid(path_ptg):
    # Read the Ps and Ts
    p_ts = np.genfromtxt(path_ptg)

    # Read the file names
    f = open(path_ptg)
    lines = f.readlines()
    f.close()

    n_entries = len(lines)

    # Prepare the array to contain the
    # pressures, temperatures, indices in the unsorted list.
    # Also prepare the list of unsorted names
    p_tind = np.ones(n_entries * 3).reshape(n_entries, 3)
    names = []

    # Fill the array and name list
    for i_line in range(n_entries):

        line = lines[i_line]
        lsp = line.split(' ')

        p_tind[i_line, 0], p_tind[i_line, 1], p_tind[i_line, 2] = \
            p_ts[i_line, 0], p_ts[i_line, 1], i_line

        if lsp[-1][-1] == '\n':
            names.append(lsp[-1][:-1])
        else:
            names.append(lsp[-1])

    # Sort the array by temperature
    tsortind = np.argsort(p_tind[:, 1])
    p_tind = p_tind[tsortind, :]

    # Sort the array entries with constant
    # temperatures by pressure
    diff_ps = 0
    t_start = p_tind[0, 1]

    for i in range(n_entries):
        if np.abs(p_tind[i, 1] - t_start) > 1e-10:
            break
        diff_ps = diff_ps + 1

    diff_ts = int(n_entries / diff_ps)
    for i_dT in range(diff_ts):
        subsort = p_tind[i_dT * diff_ps:(i_dT + 1) * diff_ps, :]
        psortind = np.argsort(subsort[:, 0])
        subsort = subsort[psortind, :]
        p_tind[i_dT * diff_ps:(i_dT + 1) * diff_ps, :] = subsort

    names_sorted = []
    for i_line in range(n_entries):
        names_sorted.append(names[int(p_tind[i_line, 2] + 0.01)])

    # Convert from bars to cgs
    p_tind[:, 0] = p_tind[:, 0] * 1e6

    return [p_tind[:, :-1][:, ::-1], names_sorted, diff_ts, diff_ps]


# Check if custom grid exists, if yes return sorted P-T array with
# corresponding sorted path_input_data names, retutn None otherwise.

def get_custom_PT_grid(path, mode, species):
    import os as os

    path_test = path + '/opacities/lines/'
    if mode == 'lbl':
        path_test = path_test + 'line_by_line/'
    elif mode == 'c-k':
        path_test = path_test + 'corr_k/'
    path_test = path_test + species + '/PTpaths.ls'
    if not os.path.isfile(path_test):
        return None
    else:
        return sort_opa_PTgrid(path_test)


'''
import pylab as plt

demo_temp = 2880
demo_press = 0.33*1e6
ab_em = 1e-6*5.485799e-4/2.33
ab_h = 0.33*1./2.33
ab_hm = 2e-9*1./2.33
mww_val = 2.33

abunds = {}
abunds['e-'] = np.array([ab_em,ab_em,ab_em])
abunds['H'] = np.array([ab_h,ab_h,ab_h])
abunds['H-'] = np.array([ab_hm,ab_hm,ab_hm])

temp = np.array([demo_temp, demo_temp, demo_temp])
press = np.array([demo_press, demo_press, demo_press])
mmw = np.array([mww_val, mww_val, mww_val])

lamb = np.logspace(np.log10(0.9),np.log10(10.),7000)*1e4
lamb_coarse = np.logspace(np.log10(0.9),np.log10(10.),10)*1e4

def calc_borders(x):
        xn = []
        xn.append(x[0]-(x[1]-x[0])/2.)
        for i in range(int(len(x))-1):
            xn.append(x[i]+(x[i+1]-x[i])/2.)
        xn.append(x[int(len(x))-1]+(x[int(len(x))-1]-x[int(len(x))-2])/2.)
        return np.array(xn)

lamb_bord = calc_borders(lamb)
print('a')
for i in range(33):
    opa = hminus_opacity(lamb, lamb_bord, temp, press, mmw, abunds)
print('b')

plt.plot(lamb/1e4, opa*2.33*nc.amu)

lamb_coarse_bord = calc_borders(lamb_coarse)
opa = hminus_opacity(lamb_coarse, lamb_coarse_bord, temp, press, mmw, abunds)
plt.plot(lamb_coarse/1e4, opa*2.33*nc.amu)

plt.ylim([1e-28,1e-22])
plt.xscale('log')
plt.yscale('log')
plt.show()
'''
