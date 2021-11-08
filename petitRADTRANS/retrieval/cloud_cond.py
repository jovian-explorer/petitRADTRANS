import copy as cp

import numpy as np
from scipy.interpolate import interp1d

plotting = False

if plotting:
    import pylab as plt

#############################################################
# Cloud Cond
#############################################################
# This file allows the calculation of equilibrium cloud abundances
# and base pressures
#
# TODO: Make a better cloud module.

#############################################################
# To calculate X_Fe from [Fe/H], C/O
#############################################################

# metal species
metals = ['C', 'N', 'O', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'K', 'Ca', 'Ti', 'V', 'Fe', 'Ni']

# solar abundances, [Fe/H] = 0, from Asplund+ 2009
nfracs = {
    'H': 0.9207539305,
    'He': 0.0783688694,
    'C': 0.0002478241,
    'N': 6.22506056949881e-05,
    'O': 0.0004509658,
    'Na': 1.60008694353205e-06,
    'Mg': 3.66558742055362e-05,
    'Al': 2.595e-06,
    'Si': 2.9795e-05,
    'P': 2.36670201997668e-07,
    'S': 1.2137900734604e-05,
    'Cl': 2.91167958499589e-07,
    'K': 9.86605611925677e-08,
    'Ca': 2.01439011429255e-06,
    'Ti': 8.20622804366359e-08,
    'V': 7.83688694089992e-09,
    'Fe': 2.91167958499589e-05,
    'Ni': 1.52807116806281e-06
}

# atomic masses  TODO use molmass instead
masses = {
    'H': 1.,
    'He': 4.,
    'C': 12.,
    'N': 14.,
    'O': 16.,
    'Na': 23.,
    'Mg': 24.3,
    'Al': 27.,
    'Si': 28.,
    'P': 31.,
    'S': 32.,
    'Cl': 35.45,
    'K': 39.1,
    'Ca': 40.,
    'Ti': 47.9,
    'V': 51.,
    'Fe': 55.8,
    'Ni': 58.7
}


def return_XFe(FeH, CO):
    nfracs_use = cp.copy(nfracs)

    for spec in nfracs.keys():

        if (spec != 'H') and (spec != 'He'):
            nfracs_use[spec] = nfracs[spec] * 1e1 ** FeH

    nfracs_use['O'] = nfracs_use['C'] / CO

    x_fe = masses['Fe'] * nfracs_use['Fe']
    add = 0.
    for spec in nfracs_use.keys():
        add += masses[spec] * nfracs_use[spec]

    x_fe = x_fe / add

    return x_fe


def return_XMgSiO3(FeH, CO):
    nfracs_use = cp.copy(nfracs)

    for spec in nfracs.keys():

        if (spec != 'H') and (spec != 'He'):
            nfracs_use[spec] = nfracs[spec] * 1e1 ** FeH

    nfracs_use['O'] = nfracs_use['C'] / CO

    nfracs_mgsio3 = np.min([nfracs_use['Mg'],
                            nfracs_use['Si'],
                            nfracs_use['O'] / 3.])
    masses_mgsio3 = masses['Mg'] \
        + masses['Si'] \
        + 3. * masses['O']

    xmgsio3 = masses_mgsio3 * nfracs_mgsio3
    add = 0.
    for spec in nfracs_use.keys():
        add += masses[spec] * nfracs_use[spec]

    xmgsio3 = xmgsio3 / add

    return xmgsio3


def return_XNa2S(FeH, CO):
    nfracs_use = cp.copy(nfracs)

    for spec in nfracs.keys():

        if (spec != 'H') and (spec != 'He'):
            nfracs_use[spec] = nfracs[spec] * 1e1 ** FeH

    nfracs_use['O'] = nfracs_use['C'] / CO

    nfracs_na2s = np.min([nfracs_use['Na'] / 2.,
                          nfracs_use['S']])
    masses_na2s = 2. * masses['Na'] \
        + masses['S']

    xna2s = masses_na2s * nfracs_na2s
    add = 0.
    for spec in nfracs_use.keys():
        add += masses[spec] * nfracs_use[spec]

    xna2s = xna2s / add

    return xna2s


def return_XKCL(FeH, CO):
    nfracs_use = cp.copy(nfracs)

    for spec in nfracs.keys():

        if (spec != 'H') and (spec != 'He'):
            nfracs_use[spec] = nfracs[spec] * 1e1 ** FeH

    nfracs_use['O'] = nfracs_use['C'] / CO

    nfracs_kcl = np.min([nfracs_use['K'],
                         nfracs_use['Cl']])
    masses_kcl = masses['K'] \
        + masses['Cl']

    xkcl = masses_kcl * nfracs_kcl
    add = 0.
    for spec in nfracs_use.keys():
        add += masses[spec] * nfracs_use[spec]

    xkcl = xkcl / add

    return xkcl


#############################################################
# Fe saturation pressure, from Ackerman & Marley (2001), including erratum (P_vap is in bar, not cgs!)
#############################################################

def return_T_cond_Fe(FeH, CO, MMW=2.33):
    t = np.linspace(100., 10000., 1000)
    # Taken from Ackerman & Marley (2001)
    # including their erratum

    def p_vap(x):
        return np.exp(15.71 - 47664. / x)

    x_fe = return_XFe(FeH, CO)

    return p_vap(t) / (x_fe * MMW / masses['Fe']), t


def return_T_cond_Fe_l(FeH, CO, MMW=2.33):
    t = np.linspace(100., 10000., 1000)
    # Taken from Ackerman & Marley (2001)
    # including their erratum

    def p_vap(x):
        return np.exp(9.86 - 37120. / x)

    x_fe = return_XFe(FeH, CO)

    return p_vap(t) / (x_fe * MMW / masses['Fe']), t


def return_T_cond_Fe_comb(FeH, CO, MMW=2.33):
    p1, t1 = return_T_cond_Fe(FeH, CO, MMW)
    p2, t2 = return_T_cond_Fe_l(FeH, CO, MMW)

    ret_p = np.zeros_like(p1)
    index = p1 < p2
    ret_p[index] = p1[index]
    ret_p[~index] = p2[~index]
    return ret_p, t2


def return_T_cond_Fe_free(XFe, MMW=2.33):
    t = np.linspace(100., 10000., 1000)
    # Taken from Ackerman & Marley (2001)
    # including their erratum

    def p_vap(x):
        return np.exp(15.71 - 47664. / x)

    return p_vap(t) / (XFe * MMW / masses['Fe']), t


def return_T_cond_Fe_l_free(XFe, MMW=2.33):
    t = np.linspace(100., 10000., 1000)
    # Taken from Ackerman & Marley (2001)
    # including their erratum

    def p_vap(x):
        return np.exp(9.86 - 37120. / x)

    return p_vap(t) / (XFe * MMW / masses['Fe']), t


def return_T_cond_Fe_comb_free(XFe, MMW=2.33):
    p1, t1 = return_T_cond_Fe_free(XFe, MMW)
    p2, t2 = return_T_cond_Fe_l_free(XFe, MMW)
    ret_p = np.zeros_like(p1)
    index = p1 < p2
    ret_p[index] = p1[index]
    ret_p[~index] = p2[~index]
    return ret_p, t2


def return_T_cond_MgSiO3(FeH, CO, MMW=2.33):
    t = np.linspace(100., 10000., 1000)
    # Taken from Ackerman & Marley (2001)
    # including their erratum

    def p_vap(x):
        return np.exp(25.37 - 58663. / x)

    xmgsio3 = return_XMgSiO3(FeH, CO)

    m_mgsio3 = masses['Mg'] \
        + masses['Si'] \
        + 3. * masses['O']
    return p_vap(t) / (xmgsio3 * MMW / m_mgsio3), t


def return_T_cond_MgSiO3_free(Xmgsio3, MMW=2.33):
    t = np.linspace(100., 10000., 1000)
    # Taken from Ackerman & Marley (2001)
    # including their erratum

    def p_vap(x):
        return np.exp(25.37 - 58663. / x)

    m_mgsio3 = masses['Mg'] \
        + masses['Si'] \
        + 3. * masses['O']
    return p_vap(t) / (Xmgsio3 * MMW / m_mgsio3), t


def return_T_cond_Na2S(FeH, CO, MMW=2.33):
    # Taken from Charnay+2018
    t = np.linspace(100., 10000., 1000)
    # This is the partial pressure of Na, so
    # Divide by factor 2 to get the partial
    # pressure of the hypothetical Na2S gas
    # particles, this is OK: there are
    # more S than Na atoms at solar
    # abundance ratios.

    def p_vap(x):
        return 1e1 ** (8.55 - 13889. / x - 0.5 * FeH) / 2.

    xna2s = return_XNa2S(FeH, CO)

    m_na2s = 2. * masses['Na'] \
        + masses['S']
    return p_vap(t) / (xna2s * MMW / m_na2s), t


def return_T_cond_Na2S_free(Xna2s, FeH, MMW=2.33):
    # Taken from Charnay+2018
    t = np.linspace(100., 10000., 1000)
    # This is the partial pressure of Na, so
    # Divide by factor 2 to get the partial
    # pressure of the hypothetical Na2S gas
    # particles, this is OK: there are
    # more S than Na atoms at solar
    # abundance ratios.

    def p_vap(x):
        return 1e1 ** (8.55 - 13889. / x - 0.5 * FeH) / 2.

    m_na2s = 2. * masses['Na'] \
        + masses['S']

    return p_vap(t) / (Xna2s * MMW / m_na2s), t


def return_T_cond_KCL(FeH, CO, MMW=2.33):
    # Taken from Charnay+2018
    t = np.linspace(100., 10000., 1000)

    def p_vap(x):
        return 1e1 ** (7.611 - 11382. / x)  # TODO check if this p_vap is alright

    xkcl = return_XKCL(FeH, CO)

    m_kcl = masses['K'] \
        + masses['Cl']
    return p_vap(t) / (xkcl * MMW / m_kcl), t


def return_T_cond_KCL_free(Xkcl, MMW=2.33):
    # Taken from Charnay+2018
    t = np.linspace(100., 10000., 1000)

    def p_vap(x):
        return 1e1 ** (7.611 - 11382. / x)  # TODO check if this p_vap is alright

    m_kcl = masses['K'] \
        + masses['Cl']
    return p_vap(t) / (Xkcl * MMW / m_kcl), t


def simple_cdf_Fe(press, temp, FeH, CO, MMW=2.33):
    pc, tc = return_T_cond_Fe_comb(FeH, CO, MMW)
    index = (pc > 1e-8) & (pc < 1e5)
    pc, tc = pc[index], tc[index]
    tcond_p = interp1d(pc, tc)
    # print(Pc, press)
    tcond_on_input_grid = tcond_p(press)

    tdiff = tcond_on_input_grid - temp
    diff_vec = tdiff[1:] * tdiff[:-1]
    ind_cdf = (diff_vec < 0.)
    if len(diff_vec[ind_cdf]) > 0:
        p_clouds = (press[1:] + press[:-1])[ind_cdf] / 2.
        p_cloud = p_clouds[-1]
    else:
        p_cloud = 1e-8

    if plotting:
        plt.plot(temp, press)
        plt.plot(tcond_on_input_grid, press)
        plt.axhline(p_cloud, color='red', linestyle='--')
        plt.yscale('log')
        plt.xlim([0., 3000.])
        plt.ylim([1e2, 1e-6])
        plt.show()

    return p_cloud


def simple_cdf_Fe_free(press, temp, XFe, MMW=2.33):
    pc, tc = return_T_cond_Fe_comb_free(XFe, MMW)
    index = (pc > 1e-8) & (pc < 1e5)
    pc, tc = pc[index], tc[index]
    tcond_p = interp1d(pc, tc)
    # print(Pc, press)
    tcond_on_input_grid = tcond_p(press)

    tdiff = tcond_on_input_grid - temp
    diff_vec = tdiff[1:] * tdiff[:-1]
    ind_cdf = (diff_vec < 0.)
    if len(diff_vec[ind_cdf]) > 0:
        p_clouds = (press[1:] + press[:-1])[ind_cdf] / 2.
        p_cloud = p_clouds[-1]
    else:
        p_cloud = 1e-8

    if plotting:
        plt.plot(temp, press)
        plt.plot(tcond_on_input_grid, press)
        plt.axhline(p_cloud, color='red', linestyle='--')
        plt.yscale('log')
        plt.xlim([0., 3000.])
        plt.ylim([1e2, 1e-6])
        plt.show()

    return p_cloud


def simple_cdf_MgSiO3(press, temp, FeH, CO, MMW=2.33):
    pc, tc = return_T_cond_MgSiO3(FeH, CO, MMW)
    index = (pc > 1e-8) & (pc < 1e5)
    pc, tc = pc[index], tc[index]
    tcond_p = interp1d(pc, tc)
    # print(Pc, press)
    tcond_on_input_grid = tcond_p(press)

    tdiff = tcond_on_input_grid - temp
    diff_vec = tdiff[1:] * tdiff[:-1]
    ind_cdf = (diff_vec < 0.)
    if len(diff_vec[ind_cdf]) > 0:
        p_clouds = (press[1:] + press[:-1])[ind_cdf] / 2.
        p_cloud = p_clouds[-1]
    else:
        p_cloud = 1e-8

    if plotting:
        plt.plot(temp, press)
        plt.plot(tcond_on_input_grid, press)
        plt.axhline(p_cloud, color='red', linestyle='--')
        plt.yscale('log')
        plt.xlim([0., 3000.])
        plt.ylim([1e2, 1e-6])
        plt.show()

    return p_cloud


def simple_cdf_MgSiO3_free(press, temp, Xmgsio3, MMW=2.33):
    pc, tc = return_T_cond_MgSiO3_free(Xmgsio3, MMW)
    index = (pc > 1e-8) & (pc < 1e5)
    pc, tc = pc[index], tc[index]
    tcond_p = interp1d(pc, tc)
    # print(Pc, press)
    tcond_on_input_grid = tcond_p(press)

    tdiff = tcond_on_input_grid - temp
    diff_vec = tdiff[1:] * tdiff[:-1]
    ind_cdf = (diff_vec < 0.)
    if len(diff_vec[ind_cdf]) > 0:
        p_clouds = (press[1:] + press[:-1])[ind_cdf] / 2.
        p_cloud = p_clouds[-1]
    else:
        p_cloud = 1e-8

    if plotting:
        plt.plot(temp, press)
        plt.plot(tcond_on_input_grid, press)
        plt.axhline(p_cloud, color='red', linestyle='--')
        plt.yscale('log')
        plt.xlim([0., 3000.])
        plt.ylim([1e2, 1e-6])
        plt.show()

    return p_cloud


def simple_cdf_Na2S(press, temp, FeH, CO, MMW=2.33):
    pc, tc = return_T_cond_Na2S(FeH, CO, MMW)
    index = (pc > 1e-8) & (pc < 1e5)
    pc, tc = pc[index], tc[index]
    tcond_p = interp1d(pc, tc)
    # print(Pc, press)
    tcond_on_input_grid = tcond_p(press)

    tdiff = tcond_on_input_grid - temp
    diff_vec = tdiff[1:] * tdiff[:-1]
    ind_cdf = (diff_vec < 0.)
    if len(diff_vec[ind_cdf]) > 0:
        p_clouds = (press[1:] + press[:-1])[ind_cdf] / 2.
        p_cloud = p_clouds[-1]
    else:
        p_cloud = 1e-8

    if plotting:
        plt.plot(temp, press)
        plt.plot(tcond_on_input_grid, press)
        plt.axhline(p_cloud, color='red', linestyle='--')
        plt.yscale('log')
        plt.xlim([0., 3000.])
        plt.ylim([1e2, 1e-6])
        plt.show()

    return p_cloud


def simple_cdf_KCL(press, temp, FeH, CO, MMW=2.33):
    pc, tc = return_T_cond_KCL(FeH, CO, MMW)
    index = (pc > 1e-8) & (pc < 1e5)
    pc, tc = pc[index], tc[index]
    tcond_p = interp1d(pc, tc)
    # print(Pc, press)
    tcond_on_input_grid = tcond_p(press)

    tdiff = tcond_on_input_grid - temp
    diff_vec = tdiff[1:] * tdiff[:-1]
    ind_cdf = (diff_vec < 0.)
    if len(diff_vec[ind_cdf]) > 0:
        p_clouds = (press[1:] + press[:-1])[ind_cdf] / 2.
        p_cloud = p_clouds[-1]
    else:
        p_cloud = 1e-8

    if plotting:
        plt.plot(temp, press)
        plt.plot(tcond_on_input_grid, press)
        plt.axhline(p_cloud, color='red', linestyle='--')
        plt.yscale('log')
        plt.xlim([0., 3000.])
        plt.ylim([1e2, 1e-6])
        plt.show()

    return p_cloud


def plot_all():
    from petitRADTRANS.physics import guillot_global

    # FeHs = np.linspace(-0.5, 2., 5)
    # COs = np.linspace(0.3, 1.2, 5)
    fehs = [0.]
    co_ratios = [0.55]

    for FeH in fehs:
        for CO in co_ratios:
            p, t = return_T_cond_Fe(FeH, CO)
            plt.plot(t, p, label='Fe(c), [Fe/H] = ' + str(FeH) + ', C/O = ' + str(CO), color='black')
            p, t = return_T_cond_Fe_l(FeH, CO)
            plt.plot(t, p, '--', label='Fe(l), [Fe/H] = ' + str(FeH) + ', C/O = ' + str(CO))
            p, t = return_T_cond_Fe_comb(FeH, CO)
            plt.plot(t, p, ':', label='Fe(c+l), [Fe/H] = ' + str(FeH) + ', C/O = ' + str(CO))
            p, t = return_T_cond_MgSiO3(FeH, CO)
            plt.plot(t, p, label='MgSiO3, [Fe/H] = ' + str(FeH) + ', C/O = ' + str(CO))
            p, t = return_T_cond_Na2S(FeH, CO)
            plt.plot(t, p, label='Na2S, [Fe/H] = ' + str(FeH) + ', C/O = ' + str(CO))
            p, t = return_T_cond_KCL(FeH, CO)
            plt.plot(t, p, label='KCL, [Fe/H] = ' + str(FeH) + ', C/O = ' + str(CO))

    plt.yscale('log')
    '''
    plt.xlim([0., 5000.])
    plt.ylim([1e5,1e-10])
    '''
    plt.xlim([0., 2000.])
    plt.ylim([1e2, 1e-3])
    plt.legend(loc='best', frameon=False)
    plt.show()

    kappa_ir = 0.01
    gamma = 0.4
    t_int = 200.
    t_equ = 1550.
    gravity = 1e1 ** 2.45

    pressures = np.logspace(-6, 2, 100)

    temperature = guillot_global(pressures, kappa_ir, gamma, gravity, t_int, t_equ)

    simple_cdf_Fe(pressures, temperature, 0., 0.55)
    simple_cdf_MgSiO3(pressures, temperature, 0., 0.55)

    t_int = 200.
    t_equ = 800.
    temperature = guillot_global(pressures, kappa_ir, gamma, gravity, t_int, t_equ)
    simple_cdf_Na2S(pressures, temperature, 0., 0.55)

    t_int = 150.
    t_equ = 650.
    temperature = guillot_global(pressures, kappa_ir, gamma, gravity, t_int, t_equ)
    simple_cdf_KCL(pressures, temperature, 0., 0.55)


if __name__ == '__main__':
    if plotting:
        plot_all()
