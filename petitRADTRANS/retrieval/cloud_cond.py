import numpy as np
import copy as cp
from scipy.interpolate import interp1d
import logging
plotting = False
if plotting:
    import pylab as plt
    from petitRADTRANS import nat_cst as nc

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
metals = ['C','N','O','Na','Mg','Al','Si','P','S','Cl','K','Ca','Ti','V','Fe','Ni']

# solar abundances, [Fe/H] = 0, from Asplund+ 2009
nfracs = {}
nfracs['H'] = 0.9207539305
nfracs['He'] = 0.0783688694
nfracs['C'] = 0.0002478241
nfracs['N'] = 6.22506056949881e-05
nfracs['O'] = 0.0004509658
nfracs['Na'] = 1.60008694353205e-06
nfracs['Mg'] = 3.66558742055362e-05
nfracs['Al'] = 2.595e-06
nfracs['Si'] = 2.9795e-05
nfracs['P'] = 2.36670201997668e-07
nfracs['S'] = 1.2137900734604e-05
nfracs['Cl'] = 2.91167958499589e-07
nfracs['K'] = 9.86605611925677e-08
nfracs['Ca'] = 2.01439011429255e-06
nfracs['Ti'] = 8.20622804366359e-08
nfracs['V'] = 7.83688694089992e-09
nfracs['Fe'] = 2.91167958499589e-05
nfracs['Ni'] = 1.52807116806281e-06

# atomic masses
masses = {}
masses['H'] = 1.
masses['He'] = 4.
masses['C'] = 12.
masses['N'] = 14.
masses['O'] = 16.
masses['Na'] = 23.
masses['Mg'] = 24.3
masses['Al'] = 27.
masses['Si'] = 28.
masses['P'] = 31.
masses['S'] = 32.
masses['Cl'] = 35.45
masses['K'] = 39.1
masses['Ca'] = 40.
masses['Ti'] = 47.9
masses['V'] = 51.
masses['Fe'] = 55.8
masses['Ni'] = 58.7

def return_cloud_mass_fraction(name,FeH,CO):
    if "Fe(c)" in name:
        return return_XFe(FeH,CO)
    if "MgSiO3(c)" in name:
        return return_XMgSiO3(FeH,CO)
    if "Na2S(c)" in name:
        return return_XNa2S(FeH,CO)
    if "KCL(c)" in name:
        return return_XKCL(FeH,CO)
    else:
        logging.warn(f"The cloud {name} is not currently implemented.")
        return np.zeros_like(FeH)

def simple_cdf(name,press, temp, FeH, CO, MMW = 2.33):
        if "Fe(c)" in name:
            return simple_cdf_Fe(press, temp, FeH, CO, MMW)
        if "MgSiO3(c)" in name:
            return simple_cdf_MgSiO3(press, temp, FeH, CO, MMW)
        if "Na2S(c)" in name:
            return simple_cdf_Na2S(press, temp, FeH, CO, MMW)
        if "KCL(c)" in name:
            return simple_cdf_KCL(press, temp, FeH, CO, MMW)
        else:
            logging.warn(f"The cloud {name} is not currently implemented.")
            return np.zeros_like(FeH)
def simple_cdf_free(name,press, temp, mfrac, MMW = 2.33):
        if "Fe(c)" in name:
            return simple_cdf_Fe_free(press, temp, mfrac, MMW)
        if "MgSiO3(c)" in name:
            return simple_cdf_MgSiO3_free(press, temp, mfrac, MMW)
        if "Na2S(c)" in name:
            return simple_cdf_Na2S_free(press, temp, mfrac, MMW)
        if "KCL(c)" in name:
            return simple_cdf_KCL_free(press, temp, mfrac, MMW)
        else:
            logging.warn(f"The cloud {name} is not currently implemented.")
            return np.zeros_like(FeH)
def return_XFe(FeH, CO):

    nfracs_use = cp.copy(nfracs)

    for spec in nfracs.keys():

        if (spec != 'H') and (spec != 'He'):
            nfracs_use[spec] = nfracs[spec]*1e1**FeH

    nfracs_use['O'] = nfracs_use['C']/CO

    XFe = masses['Fe']*nfracs_use['Fe']
    add = 0.
    for spec in nfracs_use.keys():
        add += masses[spec]*nfracs_use[spec]

    XFe = XFe / add

    return XFe

def return_XMgSiO3(FeH, CO):

    nfracs_use = cp.copy(nfracs)

    for spec in nfracs.keys():

        if (spec != 'H') and (spec != 'He'):
            nfracs_use[spec] = nfracs[spec]*1e1**FeH

    nfracs_use['O'] = nfracs_use['C']/CO

    nfracs_mgsio3 = np.min([nfracs_use['Mg'], \
                            nfracs_use['Si'], \
                            nfracs_use['O']/3.])
    masses_mgsio3 = masses['Mg'] \
      + masses['Si'] \
      + 3. * masses['O']

    Xmgsio3 = masses_mgsio3*nfracs_mgsio3
    add = 0.
    for spec in nfracs_use.keys():
        add += masses[spec]*nfracs_use[spec]

    Xmgsio3 = Xmgsio3 / add

    return Xmgsio3

def return_XNa2S(FeH, CO):

    nfracs_use = cp.copy(nfracs)

    for spec in nfracs.keys():

        if (spec != 'H') and (spec != 'He'):
            nfracs_use[spec] = nfracs[spec]*1e1**FeH

    nfracs_use['O'] = nfracs_use['C']/CO

    nfracs_na2s = np.min([nfracs_use['Na']/2., \
                            nfracs_use['S']])
    masses_na2s = 2.*masses['Na'] \
      + masses['S']

    Xna2s = masses_na2s*nfracs_na2s
    add = 0.
    for spec in nfracs_use.keys():
        add += masses[spec]*nfracs_use[spec]

    Xna2s = Xna2s / add

    return Xna2s

def return_XKCL(FeH, CO):

    nfracs_use = cp.copy(nfracs)

    for spec in nfracs.keys():

        if (spec != 'H') and (spec != 'He'):
            nfracs_use[spec] = nfracs[spec]*1e1**FeH

    nfracs_use['O'] = nfracs_use['C']/CO

    nfracs_kcl = np.min([nfracs_use['K'], \
                            nfracs_use['Cl']])
    masses_kcl = masses['K'] \
      + masses['Cl']

    Xkcl = masses_kcl*nfracs_kcl
    add = 0.
    for spec in nfracs_use.keys():
        add += masses[spec]*nfracs_use[spec]

    Xkcl = Xkcl / add

    return Xkcl

#############################################################
# Fe saturation pressure, from Ackerman & Marley (2001), including erratum (P_vap is in bar, not cgs!)
#############################################################

def return_T_cond_Fe(FeH, CO, MMW = 2.33):

    T = np.linspace(100.,10000.,1000)
    # Taken from Ackerman & Marley (2001)
    # including their erratum
    P_vap = lambda x: np.exp(15.71 - 47664./x)

    XFe = return_XFe(FeH, CO)

    return P_vap(T)/(XFe*MMW/masses['Fe']), T

def return_T_cond_Fe_l(FeH, CO, MMW = 2.33):

    T = np.linspace(100.,10000.,1000)
    # Taken from Ackerman & Marley (2001)
    # including their erratum
    P_vap = lambda x: np.exp(9.86 - 37120./x)

    XFe = return_XFe(FeH, CO)

    return P_vap(T)/(XFe*MMW/masses['Fe']), T

def return_T_cond_Fe_comb(FeH, CO, MMW = 2.33):

    P1, T1 = return_T_cond_Fe(FeH, CO, MMW)
    P2, T2 = return_T_cond_Fe_l(FeH, CO, MMW)

    retP = np.zeros_like(P1)
    index = P1<P2
    retP[index] = P1[index]
    retP[~index] = P2[~index]
    return retP, T2

def return_T_cond_Fe_free(XFe, MMW = 2.33):

    T = np.linspace(100.,10000.,1000)
    # Taken from Ackerman & Marley (2001)
    # including their erratum
    P_vap = lambda x: np.exp(15.71 - 47664./x)
    return P_vap(T)/(XFe*MMW/masses['Fe']), T

def return_T_cond_Fe_l_free(XFe, MMW = 2.33):

    T = np.linspace(100.,10000.,1000)
    # Taken from Ackerman & Marley (2001)
    # including their erratum
    P_vap = lambda x: np.exp(9.86 - 37120./x)
    return P_vap(T)/(XFe*MMW/masses['Fe']), T

def return_T_cond_Fe_comb_free(XFe, MMW = 2.33):

    P1, T1 = return_T_cond_Fe_free(XFe, MMW)
    P2, T2 = return_T_cond_Fe_l_free(XFe, MMW)
    retP = np.zeros_like(P1)
    index = P1<P2
    retP[index] = P1[index]
    retP[~index] = P2[~index]
    return retP, T2

def return_T_cond_MgSiO3(FeH, CO, MMW = 2.33):

    T = np.linspace(100.,10000.,1000)
    # Taken from Ackerman & Marley (2001)
    # including their erratum
    P_vap = lambda x: np.exp(25.37 - 58663./x)

    Xmgsio3 = return_XMgSiO3(FeH, CO)

    m_mgsio3 =  masses['Mg'] \
      + masses['Si'] \
      + 3. * masses['O']
    return P_vap(T)/(Xmgsio3*MMW/m_mgsio3), T
def return_T_cond_MgSiO3_free(Xmgsio3, MMW = 2.33):

    T = np.linspace(100.,10000.,1000)
    # Taken from Ackerman & Marley (2001)
    # including their erratum
    P_vap = lambda x: np.exp(25.37 - 58663./x)
    m_mgsio3 =  masses['Mg'] \
      + masses['Si'] \
      + 3. * masses['O']
    return P_vap(T)/(Xmgsio3*MMW/m_mgsio3), T

def return_T_cond_Na2S(FeH, CO, MMW = 2.33):

    # Taken from Charnay+2018
    T = np.linspace(100.,10000.,1000)
    # This is the partial pressure of Na, so
    # Divide by factor 2 to get the partial
    # pressure of the hypothetical Na2S gas
    # particles, this is OK: there are
    # more S than Na atoms at solar
    # abundance ratios.
    P_vap = lambda x: 1e1**(8.55 - 13889./x - 0.5*FeH)/2.

    Xna2s = return_XNa2S(FeH, CO)

    m_na2s =  2.*masses['Na'] \
      + masses['S']
    return P_vap(T)/(Xna2s*MMW/m_na2s), T
def return_T_cond_Na2S_free(Xna2s, MMW = 2.33):

    # Taken from Charnay+2018
    T = np.linspace(100.,10000.,1000)
    # This is the partial pressure of Na, so
    # Divide by factor 2 to get the partial
    # pressure of the hypothetical Na2S gas
    # particles, this is OK: there are
    # more S than Na atoms at solar
    # abundance ratios.

    # We're also using [Na/H] as a proxy for [Fe/H]
    # Definitely not strictly correct, but should be
    # good enough for ~ solar compositions. [+- 1 for Fe/H]
    # Assumes constant vertical abundance
    m_na2s =  2.*masses['Na'] \
      + masses['S']
    P_vap = lambda x: 10**(8.55 - 13889./x - 0.5*(np.log10(2*Xna2s*MMW/m_na2s)+5.7))/2

    return P_vap(T)/(Xna2s*MMW/m_na2s), T

def return_T_cond_KCL(FeH, CO, MMW = 2.33):

    # Taken from Charnay+2018
    T = np.linspace(100.,10000.,1000)
    P_vap = lambda x: 1e1**(7.611 - 11382./x)

    Xkcl = return_XKCL(FeH, CO)

    m_kcl =  masses['K'] \
      + masses['Cl']
    return P_vap(T)/(Xkcl*MMW/m_kcl), T
def return_T_cond_KCL_free(Xkcl, MMW = 2.33):

    # Taken from Charnay+2018
    T = np.linspace(100.,10000.,1000)
    P_vap = lambda x: 1e1**(7.611 - 11382./x)
    m_kcl =  masses['K'] \
      + masses['Cl']
    return P_vap(T)/(Xkcl*MMW/m_kcl), T


if plotting:

    #FeHs = np.linspace(-0.5, 2., 5)
    #COs = np.linspace(0.3, 1.2, 5)
    FeHs = [0.]
    COs = [0.55]

    for FeH in FeHs:
        for CO in COs:
            P, T = return_T_cond_Fe(FeH, CO)
            plt.plot(T,P, label = 'Fe(c), [Fe/H] = '+str(FeH)+', C/O = '+str(CO), color = 'black')
            P, T = return_T_cond_Fe_l(FeH, CO)
            plt.plot(T,P, '--', label = 'Fe(l), [Fe/H] = '+str(FeH)+', C/O = '+str(CO))
            P, T = return_T_cond_Fe_comb(FeH, CO)
            plt.plot(T,P, ':', label = 'Fe(c+l), [Fe/H] = '+str(FeH)+', C/O = '+str(CO))
            P, T = return_T_cond_MgSiO3(FeH, CO)
            plt.plot(T,P, label = 'MgSiO3, [Fe/H] = '+str(FeH)+', C/O = '+str(CO))
            P, T = return_T_cond_Na2S(FeH, CO)
            plt.plot(T,P, label = 'Na2S, [Fe/H] = '+str(FeH)+', C/O = '+str(CO))
            P, T = return_T_cond_KCL(FeH, CO)
            plt.plot(T,P, label = 'KCL, [Fe/H] = '+str(FeH)+', C/O = '+str(CO))


    plt.yscale('log')
    '''
    plt.xlim([0., 5000.])
    plt.ylim([1e5,1e-10])
    '''
    plt.xlim([0., 2000.])
    plt.ylim([1e2,1e-3])
    plt.legend(loc = 'best', frameon = False)
    plt.show()

def simple_cdf_Fe(press, temp, FeH, CO, MMW = 2.33):

    Pc, Tc = return_T_cond_Fe_comb(FeH, CO, MMW)
    index = (Pc > 1e-8) & (Pc < 1e5)
    Pc, Tc = Pc[index], Tc[index]
    tcond_p = interp1d(Pc, Tc)
    #print(Pc, press)
    Tcond_on_input_grid = tcond_p(press)

    Tdiff = Tcond_on_input_grid - temp
    diff_vec = Tdiff[1:]*Tdiff[:-1]
    ind_cdf = (diff_vec < 0.)
    if len(diff_vec[ind_cdf]) > 0:
        P_clouds = (press[1:]+press[:-1])[ind_cdf]/2.
        P_cloud = P_clouds[-1]
    else:
        P_cloud = np.min(press)

    if plotting:
        plt.plot(temp, press)
        plt.plot(Tcond_on_input_grid, press)
        plt.axhline(P_cloud, color = 'red', linestyle = '--')
        plt.yscale('log')
        plt.xlim([0., 3000.])
        plt.ylim([1e2,1e-6])
        plt.show()

    return P_cloud
def simple_cdf_Fe_free(press, temp, XFe, MMW = 2.33):

    Pc, Tc = return_T_cond_Fe_comb_free(XFe, MMW)
    index = (Pc > 1e-8) & (Pc < 1e5)
    Pc, Tc = Pc[index], Tc[index]
    tcond_p = interp1d(Pc, Tc)
    #print(Pc, press)
    Tcond_on_input_grid = tcond_p(press)

    Tdiff = Tcond_on_input_grid - temp
    diff_vec = Tdiff[1:]*Tdiff[:-1]
    ind_cdf = (diff_vec < 0.)
    if len(diff_vec[ind_cdf]) > 0:
        P_clouds = (press[1:]+press[:-1])[ind_cdf]/2.
        P_cloud = P_clouds[-1]
    else:
        P_cloud = np.min(press)

    if plotting:
        plt.plot(temp, press)
        plt.plot(Tcond_on_input_grid, press)
        plt.axhline(P_cloud, color = 'red', linestyle = '--')
        plt.yscale('log')
        plt.xlim([0., 3000.])
        plt.ylim([1e2,1e-6])
        plt.show()

    return P_cloud
def simple_cdf_MgSiO3(press, temp, FeH, CO, MMW = 2.33):

    Pc, Tc = return_T_cond_MgSiO3(FeH, CO, MMW)
    index = (Pc > 1e-8) & (Pc < 1e5)
    Pc, Tc = Pc[index], Tc[index]
    tcond_p = interp1d(Pc, Tc)
    #print(Pc, press)
    Tcond_on_input_grid = tcond_p(press)

    Tdiff = Tcond_on_input_grid - temp
    diff_vec = Tdiff[1:]*Tdiff[:-1]
    ind_cdf = (diff_vec < 0.)
    if len(diff_vec[ind_cdf]) > 0:
        P_clouds = (press[1:]+press[:-1])[ind_cdf]/2.
        P_cloud = P_clouds[-1]
    else:
        P_cloud = np.min(press)

    if plotting:
        plt.plot(temp, press)
        plt.plot(Tcond_on_input_grid, press)
        plt.axhline(P_cloud, color = 'red', linestyle = '--')
        plt.yscale('log')
        plt.xlim([0., 3000.])
        plt.ylim([1e2,1e-6])
        plt.show()

    return P_cloud
def simple_cdf_MgSiO3_free(press, temp, Xmgsio3, MMW = 2.33):

    Pc, Tc = return_T_cond_MgSiO3_free(Xmgsio3, MMW)
    index = (Pc > 1e-8) & (Pc < 1e5)
    Pc, Tc = Pc[index], Tc[index]
    tcond_p = interp1d(Pc, Tc)
    #print(Pc, press)
    Tcond_on_input_grid = tcond_p(press)

    Tdiff = Tcond_on_input_grid - temp
    diff_vec = Tdiff[1:]*Tdiff[:-1]
    ind_cdf = (diff_vec < 0.)
    if len(diff_vec[ind_cdf]) > 0:
        P_clouds = (press[1:]+press[:-1])[ind_cdf]/2.
        P_cloud = P_clouds[-1]
    else:
        P_cloud = np.min(press)

    if plotting:
        plt.plot(temp, press)
        plt.plot(Tcond_on_input_grid, press)
        plt.axhline(P_cloud, color = 'red', linestyle = '--')
        plt.yscale('log')
        plt.xlim([0., 3000.])
        plt.ylim([1e2,1e-6])
        plt.show()

    return P_cloud

def simple_cdf_Na2S(press, temp, FeH, CO, MMW = 2.33):

    Pc, Tc = return_T_cond_Na2S(FeH, CO, MMW)
    index = (Pc > 1e-8) & (Pc < 1e5)
    Pc, Tc = Pc[index], Tc[index]
    tcond_p = interp1d(Pc, Tc)
    #print(Pc, press)
    Tcond_on_input_grid = tcond_p(press)

    Tdiff = Tcond_on_input_grid - temp
    diff_vec = Tdiff[1:]*Tdiff[:-1]
    ind_cdf = (diff_vec < 0.)
    if len(diff_vec[ind_cdf]) > 0:
        P_clouds = (press[1:]+press[:-1])[ind_cdf]/2.
        P_cloud = P_clouds[-1]
    else:
        P_cloud = np.min(press)

    if plotting:
        plt.plot(temp, press)
        plt.plot(Tcond_on_input_grid, press)
        plt.axhline(P_cloud, color = 'red', linestyle = '--')
        plt.yscale('log')
        plt.xlim([0., 3000.])
        plt.ylim([1e2,1e-6])
        plt.show()

    return P_cloud

def simple_cdf_Na2S_free(press, temp, XNa2S, MMW = 2.33):

    Pc, Tc = return_T_cond_Na2S_free(XNa2S, MMW)
    index = (Pc > 1e-8) & (Pc < 1e5)

    Pc, Tc = Pc[index], Tc[index]
    tcond_p = interp1d(Pc, Tc)
    #print(Pc, press)
    Tcond_on_input_grid = tcond_p(press)

    Tdiff = Tcond_on_input_grid - temp
    diff_vec = Tdiff[1:]*Tdiff[:-1]
    ind_cdf = (diff_vec < 0.)
    if len(diff_vec[ind_cdf]) > 0:
        P_clouds = (press[1:]+press[:-1])[ind_cdf]/2.
        P_cloud = P_clouds[-1]
    else:
        P_cloud = np.min(press)

    if plotting:
        plt.plot(temp, press)
        plt.plot(Tcond_on_input_grid, press)
        plt.axhline(P_cloud, color = 'red', linestyle = '--')
        plt.yscale('log')
        plt.xlim([0., 3000.])
        plt.ylim([1e2,1e-6])
        plt.show()

    return P_cloud

def simple_cdf_KCL(press, temp, FeH, CO, MMW = 2.33):

    Pc, Tc = return_T_cond_KCL(FeH, CO, MMW)
    index = (Pc > 1e-8) & (Pc < 1e5)
    Pc, Tc = Pc[index], Tc[index]
    tcond_p = interp1d(Pc, Tc)
    #print(Pc, press)
    Tcond_on_input_grid = tcond_p(press)

    Tdiff = Tcond_on_input_grid - temp
    diff_vec = Tdiff[1:]*Tdiff[:-1]
    ind_cdf = (diff_vec < 0.)
    if len(diff_vec[ind_cdf]) > 0:
        P_clouds = (press[1:]+press[:-1])[ind_cdf]/2.
        P_cloud = P_clouds[-1]
    else:
        P_cloud = np.min(press)

    if plotting:
        plt.plot(temp, press)
        plt.plot(Tcond_on_input_grid, press)
        plt.axhline(P_cloud, color = 'red', linestyle = '--')
        plt.yscale('log')
        plt.xlim([0., 3000.])
        plt.ylim([1e2,1e-6])
        plt.show()

    return P_cloud

def simple_cdf_KCL_free(press, temp, XKCL, MMW = 2.33):

    Pc, Tc = return_T_cond_KCL_free(XKCL, MMW)
    index = (Pc > 1e-8) & (Pc < 1e5)
    Pc, Tc = Pc[index], Tc[index]
    tcond_p = interp1d(Pc, Tc)
    #print(Pc, press)
    Tcond_on_input_grid = tcond_p(press)

    Tdiff = Tcond_on_input_grid - temp
    diff_vec = Tdiff[1:]*Tdiff[:-1]
    ind_cdf = (diff_vec < 0.)
    if len(diff_vec[ind_cdf]) > 0:
        P_clouds = (press[1:]+press[:-1])[ind_cdf]/2.
        P_cloud = P_clouds[-1]
    else:
        P_cloud = np.min(press)

    if plotting:
        plt.plot(temp, press)
        plt.plot(Tcond_on_input_grid, press)
        plt.axhline(P_cloud, color = 'red', linestyle = '--')
        plt.yscale('log')
        plt.xlim([0., 3000.])
        plt.ylim([1e2,1e-6])
        plt.show()

    return P_cloud
if plotting:
    kappa_IR = 0.01
    gamma = 0.4
    T_int = 200.
    T_equ = 1550.
    gravity = 1e1**2.45

    pressures = np.logspace(-6, 2, 100)

    temperature = nc.guillot_global(pressures, kappa_IR, gamma, gravity, T_int, T_equ)

    simple_cdf_Fe(pressures, temperature, 0., 0.55)
    simple_cdf_MgSiO3(pressures, temperature, 0., 0.55)

    T_int = 200.
    T_equ = 800.
    temperature = nc.guillot_global(pressures, kappa_IR, gamma, gravity, T_int, T_equ)
    simple_cdf_Na2S(pressures, temperature, 0., 0.55)

    T_int = 150.
    T_equ = 650.
    temperature = nc.guillot_global(pressures, kappa_IR, gamma, gravity, T_int, T_equ)
    simple_cdf_KCL(pressures, temperature, 0., 0.55)

