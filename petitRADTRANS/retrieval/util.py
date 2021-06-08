import sys, os
# To not have numpy start parallelizing on its own
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from scipy.special import gamma,erfcinv

import numpy as np
import math as math
from sys import platform
#import threading, subprocess

SQRT2 = math.sqrt(2.)

#################
# Flux scaling
#################
def surf_to_meas(flux,p_rad,dist):
    """
    surf_to_meas
    Convert from emission flux to measured flux at earth
    parameters:
    -----------
    flux : numpy.ndarray
        Absolute flux value or spectrum as emitted by a source of radius p_rad
    p_rad : float
        Planet radius, in same units as dist
    dist : float
        Distance to the object, in the same units as p_rad
    returns:
    --------
    m_flux : numpy.ndarray
        Apparent flux
    """
    m_flux = flux * p_rad**2/dist**2
    return m_flux


#################
# Prior Functions
#################
# Stolen from https://github.com/JohannesBuchner/MultiNest/blob/master/src/priors.f90
def log_prior(cube,lx1,lx2):
    return 10**(lx1+cube*(lx2-lx1))

def uniform_prior(cube,x1,x2):
    return x1+cube*(x2-x1)

def gaussian_prior(cube,mu,sigma):
    return mu + sigma*SQRT2*erfcinv(2.0*(1.0 - cube))
    #return -(((cube-mu)/sigma)**2.)/2.

def log_gaussian_prior(cube,mu,sigma):
    bracket = sigma*sigma + sigma*SQRT2*erfcinv(2.0*cube)
    return mu*np.exp(bracket)

def delta_prior(cube,x1,x2):
    return x1

# Sanity checks on parameter ranges
def b_range(x, b):
    if x > b:
        return -np.inf
    else:
        return 0.

def a_b_range(x, a, b):
    if x < a:
        return -np.inf
    elif x > b:
        return -np.inf
    else:
        return 0.


########################
# Mean Molecular Weights
########################
MMWs = {}
MMWs['H2'] = 2.
MMWs['He'] = 4.
MMWs['H2O'] = 18.
MMWs['D2O'] = 20.
MMWs['CH4'] = 16.
MMWs['CO2'] = 44.
MMWs['CO'] = 28.
MMWs['CO_all_iso'] = 28.01
MMWs['Na'] = 23.
MMWs['K'] = 39.
MMWs['NH3'] = 17.
MMWs['HCN'] = 27.
MMWs['C2H2'] = 26.
MMWs['C2H2,acetylene'] = 26.
MMWs['PH3'] = 34.
MMWs['H2S'] = 34.
MMWs['VO'] = 67.
MMWs['TiO'] = 64.

def calc_MMW(abundances):
    """
    calc_MMW
    Calculate the mean molecular weight in each layer.
    
    parameters
    ----------
    abundances : dict
        dictionary of abundance arrays, each array must have the shape of the pressure array used in pRT,
        and contain the abundance at each layer in the atmosphere.
    """
    MMW = 0.
    for key in abundances.keys():
        # exo_k resolution
        spec = key.split("_R_")[0]
        if spec == 'CO_all_iso':
            MMW += abundances[key]/MMWs['CO']
        else:
            MMW += abundances[key]/MMWs[spec]
    
    return 1./MMW

def get_MMW_from_mfrac(m_frac):
    """
    wraps calc_MMW
    """
    calc_MMW(m_frac)

def get_MMW_from_nfrac(n_frac):
    """
    get_MMW_from_nfrac
    Calculate the mean molecular weight from a number fraction

    parameters:
    -----------
    n_fracs : dict
        A dictionary of number fractions
    """
    mass = 0.0
    for key,value in n_frac.items():
        spec = key.split("_R_")[0]
        mass += value*MMWs[spec]
    return mass

def mass_to_number(m_frac):
    """
    mass_to_number
    Convert mass fractions to number fractions

    parameters:
    -----------
    m_fracs : dict
        A dictionary of mass fractions
    """
    n_frac = {}
    MMW = get_MMW_from_mfrac(m_frac)
    for key,value in m_frac.items():
        spec = key.split("_R_")[0]
        n_frac[key]=value/MMWs[spec]*MMW
    return n_frac

def number_to_mass(n_fracs):
    """
    number_to_mass
    Convert number fractions to mass fractions

    parameters:
    -----------
    n_fracs : dict
        A dictionary of number fractions
    """
    m_frac = {}
    MMW = get_MMW_from_nfrac(n_fracs)
    for key,value in n_fracs.items():
        spec = key.split("_R_")[0]
        m_frac[key] = value*MMWs[spec]/MMW
    return m_frac