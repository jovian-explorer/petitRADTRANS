from __future__ import print_function
import astropy.constants as anc
import numpy as np
import scipy.constants as snc

import os as os
import sys

# Natural constants
# Everything is in cgs!
# Defined constants
c = snc.c * 1e2
h = snc.h * 1e7
kB = snc.k * 1e7
nA = snc.N_A
e = snc.e * np.sqrt(1e9 / (4 * snc.pi * snc.epsilon_0))

# Measured constants
G = snc.G * 1e3
m_elec = snc.m_e * 1e3

# Derived exact constants
sigma = snc.sigma * 1e3
L0 = snc.physical_constants['Loschmidt constant (273.15 K, 101.325 kPa)'][0] * 1e-6
R = snc.R

# Units definitions
bar = 1e6
atm = snc.atm * 1e1
AU = snc.au * 1e2
pc = snc.parsec * 1e2
light_year = snc.light_year * 1e2
amu = snc.physical_constants['atomic mass constant'][0] * 1e3

# Astronomical constants
r_sun = anc.R_sun.cgs.value
r_jup = anc.R_jup.cgs.value
r_earth = anc.R_earth.cgs.value
m_sun = anc.M_sun.cgs.value
m_jup = anc.M_jup.cgs.value
m_earth = anc.M_earth.cgs.value
l_sun = anc.L_sun.cgs.value

r_jup_mean = 6.9911e9
s_earth = 1.3654e6  # erg.s-1.cm-2, source: https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2010GL045777

# Molecular weights in amu
molecular_weight = {
    'H2O': 18.,
    'O2': 32.,
    'N2': 28.,
    'CH4': 16.,
    'CO2': 44.,
    'CO': 28.,
    'H2': 2.,
    'He': 4.
}

def b(T,nu):
    ''' Returns the Planck function :math:`B_{\\nu}(T)` in units of
    :math:`\\rm erg/s/cm^2/Hz/steradian`.

    Args:
        T (float):
            Temperature in K.
        nu:
            numpy array containing the frequency in Hz.
    '''

    retVal = 2.*h*nu**3./c**2.
    retVal = retVal / (np.exp(h*nu/kB/T)-1.)
    return retVal

def guillot_global(P,kappa_IR,gamma,grav,T_int,T_equ):
    ''' Returns a temperature array, in units of K,
    of the same dimensions as the pressure P
    (in bar). For this the temperature model of Guillot (2010)
    is used (his Equation 29).

    Args:
        P:
            numpy array of floats, containing the input pressure in bars.
        kappa_IR (float):
            The infrared opacity in units of :math:`\\rm cm^2/s`.
        gamma (float):
            The ratio between the visual and infrated opacity.
        grav (float):
            The planetary surface gravity in units of :math:`\\rm cm/s^2`.
        T_int (float):
            The planetary internal temperature (in units of K).
        T_equ (float):
            The planetary equilibrium temperature (in units of K).
    '''
    tau = P*1e6*kappa_IR/grav
    T_irr = T_equ*np.sqrt(2.)
    T = (0.75 * T_int**4. * (2. / 3. + tau) + \
      0.75 * T_irr**4. / 4. * (2. / 3. + 1. / gamma / 3.**0.5 + \
      (gamma / 3.**0.5 - 1. / 3.**0.5 / gamma)* \
      np.exp(-gamma * tau *3.**0.5)))**0.25
    return T

pathinp = os.environ.get("pRT_input_data_path")
if pathinp == None:
    raise OSError(f"Path to input data not specified!\n"
                  f"Please set pRT_input_data_path variable in .bashrc / .bash_profile or specify path via\n"
                  f">>> import os"
                  f">>> os.environ['pRT_input_data_path'] = 'absolute/path/of/the/folder/input_data'\n"
                  f"before creating a Radtrans object or loading the nat_cst module.\n"
                  f"(this will become unnecessary in a future update)"
                  )
spec_path = pathinp + '/stellar_specs'
    
description = np.genfromtxt(spec_path+'/stellar_params.dat')
logTempGrid = description[:,0]
StarRadGrid = description[:,1]

specDats = []
for i in range(len(logTempGrid)):
    specDats.append(np.genfromtxt(spec_path+'/spec_'+ \
                    str(int(i)).zfill(2)+'.dat'))

def get_PHOENIX_spec(temperature):
    ''' Returns a matrix where the first column is the wavelength in cm
    and the second is the stellar flux :math:`F_\\nu` in units of
    :math:`\\rm erg/cm^2/s/Hz`, at the surface of the star.
    The spectra are PHOENIX models from (Husser et al. 2013), the spectral
    grid used here was described in van Boekel et al. (2012).

    Args:
        temperature (float):
            stellar effective temperature in K.
    '''
    logTemp = np.log10(temperature)
    interpolationIndex = np.searchsorted(logTempGrid, logTemp)

    if interpolationIndex == 0:

        specDat = specDats[0]
        print('Warning, input temperature is lower than minimum grid temperature.')
        print('Taking F = F_grid(minimum grid temperature), normalized to desired')
        print('input temperature.')

    elif interpolationIndex == len(logTempGrid):

        specDat = specDats[int(len(logTempGrid)-1)]
        print('Warning, input temperature is higher than maximum grid temperature.')
        print('Taking F = F_grid(maximum grid temperature), normalized to desired')
        print('input temperature.')

    else:

        weightHigh = (logTemp-logTempGrid[interpolationIndex-1]) / \
          (logTempGrid[interpolationIndex]-logTempGrid[interpolationIndex-1])

        weightLow = 1. - weightHigh

        specDatLow = specDats[int(interpolationIndex-1)]

        specDatHigh = specDats[int(interpolationIndex)]

        specDat = np.zeros_like(specDatLow)

        specDat[:,0] = specDatLow[:,0]
        specDat[:,1] = weightLow * specDatLow[:,1] + \
          weightHigh * specDatHigh[:,1]

    freq = c/specDat[:,0]
    flux = specDat[:,1]
    norm = -np.sum((flux[1:]+flux[:-1])*np.diff(freq))/2.

    specDat[:,1] = flux/norm*sigma*temperature**4.

    return specDat

def get_PHOENIX_spec_rad(temperature):
    ''' 
    Returns a matrix where the first column is the wavelength in cm
    and the second is the stellar flux :math:`F_\\nu` in units of
    :math:`\\rm erg/cm^2/s/Hz`, at the surface of the star.
    The spectra are PHOENIX models from (Husser et al. 2013), the spectral
    grid used here was described in van Boekel et al. (2012).

    UPDATE: It also returns a float that is the corresponding estimate
    of the stellar radius.

    Args:
        temperature (float):
            stellar effective temperature in K.
    '''
    
    logTemp = np.log10(temperature)
    interpolationIndex = np.searchsorted(logTempGrid, logTemp)

    if interpolationIndex == 0:

        specDat = specDats[0]
        radius = StarRadGrid[0]
        print('Warning, input temperature is lower than minimum grid temperature.')
        print('Taking F = F_grid(minimum grid temperature), normalized to desired')
        print('input temperature.')

    elif interpolationIndex == len(logTempGrid):

        specDat = specDats[int(len(logTempGrid)-1)]
        radius = StarRadGrid[int(len(logTempGrid)-1)]
        print('Warning, input temperature is higher than maximum grid temperature.')
        print('Taking F = F_grid(maximum grid temperature), normalized to desired')
        print('input temperature.')

    else:

        weightHigh = (logTemp-logTempGrid[interpolationIndex-1]) / \
          (logTempGrid[interpolationIndex]-logTempGrid[interpolationIndex-1])

        weightLow = 1. - weightHigh

        specDatLow = specDats[int(interpolationIndex-1)]

        specDatHigh = specDats[int(interpolationIndex)]

        specDat = np.zeros_like(specDatLow)

        specDat[:,0] = specDatLow[:,0]
        specDat[:,1] = weightLow * specDatLow[:,1] + \
          weightHigh * specDatHigh[:,1]
        radius = weightLow * StarRadGrid[int(interpolationIndex-1)] + \
          weightHigh * StarRadGrid[int(interpolationIndex)]

    freq = c/specDat[:,0]
    flux = specDat[:,1]
    norm = -np.sum((flux[1:]+flux[:-1])*np.diff(freq))/2.

    specDat[:,1] = flux/norm*sigma*temperature**4.

    return specDat,radius
##################################################################
### Radtrans utility for retrieval temperature model computation
##################################################################

### Box car conv. average, found on stackoverflow somewhere
def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

### Global Guillot P-T formula with kappa/grav replaced by delta
def guillot_global_ret(P,delta,gamma,T_int,T_equ):

    delta = np.abs(delta)
    gamma = np.abs(gamma)
    T_int = np.abs(T_int)
    T_equ = np.abs(T_equ)
    tau = P*1e6*delta
    T_irr = T_equ*np.sqrt(2.)
    T = (0.75*T_int**4.*(2./3.+tau) + \
      0.75*T_irr**4./4.*(2./3.+1./gamma/3.**0.5+ \
                         (gamma/3.**0.5-1./3.**0.5/gamma)* \
                             np.exp(-gamma*tau*3.**0.5)))**0.25
    return T

### Modified Guillot P-T formula
def guillot_modif(P,delta,gamma,T_int,T_equ,ptrans,alpha):
    return guillot_global_ret(P,np.abs(delta),np.abs(gamma), \
                                  np.abs(T_int),np.abs(T_equ))* \
                                  (1.-alpha*(1./(1.+ \
                                                np.exp((np.log(P/ptrans))))))

### Function to make temp
def make_press_temp(rad_trans_params):

    press_many = np.logspace(-8,5,260)
    t_no_ave = guillot_modif(press_many, \
        1e1**rad_trans_params['log_delta'],1e1**rad_trans_params['log_gamma'], \
        rad_trans_params['t_int'],rad_trans_params['t_equ'], \
        1e1**rad_trans_params['log_p_trans'],rad_trans_params['alpha'])

    # new
    press_many_new = 1e1**running_mean(np.log10(press_many), 25)
    t_new          = running_mean(t_no_ave  , 25)
    index_new      = (press_many_new <= 1e3) & (press_many_new >= 1e-6)
    temp_new       = t_new[index_new][::2]
    press_new      = press_many_new[index_new][::2]

    return press_new, temp_new

### Function to make temp
def make_press_temp_iso(rad_trans_params):

    press_many = np.logspace(-8,5,260)
    t_no_ave = rad_trans_params['t_equ']  * np.ones_like(press_many)

    # new
    press_many_new = 1e1**running_mean(np.log10(press_many), 25)
    t_new          = running_mean(t_no_ave  , 25)
    index_new      = (press_many_new <= 1e3) & (press_many_new >= 1e-6)
    temp_new       = t_new[index_new][::2]
    press_new      = press_many_new[index_new][::2]

    return press_new, temp_new


# Future functions
def convolve_rebin(input_wavelengths, input_flux,
                   instrument_resolving_power, pixel_sampling, instrument_wavelength_range):
    from scipy.ndimage import gaussian_filter1d
    import petitRADTRANS.fort_rebin as fr
    """
    Function to convolve observation with instrument obs and rebin to pixels of detector.
    Create mock observation for high-res spectrograph.

    Args:
        input_wavelengths: (cm) wavelengths of the input spectrum
        input_flux: flux of the input spectrum
        instrument_resolving_power: resolving power of the instrument
        pixel_sampling: number of pixels per resolution elements (i.e. how many px in one LSF FWHM, usually 2)
        instrument_wavelength_range: (um) wavelength range of the instrument

    Returns:
        flux_lsf: flux altered by the instrument's LSF
        freq_out: (Hz) frequencies of the rebinned flux, in descending order
        flux_rebin: the rebinned flux
    """
    # From talking to Ignas: delta lambda of resolution element is the FWHM of the instrument's LSF (here: a gaussian)
    sigma_lsf = 1. / instrument_resolving_power / (2. * np.sqrt(2. * np.log(2.)))

    # The input resolution of petitCODE is 1e6, but just compute
    # it to be sure, or more versatile in the future.
    # Also, we have a log-spaced grid, so the resolution is constant
    # as a function of wavelength
    model_resolving_power = np.mean(
        (input_wavelengths[1:] + input_wavelengths[:-1]) / (2. * np.diff(input_wavelengths))
    )

    # Calculate the sigma to be used in the gauss filter in units of input frequency bins
    sigma_lsf_gauss_filter = sigma_lsf * model_resolving_power

    flux_lsf = gaussian_filter1d(
        input=input_flux,
        sigma=sigma_lsf_gauss_filter,
        mode='reflect'
    )

    if np.size(instrument_wavelength_range) == 2:  # TODO check if this is still working
        wavelength_out_borders = np.logspace(
            np.log10(instrument_wavelength_range[0]),
            np.log10(instrument_wavelength_range[1]),
            int(pixel_sampling * instrument_resolving_power
                * np.log(instrument_wavelength_range[1] / instrument_wavelength_range[0]))
        )
        wavelengths_out = (wavelength_out_borders[1:] + wavelength_out_borders[:-1]) / 2.
    elif np.size(instrument_wavelength_range) > 2:
        wavelengths_out = instrument_wavelength_range
    else:
        raise ValueError(f"instrument wavelength must be of size 2 or more, "
                         f"but is of size {np.size(instrument_wavelength_range)}: {instrument_wavelength_range}")

    flux_rebin = fr.rebin_spectrum(input_wavelengths, flux_lsf, wavelengths_out)

    return flux_lsf, wavelengths_out, flux_rebin


def radiosity_erg_hz2radiosity_erg_cm(radiosity_erg_hz, frequency):
    """Convert a radiosity from erg.s-1.cm-2.sr-1/Hz to erg.s-1.cm-2.sr-1/cm at a given frequency.  # TODO move to physics

    Steps:
        [cm] = c[cm.s-1] / [Hz]
        => d[cm]/d[Hz] = d(c / [Hz])/d[Hz]
        => d[cm]/d[Hz] = c / [Hz]**2
        => d[Hz]/d[cm] = [Hz]**2 / c
        integral of flux must be conserved: radiosity_erg_cm * d[cm] = radiosity_erg_hz * d[Hz]
        radiosity_erg_cm = radiosity_erg_hz * d[Hz]/d[cm]
        => radiosity_erg_cm = radiosity_erg_hz * frequency**2 / c

    Args:
        radiosity_erg_hz: (erg.s-1.cm-2.sr-1/Hz)
        frequency: (Hz)

    Returns:
        (erg.s-1.cm-2.sr-1/cm) the radiosity in converted units
    """
    return radiosity_erg_hz * frequency ** 2 / c
