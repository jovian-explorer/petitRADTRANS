from __future__ import print_function

import os

import astropy.constants as anc
import h5py
import numpy as np
import scipy.constants as snc
from petitRADTRANS import petitradtrans_config

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


def b(temperature, nu):
    """ Returns the Planck function :math:`B_{\\nu}(T)` in units of
    :math:`\\rm erg/s/cm^2/Hz/steradian`.

    Args:
        temperature (float):
            Temperature in K.
        nu:
            numpy array containing the frequency in Hz.
    """

    ret_val = 2. * h * nu ** 3. / c ** 2.
    ret_val = ret_val / (np.exp(h * nu / kB / temperature) - 1.)
    return ret_val


def d_b_d_temperature(temperature, nu):
    ret_val = 2. * h * nu ** 3. / c ** 2.
    ret_val = ret_val / (np.exp(h * nu / kB / temperature) - 1.) ** 2.
    ret_val = ret_val * np.exp(h * nu / kB / temperature) * h * nu / kB / temperature ** 2.
    return ret_val


def guillot_day(pressure, kappa_ir, gamma, grav, t_int, t_irr):
    tau = pressure * 1e6 * kappa_ir / grav
    temperature = (
                          0.75 * t_int ** 4. * (2. / 3. + tau) + 0.75 * t_irr ** 4. / 2. * (
                                  2. / 3. + 1. / gamma / 3. ** 0.5
                                  + (gamma / 3. ** 0.5 - 1. / 3. ** 0.5 / gamma)
                                  * np.exp(-gamma * tau * 3. ** 0.5)
                          )
                   ) ** 0.25
    return temperature


def guillot_global(pressure, kappa_ir, gamma, grav, t_int, t_equ):
    """ Returns a temperature array, in units of K,
    of the same dimensions as the pressure P
    (in bar). For this the temperature model of Guillot (2010)
    is used (his Equation 29).

    Args:
        pressure:
            numpy array of floats, containing the input pressure in bars.
        kappa_ir (float):
            The infrared opacity in units of :math:`\\rm cm^2/s`.
        gamma (float):
            The ratio between the visual and infrated opacity.
        grav (float):
            The planetary surface gravity in units of :math:`\\rm cm/s^2`.
        t_int (float):
            The planetary internal temperature (in units of K).
        t_equ (float):
            The planetary equilibrium temperature (in units of K).
    """
    tau = pressure * 1e6 * kappa_ir / grav
    t_irr = t_equ * np.sqrt(2.)
    temperature = (
                0.75 * t_int ** 4. * (2. / 3. + tau)
                + 0.75 * t_irr ** 4. / 4.
                * (2. / 3. + 1. / gamma / 3. ** 0.5
                   + (gamma / 3. ** 0.5 - 1. / 3. ** 0.5 / gamma) * np.exp(-gamma * tau * 3. ** 0.5))
         ) ** 0.25
    return temperature


def get_dist(t_irr, dist, t_star, r_star, mode, mode_what):
    mu_star = 0.
    angle_use = False
    if (mode != 'p') & (mode != 'd'):
        mu_star = float(mode)
        angle_use = True

    if mode_what == 'temp':
        if angle_use:
            t_irr = ((r_star * r_sun / (dist * AU)) ** 2. * t_star ** 4. * mu_star) ** 0.25
        elif mode == 'p':
            t_irr = ((r_star * r_sun / (dist * AU)) ** 2. * t_star ** 4. / 4.) ** 0.25
        else:
            t_irr = ((r_star * r_sun / (dist * AU)) ** 2. * t_star ** 4. / 2.) ** 0.25
        return t_irr
    elif mode_what == 'dist':
        if angle_use:
            dist = np.sqrt((r_star * r_sun) ** 2. * (t_star / t_irr) ** 4. * mu_star) / AU
        elif mode == 'p':
            dist = np.sqrt((r_star * r_sun) ** 2. * (t_star / t_irr) ** 4. / 4.) / AU
        else:
            dist = np.sqrt((r_star * r_sun) ** 2. * (t_star / t_irr) ** 4. / 2.) / AU
        return dist


logs_g = np.array([12., 10.93])

logs_met = np.array([1.05, 1.38, 2.7, 8.43, 7.83, 8.69, 4.56, 7.93, 6.24, 7.6, 6.45, 7.51, 5.41,
                     7.12, 5.5, 6.4, 5.03, 6.34, 3.15, 4.95, 3.93, 5.64, 5.43, 7.5,
                     4.99, 6.22, 4.19, 4.56, 3.04, 3.65, 3.25, 2.52, 2.87, 2.21, 2.58,
                     1.46, 1.88])


def calc_met(f):
    return np.log10((f / (np.sum(1e1 ** logs_g) + f * np.sum(1e1 ** logs_met)))
                    / (1. / (np.sum(1e1 ** logs_g) + np.sum(1e1 ** logs_met))))


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


def __load_stellar_spectra():
    with h5py.File(spec_path + os.path.sep + "stellar_spectra.h5", "r") as f:
        log_temp_grid = f['log10_effective_temperature'][()]
        star_rad_grid = f['radius'][()]
        spec_dats = f['spectral_radiance'][()]
        wavelength = f['wavelength'][()]

    return log_temp_grid, star_rad_grid, spec_dats, wavelength


pathinp = petitradtrans_config['Paths']['pRT_input_data_path']
spec_path = pathinp + os.path.sep + 'stellar_specs'

logTempGrid, StarRadGrid, specDats, wavelength_stellar = __load_stellar_spectra()


def __get_phoenix_spec_wrap(temperature):
    log_temp = np.log10(temperature)
    interpolation_index = np.searchsorted(logTempGrid, log_temp)

    if interpolation_index == 0:
        spec_dat = specDats[0]
        radius = StarRadGrid[0]
        print('Warning, input temperature is lower than minimum grid temperature.')
        print('Taking F = F_grid(minimum grid temperature), normalized to desired')
        print('input temperature.')

    elif interpolation_index == len(logTempGrid):
        spec_dat = specDats[int(len(logTempGrid) - 1)]
        radius = StarRadGrid[int(len(logTempGrid) - 1)]
        print('Warning, input temperature is higher than maximum grid temperature.')
        print('Taking F = F_grid(maximum grid temperature), normalized to desired')
        print('input temperature.')

    else:
        weight_high = (log_temp - logTempGrid[interpolation_index - 1]) / \
                     (logTempGrid[interpolation_index] - logTempGrid[interpolation_index - 1])

        weight_low = 1. - weight_high

        spec_dat_low = specDats[int(interpolation_index - 1)]

        spec_dat_high = specDats[int(interpolation_index)]

        spec_dat = weight_low * spec_dat_low \
            + weight_high * spec_dat_high

        radius = weight_low * StarRadGrid[int(interpolation_index - 1)] \
            + weight_high * StarRadGrid[int(interpolation_index)]

    freq = c / wavelength_stellar
    flux = spec_dat
    norm = -np.sum((flux[1:] + flux[:-1]) * np.diff(freq)) / 2.

    spec_dat = flux / norm * sigma * temperature ** 4.

    spec_dat = np.transpose(np.stack((wavelength_stellar, spec_dat)))

    return spec_dat, radius


def get_PHOENIX_spec(temperature):
    """
    Returns a matrix where the first column is the wavelength in cm
    and the second is the stellar flux :math:`F_\\nu` in units of
    :math:`\\rm erg/cm^2/s/Hz`, at the surface of the star.
    The spectra are PHOENIX models from (Husser et al. 2013), the spectral
    grid used here was described in van Boekel et al. (2012).

    Args:
        temperature (float):
            stellar effective temperature in K.
    """
    spec_dat, radius = __get_phoenix_spec_wrap(temperature)

    return spec_dat


def get_PHOENIX_spec_rad(temperature):
    """
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
    """
    spec_dat, radius = __get_phoenix_spec_wrap(temperature)

    return spec_dat, radius


def __phoenix_spec_dat2h5():
    """
    Convert a PHOENIX stellar spectrum in .dat format to HDF5 format.
    """
    # Load the stellar parameters
    description = np.genfromtxt(spec_path + os.path.sep + 'stellar_params.dat')

    # Initialize the grids
    log_temp_grid = description[:, 0]
    star_rad_grid = description[:, 1]

    # Load the corresponding numbered spectral files
    spec_dats = []

    for spec_num in range(len(log_temp_grid)):
        spec_dats.append(np.genfromtxt(spec_path + '/spec_'
                                       + str(int(spec_num)).zfill(2) + '.dat'))

    # Write the HDF5 file
    with h5py.File("stellar_spectra.h5", "w") as f:
        t_eff = f.create_dataset(
            name='log10_effective_temperature',
            data=log_temp_grid
        )
        t_eff.attrs['units'] = 'log10(K)'

        radius = f.create_dataset(
            name='radius',
            data=star_rad_grid
        )
        radius.attrs['units'] = 'R_sun'

        mass = f.create_dataset(
            name='mass',
            data=description[:, 2]
        )
        mass.attrs['units'] = 'M_sun'

        wavelength = f.create_dataset(
            name='wavelength',
            data=np.asarray(spec_dats)[0, :, 0]
        )
        wavelength.attrs['units'] = 'cm'

        intensity = f.create_dataset(
            name='spectral_radiance',
            data=np.asarray(spec_dats)[:, :, 1]
        )
        intensity.attrs['units'] = 'erg/s/sr/cm^2/Hz'


# Radtrans utility for retrieval temperature model computation

# Box car conv. average, found on stackoverflow somewhere
def running_mean(x, n):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[n:] - cumsum[:-n]) / float(n)


# Global Guillot P-T formula with kappa/grav replaced by delta
def guillot_global_ret(pressure, delta, gamma, t_int, t_equ):
    delta = np.abs(delta)
    gamma = np.abs(gamma)
    t_int = np.abs(t_int)
    t_equ = np.abs(t_equ)
    tau = pressure * 1e6 * delta
    t_irr = t_equ * np.sqrt(2.)
    temperature = (0.75 * t_int ** 4. * (2. / 3. + tau)
                   + 0.75 * t_irr ** 4. / 4.
                   * (2. / 3. + 1. / gamma / 3. ** 0.5
                      + (gamma / 3. ** 0.5 - 1. / 3. ** 0.5 / gamma)
                      * np.exp(-gamma * tau * 3. ** 0.5))) ** 0.25
    return temperature


# Modified Guillot P-T formula
def guillot_modif(pressure, delta, gamma, t_int, t_equ, ptrans, alpha):
    return guillot_global_ret(pressure, np.abs(delta), np.abs(gamma),
                              np.abs(t_int), np.abs(t_equ)) * \
           (1. - alpha * (1. / (1. +
                                np.exp((np.log(pressure / ptrans))))))


# Function to make temp
def make_press_temp(rad_trans_params):  # TODO pressure grid in input?
    press_many = np.logspace(-8, 5, 260)
    t_no_ave = guillot_modif(press_many,
                             1e1 ** rad_trans_params['log_delta'], 1e1 ** rad_trans_params['log_gamma'],
                             rad_trans_params['t_int'], rad_trans_params['t_equ'],
                             1e1 ** rad_trans_params['log_p_trans'], rad_trans_params['alpha'])

    # new
    press_many_new = 1e1 ** running_mean(np.log10(press_many), 25)
    t_new = running_mean(t_no_ave, 25)
    index_new = (press_many_new <= 1e3) & (press_many_new >= 1e-6)
    temp_new = t_new[index_new][::2]
    press_new = press_many_new[index_new][::2]

    return press_new, temp_new


# Function to make temp
def make_press_temp_iso(rad_trans_params):
    press_many = np.logspace(-8, 5, 260)
    t_no_ave = rad_trans_params['t_equ'] * np.ones_like(press_many)

    # new
    press_many_new = 1e1 ** running_mean(np.log10(press_many), 25)
    t_new = running_mean(t_no_ave, 25)
    index_new = (press_many_new <= 1e3) & (press_many_new >= 1e-6)
    temp_new = t_new[index_new][::2]
    press_new = press_many_new[index_new][::2]

    return press_new, temp_new
