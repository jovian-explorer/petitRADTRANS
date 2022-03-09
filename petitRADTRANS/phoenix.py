"""
"""
import os

import h5py
import numpy as np

import petitRADTRANS.nat_cst as nc
from petitRADTRANS.config import petitradtrans_config


def __load_stellar_spectra():
    with h5py.File(spec_path + os.path.sep + "stellar_spectra.h5", "r") as f:
        log_temp_grid = f['log10_effective_temperature'][()]
        star_rad_grid = f['radius'][()]
        spec_dats = f['spectral_radiosity'][()]
        wavelength = f['wavelength'][()]

    return log_temp_grid, star_rad_grid, spec_dats, wavelength


spec_path = os.path.join(petitradtrans_config['Paths']['pRT_input_data_path'], 'stellar_specs')

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

    freq = nc.c / wavelength_stellar
    flux = spec_dat
    norm = -np.sum((flux[1:] + flux[:-1]) * np.diff(freq)) / 2.

    spec_dat = flux / norm * nc.sigma * temperature ** 4.

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
    spec_dat, _ = __get_phoenix_spec_wrap(temperature)

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
