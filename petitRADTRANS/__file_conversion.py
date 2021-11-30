"""Stores functions that convert files from a format to another.

The functions in this module are stored for the sake of keeping trace of changes made to files. They are intended to be
used only once.
"""
import os

import h5py
import numpy as np

import petitRADTRANS.nat_cst as nc


def __phoenix_spec_dat2h5():
    """
    Convert a PHOENIX stellar spectrum in .dat format to HDF5 format.
    """
    # Load the stellar parameters
    description = np.genfromtxt(nc.spec_path + os.path.sep + 'stellar_params.dat')

    # Initialize the grids
    log_temp_grid = description[:, 0]
    star_rad_grid = description[:, 1]

    # Load the corresponding numbered spectral files
    spec_dats = []

    for spec_num in range(len(log_temp_grid)):
        spec_dats.append(np.genfromtxt(nc.spec_path + '/spec_'
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

        spectral_type = f.create_dataset(
            name='spectral_type',
            data=description[:, -1]
        )
        spectral_type.attrs['units'] = 'None'

        wavelength = f.create_dataset(
            name='wavelength',
            data=np.asarray(spec_dats)[0, :, 0]
        )
        wavelength.attrs['units'] = 'cm'

        spectral_radiosity = f.create_dataset(
            name='spectral_radiosity',
            data=np.asarray(spec_dats)[:, :, 1]
        )
        spectral_radiosity.attrs['units'] = 'erg/s/cm^2/Hz'
