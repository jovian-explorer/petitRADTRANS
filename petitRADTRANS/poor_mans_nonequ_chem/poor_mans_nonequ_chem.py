"""
Created on Mon May 29 14:44:35 2017

@author: Paul Mollière
"""
import numpy as np
import h5py
import os
from .chem_fortran_util import chem_fortran_util as cfu
import copy as cp
from petitRADTRANS import petitradtrans_config

path = petitradtrans_config['Paths']['pRT_input_data_path']


def __chem_table_dat2h5():
    # Read in parameters of chemistry grid
    feh = np.genfromtxt(os.path.join(path, "abundance_files/FEHs.dat"))
    co_ratios = np.genfromtxt(os.path.join(path, "abundance_files/COs.dat"))
    temperature = np.genfromtxt(os.path.join(path, "abundance_files/temps.dat"))
    pressure = np.genfromtxt(os.path.join(path, "abundance_files/pressures.dat"))

    with open(os.path.join(path, "abundance_files/species.dat"), 'r') as f:
        species_name = f.readlines()

    for i in range(len(species_name)):
        species_name[i] = species_name[i][:-1]

    chemistry_table = cfu.read_data(
        int(len(feh)), int(len(co_ratios)), int(len(temperature)), int(len(pressure)), int(len(species_name)),
        path + '/'
    )

    chemistry_table = np.array(chemistry_table, dtype='d', order='F')

    with h5py.File(f"{path}{os.path.sep}abundance_files{os.path.sep}mass_mixing_ratios.h5", 'w') as f:
        feh = f.create_dataset(
            name='iron_to_hydrogen_ratios',
            data=feh
        )
        feh.attrs['units'] = 'dex'

        co = f.create_dataset(
            name='carbon_to_oxygen_ratios',
            data=co_ratios
        )
        co.attrs['units'] = 'None'

        temp = f.create_dataset(
            name='temperatures',
            data=temperature
        )
        temp.attrs['units'] = 'K'

        p = f.create_dataset(
            name='pressures',
            data=pressure
        )
        p.attrs['units'] = 'bar'

        name = f.create_dataset(
            name='species_names',
            data=species_name
        )
        name.attrs['units'] = 'N/A'

        table = f.create_dataset(
            name='mass_mixing_ratios',
            data=chemistry_table
        )
        table.attrs['units'] = 'None'


def __load_mass_mixing_ratios():
    with h5py.File(f"{path}{os.path.sep}abundance_files{os.path.sep}mass_mixing_ratios.h5", 'r') as f:
        feh = f['iron_to_hydrogen_ratios'][()]
        co = f['carbon_to_oxygen_ratios'][()]
        temperatures = f['temperatures'][()]
        pressures_ = f['pressures'][()]

        species_names = f['species_names'][()]
        species_names = np.array([name.decode('utf-8') for name in species_names])

        mass_mixing_ratios = f['mass_mixing_ratios'][()]

    return feh, co, temperatures, pressures_, species_names, mass_mixing_ratios


# Read in parameters of chemistry grid
FEHs, COs, temps, pressures, names, chem_table = __load_mass_mixing_ratios()

chem_table = np.array(chem_table, order='F')  # change the order to column-wise (Fortran) to increase the speed


def interpol_abundances(COs_goal_in, FEHs_goal_in, temps_goal_in, pressures_goal_in,
                        Pquench_carbon = None):
    """
    Interpol abundances to desired coordinates.
    """

    COs_goal, FEHs_goal, temps_goal, pressures_goal = \
      cp.copy(COs_goal_in), cp.copy(FEHs_goal_in), cp.copy(temps_goal_in), cp.copy(pressures_goal_in)

    # Apply boundary treatment
    COs_goal[COs_goal <= np.min(COs)] = np.min(COs) + 1e-6
    COs_goal[COs_goal >= np.max(COs)] = np.max(COs) - 1e-6

    FEHs_goal[FEHs_goal <= np.min(FEHs)] = np.min(FEHs) + 1e-6
    FEHs_goal[FEHs_goal >= np.max(FEHs)] = np.max(FEHs) - 1e-6

    temps_goal[temps_goal <= np.min(temps)] = np.min(temps) + 1e-6
    temps_goal[temps_goal >= np.max(temps)] = np.max(temps) - 1e-6

    pressures_goal[pressures_goal <= np.min(pressures)] = np.min(pressures) \
        + 1e-6
    pressures_goal[pressures_goal >= np.max(pressures)] = np.max(pressures) \
        - 1e-6

    # Get interpolation indices
    COs_large_int = np.searchsorted(COs, COs_goal)+1
    FEHs_large_int = np.searchsorted(FEHs, FEHs_goal)+1
    temps_large_int = np.searchsorted(temps, temps_goal)+1
    pressures_large_int = np.searchsorted(pressures, pressures_goal)+1

    # Get the interpolated values from Fortran routine
    abundances_arr = cfu.interpolate(COs_goal, FEHs_goal, temps_goal,
                                      pressures_goal, COs_large_int,
                                      FEHs_large_int, temps_large_int,
                                      pressures_large_int, FEHs, COs, temps,
                                      pressures, chem_table)

    # Sort in output format of this function
    abundances = {}
    for id, name in enumerate(names):
        abundances[name] = abundances_arr[id, :]

    # Carbon quenching? Assumes pressures_goal is sorted in ascending order
    if Pquench_carbon is not None:
        if Pquench_carbon > np.min(pressures_goal):

            q_index = min(np.searchsorted(pressures_goal, Pquench_carbon),
                          int(len(pressures_goal))-1)

            methane_abb = abundances['CH4']
            methane_abb[pressures_goal < Pquench_carbon] = \
                abundances['CH4'][q_index]
            abundances['CH4'] = methane_abb

            co_abb = abundances['CO']
            co_abb[pressures_goal < Pquench_carbon] = \
                abundances['CO'][q_index]
            abundances['CO'] = co_abb

            h2o_abb = abundances['H2O']
            h2o_abb[pressures_goal < Pquench_carbon] = \
                abundances['H2O'][q_index]
            abundances['H2O'] = h2o_abb

    return abundances
