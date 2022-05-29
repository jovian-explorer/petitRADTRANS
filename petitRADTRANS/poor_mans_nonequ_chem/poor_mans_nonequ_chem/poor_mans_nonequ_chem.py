#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 29 14:44:35 2017

@author: Paul Mollière
"""
import numpy as np
import os
from ..chem_fortran_util import read_data, interpolate
import copy as cp

path = os.environ.get("pRT_input_data_path")
if path == None:
    raise OSError(f"Path to input data not specified!\n"
                  f"Please set pRT_input_data_path variable in .bashrc / .bash_profile or specify path via\n"
                  f">>> import os"
                  f">>> os.environ['pRT_input_data_path'] = 'absolute/path/of/the/folder/input_data'\n"
                  f"before creating a Radtrans object or loading the nat_cst module.\n"
                  f"(this will become unnecessary in a future update)"
                  )

# Read in parameters of chemistry grid
FEHs = np.genfromtxt(os.path.join(path, "abundance_files/FEHs.dat"))
COs = np.genfromtxt(os.path.join(path, "abundance_files/COs.dat"))
temps = np.genfromtxt(os.path.join(path, "abundance_files/temps.dat"))
pressures = np.genfromtxt(os.path.join(path, "abundance_files/pressures.dat"))
f = open(os.path.join(path, "abundance_files/species.dat"))
names = f.readlines()
f.close()
for id in range(len(names)):
    names[id] = names[id][:-1]

chem_table = read_data(int(len(FEHs)), int(len(COs)), int(len(temps)),
                            int(len(pressures)), int(len(names)), path+'/')

chem_table = np.array(chem_table, dtype='d', order='F')

def interpol_abundances(COs_goal_in, FEHs_goal_in, temps_goal_in, pressures_goal_in,
                        Pquench_carbon = None):
    """
    Interpol abundances to desired coordinates.
    """

    COs_goal, FEHs_goal, temps_goal, pressures_goal = \
      cp.copy(COs_goal_in), cp.copy(FEHs_goal_in), cp.copy(temps_goal_in), cp.copy(pressures_goal_in)
    COs_goal, FEHs_goal, temps_goal, pressures_goal = \
        np.array(COs_goal).reshape(-1), \
        np.array(FEHs_goal).reshape(-1), \
        np.array(temps_goal).reshape(-1), \
        np.array(pressures_goal).reshape(-1)

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
    abundances_arr = interpolate(COs_goal, FEHs_goal, temps_goal,
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
