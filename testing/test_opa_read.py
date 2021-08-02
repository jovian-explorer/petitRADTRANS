import numpy as np
import pylab as plt

from petitRADTRANS import Radtrans
from petitRADTRANS import nat_cst as nc

import os
os.environ["pRT_input_data_path"] = "/Applications/ownpy/petitRADTRANS/petitRADTRANS/input_data"

wlen_ranges = [[0.04, 20],
               [0.5, 300],
               [0.5, 20],
               [0.04, 300]]

combinations = [['CO_all_iso', 'CH4_Chubb'],
                ['CH4_Chubb', 'CO_all_iso'],
                ['CO_all_iso', 'CH4'],
                ['CH4', 'CO_all_iso'],
                ['CO_all_iso_R_200', 'CH4_R_200'],
                ['CH4_R_200', 'CO_all_iso_R_200'],
                ['CO_all_iso_R_50', 'CH4_R_50'],
                ['CH4_R_50', 'CO_all_iso_R_50'],
                ['CO_all_iso_R_10', 'CH4_R_10'],
                ['CH4_R_10', 'CO_all_iso_R_10']]

pressures = np.logspace(-6, 2, 70)
temperature = 1200. * np.ones_like(pressures)

mass_fractions = {}
mass_fractions['H2_R_10'] = 0.85 * np.ones_like(temperature)
mass_fractions['CO_all_iso'] = 0.01 * np.ones_like(temperature)
mass_fractions['CH4'] = 0.001 * np.ones_like(temperature)
mass_fractions['CH4_Chubb'] = 0.001 * np.ones_like(temperature)
mass_fractions['CO_all_iso_R_10'] = 0.01 * np.ones_like(temperature)
mass_fractions['CH4_R_10'] = 0.001 * np.ones_like(temperature)
mass_fractions['CO_all_iso_R_50'] = 0.01 * np.ones_like(temperature)
mass_fractions['CH4_R_50'] = 0.001 * np.ones_like(temperature)
mass_fractions['CO_all_iso_R_200'] = 0.01 * np.ones_like(temperature)
mass_fractions['CH4_R_200'] = 0.001 * np.ones_like(temperature)


MMW = 2.33 * np.ones_like(temperature)

R_pl = 1.838*nc.r_jup_mean
gravity = 1e1**2.45
P0 = 0.01

for rang in wlen_ranges:
    for comb in combinations:

        atmosphere = Radtrans(line_species=comb,
                      wlen_bords_micron= rang)
        atmosphere.setup_opa_structure(pressures)

        atmosphere.calc_transm(temperature, mass_fractions, gravity, MMW, R_pl=R_pl, P0_bar=P0)

        plt.plot(nc.c / atmosphere.freq / 1e-4, atmosphere.transm_rad / nc.r_jup_mean, label=comb[0]+', '+comb[1])

    plt.xscale('log')
    plt.xlabel('Wavelength (microns)')
    plt.ylabel(r'Transit radius ($\rm R_{Jup}$)')
    plt.legend(loc='best')
    plt.title(str(rang[0])+'-'+str(rang[1]))
    plt.show()