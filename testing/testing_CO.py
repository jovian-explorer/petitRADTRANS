import numpy as np
from petitRADTRANS import Radtrans

import os
os.environ["pRT_input_data_path"] = "/Users/molliere/Documents/programm_data/petitRADTRANS_public/input_data"

atmosphere = Radtrans(line_species = ['H2O_HITEMP', 'CO_all_iso_HITEMP', 'CH4', 'CO2', 'Na_allard', 'K_allard'], \
                          rayleigh_species = ['H2', 'He'], \
                          continuum_opacities = ['H2-H2', 'H2-He'], \
                          wlen_bords_micron = [0.3, 15])

atmosphere2 = Radtrans(line_species = ['H2O_HITEMP', 'CO_all_iso_Chubb', 'CH4', 'CO2', 'Na_allard', 'K_allard'], \
                          rayleigh_species = ['H2', 'He'], \
                          continuum_opacities = ['H2-H2', 'H2-He'], \
                          wlen_bords_micron = [0.3, 15])


pressures = np.logspace(-6, 2, 100)
atmosphere.setup_opa_structure(pressures)
atmosphere2.setup_opa_structure(pressures)

temperature = 1200. * np.ones_like(pressures)

abundances = {}
abundances['H2'] = 0.74 * np.ones_like(temperature)
abundances['He'] = 0.24 * np.ones_like(temperature)

abundances['H2O_HITEMP'] = 0.001 * np.ones_like(temperature)
abundances['Na_allard'] = 0.00001 * np.ones_like(temperature)
abundances['K_allard'] = 0.000001 * np.ones_like(temperature)
abundances['Na_burrows'] = 0.00001 * np.ones_like(temperature)
abundances['K_burrows'] = 0.000001 * np.ones_like(temperature)
abundances['Na_lor_cut'] = 0.00001 * np.ones_like(temperature)
abundances['K_lor_cut'] = 0.000001 * np.ones_like(temperature)

abundances['CO_all_iso_HITEMP'] = 0.01 * np.ones_like(temperature)
abundances['CO_all_iso_Chubb'] = 0.01 * np.ones_like(temperature)
abundances['CO2'] = 0.00001 * np.ones_like(temperature)
abundances['CH4'] = 0.000001 * np.ones_like(temperature)

MMW = 2.33 * np.ones_like(temperature)

from petitRADTRANS import nat_cst as nc
R_pl = 1.838*nc.r_jup_mean
gravity = 1e1**2.45
P0 = 0.01

atmosphere.calc_transm(temperature, abundances, gravity, MMW, R_pl=R_pl, P0_bar=P0)
atmosphere2.calc_transm(temperature, abundances, gravity, MMW, R_pl=R_pl, P0_bar=P0)

import pylab as plt
plt.rcParams['figure.figsize'] = (10, 6)

plt.plot(nc.c/atmosphere.freq/1e-4, atmosphere.transm_rad/nc.r_jup_mean, label = 'HITEMP')
plt.plot(nc.c/atmosphere.freq/1e-4, atmosphere2.transm_rad/nc.r_jup_mean, label = 'Exomol')

plt.xscale('log')
plt.xlabel('Wavelength (microns)')
plt.ylabel(r'Transit radius ($\rm R_{Jup}$)')
plt.legend()
plt.title('T = 1200 K')
plt.show()

temperature = 200. * np.ones_like(pressures)

atmosphere.calc_transm(temperature, abundances, gravity, MMW, R_pl=R_pl, P0_bar=P0)
atmosphere2.calc_transm(temperature, abundances, gravity, MMW, R_pl=R_pl, P0_bar=P0)

plt.plot(nc.c/atmosphere.freq/1e-4, atmosphere.transm_rad/nc.r_jup_mean, label = 'HITEMP')
plt.plot(nc.c/atmosphere.freq/1e-4, atmosphere2.transm_rad/nc.r_jup_mean, label = 'Exomol')

plt.xscale('log')
plt.xlabel('Wavelength (microns)')
plt.ylabel(r'Transit radius ($\rm R_{Jup}$)')
plt.legend()
plt.title('T = 200 K')
plt.show()
