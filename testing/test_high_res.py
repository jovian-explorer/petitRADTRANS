import numpy as np
from petitRADTRANS import Radtrans

import os

atmosphere = Radtrans(line_species = ['H2O_main_iso', 'CO_all_iso', \
                'CH4_main_iso', 'CO2_main_iso', 'Na', 'K'], \
                rayleigh_species = ['H2', 'He'], \
                continuum_opacities = ['H2-H2', 'H2-He'], \
                wlen_bords_micron = [2.2, 2.4], \
                mode = 'lbl', \
                do_scat_emis = True)

pressures = np.logspace(-10, 2, 130)
atmosphere.setup_opa_structure(pressures)

import petitRADTRANS.nat_cst as nc

R_pl = 1.838*nc.r_jup_mean
gravity = 1e1**2.45
P0 = 0.01

kappa_IR = 0.01
gamma = 0.4
T_int = 200.
T_equ = 1500.
temperature = nc.guillot_global(pressures, kappa_IR, gamma, gravity, T_int, T_equ)

abundances = {}
abundances['H2'] = 0.74 * np.ones_like(temperature)
abundances['He'] = 0.24 * np.ones_like(temperature)
abundances['H2O_main_iso'] = 0.001 * np.ones_like(temperature)
abundances['CO_all_iso'] = 0.01 * np.ones_like(temperature)
abundances['CO2_main_iso'] = 0.00001 * np.ones_like(temperature)
abundances['CH4_main_iso'] = 0.000001 * np.ones_like(temperature)
abundances['Na'] = 0.00001 * np.ones_like(temperature)
abundances['K'] = 0.000001 * np.ones_like(temperature)

MMW = 2.33 * np.ones_like(temperature)

atmosphere.calc_transm(temperature, abundances, gravity, MMW, R_pl=R_pl, P0_bar=P0)

import pylab as plt
plt.rcParams['figure.figsize'] = (10, 6)

plt.plot(nc.c/atmosphere.freq/1e-4, atmosphere.transm_rad/nc.r_jup_mean)

plt.xlabel('Wavelength (microns)')
plt.ylabel(r'Transit radius ($\rm R_{Jup}$)')
plt.show()
plt.clf()

plt.plot(nc.c/atmosphere.freq/1e-4, atmosphere.transm_rad/nc.r_jup_mean)

plt.xlim([2.3,2.3025])
plt.xscale('log')
plt.xlabel('Wavelength (microns)')
plt.ylabel(r'Transit radius ($\rm R_{Jup}$)')
plt.show()
plt.clf()

atmosphere.calc_flux(temperature, abundances, gravity, MMW)

plt.plot(nc.c/atmosphere.freq/1e-4, atmosphere.flux/1e-6)

plt.xlabel('Wavelength (microns)')
plt.ylabel(r'Planet flux $F_\nu$ (10$^{-6}$ erg cm$^{-2}$ s$^{-1}$ Hz$^{-1}$)')
plt.show()
plt.clf()

plt.plot(nc.c/atmosphere.freq/1e-4, atmosphere.flux/1e-6)

plt.xlim([2.3,2.3025])
plt.xlabel('Wavelength (microns)')
plt.ylabel(r'Planet flux $F_\nu$ (10$^{-6}$ erg cm$^{-2}$ s$^{-1}$ Hz$^{-1}$)')
plt.show()
plt.clf()
