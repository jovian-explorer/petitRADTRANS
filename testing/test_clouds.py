import numpy as np
from petitRADTRANS import Radtrans

import os

Chubb = False

if not Chubb:
    atmosphere = Radtrans(line_species = ['H2O', 'CO_all_iso', 'CH4', 'CO2', 'Na', 'K'], \
                          rayleigh_species = ['H2', 'He'], \
                          continuum_opacities = ['H2-H2', 'H2-He'], \
                          wlen_bords_micron = [0.3, 15.],
                          path_input_data = "/Applications/ownpy/petitRADTRANS/petitRADTRANS/input_data")

pressures = np.logspace(-6, 2, 100)
atmosphere.setup_opa_structure(pressures)

from petitRADTRANS import nat_cst as nc
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
if not Chubb:
    abundances['H2O'] = 0.001 * np.ones_like(temperature)
    abundances['Na'] = 0.00001 * np.ones_like(temperature)
    abundances['K'] = 0.000001 * np.ones_like(temperature)
else:
    abundances['H2O_Chubb_mass'] = 0.001 * np.ones_like(temperature)
    abundances['Na_Chubb_mass'] = 0.00001 * np.ones_like(temperature)
    abundances['K_Chubb_mass'] = 0.000001 * np.ones_like(temperature)
abundances['CO_all_iso'] = 0.01 * np.ones_like(temperature)
abundances['CO2'] = 0.00001 * np.ones_like(temperature)
abundances['CH4'] = 0.000001 * np.ones_like(temperature)

MMW = 2.33 * np.ones_like(temperature)

import pylab as plt
plt.rcParams['figure.figsize'] = (10, 6)

# Clear
atmosphere.calc_transm(temperature, abundances, \
                       gravity, MMW, R_pl=R_pl, P0_bar=P0)
plt.plot(nc.c/atmosphere.freq/1e-4, \
         atmosphere.transm_rad/nc.r_jup_mean, label = 'Clear')

kappa_zero = 0.01
gamma_scat = -4.

atmosphere.calc_transm(temperature, abundances, \
                    gravity, MMW, R_pl=R_pl, P0_bar=P0, \
                    kappa_zero = kappa_zero, gamma_scat = gamma_scat)

plt.plot(nc.c/atmosphere.freq/1e-4, \
         atmosphere.transm_rad/nc.r_jup_mean, \
         label = r'Powerlaw cloud, $\gamma = -4$')

kappa_zero = 0.01
gamma_scat = -2.

atmosphere.calc_transm(temperature, abundances, \
                    gravity, MMW, R_pl=R_pl, P0_bar=P0, \
                    kappa_zero = kappa_zero, gamma_scat = gamma_scat)

plt.plot(nc.c/atmosphere.freq/1e-4, \
         atmosphere.transm_rad/nc.r_jup_mean, \
         label = r'Powerlaw cloud, $\gamma = -2$')

kappa_zero = 0.01
gamma_scat = 0.

atmosphere.calc_transm(temperature, abundances, \
                    gravity, MMW, R_pl=R_pl, P0_bar=P0, \
                    kappa_zero = kappa_zero, gamma_scat = gamma_scat)

plt.plot(nc.c/atmosphere.freq/1e-4, \
         atmosphere.transm_rad/nc.r_jup_mean, \
         label = r'Powerlaw cloud, $\gamma = 0$')

kappa_zero = 0.01
gamma_scat = 1.

atmosphere.calc_transm(temperature, abundances, \
                    gravity, MMW, R_pl=R_pl, P0_bar=P0, \
                    kappa_zero = kappa_zero, gamma_scat = gamma_scat)

plt.plot(nc.c/atmosphere.freq/1e-4, \
         atmosphere.transm_rad/nc.r_jup_mean,\
         label = r'Powerlaw cloud, $\gamma = 1$')

# Make plot

plt.xscale('log')
plt.xlabel('Wavelength (microns)')
plt.ylabel(r'Transit radius ($\rm R_{Jup}$)')
plt.legend(loc = 'best')
plt.show()
plt.clf()

import pylab as plt
plt.rcParams['figure.figsize'] = (10, 6)

# Clear
atmosphere.calc_transm(temperature, abundances, gravity, MMW, R_pl=R_pl, P0_bar=P0)
plt.plot(nc.c/atmosphere.freq/1e-4, \
         atmosphere.transm_rad/nc.r_jup_mean, label = 'Clear')

# Gray cloud deck at 0.01 bar
atmosphere.calc_transm(temperature, abundances, gravity, MMW, R_pl=R_pl, P0_bar=P0, \
                      Pcloud = 0.01)
plt.plot(nc.c/atmosphere.freq/1e-4, \
         atmosphere.transm_rad/nc.r_jup_mean, label = 'Gray cloud deck at 0.01 bar')

# Haze (10 x gas Rayleigh scattering)
atmosphere.calc_transm(temperature, abundances, gravity, MMW, R_pl=R_pl, P0_bar=P0, \
                      haze_factor = 10)
plt.plot(nc.c/atmosphere.freq/1e-4, \
         atmosphere.transm_rad/nc.r_jup_mean, label = 'Rayleigh haze')

# Haze + cloud deck
atmosphere.calc_transm(temperature, abundances, gravity, MMW, R_pl=R_pl, P0_bar=P0, \
                      haze_factor = 10, Pcloud = 0.01)
plt.plot(nc.c/atmosphere.freq/1e-4, \
         atmosphere.transm_rad/nc.r_jup_mean, label = 'Rayleigh haze + cloud deck')

plt.xscale('log')
plt.xlabel('Wavelength (microns)')
plt.ylabel(r'Transit radius ($\rm R_{Jup}$)')
plt.legend(loc = 'best')
plt.show()
plt.clf()

import numpy as np

if not Chubb:
    atmosphere = Radtrans(line_species = ['H2O', 'CO_all_iso', 'CH4', 'CO2', 'Na', 'K'], \
                          cloud_species = ['Mg2SiO4(c)_cd'], \
                          rayleigh_species = ['H2', 'He'], \
                          continuum_opacities = ['H2-H2', 'H2-He'], \
                          wlen_bords_micron = [0.3, 15.],
                          path_input_data = "/Applications/ownpy/petitRADTRANS/petitRADTRANS/input_data")
else:
    atmosphere = Radtrans(line_species = ['H2O_Chubb_mass', 'CO_all_iso', 'CH4', 'CO2', 'Na_Chubb_mass', 'K_Chubb_mass'], \
                          cloud_species = ['Mg2SiO4(c)_cd'], \
                          rayleigh_species = ['H2', 'He'], \
                          continuum_opacities = ['H2-H2', 'H2-He'], \
                          wlen_bords_micron = [0.3, 15.])

pressures = np.logspace(-6, 2, 100)
atmosphere.setup_opa_structure(pressures)

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
abundances['CO_all_iso'] = 0.01 * np.ones_like(temperature)
abundances['CO2'] = 0.00001 * np.ones_like(temperature)
abundances['CH4'] = 0.000001 * np.ones_like(temperature)
if not Chubb:
    abundances['H2O'] = 0.001 * np.ones_like(temperature)
    abundances['Na'] = 0.00001 * np.ones_like(temperature)
    abundances['K'] = 0.000001 * np.ones_like(temperature)
else:
    abundances['H2O_Chubb_mass'] = 0.001 * np.ones_like(temperature)
    abundances['Na_Chubb_mass'] = 0.00001 * np.ones_like(temperature)
    abundances['K_Chubb_mass'] = 0.000001 * np.ones_like(temperature)
    
abundances['Mg2SiO4(c)'] = 0.0000005 * np.ones_like(temperature)

MMW = 2.33 * np.ones_like(temperature)

radius = {}
radius['Mg2SiO4(c)'] = 0.00005*np.ones_like(temperature) # I.e. a 0.5-micron particle size (0.00005 cm)

sigma_lnorm = 1.05

atmosphere.calc_transm(temperature, abundances, gravity, MMW, \
                       R_pl=R_pl, P0_bar=P0, \
                       radius = radius, sigma_lnorm = sigma_lnorm)

plt.plot(nc.c/atmosphere.freq/1e-4, atmosphere.transm_rad/nc.r_jup_mean, label = 'cloudy', zorder = 2)

abundances['Mg2SiO4(c)'] = np.zeros_like(temperature)

atmosphere.calc_transm(temperature, abundances, gravity, MMW, \
                       R_pl=R_pl, P0_bar=P0, \
                       radius = radius, sigma_lnorm = sigma_lnorm)
plt.plot(nc.c/atmosphere.freq/1e-4, atmosphere.transm_rad/nc.r_jup_mean, label = 'clear', zorder = 1)

plt.xscale('log')
plt.xlabel('Wavelength (microns)')
plt.ylabel(r'Transit radius ($\rm R_{Jup}$)')
plt.legend(loc='best')
plt.show()
plt.clf()

Kzz = np.ones_like(temperature)*1e1**7.5
fsed = 2.
sigma_lnorm = 1.05

abundances['Mg2SiO4(c)'] = 0.0000005 * np.ones_like(temperature)

atmosphere.calc_transm(temperature, abundances, gravity, MMW, \
                       R_pl=R_pl, P0_bar=P0, \
                       Kzz = Kzz, fsed=fsed, sigma_lnorm = sigma_lnorm)

plt.plot(nc.c/atmosphere.freq/1e-4, atmosphere.transm_rad/nc.r_jup_mean, label = 'cloudy', zorder = 2)

abundances['Mg2SiO4(c)'] = np.zeros_like(temperature)

atmosphere.calc_transm(temperature, abundances, gravity, MMW, \
                       R_pl=R_pl, P0_bar=P0, \
                       Kzz = Kzz, fsed=fsed, sigma_lnorm = sigma_lnorm)

plt.plot(nc.c/atmosphere.freq/1e-4, atmosphere.transm_rad/nc.r_jup_mean, label = 'clear', zorder = 1)

plt.xscale('log')
plt.xlabel('Wavelength (microns)')
plt.ylabel(r'Transit radius ($\rm R_{Jup}$)')
plt.legend(loc='best')
plt.show()
plt.clf()

plt.yscale('log')
plt.xscale('log')

plt.ylim([1e2,1e-6])

plt.ylabel('P (bar)')
plt.xlabel('Average particle size (microns)')

plt.plot(atmosphere.r_g[:,atmosphere.cloud_species.index('Mg2SiO4(c)')]/1e-4, pressures)
plt.show()
plt.clf()

abundances['Mg2SiO4(c)'] = 0.0000005 * np.ones_like(temperature)

atmosphere.calc_flux(temperature, abundances, gravity, MMW, \
                       Kzz = Kzz, fsed=fsed, sigma_lnorm = sigma_lnorm)

plt.plot(nc.c/atmosphere.freq/1e-4, atmosphere.flux/1e-6, \
         color = 'black', label = 'cloudy', zorder = 1)

atmosphere.calc_flux(temperature, abundances, gravity, MMW, \
                       Kzz = Kzz, fsed=fsed, sigma_lnorm = sigma_lnorm, \
                       add_cloud_scat_as_abs = True)
plt.plot(nc.c/atmosphere.freq/1e-4, atmosphere.flux/1e-6, \
         label = 'cloudy, add scattering to absorption', zorder = 2)

abundances['Mg2SiO4(c)'] = np.zeros_like(temperature)

atmosphere.calc_flux(temperature, abundances, gravity, MMW, \
                       Kzz = Kzz, fsed=fsed, sigma_lnorm = sigma_lnorm)

plt.plot(nc.c/atmosphere.freq/1e-4, atmosphere.flux/1e-6, '-', \
         color = 'red', label = 'clear', zorder = 0)

plt.legend(loc='best')
plt.xscale('log')
plt.xlabel('Wavelength (microns)')
plt.ylabel(r'Planet flux $F_\nu$ (10$^{-6}$ erg cm$^{-2}$ s$^{-1}$ Hz$^{-1}$)')
plt.show()
plt.clf()
