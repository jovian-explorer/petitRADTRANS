import numpy as np
from petitRADTRANS import Radtrans

Chubb = False
do_scat_emis = True

if not Chubb:
    atmosphere = Radtrans(line_species = ['H2O', 'CO_all_iso', 'CH4', 'CO2', 'Na', 'K'], \
                          rayleigh_species = ['H2', 'He'], \
                          continuum_opacities = ['H2-H2', 'H2-He'], \
                          wlen_bords_micron = [1, 2.5], \
                          do_scat_emis = do_scat_emis,
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
abundances['CO_all_iso'] = 0.01 * np.ones_like(temperature)
abundances['CO2'] = 0.00001 * np.ones_like(temperature)
abundances['CH4'] = 0.000001 * np.ones_like(temperature)
if not Chubb:
    abundances['Na'] = 0.00001 * np.ones_like(temperature)
    abundances['K'] = 0.000001 * np.ones_like(temperature)
    abundances['H2O'] = 0.001 * np.ones_like(temperature)
else:
    abundances['Na_Chubb_mass'] = 0.00001 * np.ones_like(temperature)
    abundances['K_Chubb_mass'] = 0.000001 * np.ones_like(temperature)
    abundances['H2O_Chubb_mass'] = 0.001 * np.ones_like(temperature)

MMW = 2.33 * np.ones_like(temperature)

'''
atmosphere.calc_transm(temperature, abundances, \
                       gravity, MMW, R_pl=R_pl, P0_bar=P0, \
                       contribution = True)

import pylab as plt
plt.rcParams['figure.figsize'] = (10, 6)

wlen_mu = nc.c/atmosphere.freq/1e-4
X, Y = np.meshgrid(wlen_mu, pressures)
plt.contourf(X,Y,atmosphere.contr_tr,30,cmap=plt.cm.bone_r)

plt.yscale('log')
plt.xscale('log')
plt.ylim([1e2,1e-6])
plt.xlim([np.min(wlen_mu),np.max(wlen_mu)])

plt.xlabel('Wavelength (microns)')
plt.ylabel('P (bar)')
plt.title('Transmission contribution function')
plt.show()
plt.clf()
'''
import time
b = time.time()
for i in range(10):
    atmosphere.calc_flux(temperature, abundances, \
                         gravity, MMW, \
                         contribution = True)
b = time.time() - b
print('Time: ', b/10)

import pylab as plt
plt.rcParams['figure.figsize'] = (10, 6)

wlen_mu = nc.c/atmosphere.freq/1e-4
X, Y = np.meshgrid(wlen_mu, pressures)
plt.contourf(X,Y,atmosphere.contr_em,30,cmap=plt.cm.bone_r)

plt.yscale('log')
plt.xscale('log')
plt.ylim([1e2,1e-6])
plt.xlim([np.min(wlen_mu),np.max(wlen_mu)])

plt.xlabel('Wavelength (microns)')
plt.ylabel('P (bar)')
plt.title('Emission contribution function')
plt.show()
plt.clf()
