from __future__ import print_function

import numpy as np
from . import nat_cst as nc

def py_calc_cloud_opas(rho,
                    rho_p,
                    cloud_mass_fracs,
                    r_g,
                    sigma_n,
                    cloud_rad_bins,
                    cloud_radii,
                    cloud_lambdas,
                    cloud_specs_abs_opa,
                    cloud_specs_scat_opa,
                    cloud_aniso):
    """
    This function reimplements calc_cloud_opas from fort_spec.f90. For some reason
    it runs faster in python than in fortran, so we'll use this from now on.
    This function integrates the cloud opacity throught the different layers of
    the atmosphere to get the total optical depth, scattering and anisotropic fraction.

    See the fortran implementation for details of the input arrays.
    """
    ncloud = int(cloud_mass_fracs.shape[1])
    N_cloud_rad_bins = int(cloud_radii.shape[0])
    N_cloud_lambda_bins = int(cloud_specs_abs_opa.shape[1])
    nstruct = int(rho.shape[0])

    cloud_abs_opa_TOT = np.zeros((N_cloud_lambda_bins,nstruct))
    cloud_scat_opa_TOT = np.zeros((N_cloud_lambda_bins,nstruct))
    cloud_red_fac_aniso_TOT = np.zeros((N_cloud_lambda_bins,nstruct))

    for i_struct in range(nstruct):
        for i_c in range(ncloud):
            N = 3.0 * cloud_mass_fracs[i_struct,i_c] * rho[i_struct] /(4.0 *np.pi*rho_p[i_c] * (r_g[i_struct,i_c]**3.0))* \
                np.exp(-4.5 * np.log(sigma_n)**2.0)

            dndr = N/(cloud_radii * np.sqrt(2.0*np.pi)*np.log(sigma_n)) *\
                    np.exp(-np.log(cloud_radii/r_g[i_struct,i_c])**2.0 /(2.0*(np.log(sigma_n)**2.0)))
            integrand_abs = (4.0*np.pi/3.0)*(cloud_radii[:,np.newaxis]**3.0)*rho_p[i_c]*\
                dndr[:,np.newaxis]*cloud_specs_abs_opa[:,:,i_c]
            integrand_scat = (4.0*np.pi/3.0)*(cloud_radii[:,np.newaxis]**3.0)*rho_p[i_c]*dndr[:,np.newaxis]* \
                cloud_specs_scat_opa[:,:,i_c]
            integrand_aniso = integrand_scat*(1.0-cloud_aniso[:,:,i_c])
            add_abs = np.sum(integrand_abs*(cloud_rad_bins[1:N_cloud_rad_bins+1,np.newaxis]- \
                cloud_rad_bins[0:N_cloud_rad_bins,np.newaxis]),axis = 0)

            cloud_abs_opa_TOT[:,i_struct] = cloud_abs_opa_TOT[:,i_struct] + add_abs

            add_scat = np.sum(integrand_scat*(cloud_rad_bins[1:N_cloud_rad_bins+1,np.newaxis]- \
                cloud_rad_bins[0:N_cloud_rad_bins,np.newaxis]),axis = 0)
            cloud_scat_opa_TOT[:,i_struct] = cloud_scat_opa_TOT[:,i_struct] + add_scat

            add_aniso = np.sum(integrand_aniso*(cloud_rad_bins[1:N_cloud_rad_bins+1,np.newaxis]- \
                cloud_rad_bins[0:N_cloud_rad_bins,np.newaxis]),axis = 0)
            cloud_red_fac_aniso_TOT[:,i_struct] = cloud_red_fac_aniso_TOT[:,i_struct] + add_aniso

        cloud_red_fac_aniso_TOT[:,i_struct] = cloud_red_fac_aniso_TOT[:,i_struct]/cloud_scat_opa_TOT[:,i_struct]
        cloud_red_fac_aniso_TOT[cloud_scat_opa_TOT<1e-200] = 0.0
        cloud_abs_opa_TOT[:,i_struct] /= rho[i_struct]
        cloud_scat_opa_TOT[:,i_struct] /= rho[i_struct]
    return cloud_abs_opa_TOT,cloud_scat_opa_TOT,cloud_red_fac_aniso_TOT


def sigma_hm_ff(lambda_angstroem, temp, P_e):
    '''
    Returns the H- free-free cross-section in units of cm^2
    per H per e- pressure (in cgs), as defined on page 156 of
    "The Observation and Analysis of Stellar Photospheres"
    by David F. Gray
    '''

    index = (lambda_angstroem >= 2600.) & (lambda_angstroem <= 113900.)
    lamb_use = lambda_angstroem[index]

    if temp >= 2500.:
        # Convert to Angstrom (from cgs)
        theta = 5040./temp

        f0 = -2.2763 - 1.6850 * np.log10(lamb_use) + \
          0.76661*np.log10(lamb_use)**2. \
          - 0.053346*np.log10(lamb_use)**3.
        f1 = 15.2827 - 9.2846 * np.log10(lamb_use) + \
          1.99381*np.log10(lamb_use)**2. \
          - 0.142631*np.log10(lamb_use)**3.
        f2 = 0. # -197.789 + 190.266 * np.log10(lamb_use) - 67.9775*np.log10(lamb_use)**2. \
                # + 10.6913*np.log10(lamb_use)**3. - 0.625151*np.log10(lamb_use)*4.
        # Once last term was commented out agreement was good. Otherwise opacity was way too large.

        retVal = np.zeros_like(lambda_angstroem)
        retVal[index] = 1e-26*P_e*1e1**(f0+f1*np.log10(theta)+ \
                                            f2*np.log10(theta)**2.)
        return retVal

    else:
        
        return np.zeros_like(lambda_angstroem)

def sigma_bf_mean(border_lambda_angstroem):
    '''
    Returns the H- bound-free cross-section in units of cm^2 \
    per H-, as defined on page 155 of
    "The Observation and Analysis of Stellar Photospheres"
    by David F. Gray
    '''

    left = border_lambda_angstroem[:-1]
    right = border_lambda_angstroem[1:]
    diff = np.diff(border_lambda_angstroem)

    a = []
    a.append(1.99654)
    a.append(-1.18267e-5)
    a.append(2.64243e-6)
    a.append(-4.40524e-10)
    a.append(3.23992e-14)
    a.append(-1.39568e-18)
    a.append(2.78701e-23)

    retVal = np.zeros_like(border_lambda_angstroem[1:])

    index = right <= 1.64e4

    for i_a in range(len(a)):
        retVal[index] += a[i_a]*(right[index]**(i_a+1)- \
                                     left[index]**(i_a+1))/(i_a+1)

    index_bracket = (left < 1.64e4) & (right > 1.64e4)
    for i_a in range(len(a)):
        retVal[index_bracket] += a[i_a]*((1.64e4)**(i_a+1)- \
                            left[index_bracket]**(i_a+1))/(i_a+1)
    #print(len(retVal[index_bracket]))

    index = (left+right)/2. > 1.64e4
    retVal[index] = 0.
    index = retVal < 0.
    retVal[index] = 0.

    return retVal*1e-18/diff


def hminus_opacity(lambda_angstroem, border_lambda_angstroem, \
                       temp, press, mmw, abundances):
    ''' Calc the H- opacity.'''

    retVal = np.array(np.zeros(len(lambda_angstroem)*len(press)).reshape( \
                        len(lambda_angstroem), \
                        len(press)),dtype='d',order='F')
    
    # Calc. electron number fraction
    # e- mass in amu:
    m_e = 5.485799e-4
    n_e = mmw/m_e * abundances['e-']

    # Calc. e- partial pressure
    P_e = press*n_e

    kappa_hminus_bf = sigma_bf_mean(border_lambda_angstroem)/nc.amu
    
    for i_struct in range(len(n_e)):
        #print(i_struct)
        kappa_hminus_ff = sigma_hm_ff(lambda_angstroem, temp[i_struct], \
          P_e[i_struct])/nc.amu*abundances['H'][i_struct]
        
        retVal[:,i_struct] = kappa_hminus_bf*abundances['H-'][i_struct] + \
          kappa_hminus_ff

    return retVal





#################################################
# Functions to read custom PT grids
#################################################

# Function to sort custom (potentially randomly sorted) PT grid of opacities
def sort_opa_PTgrid(path_ptg):

    # Read the Ps and Ts
    PTs = np.genfromtxt(path_ptg)

    # Read the file names
    f = open(path_ptg)
    lines = f.readlines()
    f.close()
    
    n_entries = len(lines)

    # Prepare the array to contain the
    # pressures, temperatures, indices in the unsorted list.
    # Also prepare the list of unsorted names
    PTind = np.ones(n_entries*3).reshape(n_entries, 3)
    names = []

    # Fill the array and name list
    for i_line in range(n_entries):

        line = lines[i_line]
        lsp = line.split(' ')

        PTind[i_line, 0], \
        PTind[i_line, 1], \
        PTind[i_line, 2] = \
        PTs[i_line, 0], PTs[i_line, 1], i_line
        if lsp[-1][-1] == '\n':
            names.append(lsp[-1][:-1])
        else:
            names.append(lsp[-1])

    # Sort the array by temperature
    Tsortind = np.argsort(PTind[:,1])
    PTind = PTind[Tsortind, :]
    #print(PTind)
    #input()
    
    # Sort the array entries with constant
    # temperatures by pressure
    diffPs = 0
    T_start = PTind[0, 1]
    for i in range(n_entries):
        if np.abs(PTind[i, 1]-T_start) > 1e-10:
            break
        diffPs = diffPs+1
    #print(diffPs)

    diffTs = int(n_entries / diffPs)
    for i_dT in range(diffTs):
        subsort = PTind[i_dT*diffPs:(i_dT+1)*diffPs, :]
        Psortind = np.argsort(subsort[:,0])
        subsort = subsort[Psortind, :]
        PTind[i_dT*diffPs:(i_dT+1)*diffPs, :] = subsort
        #print(subsort)
        #input()

    names_sorted = []
    for i_line in range(n_entries):
        names_sorted.append(names[int(PTind[i_line, 2]+0.01)])

    # Convert from bars to cgs
    PTind[:,0] = PTind[:,0]*1e6

    return [PTind[:,:-1][:,::-1], names_sorted, diffTs, diffPs]

# Check if custom grid exists, if yes return sorted P-T array with
# corresponding sorted path names, retutn None otherwise.

def get_custom_PT_grid(path, mode, species):

    import os as os

    path_test = path+'/opacities/lines/'
    if mode == 'lbl':
        path_test = path_test + 'line_by_line/'
    elif mode == 'c-k':
        path_test = path_test + 'corr_k/'
    path_test = path_test + species + '/PTpaths.ls'
    if not os.path.isfile(path_test):
        return None
    else:
        return sort_opa_PTgrid(path_test)

'''
import pylab as plt

demo_temp = 2880
demo_press = 0.33*1e6
ab_em = 1e-6*5.485799e-4/2.33
ab_h = 0.33*1./2.33
ab_hm = 2e-9*1./2.33
mww_val = 2.33

abunds = {}
abunds['e-'] = np.array([ab_em,ab_em,ab_em])
abunds['H'] = np.array([ab_h,ab_h,ab_h])
abunds['H-'] = np.array([ab_hm,ab_hm,ab_hm])

temp = np.array([demo_temp, demo_temp, demo_temp])
press = np.array([demo_press, demo_press, demo_press])
mmw = np.array([mww_val, mww_val, mww_val])

lamb = np.logspace(np.log10(0.9),np.log10(10.),7000)*1e4
lamb_coarse = np.logspace(np.log10(0.9),np.log10(10.),10)*1e4

def calc_borders(x):
        xn = []
        xn.append(x[0]-(x[1]-x[0])/2.)
        for i in range(int(len(x))-1):
            xn.append(x[i]+(x[i+1]-x[i])/2.)
        xn.append(x[int(len(x))-1]+(x[int(len(x))-1]-x[int(len(x))-2])/2.)
        return np.array(xn)

lamb_bord = calc_borders(lamb)
print('a')
for i in range(33):
    opa = hminus_opacity(lamb, lamb_bord, temp, press, mmw, abunds)
print('b')

plt.plot(lamb/1e4, opa*2.33*nc.amu)

lamb_coarse_bord = calc_borders(lamb_coarse)
opa = hminus_opacity(lamb_coarse, lamb_coarse_bord, temp, press, mmw, abunds)
plt.plot(lamb_coarse/1e4, opa*2.33*nc.amu)

plt.ylim([1e-28,1e-22])
plt.xscale('log')
plt.yscale('log')
plt.show()
'''
