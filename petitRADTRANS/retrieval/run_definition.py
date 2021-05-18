import sys, os
# To not have numpy start parallelizing on its own
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np

from petitRADTRANS import Radtrans
from petitRADTRANS import nat_cst as nc

from .retrieval_config import RetrievalConfig

# Import Priors
from .util import gaussian_prior

# Import atmospheric model function
# You could also write your own!
from .models import emission_model_diseq

##########################
# Define the pRT run setup
##########################
RunDefinition = RetrievalConfig(retrieval_name = "HR8799b_v3",
                                run_mode = "retrieval",
                                AMR = True,
                                plotting = False,
                                scattering = True,
                                write_out_spec_sample=True)
##################
# Read in Data
##################
RunDefinition.add_data('GRAVITY',
                       'observations/HR8799b_Spectra.fits',
                       model_generating_function = emission_model_diseq,
                       data_resolution = 500,
                       model_resolution = 500)
#RunDefinition.data['GRAVITY'].wlen_range_pRT = [1.4,2.6]

"""RunDefinition.add_data('GPI',
                       'observations/hr8799c_gpi_2018.txt',
                       model_generating_function = emission_model_diseq, 
                       scale = True,
                       external_pRT_reference='GRAVITY')
RunDefinition.add_data('OSIRIS',
                       'observations/HR8799c_OSIRIS_full.dat',
                       model_generating_function = emission_model_diseq, 
                       scale = True,
                       external_pRT_reference='GRAVITY')"""
#RunDefinition.add_data('OSIRIS2010',
#                       'observations/hr8799b_osiris_2010.txt',
#                       model_generating_function = emission_model_diseq, 
#                       scale = True,
#                       external_pRT_reference='GRAVITY')
RunDefinition.add_data('OSIRIS2011',
                       'observations/hr8799b_osiris_hk_2011.txt',
                       model_generating_function = emission_model_diseq, 
                       scale = False,
                       data_resolution = 150,
                       model_resolution = 150,
                       external_pRT_reference=None)
RunDefinition.data['GRAVITY'].flux = RunDefinition.data['GRAVITY'].flux *(39.4/10.)**2
RunDefinition.data['GRAVITY'].covariance *= (39.4/10.0)**4
RunDefinition.data['GRAVITY'].inv_cov = np.linalg.inv(RunDefinition.data['GRAVITY'].covariance)
RunDefinition.data['GRAVITY'].flux_error = np.sqrt(RunDefinition.data['GRAVITY'].covariance.diagonal())# Convert mJy to W/m^2/micron

RunDefinition.data['OSIRIS2011'].flux = 3e-12*RunDefinition.data['OSIRIS2011'].flux*1e-3/(RunDefinition.data['OSIRIS2011'].wlen)**2 
RunDefinition.data['OSIRIS2011'].flux_error = 3e-12*RunDefinition.data['OSIRIS2011'].flux_error*1e-3/(RunDefinition.data['OSIRIS2011'].wlen)**2 

#OSIRIS errors are relative
#RunDefinition.data['OSIRIS'].flux_error = RunDefinition.data['OSIRIS'].flux_error * RunDefinition.data['OSIRIS'].flux
#################################################
# Add parameters, and priors for free parameters. 
#################################################
RunDefinition.add_parameter(name = 'D_pl', free = False, value = 10*nc.pc)

# This run uses the model of Molliere (2020) for HR8799e
# Check out models.py for a description of the parameters.

# Orig
#RunDefinition.add_parameter('log_g',True, 
#                            transform_prior_cube_coordinate = \
#                            lambda x : 1.5+4.5*x)
RunDefinition.add_parameter('log_g',True, 
                            transform_prior_cube_coordinate = \
                            lambda x : 2.0+3.5*x)
# Orig
#RunDefinition.add_parameter('R_pl', True, \
#                            transform_prior_cube_coordinate = \
#                            lambda x : ( 0.6+1.4*x)*nc.r_jup_mean)
RunDefinition.add_parameter('R_pl', True, \
                            transform_prior_cube_coordinate = \
                            lambda x : ( 0.7+1.3*x)*nc.r_jup_mean)
# Orig run
#RunDefinition.add_parameter('T_int', True, \
#                            transform_prior_cube_coordinate = \
#                            lambda x : 300.+3000.*x)
RunDefinition.add_parameter('T_int', True, \
                            transform_prior_cube_coordinate = \
                            lambda x : 300.+3000.*x)
RunDefinition.add_parameter('T3', True, \
                            transform_prior_cube_coordinate = \
                            lambda x : x)
RunDefinition.add_parameter('T2', True, \
                            transform_prior_cube_coordinate = \
                            lambda x : x)
RunDefinition.add_parameter('T1', True, \
                            transform_prior_cube_coordinate = \
                            lambda x : x)
RunDefinition.add_parameter('alpha', True, \
                            transform_prior_cube_coordinate = \
                            lambda x :1.0+x)
RunDefinition.add_parameter('log_delta', True, \
                            transform_prior_cube_coordinate = \
                            lambda x : x) 
RunDefinition.add_parameter('sigma_lnorm', True,
                            transform_prior_cube_coordinate = \
                            lambda x : 1.05 + 1.95*x) 
RunDefinition.add_parameter('log_pquench',True,\
                            transform_prior_cube_coordinate = \
                            lambda x : -6.0+9.0*x)
# Orig
#RunDefinition.add_parameter('Fe/H',True,\
#                            transform_prior_cube_coordinate = \
#                            lambda x : -1.5+3.0*x)
RunDefinition.add_parameter('Fe/H',True,\
                            transform_prior_cube_coordinate = \
                            lambda x : -1.5+2.0*x)
RunDefinition.add_parameter('C/O',True,\
                            transform_prior_cube_coordinate = \
                            lambda x : 0.1+1.5*x)
RunDefinition.add_parameter('log_kzz',True,\
                            transform_prior_cube_coordinate = \
                            lambda x : 5.0+8.*x) 
# Orig run
#RunDefinition.add_parameter('fsed',True,\
#                            transform_prior_cube_coordinate = \
#                            lambda x : 1.0 + 10.0*x)
RunDefinition.add_parameter('fsed',True,\
                            transform_prior_cube_coordinate = \
                            lambda x : 0.0 + 10.0*x)
# Should we be doing a linear transform rather than a multiplicative scaling (a + b*x?)
# Scaling factor name must be the name of the Data class for
# the instrument, + "_scale_factor"                            
#RunDefinition.add_parameter('GPI_scale_factor',True,\
#                            transform_prior_cube_coordinate = \
#                            lambda x : 0.8 + 0.4 *x)

#Osiris spectrum isn't flux calibrated.
#RunDefinition.add_parameter('OSIRIS',True,\
#                            transform_prior_cube_coordinate = \
#                            lambda x : 1e-19 + 1e-13 *x)
# Orig
#RunDefinition.add_parameter('OSIRIS2010_scale_factor',True,\
#                            transform_prior_cube_coordinate = \
#                            lambda x : 10**(-19.+9.*x))
RunDefinition.add_parameter('OSIRIS2010_scale_factor',True,\
                            transform_prior_cube_coordinate = \
                            lambda x : 10**(-18.+4.*x))

#######################################################
# Define species to be included as absorbers
#######################################################
RunDefinition.set_rayleigh_species(['H2', 'He'])
RunDefinition.set_continuum_opacities(['H2-H2', 'H2-He'])
RunDefinition.set_line_species(['CH4', 'H2O', 'CO2', 'HCN', 'CO_all_iso', 'H2', 'H2S', 'NH3', 'PH3', 'Na', 'K'])
# Origin run
#RunDefinition.add_cloud_species("Fe(c)_cd",eq = True,abund_lim = (-3.5,4.5))
#RunDefinition.add_cloud_species("MgSiO3(c)_cd",eq = True,abund_lim = (-3.5,4.5))
RunDefinition.add_cloud_species("Fe(c)_cd",eq = True,abund_lim = (-2.5,4.5))
RunDefinition.add_cloud_species("MgSiO3(c)_cd",eq = True,abund_lim = (-2.5,4.5))


##################################################################
# Define what to put into corner plot if run_mode == 'evaluate'
##################################################################
RunDefinition.parameters['R_pl'].plot_in_corner = True
RunDefinition.parameters['R_pl'].corner_label = r'$R_{\rm P}$ ($\rm R_{Jup}$)'
RunDefinition.parameters['R_pl'].corner_transform = lambda x : x/nc.r_jup_mean
RunDefinition.parameters['log_g'].plot_in_corner = True
RunDefinition.parameters['log_g'].corner_ranges = [2., 5.]
RunDefinition.parameters['log_g'].corner_label = "log g"
RunDefinition.parameters['fsed'].plot_in_corner = True
RunDefinition.parameters['log_kzz'].plot_in_corner = True
RunDefinition.parameters['log_kzz'].corner_label = "log Kzz"
RunDefinition.parameters['C/O'].plot_in_corner = True
RunDefinition.parameters['Fe/H'].plot_in_corner = True
RunDefinition.parameters['log_pquench'].plot_in_corner = True
RunDefinition.parameters['log_pquench'].corner_label = "log pquench"

"""for spec in RunDefinition.line_species:
    if 'all_iso' in spec:
        RunDefinition.parameters[spec].corner_label = 'CO'
    RunDefinition.parameters[spec].plot_in_corner = True
    RunDefinition.parameters[spec].corner_transform = lambda x : np.log10(x)
    RunDefinition.parameters[spec].corner_ranges = [-7,-1.0]
"""
for spec in RunDefinition.cloud_species:
    cname = spec.split('_')[0]
    RunDefinition.parameters['log_X_cb_'+cname].plot_in_corner = True
    RunDefinition.parameters['log_X_cb_'+cname].corner_label = cname

##################################################################
# Define axis properties of spectral plot if run_mode == 'evaluate'
##################################################################
RunDefinition.plot_kwargs["spec_xlabel"] = 'Wavelength (micron)'

RunDefinition.plot_kwargs["spec_ylabel"] = r'$(R_{\rm P}/R_*)^2$ (ppm)'
RunDefinition.plot_kwargs["y_axis_scaling"] = 1.0
RunDefinition.plot_kwargs["xscale"] = 'log'
RunDefinition.plot_kwargs["yscale"] = 'linear'
RunDefinition.plot_kwargs["nsample"] = 100
RunDefinition.plot_kwargs["resolution"] = 1500.

##################################################################
# Define from which observation object to take P-T
# in evaluation mode (if run_mode == 'evaluate'),
# add PT-envelope plotting options
##################################################################
RunDefinition.plot_kwargs["take_PTs_from"] = 'GRAVITY'
RunDefinition.plot_kwargs["temp_limits"] = [150, 3000]
RunDefinition.plot_kwargs["press_limits"] = [1e2, 1e-5]
