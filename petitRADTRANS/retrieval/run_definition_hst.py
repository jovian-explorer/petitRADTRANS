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
from .models import isothermal_free_transmission

##########################
# Define the pRT run setup
##########################
RunDefinition = RetrievalConfig(retrieval_name = "hst_clear_spec",
                                run_mode = "retrieval",
                                AMR = False,
                                plotting = False,
                                scattering = False)
##################
# Read in Data
##################
RunDefinition.add_data('HST',
                       '../retrieval_examples/transmission/hst_example_clear_spec.txt',
                       model_generating_function = isothermal_free_transmission)
#################################################
# Add parameters, and priors for free parameters. 
#################################################
RunDefinition.add_parameter(name = 'D_pl', free = False, value = 62.8121*nc.pc)
RunDefinition.add_parameter(name = 'Rstar', free = False, value = 0.651*nc.r_sun)

# This run uses the model of Molliere (2020) for HR8799e
# Check out models.py for a description of the parameters.
RunDefinition.add_parameter('log_g',True, 
                            transform_prior_cube_coordinate = \
                            lambda x : 1.5+3.5*x)
RunDefinition.add_parameter('R_pl', True, \
                            transform_prior_cube_coordinate = \
                            lambda x : ( 0.5+4.0*x)*nc.r_earth)
RunDefinition.add_parameter('Temp', True, \
                            transform_prior_cube_coordinate = \
                            lambda x : 200.+1500.*x)
RunDefinition.add_parameter('log_Pcloud',True,\
                            transform_prior_cube_coordinate = \
                            lambda x : -4.0+6.0*x) 

# Should we be doing a linear transform rather than a multiplicative scaling (a + b*x?)
# Scaling factor name must be the name of the Data class for
# the instrument, + "_scale_factor"                            
#RunDefinition.add_parameter('GPI_scale_factor',True,\
#                            transform_prior_cube_coordinate = \
#                            lambda x : 0.8 + 0.4 *x)

#######################################################
# Define species to be included as absorbers
#######################################################
RunDefinition.set_rayleigh_species(['H2', 'He'])
RunDefinition.set_continuum_opacities(['H2-H2', 'H2-He'])
RunDefinition.set_line_species(["H2O","CH4","CO_all_iso","CO2","NH3","HCN"],
                                free = True, abund_lim=(-6.0,6.0))

##################################################################
# Define what to put into corner plot if run_mode == 'evaluate'
##################################################################
#RunDefinition.parameters['R_pl'].plot_in_corner = True
#RunDefinition.parameters['R_pl'].corner_label = r'$R_{\rm P}$ ($\rm R_{Jup}$)'
#RunDefinition.parameters['R_pl'].corner_transform = lambda x : x/nc.r_earth
#RunDefinition.parameters['log_g'].plot_in_corner = True
#RunDefinition.parameters['log_g'].corner_ranges = [2., 5.]
#RunDefinition.parameters['log_g'].corner_label = "log g"
#RunDefinition.parameters['C/O'].plot_in_corner = True
#RunDefinition.parameters['Fe/H'].plot_in_corner = True


for spec in RunDefinition.line_species:
    if 'all_iso' in spec:
        RunDefinition.parameters[spec].corner_label = 'CO'
    RunDefinition.parameters[spec].plot_in_corner = True
    #RunDefinition.parameters[spec].corner_transform = lambda x : np.log10(x)
    RunDefinition.parameters[spec].corner_ranges = [-6.0,0.0]
#for spec in RunDefinition.cloud_species:
#    cname = spec.split('_')[0]
#    RunDefinition.parameters['log_X_cb_'+cname].plot_in_corner = True
#    RunDefinition.parameters['log_X_cb_'+cname].corner_label = cname

##################################################################
# Define axis properties of spectral plot if run_mode == 'evaluate'
##################################################################
RunDefinition.plot_kwargs["spec_xlabel"] = 'Wavelength (micron)'
RunDefinition.plot_kwargs["spec_ylabel"] = r'$(R_{\rm P}/R_*)^2$ (ppm)'
RunDefinition.plot_kwargs["y_axis_scaling"] = 1e6
RunDefinition.plot_kwargs["xscale"] = "log"
RunDefinition.plot_kwargs["yscale"] = "linear"
RunDefinition.plot_kwargs["resolution"] = 50.
RunDefinition.plot_kwargs["nsample"] = 100.

##################################################################
# Define from which observation object to take P-T
# in evaluation mode (if run_mode == 'evaluate'),
# add PT-envelope plotting options
##################################################################
RunDefinition.plot_kwargs["take_PTs_from"] = 'HST'
RunDefinition.plot_kwargs["temp_limits"] = [100, 1000]
RunDefinition.plot_kwargs["press_limits"] = [1e2, 1e-5]
