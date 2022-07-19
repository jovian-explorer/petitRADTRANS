#conda activate petitradtrans
#python3 retran.py


# Let's start by importing everything we need.
import os
# To not have numpy start parallelizing on its own
os.environ["OMP_NUM_THREADS"] = "8"
import numpy as np
import matplotlib.pyplot as plt

from petitRADTRANS import Radtrans
from petitRADTRANS import nat_cst as nc

# Import the class used to set up the retrieval.
from petitRADTRANS.retrieval import Retrieval,RetrievalConfig
# Import Prior functions, if necessary.
from petitRADTRANS.retrieval.util import gaussian_prior
# Import atmospheric model function
# You could also write your own!
from petitRADTRANS.retrieval.models import emission_model_diseq


# Lets start out by setting up a simple run definition
# We'll add the data AFTER we define the model function below
# Full details of the parameters can be found in retrieval_config.py

# Since our retrieval has already ran before, we'll set the mode to 'evaluate' so we can make some plots.
RunDefinitionSimple = RetrievalConfig(retrieval_name = "hst_example_clear_spec",
                                      run_mode = "retrieval", # This must be 'retrieval' to run PyMultiNest
                                      AMR = False, # We won't be using adaptive mesh refinement for the pressure grid
                                      scattering = False) # This would turn on scattering when calculating emission spectra.
                                                          # Scattering is automatically included for transmission spectra.

# Let's start with the parameters that will remain fixed during the retrieval
RunDefinitionSimple.add_parameter('Rstar', # name
                                  False,   # is_free_parameter, So Rstar is not retrieved here.
                                  value = 0.651*nc.r_sun)

# Log of the surface gravity in cgs units.
# The transform_prior_cube argument transforms a uniform random sample x, drawn between 0 and 1.
# to the physical units we're using in pRT.
RunDefinitionSimple.add_parameter('log_g',
                                  True, # is_free_parameter: we are retrieving log(g) here
                                  transform_prior_cube_coordinate = \
                                              lambda x : 2.+3.5*x) # This means that log g
                                                                   # can vary between 2 and 5.5
                                  # Note that logg is usually not a free parameter in retrievals of
                                  # transmission spectra, at least it is much better constrained as being
                                  # assumed here, as the planetary mass and radius are usually known.

# Planet radius in cm
RunDefinitionSimple.add_parameter('R_pl', True, \
                            transform_prior_cube_coordinate = \
                            lambda x : ( 0.2+0.2*x)*nc.r_jup_mean) # Radius varies between 0.2 and 0.4

# Interior temperature in Kelvin
RunDefinitionSimple.add_parameter('Temperature', True, \
                            transform_prior_cube_coordinate = \
                            lambda x : 300.+2000.*x)

# Let's include a grey cloud as well to see what happens, even though we assumed a clear atmosphere.
RunDefinitionSimple.add_parameter("log_Pcloud", True, \
                            transform_prior_cube_coordinate = \
                            lambda x : -6.0+8.0*x) # The atmosphere can thus have an opaque cloud
                                                   # between 1e-6 and 100 bar
                                                   
                                                   
RunDefinitionSimple.list_available_line_species()
RunDefinitionSimple.set_rayleigh_species(['H2', 'He'])
RunDefinitionSimple.set_continuum_opacities(['H2-H2', 'H2-He'])

# Here we setup the line species for a free retrieval, setting the prior bounds with the abund_lim parameter
# The retrieved value is the log mass fraction.
# RunDefinitionSimple.set_line_species(["CH4", "H2O", "CO2", "CO_all_iso"], free=True, abund_lim = (-6.0,6.0))

# Let's use the most up-to-date line lists
RunDefinitionSimple.set_line_species(["CH4", "H2O_Exomol", "CO2", "CO_all_iso_HITEMP"],
                                     eq=False, abund_lim = (-6.0,6.0))
                                     
                                     
# Now we can define the atmospheric model we want to use
def retrieval_model_spec_iso(pRT_object, \
                             parameters, \
                             PT_plot_mode = False,
                             AMR = False):
    """
    retrieval_model_eq_transmission
    This model computes a transmission spectrum based on free retrieval chemistry
    and an isothermal temperature-pressure profile.

    parameters
    -----------
    pRT_object : object
        An instance of the pRT class, with optical properties as defined in the RunDefinition.
    parameters : dict
        Dictionary of required parameters:
            Rstar : Radius of the host star [cm]
            log_g : Log of surface gravity
            R_pl : planet radius [cm]
            Temperature : Isothermal temperature [K]
            species : Log abundances for each species used in the retrieval
            log_Pcloud : optional, cloud base pressure of a grey cloud deck.
    PT_plot_mode : bool
        Return only the pressure-temperature profile for plotting. Evaluate mode only.
    AMR :
        Adaptive mesh refinement. Use the high resolution pressure grid around the cloud base.

    returns
    -------
    wlen_model : np.array
        Wavlength array of computed model, not binned to data [um]
    spectrum_model : np.array
        Computed transmission spectrum R_pl**2/Rstar**2
    """
    # Make the P-T profile
    pressures = pRT_object.press/1e6 # Convert from bar to cgs; internally pRT uses cgs

    # Note how we access the values of the Parameter class objects
    temperatures = parameters['Temperature'].value * np.ones_like(pressures)

    # If in evaluation mode, and PTs are supposed to be plotted
    if PT_plot_mode:
        return pressures, temperatures

    # Make the abundance profiles
    abundances = {}
    msum = 0.0 # Check that the total massfraction of all species is <1
    for species in pRT_object.line_species:
        spec = species.split('_R_')[0] # Dealing with the naming scheme
                                       # for binned down opacities (see below).
        abundances[species] = 10**parameters[spec].value * np.ones_like(pressures)
        msum += 10**parameters[spec].value
    #if msum > 1.0:
    #    return None, None
    abundances['H2'] = 0.766 * (1.0-msum) * np.ones_like(pressures)
    abundances['He'] = 0.234 * (1.0-msum) * np.ones_like(pressures)

    # Find the mean molecular weight in each layer
    from petitRADTRANS.retrieval.util import calc_MMW
    MMW = calc_MMW(abundances)

    # Calculate the spectrum
    pRT_object.calc_transm(temperatures, \
                           abundances, \
                           10**parameters['log_g'].value, \
                           MMW, \
                           R_pl=parameters['R_pl'].value, \
                           P0_bar=0.01,
                           Pcloud = 10**parameters['log_Pcloud'].value)

    # Transform the outputs into the units of our data.
    wlen_model = nc.c/pRT_object.freq/1e-4 # wlen in micron
    spectrum_model = (pRT_object.transm_rad/parameters['Rstar'].value)**2.
    return wlen_model, spectrum_model


# Finally, we associate the model with the data, and we can run our retrieval!
import petitRADTRANS # need to get the name for the example data
path_to_data = "/home/keshav/petitRADTRANS/petitRADTRANS/"
path_to_data = petitRADTRANS.__file__.split('__init__.py')[0] # Default location for the example
RunDefinitionSimple.add_data('HST',   # Simulated HST data
                       path_to_data + \
                       'retrieval/examples/transmission/hst_example_clear_spec.txt', # Where is the data
                       model_generating_function = retrieval_model_spec_iso, # The atmospheric model to use
                       opacity_mode = 'c-k',
                       data_resolution = 60, # The spectral resolution of the data
                       model_resolution = 120) # The resolution of the c-k tables for the lines

# This model is a noise-free, clear atmosphere transmission spectrum for a sub-neptune type planet
# The model function used to calculate it was slightly different, and used different opacity tables
# than what we'll use in the retrieval, but the results don't significantly change.
# In general, it is also useful to use the data_resolution and model_resolution arguments.
# The data_resolution argument will convolve the generated model spectrum by the instrumental
# resolution prior to calculating the chi squared value.
#
# The model_resolution function uses exo-k to generate low-resolution correlated-k opacity tables
# in order to speed up the radiative transfer calculations.
# We recommend choosing a model_resolution of about 2x the data_resolution.


##################################################################
# Define what to put into corner plot if run_mode == 'evaluate'
##################################################################
RunDefinitionSimple.parameters['R_pl'].plot_in_corner = True
RunDefinitionSimple.parameters['R_pl'].corner_label = r'$R_{\rm P}$ ($\rm R_{Jup}$)'
RunDefinitionSimple.parameters['R_pl'].corner_transform = lambda x : x/nc.r_jup_mean
RunDefinitionSimple.parameters['log_g'].plot_in_corner = True
RunDefinitionSimple.parameters['log_g'].corner_ranges = [2., 5.]
RunDefinitionSimple.parameters['log_g'].corner_label = "log g"
RunDefinitionSimple.parameters['Temperature'].plot_in_corner = True
RunDefinitionSimple.parameters['Temperature'].corner_label ="Temp"
RunDefinitionSimple.parameters['log_Pcloud'].plot_in_corner = True
RunDefinitionSimple.parameters['log_Pcloud'].corner_label =r"log P$_{\rm cloud}$"
RunDefinitionSimple.parameters['log_Pcloud'].corner_ranges = [-6, 2]


# Adding all of the chemical species in our atmosphere to the corner plot
for spec in RunDefinitionSimple.line_species:
    if 'all_iso' in spec:
        # CO is named CO_all_iso, watch out
        RunDefinitionSimple.parameters[spec].corner_label = 'CO'
    RunDefinitionSimple.parameters[spec].plot_in_corner = True
    RunDefinitionSimple.parameters[spec].corner_ranges = [-6.2,0.2]

    # Note, the post_equal_weights file has the actual abundances.
    # If the retrieval is rerun, it will contain the log10(abundances)
    #RunDefinitionSimple.parameters[spec].corner_transform = lambda x : np.log10(x)

##################################################################
# Define axis properties of spectral plot if run_mode == 'evaluate'
##################################################################
RunDefinitionSimple.plot_kwargs["spec_xlabel"] = 'Wavelength [micron]'
RunDefinitionSimple.plot_kwargs["spec_ylabel"] = r'$(R_{\rm P}/R_*)^2$ [ppm]'
RunDefinitionSimple.plot_kwargs["y_axis_scaling"] = 1e6 # so we have units of ppm
RunDefinitionSimple.plot_kwargs["xscale"] = 'linear'
RunDefinitionSimple.plot_kwargs["yscale"] = 'linear'

# Use at least ~100 samples to plot 3 sigma curves
RunDefinitionSimple.plot_kwargs["nsample"] = 10

##################################################################
# Define from which observation object to take P-T
# in evaluation mode (if run_mode == 'evaluate'),
# add PT-envelope plotting options
##################################################################
RunDefinitionSimple.plot_kwargs["take_PTs_from"] = 'HST'
RunDefinitionSimple.plot_kwargs["temp_limits"] = [150, 3000]
RunDefinitionSimple.plot_kwargs["press_limits"] = [1e2, 1e-5]

# If in doubt, define all of the plot_kwargs used here.


# If you want to run the retrieval, you need to choose a different output directory name,
# due to pRT requirements.
#output_dir = path_to_data + 'retrieval_examples/transmission/'
output_dir = ""

retrieval = Retrieval(RunDefinitionSimple,
                      output_dir = output_dir,
                      sample_spec = False,         # Output the spectrum from nsample random samples.
                      pRT_plot_style=True,         # We think that our plots look nice.
                      ultranest=False)             # Let's use pyMultiNest rather than Ultranest

retrieval.run(n_live_points = 400,         # PMN number of live points. 400 is good for small retrievals, 4000 for large
              const_efficiency_mode=False, # Turn PMN const efficiency mode on or off (recommend on for large retrievals)
              resume=True)                # Continue retrieval from where it left off.)
#retrieval.plot_all() # We'll make all the plots individually for now.


# These are dictionaries in case we want to look at multiple retrievals.
# The keys for the dictionaries are the retrieval_names
sample_dict, parameter_dict = retrieval.get_samples()

# Pick the current retrieval to look at.
samples_use = sample_dict[retrieval.retrieval_name]
parameters_read = parameter_dict[retrieval.retrieval_name]

# Plotting the best fit spectrum
# This will generate a few warnings, but it's fine.
fig,ax,ax_r = retrieval.plot_spectra(samples_use,parameters_read)
plt.show()


# Plotting the PT profile
fig,ax =retrieval.plot_PT(sample_dict,parameters_read)
plt.show()


# Corner plot
# The corner plot produces 1,2 and 3 sigma contours for the 2D plots
print(sample_dict['hst_example_clear_spec'].shape)
retrieval.plot_corner(sample_dict,parameter_dict,parameters_read,title_kwargs = {"fontsize" : 10})
plt.show()



