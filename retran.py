#conda activate petitradtrans
#python3 retran.py
import os
os.environ["OMP_NUM_THREADS"] = "8"
import numpy as np
import matplotlib.pyplot as plt
from petitRADTRANS import Radtrans
from petitRADTRANS import nat_cst as nc
from petitRADTRANS.retrieval import Retrieval,RetrievalConfig
from petitRADTRANS.retrieval.util import gaussian_prior
from petitRADTRANS.retrieval.models import emission_model_diseq


RunDefinitionSimple = RetrievalConfig(retrieval_name = "hst_example_clear_spec",
                                      run_mode = "evaluate", 
                                      AMR = False,
                                      scattering = False) 
RunDefinitionSimple.add_parameter('Rstar',
                                  False,   
                                  value = 0.651*nc.r_sun)
RunDefinitionSimple.add_parameter('log_g',
                                  True, # is_free_parameter: we are retrieving log(g) here
                                  transform_prior_cube_coordinate = \
                                              lambda x : 2.+3.5*x)
RunDefinitionSimple.add_parameter('R_pl', True, \
                            transform_prior_cube_coordinate = \
                            lambda x : ( 0.2+0.2*x)*nc.r_jup_mean) # Radius varies between 0.2 and 0.4
RunDefinitionSimple.add_parameter('Temperature', True, \
                            transform_prior_cube_coordinate = \
                            lambda x : 300.+2000.*x)
RunDefinitionSimple.add_parameter("log_Pcloud", True, \
                            transform_prior_cube_coordinate = \
                            lambda x : -6.0+8.0*x)
                            
                            
RunDefinitionSimple.list_available_line_species()
RunDefinitionSimple.set_rayleigh_species(['H2', 'He'])
RunDefinitionSimple.set_continuum_opacities(['H2-H2', 'H2-He'])
RunDefinitionSimple.set_line_species(["CH4", "H2O_Exomol", "CO2", "CO_all_iso_HITEMP"],
                                     eq=False, abund_lim = (-6.0,6.0))



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
    pressures = pRT_object.press/1e6 # Convert from bar to cgs; internally pRT uses cgs
    temperatures = parameters['Temperature'].value * np.ones_like(pressures)
    
    if PT_plot_mode:
        return pressures, temperatures
        
    abundances = {}
    msum = 0.0
    
    for species in pRT_object.line_species:
        spec = species.split('_R_')[0] # Dealing with the naming scheme
                                       # for binned down opacities (see below).
        abundances[species] = 10**parameters[spec].value * np.ones_like(pressures)
        msum += 10**parameters[spec].value
        
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

    wlen_model = nc.c/pRT_object.freq/1e-4 # wlen in micron
    spectrum_model = (pRT_object.transm_rad/parameters['Rstar'].value)**2.
    return wlen_model, spectrum_model
    
import petitRADTRANS # need to get the name for the example data
path_to_data = "/home/keshav/petitRADTRANS/petitRADTRANS/"
#path_to_data = petitRADTRANS.__file__.split('__init__.py')[0] # Default location for the example

RunDefinitionSimple.add_data('HST',   # Simulated HST data
                       path_to_data + \
                       'retrieval/examples/transmission/hst_example_clear_spec.txt', # Where is the data
                       model_generating_function = retrieval_model_spec_iso, 
                       opacity_mode = 'c-k',
                       data_resolution = 60, 
                       model_resolution = 120)
                       
                       
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

RunDefinitionSimple.plot_kwargs["spec_xlabel"] = 'Wavelength [micron]'
RunDefinitionSimple.plot_kwargs["spec_ylabel"] = r'$(R_{\rm P}/R_*)^2$ [ppm]'
RunDefinitionSimple.plot_kwargs["y_axis_scaling"] = 1e6 # so we have units of ppm
RunDefinitionSimple.plot_kwargs["xscale"] = 'linear'
RunDefinitionSimple.plot_kwargs["yscale"] = 'linear'
RunDefinitionSimple.plot_kwargs["nsample"] = 10
RunDefinitionSimple.plot_kwargs["take_PTs_from"] = 'HST'
RunDefinitionSimple.plot_kwargs["temp_limits"] = [150, 3000]
RunDefinitionSimple.plot_kwargs["press_limits"] = [1e2, 1e-5]


output_dir = "/home/keshav/petitRADTRANS/petitRADTRANS/outputs"

retrieval = Retrieval(RunDefinitionSimple,
                      output_dir = output_dir,
                      sample_spec = False,         # Output the spectrum from nsample random samples.
                      pRT_plot_style=True,         # We think that our plots look nice.
                      ultranest=False)             # Let's use pyMultiNest rather than Ultranest

retrieval.run(n_live_points = 400,         # PMN number of live points. 400 is good for small retrievals, 4000 for large
              const_efficiency_mode=False, # Turn PMN const efficiency mode on or off (recommend on for large retrievals)
              resume=True)
              
              
              
sample_dict, parameter_dict = retrieval.get_samples()
samples_use = sample_dict[retrieval.retrieval_name]
parameters_read = parameter_dict[retrieval.retrieval_name]
fig,ax,ax_r = retrieval.plot_spectra(samples_use,parameters_read)
plt.show()

fig,ax =retrieval.plot_PT(sample_dict,parameters_read)
plt.show()

print(sample_dict['hst_example_clear_spec'].shape)
retrieval.plot_corner(sample_dict,parameter_dict,parameters_read,title_kwargs = {"fontsize" : 10})
plt.show()


