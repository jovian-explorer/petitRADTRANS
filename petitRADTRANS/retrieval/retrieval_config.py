import sys, os
# To not have numpy start parallelizing on its own
os.environ["OMP_NUM_THREADS"] = "1"

from .data_class import Data
from .parameter_class import Parameter
from petitRADTRANS import Radtrans
from petitRADTRANS import nat_cst as nc

import numpy as np

class RetrievalConfig:
    def __init__(self,
                 retrieval_name = "retrieval_name",
                 run_mode = "retrieval",
                 AMR = False,
                 plotting = False,
                 scattering = False,
                 pressures = None,
                 write_out_spec_sample = True):
        """
        parameters
        ----------
        retrieval_name : str
            Name of this retrieval. Make it informative so that you can keep track of the outputs!
        run_mode : str
            Can be either 'retrieval', which runs the retrieval normally using pymultinest,
            or 'evaluate', which produces plots from the best fit parameters stored in the
            output post_equal_weights file.
        AMR : bool
            Use an adaptive high resolution pressure grid around the location of cloud condensation.
        plotting : bool
            Produce plots for each sample to check that the retrieval is running properly. Only
            set to true on a local machine.
        scattering : bool
            If using emission spectra, turn scattering on or off.
                pressures : np.array
            A log-spaced array of pressures over which to retrieve. 100 points is standard, to use AMR
            a higher resolution grid is recommended. If None, defaults to 100 points if AMR is off, and
            1440 points if AMR is on, ranging from 10^-6 to 10^3 bar.
        """
        self.retrieval_name =  retrieval_name
        self.run_mode = run_mode
        self.AMR = AMR
        if pressures is not None:
            self.p_global = pressures
        else: 
            self.p_global = np.logspace(-6,3,100)
        self.plotting = plotting
        self.scattering = scattering 

        self.parameters = {}
        self.data = {}
        self.instruments = []
        self.line_species = []
        self.cloud_species = []
        self.rayleigh_species = ["H2", "He"]
        self.continuum_opacities = ["H2-H2", "H2-He"]
        self.plot_kwargs = {}
        self.plot_defaults()
        self.write_out_spec_sample = write_out_spec_sample

        self.add_parameter("pressure_scaling",False,value = 1)
        self.add_parameter("pressure_width",False,value = 1)
        self.add_parameter("pressure_simple",False,value = self.p_global.shape[0])


    def plot_defaults(self):
        ##################################################################
        # Define axis properties of spectral plot if run_mode == 'evaluate'
        ##################################################################
        self.plot_kwargs["spec_xlabel"] = 'Wavelength [micron]'
        self.plot_kwargs["spec_ylabel"] =  "Flux [W/m2/micron]"
        self.plot_kwargs["y_axis_scaling"] = 1.0
        self.plot_kwargs["xscale"] = 'log'
        self.plot_kwargs["yscale"] = 'linear'
        self.plot_kwargs["resolution"] = 1500.

        ##################################################################
        # Define from which observation object to take P-T
        # in evaluation mode (if run_mode == 'evaluate'),
        # add PT-envelope plotting options
        ##################################################################
        self.plot_kwargs["temp_limits"] = [150, 3000]
        self.plot_kwargs["press_limits"] = [1e2, 1e-5]

    def setup_pres(self, scaling = 10, width = 3):
        # Maybe should read the params from somewhere?
        print("Setting up AMR pressure grid.")
        self.scaling = scaling
        self.width = width
        nclouds = len(self.cloud_species)
        if nclouds == 0:
            print("WARNING: there are no clouds in the retrieval, please add cloud species before setting up AMR")
        new_len = self.p_global.shape[0]  + nclouds*width*(scaling-1)
        self.amr_pressure = np.logspace(np.log10(self.p_global[0]),np.log10(self.p_global[-1]),new_len)
        self.add_parameter("pressure_scaling",False,value = scaling)
        self.add_parameter("pressure_width",False,value = width)
        self.add_parameter("pressure_simple",False,value = self.p_global.shape[0])

        return self.amr_pressure
    def add_parameter(self,name,free, value = None, transform_prior_cube_coordinate  = None):
        self.parameters[name] = Parameter(name, free, value, 
                                          transform_prior_cube_coordinate = transform_prior_cube_coordinate)
    def set_line_species(self,linelist,free=False,abund_lim=(-6.0,6.0)):
        self.line_species = linelist
        if free:
            for spec in self.line_species:
                self.parameters[spec] = Parameter(spec,True,\
                                    transform_prior_cube_coordinate = \
                                    lambda x : abund_lim[0]+abund_lim[1]*x)
    def set_rayleigh_species(self,linelist):
        self.rayleigh_species = linelist
    def set_continuum_opacities(self,linelist):
        self.continuum_opacities = linelist
    def add_line_species(self,species,free=False,abund_lim=(-8.0,7.0)):
        # parameter passed through loglike is log10 abundance
        self.line_species.append(species)
        if free:
            self.parameters[species] = Parameter(species,True,\
                                    transform_prior_cube_coordinate = \
                                    lambda x : abund_lim[0] + abund_lim[1]*x)
    def remove_species_lines(self,species,free=False):
        if species in self.line_species:
            self.line_species.remove(species)
        if free:
            self.parameters.pop(species,None)

    def add_cloud_species(self,species,eq = True, abund_lim = (-3.5,4.5), PBase_lim = (-5.0,7.0)):
        """
        add_cloud_species
        This function adds a single cloud species to the list of species. Optionally,
        it will add parameters to allow for a retrieval using an ackermann-marley model. 
        If an equilibrium condensation model is used in th retrieval model function (eq=True),
        then a parameter is added that scales the equilibrium cloud abundance, as in Molliere (2020).
        If eq is false, two parameters are added, the cloud abundnace and the cloud base pressure.
        The limits set the prior ranges, both on a log scale.

        parameters
        ----------
        species : str
            Name of the pRT cloud species, including the cloud shape tag.
        eq : bool
            Does the retrieval model use an equilibrium cloud model. This restricts the available species!
        abund_lim : tuple(float,float)
            If eq is True, this sets the scaling factor for the equilibrium condensate abundance, typical
            range would be (-3,0). If eq is false, this sets the the range on the actual cloud abundance,
            with a typical range being (-5,7). Note that the upper limit is set from abund_lim[0] + abund_lim[1].
        PBase_lim : tuple(float,float)
            Only used if not using an equilibrium model. Sets the limits on teh log of the cloud base pressure
        """
        if species.endswith("(c)"):
            print("Ensure you set the cloud particle shape, typically with the _cd tag!")
            print(species + " was not added to the list of cloud species")
            return
        self.cloud_species.append(species)
        cname = species.split('_')[0]
        self.parameters['log_X_cb_'+cname] = Parameter('log_X_cb_'+cname,True,\
                                       transform_prior_cube_coordinate = \
                                       lambda x : abund_lim[0] + abund_lim[1]*x)
        #self.parameters['Pbase_'+cname] = Parameter('Pbase_'+cname,True,\
        #                     transform_prior_cube_coordinate = \
        #                     lambda x : PBase_lim[0] + PBase_lim[1]*x)
        
    def add_data(self, name, path,
                 model_generating_function,
                 data_resolution = None,
                 model_resolution = None,
                 scale = False,
                 external_pRT_reference = None):
        """
        add_data
        Create a Data class object.

        parameters
        -----------
        name : str
            Identifier for this data set.
        path_to_observations : str
            Path to observations file, including filename. This can be a txt or dat file containing the wavelength,
            flux, transit depth and error, or a fits file containing the wavelength, spectrum and covariance matrix.
        data_resolution : float
            Spectral resolution of the instrument. Optional, allows convolution of model to instrumental line width.
        model_generating_function : fnc
            A function, typically defined in run_definition.py that returns the model wavelength and spectrum (emission or transmission).
            This is the function that contains the physics of the model, and calls pRT in order to compute the spectrum.
        scale : bool
            Turn on or off scaling the data by a constant factor.
        """
        self.data[name] = Data(name, path, 
                                model_generating_function = model_generating_function,
                                data_resolution = data_resolution,
                                model_resolution = model_resolution,
                                scale = scale,
                                external_pRT_reference=external_pRT_reference)
