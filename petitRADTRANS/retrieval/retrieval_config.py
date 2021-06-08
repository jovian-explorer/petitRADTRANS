import sys, os
# To not have numpy start parallelizing on its own
os.environ["OMP_NUM_THREADS"] = "1"

from .data import Data
from .parameter import Parameter
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
                 write_out_spec_sample = False):
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
            This will increase the size of the pressure grid by a constant factor that can be adjusted
            in the setup_pres function.
        plotting : bool
            Produce plots for each sample to check that the retrieval is running properly. Only
            set to true on a local machine.
        scattering : bool
            If using emission spectra, turn scattering on or off.
        pressures : numpy.array
            A log-spaced array of pressures over which to retrieve. 100 points is standard, between 
            10^-6 and 10^3.
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
        self.rayleigh_species = []
        self.continuum_opacities = []
        self.plot_kwargs = {}

        self._plot_defaults()
        self.write_out_spec_sample = write_out_spec_sample

        self.add_parameter("pressure_scaling",False,value = 1)
        self.add_parameter("pressure_width",False,value = 1)
        self.add_parameter("pressure_simple",False,value = self.p_global.shape[0])


    def _plot_defaults(self):
        ##################################################################
        # Define axis properties of spectral plot if run_mode == 'evaluate'
        ##################################################################
        self.plot_kwargs["spec_xlabel"] = 'Wavelength [micron]'
        self.plot_kwargs["spec_ylabel"] =  "Flux [W/m2/micron]"
        self.plot_kwargs["y_axis_scaling"] = 1.0
        self.plot_kwargs["xscale"] = 'log'
        self.plot_kwargs["yscale"] = 'linear'
        self.plot_kwargs["resolution"] = 1500.
        self.plot_kwargs["nsample"] = 10.

        ##################################################################
        # Define from which observation object to take P-T
        # in evaluation mode (if run_mode == 'evaluate'),
        # add PT-envelope plotting options
        ##################################################################
        self.plot_kwargs["temp_limits"] = [150, 3000]
        self.plot_kwargs["press_limits"] = [1e2, 1e-5]

    def _setup_pres(self, scaling = 10, width = 3):
        """
        setup_pres
        This converts the standard pressure grid into the correct length
        for the AMR pressure grid. The scaling adjusts the resolution of the
        high resolution grid, while the width determines the size of the high 
        pressure region. This function is automatically called in
        Retrieval.setupData().

        parameters
        ----------
        scaling : int
            A multiplicative factor that determines the size of the full high resolution pressure grid,
            which will have length self.p_global.shape[0] * scaling.
        width : int
            The number of cells in the low pressure grid to replace with the high resolution grid.
        """
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
        """
        add_parameter
        This function adds a Parameter (see parameter.py) to the dictionary of parameters. A Parameter
        has a name and a boolean parameter to set whether it is a free or fixed parameter during the retrieval.
        In addition, a value can be set, or a prior function can be given that transforms a random variable in
        [0,1] to the physical dimensions of the Parameter.

        parameters
        ----------
        name : str
            The name of the parameter. Must match the name used in the model function for the retrieval.
        free : bool
            True if the parameter is a free parameter in the retrieval, false if it is fixed.
        value : float
            The value of the parameter in the units used by the model function.
        transform_prior_cube_coordinate : method
            A function that transforms the unit interval to the physical units of the parameter. 
            Typically given as a lambda function.
        """
        self.parameters[name] = Parameter(name, free, value, 
                                          transform_prior_cube_coordinate = transform_prior_cube_coordinate)

    def set_line_species(self,linelist,free=False,abund_lim=(-6.0,6.0)):
        """
        set_line_species
        This function adds a list of species to the pRT object that will define the line opacities of the model.
        The values in the list are strings, with the names matching the pRT opacity names, which vary between 
        the c-k line opacities and the line-by-line opacities.

        parameters
        ----------
        linelist : List(str)
            The list of species to include in the retrieval
        free : bool
            If true, the retrieval should use free chemistry, and Parameters for the abundance of each
            species in the linelist will be added to the retrieval
        abund_lim : Tuple(float,float)
            If free is True, this sets the boundaries of the uniform prior that will be applied for 
            each species in linelist. The range of the prior goes from abund_lim[0] to abund_lim[0] + abund_lim[1].
            The abundance limits must be given in log10 units of the mass fraction.
        """
        self.line_species = linelist
        if free:
            for spec in self.line_species:
                self.parameters[spec] = Parameter(spec,True,\
                                    transform_prior_cube_coordinate = \
                                    lambda x : abund_lim[0]+abund_lim[1]*x)
    def set_rayleigh_species(self,linelist):
        """
        set_rayleigh_species
        Set the list of species that contribute to the rayleigh scattering in the pRT object.

        parameters
        ----------
        linelist : List(str)
            A list of species that contribute to the rayleigh opacity.
        """
        self.rayleigh_species = linelist

    def set_continuum_opacities(self,linelist):
        """
        set_continuum_opacities
        Set the list of species that contribute to the continuum opacity in the pRT object.

        parameters
        ----------
        linelist : List(str)
            A list of species that contribute to the continuum opacity.
        """
        self.continuum_opacities = linelist

    def add_line_species(self,species,free=False,abund_lim=(-8.0,7.0)):
        """
        set_line_species
        This function adds a single species to the pRT object that will define the line opacities of the model.
        The name must match the pRT opacity name, which vary between the c-k line opacities and the line-by-line opacities.

        parameters
        ----------
        species : str
            The species to include in the retrieval
        free : bool
            If true, the retrieval should use free chemistry, and Parameters for the abundance of the
            species will be added to the retrieval
        abund_lim : Tuple(float,float)
            If free is True, this sets the boundaries of the uniform prior that will be applied the species given. 
            The range of the prior goes from abund_lim[0] to abund_lim[0] + abund_lim[1].
            The abundance limits must be given in log10 units of the mass fraction.
        """
        # parameter passed through loglike is log10 abundance
        self.line_species.append(species)
        if free:
            self.parameters[species] = Parameter(species,True,\
                                    transform_prior_cube_coordinate = \
                                    lambda x : abund_lim[0] + abund_lim[1]*x)

    def remove_species_lines(self,species,free=False):
        """
        remove_species_lines
        This function removes a species from the pRT line list, and if using a free chemistry retrieval,
        removes the associated Parameter of the species.

        parameters
        ----------
        species : str
            The species to remove from the retrieval
        free : bool
            If true, the retrieval should use free chemistry, and Parameters for the abundance of the
            species will be removed to the retrieval    
        """
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
            Only used if not using an equilibrium model. Sets the limits on teh log of the cloud base pressure.
            Obsolete.
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
                 distance = None,
                 scale = False,
                 wlen_range_micron = None,
                 external_pRT_reference = None):
        """
        add_data
        Create a Data class object.

        parameters
        -----------
        name : str
            Identifier for this data set.
        path : str
            Path to observations file, including filename. This can be a txt or dat file containing the wavelength,
            flux, transit depth and error, or a fits file containing the wavelength, spectrum and covariance matrix.
        model_generating_function : fnc
            A function, typically defined in run_definition.py that returns the model wavelength and spectrum (emission or transmission).
            This is the function that contains the physics of the model, and calls pRT in order to compute the spectrum.
        data_resolution : float
            Spectral resolution of the instrument. Optional, allows convolution of model to instrumental line width.
        model_resolution : float
            Spectral resolution of the model, allowing for low resolution correlated k tables from exo-k.
        distance : float
            The distance to the object in cgs units. Defaults to a 10pc normalized distance. All data must be scaled to the 
            same distance before running the retrieval, which can be done using the scale_to_distance method in the Data class.
        scale : bool
            Turn on or off scaling the data by a constant factor.
        wlen_range_micron : Tuple
            A pair of wavelenths in units of micron that determine the lower and upper boundaries of the model computation.
        external_pRT_reference : str
            The name of an existing Data object. This object's pRT_object will be used to calculate the chi squared
            of the new Data object. This is useful when two datasets overlap, as only one model computation is required
            to compute the log likelihood of both datasets.
        """
        self.data[name] = Data(name, path, 
                                model_generating_function = model_generating_function,
                                data_resolution = data_resolution,
                                model_resolution = model_resolution,
                                distance = distance,
                                scale = scale,
                                wlen_range_micron = wlen_range_micron,
                                external_pRT_reference=external_pRT_reference)
        return
    def add_photometry(self, path, 
                       model_resolution = 10, 
                       distance = None,
                       scale = False, 
                       wlen_range_micron = None,
                       transform_func = None,
                       external_pRT_reference = None):
        """
        add_photometry
        Create a Data class object for each photometric point in a photometry file.
        The photometry file must be a csv file and have the following structure:
        name, lower wavelength bound [um], upper wavelength boundary[um], flux [W/m2/micron], flux error [W/m2/micron]

        Photometric data requires a transformation function to conver a spectrum into synthetic photometry.
        You must provide this function yourself, or have the species package installed.
        If using species, the name in the data file must be of the format instrument/filter.

        parameters
        -----------
        name : str
            Identifier for this data set.
        path : str
            Path to observations file, including filename.
        model_resolution : float
            Spectral resolution of the model, allowing for low resolution correlated k tables from exo-k.
        scale : bool
            Turn on or off scaling the data by a constant factor. Currently only set up to scale all photometric data
            in a given file.
        distance : float
            The distance to the object in cgs units. Defaults to a 10pc normalized distance. All data must be scaled to the 
            same distance before running the retrieval, which can be done using the scale_to_distance method in the Data class.
        wlen_range_micron : Tuple
            A pair of wavelenths in units of micron that determine the lower and upper boundaries of the model computation.
        external_pRT_reference : str
            The name of an existing Data object. This object's pRT_object will be used to calculate the chi squared
            of the new Data object. This is useful when two datasets overlap, as only one model computation is required
            to compute the log likelihood of both datasets.
        photometric_transfomation_function : method
            A function that will transform a spectrum into an average synthetic photometric point, typicall accounting for 
            filter transmission.
        """
        photometry = open(path)
        if transform_func is None:
            try:
                import species
                species.SpeciesInit()
            except:
                print("Please provide a function to transform a spectrum into photometry, or pip install species")
        for line in photometry:
            # # must be the comment character
            if line[0] == '#':
                continue
            vals = line.split(',')
            name = vals[0]
            wlow = float(vals[1])
            whigh = float(vals[2])
            flux = float(vals[3])
            err = float(vals[4])
            if transform_func is None:
                transform = species.SyntheticPhotometry(name).spectrum_to_flux
            else:
                transform = transform_func
            
            if wlen_range_micron is None:
                wbins = [0.95*wlow,1.05*whigh]
            else:
                wbins = wlen_range_micron
            print("Adding " +name)
            self.data[name] = Data(name, 
                                    path,    
                                    distance = distance,
                                    photometry = True,
                                    wlen_range_micron = wbins,
                                    photometric_bin_edges = [wlow,whigh],
                                    data_resolution = np.mean([wlow,whigh])/(whigh-wlow),
                                    model_resolution = model_resolution,
                                    scale = scale,
                                    photometric_transfomation_function = transform,
                                    external_pRT_reference=external_pRT_reference)
            self.data[name].flux = flux
            self.data[name].flux_error = err
        return
