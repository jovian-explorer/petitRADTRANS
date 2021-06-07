# Input / output, general run definitions
import sys, os
# To not have numpy start parallelizing on its own
os.environ["OMP_NUM_THREADS"] = "1"

# Read external packages
import numpy as np
import copy as cp
import pymultinest
import json
from scipy.stats import binned_statistic

# Plotting
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, AutoMinorLocator, LogLocator, NullFormatter

# Read own packages
from petitRADTRANS import Radtrans
from petitRADTRANS import nat_cst as nc
from .parameter import Parameter
from .data import Data
from .plotting import plot_specs,plot_data,contour_corner
from .rebin_give_width import rebin_give_width as rgw

class Retrieval:
    def __init__(self,
                 run_definition,
                 output_dir = "",
                 test_plotting = False,
                 sample_spec = False,
                 sampling_efficiency = 0.05,\
                 const_efficiency_mode = True, \
                 n_live_points = 4000,
                 resume = True,
                 bayes_factor_species = None,
                 corner_plot_names = None,
                 short_names = None,
                 pRT_plot_style = True,
                 plot_multiple_retrieval_names = None):
        """
        Retrieval Class
        This class implements the retrieval method using petitRADTRANS and pymultinest.
        A RetrievalConfig object is passed to this class to describe the retrieval data, parameters
        and priors. The run() method then uses pymultinest to sample the parameter space, producing
        posterior distributions for parameters and bayesian evidence for models.
        Various useful plotting functions have also been included, and can be run once the retrieval is
        complete.

        Parameters
        ----------
        run_definition : RetrievalConfig
            A RetrievalConfig object that describes the retrieval to be run. This is the user facing class
            that must be setup for every retrieval.
        output_dir : Str
            The directory in which the output folders should be written
        test_plotting : Bool
            Only use when running locally. A boolean flag that will produce plots for each sample when pymultinest is run.
        sample_spec : Bool
            Produce plots and data files for 100 randomly sampled outputs from pymultinest.
        sampling_efficiency : Float
            pymultinest sampling efficiency
        const_efficiency_mode : Bool
            pymultinest constant efficiency mode
        n_live_points : Int
            Number of live points to use in pymultinest.
        resume : bool
            Continue existing retrieval. If FALSE THIS WILL OVERWRITE YOUR EXISTING RETRIEVAL.
        bayes_factor_species : Str
            A pRT species that should be removed to test for the bayesian evidence for it's presence.
        corner_plot_names : List(Str)
            List of additional retrieval names that should be included in the corner plot.
        short_names : List(Str)
            For each corner_plot_name, a shorter name to be included when plotting.
        plot_multiple_retrieval_names : List(Str)
            List of additional retrievals to include when plotting spectra and PT profiles. Not yet implemented.
        """
        self.rd = run_definition

        # Maybe inherit from retrieval config class?
        self.retrieval_name = self.rd.retrieval_name
        self.data = self.rd.data
        self.run_mode = self.rd.run_mode
        self.parameters = self.rd.parameters

        self.output_dir = output_dir
        if self.output_dir != "" and not self.output_dir.endswith("/"):
            self.output_dir += "/"

        self.remove_species = bayes_factor_species
        self.corner_files = corner_plot_names
        if self.corner_files is None:
            self.corner_files = [self.retrieval_name]
        self.short_names = short_names

        # Plotting variables
        self.best_fit_specs = {}
        self.best_fit_params = {}
        self.posterior_sample_specs = {}
        self.plotting = test_plotting
        self.PT_plot_mode = test_plotting
        self.evaluate_sample_spectra = sample_spec

        # Pymultinest stuff
        self.sampling_efficiency = sampling_efficiency
        self.const_efficiency_mode = const_efficiency_mode
        self.n_live_points = n_live_points
        self.resume = resume
        # TODO
        self.retrieval_list = plot_multiple_retrieval_names

        # Set up pretty plotting
        if pRT_plot_style:
            import petitRADTRANS.retrieval.plot_style
        # Path to input opacities
        self.path = os.environ.get("pRT_input_data_path")
        if self.path == None:
            print('Path to input data not specified!')
            print('Please set pRT_input_data_path variable in .bashrc / .bash_profile or specify path via')
            print('    import os')
            print('    os.environ["pRT_input_data_path"] = "absolute/path/of/the/folder/input_data"')
            print('before creating a Radtrans object or loading the nat_cst module.')
            sys.exit(1)
        if not self.path.endswith("/"):
            self.path += "/"
        # Setup Directories
        if not os.path.isdir(self.output_dir + 'out_PMN/'):
            os.makedirs(self.output_dir + 'out_PMN', exist_ok=True)  
        if not os.path.isdir(self.output_dir + 'evaluate_' + self.retrieval_name +'/'):
            os.makedirs(self.output_dir + 'evaluate_' + self.retrieval_name, exist_ok=True) 
        
        # Setup pRT Objects for each data structure.
        print("Setting up PRT Objects")
        self.setupData()
        return

    def run(self):
        """
        run
        Run mode for the class. Uses pynultinest to sample parameter space
        and produce standard PMN outputs.
        """

        prefix = self.output_dir + 'out_PMN/'+self.retrieval_name+'_'

        if len(self.output_dir + 'out_PMN/') > 100:
            print("PyMultinest requires output directory names to be <100 characters. Please use a short path name.")
            sys.exit(3)
        if self.run_mode == 'retrieval':
            print("Starting retrieval: " + self.retrieval_name+'\n')
            # How many free parameters?
            n_params = 0
            free_parameter_names = []
            for pp in self.parameters:
                if self.parameters[pp].is_free_parameter:
                    free_parameter_names.append(self.parameters[pp].name)
                    n_params += 1
            json.dump(free_parameter_names, \
                    open(self.output_dir + 'out_PMN/'+self.retrieval_name+'_params.json', 'w'))
            pymultinest.run(self.LogLikelihood, 
                            self.Prior, 
                            n_params, 
                            outputfiles_basename=prefix, 
                            resume = self.resume, 
                            verbose = True, 
                            sampling_efficiency = self.sampling_efficiency,
                            const_efficiency_mode = self.const_efficiency_mode, 
                            n_live_points = self.n_live_points)
            self.analyzer = pymultinest.Analyzer(n_params = n_params, 
                                                 outputfiles_basename = prefix)
            s = self.analyzer.get_stats()

            json.dump(s, open(prefix + 'stats.json', 'w'), indent=4)
            print('  marginal likelihood:')
            print('    ln Z = %.1f +- %.1f' % (s['global evidence'], s['global evidence error']))
            print('  parameters:')
            for p, m in zip(self.parameters.keys(), s['marginals']):
                lo, hi = m['1sigma']
                med = m['median']
                sigma = (hi - lo) / 2
                if sigma == 0:
                    i = 3
                else:
                    i = max(0, int(-np.floor(np.log10(sigma))) + 1)
                fmt = '%%.%df' % i
                fmts = '\t'.join(['    %-15s' + fmt + " +- " + fmt])
                print(fmts % (p, med, sigma))
        return

    def run_ultranest(self):
        """
        run
        Run mode for the class. Uses ultranest to sample parameter space
        and produce standard outputs.
        """
        print("Warning, ultranest mode is still in development. Proceed with caution")
        try:
             import ultranest as un
        except ImportError:
            print("Could not import ultranest. Exiting.")
            sys.exit(1)
        # Todo: autodetect PMN vs UN outputs
        prefix = self.output_dir + 'out_PMN/'+self.retrieval_name+'_'
        if self.run_mode == 'retrieval':
            print("Starting retrieval: " + self.retrieval_name+'\n')
            # How many free parameters?
            n_params = 0
            free_parameter_names = []
            for pp in self.parameters:
                if self.parameters[pp].is_free_parameter:
                    free_parameter_names.append(self.parameters[pp].name)
                    n_params += 1
            json.dump(free_parameter_names, \
                    open(self.output_dir + 'out_PMN/'+self.retrieval_name+'_params.json', 'w'))
            sampler = un.ReactiveNestedSampler(free_parameter_names, 
                                               self.LogLikelihood,
                                               self.Prior,
                                               log_dir=self.output_dir + "out_" + self.retrieval_name, 
                                               resume=self.resume)

            result = sampler.run()
            sampler.print_results()
        return result

    def setupData(self,scaling=10,width = 3):
        """
        setupData
        Creates a pRT object for each data set that asks for a unique object.
        Checks if there are low resolution c-k models from exo-k, and creates them if necessary.
        The scaling and width parameters adjust the AMR grid as described in RetrievalConfig.setup_pres
        and models.fixed_length_amr. It is recommended to keep the defaults.

        arameters
        ----------
        scaling : int
            A multiplicative factor that determines the size of the full high resolution pressure grid,
            which will have length self.p_global.shape[0] * scaling.
        width : int
            The number of cells in the low pressure grid to replace with the high resolution grid.
        """
        for name,dd in self.data.items():
            # Only create if there's no other data
            # object using the same pRT object
            if dd.external_pRT_reference == None:  
                # Use ExoK to have low res models.
                if dd.model_resolution is not None:
                    species = []
                    # Check if low res opacities already exist
                    for line in self.rd.line_species:
                        if not os.path.isdir(self.path + "opacities/lines/corr_k/" +line + "_R_" + str(dd.model_resolution)):
                            species.append(line)
                    # If not, setup low-res c-k tables
                    if len(species)>0:
                        from .util import MMWs as masses
                        atmosphere = Radtrans(line_species = species, wlen_bords_micron = [0.1, 251.])
                        prt_path = self.path
                        ck_path = prt_path + 'opacities/lines/corr_k/'
                        print("Saving to " + ck_path)
                        print("Resolution: ", dd.model_resolution)
                        atmosphere.write_out_rebin(int(dd.model_resolution), 
                                                    path = ck_path, 
                                                    species = species, 
                                                    masses = masses)   
                    species = []
                    # Setup the pRT objects for the given dataset
                    for spec in self.rd.line_species:
                        species.append(spec + "_R_" + str(dd.model_resolution))
                    rt_object = Radtrans(line_species = cp.copy(species), \
                                        rayleigh_species= cp.copy(self.rd.rayleigh_species), \
                                        continuum_opacities = cp.copy(self.rd.continuum_opacities), \
                                        cloud_species = cp.copy(self.rd.cloud_species), \
                                        mode='c-k', \
                                        wlen_bords_micron = dd.wlen_range_pRT,
                                        do_scat_emis = self.rd.scattering)
                else:
                    rt_object = Radtrans(line_species = cp.copy(self.rd.line_species), \
                                        rayleigh_species= cp.copy(self.rd.rayleigh_species), \
                                        continuum_opacities = cp.copy(self.rd.continuum_opacities), \
                                        cloud_species = cp.copy(self.rd.cloud_species), \
                                        mode='c-k', \
                                        wlen_bords_micron = dd.wlen_range_pRT,
                                        do_scat_emis = self.rd.scattering)

                # Create random P-T profile to create RT arrays of the Radtrans object.
                if self.rd.AMR:
                    p = self.rd.setup_pres(scaling,width)
                else:
                    p = self.rd.p_global
                rt_object.setup_opa_structure(p)
                dd.pRT_object = rt_object
        return

    def Prior(self, cube, ndim, nparams):
        """
        Prior
        pymultinest prior function. Transforms unit hypercube into physical space.
        """
        i_p = 0
        for pp in self.parameters:
            if self.parameters[pp].is_free_parameter:
                cube[i_p] = self.parameters[pp].get_param_uniform(cube[i_p])
                i_p += 1

    def LogLikelihood(self,cube,ndim,nparam):
        """
        LogLikelihood
        pymultinest required likelihood function.
        """
        log_likelihood = 0.
        log_prior      = 0.

        i_p = 0 # parameter count
        for pp in self.parameters:
            if self.parameters[pp].is_free_parameter:
                self.parameters[pp].set_param(cube[i_p])
                i_p += 1

        for name,dd in self.data.items():
            # Only calculate spectra within a given
            # wlen range once
            if dd.scale:
                dd.scale_factor = self.parameters[name + "_scale_factor"].value
            if dd.external_pRT_reference == None:
                if not self.PT_plot_mode:
                    # Compute the model
                    wlen_model, spectrum_model = \
                        dd.model_generating_function(dd.pRT_object, 
                                                    self.parameters, 
                                                    self.PT_plot_mode,
                                                    AMR = self.rd.AMR)
                    # Sanity checks on outputs
                    #print(spectrum_model)
                    if spectrum_model is None:
                        return -np.inf
                    if np.isnan(spectrum_model).any():
                        return -np.inf
                else:
                    # Get the PT profile
                    if name == self.rd.plot_kwargs["take_PTs_from"]:
                        pressures, temperatures = \
                            dd.model_generating_function(dd.pRT_object, 
                                                         self.parameters, 
                                                         self.PT_plot_mode,
                                                         AMR = self.rd.AMR)
                        return pressures, temperatures     
                log_likelihood += dd.get_chisq(wlen_model, 
                                            spectrum_model, 
                                            self.plotting)
                # Save sampled outputs if necessary.
                if self.run_mode == 'evaluate':
                    np.savetxt(self.output_dir + 'evaluate_' + self.retrieval_name + '/model_spec_best_fit_'+ 
                            name+'.dat', 
                            np.column_stack((wlen_model, 
                                                spectrum_model)))

                    self.best_fit_specs[name] = [wlen_model, \
                                            spectrum_model]
                    if self.evaluate_sample_spectra:
                        self.posterior_sample_specs[name] = [wlen_model, \
                                                spectrum_model]
                # Check for data using the same pRT object,
                # calculate log_likelihood
                for de_name,dede in self.data.items():
                    if dede.external_pRT_reference != None:
                        if dede.scale:
                            dd.scale_factor = self.parameters[de_name + "_scale_factor"].value
                        if dede.external_pRT_reference == name:
                            log_likelihood += dede.get_chisq(wlen_model, \
                                            spectrum_model, \
                                            self.plotting)
        #print(log_likelihood)
        return log_likelihood + log_prior  
    
    def getSamples(self, output_dir = None):
        """
        getSamples
        This function looks in the given output directory and finds the post_equal_weights
        file associated with the current retrieval name.

        parameters
        ----------
        output_dir : str
            Parent directory of the out_PMN/*post_equal_weights.dat file
        
        returns
        -------
        sample_dict : dict
            A dictionary with keys being the name of the retrieval, and values are a numpy
            ndarray containing the samples in the post_equal_weights file
        parameter_dict : dict
            A dictionary with keys being the name of the retrieval, and values are a list of names
            of the parameters used in the retrieval. The first name corresponds to the first column
            of the samples, and so on.
        """
        sample_dict = {}
        parameter_dict = {}
        if output_dir is None:
            output_dir = self.output_dir
        for name in self.corner_files:
            samples = np.genfromtxt(output_dir +'out_PMN/'+ \
                                    name+ \
                                    '_post_equal_weights.dat')

            parameters_read = json.load(open(output_dir + 'out_PMN/'+ \
                                        name+ \
                                        '_params.json'))
            sample_dict[name] = samples
            parameter_dict[name] = parameters_read
        return sample_dict, parameter_dict

    def getBestFitParams(self,best_fit_params,parameters_read):
        """
        getBestFitParams
        This function converts the sample from the post_equal_weights file with the maximum
        log likelihood, and converts it into a dictionary of Parameters that can be used in
        a model function.

        parameters
        ----------
        best_fit_params : numpy.ndarray
            An array of the best fit parameter values (or any other sample)
        parameters_read : list
            A list of the free parameters as read from the output files.
        """
        i_p = 0
        for pp in self.parameters:
            if self.parameters[pp].is_free_parameter:
                for i_s in range(len(parameters_read)):
                    if parameters_read[i_s] == self.parameters[pp].name:
                        self.best_fit_params[self.parameters[pp].name] = Parameter(pp,False,value=best_fit_params[i_p])
                        i_p += 1
            else:
                self.best_fit_params[pp] = Parameter(pp,False,value=self.parameters[pp].value)
        return self.best_fit_params

    def getBestFitModel(self,best_fit_params,parameters_read,model_generating_func = None,ret_name = None):
        """
        getBestFitModel
        This function uses the best fit parameters to generate a pRT model that spans the entire wavelength
        range of the retrieval, to be used in plots.

        parameters
        ----------
        best_fit_params : numpy.ndarray
            A numpy array containing the best fit parameters, to be passed to getBestFitParams
        parameters_read : list
            A list of the free parameters as read from the output files.
        model_generating_fun : method
            A function that will take in the standard 'model' arguments (pRT_object, params, pt_plot_mode, AMR, resolution)
            and will return the wavlength and flux arrays as calculated by petitRadTrans.
            If no argument is given, it uses the method of the first dataset included in the retrieval.
        ret_name : str
            If plotting a fit from a different retrieval, input the retrieval name to be included.
        
        returns
        -------
        bf_wlen : numpy.ndarray
            The wavelength array of the best fit model
        bf_spectrum : numpy.ndarray
            The emission or transmission spectrum array, with the same shape as bf_wlen
        """
        print("Computing Best Fit Model, this may take a minute...")
        if ret_name == None:
            ret_name = self.retrieval_name

        # Find the boundaries of the wavelength range to calculate
        wmin = 99999.0
        wmax = 0.0
        for name,dd in self.data.items():
            if dd.wlen_range_pRT[0] < wmin:
                wmin = dd.wlen_range_pRT[0]
            if dd.wlen_range_pRT[1] > wmax:
                wmax = dd.wlen_range_pRT[1]
        # Set up parameter dictionary
        if not self.retrieval_name in self.best_fit_specs.keys():
            self.getBestFitParams(best_fit_params,parameters_read)

        # Setup the pRT object
        bf_prt = Radtrans(line_species = cp.copy(self.rd.line_species), \
                            rayleigh_species= cp.copy(self.rd.rayleigh_species), \
                            continuum_opacities = cp.copy(self.rd.continuum_opacities), \
                            cloud_species = cp.copy(self.rd.cloud_species), \
                            mode='c-k', \
                            wlen_bords_micron = [wmin*0.98,wmax*1.02],
                            do_scat_emis = self.rd.scattering)
        if self.rd.AMR:
            p = self.rd.setup_pres()
        else:
            p = self.rd.p_global
        bf_prt.setup_opa_structure(p)

        # Check what model function we're using
        if model_generating_func is None:
            mg_func = list(self.data.values())[0].model_generating_function
        else:
            mg_func = model_generating_func 

        # get the spectrum
        bf_wlen, bf_spectrum= mg_func(bf_prt, 
                                      self.best_fit_params, 
                                      PT_plot_mode= False,
                                      AMR = True,
                                      resolution = None)
        # Add to the dictionary.
        self.best_fit_specs[ret_name]= [bf_wlen,bf_spectrum]
        return bf_wlen, bf_spectrum


#############################################################
# Plotting functions
#############################################################
    def plotAll(self, output_dir = None):
        """
        plotAll
        Produces plots for the best fit spectrum, a sample of 100 output spectra,
        the best fit PT profile and a corner plot for parameters specified in the
        run definition.
        """
        if output_dir is None:
            output_dir = self.output_dir
        sample_dict, parameter_dict = self.getSamples(output_dir)

        ###########################################
        # Plot best-fit spectrum
        ###########################################
        samples_use = cp.copy(sample_dict[self.retrieval_name])
        parameters_read = cp.copy(parameter_dict[self.retrieval_name])
        i_p = 0
        for pp in self.parameters:
            if self.parameters[pp].is_free_parameter:
                for i_s in range(len(parameters_read)):
                    if parameters_read[i_s] == self.parameters[pp].name:
                        samples_use[:,i_p] = sample_dict[self.retrieval_name][:, i_s]
                i_p += 1
                
        print("Best fit parameters")
        i_p = 0
        # Get best-fit index
        logL = samples_use[:,-1]
        best_fit_index = np.argmax(logL)
        for pp in self.parameters:
            if self.parameters[pp].is_free_parameter:
                for i_s in range(len(parameters_read)):
                    if parameters_read[i_s] == self.parameters[pp].name:
                        print(self.parameters[pp].name, samples_use[best_fit_index][i_p])
                        i_p += 1
                        
        # Plotting
        self.plotSpectra(samples_use,parameters_read)
        self.plotSampled(samples_use, parameters_read)
        self.plotPT(sample_dict,parameters_read)
        self.plotCorner(sample_dict,parameter_dict,parameters_read)
        print("Done!")
        return

    def plotSpectra(self,samples_use,parameters_read,model_generating_func = None):
        """
        plotSpectra
        Plot the best fit spectrum, the data from each dataset and the residuals between the two.
        Saves a file to $OUTPUT_DIR/evaluate_$RETRIEVAL_NAME/best_fit_spec.pdf
        TODO: include plotting of multiple retrievals

        parameters:
        samples_use : numpy.ndarray
            An array of the samples from the post_equal_weights file, used to find the best fit sample
        parameters_read : list
            A list of the free parameters as read from the output files.
        model_generating_fun : method
            A function that will take in the standard 'model' arguments (pRT_object, params, pt_plot_mode, AMR, resolution)
            and will return the wavlength and flux arrays as calculated by petitRadTrans.
            If no argument is given, it uses the method of the first dataset included in the retrieval.
        
        returns
        -------
        fig : matplotlib.figure
        ax : matplotlib.axes
            The upper pane of the plot, containing the best fit spectrum and data
        ax_r : matplotlib.axes
            The lower pane of the plot, containing the residuals between the fit and the data
        """
        print("Plotting Best-fit spectrum")
        fig, axes = fig, axes = plt.subplots(nrows=2, ncols=1, sharex='col', sharey=False,
                               gridspec_kw={'height_ratios': [2.5, 1],'hspace':0.1},
                               figsize=(20, 10))
        ax = axes[0] # Normal Spectrum axis
        ax_r = axes[1] # residual axis
                
        # Get best-fit index
        logL = samples_use[:,-1]
        best_fit_index = np.argmax(logL)

        # Setup best fit spectrum
        # First get the fit for each dataset for the residual plots
        self.LogLikelihood(samples_use[best_fit_index, :-1], 0, 0)
        # Then get the full wavelength range
        bf_wlen, bf_spectrum = self.getBestFitModel(samples_use[best_fit_index, :-1],parameters_read,model_generating_func)

        # Iterate through each dataset, plotting the data and the residuals.
        for name,dd in self.data.items():
            # If the user has specified a resolution, rebin to that
            try:
                # Sometimes this fails, I'm not super sure why.
                resolution_data = np.mean(dd.wlen[1:]/np.diff(dd.wlen))
                ratio = resolution_data / self.rd.plot_kwargs["resolution"]
                if int(ratio) > 1:
                    flux,edges,_ = binned_statistic(dd.wlen,dd.flux,'mean',dd.wlen.shape[0]/ratio)
                    error,_,_ = binned_statistic(dd.wlen,dd.flux_error,'mean',dd.wlen.shape[0]/ratio)/np.sqrt(ratio)
                    wlen = np.array([(edges[i]+edges[i+1])/2.0 for i in range(edges.shape[0]-1)])
                    # Old method
                    #wlen = nc.running_mean(dd.wlen, int(ratio))[::int(ratio)]
                    #error = nc.running_mean(dd.flux_error / int(np.sqrt(ratio)), \
                    #                        int(ratio))[::int(ratio)]
                    #flux = nc.running_mean(dd.flux, \
                    #                        int(ratio))[::int(ratio)]
                else:
                    wlen = dd.wlen
                    error = dd.flux_error
                    flux = dd.flux
            except:
                wlen = dd.wlen
                error = dd.flux_error
                flux = dd.flux

            # If the data has an arbitrary retrieved scaling factor
            scale = dd.scale_factor

            # Setup bins to rebin the best fit model to find the residuals
            wlen_bins = np.zeros_like(wlen)
            wlen_bins[:-1] = np.diff(wlen)
            wlen_bins[-1] = wlen_bins[-2]
            if dd.external_pRT_reference == None:
                best_fit_binned = rgw(self.best_fit_specs[name][0], \
                                        self.best_fit_specs[name][1], \
                                        wlen, \
                                        wlen_bins)
            else:
                best_fit_binned = rgw(self.best_fit_specs[dd.external_pRT_reference][0], \
                            self.best_fit_specs[dd.external_pRT_reference][1], \
                            wlen, \
                            wlen_bins)
            
            # Plot the data
            if not dd.photometry:
                ax.errorbar(wlen, \
                            flux * self.rd.plot_kwargs["y_axis_scaling"] * scale, \
                            yerr = error * self.rd.plot_kwargs["y_axis_scaling"] *scale, \
                            marker='o', markeredgecolor='k', linewidth = 0, elinewidth = 2, \
                            label = dd.name, zorder =10, alpha = 0.9,)
            else:
                ax.errorbar(wlen, \
                            flux * self.rd.plot_kwargs["y_axis_scaling"] * scale, \
                            yerr = error * self.rd.plot_kwargs["y_axis_scaling"] *scale, \
                            xerr = dd.width_photometry/2., linewidth = 0, elinewidth = 2, \
                            marker='o', markeredgecolor='k', zorder = 10, \
                            label = dd.name, alpha = 0.9)
            # Plot the residuals
            col = ax.get_lines()[-1].get_color()
            if dd.external_pRT_reference == None:
                ax_r.errorbar(wlen, \
                            ((flux*scale) - best_fit_binned )/error , 
                            yerr = error/error,
                            color = col,
                            linewidth = 0, elinewidth = 2, \
                            marker='o', markeredgecolor='k', zorder = 10,
                            alpha = 0.9)
            else:
                ax_r.errorbar(wlen, \
                        ((flux*scale) - best_fit_binned )/error,
                        yerr = error/error,
                        color = col,
                        linewidth = 0, elinewidth = 2, \
                        marker='o', markeredgecolor='k', zorder = 10,
                        alpha = 0.9)
        # Plot the best fit model
        ax.plot(bf_wlen, \
                bf_spectrum * self.rd.plot_kwargs["y_axis_scaling"],
                label = 'Best Fit Model',
                linewidth=4,
                alpha = 0.5,
                color = 'r')
        # Plot the shading in the residual plot        
        yabs_max = abs(max(ax_r.get_ylim(), key=abs))
        lims = ax.get_xlim()
        lim_y = ax.get_ylim()
        lim_y = [lim_y[0],lim_y[1]*1.12]
        ax.set_ylim(lim_y)
        # weird scaling to get axis to look ok on log plots
        if self.rd.plot_kwargs["xscale"] == 'log':
            lims = [lims[0]*1.09,lims[1]*1.02]
        else:
            lims = [bf_wlen[0]*0.98,bf_wlen[-1]*1.02]
        ax.set_xlim(lims)
        ax_r.set_xlim(lims)
        ax_r.set_ylim(ymin=-yabs_max, ymax=yabs_max)
        ax_r.fill_between(lims,-1,1,color='dimgrey',alpha=0.4,zorder = -10)
        ax_r.fill_between(lims,-3,3,color='darkgrey',alpha=0.3,zorder = -9)
        ax_r.fill_between(lims,-5,5,color='lightgrey',alpha=0.3,zorder = -8)
        ax_r.axhline(linestyle = '--', color = 'k',alpha=0.8, linewidth=2)

        # Making the plots pretty
        try:
            ax.set_xscale(self.rd.plot_kwargs["xscale"])
        except:
            pass
        try:
            ax.set_yscale(self.rd.plot_kwargs["yscale"])
        except:
            pass

        # Fancy ticks for upper pane
        ax.tick_params(axis="both",direction="in",length=10,bottom=True, top=True, left=True, right=True)
        try:
            ax.xaxis.set_major_formatter('{x:.1f}')
        except:
            print("Please update to matplotlib 3.3.4 or greater")
            pass

        if self.rd.plot_kwargs["xscale"] == 'log':
            # For the minor ticks, use no labels; default NullFormatter.
            x_major = LogLocator(base = 10.0, subs = (1,2,3,4), numticks = 4)
            ax.xaxis.set_major_locator(x_major)
            x_minor = LogLocator(base = 10.0, subs = np.arange(0.1,10.1,0.1)*0.1, numticks = 100)
            ax.xaxis.set_minor_locator(x_minor)
            ax.xaxis.set_minor_formatter(NullFormatter())
        else:
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.tick_params(axis='both', which='minor', bottom=True, top=True, left=True, right=True, direction='in',length=5)
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(axis='both', which='minor', bottom=True, top=True, left=True, right=True, direction='in',length=5)
        ax.set_ylabel(self.rd.plot_kwargs["spec_ylabel"])

        # Fancy ticks for lower pane
        ax_r.tick_params(axis="both",direction="in",length=10,bottom=True, top=True, left=True, right=True)

        try:
            ax_r.xaxis.set_major_formatter('{x:.1f}')
        except:
            print("Please update to matplotlib 3.3.4 or greater")
            pass

        if self.rd.plot_kwargs["xscale"] == 'log':
            # For the minor ticks, use no labels; default NullFormatter.
            x_major = LogLocator(base = 10.0, subs = (1,2,3,4), numticks = 4)
            ax_r.xaxis.set_major_locator(x_major)
            x_minor = LogLocator(base = 10.0, subs = np.arange(0.1,10.1,0.1)*0.1, numticks = 100)
            ax_r.xaxis.set_minor_locator(x_minor)
            ax_r.xaxis.set_minor_formatter(NullFormatter())
        else:
            ax_r.xaxis.set_minor_locator(AutoMinorLocator())
            ax_r.tick_params(axis='both', which='minor', bottom=True, top=True, left=True, right=True, direction='in',length=5)
        ax_r.yaxis.set_minor_locator(AutoMinorLocator())
        ax_r.tick_params(axis='both', which='minor', bottom=True, top=True, left=True, right=True, direction='in',length=5)
        ax_r.set_ylabel("Residuals [$\sigma$]")
        ax_r.set_xlabel(self.rd.plot_kwargs["spec_xlabel"])
        ax.legend(loc='upper center',ncol = len(self.data.keys())+1).set_zorder(1002) 
        plt.tight_layout()
        plt.savefig(self.output_dir + 'evaluate_'+self.rd.retrieval_name +'/best_fit_spec.pdf')
        return fig, ax, ax_r

    def plotSampled(self,samples_use,parameters_read):
        """
        plotSampled
        Plot a set of randomly sampled output spectra

        Parameters:
        -----------
        samples_use : np.ndarray
            posterior samples from pynmultinest outputs (post_equal_weights)
        """
        print("Plotting Best-fit spectrum with "+ str(self.rd.plot_kwargs["nsample"]) + " samples.")
        print("This could take some time...")
        len_samples = samples_use.shape[0]
        path = self.output_dir + 'evaluate_'+self.retrieval_name + "/"

        data_use= {}
        for name, dd in self.data.items():
            if not os.path.exists(path + name.replace(' ','_')+'_sampled_'+ 
                        str(int(self.rd.plot_kwargs["nsample"])).zfill(int(np.log10(self.rd.plot_kwargs["nsample"])+1))+'.dat'):
                data_use[name] = dd

        for i_sample in range(int(self.rd.plot_kwargs["nsample"])):
            random_index = int(np.random.uniform()*len_samples)
            self.LogLikelihood(samples_use[random_index, :-1], 0, 0)
            for name,dd in data_use.items():
                if dd.external_pRT_reference == None:
                    np.savetxt(path +name.replace(' ','_')+'_sampled_'+ 
                                str(int(i_sample+1)).zfill(int(np.log10(self.rd.plot_kwargs["nsample"])+1))+'.dat',
                                np.column_stack((self.posterior_sample_specs[name][0],
                                                 self.posterior_sample_specs[name][1])))
            
        fig,ax = plt.subplots(figsize = (16,10))
        plot_specs(fig,ax,path, 'Retrieved', '#ff9f9f', '#ff3d3d', -10, rebin_val = 5)
        #print("Plotting sample spectrum failed, are you sure you allowed sampling?")
        #return None
        
        for name,dd in self.data.items():
            plot_data(fig,ax,dd, name, 'white', 0, rebin_val = 5)
            #plt.ylim([0.006, 0.0085])

        ax.set_xlabel('Wavelength [micron]')
        ax.set_ylabel(self.rd.plot_kwargs["spec_ylabel"])
        ax.legend(loc='best')
        plt.tight_layout()
        plt.savefig(path +'sampled_data.pdf',bbox_inches = 0.)
        return fig, ax

    def plotPT(self,sample_dict,parameters_read):
        """
        plotPT
        Plot the PT profile with error contours

        Parameters:
        -----------
        samples_use : np.ndarray
            posterior samples from pynmultinest outputs (post_equal_weights)
        parameters_read : List
            Used to plot correct parameters, as some in self.parameters are not free, and
            aren't included in the PMN outputs

        returns
        -------
        fig : matplotlib.figure
        ax : matplotlib.axes
        """
        print("Plotting PT profiles")
        self.PT_plot_mode = True
        samples_use = cp.copy(sample_dict[self.retrieval_name])
        i_p = 0
        for pp in self.parameters:
            if self.parameters[pp].is_free_parameter:
                for i_s in range(len(parameters_read)):
                    if parameters_read[i_s] == self.parameters[pp].name:
                        samples_use[:,i_p] = sample_dict[self.retrieval_name][:, i_s]
                i_p += 1

        temps = []
        for i_s in range(len(samples_use)):
            pressures, t = self.LogLikelihood(samples_use[i_s, :-1], 0, 0)
            temps.append(t)

        temps = np.array(temps)
        temps_sort = np.sort(temps, axis=0)
        fig,ax = plt.subplots(figsize=(16, 10))
        len_samp = len(samples_use)
        ax.fill_betweenx(pressures, \
                        x1 = temps_sort[0, :], \
                        x2 = temps_sort[-1, :], \
                        color = 'cyan', label = 'all')
        ax.fill_betweenx(pressures, \
                        x1 = temps_sort[int(len_samp*(0.5-0.997/2.)), :], \
                        x2 = temps_sort[int(len_samp*(0.5+0.997/2.)), :], \
                        color = 'brown', label = '3 sig')
        ax.fill_betweenx(pressures, \
                        x1 = temps_sort[int(len_samp*(0.5-0.95/2.)), :], \
                        x2 = temps_sort[int(len_samp*(0.5+0.95/2.)), :], \
                        color = 'orange', label = '2 sig')
        ax.fill_betweenx(pressures, \
                        x1 = temps_sort[int(len_samp*(0.5-0.68/2.)), :], \
                        x2 = temps_sort[int(len_samp*(0.5+0.68/2.)), :], \
                        color = 'red', label = '1 sig')

        ax.set_yscale('log')
        try:
            ax.set_ylim(self.rd.plot_kwargs["press_limits"])
        except:
            ax.set_ylim([pressures[-1], pressures[0]])
        try:
            ax.set_xlim(self.rd.plot_kwargs["temp_limits"])
        except:
            pass
        ax.set_xlabel('Temperature [K]')
        ax.set_ylabel('Pressure [bar]')
        ax.legend(loc='best')
        plt.savefig(self.output_dir + 'evaluate_'+self.retrieval_name +'/PT_envelopes.pdf')
        return fig, ax

    def plotCorner(self,sample_dict,parameter_dict,parameters_read):
        """
        plotCorner
        Make the corner plots

        Parameters:
        -----------
        samples_dict : Dict
            Dictionary of samples from PMN outputs, with keys being retrieval names
        parameter_dict : Dict
            Dictionary of parameters for each of the retrievals to be plotted.
        parameters_read : List
            Used to plot correct parameters, as some in self.parameters are not free, and
            aren't included in the PMN outputs
        """
        print("Making corner plot")
        sample_use_dict = {}
        p_plot_inds = {}
        p_ranges = {}
        p_use_dict = {}
        for name,params in parameter_dict.items():
            samples_use = cp.copy(sample_dict[name])
            parameters_use = cp.copy(params)
            parameter_plot_indices = []
            parameter_ranges       = []
            i_p = 0
            for pp in parameters_read:
                parameter_ranges.append(self.parameters[pp].corner_ranges)
                if self.parameters[pp].plot_in_corner:
                    parameter_plot_indices.append(i_p)
                if self.parameters[pp].corner_label != None:
                    parameters_use[i_p] = self.parameters[pp].corner_label
                if self.parameters[pp].corner_transform != None:
                    samples_use[:, i_p] = \
                        self.parameters[pp].corner_transform(samples_use[:, i_p])
                i_p += 1
            p_plot_inds[name] = parameter_plot_indices
            p_ranges[name] = parameter_ranges
            p_use_dict[name] = parameters_use
            sample_use_dict[name] = samples_use
            

        output_file = self.output_dir + 'evaluate_'+self.retrieval_name +'/corner_nice.pdf'
        max_val_ratio = 5.

        # from Plotting
        contour_corner(sample_use_dict, \
                        p_use_dict, \
                        output_file, \
                        parameter_plot_indices = p_plot_inds, \
                        max_val_ratio = max_val_ratio, \
                        parameter_ranges = p_ranges, \
                        true_values = None)
