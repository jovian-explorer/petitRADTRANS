###########################################
# Input / output, general run definitions
###########################################
import sys, os
# To not have numpy start parallelizing on its own
os.environ["OMP_NUM_THREADS"] = "1"

# Read external packages
import numpy as np
import copy as cp
import matplotlib
matplotlib.use('Agg') # set the backend before importing pyplot
font = {'family' : 'normal',
        'size'   : 24}
matplotlib.rc('font', **font)

import pymultinest
import json
import argparse as ap
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, AutoMinorLocator, LogLocator, NullFormatter

# Read own packages
import petitRADTRANS # Just need the filename actually
from petitRADTRANS import Radtrans
from petitRADTRANS import nat_cst as nc
from .parameter_class import Parameter
from .data_class import Data
from .plotting import plot_specs,plot_data,contour_corner

class Retrieval:
    def __init__(self,
                 run_definition,
                 output_dir = "",
                 test_plotting = False,
                 sample_spec = False,
                 sampling_efficiency = 0.05,\
                 const_efficiency_mode = True, \
                 n_live_points = 4000,
                 bayes_factor_species = None,
                 corner_plot_names = None,
                 short_names = None,
                 plot_multiple_retrieval_names = None):
        """
        Retrieve Class
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
        self.short_names = short_names

        # Plotting variables
        self.best_fit_specs = {}
        self.posterior_sample_specs = {}
        self.plotting = test_plotting
        self.PT_plot_mode = test_plotting
        self.evaluate_sample_spectra = sample_spec

        # Pymultinest stuff
        self.sampling_efficiency = sampling_efficiency
        self.const_efficiency_mode = const_efficiency_mode
        self.n_live_points = n_live_points

        # TODO
        self.retrieval_list = plot_multiple_retrieval_names

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
                            resume = True, 
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
    def plotAll(self, output_dir = None):
        """
        plotAll
        Produces plots for the best fit spectrum, a sample of 100 output spectra,
        the best fit PT profile and a corner plot for parameters specified in the
        run definition.
        """
        # Read in samples
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
                
        # Get best-fit index
        logL = samples_use[:,-1]
        best_fit_index = np.argmax(logL)

        # Setup best fit spectrum
        self.LogLikelihood(samples_use[best_fit_index, :-1], 0, 0)

        print("Best fit parameters")
        i_p = 0
        for pp in self.parameters:
            if self.parameters[pp].is_free_parameter:
                for i_s in range(len(parameters_read)):
                    if parameters_read[i_s] == self.parameters[pp].name:
                        print(self.parameters[pp].name, samples_use[best_fit_index][i_p])
                        i_p += 1
                        
        # Plotting
        self.plotSpectra(samples_use[best_fit_index, :-1])
        self.plotSampled(samples_use, parameters_read)
        self.plotPT(sample_dict,parameters_read)
        self.plotCorner(sample_dict,parameter_dict,parameters_read)
        print("Done!")
        return

    def setupData(self):
        """
        setupData
        Creates a pRT object for each data set that asks for a unique object.
        Checks if there are low resolution c-k models from exo-k, and creates them if necessary.
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
                        from util import MMWs as masses
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
                    p = self.rd.setup_pres()
                else:
                    p = self.rd.p_global
                rt_object.setup_opa_structure(p)
                dd.pRT_object = rt_object
        return

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
                                                    AMR = self.rd.AMR,
                                                    resolution = dd.model_resolution)
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
                    if not self.evaluate_sample_spectra:
                        np.savetxt(self.output_dir + 'evaluate_data/model_spec_best_fit'+ 
                                name+'.dat', 
                                np.column_stack((wlen_model, 
                                                    spectrum_model)))
                        self.best_fit_specs[name] = [wlen_model, \
                                                spectrum_model]
                    else:
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
    def getBestFitModel(self,best_fit_params,model_generating_func = None):
        wmin = 99999.0
        wmax = 0.0
        for name,dd in self.data.items():
            if dd.wlen_range_pRT[0] < wmin:
                wmin = dd.wlen_range_pRT[0]
            if dd.wlen_range_pRT[1] < wmax:
                wmax = dd.wlen_range_pRT[1]
        
        bf_prt = Radtrans(line_species = cp.copy(self.rd.line_species), \
                            rayleigh_species= cp.copy(self.rd.rayleigh_species), \
                            continuum_opacities = cp.copy(self.rd.continuum_opacities), \
                            cloud_species = cp.copy(self.rd.cloud_species), \
                            mode='c-k', \
                            wlen_bords_micron = [wmin*0.98,wmax*1.02],
                            do_scat_emis = self.rd.scattering)
        if model_generating_func is None:
            mg_func = self.data.values()[0].model_generating_function
        else:
            mg_func = model_generating_func 

        bf_wlen, bf_spectrum= mg_func(bf_prt, 
                                        best_fit_params, 
                                        self.PT_plot_mode,
                                        AMR = self.rd.AMR,
                                        resolution = dd.model_resolution)
        return bf_wlen, bf_spectrum

    def plotSpectra(self,best_fit_params,model_generating_func = None):
        """
        plotSpectra
        Plot the best fit spectrum, the data from each dataset and the residuals between the two.
        """
        print("Plotting Best-fit spectrum")
        fig, axes = fig, axes = plt.subplots(nrows=2, ncols=1, sharex='col', sharey=False,
                               gridspec_kw={'height_ratios': [2.5, 1],'hspace':0.1},
                               figsize=(24, 12))
        ax = axes[0] # Normal Spectrum axis
        ax_r = axes[1] # residual axis
        bf_wlen, bf_spectrum = self.getBestFitModel(best_fit_params,model_generating_func)

        for name,dd in self.data.items():
            try:
                resolution_data = np.mean(dd.wlen[1:]/np.diff(dd.wlen))
                ratio = resolution_data / self.rd.plot_kwargs["resolution"]
                if int(ratio) > 1:
                    wlen = nc.running_mean(dd.wlen, int(ratio))[::int(ratio)]
                    error = nc.running_mean(dd.flux_error / int(np.sqrt(ratio)), \
                                            int(ratio))[::int(ratio)]
                    flux = nc.running_mean(dd.flux, \
                                            int(ratio))[::int(ratio)]
                else:
                    wlen = dd.wlen
                    error = dd.flux_error
                    flux = dd.flux
            except:
                wlen = dd.wlen
                error = dd.flux_error
                flux = dd.flux

        scale = dd.scale_factor
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
        col = ax._get_lines.get_color()
        if dd.external_pRT_reference == None:
            print(self.best_fit_specs.keys())
            ax_r.plot(self.best_fit_specs[name][0], \
                     (flux - self.best_fit_specs[name][1])/np.std(flux - self.best_fit_specs[name][1]) , \
                     color = col,
                     linewidth = 0)
        else:
            # TODO might need to add rebinning step here
            ax_r.plot(self.best_fit_specs[dd.external_pRT_reference][0], \
                     (flux - self.best_fit_specs[dd.external_pRT_reference][1])/np.std(flux - self.best_fit_specs[name][1]) , \
                     color = col, 
                     zorder = -10,
                     linewidth = 0)
        ax.plot(bf_wlen, \
                bf_spectrum * self.rd.plot_kwargs["y_axis_scaling"],
                linewidth=4,
                alpha = 0.5,
                color = 'r')
        try:
            ax.set_xscale(self.rd.plot_kwargs["xscale"])
        except:
            pass

        try:
            ax.set_yscale(self.rd.plot_kwargs["yscale"])
        except:
            pass

        ax.tick_params(axis="both",direction="in",length=10,bottom=True, top=True, left=True, right=True)
        #ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.xaxis.set_major_formatter('{x:.0f}')
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

        ax.set_xlabel(self.rd.plot_kwargs["spec_xlabel"])
        ax.set_ylabel(self.rd.plot_kwargs["spec_ylabel"])

        ax_r.tick_params(axis="both",direction="in",length=10,bottom=True, top=True, left=True, right=True)
        #ax.xaxis.set_major_locator(MultipleLocator(1))
        ax_r.xaxis.set_major_formatter('{x:.0f}')
        if self.rd.plot_kwargs["xscale"] == 'log':
            # For the minor ticks, use no labels; default NullFormatter.
            x_major = LogLocator(base = 10.0, subs = (1,2,3,4), numticks = 4)
            ax_r.xaxis.set_major_locator(x_major)

            x_minor = LogLocator(base = 10.0, subs = np.arange(0.1,10.1,0.1)*0.1, numticks = 100)
            ax_r.xaxis.set_minor_locator(x_minor)
            ax_r.xaxis.set_minor_formatter(NullFormatter())
        else:
            ax_r.xaxis.set_minor_locator(AutoMinorLocator())
            ax._rtick_params(axis='both', which='minor', bottom=True, top=True, left=True, right=True, direction='in',length=5)

        ax_r.yaxis.set_minor_locator(AutoMinorLocator())
        ax_r.tick_params(axis='both', which='minor', bottom=True, top=True, left=True, right=True, direction='in',length=5)

        ax_r.set_xlabel("Residuals")
        ax_r.set_ylabel(self.rd.plot_kwargs["spec_ylabel"])
        #([model_min*0.98, model_max*1.02])
        plt.tight_layout()
        plt.legend(loc='best')
        plt.savefig('evaluate_'+self.rd.retrieval_name +'/best_fit_spec.pdf')

    def plotSampled(self,samples_use):
        """
        plotSampled
        Plot a set of randomly sampled output spectra

        Parameters:
        -----------
        samples_use : np.ndarray
            posterior samples from pynmultinest outputs (post_equal_weights)
        """
        print("Plotting Best-fit spectrum with 100 samples")
        len_samples = np.shape(samples_use)[0]
        evaluate_sample_spectra = True
        if self.rd.write_out_spec_sample:
            for i_sample in range(self.rd.plot_kwargs["nsample"]):
                random_index = int(np.random.uniform()*len_samples)
                self.LogLikelihood(samples_use[random_index, :-1], 0, 0)
                for name,dd in self.data.items():
                    if dd.external_pRT_reference == None:
                        np.savetxt('evaluate_'+self.retrieval_name + '/' +\
                                name.replace(' ','_')+'_sampled_'+ \
                                    str(int(i_sample+1)).zfill(int(np.log10(self.rd.plot_kwargs["nsample"])+1))+'.dat', \
                                np.column_stack((self.posterior_sample_specs[name][0], \
                                                self.posterior_sample_specs[name][1])))
            
        path = self.output_dir + 'evaluate_'+self.retrieval_name
        fig,ax = plt.subplots(figsize = (16,10))
        plot_specs(fig,ax,path, 'Retrieved', '#ff9f9f', '#ff3d3d', -10, rebin_val = 5)
        for name,dd in self.data.items():
            plot_data(fig,ax,dd, name, 'white', 0, rebin_val = 5)
            #plt.ylim([0.006, 0.0085])
        #TODO: autoset xlim
        #ax.set_xlim([1.4,2.6])
        ax.set_xlabel('Wavelength [micron]')
        ax.set_ylabel(r'$F_{\rm P}/F_*$ (\%)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(path +'/sampled_data.pdf',bbox_inches = 0.)
    
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

        len_samp = len(samples_use)
        plt.fill_betweenx(pressures, \
                        x1 = temps_sort[0, :], \
                        x2 = temps_sort[-1, :], \
                        color = 'cyan', label = 'all')
        plt.fill_betweenx(pressures, \
                        x1 = temps_sort[int(len_samp*(0.5-0.997/2.)), :], \
                        x2 = temps_sort[int(len_samp*(0.5+0.997/2.)), :], \
                        color = 'brown', label = '3 sig')
        plt.fill_betweenx(pressures, \
                        x1 = temps_sort[int(len_samp*(0.5-0.95/2.)), :], \
                        x2 = temps_sort[int(len_samp*(0.5+0.95/2.)), :], \
                        color = 'orange', label = '2 sig')
        plt.fill_betweenx(pressures, \
                        x1 = temps_sort[int(len_samp*(0.5-0.68/2.)), :], \
                        x2 = temps_sort[int(len_samp*(0.5+0.68/2.)), :], \
                        color = 'red', label = '1 sig')

        plt.yscale('log')
        try:
            plt.ylim(self.rd.plot_kwargs["press_limits"])
        except:
            plt.ylim([pressures[-1], pressures[0]])
        try:
            plt.xlim(self.rd.plot_kwargs["temp_limits"])
        except:
            pass
        plt.xlabel('T (K)')
        plt.ylabel('P (bar)')
        plt.legend(loc='best')
        plt.savefig(self.output_dir + 'evaluate_'+self.retrieval_name +'/PT_envelopes.pdf')
        return 

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
