import numpy as np
from .rebin_give_width import rebin_give_width as rgw
from scipy.ndimage.filters import gaussian_filter
from astropy.io import fits

class Data:
    def __init__(self,
                 name,
                 path_to_observations = None,
                 data_resolution = None,
                 model_resolution = None,
                 external_pRT_reference = None,
                 model_generating_function = None,
                 wlen_range_micron = None,
                 scale = False,
                 wlen_bins = None,
                 photometry = False,
                 photometric_transfomation_function = None, 
                 photometry_range = None,
                 width_photometry = None):
        """
        Data class initializer
        This class stores the spectral data to be retrieved from a single instrument or observation.
        Each dataset is associated with an instance of petitRadTrans and an atmospheric model.
        The pRT instance can be overwritten, and associated with an existing pRT instance with the 
        external_pRT_reference parameter.
        This setup allows for joint or independant retrievals on multiple datasets.

        parameters
        -----------
        name : str
            Identifier for this data set.
        path_to_observations : str
            Path to observations file, including filename. This can be a txt or dat file containing the wavelength,
            flux, transit depth and error, or a fits file containing the wavelength, spectrum and covariance matrix.
        data_resolution : float
            Spectral resolution of the instrument. Optional, allows convolution of model to instrumental line width.
        model_resolution : float
            The resolution of the c-k opacity tables in pRT. This will generate a new c-k table using exo-k. The default 
            (and maximum) correlated k resolution in pRT is λ/∆λ = 1000 (R=500). Lowering the resolution will speed up the computation.
        external_pRT_instance : object
            An existing RadTrans object. Leave as none unless you're sure of what you're doing.
        model_generating_function : method
            A function, typically defined in run_definition.py that returns the model wavelength and spectrum (emission or transmission).
            This is the function that contains the physics of the model, and calls pRT in order to compute the spectrum.
        wlen_range_micron : tuple,list
            Set the wavelength range of the pRT object. Defaults to a range ±5% greater than that of the data. Must at least be 
            equal to the range of the data. 
        scale : bool
            Turn on or off scaling the data by a constant factor. Set to True if scaling the data during the retrieval.
        wlen_bins : numpy.ndarray
            Set the wavelength bins to bin the pRT model to the data. Defaults to the data bins.
        photometry : bool
            Set to True if using photometric data.
        photometric_transfomation_function : method
            Transform the photometry (account for filter transmission etc.)
        photometry_range : Tuple, numpy.ndarray
            The wavelength range of the pRT object, must be greater than the width of the photometric band. [low,high]
        width_photometry : Tuple, numpy.ndarray
            The width of the photometric bin. [low,high]
        """
        self.name = name
        self.path_to_observations = path_to_observations
        self.data_resolution = data_resolution
        self.model_resolution = model_resolution

        self.external_pRT_reference = external_pRT_reference
        self.model_generating_function = model_generating_function
        self.generate_spectrum_wlen_range_micron = wlen_range_micron
        self.covariance = None
        self.inv_cov = None
        self.flux_error = None
        self.scale = scale
        self.scale_factor = 1.0
        
        self.wlen_bins = wlen_bins
        self.photometry = photometry
        self.photometric_transfomation_function = \
            photometric_transfomation_function
        self.photometry_range = photometry_range
        self.width_photometry = width_photometry

        # Read in data
        if path_to_observations != None:
            if path_to_observations.endswith('.fits'):
                self.loadfits(path_to_observations)
            else:
                self.loadtxt(path_to_observations)
            
            self.wlen_range_pRT = [0.95 * self.wlen[0], \
                                   1.05 * self.wlen[-1]]
            # For binning later
            if not self.photometry:
                if wlen_bins != None:
                    self.wlen_bins = wlen_bins
                else:
                    self.wlen_bins = np.zeros_like(self.wlen)
                    self.wlen_bins[:-1] = np.diff(self.wlen)
                    self.wlen_bins[-1] = self.wlen_bins[-2]

            # For binning later
            self.wlen_bins = np.zeros_like(self.wlen)
            self.wlen_bins[:-1] = np.diff(self.wlen)
            self.wlen_bins[-1] = self.wlen_bins[-2]
        else:
            self.wlen_range_pRT = \
                self.generate_spectrum_wlen_range_micron
        if self.photometry:
            self.wlen_range_pRT = [self.photometry_range[0], \
                                       self.photometry_range[1]]

        # To be filled later
        self.pRT_object = None

    def loadtxt(self, path):
        """
        loadtxt
        This function reads in a .txt or .dat file containing the spectrum. Headers should be commented out with #,
        the first column must be the wavelength in micron, the second column the flux or transit depth, 
        and the final column must be the error on each data point.

        parameters
        ----------
        path : str
            Directory and filename of the data.
        """
        obs = np.genfromtxt(path,delimiter = ',')
        if np.isnan(obs).any():
            #print("Nans in " + path + ", trying different delimiter")
            obs = np.genfromtxt(path, delimiter = ' ')
        if len(obs.shape) < 2:
            #print("Incorrect shape in " + path + ", trying different delimiter")
            obs = np.genfromtxt(path)
        if obs.shape[1] != 3:
            #print("Incorrect shape in " + path + ", trying different delimiter")
            obs= np.genfromtxt(path)
        #print(obs.shape,len(obs.shape))
        self.wlen = obs[:,0]
        self.flux = obs[:,1]
        self.flux_error = obs[:,-1]

    def loadfits(self,path):
        """
        loadfits
        Load in a particular style of fits file.
        Must include extension SPECTRUM with fields WAVLENGTH, FLUX
        and COVARIANCE.
        """
        self.wlen = fits.getdata(path, 'SPECTRUM').field("WAVELENGTH")
        self.flux = fits.getdata(path, 'SPECTRUM').field("FLUX")
        self.covariance = fits.getdata(path,'SPECTRUM').field("COVARIANCE")
        self.inv_cov = np.linalg.inv(self.covariance)
        self.flux_error = np.sqrt(self.covariance.diagonal())
        return

    def get_chisq(self, wlen_model, \
                  spectrum_model, \
                  plotting):
        """
        get_chisq
        Calculate the chi square between the model and the data.

        parameters
        ----------
        wlen_model : numpy.ndarray
            The wavlengths of the model
        spectrum_model : numpy.ndarray
            The model flux in the same units as the data.
        plotting : bool
            Show test plots. 
        """
        if plotting:
            import pylab as plt
        # Convolve to data resolution
        if not self.photometry:
            # Convolve to data resolution
            if self.data_resolution != None:
                spectrum_model = self.convolve(wlen_model, \
                            spectrum_model, \
                            self.data_resolution)

            # Rebin to model observation
            flux_rebinned = rgw.rebin_give_width(wlen_model, \
                                                 spectrum_model, \
                                                 self.wlen, \
                                                 self.wlen_bins)
        else:
            flux_rebinned = \
                self.photometric_transfomation_function(wlen_model, \
                                                  spectrum_model)


        diff = (flux_rebinned - self.flux*self.scale_factor)
        f_err = self.flux_error*self.scale_factor
        logL=0.0
        if self.covariance is not None:
            logL += -1*np.dot(diff, np.dot(self.inv_cov, diff))/2.
        else:
            logL += -1*np.sum( (diff / f_err)**2. ) / 2.
        if plotting:
            plt.plot(self.wlen, flux_rebinned)
            plt.errorbar(self.wlen, \
                         self.flux*self.scale_factor, \
                         yerr = f_err, \
                         fmt = '+')
            plt.show()
        #print("LogL, " + self.name + " = "  + str(logL))
        return logL

    def convolve(self, \
                 input_wavelength, \
                 input_flux, \
                 instrument_res):
    
        # From talking to Ignas: delta lambda of resolution element
        # is FWHM of the LSF's standard deviation, hence:
        sigma_LSF = 1./instrument_res/(2.*np.sqrt(2.*np.log(2.)))

        # The input spacing of petitRADTRANS is 1e3, but just compute
        # it to be sure, or more versatile in the future.
        # Also, we have a log-spaced grid, so the spacing is constant
        # as a function of wavelength
        spacing = np.mean(2.*np.diff(input_wavelength)/ \
                          (input_wavelength[1:]+input_wavelength[:-1]))

        # Calculate the sigma to be used in the gauss filter in units
        # of input wavelength bins
        sigma_LSF_gauss_filter = sigma_LSF/spacing
    
        flux_LSF = gaussian_filter(input_flux, \
                                   sigma = sigma_LSF_gauss_filter, \
                                   mode = 'nearest')

        return flux_LSF
