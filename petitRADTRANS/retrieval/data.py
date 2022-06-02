import sys
import os
import numpy as np
import logging
from scipy.ndimage import gaussian_filter
import petitRADTRANS.nat_cst as nc
from .rebin_give_width import rebin_give_width

class Data:
    """
    This class stores the spectral data to be retrieved from a single instrument or observation.

    Each dataset is associated with an instance of petitRadTrans and an atmospheric model.
    The pRT instance can be overwritten, and associated with an existing pRT instance with the
    external_pRT_reference parameter.
    This setup allows for joint or independant retrievals on multiple datasets.

    Args:
        name : str
            Identifier for this data set.
        path_to_observations : str
            Path to observations file, including filename. This can be a txt or dat file
            containing the wavelength, flux, transit depth and error, or a fits file
            containing the wavelength, spectrum and covariance matrix.
        distance : float
            The distance to the object in cgs units. Defaults to a 10pc normalized distance.
        data_resolution : float
            Spectral resolution of the instrument. Optional, allows convolution of model to
            instrumental line width.
        model_resolution : float
            Will be ``None`` by default.  The resolution of the c-k opacity tables in pRT.
            This will generate a new c-k table using exo-k. The default (and maximum)
            correlated k resolution in pRT is :math:`\\lambda/\\Delta \\lambda > 1000` (R=500).
            Lowering the resolution will speed up the computation.
            If integer positive value, and if ``opacities == 'lbl'`` is ``True``, then this
            will sample the the high-resolution opacities at the specified resolution.
            This may be desired in the case where medium-resolution spectra are
            required with a :math:`\\lambda/\\Delta \\lambda > 1000`, but much smaller than
            :math:`10^6`, which is the resolution of the ``lbl`` mode. In this case it
            may make sense to carry out the calculations with lbl_opacity_sampling = 10e5,
            for example, and then rebinning to the final desired resolution:
            this may save time! The user should verify whether this leads to
            solutions which are identical to the rebinned results of the fiducial
            :math:`10^6` resolution. If not, this parameter must not be used.
            Note the difference between this parameter and the lbl_opacity_sampling
            parameter in the RadTrans class - the actual desired resolution should
            be set here.
        external_pRT_instance : object
            An existing RadTrans object. Leave as none unless you're sure of what you're doing.
        model_generating_function : method
            A function, typically defined in run_definition.py that returns the model wavelength and spectrum (emission or transmission).
            This is the function that contains the physics of the model, and calls pRT in order to compute the spectrum.
        wlen_range_micron : tuple,list
            Set the wavelength range of the pRT object. Defaults to a range +/-5% greater than that of the data. Must at least be
            equal to the range of the data.
        scale : bool
            Turn on or off scaling the data by a constant factor. Set to True if scaling the data during the retrieval.
        wlen_bins : numpy.ndarray
            Set the wavelength bins to bin the pRT model to the data. Defaults to the data bins.
        photometry : bool
            Set to True if using photometric data.
        photometric_transformation_function : method
            Transform the photometry (account for filter transmission etc.).
            This function must take in the wavelength and flux of a spectrum,
            and output a single photometric point (and optionally flux error).
        photometric_bin_edges : Tuple, numpy.ndarray
            The edges of the photometric bin in micron. [low,high]
        opacity_mode : str
            Should the retrieval be run using correlated-k opacities (default, 'c-k'),
            or line by line ('lbl') opacities? If 'lbl' is selected, it is HIGHLY
            recommended to set the model_resolution parameter. In general,
            'c-k' mode is recommended for retrievals of everything other than
            high-resolution (R>40000) spectra.

    """

    def __init__(self,
                 name,
                 path_to_observations = None,
                 data_resolution = None,
                 model_resolution = None,
                 distance = None,
                 external_pRT_reference = None,
                 model_generating_function = None,
                 wlen_range_micron = None,
                 scale = False,
                 scale_err = False,
                 wlen_bins = None,
                 photometry = False,
                 photometric_transformation_function = None,
                 photometric_bin_edges = None,
                 opacity_mode = 'c-k'):

        self.name = name
        self.path_to_observations = path_to_observations

        # To be filled later
        self.pRT_object = None
        self.wlen = None #: The wavelength bin centers
        self.flux = None #: The flux or transit depth
        self.flux_error = None #: The error on the flux or transit depth

        # Data file
        if not os.path.exists(path_to_observations):
            logging.error(path_to_observations + " Does not exist!")
            sys.exit(7)

        # Sanity check distance
        self.distance = distance
        if not distance:
            self.distance = 10.* nc.pc
        if self.distance < 1.0*nc.pc:
            logging.warning("Your distance is less than 1pc, are you sure you're using cgs units?")


        self.data_resolution = data_resolution
        self.model_resolution = model_resolution
        self.external_pRT_reference = external_pRT_reference
        self.model_generating_function = model_generating_function
        self.opacity_mode = opacity_mode
        if opacity_mode not in ['c-k','lbl']:
            logging.error("opacity_mode must be either 'c-k' or 'lbl'!")
            sys.exit(10)
        # Sanity check model function
        if not model_generating_function and not external_pRT_reference:
            logging.error("Please provide a model generating function or external reference for " + name + "!")
            sys.exit(8)
        if model_resolution is not None:
            if opacity_mode == 'c_k' and model_resolution > 1000:
                logging.warning("The maximum opacity for c-k mode is 1000!")
                self.model_resolution = None
            if opacity_mode == 'lbl' and model_resolution < 1000:
                logging.warning("Your resolution is lower than R=1000, it's recommended to use 'c-k' mode.")

        # Optional, covariance and scaling
        self.covariance = None
        self.inv_cov = None
        self.flux_error = None
        self.scale = scale
        self.scale_err = scale_err

        self.scale_factor = 1.0

        # Bins and photometry
        self.wlen_bins = wlen_bins
        self.photometry = photometry
        self.photometric_transformation_function = \
            photometric_transformation_function
        if photometry:
            if photometric_transformation_function is None:
                logging.error("Please provide a photometry transformation function for " + name + "!")
                sys.exit(9)
            if photometric_bin_edges is None:
                logging.error("You must include the photometric bin size if photometry is True!")
                sys.exit(9)
        self.photometry_range = wlen_range_micron
        self.width_photometry = photometric_bin_edges

        # Read in data
        if path_to_observations is not None:
            if not photometry:
                if path_to_observations.endswith('.fits'):
                    self.loadfits(path_to_observations)
                else:
                    self.loadtxt(path_to_observations)

                if wlen_range_micron is not None:
                    self.wlen_range_pRT = wlen_range_micron
                else:
                    self.wlen_range_pRT = [0.95 * self.wlen[0], \
                                    1.05 * self.wlen[-1]]

                if wlen_bins is not None:
                    self.wlen_bins = wlen_bins
                else:
                    self.wlen_bins = np.zeros_like(self.wlen)
                    self.wlen_bins[:-1] = np.diff(self.wlen)
                    self.wlen_bins[-1] = self.wlen_bins[-2]
            else:
                if wlen_range_micron is not None:
                    self.wlen_range_pRT = wlen_range_micron
                else:
                    self.wlen_range_pRT = [0.95*self.width_photometry[0],
                                            1.05*self.width_photometry[1]]
                # For binning later
                self.wlen_bins = self.width_photometry[1]-self.width_photometry[0]
                if self.data_resolution is None:
                    self.data_resolution = np.mean(self.width_photometry)/self.wlen_bins

    def loadtxt(self, path, delimiter = ',', comments = '#'):
        """
        This function reads in a .txt or .dat file containing the spectrum. Headers should be commented out with '#',
        the first column must be the wavelength in micron, the second column the flux or transit depth,
        and the final column must be the error on each data point.
        Checks will be performed to determine the correct delimiter, but the recommended format is to use a
        csv file with columns for wavlength, flux and error.

        Args:
            path : str
                Directory and filename of the data.
            delimiter : string, int
                The string used to separate values. By default, commas act as delimiter.
                An integer or sequence of integers can also be provided as width(s) of each field.
            comments : string
                The character used to indicate the start of a comment.
                All the characters occurring on a line after a comment are discarded
        """

        if self.photometry:
            return
        obs = np.genfromtxt(path,delimiter = delimiter, comments = comments)
        # Input sanity checks
        if np.isnan(obs).any():
            obs = np.genfromtxt(path, delimiter = ' ', comments = comments)
        if len(obs.shape) < 2:
            obs = np.genfromtxt(path, comments = comments)
        if obs.shape[1] != 3:
            obs= np.genfromtxt(path)

        # Warnings and errors
        if obs.shape[1] != 3:
            logging.error("Failed to properly load data in " + path + "!!!")
            sys.exit(6)
        if np.isnan(obs).any():
            logging.warning("nans present in " + path + ", please verify your data before running the retrieval!")
        self.wlen = obs[:,0]
        self.flux = obs[:,1]
        self.flux_error = obs[:,-1]

    def loadfits(self,path):
        """
        Load in a particular style of fits file.
        Must include extension SPECTRUM with fields WAVLENGTH, FLUX
        and COVARIANCE (or ERROR).

        Args:
            path : str
                Directory and filename of the data.
        """

        from astropy.io import fits
        if self.photometry:
            return
        self.wlen = fits.getdata(path, 'SPECTRUM').field("WAVELENGTH")
        self.flux = fits.getdata(path, 'SPECTRUM').field("FLUX")
        try:
            self.covariance = fits.getdata(path,'SPECTRUM').field("COVARIANCE")
            self.inv_cov = np.linalg.inv(self.covariance)

            # Note that this will only be the uncorrelated error.
            # Dot with the correlation matrix (if available) to get
            # the full error.
            try:
                self.flux_error = fits.getdata(path,'SPECTRUM').field("ERROR")
            except:
                self.flux_error = np.sqrt(self.covariance.diagonal())
        except:
            self.flux_error = fits.getdata(path,'SPECTRUM').field("ERROR")

    def set_distance(self,distance):
        """
        Sets the distance variable in the data class.
        This does NOT rescale the flux to the new distance.
        In order to rescale the flux and error, use the scale_to_distance method.

        Args:
            distance : float
                The distance to the object in cgs units.
        """

        self.distance = distance
        return self.distance

    def update_bins(self,wlens):
        self.wlen_bins = np.zeros_like(wlens)
        self.wlen_bins[:-1] = np.diff(wlens)
        self.wlen_bins[-1] = self.wlen_bins[-2]

    def scale_to_distance(self, new_dist):
        """
        Updates the distance variable in the data class.
        This will rescale the flux to the new distance.

        Args:
            new_dist : float
                The distance to the object in cgs units.
        """

        scale = (self.distance/new_dist)**2
        self.flux *= scale
        if self.covariance is not None:
            self.covariance *= scale**2
            self.inv_cov = np.linalg.inv(self.covariance)
            self.flux_error = np.sqrt(self.covariance.diagonal())
        else:
            self.flux_error *= scale
        self.distance = new_dist
        return scale

    def get_chisq(self, wlen_model, \
                  spectrum_model, \
                  plotting):
        """
        Calculate the chi square between the model and the data.

        Args:
            wlen_model : numpy.ndarray
                The wavlengths of the model
            spectrum_model : numpy.ndarray
                The model flux in the same units as the data.
            plotting : bool
                Show test plots.

        Returns:
            logL : float
                The log likelihood of the model given the data.
        """

        if plotting:
            import matplotlib.pyplot as plt
        # Convolve to data resolution
        if self.data_resolution is not None:
            spectrum_model = self.convolve(wlen_model,
                                           spectrum_model,
                                           self.data_resolution)

        if not self.photometry:
            # Rebin to model observation
            flux_rebinned = rebin_give_width(wlen_model,
                                             spectrum_model,
                                             self.wlen,
                                             self.wlen_bins)
        else:
            flux_rebinned = \
                self.photometric_transformation_function(wlen_model,
                                                         spectrum_model)
            # species spectrum_to_flux functions return (flux,error)
            if isinstance(flux_rebinned,(tuple,list)):
                flux_rebinned = flux_rebinned[0]


        diff = (flux_rebinned - self.flux*self.scale_factor)
        f_err = self.flux_error
        if self.scale_err:
            f_err = self.flux_error*self.scale_factor
        logL=0.0
        if self.covariance is not None:
            logL += -1*np.dot(diff, np.dot(self.inv_cov, diff))/2.
        else:
            logL += -1*np.sum( (diff / f_err)**2. ) / 2.
        if plotting:
            if not self.photometry:
                plt.plot(self.wlen, flux_rebinned)
                plt.errorbar(self.wlen,
                             self.flux*self.scale_factor,
                             yerr = f_err,
                             fmt = '+')
                plt.show()
        return logL

    def convolve(self, \
                 input_wavelength, \
                 input_flux, \
                 instrument_res):
        """
        This function convolves a model spectrum to the instrumental wavelength
        using the provided data_resolution
        Args:
            input_wavelength : numpy.ndarray
                The wavelength grid of the model spectrum
            input_flux : numpy.ndarray
                The flux as computed by the model
            instrument_res : float
                :math:`\\lambda/\\Delta \\lambda`, the width of the gaussian kernel to convolve with the model spectrum.

        Returns:
            flux_LSF
                The convolved spectrum.
        """

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
