import copy
import os
import pickle
import sys
import warnings

import h5py
import numpy as np
import pyvo
from astropy.table.table import Table
from scipy.ndimage import gaussian_filter1d

from petitRADTRANS import nat_cst as nc
from petitRADTRANS.ccf.pipeline import simple_pipeline
from petitRADTRANS.ccf.utils import calculate_uncertainty, module_dir, dict2hdf5, hdf52dict
from petitRADTRANS.fort_rebin import fort_rebin as fr
from petitRADTRANS.phoenix import get_PHOENIX_spec
from petitRADTRANS.physics import doppler_shift, guillot_global, guillot_metallic_temperature_profile
from petitRADTRANS.radtrans import Radtrans
from petitRADTRANS.retrieval import RetrievalConfig
from petitRADTRANS.retrieval.util import calc_MMW, uniform_prior

# from petitRADTRANS.config import petitradtrans_config

# planet_models_directory = os.path_input_data.abspath(Path.home()) + os.path_input_data.sep + 'Downloads' + os.path_input_data.sep + 'tmp' #os.path_input_data.abspath(os.path_input_data.dirname(__file__) + os.path_input_data.sep + 'planet_models')
planet_models_directory = os.path.abspath(os.path.dirname(__file__) + os.path.sep + 'planet_models')  # TODO change that


# planet_models_directory = petitradtrans_config['Paths']['pRT_outputs_path']

class BaseSpectralModel:
    # TODO ideally this should inherit from Radtrans, but it cannot be done right now because when Radtrans is init, it takes ages to load opacity data
    def __init__(self, pressures,
                 line_species=None, rayleigh_species=None, continuum_opacities=None, cloud_species=None,
                 opacity_mode='lbl', do_scat_emis=True, lbl_opacity_sampling=1,
                 temperatures=None, mass_mixing_ratios=None, mean_molar_masses=None,
                 wavelengths_boundaries=None, wavelengths=None, transit_radii=None, spectral_radiosities=None,
                 times=None,
                 **model_parameters):
        # Atmosphere/Radtrans parameters
        self.pressures = pressures
        self.lbl_opacity_sampling = lbl_opacity_sampling
        self.do_scat_emis = do_scat_emis
        self.opacity_mode = opacity_mode

        if line_species is None:
            self.line_species = []
        else:
            self.line_species = line_species

        if rayleigh_species is None:
            self.rayleigh_species = []
        else:
            self.rayleigh_species = rayleigh_species

        if continuum_opacities is None:
            self.continuum_opacities = []
        else:
            self.continuum_opacities = continuum_opacities

        if cloud_species is None:
            self.cloud_species = []
        else:
            self.cloud_species = cloud_species

        # Spectrum generation base parameters
        self.temperatures = temperatures
        self.mass_mixing_ratios = mass_mixing_ratios
        self.mean_molar_masses = mean_molar_masses

        # Spectrum parameters
        self.wavelengths = wavelengths
        self.transit_radii = transit_radii
        self.spectral_radiosities = spectral_radiosities

        # Time-dependent parameters
        self.times = times

        # Other model parameters
        self.model_parameters = model_parameters

        # Wavelength boundaries
        if wavelengths_boundaries is None:  # calculate the optimal wavelength boundaries
            if self.wavelengths is not None:
                self.model_parameters['output_wavelengths'] = copy.deepcopy(self.wavelengths)
            elif 'output_wavelengths' not in self.model_parameters:
                raise TypeError(f"missing required argument "
                                f"'wavelengths_boundaries', add this argument to manually set the boundaries or "
                                f"add keyword argument 'output_wavelengths' to set the boundaries automatically")

            # Calculate relative velocity to take Doppler shift into account
            if 'relative_velocities' in self.model_parameters \
                    or 'orbital_longitudes' in self.model_parameters \
                    or 'orbital_phases' in self.model_parameters:
                self.model_parameters['relative_velocities'], \
                    self.model_parameters['planet_max_radial_orbital_velocity'], \
                    self.model_parameters['orbital_longitudes'] = \
                    self.__calculate_relative_velocities_wrap(
                        calculate_max_radial_orbital_velocity_function=self.calculate_max_radial_orbital_velocity,
                        calculate_relative_velocities_function=self.calculate_relative_velocities,
                        **self.model_parameters
                    )
            else:
                self.model_parameters['relative_velocities'] = np.zeros(1)

            wavelengths_boundaries = self.get_optimal_wavelength_boundaries()

        self.wavelengths_boundaries = wavelengths_boundaries

    @staticmethod
    def __calculate_relative_velocities_wrap(calculate_max_radial_orbital_velocity_function,
                                             calculate_relative_velocities_function, **kwargs):
        if 'planet_max_radial_orbital_velocity' in kwargs:
            if kwargs['planet_max_radial_orbital_velocity'] is None:
                planet_max_radial_orbital_velocity = calculate_max_radial_orbital_velocity_function(
                    **kwargs
                )
            else:
                planet_max_radial_orbital_velocity = kwargs['planet_max_radial_orbital_velocity']
        else:
            planet_max_radial_orbital_velocity = calculate_max_radial_orbital_velocity_function(
                **kwargs
            )

        kwargs['planet_max_radial_orbital_velocity'] = planet_max_radial_orbital_velocity

        if 'orbital_phases' in kwargs and 'orbital_longitudes' not in kwargs:
            if kwargs['orbital_phases'] is not None:
                orbital_longitudes = np.rad2deg(2 * np.pi * kwargs['orbital_phases'])
            else:
                orbital_longitudes = np.zeros(1)
        else:
            orbital_longitudes = kwargs['orbital_longitudes']

        kwargs['orbital_longitudes'] = orbital_longitudes

        if 'relative_velocities' in kwargs:
            if kwargs['relative_velocities'] is None:
                relative_velocities = calculate_relative_velocities_function(
                    **kwargs
                )
            else:
                relative_velocities = kwargs['relative_velocities']
        else:
            relative_velocities = calculate_relative_velocities_function(
                **kwargs
            )

        return relative_velocities, planet_max_radial_orbital_velocity, orbital_longitudes

    @staticmethod
    def calculate_mass_mixing_ratios(pressures, **kwargs):
        """Template for mass mixing ratio profile function.
        Here, generate iso-abundant mass mixing ratios profiles.

        Args:
            pressures: (bar) pressures of the temperature profile
            **kwargs: other parameters needed to generate the temperature profile

        Returns:
            A 1D-array containing the temperatures as a function of pressures
        """
        return {
            species: mass_mixing_ratio * np.ones(np.size(pressures))
            for species, mass_mixing_ratio in kwargs['imposed_mass_mixing_ratios'].items()
        }

    @staticmethod
    def calculate_max_radial_orbital_velocity(star_mass, semi_major_axis, **kwargs):
        return Planet.calculate_orbital_velocity(
            star_mass=star_mass,
            semi_major_axis=semi_major_axis
        )

    @staticmethod
    def calculate_mean_molar_masses(mass_mixing_ratios, **kwargs):
        return calc_MMW(mass_mixing_ratios)

    @staticmethod
    def calculate_model_parameters(**kwargs):
        """Function to update model parameters.

        Args:
            **kwargs: parameters to update

        Returns:
            Updated parameters
        """
        # This function can be expanded to include anything
        if 'star_spectral_radiosities' in kwargs:
            if kwargs['star_spectral_radiosities'] is None:
                kwargs['star_spectral_radiosities'] = BaseSpectralModel.calculate_star_spectral_radiosity(
                    **kwargs
                )

        if 'planet_max_radial_orbital_velocity' in kwargs \
                or 'relative_velocities' in kwargs \
                or 'orbital_longitudes' in kwargs\
                or 'orbital_phases' in kwargs:
            kwargs['relative_velocities'], \
                kwargs['planet_max_radial_orbital_velocity'], \
                kwargs['orbital_longitudes'] = \
                BaseSpectralModel.__calculate_relative_velocities_wrap(
                    calculate_max_radial_orbital_velocity_function=
                    BaseSpectralModel.calculate_max_radial_orbital_velocity,
                    calculate_relative_velocities_function=
                    BaseSpectralModel.calculate_relative_velocities,
                    **kwargs
                )

        if 'line_species' in kwargs:
            del kwargs['line_species']

        if 'pressures' in kwargs:
            del kwargs['pressures']

        if 'wavelengths' in kwargs:
            del kwargs['wavelengths']

        return kwargs

    @staticmethod
    def calculate_relative_velocities(orbital_longitudes, planet_orbital_inclination=90.0,
                                      planet_max_radial_orbital_velocity=None, system_observer_radial_velocities=0.0,
                                      planet_rest_frame_shift=0.0, **kwargs):
        if planet_max_radial_orbital_velocity is None:
            planet_max_radial_orbital_velocity = BaseSpectralModel.calculate_max_radial_orbital_velocity(
                **kwargs
            )

        if -sys.float_info.min < planet_max_radial_orbital_velocity < sys.float_info.min:
            relative_velocities = 0.0
        else:
            relative_velocities = Planet.calculate_planet_radial_velocity(
                planet_max_radial_orbital_velocity=planet_max_radial_orbital_velocity,
                planet_orbital_inclination=planet_orbital_inclination,
                orbital_longitude=orbital_longitudes
            )

        relative_velocities += system_observer_radial_velocities + planet_rest_frame_shift  # planet + system velocity

        return relative_velocities

    @staticmethod
    def calculate_spectral_parameters(temperature_profile_function, mass_mixing_ratios_function,
                                      mean_molar_masses_function, spectral_parameters_function, **kwargs):
        """Calculate the temperature profile, the mass mixing ratios, the mean molar masses and other parameters
        required for spectral calculation.

        This function define how these parameters are calculated and how they are combined.

        Args:
            temperature_profile_function:
            mass_mixing_ratios_function:
            mean_molar_masses_function:
            spectral_parameters_function:
            **kwargs:

        Returns:

        """
        temperatures = temperature_profile_function(
            **kwargs
        )

        mass_mixing_ratios = mass_mixing_ratios_function(
            **kwargs
        )

        mean_molar_masses = mean_molar_masses_function(
            mass_mixing_ratios=mass_mixing_ratios,
            **kwargs
        )

        model_parameters = spectral_parameters_function(
            **kwargs
        )

        return temperatures, mass_mixing_ratios, mean_molar_masses, model_parameters

    @staticmethod
    def calculate_spectral_radiosity_spectrum(radtrans: Radtrans, temperatures, mass_mixing_ratios,
                                              planet_surface_gravity, mean_molar_mass, star_effective_temperature,
                                              star_spectral_radiosities, cloud_pressure=None, **kwargs):
        """Wrapper of Radtrans.calc_flux that output wavelengths in um and spectral radiosity in erg.s-1.cm-2.sr-1/cm.
        # TODO move to Radtrans or outside of object
        Args:
            radtrans:
            temperatures:
            mass_mixing_ratios:
            planet_surface_gravity:
            mean_molar_mass:
            star_effective_temperature:
            star_spectral_radiosities:
            cloud_pressure:

        Returns:

        """
        # Calculate the spectrum
        # TODO units in native calc_flux units for more performances?
        radtrans.calc_flux(
            temp=temperatures,
            abunds=mass_mixing_ratios,
            gravity=planet_surface_gravity,
            mmw=mean_molar_mass,
            Tstar=star_effective_temperature,
            Pcloud=cloud_pressure,
            stellar_intensity=BaseSpectralModel.radiosity_erg_cm2radiosity_erg_hz(
                star_spectral_radiosities, nc.c / radtrans.freq  # Hz to cm
            )
            # **kwargs  # TODO add kwargs once arguments names are made unambiguous
        )

        # Transform the outputs into the units of our data
        spectral_radiosity = BaseSpectralModel.radiosity_erg_hz2radiosity_erg_cm(radtrans.flux, radtrans.freq)
        wavelengths = BaseSpectralModel.hz2um(radtrans.freq)

        return wavelengths, spectral_radiosity

    @staticmethod
    def calculate_star_spectral_radiosity(wavelengths, star_effective_temperature, star_radius, semi_major_axis,
                                          **kwargs):
        star_data = get_PHOENIX_spec(star_effective_temperature)

        star_radiosities = star_data[:, 1]
        star_wavelengths = star_data[:, 0] * 1e4  # cm to um

        star_radiosities = fr.rebin_spectrum(star_wavelengths, star_radiosities, wavelengths)
        star_radiosities *= (star_radius / semi_major_axis) ** 2 / np.pi

        star_radiosities = BaseSpectralModel.radiosity_erg_hz2radiosity_erg_cm(
            star_radiosities, BaseSpectralModel.um2hz(wavelengths)
        )

        return star_radiosities

    @staticmethod
    def calculate_temperature_profile(pressures, **kwargs):
        """Template for temperature profile function.
        Here, generate an isothermal temperature profile.

        Args:
            pressures: (bar) pressures of the temperature profile
            **kwargs: other parameters needed to generate the temperature profile

        Returns:
            A 1D-array containing the temperatures as a function of pressures
        """
        return np.ones(np.size(pressures)) * kwargs['temperature']

    @staticmethod
    def calculate_transit_spectrum(radtrans: Radtrans, temperatures, mass_mixing_ratios, mean_molar_masses,
                                   planet_surface_gravity, reference_pressure, planet_radius, cloud_pressure=None,
                                   **kwargs):
        """Wrapper of Radtrans.calc_transm that output wavelengths in um and transit radius in cm.
        # TODO move to Radtrans or outside of object

        Args:
            radtrans:
            temperatures:
            mass_mixing_ratios:
            planet_surface_gravity:
            mean_molar_masses:
            reference_pressure:
            planet_radius:
            cloud_pressure:

        Returns:

        """
        # Calculate the spectrum
        radtrans.calc_transm(
            temp=temperatures,
            abunds=mass_mixing_ratios,
            gravity=planet_surface_gravity,
            mmw=mean_molar_masses,
            P0_bar=reference_pressure,
            R_pl=planet_radius,
            Pcloud=cloud_pressure,
            # **kwargs  # TODO add kwargs once arguments names are made unambiguous
        )

        # Convert into more useful units
        planet_transit_radius = copy.copy(radtrans.transm_rad)
        wavelengths = BaseSpectralModel.hz2um(radtrans.freq)

        return wavelengths, planet_transit_radius

    @staticmethod
    def convolve(input_wavelengths, input_spectrum, new_resolving_power, **kwargs):
        """Convolve a spectrum to a new resolving power.
        The spectrum is convolved using a Gaussian filter with a standard deviation ~ R_in / R_new input wavelengths bins.
        The spectrum must have a constant resolving power as a function of wavelength.
        The input resolving power is given by:
            lambda / Delta_lambda
        where lambda is the center of a wavelength bin and Delta_lambda the difference between the edges of the bin.

        Args:
            input_wavelengths: (cm) wavelengths of the input spectrum
            input_spectrum: input spectrum
            new_resolving_power: resolving power of output spectrum  # TODO change to actual resolution of output wvl

        Returns:
            convolved_spectrum: the convolved spectrum at the new resolving power
        """
        # Compute resolving power of the model
        # In petitRADTRANS, the wavelength grid is log-spaced, so the resolution is constant as a function of wavelength
        model_resolving_power = np.mean(
            (input_wavelengths[1:] + input_wavelengths[:-1]) / (2 * np.diff(input_wavelengths))
        )

        # Calculate the sigma to be used in the gauss filter in units of input wavelength bins
        # Delta lambda of resolution element is the FWHM of the instrument's LSF (here: a gaussian)
        sigma_lsf_gauss_filter = model_resolving_power / new_resolving_power / (2 * np.sqrt(2 * np.log(2)))

        convolved_spectrum = gaussian_filter1d(
            input=input_spectrum,
            sigma=sigma_lsf_gauss_filter,
            mode='reflect'
        )

        return convolved_spectrum

    @staticmethod
    def cnvl2(y, filter):
        # TODO implement sliding convolution
        yc = np.zeros(y.size + filter.shape[-1] - 1)
        yp = copy.copy(yc)
        filterc = np.zeros((yc.size, filter.shape[-1]))
        yp[:y.size] = y
        filterc[int(filter.shape[-1] / 2):y.size + int(filter.shape[-1] / 2), :] = filter
        filterc[y.size + int(filter.shape[-1] / 2):, :] = filter[-1]
        filterc[:int(filter.shape[-1] / 2), :] = filter[0]

        for i, yy in enumerate(yp):
            for j, g in enumerate(filterc[i, :np.min((i, filterc.shape[-1]))]):
                yc[i] += yp[i - j] * g

        return yc[int(filter.shape[-1] / 2):y.size + int(filter.shape[-1] / 2)]

    def get_instrument_model(self, wavelengths, spectrum, relative_velocities=None, shift=False, convolve=False, rebin=False):
        if shift:
            if 'relative_velocities' not in self.model_parameters:
                raise TypeError(f"missing required parameter 'relative_velocities' for shifting")

            relative_velocities_tmp = copy.deepcopy(self.model_parameters['relative_velocities'])
            del self.model_parameters['relative_velocities']

            if relative_velocities is None:
                relative_velocities = copy.deepcopy(relative_velocities_tmp)

            wavelengths = self.shift_wavelengths(
                wavelengths_rest=wavelengths,
                relative_velocities=relative_velocities,
                **self.model_parameters
            )

            self.model_parameters['relative_velocities'] = relative_velocities_tmp

        if convolve:
            if np.ndim(wavelengths) <= 1:
                spectrum = self.convolve(
                    input_wavelengths=wavelengths,
                    input_spectrum=spectrum,
                    **self.model_parameters
                )
            else:
                spectrum = self.convolve(
                    input_wavelengths=wavelengths[0],  # assuming Doppler shifting doesn't change the resolving power
                    input_spectrum=spectrum,
                    **self.model_parameters
                )

        if rebin:
            if np.ndim(wavelengths) <= 1:
                wavelengths_tmp, spectrum = self.rebin_spectrum(
                    input_wavelengths=wavelengths,
                    input_spectrum=spectrum,
                    **self.model_parameters
                )
            elif np.ndim(wavelengths) == 2:
                spectrum_tmp = []
                wavelengths_tmp = None

                for i, wavelength_shift in enumerate(wavelengths):
                    spectrum_tmp.append([])
                    wavelengths_tmp, spectrum_tmp[-1] = self.rebin_spectrum(
                        input_wavelengths=wavelength_shift,
                        input_spectrum=spectrum,
                        **self.model_parameters
                    )

                spectrum = np.array(spectrum_tmp)

                if np.ndim(spectrum) == 3:
                    spectrum = np.moveaxis(spectrum, 0, 1)
                elif np.ndim(spectrum) > 3:
                    raise ValueError(f"spectrum must have at most 3 dimensions, but has {np.ndim(spectrum)}")
            else:
                raise ValueError(f"argument 'wavelength' must have at most 2 dimensions, "
                                 f"but has {np.ndim(wavelengths)}")

            wavelengths = wavelengths_tmp

        return wavelengths, spectrum

    def get_optimal_wavelength_boundaries(self, output_wavelengths=None, relative_velocities=None):
        """Return the optimal wavelength boundaries for rebin on output wavelengths.
        This minimises the number of wavelengths to load and over which to calculate the spectra.
        Doppler shifting is also taken into account.

        The SpectralModel must have in its model_parameters keys:
            -  'output_wavelengths': (um) the wavelengths to rebin to

        The SpectralModel can have in its model_parameters keys:
            - 'relative_velocities' (cm.s-1) the velocities of the source relative to the observer, in that case the
                wavelength range is increased to take into account Doppler shifting

        Returns:
            rebin_required_interval: (um) the optimal wavelengths boundaries for the spectrum
        """
        if output_wavelengths is None:
            output_wavelengths = self.model_parameters['output_wavelengths']

        if relative_velocities is None and 'relative_velocities' in self.model_parameters:
            relative_velocities = self.model_parameters['relative_velocities']

        # Re-bin requirement is an interval half a bin larger then re-binning interval
        wavelengths_flat = np.array(output_wavelengths).flatten()
        rebin_required_interval = [
            wavelengths_flat[0]
            - (wavelengths_flat[1] - wavelengths_flat[0]) / 2,
            wavelengths_flat[-1]
            + (wavelengths_flat[-1] - wavelengths_flat[-2]) / 2,
        ]

        # Take Doppler shifting into account
        rebin_required_interval_shifted = copy.copy(rebin_required_interval)

        if relative_velocities is not None:
            if 'relative_velocities' in self.model_parameters:
                relative_velocities_tmp = copy.deepcopy(self.model_parameters['relative_velocities'])
                del self.model_parameters['relative_velocities']  # tmp rm parameters to prevent multiple argument def
            else:
                relative_velocities_tmp = None

            rebin_required_interval_shifted[0] = self.shift_wavelengths(
                wavelengths_rest=np.array([rebin_required_interval[0]]),
                relative_velocities=np.array([
                    -np.max(relative_velocities)
                ]),
                **self.model_parameters
            )[0][0]

            rebin_required_interval_shifted[1] = self.shift_wavelengths(
                wavelengths_rest=np.array([rebin_required_interval[1]]),
                relative_velocities=np.array([
                    -np.min(relative_velocities)
                ]),
                **self.model_parameters
            )[0][0]

            if relative_velocities_tmp is not None:
                self.model_parameters['relative_velocities'] = relative_velocities_tmp

        # Ensure that non-shifted spectrum can still be re-binned
        rebin_required_interval[0] = np.min((rebin_required_interval_shifted[0], rebin_required_interval[0]))
        rebin_required_interval[1] = np.max((rebin_required_interval_shifted[1], rebin_required_interval[1]))

        # Satisfy re-bin requirement by increasing the range by the smallest possible significant value
        rebin_required_interval[0] -= 10 ** (np.floor(np.log10(rebin_required_interval[0])) - sys.float_info.dig)
        rebin_required_interval[1] += 10 ** (np.floor(np.log10(rebin_required_interval[1])) - sys.float_info.dig)

        return rebin_required_interval

    def get_parameters_dict(self):
        parameters_dict = {}

        for key, value in self.__dict__.items():
            if key == 'model_parameters':  # model_parameters is a dictionary, extract its values
                for parameter, parameter_value in value.items():
                    parameters_dict[parameter] = copy.copy(parameter_value)
            else:
                parameters_dict[key] = copy.copy(value)

        return parameters_dict

    def get_radtrans(self):
        """Return the Radtrans object corresponding to this SpectrumModel."""
        return self.init_radtrans(
            wavelengths_boundaries=self.wavelengths_boundaries,
            pressures=self.pressures,
            line_species=self.line_species,
            rayleigh_species=self.rayleigh_species,
            continuum_opacities=self.continuum_opacities,
            cloud_species=None,
            opacity_mode=self.opacity_mode,
            do_scat_emis=self.do_scat_emis,
            lbl_opacity_sampling=self.lbl_opacity_sampling
        )

    @staticmethod
    def get_reduced_spectrum(spectrum, pipeline, **kwargs):
        return pipeline(spectrum, **kwargs)

    def get_spectral_calculation_parameters(self, pressures=None, wavelengths=None, **kwargs):
        if pressures is None:
            pressures = self.pressures

        if wavelengths is None:
            wavelengths = self.wavelengths

        return self.calculate_spectral_parameters(
            temperature_profile_function=self.calculate_temperature_profile,
            mass_mixing_ratios_function=self.calculate_mass_mixing_ratios,
            mean_molar_masses_function=self.calculate_mean_molar_masses,
            spectral_parameters_function=self.calculate_model_parameters,
            pressures=pressures,
            wavelengths=wavelengths,
            **kwargs
        )

    def get_spectral_radiosity_spectrum_model(self, radtrans: Radtrans, parameters):
        self.wavelengths, self.spectral_radiosities = self.calculate_spectral_radiosity_spectrum(
            radtrans=radtrans,
            temperatures=self.temperatures,
            mass_mixing_ratios=self.mass_mixing_ratios,
            mean_molar_mass=self.mean_molar_masses,
            **parameters
        )

        return self.wavelengths, self.spectral_radiosities

    def get_spectrum_model(self, radtrans: Radtrans, mode='emission', parameters=None, update_parameters=False,
                           deformation_matrix=None, noise_matrix=None,
                           scale=False, shift=False, convolve=False, rebin=False, reduce=False):
        if parameters is None:
            parameters = self.model_parameters

        if update_parameters:
            self.update_spectral_calculation_parameters(
                radtrans=radtrans,
                **parameters
            )

            self.model_parameters['mode'] = mode
            self.model_parameters['deformation_matrix'] = deformation_matrix
            self.model_parameters['noise_matrix'] = noise_matrix
            self.model_parameters['scale'] = scale
            self.model_parameters['shift'] = shift
            self.model_parameters['convolve'] = convolve
            self.model_parameters['rebin'] = rebin
            self.model_parameters['reduce'] = reduce

            parameters = copy.deepcopy(self.model_parameters)

        if mode == 'emission':
            self.wavelengths, self.spectral_radiosities = self.get_spectral_radiosity_spectrum_model(
                radtrans=radtrans,
                parameters=parameters
            )
            spectrum = copy.copy(self.spectral_radiosities)
        elif mode == 'transmission':
            self.wavelengths, self.transit_radii = self.get_transit_spectrum_model(
                radtrans=radtrans,
                parameters=parameters
            )
            spectrum = copy.copy(self.transit_radii)
        else:
            raise ValueError(f"mode must be 'emission' or 'transmission', not '{mode}'")

        wavelengths = copy.copy(self.wavelengths)

        wavelengths, spectrum = self.get_instrument_model(
            wavelengths=wavelengths,
            spectrum=spectrum,
            shift=shift,
            convolve=convolve,
            rebin=rebin
        )

        if scale:
            if mode == 'emission':  # shift the star spectrum as well for scaling
                if 'star_observed_spectral_radiosities' not in parameters:
                    missing = []

                    if 'star_spectral_radiosities' not in parameters:
                        missing.append('star_spectral_radiosities')

                    if shift:
                        if 'system_observer_radial_velocities' not in parameters:
                            if 'relative_velocities' in parameters:
                                parameters['system_observer_radial_velocities'] = \
                                    np.zeros(parameters['relative_velocities'].shape)
                            else:
                                missing.append('system_observer_radial_velocities')

                    if len(missing) > 0:
                        joint = "', '".join(missing)

                        raise TypeError(f"missing {len(missing)} parameters for scaling: '{joint}'")

                    _, parameters['star_observed_spectral_radiosities'] = self.get_instrument_model(
                        wavelengths=copy.copy(self.wavelengths),
                        spectrum=parameters['star_spectral_radiosities'],
                        relative_velocities=parameters['system_observer_radial_velocities'],
                        shift=shift,
                        convolve=convolve,
                        rebin=rebin
                    )

                    if update_parameters:
                        self.model_parameters['star_observed_spectral_radiosities'] = \
                            copy.deepcopy(parameters['star_observed_spectral_radiosities'])

            spectrum = self.scale_spectrum(
                spectrum=spectrum,
                **parameters
            )

        if deformation_matrix is not None:
            spectrum *= deformation_matrix

        if noise_matrix is not None:
            spectrum += noise_matrix

        if reduce:
            spectrum, parameters['reduction_matrix'], parameters['reduced_uncertainties'] = \
                self.get_reduced_spectrum(
                    spectrum=spectrum,
                    pipeline=self.pipeline,
                    wavelengths=wavelengths,
                    **parameters
                )
        else:
            parameters['reduction_matrix'] = np.ones(spectrum.shape)

            if 'data_uncertainties' in parameters:
                parameters['reduced_uncertainties'] = \
                    copy.deepcopy(parameters['data_uncertainties'])
            else:
                parameters['reduced_uncertainties'] = None

        if update_parameters:
            self.model_parameters['reduction_matrix'] = parameters['reduction_matrix']
            self.model_parameters['reduced_uncertainties'] = parameters['reduced_uncertainties']

        return wavelengths, spectrum

    def get_transit_spectrum_model(self, radtrans: Radtrans, parameters):
        self.wavelengths, self.transit_radii = self.calculate_transit_spectrum(
            radtrans=radtrans,
            temperatures=self.temperatures,
            mass_mixing_ratios=self.mass_mixing_ratios,
            mean_molar_masses=self.mean_molar_masses,
            **parameters
        )

        return self.wavelengths, self.transit_radii

    @staticmethod
    def hz2um(frequency):
        return nc.c / frequency * 1e4  # cm to um

    @staticmethod
    def init_radtrans(wavelengths_boundaries, pressures,
                      line_species=None, rayleigh_species=None, continuum_opacities=None, cloud_species=None,
                      opacity_mode='lbl', do_scat_emis=True, lbl_opacity_sampling=1):
        print('Generating atmosphere...')

        radtrans = Radtrans(
            line_species=line_species,
            rayleigh_species=rayleigh_species,
            continuum_opacities=continuum_opacities,
            cloud_species=cloud_species,
            wlen_bords_micron=wavelengths_boundaries,
            mode=opacity_mode,
            do_scat_emis=do_scat_emis,
            lbl_opacity_sampling=lbl_opacity_sampling,
            pressures=pressures
        )

        return radtrans

    @classmethod
    def load(cls, filename):
        new_spectrum_model = cls(pressures=None, wavelengths_boundaries=[0.0, 0.0])

        with h5py.File(filename, 'r') as f:
            new_spectrum_model.__dict__ = hdf52dict(f)

        return new_spectrum_model

    @staticmethod
    def pipeline(spectrum):
        """Simplistic pipeline model. Do nothing.
        To be updated when initializing an instance of retrieval model.

        Args:
            spectrum: a spectrum

        Returns:
            spectrum: the spectrum reduced by the pipeline
        """
        return spectrum

    @staticmethod
    def radiosity_erg_cm2radiosity_erg_hz(radiosity_erg_cm, wavelength):
        """
        Convert a radiosity from erg.s-1.cm-2.sr-1/cm to erg.s-1.cm-2.sr-1/Hz at a given wavelength.
        Steps:
            [cm] = c[cm.s-1] / [Hz]
            => d[cm]/d[Hz] = d(c / [Hz])/d[Hz]
            => d[cm]/d[Hz] = c / [Hz]**2
            integral of flux must be conserved: radiosity_erg_cm * d[cm] = radiosity_erg_hz * d[Hz]
            radiosity_erg_hz = radiosity_erg_cm * d[cm]/d[Hz]
            => radiosity_erg_hz = radiosity_erg_cm * wavelength**2 / c

        Args:
            radiosity_erg_cm: (erg.s-1.cm-2.sr-1/cm)
            wavelength: (cm)

        Returns:
            (erg.s-1.cm-2.sr-1/cm) the radiosity in converted units
        """
        return radiosity_erg_cm * wavelength ** 2 / nc.c

    @staticmethod
    def radiosity_erg_hz2radiosity_erg_cm(radiosity_erg_hz, frequency):
        """
        Convert a radiosity from erg.s-1.cm-2.sr-1/Hz to erg.s-1.cm-2.sr-1/cm at a given frequency.
        Steps:
            [cm] = c[cm.s-1] / [Hz]
            => d[cm]/d[Hz] = d(c / [Hz])/d[Hz]
            => d[cm]/d[Hz] = c / [Hz]**2
            => d[Hz]/d[cm] = [Hz]**2 / c
            integral of flux must be conserved: radiosity_erg_cm * d[cm] = radiosity_erg_hz * d[Hz]
            radiosity_erg_cm = radiosity_erg_hz * d[Hz]/d[cm]
            => radiosity_erg_cm = radiosity_erg_hz * frequency**2 / c

        Args:
            radiosity_erg_hz: (erg.s-1.cm-2.sr-1/Hz)
            frequency: (Hz)

        Returns:
            (erg.s-1.cm-2.sr-1/cm) the radiosity in converted units
        """
        return radiosity_erg_hz * frequency ** 2 / nc.c

    @staticmethod
    def rebin_spectrum(input_wavelengths, input_spectrum, output_wavelengths, **kwargs):
        if np.ndim(output_wavelengths) <= 1:
            return output_wavelengths, fr.rebin_spectrum(input_wavelengths, input_spectrum, output_wavelengths)
        elif np.ndim(output_wavelengths) == 2:
            spectra = []
            lengths = []

            for wavelengths in output_wavelengths:
                spectra.append(fr.rebin_spectrum(input_wavelengths, input_spectrum, wavelengths))
                lengths.append(spectra[-1].size)

            if np.all(np.array(lengths) == lengths[0]):
                spectra = np.array(spectra)
            else:
                spectra = np.array(spectra, dtype=object)

            return output_wavelengths, spectra
        else:
            raise ValueError(f"parameter 'output_wavelengths' must have at most 2 dimensions, "
                             f"but has {np.ndim(output_wavelengths)}")

    def save(self, file):
        with h5py.File(file, 'w') as f:
            dict2hdf5(
                dictionary=self.__dict__,
                hdf5_file=f
            )
            f.create_dataset(
                name='units',
                data='pressures are in bar, radiosities are in erg.s-1.cm-2.sr-1/cm, wavelengths are in um, '
                     'otherwise all other units are in CGS'
            )

    @staticmethod
    def scale_spectrum(spectrum, star_radius, planet_radius=None, star_observed_spectral_radiosities=None,
                       mode='emission', **kwargs):
        if mode == 'emission':
            if planet_radius is None or star_observed_spectral_radiosities is None:
                missing = []

                if planet_radius is None:
                    missing.append('planet_radius')

                if star_observed_spectral_radiosities is None:
                    missing.append('star_spectral_radiosities')

                joint = "', '".join(missing)

                raise TypeError(f"missing {len(missing)} positional arguments: '{joint}'")

            return 1 + spectrum / star_observed_spectral_radiosities * (planet_radius / star_radius) ** 2
        elif mode == 'transmission':
            return 1 - (spectrum / star_radius) ** 2
        else:
            raise ValueError(f"mode must be 'emission' or 'transmission', not '{mode}'")

    @staticmethod
    def shift_wavelengths(wavelengths_rest, relative_velocities, **kwargs):
        wavelengths_shift = np.zeros((relative_velocities.size, wavelengths_rest.size))

        for i, relative_velocity in enumerate(relative_velocities):
            wavelengths_shift[i] = doppler_shift(wavelengths_rest, relative_velocity)

        return wavelengths_shift

    @staticmethod
    def um2hz(wavelength):
        return nc.c / (wavelength * 1e-4)  # um to cm

    def update_spectral_calculation_parameters(self, radtrans: Radtrans, **kwargs):
        self.temperatures, self.mass_mixing_ratios, self.mean_molar_masses, self.model_parameters = \
            self.get_spectral_calculation_parameters(
                pressures=radtrans.press * 1e-6,  # cgs to bar
                wavelengths=BaseSpectralModel.hz2um(radtrans.freq),
                **kwargs
            )


class Param:
    """Object used only to satisfy the requirements of the retrieval module."""
    def __init__(self, value):
        self.value = value


class ParametersDict(dict):
    def __init__(self, t_int, metallicity, co_ratio, p_cloud):
        super().__init__()

        self['intrinsic_temperature'] = t_int
        self['metallicity'] = metallicity
        self['co_ratio'] = co_ratio
        self['p_cloud'] = p_cloud

    def to_str(self):
        return f"T_int = {self['intrinsic_temperature']}, [Fe/H] = {self['metallicity']}, C/O = {self['co_ratio']}, " \
               f"P_cloud = {self['p_cloud']}"


class Planet:
    def __init__(
            self,
            name,
            mass=0.,
            mass_error_upper=0.,
            mass_error_lower=0.,
            radius=0.,
            radius_error_upper=0.,
            radius_error_lower=0.,
            orbit_semi_major_axis=0.,
            orbit_semi_major_axis_error_upper=0.,
            orbit_semi_major_axis_error_lower=0.,
            orbital_eccentricity=0.,
            orbital_eccentricity_error_upper=0.,
            orbital_eccentricity_error_lower=0.,
            orbital_inclination=0.,
            orbital_inclination_error_upper=0.,
            orbital_inclination_error_lower=0.,
            orbital_period=0.,
            orbital_period_error_upper=0.,
            orbital_period_error_lower=0.,
            argument_of_periastron=0.,
            argument_of_periastron_error_upper=0.,
            argument_of_periastron_error_lower=0.,
            epoch_of_periastron=0.,
            epoch_of_periastron_error_upper=0.,
            epoch_of_periastron_error_lower=0.,
            ra=0.,
            dec=0.,
            x=0.,
            y=0.,
            z=0.,
            reference_pressure=0.01,
            density=0.,
            density_error_upper=0.,
            density_error_lower=0.,
            surface_gravity=0.,
            surface_gravity_error_upper=0.,
            surface_gravity_error_lower=0.,
            equilibrium_temperature=0.,
            equilibrium_temperature_error_upper=0.,
            equilibrium_temperature_error_lower=0.,
            insolation_flux=0.,
            insolation_flux_error_upper=0.,
            insolation_flux_error_lower=0.,
            bond_albedo=0.,
            bond_albedo_error_upper=0.,
            bond_albedo_error_lower=0.,
            transit_depth=0.,
            transit_depth_error_upper=0.,
            transit_depth_error_lower=0.,
            transit_midpoint_time=0.,
            transit_midpoint_time_error_upper=0.,
            transit_midpoint_time_error_lower=0.,
            transit_duration=0.,
            transit_duration_error_upper=0.,
            transit_duration_error_lower=0.,
            projected_obliquity=0.,
            projected_obliquity_error_upper=0.,
            projected_obliquity_error_lower=0.,
            true_obliquity=0.,
            true_obliquity_error_upper=0.,
            true_obliquity_error_lower=0.,
            radial_velocity_amplitude=0.,
            radial_velocity_amplitude_error_upper=0.,
            radial_velocity_amplitude_error_lower=0.,
            planet_stellar_radius_ratio=0.,
            planet_stellar_radius_ratio_error_upper=0.,
            planet_stellar_radius_ratio_error_lower=0.,
            semi_major_axis_stellar_radius_ratio=0.,
            semi_major_axis_stellar_radius_ratio_error_upper=0.,
            semi_major_axis_stellar_radius_ratio_error_lower=0.,
            reference='',
            discovery_year=0,
            discovery_method='',
            discovery_reference='',
            confirmation_status='',
            host_name='',
            star_spectral_type='',
            star_mass=0.,
            star_mass_error_upper=0.,
            star_mass_error_lower=0.,
            star_radius=0.,
            star_radius_error_upper=0.,
            star_radius_error_lower=0.,
            star_age=0.,
            star_age_error_upper=0.,
            star_age_error_lower=0.,
            star_metallicity=0.,
            star_metallicity_error_upper=0.,
            star_metallicity_error_lower=0.,
            star_effective_temperature=0.,
            star_effective_temperature_error_upper=0.,
            star_effective_temperature_error_lower=0.,
            star_luminosity=0.,
            star_luminosity_error_upper=0.,
            star_luminosity_error_lower=0.,
            star_rotational_period=0.,
            star_rotational_period_error_upper=0.,
            star_rotational_period_error_lower=0.,
            star_radial_velocity=0.,
            star_radial_velocity_error_upper=0.,
            star_radial_velocity_error_lower=0.,
            star_rotational_velocity=0.,
            star_rotational_velocity_error_upper=0.,
            star_rotational_velocity_error_lower=0.,
            star_density=0.,
            star_density_error_upper=0.,
            star_density_error_lower=0.,
            star_surface_gravity=0.,
            star_surface_gravity_error_upper=0.,
            star_surface_gravity_error_lower=0.,
            star_reference='',
            system_star_number=0,
            system_planet_number=0,
            system_moon_number=0,
            system_distance=0.,
            system_distance_error_upper=0.,
            system_distance_error_lower=0.,
            system_apparent_magnitude_v=0.,
            system_apparent_magnitude_v_error_upper=0.,
            system_apparent_magnitude_v_error_lower=0.,
            system_apparent_magnitude_j=0.,
            system_apparent_magnitude_j_error_upper=0.,
            system_apparent_magnitude_j_error_lower=0.,
            system_apparent_magnitude_k=0.,
            system_apparent_magnitude_k_error_upper=0.,
            system_apparent_magnitude_k_error_lower=0.,
            system_proper_motion=0.,
            system_proper_motion_error_upper=0.,
            system_proper_motion_error_lower=0.,
            system_proper_motion_ra=0.,
            system_proper_motion_ra_error_upper=0.,
            system_proper_motion_ra_error_lower=0.,
            system_proper_motion_dec=0.,
            system_proper_motion_dec_error_upper=0.,
            system_proper_motion_dec_error_lower=0.,
            units=None
    ):
        self.name = name
        self.mass = mass
        self.mass_error_upper = mass_error_upper
        self.mass_error_lower = mass_error_lower
        self.radius = radius
        self.radius_error_upper = radius_error_upper
        self.radius_error_lower = radius_error_lower
        self.orbit_semi_major_axis = orbit_semi_major_axis
        self.orbit_semi_major_axis_error_upper = orbit_semi_major_axis_error_upper
        self.orbit_semi_major_axis_error_lower = orbit_semi_major_axis_error_lower
        self.orbital_eccentricity = orbital_eccentricity
        self.orbital_eccentricity_error_upper = orbital_eccentricity_error_upper
        self.orbital_eccentricity_error_lower = orbital_eccentricity_error_lower
        self.orbital_inclination = orbital_inclination
        self.orbital_inclination_error_upper = orbital_inclination_error_upper
        self.orbital_inclination_error_lower = orbital_inclination_error_lower
        self.orbital_period = orbital_period
        self.orbital_period_error_upper = orbital_period_error_upper
        self.orbital_period_error_lower = orbital_period_error_lower
        self.argument_of_periastron = argument_of_periastron
        self.argument_of_periastron_error_upper = argument_of_periastron_error_upper
        self.argument_of_periastron_error_lower = argument_of_periastron_error_lower
        self.epoch_of_periastron = epoch_of_periastron
        self.epoch_of_periastron_error_upper = epoch_of_periastron_error_upper
        self.epoch_of_periastron_error_lower = epoch_of_periastron_error_lower
        self.ra = ra
        self.dec = dec
        self.x = x
        self.y = y
        self.z = z
        self.reference_pressure = reference_pressure
        self.density = density
        self.density_error_upper = density_error_upper
        self.density_error_lower = density_error_lower
        self.surface_gravity = surface_gravity
        self.surface_gravity_error_upper = surface_gravity_error_upper
        self.surface_gravity_error_lower = surface_gravity_error_lower
        self.equilibrium_temperature = equilibrium_temperature
        self.equilibrium_temperature_error_upper = equilibrium_temperature_error_upper
        self.equilibrium_temperature_error_lower = equilibrium_temperature_error_lower
        self.insolation_flux = insolation_flux
        self.insolation_flux_error_upper = insolation_flux_error_upper
        self.insolation_flux_error_lower = insolation_flux_error_lower
        self.bond_albedo = bond_albedo
        self.bond_albedo_error_upper = bond_albedo_error_upper
        self.bond_albedo_error_lower = bond_albedo_error_lower
        self.transit_depth = transit_depth
        self.transit_depth_error_upper = transit_depth_error_upper
        self.transit_depth_error_lower = transit_depth_error_lower
        self.transit_midpoint_time = transit_midpoint_time
        self.transit_midpoint_time_error_upper = transit_midpoint_time_error_upper
        self.transit_midpoint_time_error_lower = transit_midpoint_time_error_lower
        self.transit_duration = transit_duration
        self.transit_duration_error_upper = transit_duration_error_upper
        self.transit_duration_error_lower = transit_duration_error_lower
        self.projected_obliquity = projected_obliquity
        self.projected_obliquity_error_upper = projected_obliquity_error_upper
        self.projected_obliquity_error_lower = projected_obliquity_error_lower
        self.true_obliquity = true_obliquity
        self.true_obliquity_error_upper = true_obliquity_error_upper
        self.true_obliquity_error_lower = true_obliquity_error_lower
        self.radial_velocity_amplitude = radial_velocity_amplitude
        self.radial_velocity_amplitude_error_upper = radial_velocity_amplitude_error_upper
        self.radial_velocity_amplitude_error_lower = radial_velocity_amplitude_error_lower
        self.planet_stellar_radius_ratio = planet_stellar_radius_ratio
        self.planet_stellar_radius_ratio_error_upper = planet_stellar_radius_ratio_error_upper
        self.planet_stellar_radius_ratio_error_lower = planet_stellar_radius_ratio_error_lower
        self.semi_major_axis_stellar_radius_ratio = semi_major_axis_stellar_radius_ratio
        self.semi_major_axis_stellar_radius_ratio_error_upper = semi_major_axis_stellar_radius_ratio_error_upper
        self.semi_major_axis_stellar_radius_ratio_error_lower = semi_major_axis_stellar_radius_ratio_error_lower
        self.reference = reference
        self.discovery_year = discovery_year
        self.discovery_method = discovery_method
        self.discovery_reference = discovery_reference
        self.confirmation_status = confirmation_status
        self.host_name = host_name
        self.star_spectral_type = star_spectral_type
        self.star_mass = star_mass
        self.star_mass_error_upper = star_mass_error_upper
        self.star_mass_error_lower = star_mass_error_lower
        self.star_radius = star_radius
        self.star_radius_error_upper = star_radius_error_upper
        self.star_radius_error_lower = star_radius_error_lower
        self.star_age = star_age
        self.star_age_error_upper = star_age_error_upper
        self.star_age_error_lower = star_age_error_lower
        self.star_metallicity = star_metallicity
        self.star_metallicity_error_upper = star_metallicity_error_upper
        self.star_metallicity_error_lower = star_metallicity_error_lower
        self.star_effective_temperature = star_effective_temperature
        self.star_effective_temperature_error_upper = star_effective_temperature_error_upper
        self.star_effective_temperature_error_lower = star_effective_temperature_error_lower
        self.star_luminosity = star_luminosity
        self.star_luminosity_error_upper = star_luminosity_error_upper
        self.star_luminosity_error_lower = star_luminosity_error_lower
        self.star_rotational_period = star_rotational_period
        self.star_rotational_period_error_upper = star_rotational_period_error_upper
        self.star_rotational_period_error_lower = star_rotational_period_error_lower
        self.star_radial_velocity = star_radial_velocity
        self.star_radial_velocity_error_upper = star_radial_velocity_error_upper
        self.star_radial_velocity_error_lower = star_radial_velocity_error_lower
        self.star_rotational_velocity = star_rotational_velocity
        self.star_rotational_velocity_error_upper = star_rotational_velocity_error_upper
        self.star_rotational_velocity_error_lower = star_rotational_velocity_error_lower
        self.star_density = star_density
        self.star_density_error_upper = star_density_error_upper
        self.star_density_error_lower = star_density_error_lower
        self.star_surface_gravity = star_surface_gravity
        self.star_surface_gravity_error_upper = star_surface_gravity_error_upper
        self.star_surface_gravity_error_lower = star_surface_gravity_error_lower
        self.star_reference = star_reference
        self.system_star_number = system_star_number
        self.system_planet_number = system_planet_number
        self.system_moon_number = system_moon_number
        self.system_distance = system_distance
        self.system_distance_error_upper = system_distance_error_upper
        self.system_distance_error_lower = system_distance_error_lower
        self.system_apparent_magnitude_v = system_apparent_magnitude_v
        self.system_apparent_magnitude_v_error_upper = system_apparent_magnitude_v_error_upper
        self.system_apparent_magnitude_v_error_lower = system_apparent_magnitude_v_error_lower
        self.system_apparent_magnitude_j = system_apparent_magnitude_j
        self.system_apparent_magnitude_j_error_upper = system_apparent_magnitude_j_error_upper
        self.system_apparent_magnitude_j_error_lower = system_apparent_magnitude_j_error_lower
        self.system_apparent_magnitude_k = system_apparent_magnitude_k
        self.system_apparent_magnitude_k_error_upper = system_apparent_magnitude_k_error_upper
        self.system_apparent_magnitude_k_error_lower = system_apparent_magnitude_k_error_lower
        self.system_proper_motion = system_proper_motion
        self.system_proper_motion_error_upper = system_proper_motion_error_upper
        self.system_proper_motion_error_lower = system_proper_motion_error_lower
        self.system_proper_motion_ra = system_proper_motion_ra
        self.system_proper_motion_ra_error_upper = system_proper_motion_ra_error_upper
        self.system_proper_motion_ra_error_lower = system_proper_motion_ra_error_lower
        self.system_proper_motion_dec = system_proper_motion_dec
        self.system_proper_motion_dec_error_upper = system_proper_motion_dec_error_upper
        self.system_proper_motion_dec_error_lower = system_proper_motion_dec_error_lower

        if units is None:
            self.units = {
                'name': 'N/A',
                'mass': 'g',
                'mass_error_upper': 'g',
                'mass_error_lower': 'g',
                'radius': 'cm',
                'radius_error_upper': 'cm',
                'radius_error_lower': 'cm',
                'orbit_semi_major_axis': 'cm',
                'orbit_semi_major_axis_error_upper': 'cm',
                'orbit_semi_major_axis_error_lower': 'cm',
                'orbital_eccentricity': 'None',
                'orbital_eccentricity_error_upper': 'None',
                'orbital_eccentricity_error_lower': 'None',
                'orbital_inclination': 'deg',
                'orbital_inclination_error_upper': 'deg',
                'orbital_inclination_error_lower': 'deg',
                'orbital_period': 's',
                'orbital_period_error_upper': 's',
                'orbital_period_error_lower': 's',
                'argument_of_periastron': 'deg',
                'argument_of_periastron_error_upper': 'deg',
                'argument_of_periastron_error_lower': 'deg',
                'epoch_of_periastron': 's',
                'epoch_of_periastron_error_upper': 's',
                'epoch_of_periastron_error_lower': 's',
                'ra': 'deg',
                'dec': 'deg',
                'x': 'cm',
                'y': 'cm',
                'z': 'cm',
                'reference_pressure': 'bar',
                'density': 'g/cm^3',
                'density_error_upper': 'g/cm^3',
                'density_error_lower': 'g/cm^3',
                'surface_gravity': 'cm/s^2',
                'surface_gravity_error_upper': 'cm/s^2',
                'surface_gravity_error_lower': 'cm/s^2',
                'equilibrium_temperature': 'K',
                'equilibrium_temperature_error_upper': 'K',
                'equilibrium_temperature_error_lower': 'K',
                'insolation_flux': 'erg/s/cm^2',
                'insolation_flux_error_upper': 'erg/s/cm^2',
                'insolation_flux_error_lower': 'erg/s/cm^2',
                'bond_albedo': 'None',
                'bond_albedo_error_upper': 'None',
                'bond_albedo_error_lower': 'None',
                'transit_depth': 'None',
                'transit_depth_error_upper': 'None',
                'transit_depth_error_lower': 'None',
                'transit_midpoint_time': 's',
                'transit_midpoint_time_error_upper': 's',
                'transit_midpoint_time_error_lower': 's',
                'transit_duration': 's',
                'transit_duration_error_upper': 's',
                'transit_duration_error_lower': 's',
                'projected_obliquity': 'deg',
                'projected_obliquity_error_upper': 'deg',
                'projected_obliquity_error_lower': 'deg',
                'true_obliquity': 'deg',
                'true_obliquity_error_upper': 'deg',
                'true_obliquity_error_lower': 'deg',
                'radial_velocity_amplitude': 'cm/s',
                'radial_velocity_amplitude_error_upper': 'cm/s',
                'radial_velocity_amplitude_error_lower': 'cm/s',
                'planet_stellar_radius_ratio': 'None',
                'planet_stellar_radius_ratio_error_upper': 'None',
                'planet_stellar_radius_ratio_error_lower': 'None',
                'semi_major_axis_stellar_radius_ratio': 'None',
                'semi_major_axis_stellar_radius_ratio_error_upper': 'None',
                'semi_major_axis_stellar_radius_ratio_error_lower': 'None',
                'reference': 'N/A',
                'discovery_year': 'year',
                'discovery_method': 'N/A',
                'discovery_reference': 'N/A',
                'confirmation_status': 'N/A',
                'host_name': 'N/A',
                'star_spectral_type': 'N/A',
                'star_mass': 'g',
                'star_mass_error_upper': 'g',
                'star_mass_error_lower': 'g',
                'star_radius': 'cm',
                'star_radius_error_upper': 'cm',
                'star_radius_error_lower': 'cm',
                'star_age': 's',
                'star_age_error_upper': 's',
                'star_age_error_lower': 's',
                'star_metallicity': 'dex',
                'star_metallicity_error_upper': 'dex',
                'star_metallicity_error_lower': 'dex',
                'star_effective_temperature': 'K',
                'star_effective_temperature_error_upper': 'K',
                'star_effective_temperature_error_lower': 'K',
                'star_luminosity': 'erg/s',
                'star_luminosity_error_upper': 'erg/s',
                'star_luminosity_error_lower': 'erg/s',
                'star_rotational_period': 's',
                'star_rotational_period_error_upper': 's',
                'star_rotational_period_error_lower': 's',
                'star_radial_velocity': 'cm/s',
                'star_radial_velocity_error_upper': 'cm/s',
                'star_radial_velocity_error_lower': 'cm/s',
                'star_rotational_velocity': 'cm/s',
                'star_rotational_velocity_error_upper': 'cm/s',
                'star_rotational_velocity_error_lower': 'cm/s',
                'star_density': 'g/cm^3',
                'star_density_error_upper': 'g/cm^3',
                'star_density_error_lower': 'g/cm^3',
                'star_surface_gravity': 'cm/s^2',
                'star_surface_gravity_error_upper': 'cm/s^2',
                'star_surface_gravity_error_lower': 'cm/s^2',
                'star_reference': 'N/A',
                'system_star_number': 'None',
                'system_planet_number': 'None',
                'system_moon_number': 'None',
                'system_distance': 'cm',
                'system_distance_error_upper': 'cm',
                'system_distance_error_lower': 'cm',
                'system_apparent_magnitude_v': 'None',
                'system_apparent_magnitude_v_error_upper': 'None',
                'system_apparent_magnitude_v_error_lower': 'None',
                'system_apparent_magnitude_j': 'None',
                'system_apparent_magnitude_j_error_upper': 'None',
                'system_apparent_magnitude_j_error_lower': 'None',
                'system_apparent_magnitude_k': 'None',
                'system_apparent_magnitude_k_error_upper': 'None',
                'system_apparent_magnitude_k_error_lower': 'None',
                'system_proper_motion': 'deg/s',
                'system_proper_motion_error_upper': 'deg/s',
                'system_proper_motion_error_lower': 'deg/s',
                'system_proper_motion_ra': 'deg/s',
                'system_proper_motion_ra_error_upper': 'deg/s',
                'system_proper_motion_ra_error_lower': 'deg/s',
                'system_proper_motion_dec': 'deg/s',
                'system_proper_motion_dec_error_upper': 'deg/s',
                'system_proper_motion_dec_error_lower': 'deg/s',
                'units': 'N/A'
            }
        else:
            self.units = units

    def calculate_planetary_equilibrium_temperature(self):
        """
        Calculate the equilibrium temperature of a planet.
        """
        equilibrium_temperature = \
            self.star_effective_temperature * np.sqrt(self.star_radius / (2 * self.orbit_semi_major_axis)) \
            * (1 - self.bond_albedo) ** 0.25

        partial_derivatives = np.array([
            equilibrium_temperature / self.star_effective_temperature,  # dt_eq/dt_eff
            0.5 * equilibrium_temperature / self.star_radius,  # dt_eq/dr*
            - 0.5 * equilibrium_temperature / self.orbit_semi_major_axis  # dt_eq/dd
        ])
        uncertainties = np.abs(np.array([
            [self.star_effective_temperature_error_lower, self.star_effective_temperature_error_upper],
            [self.star_radius_error_lower, self.star_radius_error_upper],
            [self.orbit_semi_major_axis_error_lower, self.orbit_semi_major_axis_error_upper]
        ]))

        errors = calculate_uncertainty(partial_derivatives, uncertainties)  # lower and upper errors

        return equilibrium_temperature, errors[1], -errors[0]

    def get_filename(self):
        return self.generate_filename(self.name)

    def save(self, filename=None):
        if filename is None:
            filename = self.get_filename()

        with h5py.File(filename, 'w') as f:
            for key in self.__dict__:
                if key == 'units':
                    continue

                data_set = f.create_dataset(
                    name=key,
                    data=self.__dict__[key]
                )

                if self.units[key] != 'N/A':
                    data_set.attrs['units'] = self.units[key]

    @classmethod
    def from_tab_file(cls, filename, use_best_mass=True):
        """Read from a NASA Exoplanet Archive Database .tab file.
        Args:
            filename: file to read
            use_best_mass: if True, use NASA Exoplanet Archive Database 'bmass' argument instead of 'mass'.

        Returns:
            planets: a list of Planet objects
        """
        with open(filename, 'r') as f:
            line = f.readline()
            line = line.strip()

            # Skip header
            while line[0] == '#':
                line = f.readline()
                line = line.strip()

            # Read column names
            columns_name = line.split('\t')

            planet_name_index = columns_name.index('pl_name')

            planets = {}

            # Read data
            for line in f:
                line = line.strip()
                columns = line.split('\t')

                new_planet = cls(columns[planet_name_index])
                keys = []

                for i, value in enumerate(columns):
                    # Clearer keynames
                    keys.append(columns_name[i])

                    if value != '':
                        try:
                            value = float(value)
                        except ValueError:
                            pass

                        value, keys[i] = Planet.__convert_nasa_exoplanet_archive(
                            value, keys[i], use_best_mass=use_best_mass
                        )
                    else:
                        value = None

                    if keys[i] in new_planet.__dict__:
                        new_planet.__dict__[keys[i]] = value

                # Try to calculate the planet mass and radius if missing
                if new_planet.radius == 0 and new_planet.mass > 0 and new_planet.density > 0:
                    new_planet.radius = (3 * new_planet.mass / (4 * np.pi * new_planet.density)) ** (1 / 3)

                    partial_derivatives = np.array([
                        new_planet.radius / (3 * new_planet.mass),  # dr/dm
                        - new_planet.radius / (3 * new_planet.density)  # dr/drho
                    ])
                    uncertainties = np.abs(np.array([
                        [new_planet.mass_error_lower, new_planet.mass_error_upper],
                        [new_planet.density_error_lower, new_planet.density_error_upper]
                    ]))

                    new_planet.radius_error_lower, new_planet.radius_error_upper = \
                        calculate_uncertainty(partial_derivatives, uncertainties)  # lower and upper errors
                elif new_planet.mass == 0 and new_planet.radius > 0 and new_planet.density > 0:
                    new_planet.mass = new_planet.density * 4 / 3 * np.pi * new_planet.radius ** 3

                    partial_derivatives = np.array([
                        new_planet.mass / new_planet.density,  # dm/drho
                        3 * new_planet.radius / new_planet.radius  # dm/dr
                    ])
                    uncertainties = np.abs(np.array([
                        [new_planet.density_error_lower, new_planet.density_error_upper],
                        [new_planet.radius_error_lower, new_planet.radius_error_upper]
                    ]))

                    new_planet.mass_error_lower, new_planet.mass_error_upper = \
                        calculate_uncertainty(partial_derivatives, uncertainties)  # lower and upper errors

                # Try to calculate the star radius if missing
                if new_planet.star_radius == 0 and new_planet.star_mass > 0:
                    if new_planet.star_surface_gravity > 0:
                        new_planet.star_radius, \
                            new_planet.star_radius_error_upper, new_planet.star_radius_error_lower = \
                            new_planet.surface_gravity2radius(
                                new_planet.star_surface_gravity,
                                new_planet.star_mass,
                                surface_gravity_error_upper=new_planet.star_surface_gravity_error_upper,
                                surface_gravity_error_lower=new_planet.star_surface_gravity_error_lower,
                                mass_error_upper=new_planet.star_mass_error_upper,
                                mass_error_lower=new_planet.star_mass_error_lower
                            )
                    elif new_planet.star_density > 0:
                        new_planet.star_radius = \
                            (3 * new_planet.star_mass / (4 * np.pi * new_planet.star_density)) ** (1 / 3)

                        partial_derivatives = np.array([
                            new_planet.star_radius / (3 * new_planet.star_mass),  # dr/dm
                            - new_planet.star_radius / (3 * new_planet.star_density)  # dr/drho
                        ])
                        uncertainties = np.abs(np.array([
                            [new_planet.star_mass_error_lower, new_planet.star_mass_error_upper],
                            [new_planet.star_density_error_lower, new_planet.star_density_error_upper]
                        ]))

                        new_planet.star_radius_error_lower, new_planet.star_radius_error_upper = \
                            calculate_uncertainty(partial_derivatives, uncertainties)  # lower and upper errors

                if 'surface_gravity' not in keys and new_planet.radius > 0 and new_planet.mass > 0:
                    new_planet.surface_gravity, \
                        new_planet.surface_gravity_error_upper, new_planet.surface_gravity_error_lower = \
                        new_planet.mass2surface_gravity(
                            new_planet.mass,
                            new_planet.radius,
                            mass_error_upper=new_planet.mass_error_upper,
                            mass_error_lower=new_planet.mass_error_lower,
                            radius_error_upper=new_planet.radius_error_upper,
                            radius_error_lower=new_planet.radius_error_lower
                        )

                if 'equilibrium_temperature' not in keys \
                        and new_planet.orbit_semi_major_axis > 0 \
                        and new_planet.star_effective_temperature > 0 \
                        and new_planet.star_radius > 0:
                    new_planet.equilibrium_temperature, \
                        new_planet.equilibrium_temperature_error_upper, new_planet.equilibrium_temperature_error_lower = \
                        new_planet.calculate_planetary_equilibrium_temperature()

                planets[new_planet.name] = new_planet

        return planets

    @classmethod
    def from_votable(cls, votable):
        new_planet = cls('new_planet')
        parameter_dict = {}

        for key in votable.keys():
            # Clearer keynames
            value, key = Planet.__convert_nasa_exoplanet_archive(votable[key], key)
            parameter_dict[key] = value

        parameter_dict = new_planet.select_best_in_column(parameter_dict)

        for key in parameter_dict:
            if key in new_planet.__dict__:
                new_planet.__dict__[key] = parameter_dict[key]

        if 'surface_gravity' not in parameter_dict:
            new_planet.surface_gravity, \
                new_planet.surface_gravity_error_upper, new_planet.surface_gravity_error_lower = \
                new_planet.mass2surface_gravity(
                    new_planet.mass,
                    new_planet.radius,
                    mass_error_upper=new_planet.mass_error_upper,
                    mass_error_lower=new_planet.mass_error_lower,
                    radius_error_upper=new_planet.radius_error_upper,
                    radius_error_lower=new_planet.radius_error_lower
                )

        if 'equilibrium_temperature' not in parameter_dict:
            new_planet.equilibrium_temperature, \
                new_planet.equilibrium_temperature_error_upper, new_planet.equilibrium_temperature_error_lower = \
                new_planet.calculate_planetary_equilibrium_temperature()

        return new_planet

    @classmethod
    def from_votable_file(cls, filename):
        astro_table = Table.read(filename)

        return cls.from_votable(astro_table)

    @classmethod
    def get(cls, name):
        filename = cls.generate_filename(name)

        if not os.path.exists(filename):
            filename_vot = filename.rsplit('.', 1)[0] + '.vot'  # search for votable

            if not os.path.exists(filename_vot):
                print(f"file '{filename_vot}' not found, downloading...")
                cls.download_from_nasa_exoplanet_archive(name)

            # Save into HDF5 and remove the VO table
            new_planet = cls.from_votable_file(filename_vot)
            new_planet.save()
            os.remove(filename_vot)

            return new_planet
        else:
            return cls.load(name, filename)

    @classmethod
    def load(cls, name, filename=None):
        new_planet = cls(name)

        if filename is None:
            filename = new_planet.get_filename()

        with h5py.File(filename, 'r') as f:
            for key in f:
                if isinstance(f[key][()], bytes):
                    value = str(f[key][()], 'utf-8')
                else:
                    value = f[key][()]

                new_planet.__dict__[key] = value

                if 'units' in f[key].attrs:
                    if key in new_planet.units:
                        if f[key].attrs['units'] != new_planet.units[key]:
                            raise ValueError(f"units of key '{key}' must be '{new_planet.units[key]}', "
                                             f"but is '{f[key].attrs['units']}'")
                    else:
                        new_planet.units[key] = f[key].attrs['units'][()]
                else:
                    new_planet.units[key] = 'N/A'

        return new_planet

    @staticmethod
    def __convert_nasa_exoplanet_archive(value, key, verbose=False, use_best_mass=False):
        skip_unit_conversion = False

        # Heads
        if key[:3] == 'sy_':
            key = 'system_' + key[3:]
        elif key[:3] == 'st_':
            key = 'star_' + key[3:]
        elif key[:5] == 'disc_':
            key = 'discovery_' + key[5:]

        # Tails
        if key[-4:] == 'err1':
            key = key[:-4] + '_error_upper'
        elif key[-4:] == 'err2':
            key = key[:-4] + '_error_lower'
        elif key[-3:] == 'lim':
            key = key[:-3] + '_limit_flag'
            skip_unit_conversion = True
        elif key[-3:] == 'str':
            key = key[:-3] + '_str'
            skip_unit_conversion = True

        # Parameters of interest
        if '_orbper' in key:
            key = key.replace('_orbper', '_orbital_period')

            if not skip_unit_conversion:
                value *= nc.snc.day
        elif '_orblper' in key:
            key = key.replace('_orblper', '_argument_of_periastron')
        elif '_orbsmax' in key:
            key = key.replace('_orbsmax', '_orbit_semi_major_axis')

            if not skip_unit_conversion:
                value *= nc.AU
        elif '_orbincl' in key:
            key = key.replace('_orbincl', '_orbital_inclination')
        elif '_orbtper' in key:
            key = key.replace('_orbtper', '_epoch_of_periastron')

            if not skip_unit_conversion:
                value *= nc.snc.day
        elif '_orbeccen' in key:
            key = key.replace('_orbeccen', '_orbital_eccentricity')
        elif '_eqt' in key:
            key = key.replace('_eqt', '_equilibrium_temperature')
        elif '_occdep' in key:
            key = key.replace('_occdep', '_occultation_depth')
        elif '_insol' in key:
            key = key.replace('_insol', '_insolation_flux')

            if not skip_unit_conversion:
                value *= nc.s_earth
        elif '_dens' in key:
            key = key.replace('_dens', '_density')
        elif '_trandep' in key:
            key = key.replace('_trandep', '_transit_depth')

            if not skip_unit_conversion:
                value *= 1e2
        elif '_tranmid' in key:
            key = key.replace('_tranmid', '_transit_midpoint_time')

            if not skip_unit_conversion:
                value *= nc.snc.day
        elif '_trandur' in key:
            key = key.replace('_trandur', '_transit_duration')

            if not skip_unit_conversion:
                value *= nc.snc.hour
        elif '_spectype' in key:
            key = key.replace('_spectype', '_spectral_type')
        elif '_rotp' in key:
            key = key.replace('_rotp', '_rotational_period')

            if not skip_unit_conversion:
                value *= nc.snc.day
        elif '_projobliq' in key:
            key = key.replace('_projobliq', '_projected_obliquity')
        elif '_rvamp' in key:
            key = key.replace('_rvamp', '_radial_velocity_amplitude')

            if not skip_unit_conversion:
                value *= 1e2
        elif '_radj' in key:
            key = key.replace('_radj', '_radius')

            if not skip_unit_conversion:
                value *= nc.r_jup
        elif '_ratror' in key:
            key = key.replace('_ratror', '_planet_stellar_radius_ratio')
        elif '_trueobliq' in key:
            key = key.replace('_trueobliq', '_true_obliquity')
        elif '_ratdor' in key:
            key = key.replace('_ratdor', '_semi_major_axis_stellar_radius_ratio')
        elif '_imppar' in key:
            key = key.replace('_imppar', '_impact_parameter')
        elif '_msinij' in key:
            key = key.replace('_msinij', '_mass_sini')

            if not skip_unit_conversion:
                value *= nc.m_jup
        elif '_massj' in key:
            if not use_best_mass:
                key = key.replace('_massj', '_mass')

                if not skip_unit_conversion:
                    value *= nc.m_jup
        elif '_bmassj' in key:
            if use_best_mass:
                key = key.replace('_bmassj', '_mass')

                if not skip_unit_conversion:
                    value *= nc.m_jup
        elif '_teff' in key:
            key = key.replace('_teff', '_effective_temperature')
        elif '_met' in key:
            key = key.replace('_met', '_metallicity')
        elif '_radv' in key:
            key = key.replace('_radv', '_radial_velocity')

            if not skip_unit_conversion:
                value *= 1e5
        elif '_vsin' in key:
            key = key.replace('_vsin', '_rotational_velocity')

            if not skip_unit_conversion:
                value *= 1e5
        elif '_lum' in key:
            key = key.replace('_lum', '_luminosity')

            if not skip_unit_conversion:
                value = 10 ** value * nc.l_sun
        elif '_logg' in key:
            key = key.replace('_logg', '_surface_gravity')

            if not skip_unit_conversion:
                value = 10 ** value
        elif '_age' in key:
            if not skip_unit_conversion:
                value *= 1e9 * nc.snc.year
        elif 'star_mass' in key:
            if not skip_unit_conversion:
                value *= nc.m_sun
        elif 'star_rad' in key:
            key = key.replace('star_rad', 'star_radius')

            if not skip_unit_conversion:
                value *= nc.r_sun
        elif '_dist' in key:
            key = key.replace('_dist', '_distance')

            if not skip_unit_conversion:
                value *= nc.pc
        elif '_plx' in key:
            key = key.replace('_plx', '_parallax')

            if not skip_unit_conversion:
                value *= 3.6e-6
        elif '_pm' in key:
            if key[-3:] == '_pm':
                key = key.replace('_pm', '_proper_motion')
            else:
                i = key.find('_pm')
                key = key[:i] + '_proper_motion_' + key[i + len('_pm'):]

            if not skip_unit_conversion:
                value *= np.deg2rad(1e-3 / 3600 / nc.snc.year)

        elif key == 'hostname':
            key = 'host_name'
        elif key == 'discoverymethod':
            key = 'discovery_method'
        elif key == 'discovery_refname':
            key = 'discovery_reference'
        elif 'controv_flag' in key:
            key = 'controversy_flag'
        elif key == 'star_refname':
            key = 'star_reference'
        elif key == 'soltype':
            key = 'confirmation_status'
        elif key == 'system_snum':
            key = 'system_star_number'
        elif key == 'system_pnum':
            key = 'system_planet_number'
        elif key == 'system_mnum':
            key = 'system_moon_number'
        elif 'mag' in key:
            i = key.find('mag')

            if i + len('mag') == len(key):
                tail = ''
            else:
                tail = key[i + 3:]

                if tail[0] != '_':  # should not be necessary
                    tail = '_' + tail

            # Move magnitude band to the end
            if key[i - 2] == '_':  # one-character band
                letter = key[i - 1]
                key = key[:i - 1] + 'apparent_magnitude_' + letter + tail
            elif key[i - 3] == '_':  # two-characters band
                letters = key[i - 2:i]
                key = key[:i - 2] + 'apparent_magnitude_' + letters + tail
            elif 'kepmag' in key:
                key = key[:i - 3] + 'apparent_magnitude_' + 'kepler' + tail
            elif 'gaiamag' in key:
                key = key[:i - 4] + 'apparent_magnitude_' + 'gaia' + tail
            else:
                raise ValueError(f"unidentified apparent magnitude key '{key}'")
        elif verbose:
            print(f"unchanged key '{key}' with value {value}")

        if key[:3] == 'pl_':
            key = key[3:]

        return value, key

    @staticmethod
    def calculate_planet_radial_velocity(planet_max_radial_orbital_velocity, planet_orbital_inclination,
                                         orbital_longitude):
        """Calculate the planet radial velocity as seen by an observer.

        Args:
            planet_max_radial_orbital_velocity: maximum radial velocity for an inclination angle of 90 degree
            planet_orbital_inclination: (degree) angle between the normal of the planet orbital plane and the axis of
                observation, i.e. 90 degree: edge view, 0 degree: top view
            orbital_longitude: (degree) angle between the closest point from the observer on the planet orbit and the
                planet position, i.e. if the planet orbital inclination is 0 degree, 0 degree: mid primary transit
                point, 180 degree: mid secondary eclipse point

        Returns:

        """
        kp = planet_max_radial_orbital_velocity * np.sin(np.deg2rad(planet_orbital_inclination))  # (cm.s-1)

        return kp * np.sin(np.deg2rad(orbital_longitude))

    @staticmethod
    def calculate_orbital_velocity(star_mass, semi_major_axis):
        """Calculate an approximation of the orbital velocity.
        This equation is valid if the mass of the object is negligible compared to the mass of the star, and if the
        eccentricity of the object is close to 0.
        
        Args:
            star_mass: (g) mass of the star
            semi_major_axis: (cm) semi-major axis of the orbit of the object

        Returns: (cm.s-1) the mean orbital velocity, assuming 0 eccentricity and mass_object << mass_star
        """
        return np.sqrt(nc.G * star_mass / semi_major_axis)

    @staticmethod
    def generate_filename(name):
        return f"{planet_models_directory}{os.path.sep}planet_{name.replace(' ', '_')}.h5"

    @staticmethod
    def get_simple_transit_curve(time_from_mid_transit, planet_radius, star_radius,
                                 planet_orbital_velocity=None, star_mass=None, orbit_semi_major_axis=None):
        """
        Assume no inclination, circular orbit, observer infinitely far away, spherical objects, perfectly sharp and
        black planet and perfectly sharp and uniformly luminous star.

        Args:
            time_from_mid_transit: (s) time from mid transit, 0 is the mid transit time, < 0 before and > 0 after
            planet_radius: (cm) radius of the planet
            star_radius: (cm) radius of the star
            planet_orbital_velocity: (cm.s-1) planet velocity along its orbit.
            star_mass: (g) mass of the star
            orbit_semi_major_axis: (cm) planet orbit semi major axis

        Returns:

        """
        if planet_orbital_velocity is None:
            planet_orbital_velocity = Planet.calculate_orbital_velocity(star_mass, orbit_semi_major_axis)

        planet_center_to_star_center = planet_orbital_velocity * time_from_mid_transit

        if np.abs(planet_center_to_star_center) >= star_radius + planet_radius:
            return 1.0  # planet is not transiting yet
        elif np.abs(planet_center_to_star_center) <= star_radius - planet_radius:
            return 1 - (planet_radius / star_radius) ** 2  # planet is fully transiting
        else:
            # Get the vertical coordinate intersection between the two discs
            x_intersection = (star_radius ** 2 - planet_radius ** 2 + planet_center_to_star_center ** 2) \
                             / (2 * planet_center_to_star_center)
            y_intersection = np.sqrt(star_radius ** 2 - np.abs(x_intersection) ** 2)

            # Get the half angle between the two intersection points and the center of each disc
            theta_half_intersection_planet = np.arcsin(y_intersection / planet_radius)
            theta_half_intersection_star = np.arcsin(y_intersection / star_radius)

            if np.abs(planet_center_to_star_center) < star_radius:
                theta_half_intersection_planet = np.pi - theta_half_intersection_planet

            # Calculate the area of the sector between the 2 intersection point for the 2 discs
            planet_sector_area = planet_radius ** 2 * theta_half_intersection_planet
            star_sector_area = star_radius ** 2 * theta_half_intersection_star

            # Calculate the area of the triangles formed by the 2 intersection points and the center of each disc
            planet_triangle_area = 0.5 * planet_radius ** 2 * np.sin(2 * theta_half_intersection_planet)
            star_triangle_area = 0.5 * star_radius ** 2 * np.sin(2 * theta_half_intersection_star)

            return 1 - (planet_sector_area - planet_triangle_area + star_sector_area - star_triangle_area) \
                / (np.pi * star_radius ** 2)

    @staticmethod
    def get_orbital_phases(phase_start, orbital_period, times):
        """Calculate orbital phases assuming low eccentricity.

        Args:
            phase_start: planet phase at the start of observations
            orbital_period: (s) orbital period of the planet
            times: (s) time array

        Returns:
            The orbital phases for the given time
        """
        return np.mod(phase_start + times / orbital_period, 1.0)

    @staticmethod
    def download_from_nasa_exoplanet_archive(name):
        service = pyvo.dal.TAPService("https://exoplanetarchive.ipac.caltech.edu/TAP")
        result_set = service.search(f"select * from ps where pl_name = '{name}'")

        astro_table = result_set.to_table()
        filename = Planet.generate_filename(name).rsplit('.', 1)[0] + '.vot'

        astro_table.write(filename, format='votable')

        return astro_table

    @staticmethod
    def select_best_in_column(dictionary):
        parameter_dict = {}
        tails = ['_error_upper', '_error_lower', '_limit_flag', '_str']

        for key in dictionary.keys():
            if tails[0] in key or tails[1] in key or tails[2] in key or tails[3] in key:
                continue  # skip every tailed parameters
            elif dictionary[key].dtype == object or not (key + tails[0] in dictionary and key + tails[1] in dictionary):
                # if object or no error tailed parameters, get the first value that is not masked
                if not hasattr(dictionary[key], '__iter__'):
                    raise ValueError(f"No value found for parameter '{key}'; "
                                     f"this error is most often caused by a misspelling of a planet name")

                parameter_dict[key] = dictionary[key][0]

                for value in dictionary[key][1:]:
                    if not hasattr(value, 'mask'):
                        parameter_dict[key] = value

                        break
            else:
                value_error_upper = dictionary[key + tails[0]]
                value_error_lower = dictionary[key + tails[1]]
                error_interval = np.abs(value_error_upper) + np.abs(value_error_lower)

                wh = np.where(error_interval == np.min(error_interval))[0]

                parameter_dict[key] = dictionary[key][wh][0]

                for tail in tails:
                    if key + tail in dictionary:
                        parameter_dict[key + tail] = dictionary[key + tail][wh][0]

        return parameter_dict

    @staticmethod
    def mass2surface_gravity(mass, radius,
                             mass_error_upper=0., mass_error_lower=0., radius_error_upper=0., radius_error_lower=0.):
        """
        Convert the mass of a planet to its surface gravity.
        Args:
            mass: (g) mass of the planet
            radius: (cm) radius of the planet
            mass_error_upper: (g) upper error on the planet mass
            mass_error_lower: (g) lower error on the planet mass
            radius_error_upper: (cm) upper error on the planet radius
            radius_error_lower: (cm) lower error on the planet radius

        Returns:
            (cm.s-2) the surface gravity of the planet, and its upper and lower error
        """
        surface_gravity = nc.G * mass / radius ** 2

        partial_derivatives = np.array([
            surface_gravity / mass,  # dg/dm
            - 2 * surface_gravity / radius  # dg/dr
        ])
        uncertainties = np.abs(np.array([
            [mass_error_lower, mass_error_upper],
            [radius_error_lower, radius_error_upper]
        ]))

        errors = calculate_uncertainty(partial_derivatives, uncertainties)  # lower and upper errors

        return surface_gravity, errors[1], -errors[0]

    @staticmethod
    def surface_gravity2radius(surface_gravity, mass,
                               surface_gravity_error_upper=0., surface_gravity_error_lower=0.,
                               mass_error_upper=0., mass_error_lower=0.):
        """
        Convert the mass of a planet to its surface gravity.
        Args:
            surface_gravity: (cm.s-2) surface_gravity of the planet
            mass: (g) mass of the planet
            mass_error_upper: (g) upper error on the planet mass
            mass_error_lower: (g) lower error on the planet mass
            surface_gravity_error_upper: (cm.s-2) upper error on the planet radius
            surface_gravity_error_lower: (cm.s-2) lower error on the planet radius

        Returns:
            (cm.s-2) the surface gravity of the planet, and its upper and lower error
        """
        radius = (nc.G * mass / surface_gravity) ** 0.5

        partial_derivatives = np.array([
            radius / (2 * mass),  # dr/dm
            - radius / (2 * surface_gravity)  # dr/dg
        ])
        uncertainties = np.abs(np.array([
            [mass_error_lower, mass_error_upper],
            [surface_gravity_error_lower, surface_gravity_error_upper]
        ]))

        errors = calculate_uncertainty(partial_derivatives, uncertainties)  # lower and upper errors

        return radius, errors[1], -errors[0]

    @staticmethod
    def surface_gravity2mass(surface_gravity, radius,
                             surface_gravity_error_upper=0., surface_gravity_error_lower=0.,
                             radius_error_upper=0., radius_error_lower=0.):
        """
        Convert the surface gravity of a planet to its mass.
        Args:
            surface_gravity: (cm.s-2) surface gravity of the planet
            radius: (cm) radius of the planet
            surface_gravity_error_upper: (cm.s-2) upper error on the planet surface gravity
            surface_gravity_error_lower: (cm.s-2) lower error on the planet surface gravity
            radius_error_upper: (cm) upper error on the planet radius
            radius_error_lower: (cm) lower error on the planet radius

        Returns:
            (g) the mass of the planet, and its upper and lower error
        """
        mass = surface_gravity / nc.G * radius ** 2

        partial_derivatives = np.array([
            mass / surface_gravity,  # dm/dg
            2 * mass / radius  # dm/dr
        ])
        uncertainties = np.abs(np.array([
            [surface_gravity_error_lower, surface_gravity_error_upper],
            [radius_error_lower, radius_error_upper]
        ]))

        errors = calculate_uncertainty(partial_derivatives, uncertainties)  # lower and upper errors

        return mass, errors[1], -errors[0]


class SimplePlanet(Planet):
    def __init__(self, name, radius, surface_gravity, star_effective_temperature, star_radius, orbit_semi_major_axis,
                 reference_pressure=0.01, bond_albedo=0, equilibrium_temperature=None, mass=None):
        """

        Args:
            name: name of the planet
            radius: (cm) radius of the planet
            surface_gravity: (cm.s-2) gravity of the planet
            star_effective_temperature: (K) surface effective temperature of the star
            star_radius: (cm) mean radius of the star
            orbit_semi_major_axis: (cm) distance between the planet and the star
            reference_pressure: (bar) reference pressure for the radius and the gravity of the planet
            bond_albedo: bond albedo of the planet
        """
        super().__init__(
            name=name,
            mass=mass,
            radius=radius,
            surface_gravity=surface_gravity,
            orbit_semi_major_axis=orbit_semi_major_axis,
            reference_pressure=reference_pressure,
            equilibrium_temperature=equilibrium_temperature,
            bond_albedo=bond_albedo,
            star_radius=star_radius,
            star_effective_temperature=star_effective_temperature
        )

        if equilibrium_temperature is None:
            self.equilibrium_temperature = self.calculate_planetary_equilibrium_temperature()[0]
        else:
            self.equilibrium_temperature = equilibrium_temperature

        if mass is None:
            self.mass = self.surface_gravity2mass(self.surface_gravity, self.radius)
        else:
            self.mass = mass

    @staticmethod
    def surface_gravity2mass(surface_gravity, radius, **kwargs):
        return surface_gravity * radius ** 2 / nc.G


class SpectralModel:
    default_line_species = [
        'CH4_main_iso',
        'CO_all_iso',
        'CO2_main_iso',
        'H2O_main_iso',
        'HCN_main_iso',
        'K',
        'Na_allard',
        'NH3_main_iso',
        'TiO_all_iso_exo',
        'VO'
    ]
    default_rayleigh_species = [
        'H2',
        'He'
    ]
    default_continuum_opacities = [
        'H2-H2',
        'H2-He'
    ]

    def __init__(self, planet_name, wavelength_boundaries, lbl_opacity_sampling, do_scat_emis,
                 t_int, metallicity, co_ratio, p_cloud, kappa_ir_z0=0.01, gamma=0.4, p_quench_c=None, haze_factor=1,
                 atmosphere_file=None, wavelengths=None, transit_radius=None, eclipse_depth=None,
                 spectral_radiosity=None, star_spectral_radiosity=None, opacity_mode='lbl',
                 heh2_ratio=0.324, use_equilibrium_chemistry=False,
                 temperature=None, mass_fractions=None, planet_model_file=None, model_suffix='', filename=None):
        self.planet_name = planet_name
        self.wavelength_boundaries = wavelength_boundaries
        self.lbl_opacity_sampling = lbl_opacity_sampling
        self.do_scat_emis = do_scat_emis
        self.opacity_mode = opacity_mode
        self.t_int = t_int
        self.metallicity = metallicity
        self.co_ratio = co_ratio
        self.p_cloud = p_cloud

        self.kappa_ir_z0 = kappa_ir_z0
        self.gamma = gamma
        self.p_quench_c = p_quench_c
        self.haze_factor = haze_factor

        self.atmosphere_file = atmosphere_file

        self.temperature = temperature
        self.mass_fractions = mass_fractions

        self.wavelengths = wavelengths
        self.transit_radius = transit_radius
        self.eclipse_depth = eclipse_depth
        self.spectral_radiosity = spectral_radiosity
        self.star_spectral_radiosity = star_spectral_radiosity

        self.heh2_ratio = heh2_ratio
        self.use_equilibrium_chemistry = use_equilibrium_chemistry

        self.name_suffix = model_suffix

        if planet_model_file is None:
            self.planet_model_file = Planet(planet_name).get_filename()
        else:
            self.planet_model_file = planet_model_file

        if filename is None:
            self.filename = self.get_filename()

    @staticmethod
    def _init_equilibrium_chemistry(pressures, temperatures, co_ratio, log10_metallicity,
                                    line_species, included_line_species,
                                    carbon_pressure_quench=None, mass_mixing_ratios=None):
        from petitRADTRANS.poor_mans_nonequ_chem import poor_mans_nonequ_chem as pm  # import is here because it is long to load

        if np.size(co_ratio) == 1:
            co_ratios = np.ones_like(pressures) * co_ratio
        else:
            co_ratios = co_ratio

        if np.size(log10_metallicity) == 1:
            log10_metallicities = np.ones_like(pressures) * log10_metallicity
        else:
            log10_metallicities = log10_metallicity

        abundances = pm.interpol_abundances(
            COs_goal_in=co_ratios,
            FEHs_goal_in=log10_metallicities,
            temps_goal_in=temperatures,
            pressures_goal_in=pressures,
            Pquench_carbon=carbon_pressure_quench
        )

        # Check mass_mixing_ratios keys
        for key in mass_mixing_ratios:
            if key not in line_species and key not in abundances:
                raise KeyError(f"key '{key}' not in retrieved species list or "
                               f"standard petitRADTRANS mass fractions dict")

        # Get the right keys for the mass fractions dictionary
        mass_mixing_ratios_dict = {}

        if included_line_species == 'all':
            included_line_species = copy.copy(line_species)

        for key in abundances:
            found = False

            # Set line species mass mixing ratios into to their imposed one
            for line_species_name in line_species:
                # Correct for line species name to match pRT chemistry name
                line_species_name = line_species_name.split('_', 1)[0]

                if line_species_name == 'C2H2':  # C2H2 special case
                    line_species_name += ',acetylene'

                if key == line_species_name:
                    if key not in included_line_species:
                        # Species not included, set mass mixing ratio to 0
                        mass_mixing_ratios_dict[line_species_name] = np.zeros(np.shape(temperatures))
                    elif line_species_name in mass_mixing_ratios:
                        # Use imposed mass mixing ratio
                        mass_mixing_ratios_dict[line_species_name] = 10 ** mass_mixing_ratios[line_species_name]
                    else:
                        # Use calculated mass mixing ratio
                        mass_mixing_ratios_dict[line_species_name] = abundances[line_species_name]

                    found = True

                    break

            # Set species mass mixing ratio to their imposed one
            if not found:
                if key in mass_mixing_ratios:
                    # Use imposed mass mixing ratio
                    mass_mixing_ratios_dict[key] = mass_mixing_ratios[key]
                else:
                    # Use calculated mass mixing ratio
                    mass_mixing_ratios_dict[key] = abundances[key]

        return mass_mixing_ratios_dict

    @staticmethod
    def _init_mass_mixing_ratios(pressures, line_species,
                                 included_line_species='all', temperatures=None, co_ratio=0.55, log10_metallicity=0,
                                 carbon_pressure_quench=None,
                                 imposed_mass_mixing_ratios=None, heh2_ratio=0.324324, use_equilibrium_chemistry=False):
        """Initialize a model mass mixing ratios.
        Ensure that in any case, the sum of mass mixing ratios is equal to 1. Imposed mass mixing ratios are kept to
        their value as much as possible.
        If the sum of mass mixing ratios of all imposed species is greater than 1, the mass mixing ratios will be scaled
        down, conserving the ratio between them. In that case, non-imposed mass mixing ratios are set to 0.
        If the sum of mass mixing ratio of all imposed species is less than 1, then if equilibrium chemistry is used or
        if H2 and He are imposed species, the atmosphere will be filled with H2 and He respecting the imposed H2/He
        ratio. Otherwise, the heh2_ratio parameter is used.
        When using equilibrium chemistry with imposed mass mixing ratios, imposed mass mixing ratios are set to their
        required value regardless of chemical equilibrium consistency.

        Args:
            pressures: (bar) pressures of the mass mixing ratios
            line_species: list of line species, required to manage naming differences between opacities and chemistry
            included_line_species: which line species of the list to include, mass mixing ratio set to 0 otherwise
            temperatures: (K) temperatures of the mass mixing ratios, used with equilibrium chemistry
            co_ratio: carbon over oxygen ratios of the model, used with equilibrium chemistry
            log10_metallicity: ratio between heavy elements and H2 + He compared to solar, used with equilibrium chemistry
            carbon_pressure_quench: (bar) pressure where the carbon species are quenched, used with equilibrium chemistry
            imposed_mass_mixing_ratios: imposed mass mixing ratios
            heh2_ratio: H2 over He mass mixing ratio
            use_equilibrium_chemistry: if True, use pRT equilibrium chemistry module

        Returns:
            A dictionary containing the mass mixing ratios.
        """
        # Initialization
        mass_mixing_ratios = {}
        m_sum_imposed_species = np.zeros(np.shape(pressures))
        m_sum_species = np.zeros(np.shape(pressures))

        # Initialize imposed mass mixing ratios
        if imposed_mass_mixing_ratios is not None:
            for species, mass_mixing_ratio in imposed_mass_mixing_ratios.items():
                if np.size(mass_mixing_ratio) == 1:
                    imposed_mass_mixing_ratios[species] = np.ones(np.shape(pressures)) * mass_mixing_ratio
                elif np.size(mass_mixing_ratio) != np.size(pressures):
                    raise ValueError(f"mass mixing ratio for species '{species}' must be a scalar or an array of the"
                                     f"size of the pressure array ({np.size(pressures)}), "
                                     f"but is of size ({np.size(mass_mixing_ratio)})")
        else:
            # Nothing is imposed
            imposed_mass_mixing_ratios = {}

        # Chemical equilibrium
        if use_equilibrium_chemistry:
            mass_mixing_ratios_equilibrium = SpectralModel._init_equilibrium_chemistry(
                pressures=pressures,
                temperatures=temperatures,
                co_ratio=co_ratio,
                log10_metallicity=log10_metallicity,
                line_species=line_species,
                included_line_species=included_line_species,
                carbon_pressure_quench=carbon_pressure_quench,
                mass_mixing_ratios=imposed_mass_mixing_ratios
            )

            if imposed_mass_mixing_ratios == {}:
                imposed_mass_mixing_ratios = copy.copy(mass_mixing_ratios_equilibrium)
        else:
            mass_mixing_ratios_equilibrium = None

        # Ensure that the sum of mass mixing ratios of imposed species is <= 1
        for species in imposed_mass_mixing_ratios:
            # Ignore the non-abundances coming from the chemistry module
            if species == 'nabla_ad' or species == 'MMW':
                continue

            spec = species.split('_R_')[0]  # deal with the naming scheme for binned down opacities
            mass_mixing_ratios[species] = imposed_mass_mixing_ratios[spec]
            m_sum_imposed_species += imposed_mass_mixing_ratios[spec]

        for i in range(np.size(m_sum_imposed_species)):
            if m_sum_imposed_species[i] > 1:
                # TODO changing retrieved mmr might come problematic in some retrievals (retrieved value not corresponding to actual value in model)
                print(f"Warning: sum of mass mixing ratios of imposed species ({m_sum_imposed_species}) is > 1, "
                      f"correcting...")

                for species in imposed_mass_mixing_ratios:
                    mass_mixing_ratios[species][i] /= m_sum_imposed_species[i]

        m_sum_imposed_species = np.sum(list(mass_mixing_ratios.values()), axis=0)

        # Get the sum of mass mixing ratios of non-imposed species
        if mass_mixing_ratios_equilibrium is None:
            # TODO this is assuming an H2-He atmosphere with line species, this could be more general
            species_list = copy.copy(line_species)
        else:
            species_list = list(mass_mixing_ratios_equilibrium.keys())

        for species in species_list:
            # Ignore the non-abundances coming from the chemistry module
            if species == 'nabla_ad' or species == 'MMW':
                continue

            # Search for imposed species
            found = False

            for key in imposed_mass_mixing_ratios:
                spec = key.split('_R_')[0]  # deal with the naming scheme for binned down opacities

                if species == spec:
                    found = True

                    break

            # Only take into account non-imposed species and ignore imposed species
            if not found:
                mass_mixing_ratios[species] = mass_mixing_ratios_equilibrium[species]
                m_sum_species += mass_mixing_ratios_equilibrium[species]

        # Ensure that the sum of mass mixing ratios of all species is = 1
        m_sum_total = m_sum_species + m_sum_imposed_species

        if np.any(np.logical_or(m_sum_total > 1, m_sum_total < 1)):
            # Search for H2 and He in both imposed and non-imposed species
            h2_found_in_mass_mixing_ratios = False
            he_found_in_mass_mixing_ratios = False
            h2_found_in_abundances = False
            he_found_in_abundances = False

            for key in imposed_mass_mixing_ratios:
                if key == 'H2':
                    h2_found_in_mass_mixing_ratios = True
                elif key == 'He':
                    he_found_in_mass_mixing_ratios = True

            for key in mass_mixing_ratios:
                if key == 'H2':
                    h2_found_in_abundances = True
                elif key == 'He':
                    he_found_in_abundances = True

            if not h2_found_in_abundances or not he_found_in_abundances:
                if not h2_found_in_abundances:
                    mass_mixing_ratios['H2'] = np.zeros(np.shape(pressures))

                if not he_found_in_abundances:
                    mass_mixing_ratios['He'] = np.zeros(np.shape(pressures))

            for i in range(np.size(m_sum_total)):
                if m_sum_total[i] > 1:
                    print(f"Warning: sum of species mass fraction ({m_sum_species[i]} + {m_sum_imposed_species[i]}) "
                          f"is > 1, correcting...")

                    for species in mass_mixing_ratios:
                        found = False

                        for key in imposed_mass_mixing_ratios:
                            if species == key:
                                found = True

                                break

                        if not found:
                            mass_mixing_ratios[species][i] = \
                                mass_mixing_ratios[species][i] * (1 - m_sum_imposed_species[i]) / m_sum_species[i]
                elif m_sum_total[i] < 1:
                    # Fill atmosphere with H2 and He
                    # TODO there might be a better filling species, N2?
                    if h2_found_in_mass_mixing_ratios and he_found_in_mass_mixing_ratios:
                        # Use imposed He/H2 ratio
                        heh2_ratio = 10 ** imposed_mass_mixing_ratios['He'][i] / 10 ** imposed_mass_mixing_ratios['H2'][i]

                    if h2_found_in_abundances and he_found_in_abundances:
                        # Use calculated He/H2 ratio
                        heh2_ratio = mass_mixing_ratios['He'][i] / mass_mixing_ratios['H2'][i]

                        mass_mixing_ratios['H2'][i] += (1 - m_sum_total[i]) / (1 + heh2_ratio)
                        mass_mixing_ratios['He'][i] = mass_mixing_ratios['H2'][i] * heh2_ratio
                    else:
                        # Remove H2 and He mass mixing ratios from total for correct mass mixing ratio calculation
                        if h2_found_in_abundances:
                            m_sum_total[i] -= mass_mixing_ratios['H2'][i]
                        elif he_found_in_abundances:
                            m_sum_total[i] -= mass_mixing_ratios['He'][i]

                        # Use He/H2 ratio in argument
                        mass_mixing_ratios['H2'][i] = (1 - m_sum_total[i]) / (1 + heh2_ratio)
                        mass_mixing_ratios['He'][i] = mass_mixing_ratios['H2'][i] * heh2_ratio

        return mass_mixing_ratios

    @staticmethod
    def _init_model(atmosphere: Radtrans, parameters: dict):
        """Initialize the temperature profile, mass mixing ratios and mean molar mass of a model.

        Args:
            atmosphere: an instance of Radtrans object
            parameters: dictionary of parameters

        Returns:
            The temperature, mass mixing ratio and mean molar mass at each pressure as 1D-arrays
        """
        pressures = atmosphere.press * 1e-6  # bar to cgs

        if parameters['intrinsic_temperature'].value is not None:
            temperatures = SpectralModel._init_temperature_profile_guillot(
                pressures=pressures,
                gamma=parameters['guillot_temperature_profile_gamma'].value,
                surface_gravity=10 ** parameters['log10_surface_gravity'].value,
                intrinsic_temperature=parameters['intrinsic_temperature'].value,
                equilibrium_temperature=parameters['temperature'].value,
                kappa_ir_z0=parameters['guillot_temperature_profile_kappa_ir_z0'].value,
                metallicity=10 ** parameters['log10_metallicity'].value
            )
        elif isinstance(parameters['temperature'].value, (float, int)):
            temperatures = np.ones(np.shape(atmosphere.press)) * parameters['temperature'].value
        elif np.size(parameters['temperature'].value) == np.size(pressures):
            temperatures = np.asarray(parameters['temperature'].value)
        else:
            raise ValueError(f"could not initialize temperature profile; "
                             f"possible inputs are float, int, "
                             f"or a 1-D array of the same size of parameter 'pressures' ({np.size(atmosphere.press)})")

        imposed_mass_mixing_ratios = {}

        for species in atmosphere.line_species:
            # TODO mass mixing ratio dict initialization more general
            spec = species.split('_R_')[0]  # deal with the naming scheme for binned down opacities
            # Convert from log-abundance
            imposed_mass_mixing_ratios[species] = 10 ** parameters[spec].value * np.ones_like(pressures)

        mass_mixing_ratios = SpectralModel._init_mass_mixing_ratios(
            pressures=pressures,
            line_species=atmosphere.line_species,
            included_line_species=parameters['included_line_species'].value,
            temperatures=temperatures,
            co_ratio=parameters['co_ratio'].value,
            log10_metallicity=parameters['log10_metallicity'].value,
            carbon_pressure_quench=parameters['carbon_pressure_quench'].value,
            imposed_mass_mixing_ratios=imposed_mass_mixing_ratios,
            heh2_ratio=parameters['heh2_ratio'].value,
            use_equilibrium_chemistry=parameters['use_equilibrium_chemistry'].value
        )

        # Find the mean molar mass in each layer
        mean_molar_mass = calc_MMW(mass_mixing_ratios)

        return temperatures, mass_mixing_ratios, mean_molar_mass

    @staticmethod
    def _get_parameters_dict(surface_gravity, planet_radius=None, reference_pressure=1e-2,
                             temperature=None, mass_mixing_ratios=None, cloud_pressure=None,
                             guillot_temperature_profile_gamma=0.4, guillot_temperature_profile_kappa_ir_z0=0.01,
                             included_line_species=None, intrinsic_temperature=None, heh2_ratio=0.324,
                             use_equilibrium_chemistry=False,
                             co_ratio=0.55, metallicity=1.0, carbon_pressure_quench=None,
                             star_effective_temperature=None, star_radius=None, star_spectral_radiosity=None,
                             planet_max_radial_orbital_velocity=None, planet_orbital_inclination=None,
                             semi_major_axis=None,
                             planet_rest_frame_shift=0.0, orbital_phases=None, system_observer_radial_velocities=None,
                             wavelengths_instrument=None, instrument_resolving_power=None,
                             data=None, data_uncertainties=None,
                             reduced_data=None, reduced_data_uncertainties=None, reduction_matrix=None,
                             airmass=None, telluric_transmittance=None, variable_throughput=None
                             ):
        # Conversions to log-space
        if cloud_pressure is not None:
            cloud_pressure = np.log10(cloud_pressure)

        if metallicity is not None:
            metallicity = np.log10(metallicity)

        if surface_gravity is not None:
            surface_gravity = np.log10(surface_gravity)

        # TODO expand to include all possible parameters of transm and calc_flux
        parameters = {
            'airmass': Param(airmass),
            'carbon_pressure_quench': Param(carbon_pressure_quench),
            'co_ratio': Param(co_ratio),
            'data': Param(data),
            'data_uncertainties': Param(data_uncertainties),
            'guillot_temperature_profile_gamma': Param(guillot_temperature_profile_gamma),
            'guillot_temperature_profile_kappa_ir_z0': Param(guillot_temperature_profile_kappa_ir_z0),
            'heh2_ratio': Param(heh2_ratio),
            'included_line_species': Param(included_line_species),
            'instrument_resolving_power': Param(instrument_resolving_power),
            'intrinsic_temperature': Param(intrinsic_temperature),
            'log10_cloud_pressure': Param(cloud_pressure),
            'log10_metallicity': Param(metallicity),
            'log10_surface_gravity': Param(surface_gravity),
            'orbital_phases': Param(orbital_phases),
            'planet_max_radial_orbital_velocity': Param(planet_max_radial_orbital_velocity),
            'planet_radius': Param(planet_radius),
            'planet_rest_frame_shift': Param(planet_rest_frame_shift),
            'planet_orbital_inclination': Param(planet_orbital_inclination),
            'reduced_data': Param(reduced_data),
            'reduction_matrix': Param(reduction_matrix),
            'reduced_data_uncertainties': Param(reduced_data_uncertainties),
            'reference_pressure': Param(reference_pressure),
            'semi_major_axis': Param(semi_major_axis),
            'star_effective_temperature': Param(star_effective_temperature),
            'star_radius': Param(star_radius),
            'star_spectral_radiosity': Param(star_spectral_radiosity),
            'system_observer_radial_velocities': Param(system_observer_radial_velocities),
            'telluric_transmittance': Param(telluric_transmittance),
            'temperature': Param(temperature),
            'use_equilibrium_chemistry': Param(use_equilibrium_chemistry),
            'variable_throughput': Param(variable_throughput),
            'wavelengths_instrument': Param(wavelengths_instrument),
        }

        if mass_mixing_ratios is None:
            mass_mixing_ratios = {}

        for species, mass_mixing_ratio in mass_mixing_ratios.items():
            parameters[species] = Param(np.log10(mass_mixing_ratio))

        return parameters

    @staticmethod
    def _init_temperature_profile_guillot(pressures, gamma, surface_gravity,
                                          intrinsic_temperature, equilibrium_temperature,
                                          kappa_ir_z0=None, metallicity=None):
        if metallicity is not None:
            kappa_ir = kappa_ir_z0 * metallicity
        else:
            kappa_ir = kappa_ir_z0

        temperatures = guillot_global(
            pressure=pressures,
            kappa_ir=kappa_ir,
            gamma=gamma,
            grav=surface_gravity,
            t_int=intrinsic_temperature,
            t_equ=equilibrium_temperature
        )

        return temperatures

    @staticmethod
    def _spectral_radiosity_model(atmosphere: Radtrans, parameters: dict):
        temperatures, mass_mixing_ratios, mean_molar_mass = SpectralModel._init_model(
            atmosphere=atmosphere,
            parameters=parameters
        )

        # Calculate the spectrum
        atmosphere.calc_flux(
            temp=temperatures,
            abunds=mass_mixing_ratios,
            gravity=10 ** parameters['log10_surface_gravity'].value,
            mmw=mean_molar_mass,
            Tstar=parameters['star_effective_temperature'].value,
            Rstar=parameters['star_radius'].value,
            semimajoraxis=parameters['semi_major_axis'].value,
            Pcloud=10 ** parameters['log10_cloud_pressure'].value,
            # stellar_intensity=parameters['star_spectral_radiosity'].value
        )

        # Transform the outputs into the units of our data.
        planet_radiosity = SpectralModel.radiosity_erg_hz2radiosity_erg_cm(atmosphere.flux, atmosphere.freq)
        wlen_model = nc.c / atmosphere.freq * 1e4  # cm to um

        return wlen_model, planet_radiosity

    @staticmethod
    def _transit_radius_model(atmosphere: Radtrans, parameters: dict):
        temperatures, mass_mixing_ratios, mean_molar_mass = SpectralModel._init_model(
            atmosphere=atmosphere,
            parameters=parameters
        )

        # Calculate the spectrum
        atmosphere.calc_transm(
            temp=temperatures,
            abunds=mass_mixing_ratios,
            gravity=10 ** parameters['log10_surface_gravity'].value,
            mmw=mean_molar_mass,
            P0_bar=parameters['reference_pressure'].value,
            R_pl=parameters['planet_radius'].value
        )

        # Transform the outputs into the units of our data.
        planet_transit_radius = atmosphere.transm_rad
        wavelengths = nc.c / atmosphere.freq * 1e4  # cm to um

        return wavelengths, planet_transit_radius

    def calculate_transit_radius(self, planet: Planet, atmosphere: Radtrans = None, pressures=None,
                                 line_species=None, rayleigh_species=None, continuum_opacities=None):
        if line_species is None:
            line_species = self.default_line_species

        if rayleigh_species is None:
            rayleigh_species = self.default_rayleigh_species

        if continuum_opacities is None:
            continuum_opacities = self.default_continuum_opacities

        if atmosphere is None:
            atmosphere = self.init_atmosphere(
                pressures=pressures,
                wlen_bords_micron=self.wavelength_boundaries,
                line_species_list=line_species,
                rayleigh_species=rayleigh_species,
                continuum_opacities=continuum_opacities,
                lbl_opacity_sampling=self.lbl_opacity_sampling,
                do_scat_emis=self.do_scat_emis,
                mode=self.opacity_mode
            )

        parameters = self.get_parameters_dict(planet)

        wavelengths, transit_radius = self._transit_radius_model(
            atmosphere=atmosphere,
            parameters=parameters
        )

        self.wavelengths = wavelengths
        self.transit_radius = transit_radius

        # Initialized afterward because we need wavelengths first!
        # TODO find a way to prevent that
        parameters['star_spectral_radiosity'] = Param(self.get_phoenix_star_spectral_radiosity(planet))

        return wavelengths, transit_radius

    def calculate_spectral_radiosity(self, planet: Planet, atmosphere: Radtrans = None, pressures=None,
                                     line_species=None, rayleigh_species=None, continuum_opacities=None):
        if line_species is None:
            line_species = self.default_line_species

        if rayleigh_species is None:
            rayleigh_species = self.default_rayleigh_species

        if continuum_opacities is None:
            continuum_opacities = self.default_continuum_opacities

        if atmosphere is None:
            atmosphere = self.init_atmosphere(
                pressures=pressures,
                wlen_bords_micron=self.wavelength_boundaries,
                line_species_list=line_species,
                rayleigh_species=rayleigh_species,
                continuum_opacities=continuum_opacities,
                lbl_opacity_sampling=self.lbl_opacity_sampling,
                do_scat_emis=self.do_scat_emis,
                mode=self.opacity_mode
            )

        parameters = self.get_parameters_dict(planet)

        wavelengths, spectral_radiosity = self._spectral_radiosity_model(
            atmosphere=atmosphere,
            parameters=parameters
        )

        self.wavelengths = wavelengths
        self.spectral_radiosity = spectral_radiosity

        # Initialized afterward because we need wavelengths first!
        # TODO find a way to prevent that
        parameters['star_spectral_radiosity'] = Param(self.get_phoenix_star_spectral_radiosity(planet))

        return wavelengths, spectral_radiosity

    def calculate_eclipse_depth(self, atmosphere: Radtrans, planet: Planet, star_radiosity_filename=None):
        if star_radiosity_filename is None:
            star_radiosity_filename = self.get_star_radiosity_filename(
                planet.star_effective_temperature, path=module_dir
            )

        if not os.path.isfile(star_radiosity_filename):
            self.generate_phoenix_star_spectrum_file(star_radiosity_filename, planet.star_effective_temperature)

        data = np.loadtxt(star_radiosity_filename)
        star_wavelength = data[:, 0] * 1e6  # m to um
        star_radiosities = data[:, 1] * 1e8 * np.pi  # erg.s-1.cm-2.sr-1/A to erg.s-1.cm-2/cm

        print('Calculating eclipse depth...')
        # TODO fix stellar flux calculated multiple time if do_scat_emis is True
        wavelengths, planet_radiosity = self.calculate_emission_spectrum(atmosphere, planet)
        star_radiosities = fr.rebin_spectrum(star_wavelength, star_radiosities, wavelengths)

        eclipse_depth = (planet_radiosity * planet.radius ** 2) / (star_radiosities * planet.star_radius ** 2)

        return wavelengths, eclipse_depth, planet_radiosity

    def calculate_emission_spectrum(self, atmosphere: Radtrans, planet: Planet):
        print('Calculating emission spectrum...')

        atmosphere.calc_flux(
            self.temperature,
            self.mass_fractions,
            planet.surface_gravity,
            self.mass_fractions['MMW'],
            Tstar=planet.star_effective_temperature,
            Rstar=planet.star_radius,
            semimajoraxis=planet.orbit_semi_major_axis,
            Pcloud=self.p_cloud
        )

        flux = self.radiosity_erg_hz2radiosity_erg_cm(atmosphere.flux, atmosphere.freq)
        wavelengths = nc.c / atmosphere.freq * 1e4  # cm to um

        return wavelengths, flux

    def calculate_transmission_spectrum(self, atmosphere: Radtrans, planet: Planet):
        print('Calculating transmission spectrum...')
        # TODO better transmission spectrum with Doppler shift, RM effect, limb-darkening effect (?)
        # Doppler shift should be low, RM effect and limb-darkening might be removed by the pipeline
        atmosphere.calc_transm(
            self.temperature,
            self.mass_fractions,
            planet.surface_gravity,
            self.mass_fractions['MMW'],
            R_pl=planet.radius,
            P0_bar=planet.reference_pressure,
            Pcloud=self.p_cloud,
            haze_factor=self.haze_factor,
        )

        transit_radius = (atmosphere.transm_rad / planet.star_radius) ** 2
        wavelengths = nc.c / atmosphere.freq * 1e4  # m to um

        return wavelengths, transit_radius

    @staticmethod
    def generate_phoenix_star_spectrum_file(star_spectrum_file, star_effective_temperature):
        stellar_spectral_radiance = get_PHOENIX_spec(star_effective_temperature)

        # Convert the spectrum to units accepted by the ETC website
        # Don't take the first wavelength to avoid spike in convolution
        wavelength_stellar = \
            stellar_spectral_radiance[1:, 0]  # in cm
        stellar_spectral_radiance = SpectralModel.radiosity_erg_hz2radiosity_erg_cm(
            stellar_spectral_radiance[1:, 1],
            nc.c / wavelength_stellar  # cm to Hz
        )

        wavelength_stellar *= 1e-2  # cm to m
        stellar_spectral_radiance *= 1e-8 / np.pi  # erg.s-1.cm-2/cm to erg.s-1.cm-2.sr-1/A

        np.savetxt(star_spectrum_file, np.transpose((wavelength_stellar, stellar_spectral_radiance)))

    def get_filename(self):
        name = self.get_name()

        return planet_models_directory + os.path.sep + name + '.pkl'

    def get_parameters_dict(self, planet: Planet, included_line_species='all'):
        # star_spectral_radiosity = self.get_phoenix_star_spectral_radiosity(planet)
        planet_max_radial_orbital_velocity = planet.calculate_orbital_velocity(
            planet.star_mass, planet.orbit_semi_major_axis
        )

        return self._get_parameters_dict(
            surface_gravity=planet.surface_gravity,
            planet_radius=planet.radius,
            reference_pressure=planet.reference_pressure,
            temperature=self.temperature,
            mass_mixing_ratios=self.mass_fractions,
            cloud_pressure=self.p_cloud,
            guillot_temperature_profile_gamma=self.gamma,
            guillot_temperature_profile_kappa_ir_z0=self.kappa_ir_z0,
            included_line_species=included_line_species,
            intrinsic_temperature=self.t_int,
            heh2_ratio=self.heh2_ratio,
            use_equilibrium_chemistry=self.use_equilibrium_chemistry,
            co_ratio=self.co_ratio,
            metallicity=10 ** self.metallicity,
            carbon_pressure_quench=self.p_quench_c,
            star_effective_temperature=planet.star_effective_temperature,
            star_radius=planet.star_radius,
            # star_spectral_radiosity=star_spectral_radiosity,
            planet_max_radial_orbital_velocity=planet_max_radial_orbital_velocity,
            planet_orbital_inclination=planet.orbital_inclination,
            semi_major_axis=planet.orbit_semi_major_axis,
            planet_rest_frame_shift=0.0,
            orbital_phases=None,
            system_observer_radial_velocities=None,
            wavelengths_instrument=None,
            instrument_resolving_power=None,
            data=None,
            data_uncertainties=None,
            reduced_data=None,
            reduced_data_uncertainties=None,
            reduction_matrix=None,
            airmass=None,
            telluric_transmittance=None,
            variable_throughput=None
        )

    @staticmethod
    def _get_phoenix_star_spectral_radiosity(star_effective_temperature, wavelengths):
        star_data = get_PHOENIX_spec(star_effective_temperature)
        star_data[:, 1] = SpectralModel.radiosity_erg_hz2radiosity_erg_cm(
            star_data[:, 1], nc.c / star_data[:, 0]  # cm to Hz
        )

        star_data[:, 0] *= 1e4  # cm to um

        star_radiosities = fr.rebin_spectrum(
            star_data[:, 0],
            star_data[:, 1],
            wavelengths
        )

        return star_radiosities

    def get_phoenix_star_spectral_radiosity(self, planet: Planet):
        return self._get_phoenix_star_spectral_radiosity(planet.star_effective_temperature, self.wavelengths)

    def get_name(self):
        name = 'spectral_model_'
        name += f"{self.planet_name.replace(' ', '_')}_" \
                f"Tint{self.t_int}K_Z{self.metallicity}_co{self.co_ratio}_pc{self.p_cloud}bar_" \
                f"{self.wavelength_boundaries[0]}-{self.wavelength_boundaries[1]}um_ds{self.lbl_opacity_sampling}"

        if self.do_scat_emis:
            name += '_scat'
        else:
            name += '_noscat'

        if self.name_suffix != '':
            name += f'_{self.name_suffix}'

        return name

    @staticmethod
    def get_star_radiosity_filename(star_effective_temperature, path='.'):
        return f'{path}/crires/star_spectrum_{star_effective_temperature}K.dat'

    def init_mass_fractions(self, atmosphere, temperature, include_species, mass_fractions=None):
        from petitRADTRANS.poor_mans_nonequ_chem import poor_mans_nonequ_chem as pm  # import is here because it's long to load

        if mass_fractions is None:
            mass_fractions = {}
        elif not isinstance(mass_fractions, dict):
            raise ValueError(
                f"mass fractions must be in a dict, but the input was of type '{type(mass_fractions)}'")

        pressures = atmosphere.press * 1e-6  # cgs to bar

        if np.size(self.co_ratio) == 1:
            co_ratios = np.ones_like(pressures) * self.co_ratio
        else:
            co_ratios = self.co_ratio

        if np.size(self.metallicity) == 1:
            metallicity = np.ones_like(pressures) * self.metallicity
        else:
            metallicity = self.metallicity

        abundances = pm.interpol_abundances(
            COs_goal_in=co_ratios,
            FEHs_goal_in=metallicity,
            temps_goal_in=temperature,
            pressures_goal_in=pressures,
            Pquench_carbon=self.p_quench_c
        )

        # Check mass_mixing_ratios keys
        for key in mass_fractions:
            if key not in atmosphere.line_species and key not in abundances:
                raise KeyError(f"key '{key}' not in line species list or "
                               f"standard petitRADTRANS mass fractions dict")

        # Get the right keys for the mass fractions dictionary
        mass_fractions_dict = {}

        for key in abundances:
            found = False

            for line_species_name in atmosphere.line_species:
                line_species = line_species_name.split('_', 1)[0]

                if line_species == 'C2H2':   # C2H2 special case
                    line_species += ',acetylene'

                if key == line_species:
                    if key not in include_species:
                        mass_fractions_dict[line_species_name] = np.zeros_like(temperature)
                    elif line_species_name in mass_fractions:
                        mass_fractions_dict[line_species_name] = mass_fractions[line_species_name]
                    else:
                        mass_fractions_dict[line_species_name] = abundances[line_species]

                    found = True

                    break

            if not found:
                if key in mass_fractions:
                    mass_fractions_dict[key] = mass_fractions[key]
                else:
                    mass_fractions_dict[key] = abundances[key]

        for key in mass_fractions:
            if key not in mass_fractions_dict:
                if key not in include_species:
                    mass_fractions_dict[key] = np.zeros_like(temperature)
                else:
                    mass_fractions_dict[key] = mass_fractions[key]

        return mass_fractions_dict

    def init_temperature_guillot(self, planet: Planet, atmosphere: Radtrans):
        pressures = atmosphere.press * 1e-6  # cgs to bar
        temperatures = self._init_temperature_profile_guillot(
            pressures=pressures,
            gamma=self.gamma,
            surface_gravity=planet.surface_gravity,
            intrinsic_temperature=self.t_int,
            equilibrium_temperature=planet.equilibrium_temperature,
            kappa_ir_z0=self.kappa_ir_z0,
            metallicity=10 ** self.metallicity
        )

        return temperatures

    def save(self):
        with open(self.get_filename(), 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, file):
        with open(file, 'rb') as f:
            return pickle.load(f)

    @classmethod
    def get(cls, planet_name, wavelength_boundaries, lbl_opacity_sampling, pressures, do_scat_emis, t_int,
            metallicity, co_ratio, p_cloud, kappa_ir_z0=0.01, gamma=0.4, p_quench_c=None, haze_factor=1,
            line_species_list='default', rayleigh_species='default', continuum_opacities='default',
            include_species='all', model_suffix='', atmosphere=None, calculate_transmission_spectrum=False,
            calculate_emission_spectrum=False, calculate_eclipse_depth=False,
            rewrite=True):
        # Initialize model
        model = cls.species_init(
            include_species=include_species,
            planet_name=planet_name,
            wavelength_boundaries=wavelength_boundaries,
            lbl_opacity_sampling=lbl_opacity_sampling,
            do_scat_emis=do_scat_emis,
            t_int=t_int,
            metallicity=metallicity,
            co_ratio=co_ratio,
            p_cloud=p_cloud,
            kappa_ir_z0=kappa_ir_z0,
            gamma=gamma,
            p_quench_c=p_quench_c,
            haze_factor=haze_factor,
            model_suffix=model_suffix
        )

        # Generate or load model
        return cls.generate_from(
            model=model,
            pressures=pressures,
            line_species_list=line_species_list,
            rayleigh_species=rayleigh_species,
            continuum_opacities=continuum_opacities,
            include_species=include_species,
            model_suffix=model_suffix,
            atmosphere=atmosphere,
            calculate_transmission_spectrum=calculate_transmission_spectrum,
            calculate_emission_spectrum=calculate_emission_spectrum,
            calculate_eclipse_depth=calculate_eclipse_depth,
            rewrite=rewrite
        )

    @classmethod
    def generate_from(cls, model, pressures,
                      line_species_list='default', rayleigh_species='default', continuum_opacities='default',
                      include_species=None, model_suffix='',
                      atmosphere=None, temperature_profile=None, mass_fractions=None,
                      calculate_transmission_spectrum=False, calculate_emission_spectrum=False,
                      calculate_eclipse_depth=False,
                      rewrite=False):
        if not hasattr(include_species, '__iter__') or isinstance(include_species, str):
            include_species = [include_species]
        elif include_species is None:
            include_species = ['all']

        if len(include_species) > 1:
            raise ValueError("Please include either only one species or all of them using keyword 'all'")

        # Check if model already exists
        if os.path.isfile(model.filename) and not rewrite:
            print(f"Model '{model.filename}' already exists, loading from file...")
            return model.load(model.filename)
        else:
            if os.path.isfile(model.filename) and rewrite:
                print(f"Rewriting already existing model '{model.filename}'...")

            print(f"Generating model '{model.filename}'...")

            # Initialize species
            if line_species_list == 'default':
                line_species_list = cls.default_line_species

            if rayleigh_species == 'default':
                rayleigh_species = cls.default_rayleigh_species

            if continuum_opacities == 'default':
                continuum_opacities = cls.default_continuum_opacities

            if include_species == ['all']:
                include_species = []

                for species_name in line_species_list:
                    if species_name == 'CO_36':
                        include_species.append(species_name)
                    else:
                        include_species.append(species_name.split('_', 1)[0])

            # Generate the model
            return cls._generate(
                model, pressures, line_species_list, rayleigh_species, continuum_opacities, include_species,
                model_suffix, atmosphere, temperature_profile, mass_fractions, calculate_transmission_spectrum,
                calculate_emission_spectrum, calculate_eclipse_depth
            )

    @staticmethod
    def radiosity_erg_cm2radiosity_erg_hz(radiosity_erg_cm, wavelength):
        """
        Convert a radiosity from erg.s-1.cm-2.sr-1/cm to erg.s-1.cm-2.sr-1/Hz at a given wavelength.
        Steps:
            [cm] = c[cm.s-1] / [Hz]
            => d[cm]/d[Hz] = d(c / [Hz])/d[Hz]
            => d[cm]/d[Hz] = c / [Hz]**2
            integral of flux must be conserved: radiosity_erg_cm * d[cm] = radiosity_erg_hz * d[Hz]
            radiosity_erg_hz = radiosity_erg_cm * d[cm]/d[Hz]
            => radiosity_erg_hz = radiosity_erg_cm * wavelength**2 / c

        Args:
            radiosity_erg_cm: (erg.s-1.cm-2.sr-1/cm)
            wavelength: (cm)

        Returns:
            (erg.s-1.cm-2.sr-1/cm) the radiosity in converted units
        """
        return radiosity_erg_cm * wavelength ** 2 / nc.c

    @staticmethod
    def radiosity_erg_hz2radiosity_erg_cm(radiosity_erg_hz, frequency):
        """
        Convert a radiosity from erg.s-1.cm-2.sr-1/Hz to erg.s-1.cm-2.sr-1/cm at a given frequency.
        Steps:
            [cm] = c[cm.s-1] / [Hz]
            => d[cm]/d[Hz] = d(c / [Hz])/d[Hz]
            => d[cm]/d[Hz] = c / [Hz]**2
            => d[Hz]/d[cm] = [Hz]**2 / c
            integral of flux must be conserved: radiosity_erg_cm * d[cm] = radiosity_erg_hz * d[Hz]
            radiosity_erg_cm = radiosity_erg_hz * d[Hz]/d[cm]
            => radiosity_erg_cm = radiosity_erg_hz * frequency**2 / c

        Args:
            radiosity_erg_hz: (erg.s-1.cm-2.sr-1/Hz)
            frequency: (Hz)

        Returns:
            (erg.s-1.cm-2.sr-1/cm) the radiosity in converted units
        """
        return radiosity_erg_hz * frequency ** 2 / nc.c

    @classmethod
    def species_init(cls, include_species, planet_name, wavelength_boundaries, lbl_opacity_sampling, do_scat_emis,
                     t_int, metallicity, co_ratio, p_cloud, kappa_ir_z0=0.01, gamma=0.4, p_quench_c=None, haze_factor=1,
                     atmosphere_file=None, wavelengths=None, transit_radius=None, temperature=None,
                     mass_fractions=None, planet_model_file=None, model_suffix='', filename=None):
        # Initialize include_species
        if not hasattr(include_species, '__iter__') or isinstance(include_species, str):
            include_species = [include_species]

        if len(include_species) > 1:
            raise ValueError("Please include either only one species or all of them using keyword 'all'")
        else:
            if model_suffix == '':
                species_suffix = f'{include_species[0]}'
            else:
                species_suffix = f'_{include_species[0]}'

        # Initialize model
        return cls(
            planet_name=planet_name,
            wavelength_boundaries=wavelength_boundaries,
            lbl_opacity_sampling=lbl_opacity_sampling,
            do_scat_emis=do_scat_emis,
            t_int=t_int,
            metallicity=metallicity,
            co_ratio=co_ratio,
            p_cloud=p_cloud,
            kappa_ir_z0=kappa_ir_z0,
            gamma=gamma,
            p_quench_c=p_quench_c,
            haze_factor=haze_factor,
            atmosphere_file=atmosphere_file,
            wavelengths=wavelengths,
            transit_radius=transit_radius,
            temperature=temperature,
            mass_fractions=mass_fractions,
            planet_model_file=planet_model_file,
            model_suffix=model_suffix + species_suffix,
            filename=filename
        )

    @staticmethod
    def _generate(model, pressures, line_species_list, rayleigh_species, continuum_opacities, include_species,
                  model_suffix, atmosphere=None, temperature_profile=None, mass_fractions=None,
                  calculate_transmission_spectrum=False, calculate_emission_spectrum=False,
                  calculate_eclipse_depth=False):
        if atmosphere is None:
            atmosphere, model.atmosphere_file = model.get_atmosphere_model(
                model.wavelength_boundaries, pressures, line_species_list, rayleigh_species, continuum_opacities,
                model_suffix
            )
        else:
            model.atmosphere_file = SpectralModel._get_hires_atmosphere_filename(
                pressures, model.wavelength_boundaries, model.lbl_opacity_sampling, model_suffix
            )

        # A Planet needs to be generated and saved first
        model.planet_model_file = Planet.generate_filename(model.planet_name)
        planet = Planet.load(model.planet_name, model.planet_model_file)

        if temperature_profile is None:
            model.temperature = model.init_temperature_guillot(
                planet=planet,
                atmosphere=atmosphere
            )
        elif isinstance(temperature_profile, (float, int)):
            model.temperature = np.ones_like(atmosphere.press) * temperature_profile
        elif np.size(temperature_profile) == np.size(atmosphere.press):
            model.temperature = np.asarray(temperature_profile)
        else:
            raise ValueError(f"could not initialize temperature profile using input {temperature_profile}; "
                             f"possible inputs are None, float, int, "
                             f"or a 1-D array of the same size of argument 'pressures' ({np.size(atmosphere.press)})")

        # Generate mass fractions from equilibrium chemistry first to have all the keys
        # TODO generate the mass fractions dict without calling equilibrium chemistry
        model.mass_fractions = model.init_mass_fractions(
            atmosphere=atmosphere,
            temperature=model.temperature,
            include_species=include_species,
            mass_fractions=mass_fractions
        )

        if not calculate_transmission_spectrum and not calculate_emission_spectrum and not calculate_eclipse_depth:
            print(f"No spectrum will be calculated")

            return model

        if calculate_transmission_spectrum:
            model.wavelengths, model.transit_radius = model.calculate_transmission_spectrum(
                atmosphere=atmosphere,
                planet=planet
            )

        if calculate_emission_spectrum and not calculate_eclipse_depth:
            model.wavelengths, model.spectral_radiosity = model.calculate_emission_spectrum(
                atmosphere=atmosphere,
                planet=planet
            )
        elif calculate_eclipse_depth:
            model.wavelengths, model.eclipse_depth, model.spectral_radiosity = model.calculate_eclipse_depth(
                atmosphere=atmosphere,
                planet=planet
            )

        return model

    @staticmethod
    def _get_hires_atmosphere_filename(pressures, wlen_bords_micron, lbl_opacity_sampling, do_scat_emis,
                                       model_suffix=''):
        filename = planet_models_directory + os.path.sep \
                   + f"atmosphere_{np.max(pressures)}-{np.min(pressures)}bar_" \
                     f"{wlen_bords_micron[0]}-{wlen_bords_micron[1]}um_ds{lbl_opacity_sampling}"

        if do_scat_emis:
            filename += '_scat'

        if model_suffix != '':
            filename += f"_{model_suffix}"

        filename += '.pkl'

        return filename

    @staticmethod
    def get_atmosphere_model(wlen_bords_micron, pressures,
                             line_species_list=None, rayleigh_species=None, continuum_opacities=None,
                             lbl_opacity_sampling=1, do_scat_emis=False, save=False,
                             model_suffix=''):
        atmosphere_filename = SpectralModel._get_hires_atmosphere_filename(
            pressures, wlen_bords_micron, lbl_opacity_sampling, do_scat_emis, model_suffix
        )

        if os.path.isfile(atmosphere_filename):
            print('Loading atmosphere model...')
            with open(atmosphere_filename, 'rb') as f:
                atmosphere = pickle.load(f)
        else:
            atmosphere = SpectralModel.init_atmosphere(
                pressures, wlen_bords_micron, line_species_list, rayleigh_species, continuum_opacities,
                lbl_opacity_sampling, do_scat_emis
            )

            if save:
                print('Saving atmosphere model...')
                with open(atmosphere_filename, 'wb') as f:
                    pickle.dump(atmosphere, f)

        return atmosphere, atmosphere_filename

    @staticmethod
    def init_atmosphere(pressures, wlen_bords_micron, line_species_list, rayleigh_species, continuum_opacities,
                        lbl_opacity_sampling, do_scat_emis, mode='lbl'):
        print('Generating atmosphere...')

        atmosphere = Radtrans(
            line_species=line_species_list,
            rayleigh_species=rayleigh_species,
            continuum_opacities=continuum_opacities,
            wlen_bords_micron=wlen_bords_micron,
            mode=mode,
            do_scat_emis=do_scat_emis,
            lbl_opacity_sampling=lbl_opacity_sampling
        )

        atmosphere.setup_opa_structure(pressures)

        return atmosphere


class SpectralModel2(BaseSpectralModel):
    default_line_species = [
        'CH4_main_iso',
        'CO_all_iso',
        'CO2_main_iso',
        'H2O_main_iso',
        'HCN_main_iso',
        'K',
        'Na_allard',
        'NH3_main_iso',
        'TiO_all_iso_exo',
        'VO'
    ]
    default_rayleigh_species = [
        'H2',
        'He'
    ]
    default_continuum_opacities = [
        'H2-H2',
        'H2-He'
    ]

    def __init__(self, pressures,
                 line_species=None, rayleigh_species=None, continuum_opacities=None, cloud_species=None,
                 opacity_mode='lbl', do_scat_emis=True, lbl_opacity_sampling=1,
                 temperatures=None, mass_mixing_ratios=None, mean_molar_masses=None,
                 wavelengths_boundaries=None, wavelengths=None, transit_radii=None, spectral_radiosities=None,
                 times=None, **model_parameters):
        super().__init__(
            wavelengths_boundaries=wavelengths_boundaries,
            pressures=pressures,
            temperatures=temperatures,
            mass_mixing_ratios=mass_mixing_ratios,
            mean_molar_masses=mean_molar_masses,
            line_species=line_species,
            rayleigh_species=rayleigh_species,
            continuum_opacities=continuum_opacities,
            cloud_species=cloud_species,
            opacity_mode=opacity_mode,
            do_scat_emis=do_scat_emis,
            lbl_opacity_sampling=lbl_opacity_sampling,
            wavelengths=wavelengths,
            transit_radii=transit_radii,
            spectral_radiosities=spectral_radiosities,
            times=times,
            **model_parameters
        )

    @staticmethod
    def __calculate_metallicity_wrap(metallicity=None, log10_metallicity=None,
                                     planet_mass=None, planet_radius=None, planet_surface_gravity=None,
                                     star_metallicity=1.0, atmospheric_mixing=1.0, alpha=-0.68, beta=7.2,
                                     verbose=False, **kwargs):
        if log10_metallicity is None:
            if metallicity is None:
                if verbose:
                    print(f"log10 metallicity set to None, calculating it using scaled metallicity...")

                if planet_mass is None:
                    if planet_radius is None or planet_surface_gravity is None:
                        raise ValueError(f"both planet radius ({planet_radius}) "
                                         f"and surface gravity ({planet_surface_gravity}) "
                                         f"are required to calculate planet mass")
                    elif planet_radius <= 0:
                        raise ValueError(f"cannot calculate planet mass from surface gravity with a radius <= 0")

                    planet_mass = Planet.surface_gravity2mass(
                        surface_gravity=planet_surface_gravity,
                        radius=planet_radius
                    )[0]

                metallicity = SpectralModel2.calculate_scaled_metallicity(
                    planet_mass=planet_mass,
                    star_metallicity=star_metallicity,
                    atmospheric_mixing=atmospheric_mixing,
                    alpha=alpha,
                    beta=beta
                )

            if metallicity <= 0:
                metallicity = sys.float_info.min

            log10_metallicity = np.log10(metallicity)

        return log10_metallicity, metallicity, planet_mass, star_metallicity, atmospheric_mixing, alpha, beta

    @staticmethod
    def _calculate_equilibrium_mass_mixing_ratios(pressures, temperatures, co_ratio, log10_metallicity,
                                                  line_species, included_line_species,
                                                  carbon_pressure_quench=None, imposed_mass_mixing_ratios=None):
        from petitRADTRANS.poor_mans_nonequ_chem import poor_mans_nonequ_chem as pm  # import is here because it is long to load

        if imposed_mass_mixing_ratios is None:
            imposed_mass_mixing_ratios = {}

        if np.size(co_ratio) == 1:
            co_ratios = np.ones_like(pressures) * co_ratio
        else:
            co_ratios = co_ratio

        if np.size(log10_metallicity) == 1:
            log10_metallicities = np.ones_like(pressures) * log10_metallicity
        else:
            log10_metallicities = log10_metallicity

        equilibrium_mass_mixing_ratios = pm.interpol_abundances(
            COs_goal_in=co_ratios,
            FEHs_goal_in=log10_metallicities,
            temps_goal_in=temperatures,
            pressures_goal_in=pressures,
            Pquench_carbon=carbon_pressure_quench
        )

        # Check imposed mass mixing ratios keys
        for key in imposed_mass_mixing_ratios:
            if key not in line_species and key not in equilibrium_mass_mixing_ratios:
                raise KeyError(f"key '{key}' not in retrieved species list or "
                               f"standard petitRADTRANS mass fractions dict")

        # Get the right keys for the mass fractions dictionary
        mass_mixing_ratios = {}

        if included_line_species == 'all':
            included_line_species = []

            for line_species_name in line_species:
                included_line_species.append(line_species_name.split('_', 1)[0])

        for key in equilibrium_mass_mixing_ratios:
            found = False

            # Set line species mass mixing ratios into to their imposed one
            for line_species_name_ in line_species:
                if line_species_name_ == 'CO_36':  # CO_36 special case
                    if line_species_name_ in imposed_mass_mixing_ratios:
                        # Use imposed mass mixing ratio
                        mass_mixing_ratios[line_species_name_] = imposed_mass_mixing_ratios[line_species_name_]

                    continue

                # Correct for line species name to match pRT chemistry name
                line_species_name = line_species_name_.split('_', 1)[0]

                if line_species_name == 'C2H2':  # C2H2 special case
                    line_species_name += ',acetylene'

                if key == line_species_name:
                    if key not in included_line_species:
                        # Species not included, set mass mixing ratio to 0
                        mass_mixing_ratios[line_species_name] = np.zeros(np.shape(temperatures))
                    elif line_species_name_ in imposed_mass_mixing_ratios:
                        # Use imposed mass mixing ratio
                        mass_mixing_ratios[line_species_name_] = imposed_mass_mixing_ratios[line_species_name_]
                    else:
                        # Use calculated mass mixing ratio
                        mass_mixing_ratios[line_species_name] = equilibrium_mass_mixing_ratios[line_species_name]

                    found = True

                    break

            # Set species mass mixing ratio to their imposed one
            if not found:
                if key in imposed_mass_mixing_ratios:
                    # Use imposed mass mixing ratio
                    mass_mixing_ratios[key] = imposed_mass_mixing_ratios[key]
                else:
                    # Use calculated mass mixing ratio
                    mass_mixing_ratios[key] = equilibrium_mass_mixing_ratios[key]

        return mass_mixing_ratios

    @staticmethod
    def calculate_mass_mixing_ratios(pressures, line_species=None,
                                     included_line_species='all', temperatures=None, co_ratio=0.55,
                                     log10_metallicity=None, carbon_pressure_quench=None,
                                     imposed_mass_mixing_ratios=None, heh2_ratio=0.324324, c13c12_ratio=0.01,
                                     planet_mass=None, planet_radius=None, planet_surface_gravity=None,
                                     star_metallicity=1.0, atmospheric_mixing=1.0, alpha=-0.68, beta=7.2,
                                     use_equilibrium_chemistry=False, verbose=False, **kwargs):
        """Initialize a model mass mixing ratios.
        Ensure that in any case, the sum of mass mixing ratios is equal to 1. Imposed mass mixing ratios are kept to
        their imposed value as long as the sum of the imposed values is lower or equal to 1. H2 and He are used as
        filling gases.
        The different possible cases are dealt with as follows:
            - Sum of imposed mass mixing ratios > 1: the mass mixing ratios are scaled down, conserving the ratio
            between them. Non-imposed mass mixing ratios are set to 0.
            - Sum of imposed mass mixing ratio of all imposed species < 1: if equilibrium chemistry is used or if H2 and
            He are imposed species, the atmosphere will be filled with H2 and He respecting the imposed H2/He ratio.
            Otherwise, the heh2_ratio parameter is used.
            - Sum of imposed and non-imposed mass mixing ratios > 1: the non-imposed mass mixing ratios are scaled down,
            conserving the ratios between them. Imposed mass mixing ratios are unchanged.
            - Sum of imposed and non-imposed mass mixing ratios < 1: if equilibrium chemistry is used or if H2 and
            He are imposed species, the atmosphere will be filled with H2 and He respecting the imposed H2/He ratio.
            Otherwise, the heh2_ratio parameter is used.

        When using equilibrium chemistry with imposed mass mixing ratios, imposed mass mixing ratios are set to their
        imposed value regardless of chemical equilibrium consistency.

        Args:
            pressures: (bar) pressures of the mass mixing ratios
            line_species: list of line species, required to manage naming differences between opacities and chemistry
            included_line_species: which line species of the list to include, mass mixing ratio set to 0 otherwise
            temperatures: (K) temperatures of the mass mixing ratios, used with equilibrium chemistry
            co_ratio: carbon over oxygen ratios of the model, used with equilibrium chemistry
            log10_metallicity: ratio between heavy elements and H2 + He compared to solar, used with equilibrium chemistry
            carbon_pressure_quench: (bar) pressure where the carbon species are quenched, used with equilibrium chemistry
            imposed_mass_mixing_ratios: imposed mass mixing ratios
            heh2_ratio: H2 over He mass mixing ratio
            c13c12_ratio: 13C over 12C mass mixing ratio in equilibrium chemistry
            planet_mass: (g) mass of the planet; if None, planet mass is calculated from planet radius and surface gravity, used to calulate metallicity
            planet_radius: (cm) radius of the planet, used to calculate the mass
            planet_surface_gravity: (cm.s-2) surface gravity of the planet, used to calculate the mass
            star_metallicity: (solar metallicity) metallicity of the planet's star, used to calulate metallicity
            atmospheric_mixing: scaling factor [0, 1] representing how well metals are mixed in the atmosphere, used to calulate metallicity
            alpha: power of the mass-metallicity relation
            beta: scaling factor of the mass-metallicity relation
            use_equilibrium_chemistry: if True, use pRT equilibrium chemistry module
            verbose: if True, print additional information

        Returns:
            A dictionary containing the mass mixing ratios.
        """
        # Initialization
        mass_mixing_ratios = {}
        m_sum_imposed_species = np.zeros(np.shape(pressures))
        m_sum_species = np.zeros(np.shape(pressures))

        if line_species is None:
            line_species = []

        # Initialize imposed mass mixing ratios
        if imposed_mass_mixing_ratios is not None:
            for species, mass_mixing_ratio in imposed_mass_mixing_ratios.items():
                if np.size(mass_mixing_ratio) == 1:
                    imposed_mass_mixing_ratios[species] = np.ones(np.shape(pressures)) * mass_mixing_ratio
                elif np.size(mass_mixing_ratio) != np.size(pressures):
                    raise ValueError(f"mass mixing ratio for species '{species}' must be a scalar or an array of the"
                                     f"size of the pressure array ({np.size(pressures)}), "
                                     f"but is of size ({np.size(mass_mixing_ratio)})")
        else:
            # Nothing is imposed
            imposed_mass_mixing_ratios = {}

        # Chemical equilibrium mass mixing ratios
        if use_equilibrium_chemistry:
            # Calculate metallicity
            if log10_metallicity is None:
                if 'metallicity' in kwargs:
                    metallicity = kwargs['metallicity']
                else:
                    metallicity = None

                log10_metallicity, _, _, _, _, _, _ = SpectralModel2.__calculate_metallicity_wrap(
                    log10_metallicity=log10_metallicity,
                    metallicity=metallicity,
                    planet_mass=planet_mass,
                    planet_radius=planet_radius,
                    planet_surface_gravity=planet_surface_gravity,
                    star_metallicity=star_metallicity,
                    atmospheric_mixing=atmospheric_mixing,
                    alpha=alpha,
                    beta=beta,
                    verbose=verbose
                )

            # Interpolate chemical equilibrium
            mass_mixing_ratios_equilibrium = SpectralModel2._calculate_equilibrium_mass_mixing_ratios(
                pressures=pressures,
                temperatures=temperatures,
                co_ratio=co_ratio,
                log10_metallicity=log10_metallicity,
                line_species=line_species,
                included_line_species=included_line_species,
                carbon_pressure_quench=carbon_pressure_quench,
                imposed_mass_mixing_ratios=imposed_mass_mixing_ratios
            )

            # TODO more general handling of isotopologues (use smarter species names)
            if 'CO_main_iso' in line_species and 'CO_all_iso' in line_species:
                raise ValueError(f"cannot add main isotopologue and all isotopologues of CO at the same time")

            if 'CO_main_iso' not in imposed_mass_mixing_ratios and 'CO_36' not in imposed_mass_mixing_ratios:
                if 'CO_all_iso' not in line_species:
                    if 'CO_main_iso' in mass_mixing_ratios_equilibrium:
                        co_mass_mixing_ratio = copy.copy(mass_mixing_ratios_equilibrium['CO_main_iso'])
                    else:
                        co_mass_mixing_ratio = copy.copy(mass_mixing_ratios_equilibrium['CO'])

                    if 'CO_main_iso' in line_species:
                        mass_mixing_ratios_equilibrium['CO'] = co_mass_mixing_ratio / (1 + c13c12_ratio)
                        mass_mixing_ratios_equilibrium['CO_36'] = \
                            co_mass_mixing_ratio - mass_mixing_ratios_equilibrium['CO']
                    elif 'CO_36' in line_species:
                        mass_mixing_ratios_equilibrium['CO_36'] = co_mass_mixing_ratio / (1 + 1 / c13c12_ratio)
                        mass_mixing_ratios_equilibrium['CO'] = \
                            co_mass_mixing_ratio - mass_mixing_ratios_equilibrium['CO_36']
        else:
            mass_mixing_ratios_equilibrium = None

        # Imposed mass mixing ratios
        # Ensure that the sum of mass mixing ratios of imposed species is <= 1
        for species in imposed_mass_mixing_ratios:
            mass_mixing_ratios[species] = imposed_mass_mixing_ratios[species]
            m_sum_imposed_species += imposed_mass_mixing_ratios[species]

        for i in range(np.size(m_sum_imposed_species)):
            if m_sum_imposed_species[i] > 1:
                # TODO changing retrieved mmr might come problematic in some retrievals (retrieved value not corresponding to actual value in model)
                if verbose:
                    warnings.warn(f"sum of mass mixing ratios of imposed species ({m_sum_imposed_species}) is > 1, "
                                  f"correcting...")

                for species in imposed_mass_mixing_ratios:
                    mass_mixing_ratios[species][i] /= m_sum_imposed_species[i]

        m_sum_imposed_species = np.sum(list(mass_mixing_ratios.values()), axis=0)

        # Get the sum of mass mixing ratios of non-imposed species
        if mass_mixing_ratios_equilibrium is None:
            # TODO this is assuming an H2-He atmosphere with line species, this could be more general
            species_list = copy.copy(line_species)
        else:
            species_list = list(mass_mixing_ratios_equilibrium.keys())

        for species in species_list:
            # Ignore the non-MMR keys coming from the chemistry module
            if species == 'nabla_ad' or species == 'MMW':
                continue

            # Search for imposed species
            found = False

            for key in imposed_mass_mixing_ratios:
                spec = key.split('_R_')[0]  # deal with the naming scheme for binned down opacities

                if species == spec:
                    found = True

                    break

            # Only take into account non-imposed species and ignore imposed species
            if not found:
                if mass_mixing_ratios_equilibrium is None:
                    if verbose:
                        warnings.warn(
                            f"line species '{species}' initialised to {sys.float_info.min} ; "
                            f"to remove this warning set use_equilibrium_chemistry to True "
                            f"or add '{species}' and the desired mass mixing ratio to imposed_mass_mixing_ratios"
                        )

                    mass_mixing_ratios[species] = sys.float_info.min
                else:
                    mass_mixing_ratios[species] = mass_mixing_ratios_equilibrium[species]
                    m_sum_species += mass_mixing_ratios_equilibrium[species]

        # Ensure that the sum of mass mixing ratios of all species is = 1
        m_sum_total = m_sum_species + m_sum_imposed_species

        if np.any(np.logical_or(m_sum_total > 1, m_sum_total < 1)):
            # Search for H2 and He in both imposed and non-imposed species
            h2_in_imposed_mass_mixing_ratios = False
            he_in_imposed_mass_mixing_ratios = False
            h2_in_mass_mixing_ratios = False
            he_in_mass_mixing_ratios = False

            if 'H2' in imposed_mass_mixing_ratios:
                h2_in_imposed_mass_mixing_ratios = True

            if 'He' in imposed_mass_mixing_ratios:
                he_in_imposed_mass_mixing_ratios = True

            if 'H2' in mass_mixing_ratios:
                h2_in_mass_mixing_ratios = True

            if 'He' in mass_mixing_ratios:
                he_in_mass_mixing_ratios = True

            if not h2_in_mass_mixing_ratios or not he_in_mass_mixing_ratios:
                if not h2_in_mass_mixing_ratios:
                    mass_mixing_ratios['H2'] = np.zeros(np.shape(pressures))

                if not he_in_mass_mixing_ratios:
                    mass_mixing_ratios['He'] = np.zeros(np.shape(pressures))

            for i in range(np.size(m_sum_total)):
                if m_sum_total[i] > 1:
                    if verbose:
                        warnings.warn(f"sum of species mass fraction ({m_sum_species[i]} + {m_sum_imposed_species[i]}) "
                                      f"is > 1, correcting...")

                    for species in mass_mixing_ratios:
                        if species not in imposed_mass_mixing_ratios:
                            if m_sum_species[i] > 0:
                                mass_mixing_ratios[species][i] = \
                                    mass_mixing_ratios[species][i] * (1 - m_sum_imposed_species[i]) / m_sum_species[i]
                            else:
                                mass_mixing_ratios[species][i] = mass_mixing_ratios[species][i] / m_sum_total[i]
                elif m_sum_total[i] == 0:
                    if verbose:
                        print(f"sum of species mass fraction ({m_sum_species[i]} + {m_sum_imposed_species[i]}) "
                              f"is 0")
                elif m_sum_total[i] < 1:
                    # Fill atmosphere with H2 and He
                    # TODO there might be a better filling species, N2?
                    if h2_in_imposed_mass_mixing_ratios and he_in_imposed_mass_mixing_ratios:
                        if imposed_mass_mixing_ratios['H2'][i] > 0:
                            # Use imposed He/H2 ratio
                            heh2_ratio = 10 ** imposed_mass_mixing_ratios['He'][i] \
                                         / 10 ** imposed_mass_mixing_ratios['H2'][i]
                        else:
                            heh2_ratio = None

                    if h2_in_mass_mixing_ratios and he_in_mass_mixing_ratios:
                        # Use calculated He/H2 ratio
                        heh2_ratio = mass_mixing_ratios['He'][i] / mass_mixing_ratios['H2'][i]

                        mass_mixing_ratios['H2'][i] += (1 - m_sum_total[i]) / (1 + heh2_ratio)
                        mass_mixing_ratios['He'][i] = mass_mixing_ratios['H2'][i] * heh2_ratio
                    else:
                        # Remove H2 and He mass mixing ratios from total for correct mass mixing ratio calculation
                        if h2_in_mass_mixing_ratios:
                            m_sum_total[i] -= mass_mixing_ratios['H2'][i]
                        elif he_in_mass_mixing_ratios:
                            m_sum_total[i] -= mass_mixing_ratios['He'][i]

                        # Use He/H2 ratio in argument
                        mass_mixing_ratios['H2'][i] = (1 - m_sum_total[i]) / (1 + heh2_ratio)
                        mass_mixing_ratios['He'][i] = mass_mixing_ratios['H2'][i] * heh2_ratio
                else:
                    mass_mixing_ratios['H2'] = np.zeros(np.shape(pressures))
                    mass_mixing_ratios['He'] = np.zeros(np.shape(pressures))
        else:
            mass_mixing_ratios['H2'] = np.zeros(np.shape(pressures))
            mass_mixing_ratios['He'] = np.zeros(np.shape(pressures))

        return mass_mixing_ratios

    @staticmethod
    def calculate_scaled_metallicity(planet_mass, star_metallicity=1.0, atmospheric_mixing=1.0, alpha=-0.68, beta=7.2):
        """Calculate the scaled metallicity of a planet.
        The relation used is a power law. Default parameters come from the source.

        Source: Mordasini et al. 2014 (https://www.aanda.org/articles/aa/pdf/2014/06/aa21479-13.pdf)

        Args:
            planet_mass: (g) mass of the planet
            star_metallicity: metallicity of the planet in solar metallicity
            atmospheric_mixing: scaling factor [0, 1] representing how well metals are mixed in the atmosphere
            alpha: power of the relation
            beta: scaling factor of the relation

        Returns:
            An estimation of the planet atmospheric metallicity in solar metallicity.
        """
        return beta * (planet_mass / nc.m_jup) ** alpha * star_metallicity * atmospheric_mixing

    @staticmethod
    def calculate_spectral_parameters(temperature_profile_function, mass_mixing_ratios_function,
                                      mean_molar_masses_function, spectral_parameters_function, **kwargs):
        temperatures = temperature_profile_function(
            **kwargs
        )

        mass_mixing_ratios = mass_mixing_ratios_function(
            temperatures=temperatures,  # use the newly calculated temperature profile to obtain the mass mixing ratios
            **kwargs
        )

        # Find the mean molar mass in each layer
        mean_molar_mass = mean_molar_masses_function(
            mass_mixing_ratios=mass_mixing_ratios,
            **kwargs
        )

        model_parameters = spectral_parameters_function(
            **kwargs
        )

        return temperatures, mass_mixing_ratios, mean_molar_mass, model_parameters

    @staticmethod
    def calculate_temperature_profile(pressures, temperature_profile_mode='isothermal', temperature=None,
                                      intrinsic_temperature=None, planet_surface_gravity=None, metallicity=None,
                                      guillot_temperature_profile_gamma=0.4,
                                      guillot_temperature_profile_kappa_ir_z0=0.01, **kwargs):
        if temperature is None:
            raise TypeError(f"missing required argument 'temperature'")

        if temperature_profile_mode == 'isothermal':
            if isinstance(temperature, (float, int)):
                temperatures = np.ones(np.shape(pressures)) * temperature
            elif np.size(temperature) == np.size(pressures):
                temperatures = np.asarray(temperature)
            else:
                raise ValueError(f"could not initialize isothermal temperature profile ; "
                                 f"possible inputs are float, int, "
                                 f"or a 1-D array of the same size of parameter 'pressures' ({np.size(pressures)})")
        elif temperature_profile_mode == 'guillot':
            temperatures = guillot_metallic_temperature_profile(
                pressures=pressures,
                gamma=guillot_temperature_profile_gamma,
                surface_gravity=planet_surface_gravity,
                intrinsic_temperature=intrinsic_temperature,
                equilibrium_temperature=temperature,
                kappa_ir_z0=guillot_temperature_profile_kappa_ir_z0,
                metallicity=metallicity
            )
        else:
            raise ValueError(f"mode must be 'isothermal' or 'guillot', but was '{temperature_profile_mode}'")

        return temperatures

    def calculate_orbital_phases(self, phase_start, orbital_period):
        orbital_phases = Planet.get_orbital_phases(
            phase_start=phase_start,
            orbital_period=orbital_period,
            times=self.times
        )

        return orbital_phases

    @staticmethod
    def get_reduced_spectrum(spectrum, pipeline, **kwargs):
        # simple_pipeline interface
        if not hasattr(spectrum, 'mask'):
            spectrum = np.ma.masked_array(spectrum)

        if 'uncertainties' in kwargs:  # ensure that spectrum and uncertainties share the same mask
            if hasattr(kwargs['uncertainties'], 'mask'):
                spectrum.mask = np.zeros(spectrum.shape, dtype=bool)
                spectrum.mask[:] = copy.deepcopy(kwargs['uncertainties'].mask)

        reduced_data, reduction_matrix, reduced_data_uncertainties = \
            pipeline(spectrum=spectrum, full=True, **kwargs)

        return reduced_data, reduction_matrix, reduced_data_uncertainties

    def get_spectral_calculation_parameters(self, pressures=None, wavelengths=None,
                                            temperature_profile_mode='isothermal',
                                            temperature=None, line_species=None,
                                            included_line_species='all',
                                            imposed_mass_mixing_ratios=None,
                                            intrinsic_temperature=None, planet_surface_gravity=None, metallicity=None,
                                            guillot_temperature_profile_gamma=0.4,
                                            guillot_temperature_profile_kappa_ir_z0=0.01,
                                            co_ratio=0.55, carbon_pressure_quench=None, heh2_ratio=0.324324,
                                            use_equilibrium_chemistry=False, **kwargs
                                            ):
        """Initialize the temperature profile, mass mixing ratios and mean molar mass of a model.

        Args:
            pressures:
            wavelengths:
            temperature_profile_mode:
            temperature:
            line_species:
            included_line_species:
            imposed_mass_mixing_ratios:
            intrinsic_temperature:
            planet_surface_gravity:
            metallicity:
            guillot_temperature_profile_gamma:
            guillot_temperature_profile_kappa_ir_z0:
            co_ratio:
            carbon_pressure_quench:
            heh2_ratio:
            use_equilibrium_chemistry:

        Returns:

        """
        if pressures is None:
            pressures = self.pressures

        if metallicity is not None:
            log10_metallicity = np.log10(metallicity)
        elif use_equilibrium_chemistry:
            log10_metallicity, metallicity, planet_mass, star_metallicity, atmospheric_mixing, alpha, beta = \
                self.__calculate_metallicity_wrap(
                    log10_metallicity=None,
                    metallicity=metallicity,
                    planet_surface_gravity=planet_surface_gravity,
                    **kwargs
                )
        else:
            log10_metallicity = None

        for argument, value in locals().items():
            if argument not in kwargs and argument != 'self' and argument != 'kwargs':
                kwargs[argument] = value

        return self.calculate_spectral_parameters(
            temperature_profile_function=self.calculate_temperature_profile,
            mass_mixing_ratios_function=self.calculate_mass_mixing_ratios,
            mean_molar_masses_function=self.calculate_mean_molar_masses,
            spectral_parameters_function=self.calculate_model_parameters,
            **kwargs
        )

    @staticmethod
    def pipeline(**kwargs):
        return simple_pipeline(**kwargs)

    def update_spectral_calculation_parameters(self, radtrans: Radtrans, **parameters):
        pressures = radtrans.press * 1e-6  # cgs to bar
        # imposed_mass_mixing_ratios = {}
        kwargs = {'imposed_mass_mixing_ratios': {}}

        # for species in radtrans.line_species:
        #     # TODO mass mixing ratio dict initialization more general
        #     spec = species.split('_R_')[0]  # deal with the naming scheme for binned down opacities
        #
        #     if spec in parameters:
        #         # TODO move that to RetrievalSpectralModel
        #         kwargs['imposed_mass_mixing_ratios'][species] = 10 ** parameters[spec] * np.ones_like(pressures)

        for parameter, value in parameters.items():
            # TODO move that to RetrievalSpectralModel
            if 'log10_' in parameter and value is not None:
                kwargs[parameter.split('log10_', 1)[-1]] = 10 ** value
            else:
                if parameter == 'imposed_mass_mixing_ratios':
                    for species, mass_mixing_ratios in parameters[parameter].items():
                        if species not in kwargs[parameter]:
                            kwargs[parameter][species] = copy.copy(mass_mixing_ratios)
                else:
                    kwargs[parameter] = copy.copy(value)

        self.temperatures, self.mass_mixing_ratios, self.mean_molar_masses, self.model_parameters = \
            self.get_spectral_calculation_parameters(
                pressures=pressures,
                wavelengths=BaseSpectralModel.hz2um(radtrans.freq),
                line_species=radtrans.line_species,
                **kwargs
            )

        # Adapt chemical names to line species names, as required by Retrieval
        for species in radtrans.line_species:
            spec = species.split('_', 1)[0]

            if spec in self.mass_mixing_ratios:
                if species not in self.mass_mixing_ratios and species != 'K':
                    self.mass_mixing_ratios[species] = self.mass_mixing_ratios[spec]

                if species != 'K':  # TODO fix this K special case by choosing smarter opacities names
                    del self.mass_mixing_ratios[spec]


class RetrievalSpectralModel(SpectralModel):
    def __init__(self, planet_name, wavelength_boundaries, lbl_opacity_sampling, do_scat_emis, t_int, metallicity,
                 co_ratio, p_cloud,
                 line_species=None, rayleigh_species=None, continuum_opacities=None,
                 kappa_ir_z0=0.01, gamma=0.4, p_quench_c=None, haze_factor=1.0,
                 atmosphere_file=None, wavelengths=None, transit_radius=None, eclipse_depth=None,
                 spectral_radiosity=None, star_spectral_radiosity=None, opacity_mode='lbl',
                 heh2_ratio=0.324, use_equilibrium_chemistry=False,
                 temperature=None, pressures=None, mass_fractions=None, mean_molar_mass=None,
                 orbital_phases=None, system_observer_radial_velocities=None, planet_rest_frame_shift=0.0,
                 wavelengths_instrument=None, instrument_resolving_power=None,
                 planet_model_file=None, model_suffix='', filename=None):
        super().__init__(
            planet_name, wavelength_boundaries, lbl_opacity_sampling, do_scat_emis, t_int, metallicity,
            co_ratio, p_cloud,
            line_species=line_species,
            rayleigh_species=rayleigh_species,
            continuum_opacities=continuum_opacities,
            kappa_ir_z0=kappa_ir_z0,
            gamma=gamma,
            p_quench_c=p_quench_c,
            haze_factor=haze_factor,
            atmosphere_file=atmosphere_file,
            wavelengths=wavelengths,
            transit_radius=transit_radius,
            eclipse_depth=eclipse_depth,
            spectral_radiosity=spectral_radiosity,
            star_spectral_radiosity=star_spectral_radiosity,
            opacity_mode=opacity_mode,
            heh2_ratio=heh2_ratio,
            use_equilibrium_chemistry=use_equilibrium_chemistry,
            temperature=temperature,
            pressures=pressures,
            mass_fractions=mass_fractions,
            mean_molar_mass=mean_molar_mass,
            planet_model_file=planet_model_file,
            model_suffix=model_suffix,
            filename=filename
            )

        self.orbital_phases = orbital_phases
        self.system_observer_radial_velocities = system_observer_radial_velocities
        self.planet_rest_frame_shift = planet_rest_frame_shift
        self.wavelengths_instrument = wavelengths_instrument
        self.instrument_resolving_power = instrument_resolving_power

    def _calculate_transit_radius_from_parameters(self, atmosphere: Radtrans, **parameters):
        self.calculate_transit_radius(
            atmosphere=atmosphere,
            planet_surface_gravity=10 ** parameters['log10_surface_gravity'].value,
            planet_reference_pressure=parameters['reference_pressure'].value,
            planet_radius=parameters['planet_radius'].value
        )

    @staticmethod
    def _get_parameters_dict(surface_gravity, planet_radius=None, reference_pressure=1e-2,
                             temperature=None, mass_mixing_ratios=None, cloud_pressure=None,
                             guillot_temperature_profile_gamma=0.4, guillot_temperature_profile_kappa_ir_z0=0.01,
                             included_line_species=None, intrinsic_temperature=None, heh2_ratio=0.324,
                             use_equilibrium_chemistry=False,
                             co_ratio=0.55, metallicity=1.0, carbon_pressure_quench=None,
                             star_effective_temperature=None, star_radius=None, star_spectral_radiosity=None,
                             planet_max_radial_orbital_velocity=None, planet_orbital_inclination=None,
                             semi_major_axis=None, planet_radial_velocities=None,
                             planet_rest_frame_shift=0.0, orbital_phases=None, system_observer_radial_velocities=None,
                             wavelengths_instrument=None, instrument_resolving_power=None,
                             data=None, data_uncertainties=None,
                             reduced_data=None, reduced_data_uncertainties=None, reduction_matrix=None,
                             airmass=None, telluric_transmittance=None, variable_throughput=None
                             ):
        # Conversions to log-space
        if carbon_pressure_quench is not None:
            carbon_pressure_quench = np.log10(carbon_pressure_quench)

        if cloud_pressure is not None:
            cloud_pressure = np.log10(cloud_pressure)

        if co_ratio is not None:
            co_ratio = np.log10(co_ratio)

        if metallicity is not None:
            metallicity = np.log10(metallicity)

        if surface_gravity is not None:
            surface_gravity = np.log10(surface_gravity)

        # TODO expand to include all possible parameters of transm and calc_flux
        # TODO merge parameters and SpectralModel attributes?
        parameters = {
            'airmass': Param(airmass),
            'log10_co_ratio': Param(co_ratio),
            'data': Param(data),
            'data_uncertainties': Param(data_uncertainties),
            'guillot_temperature_profile_gamma': Param(guillot_temperature_profile_gamma),
            'guillot_temperature_profile_kappa_ir_z0': Param(guillot_temperature_profile_kappa_ir_z0),
            'heh2_ratio': Param(heh2_ratio),
            'included_line_species': Param(included_line_species),
            'instrument_resolving_power': Param(instrument_resolving_power),
            'intrinsic_temperature': Param(intrinsic_temperature),
            'log10_carbon_pressure_quench': Param(carbon_pressure_quench),
            'log10_cloud_pressure': Param(cloud_pressure),
            'log10_metallicity': Param(metallicity),
            'log10_surface_gravity': Param(surface_gravity),
            'orbital_phases': Param(orbital_phases),
            'planet_max_radial_orbital_velocity': Param(planet_max_radial_orbital_velocity),
            'planet_radius': Param(planet_radius),
            'planet_radial_velocities': Param(planet_radial_velocities),
            'planet_rest_frame_shift': Param(planet_rest_frame_shift),
            'planet_orbital_inclination': Param(planet_orbital_inclination),
            'reduced_data': Param(reduced_data),
            'reduction_matrix': Param(reduction_matrix),
            'reduced_data_uncertainties': Param(reduced_data_uncertainties),
            'reference_pressure': Param(reference_pressure),
            'semi_major_axis': Param(semi_major_axis),
            'star_effective_temperature': Param(star_effective_temperature),
            'star_radius': Param(star_radius),
            'star_spectral_radiosity': Param(star_spectral_radiosity),
            'system_observer_radial_velocities': Param(system_observer_radial_velocities),
            'telluric_transmittance': Param(telluric_transmittance),
            'temperature': Param(temperature),
            'use_equilibrium_chemistry': Param(use_equilibrium_chemistry),
            'variable_throughput': Param(variable_throughput),
            'wavelengths_instrument': Param(wavelengths_instrument),
        }

        if mass_mixing_ratios is None:
            mass_mixing_ratios = {}

        for species, mass_mixing_ratio in mass_mixing_ratios.items():
            parameters[species] = Param(np.log10(mass_mixing_ratio))

        return parameters

    def _get_shifted_transit_radius_from_parameters(self, atmosphere: Radtrans, **parameters):
        return self.get_shifted_transit_radius(
            atmosphere=atmosphere,
            planet_surface_gravity=10 ** parameters['log10_surface_gravity'].value,
            planet_reference_pressure=parameters['reference_pressure'].value,
            planet_radius=parameters['planet_radius'].value,
            planet_max_radial_orbital_velocity=parameters['planet_max_radial_orbital_velocity'].value,
            planet_orbital_inclination=parameters['planet_orbital_inclination'].value,
            orbital_phases=parameters['orbital_phases'].value,
            system_observer_radial_velocities=parameters['system_observer_radial_velocities'].value,
            planet_rest_frame_shift=parameters['planet_rest_frame_shift'].value,
            wavelengths_instrument=parameters['wavelengths_instrument'].value,
            instrument_resolving_power=parameters['instrument_resolving_power'].value,
            star_radius=parameters['star_radius'].value
        )

    @classmethod
    def from_parameters_dict(cls, planet_name, wavelength_boundaries=None, **parameters):
        # planet_orbital_velocity = Planet.calculate_orbital_velocity(
        #     star_mass=parameters['star_mass'].value,
        #     semi_major_axis=parameters['semi_major_axis'].value
        # )

        if wavelength_boundaries is None:
            wavelength_boundaries = [
                parameters['wavelengths_instrument'].value[0],
                parameters['wavelengths_instrument'].value[-1]
            ]
        else:
            # Constrain to boundaries
            within_boundaries = np.where(np.logical_and(
                parameters['wavelengths_instrument'].value > wavelength_boundaries[0],
                parameters['wavelengths_instrument'].value < wavelength_boundaries[-1]
            ))[0]
            parameters['wavelengths_instrument'] = Param(parameters['wavelengths_instrument'].value[within_boundaries])

        # Add enough spaces for Doppler shifting, taking into account a max rest frame shift of +/- K_p
        # wavelength_boundaries = [
        #     doppler_shift(wavelength_boundaries[0], -2 * planet_orbital_velocity),
        #     doppler_shift(wavelength_boundaries[-1], 2 * planet_orbital_velocity)
        # ]

        mass_mixing_ratios = {}

        for parameter, value in parameters.items():
            # TODO add default chemical species, otherwise this won't work as expected
            if parameter in cls.default_line_species or parameter in cls.default_rayleigh_species:
                mass_mixing_ratios[parameter] = value.value

        return cls(
            planet_name=planet_name,
            wavelength_boundaries=wavelength_boundaries,
            lbl_opacity_sampling=1,
            do_scat_emis=True,
            t_int=parameters['intrinsic_temperature'].value,
            metallicity=10 ** parameters['log10_metallicity'].value,
            co_ratio=10 ** parameters['log10_co_ratio'].value,
            p_cloud=10 ** parameters['log10_cloud_pressure'].value,
            kappa_ir_z0=parameters['guillot_temperature_profile_kappa_ir_z0'].value,
            gamma=parameters['guillot_temperature_profile_gamma'].value,
            p_quench_c=10 ** parameters['log10_carbon_pressure_quench'].value,
            haze_factor=1.0,
            atmosphere_file=None,
            wavelengths=None,
            transit_radius=None,
            eclipse_depth=None,
            spectral_radiosity=None,
            star_spectral_radiosity=None,
            opacity_mode='lbl',
            heh2_ratio=parameters['heh2_ratio'].value,
            use_equilibrium_chemistry=parameters['use_equilibrium_chemistry'].value,
            temperature=parameters['temperature'].value,
            pressures=None,
            mass_fractions=mass_mixing_ratios,
            mean_molar_mass=None,
            orbital_phases=parameters['orbital_phases'].value,
            system_observer_radial_velocities=parameters['system_observer_radial_velocities'].value,
            planet_rest_frame_shift=parameters['planet_rest_frame_shift'].value,
            wavelengths_instrument=parameters['wavelengths_instrument'].value,
            instrument_resolving_power=parameters['instrument_resolving_power'].value,
            planet_model_file=None,
            model_suffix='',
            filename=None
        )

    @staticmethod
    def get_orbital_phases(planet_orbital_period, integration_time, start=None, integrations_number=50,
                           mode='transit'):
        if mode == 'transit':
            orbital_phases = get_orbital_phases(
                0.0, planet_orbital_period, integration_time, integrations_number
            )
            orbital_phases -= np.max(orbital_phases) / 2
        else:
            orbital_phases = \
                get_orbital_phases(start, planet_orbital_period, integration_time, integrations_number)

        return orbital_phases

    def get_parameters_dict(self, planet: Planet, included_line_species='all'):
        # star_spectral_radiosity = self.get_phoenix_star_spectral_radiosity(planet)
        planet_max_radial_orbital_velocity = planet.calculate_orbital_velocity(
            planet.star_mass, planet.orbit_semi_major_axis
        )
        # TODO complete this
        # TODO add class function from parameter dict

        return self._get_parameters_dict(
            surface_gravity=planet.surface_gravity,
            planet_radius=planet.radius,
            reference_pressure=planet.reference_pressure,
            temperature=self.temperature,
            mass_mixing_ratios=self.mass_mixing_ratios,
            cloud_pressure=self.cloud_pressure,
            guillot_temperature_profile_gamma=self.guillot_temperature_profile_gamma,
            guillot_temperature_profile_kappa_ir_z0=self.guillot_temperature_profile_kappa_ir_z0,
            included_line_species=included_line_species,
            intrinsic_temperature=self.intrinsic_temperature,
            heh2_ratio=self.heh2_ratio,
            use_equilibrium_chemistry=self.use_equilibrium_chemistry,
            co_ratio=self.co_ratio,
            metallicity=self.metallicity,
            carbon_pressure_quench=self.carbon_pressure_quench,
            star_effective_temperature=planet.star_effective_temperature,
            star_radius=planet.star_radius,
            # star_spectral_radiosity=star_spectral_radiosity,
            planet_max_radial_orbital_velocity=planet_max_radial_orbital_velocity,
            planet_orbital_inclination=planet.orbital_inclination,
            semi_major_axis=planet.orbit_semi_major_axis,
            planet_rest_frame_shift=self.planet_rest_frame_shift,
            orbital_phases=self.orbital_phases,
            system_observer_radial_velocities=self.system_observer_radial_velocities,
            wavelengths_instrument=self.wavelengths_instrument,
            instrument_resolving_power=self.instrument_resolving_power,
            data=None,
            data_uncertainties=None,
            reduced_data=None,
            reduced_data_uncertainties=None,
            reduction_matrix=None,
            airmass=None,
            telluric_transmittance=None,
            variable_throughput=None
        )

    @staticmethod
    def get_reduced_spectrum(spectrum, pipeline, **kwargs):
        if 'data' in kwargs:
            if isinstance(kwargs['data'], np.ma.core.MaskedArray):
                spectrum = np.ma.masked_array(spectrum)
                spectrum.mask = copy.copy(kwargs['data'].mask)

        return pipeline(spectrum, **kwargs)

    def get_reduced_shifted_transit_radius_model(self, atmosphere: Radtrans, parameters, pt_plot_mode=None, AMR=False):
        self.update_spectral_calculation_parameters(atmosphere, **parameters)
        wavelengths, spectra = self._get_shifted_transit_radius_from_parameters(atmosphere, **parameters)

        value_parameters = {key: value.value for key, value in parameters.items()}

        return wavelengths, self.get_reduced_spectrum(np.array([spectra]), self.pipeline, **value_parameters)

    def get_shifted_transit_radius_model(self, atmosphere: Radtrans, parameters):
        self.update_spectral_calculation_parameters(atmosphere, **parameters)

        return self._get_shifted_transit_radius_from_parameters(atmosphere, **parameters)

    def get_transit_radius_model(self, atmosphere: Radtrans, parameters):
        self.update_spectral_calculation_parameters(atmosphere, **parameters)
        self._calculate_transit_radius_from_parameters(atmosphere, **parameters)

        return self.wavelengths, self.transit_radius

    @staticmethod
    def get_wavelengths_boundaries(wavelengths_boundaries, min_radial_velocity=0.0, max_radial_velocity=0.0):
        """Get wavelength boundaries for a source moving between two radial velocities with respect to the observer.
        A negative velocity means that the source is going toward the observer. A positive velocity means the source is
        going away from the observer.

        Args:
            wavelengths_boundaries: list containing the desired wavelengths boundaries for the source at rest.
            min_radial_velocity: (cm.s-1) minimum velocity of the source
            max_radial_velocity: (cm.s-1) maximum velocity of the source

        Returns:
            New wavelengths boundaries, taking into account the velocity of the source
        """
        return [
            doppler_shift(wavelengths_boundaries[0], min_radial_velocity),
            doppler_shift(wavelengths_boundaries[-1], max_radial_velocity)
        ]

    @staticmethod
    def pipeline(spectrum):
        """Simplistic pipeline model. Do nothing.
        To be updated when initializing an instance of retrieval model.

        Args:
            spectrum: a spectrum

        Returns:
            spectrum: the spectrum reduced by the pipeline
        """
        return spectrum

    def update_spectral_calculation_parameters(self, radtrans: Radtrans, **parameters):
        pressures = radtrans.press * 1e-6  # cgs to bar
        imposed_mass_mixing_ratios = {}

        for species in radtrans.line_species:
            # TODO mass mixing ratio dict initialization more general
            spec = species.split('_R_')[0]  # deal with the naming scheme for binned down opacities

            if spec in parameters:
                imposed_mass_mixing_ratios[species] = 10 ** parameters[spec].value * np.ones_like(pressures)

        self.temperature, self.mass_mixing_ratios, self.mean_molar_mass = \
            self._get_spectral_calculation_parameters(
                pressures=pressures,
                temperature=parameters['temperature'].value,
                line_species=radtrans.line_species,
                included_line_species=parameters['included_line_species'].value,
                imposed_mass_mixing_ratios=imposed_mass_mixing_ratios,
                intrinsic_temperature=parameters['intrinsic_temperature'].value,
                surface_gravity=10 ** parameters['log10_surface_gravity'].value,
                metallicity=10 ** parameters['log10_metallicity'].value,
                guillot_temperature_profile_gamma=parameters['guillot_temperature_profile_gamma'].value,
                guillot_temperature_profile_kappa_ir_z0=parameters['guillot_temperature_profile_kappa_ir_z0'].value,
                co_ratio=10 ** parameters['log10_co_ratio'].value,
                carbon_pressure_quench=parameters['log10_carbon_pressure_quench'].value,
                heh2_ratio=parameters['heh2_ratio'].value,
                use_equilibrium_chemistry=parameters['use_equilibrium_chemistry'].value
            )

        # Adapt chemical names to line species names, as required by Retrieval
        for species in radtrans.line_species:
            spec = species.split('_', 1)[0]

            if spec in self.mass_mixing_ratios:
                if species not in self.mass_mixing_ratios:
                    self.mass_mixing_ratios[species] = self.mass_mixing_ratios[spec]

                del self.mass_mixing_ratios[spec]


class RetrievalSpectralModel2(SpectralModel2):
    def __init__(self,  wavelengths_boundaries, pressures, temperatures=None,
                 mass_mixing_ratios=None, mean_molar_masses=None,
                 line_species=None, rayleigh_species=None, continuum_opacities=None, cloud_species=None,
                 opacity_mode='lbl', do_scat_emis=True, lbl_opacity_sampling=1,
                 wavelengths=None, transit_radii=None, spectral_radiosities=None, times=None,
                 star_spectral_radiosities=None, shifted_star_spectral_radiosities=None,
                 wavelengths_rest=None, resolving_power_rest=None,
                 shifted_transit_radii=None, shifted_spectral_radiosities=None,
                 orbital_phases=None, log10_parameters=None,
                 **model_parameters):
        super().__init__(
            wavelengths_boundaries=wavelengths_boundaries,
            pressures=pressures,
            temperatures=temperatures,
            mass_mixing_ratios=mass_mixing_ratios,
            mean_molar_masses=mean_molar_masses,
            line_species=line_species,
            rayleigh_species=rayleigh_species,
            continuum_opacities=continuum_opacities,
            cloud_species=cloud_species,
            opacity_mode=opacity_mode,
            do_scat_emis=do_scat_emis,
            lbl_opacity_sampling=lbl_opacity_sampling,
            wavelengths=wavelengths,
            transit_radii=transit_radii,
            spectral_radiosities=spectral_radiosities,
            times=times,
            star_spectral_radiosities=star_spectral_radiosities,
            shifted_star_spectral_radiosities=shifted_star_spectral_radiosities,
            wavelengths_rest=wavelengths_rest,
            resolving_power_rest=resolving_power_rest,
            shifted_transit_radii=shifted_transit_radii,
            shifted_spectral_radiosities=shifted_spectral_radiosities,
            orbital_phases=orbital_phases,
            **model_parameters
        )

        self.log10_parameters = log10_parameters

    @staticmethod
    def __parameter_to_log10(parameter, parameter_value):
        return f'log10_{parameter}', np.log10(parameter_value)

    @staticmethod
    def __log10_to_parameter(parameter, parameter_value):
        return parameter.split('log10', 1)[-1], 10 ** parameter_value

    @staticmethod
    def dict_to_param(dictionary):
        return {key: Param(value) for key, value in dictionary.items()}

    def get_parameters_dict(self):
        parameters_dict = {}

        for key, value in self.__dict__.items():
            if key == 'model_parameters':  # model_parameters is a dictionary, extract its values
                for parameter, parameter_value in value.items():
                    if parameter == 'imposed_mass_mixing_ratios':
                        for species, mass_mixing_ratio in parameter_value.items():
                            parameters_dict[species] = copy.copy(
                                np.log10(np.max((mass_mixing_ratio, sys.float_info.min)))
                            )
                    else:
                        if parameter in self.log10_parameters:
                            if parameter_value is not None:
                                parameter, parameter_value = self.__parameter_to_log10(parameter, parameter_value)
                            else:
                                parameter, _ = self.__parameter_to_log10(parameter, 1)

                        parameters_dict[parameter] = copy.copy(parameter_value)

        return self.dict_to_param(parameters_dict)

    @staticmethod
    def get_wavelengths_boundaries(wavelengths_boundaries, min_radial_velocity=0.0, max_radial_velocity=0.0):
        """Get wavelength boundaries for a source moving between two radial velocities with respect to the observer.
        A negative velocity means that the source is going toward the observer. A positive velocity means the source is
        going away from the observer.

        Args:
            wavelengths_boundaries: list containing the desired wavelengths boundaries for the source at rest.
            min_radial_velocity: (cm.s-1) minimum velocity of the source
            max_radial_velocity: (cm.s-1) maximum velocity of the source

        Returns:
            New wavelengths boundaries, taking into account the velocity of the source
        """
        return [
            doppler_shift(wavelengths_boundaries[0], min_radial_velocity),
            doppler_shift(wavelengths_boundaries[-1], max_radial_velocity)
        ]

    @staticmethod
    def init_retrieval_configuration(retrieval_name, parameters, retrieved_parameters, prior_functions,
                                     observed_spectra, observations_uncertainties, retrieval_model, prt_object,
                                     pressures=None, run_mode='retrieval', amr=False, scattering=False):
        run_configuration = RetrievalConfig(
            retrieval_name=retrieval_name,
            run_mode=run_mode,
            AMR=amr,
            scattering=scattering,  # scattering is automatically included for transmission spectra
            pressures=pressures
            # TODO add the other parameters if they are relevant
        )

        # Fixed parameters
        i_retrieved = 0

        for name, value in parameters.items():
            if name not in retrieved_parameters:
                run_configuration.add_parameter(
                    name=name,
                    free=False,
                    value=value.value,  # TODO no need to use .value?
                    transform_prior_cube_coordinate=None
                )

        for i, name in enumerate(retrieved_parameters):
            run_configuration.add_parameter(
                name=name,
                free=True,
                value=None,
                transform_prior_cube_coordinate=prior_functions[i_retrieved]
            )

        # Remove masked values if necessary to speed up retrieval
        if hasattr(observed_spectra, 'mask'):
            print('Taking care of mask...')
            data_ = []
            error_ = []
            mask_ = copy.copy(observed_spectra.mask)
            lengths = []

            for i in range(observed_spectra.shape[0]):
                data_.append([])
                error_.append([])

                for j in range(observed_spectra.shape[1]):
                    data_[i].append(np.array(
                        observed_spectra[i, j, ~mask_[i, j, :]]
                    ))
                    error_[i].append(np.array(observations_uncertainties[i, j, ~mask_[i, j, :]]))
                    lengths.append(data_[i][j].size)

            # Handle jagged arrays
            if np.all(np.asarray(lengths) == lengths[0]):
                data_ = np.asarray(data_)
                error_ = np.asarray(error_)
            else:
                print("Array is jagged, generating object array...")
                data_ = np.asarray(data_, dtype=object)
                error_ = np.asarray(error_, dtype=object)
        else:
            data_ = observed_spectra
            error_ = observations_uncertainties
            mask_ = None

        # Load data
        # TODO add multiple data (see JWST config)
        run_configuration.add_data(
            name='test',
            path=None,
            model_generating_function=retrieval_model,
            opacity_mode=prt_object.mode,  # TODO this might be mostly useless
            pRT_object=prt_object,
            wlen=nc.c / prt_object.freq * 1e4,  # TODO this might be mostly useless
            flux=data_,
            flux_error=error_,
            mask=mask_
        )

        return run_configuration

    @staticmethod
    def param_to_dict(parameters):
        return {key: value.value for key, value in parameters.items()}

    def retrieval_model(self, radtrans: Radtrans, parameters, pt_plot_mode=None, AMR=False, mode='transit',
                        update_parameters=False, scale=False, shift=False, convolve=False, rebin=False, reduce=False
                        ):
        parameter_dict = self.param_to_dict(parameters)
        wavelengths, spectra = self.get_spectrum_model(
            radtrans=radtrans,
            parameters=parameter_dict,
            mode=mode,
            update_parameters=update_parameters,
            scale=scale,
            shift=shift,
            convolve=convolve,
            rebin=rebin,
            reduce=reduce
        )

        return wavelengths, spectra


def get_orbital_phases(phase_start, orbital_period, dit, ndit, return_times=False):
    """Calculate orbital phases assuming low eccentricity.

    Args:
        phase_start: planet phase at the start of observations
        orbital_period: (s) orbital period of the planet
        dit: (s) integration duration
        ndit: number of integrations
        return_times: if true, also returns the time used to calculate the orbital phases

    Returns:
        ndit phases from start_phase at t=0 to the phase at t=dit * ndit
    """
    times = np.linspace(0, dit * ndit, ndit)
    phases = np.mod(phase_start + times / orbital_period, 1.0)

    if return_times:
        return phases, times  # the 2 * pi factors cancel out
    else:
        return phases


def _get_generic_planet_name(radius, surface_gravity, equilibrium_temperature):
    return f"generic_{radius / nc.r_jup:.2f}Rjup_logg{np.log10(surface_gravity):.2f}_teq{equilibrium_temperature:.2f}K"


def generate_model_grid(models, pressures,
                        line_species_list='default', rayleigh_species='default', continuum_opacities='default',
                        model_suffix='', atmosphere=None, temperature_profile=None, mass_fractions=None,
                        calculate_transmission_spectrum=False, calculate_emission_spectrum=False,
                        calculate_eclipse_depth=False,
                        rewrite=False, save=False):
    """
    Get a grid of models, generate it if needed.
    Models are generated using petitRADTRANS in its line-by-line mode. Clouds are modelled as a gray deck.
    Output will be organized hierarchically as follows: model string > included species

    Args:
        models: dictionary of models
        pressures: (bar) 1D-array containing the pressure grid of the models
        line_species_list: list containing all the line species to include in the models
        rayleigh_species: list containing all the rayleigh species to include in the models
        continuum_opacities: list containing all the continua to include in the models
        model_suffix: suffix of the model
        atmosphere: pre-loaded Radtrans object
        temperature_profile: if None, a Guillot temperature profile is generated, if int or float, an isothermal
            temperature profile is generated, if 1-D array of the same size of pressures, the temperature profile is
            directly used
        mass_fractions: if None, equilibrium chemistry is used, if dict, the values from the dict are used
        calculate_transmission_spectrum: if True, calculate the transmission spectrum of the model
        calculate_emission_spectrum: if True, calculate the emission spectrum of the model
        calculate_eclipse_depth: if True, calculate the eclipse depth, and the emission spectrum, of the model
        rewrite: if True, rewrite all the models, even if they already exists
        save: if True, save the models once generated

    Returns:
        models: a dictionary containing all the requested models
    """
    i = 0

    for model in models:
        for species in models[model]:
            i += 1
            print(f"Model {i}/{len(models) * len(models[model])}...")

            models[model][species] = SpectralModel.generate_from(
                model=models[model][species],
                pressures=pressures,
                include_species=species,
                line_species_list=line_species_list,
                rayleigh_species=rayleigh_species,
                continuum_opacities=continuum_opacities,
                model_suffix=model_suffix,
                atmosphere=atmosphere,
                temperature_profile=temperature_profile,
                mass_fractions=mass_fractions,
                calculate_transmission_spectrum=calculate_transmission_spectrum,
                calculate_emission_spectrum=calculate_emission_spectrum,
                calculate_eclipse_depth=calculate_eclipse_depth,
                rewrite=rewrite
            )

            if save:
                models[model][species].save()

    return models


def get_model_grid(planet_name, lbl_opacity_sampling, do_scat_emis, parameter_dicts, species_list, pressures,
                   wavelength_boundaries, line_species_list='default', rayleigh_species='default',
                   continuum_opacities='default', model_suffix='', atmosphere=None,
                   calculate_transmission_spectrum=False, calculate_emission_spectrum=False,
                   calculate_eclipse_depth=False,
                   rewrite=False, save=False):
    """
    Get a grid of models, generate it if needed.
    Models are generated using petitRADTRANS in its line-by-line mode. Clouds are modelled as a gray deck.
    Output will be organized hierarchically as follows: model string > included species

    Args:
        planet_name: name of the planet modelled
        lbl_opacity_sampling: downsampling coefficient of
        do_scat_emis: if True, include the scattering for emission spectra
        parameter_dicts: list of dictionaries containing the models parameters
        species_list: list of lists of species to include in the models (e.g. ['all', 'H2O', 'CH4'])
        pressures: (bar) 1D-array containing the pressure grid of the models
        wavelength_boundaries: (um) size-2 array containing the min and max wavelengths
        line_species_list: list containing all the line species to include in the models
        rayleigh_species: list containing all the rayleigh species to include in the models
        continuum_opacities: list containing all the continua to include in the models
        model_suffix: suffix of the model
        atmosphere: pre-loaded Radtrans object
        calculate_transmission_spectrum: if True, calculate the transmission spectrum of the model
        calculate_emission_spectrum: if True, calculate the emission spectrum of the model
        calculate_eclipse_depth: if True, calculate the eclipse depth, and the emission spectrum, of the model
        rewrite: if True, rewrite all the models, even if they already exists
        save: if True, save the models once generated

    Returns:
        models: a dictionary containing all the requested models
    """
    models = {}
    i = 0

    for parameter_dict in parameter_dicts:
        models[parameter_dict.to_str()] = {}

        for species in species_list:
            i += 1
            print(f"Model {i}/{len(parameter_dicts) * len(species_list)}...")

            models[parameter_dict.to_str()][species] = SpectralModel.get(
                planet_name=planet_name,
                wavelength_boundaries=wavelength_boundaries,
                lbl_opacity_sampling=lbl_opacity_sampling,
                do_scat_emis=do_scat_emis,
                t_int=parameter_dict['intrinsic_temperature'],
                metallicity=parameter_dict['metallicity'],
                co_ratio=parameter_dict['co_ratio'],
                p_cloud=parameter_dict['p_cloud'],
                pressures=pressures,
                include_species=species,
                kappa_ir_z0=0.01,
                gamma=0.4,
                p_quench_c=None,
                haze_factor=1,
                line_species_list=line_species_list,
                rayleigh_species=rayleigh_species,
                continuum_opacities=continuum_opacities,
                model_suffix=model_suffix,
                atmosphere=atmosphere,
                calculate_transmission_spectrum=calculate_transmission_spectrum,
                calculate_emission_spectrum=calculate_emission_spectrum,
                calculate_eclipse_depth=calculate_eclipse_depth,
                rewrite=rewrite
            )

            if save:
                models[parameter_dict.to_str()][species].save()

    return models


def get_parameter_dicts(t_int: list, metallicity: list, co_ratio: list, p_cloud: list):
    """
    Generate a parameter dictionary from parameters.
    To be used in get_model_grid()

    Args:
        t_int: (K) intrinsic temperature of the planet
        metallicity: metallicity of the planet
        co_ratio: C/O ratio of the planet
        p_cloud: (bar) cloud top pressure of the planet

    Returns:
        parameter_dict: a ParameterDict
    """

    parameter_dicts = []

    for t in t_int:
        for z in metallicity:
            for co in co_ratio:
                for pc in p_cloud:
                    parameter_dicts.append(
                        ParametersDict(
                            t_int=t,
                            metallicity=z,
                            co_ratio=co,
                            p_cloud=pc
                        )
                    )

    return parameter_dicts


def init_model_grid(planet_name, lbl_opacity_sampling, do_scat_emis, parameter_dicts, species_list,
                    wavelength_boundaries, model_suffix=''):
    # Initialize models
    models = {}
    all_models_exist = True

    for parameter_dict in parameter_dicts:
        models[parameter_dict.to_str()] = {}

        for species in species_list:
            models[parameter_dict.to_str()][species] = SpectralModel.species_init(
                include_species=species,
                planet_name=planet_name,
                wavelength_boundaries=wavelength_boundaries,
                lbl_opacity_sampling=lbl_opacity_sampling,
                do_scat_emis=do_scat_emis,
                t_int=parameter_dict['intrinsic_temperature'],
                metallicity=parameter_dict['metallicity'],
                co_ratio=parameter_dict['co_ratio'],
                p_cloud=parameter_dict['p_cloud'],
                kappa_ir_z0=0.01,
                gamma=0.4,
                p_quench_c=None,
                haze_factor=1,
                model_suffix=model_suffix
            )

            if not os.path.isfile(models[parameter_dict.to_str()][species].filename) and all_models_exist:
                all_models_exist = False

    return models, all_models_exist


def load_model_grid(models):
    i = 0

    for model in models:
        for species in models[model]:
            i += 1
            print(f"Loading model {i}/{len(models) * len(models[model])} from '{models[model][species].filename}'...")

            models[model][species] = models[model][species].load(models[model][species].filename)

    return models


def make_generic_planet(radius, surface_gravity, equilibrium_temperature,
                        star_effective_temperature=5500, star_radius=nc.r_sun, orbit_semi_major_axis=nc.AU):
    name = _get_generic_planet_name(radius, surface_gravity, equilibrium_temperature)

    return SimplePlanet(
        name,
        radius,
        surface_gravity,
        star_effective_temperature,
        star_radius,
        orbit_semi_major_axis,
        equilibrium_temperature=equilibrium_temperature
    )
