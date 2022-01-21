"""
Useful functions to generate mock observations.
"""
import copy

import numpy as np
from scipy.ndimage.filters import gaussian_filter1d

from petitRADTRANS.ccf.model_containers import Planet
from petitRADTRANS.fort_rebin import fort_rebin as fr
from petitRADTRANS.physics import doppler_shift


def convolve_rebin(input_wavelengths, input_flux,
                   instrument_resolving_power, pixel_sampling, instrument_wavelength_range):
    """
    Function to convolve observation with instrument obs and rebin to pixels of detector.
    Create mock observation for high-res spectrograph.

    Args:
        input_wavelengths: (cm) wavelengths of the input spectrum
        input_flux: flux of the input spectrum
        instrument_resolving_power: resolving power of the instrument
        pixel_sampling: number of pixels per resolution elements (i.e. how many px in one LSF FWHM, usually 2)
        instrument_wavelength_range: (um) wavelength range of the instrument

    Returns:
        flux_lsf: flux altered by the instrument's LSF
        freq_out: (Hz) frequencies of the rebinned flux, in descending order
        flux_rebin: the rebinned flux
    """
    # From talking to Ignas: delta lambda of resolution element is the FWHM of the instrument's LSF (here: a gaussian)
    sigma_lsf = 1. / instrument_resolving_power / (2. * np.sqrt(2. * np.log(2.)))

    # The input resolution of petitCODE is 1e6, but just compute
    # it to be sure, or more versatile in the future.
    # Also, we have a log-spaced grid, so the resolution is constant
    # as a function of wavelength
    model_resolving_power = np.mean(
        (input_wavelengths[1:] + input_wavelengths[:-1]) / (2. * np.diff(input_wavelengths))
    )

    # Calculate the sigma to be used in the gauss filter in units of input frequency bins
    sigma_lsf_gauss_filter = sigma_lsf * model_resolving_power

    flux_lsf = gaussian_filter1d(
        input=input_flux,
        sigma=sigma_lsf_gauss_filter,
        mode='reflect'
    )

    if np.size(instrument_wavelength_range) == 2:  # TODO check if this is still working
        wavelength_out_borders = np.logspace(
            np.log10(instrument_wavelength_range[0]),
            np.log10(instrument_wavelength_range[1]),
            int(pixel_sampling * instrument_resolving_power
                * np.log(instrument_wavelength_range[1] / instrument_wavelength_range[0]))
        )
        wavelengths_out = (wavelength_out_borders[1:] + wavelength_out_borders[:-1]) / 2.
    elif np.size(instrument_wavelength_range) > 2:
        wavelengths_out = instrument_wavelength_range
    else:
        raise ValueError(f"instrument wavelength must be of size 2 or more, "
                         f"but is of size {np.size(instrument_wavelength_range)}: {instrument_wavelength_range}")

    flux_rebin = fr.rebin_spectrum(input_wavelengths, flux_lsf, wavelengths_out)

    return flux_lsf, wavelengths_out, flux_rebin


def convolve_shift_rebin(input_wavelengths, input_flux,
                         instrument_resolving_power, output_wavelengths, planet_velocities=None):
    """
    Function to convolve and Doppler-shift observation with instrument obs and rebin to pixels of detector.

    Args:
        input_wavelengths: (cm) wavelengths of the input spectrum
        input_flux: flux of the input spectrum
        instrument_resolving_power: resolving power of the instrument
        output_wavelengths: (cm) wavelength of the instrument
        planet_velocities: (cm.s-1) array containing velocities of the planet relative to its star

    Returns:
        flux_lsf: flux altered by the instrument's LSF
        freq_out: (Hz) frequencies of the rebinned flux, in descending order
        flux_rebin: the rebinned flux
    """
    if planet_velocities is None:
        planet_velocities = np.zeros(1)

    # Delta lambda of resolution element is the FWHM of the instrument's LSF (here: a gaussian)
    sigma_lsf = 1. / instrument_resolving_power / (2. * np.sqrt(2. * np.log(2.)))

    # Compute resolving power of the model
    # In petitRADTRANS, the wavelength grid is log-spaced, so the resolution is constant as a function of wavelength
    model_resolving_power = np.mean(
        (input_wavelengths[1:] + input_wavelengths[:-1]) / (2. * np.diff(input_wavelengths))
    )

    # Calculate the sigma to be used in the gauss filter in units of input frequency bins
    sigma_lsf_gauss_filter = sigma_lsf * model_resolving_power

    flux_lsf = gaussian_filter1d(
        input=input_flux,
        sigma=sigma_lsf_gauss_filter,
        mode='reflect'
    )

    flux_rebin = np.zeros((np.size(planet_velocities), np.size(output_wavelengths)))

    for i, planet_velocity in enumerate(planet_velocities):
        wavelength_shift = doppler_shift(input_wavelengths, planet_velocity)
        flux_rebin[i, :] = fr.rebin_spectrum(wavelength_shift, flux_lsf, output_wavelengths)

    return flux_rebin


def get_mock_secondary_eclipse_spectra(wavelength_model, spectrum_model, star_spectral_radiosity,
                                       planet_radius, star_radius,
                                       wavelength_instrument, instrument_resolving_power,
                                       planet_velocities, system_observer_radial_velocities,
                                       planet_rest_frame_shift=0.0):
    planet_radiosity = convolve_shift_rebin(
        wavelength_model,
        spectrum_model,
        instrument_resolving_power,
        wavelength_instrument,
        planet_velocities=planet_velocities
            + system_observer_radial_velocities
            + planet_rest_frame_shift  # planet + system velocity
    )

    star_spectral_radiosity = convolve_shift_rebin(
        wavelength_model,
        star_spectral_radiosity,
        instrument_resolving_power,
        wavelength_instrument,
        planet_velocities=system_observer_radial_velocities + planet_rest_frame_shift
    )

    # TODO add stellar reflection, dayside/nightside model ?
    return 1 + (planet_radiosity * planet_radius ** 2) / (star_spectral_radiosity * star_radius ** 2)


def get_noise(mock_observation, instrument_snr, observing_duration, planet_visible_duration, mode='eclipse', number=1):
    """

    Args:
        mock_observation:
        instrument_snr:
        observing_duration:
            (s) total observation time
        planet_visible_duration:
            (s) in transit mode, the transit duration; in eclipse mode, the duration of the planet visibility
        mode:
        number:

    Returns:

    """
    # TODO take background noise into account
    if planet_visible_duration <= 0:
        # The planet is not visible, so no signal is coming from the planet
        raise ValueError(f"the planet is not transiting/visible (visible duration = {planet_visible_duration} s)")
    elif planet_visible_duration >= observing_duration:
        # It is not possible to extract the planet signal from the data if the signal of the star alone is not taken
        raise ValueError(f"impossible to retrieve the planet transit/eclipse depth "
                         f"if transit/visible duration is greater than observing time "
                         f"({planet_visible_duration} >= {observing_duration} s)")

    if mode == 'eclipse':
        noise_per_pixel = 1 / instrument_snr \
            * np.sqrt(
                mock_observation * observing_duration
                * (observing_duration + (mock_observation - 1) * planet_visible_duration)
                / (planet_visible_duration * (observing_duration - planet_visible_duration))
            )
    elif mode == 'transit':
        noise_per_pixel = 1 / instrument_snr \
            * np.sqrt(
                mock_observation * observing_duration
                * (observing_duration - (mock_observation + 1) * planet_visible_duration)
                / (planet_visible_duration * (observing_duration - planet_visible_duration))
            )
    else:
        raise ValueError(f"Unknown mode '{mode}', must be 'transit' or 'eclipse'")

    rng = np.random.default_rng()

    mock_observation = mock_observation + rng.normal(
        loc=0.,
        scale=noise_per_pixel,
        size=(number, np.size(mock_observation, axis=0), np.size(mock_observation, axis=1))
    )

    return mock_observation, noise_per_pixel


def get_orbital_phases(phase_start, orbital_period, dit, ndit):
    """Calculate orbital phases assuming low eccentricity.

    Args:
        phase_start: planet phase at the start of observations
        orbital_period: (s) orbital period of the planet
        dit: (s) integration duration
        ndit: number of integrations

    Returns:
        ndit phases from start_phase at t=0 to the phase at t=dit * ndit
    """
    times = np.linspace(0, dit * ndit, ndit)

    return np.mod(phase_start + times / orbital_period, 1.0)  # the 2 * pi factors cancel out


def generate_mock_observations(wavelength_model, planet_spectrum_model,
                               telluric_transmittance, variable_throughput, integration_time, integration_time_ref,
                               wavelength_instrument, instrument_snr=None, instrument_resolving_power=None,
                               planet_radius=None, star_radius=None, star_spectral_radiosity=None,
                               orbital_phases=None, system_observer_radial_velocities=None,
                               planet_max_radial_orbital_velocity=0.0, planet_orbital_inclination=0.0,
                               mode='eclipse', add_noise=True, apply_snr_mask=False, number=1):
    """Generate mock observations from model spectra.

    Args:
        wavelength_model:
            (cm) wavelength of the model.
        planet_spectrum_model:
            In eclipse mode, the spectral radiosity of the planet (erg.s-1.cm-2.cm-1) or equivalent.
            In transit mode, the transit radius of the planet.
        telluric_transmittance:
            The transmittance of the Earth's atmosphere on all observations, re-binned on the instrument wavelength grid
        variable_throughput:
            Instrument variable throughput, due to various reasons
        integration_time:
            (s) integration time of 1 observation (DIT)
        integration_time_ref:
            (s) reference integration time (DIT)
        wavelength_instrument:
            (cm) the wavelength (spectral) grid of the instrument
        instrument_snr:
            The instrument Signal to Noise Ratio, per spectral pixel
        instrument_resolving_power:
            The instrument resolving power
        planet_radius:
            (cm) radius of the planet
        star_radius:
            (cm) radius of the star
        star_spectral_radiosity:
            (same units as planet_spectrum_model) star spectral radiosity, re-binned on the model wavelength grid
        orbital_phases:
            The orbital phases of the planet, 0 corresponding to the planet eclipsing the star, 0.5 to the planet being
            eclipsed by the star. The planet is moving away from the observer from 0 to 0.5, and is moving toward the
            observer from 0.5 to 1.
        system_observer_radial_velocities:
            (cm.s-1) Velocities of the star relative to the observer. In most cases, they are the barycentric + systemic
            radial velocities.
        planet_max_radial_orbital_velocity:
            (cm.s-1) planet orbital velocity, assuming a circular orbit
        planet_orbital_inclination:
            (deg) orbital inclination of the planet, 0 degree corresponding to seeing the planet's orbit by the edge,
            and 90 degrees to seeing the planet's orbit from top.
        mode:
            'eclipse' or 'transit', mode on which to make the mock observation
        add_noise:
            If True, add a random Gaussian noise to the spectrum
        apply_snr_mask:
            If True, and add_noise is False, the SNR mask will be applied to the data
        number:
            if add_noise is True, number of time to generate mock observations, re-rolling noise each time

    Returns:

    """
    # Calculate planet radial velocities
    if orbital_phases is None:
        planet_velocities = np.zeros(1)
    else:
        if not hasattr(orbital_phases, '__iter__'):
            orbital_phases = np.asarray([orbital_phases])

        planet_velocities = Planet.calculate_planet_radial_velocity(
            planet_max_radial_orbital_velocity, planet_orbital_inclination, orbital_phases
        )

    # Check types, sizes and dimensions
    if system_observer_radial_velocities is None:
        system_observer_radial_velocities = np.zeros(1)
    elif not hasattr(system_observer_radial_velocities, '__iter__'):
        system_observer_radial_velocities = np.ones_like(orbital_phases) * system_observer_radial_velocities
    elif np.size(system_observer_radial_velocities) != np.size(orbital_phases):
        raise ValueError(f"System-observer radial velocities (size {np.size(system_observer_radial_velocities)}) "
                         f"and planet orbital phases (size {np.size(orbital_phases)}) "
                         f"must be of the same size (NDIT)")

    if telluric_transmittance is None:
        telluric_transmittance = np.ones_like(wavelength_instrument)
    elif not hasattr(telluric_transmittance, '__iter__'):
        telluric_transmittance = np.ones_like(wavelength_instrument) * telluric_transmittance

    if variable_throughput is None:
        variable_throughput = np.ones_like(orbital_phases)
    elif not hasattr(variable_throughput, '__iter__'):
        variable_throughput = np.ones_like(orbital_phases) * variable_throughput

    if np.ndim(telluric_transmittance) == 1:
        # TODO add changes in airmass
        telluric_transmittance = np.asarray([telluric_transmittance] * np.size(orbital_phases))

    if np.size(telluric_transmittance, axis=1) != np.size(wavelength_instrument):
        raise ValueError(f"Telluric transmittance (shape {np.shape(telluric_transmittance)}) "
                         f"must be on the instrument spectral grid (size {np.size(wavelength_instrument)})")

    if np.size(telluric_transmittance, axis=0) != np.size(orbital_phases):
        raise ValueError(f"There must be one telluric transmittance (shape {np.shape(telluric_transmittance)}) "
                         f"per observations/NDIT (size {np.size(orbital_phases)})")

    # Simulate observations
    if mode == 'eclipse':
        if np.size(star_spectral_radiosity) != np.size(planet_spectrum_model):
            raise ValueError(f"Star spectral radiosity (size {np.size(star_spectral_radiosity)}) "
                             f"and planet spectral radiosity (size {np.size(planet_spectrum_model)}) "
                             f"must be on the same spectral grid")

        mock_observation = get_mock_secondary_eclipse_spectra(
            wavelength_model=wavelength_model,
            spectrum_model=planet_spectrum_model,
            star_spectral_radiosity=star_spectral_radiosity,
            planet_radius=planet_radius,
            star_radius=star_radius,
            wavelength_instrument=wavelength_instrument,
            instrument_resolving_power=instrument_resolving_power,
            planet_velocities=planet_velocities,
            system_observer_radial_velocities=system_observer_radial_velocities,
            planet_rest_frame_shift=0.0
        )
    elif mode == 'transit':
        mock_observation = convolve_shift_rebin(
            wavelength_model, planet_spectrum_model, instrument_resolving_power, wavelength_instrument,
            planet_velocities + system_observer_radial_velocities  # planet + system velocity
        )

        mock_observation = np.asarray([1 - mock_observation])
    else:
        raise ValueError(f"Unknown mode '{mode}', must be 'transit' or 'eclipse'")

    # Apply Earth's atmospheric effect
    mock_observation *= telluric_transmittance

    # Apply variable throughput
    mock_observation = np.transpose(variable_throughput * np.transpose(mock_observation))

    # Save no noise observations
    mock_observation_without_noise = copy.deepcopy(mock_observation)
    mock_observation_without_noise = np.asarray([mock_observation_without_noise])

    # Add noise to the model
    if add_noise:
        if np.size(instrument_snr) != np.size(wavelength_instrument):
            raise ValueError(f"Instrument SNR (size {np.size(instrument_snr)}) "
                             f"and instrument wavelength (size {np.size(wavelength_instrument)}) "
                             f"must be on the same spectral grid")

        # The noise must take into account the Doppler shift of the star, and ideally the effect of the planet
        noise_per_pixel = 1 / instrument_snr * np.sqrt(integration_time / integration_time_ref)

        rng = np.random.default_rng()

        mock_observation = mock_observation + rng.normal(
            loc=0.,
            scale=noise_per_pixel,
            size=(number, np.size(mock_observation, axis=0), np.size(mock_observation, axis=1))
        )

        # Mask invalid columns
        mock_observation = np.ma.masked_where(
            np.asarray(noise_per_pixel.mask * np.ones(mock_observation.shape)),
            mock_observation
        )

        if mock_observation.mask.size == 1:  # no masked values in noise
            mock_observation.mask = np.zeros(mock_observation.shape, dtype=bool)  # add full-fledged mask

    else:
        mock_observation = np.ma.asarray([mock_observation])  # add a third dimension for output consistency
        mock_observation.mask = np.zeros(mock_observation.shape, dtype=bool)

        if apply_snr_mask:
            mock_observation.mask[:, :, :] = instrument_snr.mask
            noise_per_pixel = 1 / instrument_snr * np.sqrt(integration_time / integration_time_ref)
        else:
            mock_observation.mask = np.zeros(mock_observation.shape, dtype=bool)  # add full-fledged mask
            noise_per_pixel = np.zeros_like(mock_observation)

    return mock_observation, noise_per_pixel, mock_observation_without_noise


def simple_mock_observation(wavelengths, flux, snr_per_res_element, observing_time, transit_duration,
                            instrument_resolving_power, pixel_sampling, instrument_wavelength_range,
                            number=1):
    """
    Generate a mock observation from a modelled transmission spectrum.
    The noise of the transmission spectrum is estimated assuming that to retrieve the planetary spectrum, the flux of
    the star with the planet transiting in front of it was subtracted to the flux of the star alone.
    It is possible to generate multiple mock observations with the same setting, but a different random noise.

    Args:
        wavelengths: (cm) the wavelengths of the model
        flux: the flux of the model
        snr_per_res_element: the signal-to-noise ratio per resolution element of the instrument
        observing_time: (s) the total time passed observing the star (in and out of transit)
        transit_duration: (s) the duration of the planet transit (must be lower than the observing time)
        instrument_resolving_power: the instrument resolving power
        pixel_sampling: the pixel sampling of the instrument
        instrument_wavelength_range: (cm) size-2 array containing the min and max wavelengths of the instrument
        number: number of mock observations to generate

    Returns:
        observed_spectrum: the modelled spectrum rebinned, altered, and with a random white noise from the instrument
        full_lsf_ed: the modelled spectrum altered by the instrument's LSF
        freq_out: (Hz) the frequencies of the rebinned spectrum (in descending order)
        full_rebinned: the modelled spectrum, rebinned and altered by the instrument's LSF
    """
    if transit_duration <= 0:
        # There is no transit, so no signal from the planet
        raise ValueError(f"the planet is not transiting (transit duration = {transit_duration} s)")
    elif transit_duration >= observing_time:
        # It is not possible to extract the planet signal if the signal of the star alone is not taken
        raise ValueError(f"impossible to retrieve the planet transit depth "
                         f"if transit duration is greater than observing time "
                         f"({transit_duration} >= {observing_time} s)")

    # Start from the nominal model, and re-bin using the instrument LSF
    full_lsf_ed, wavelengths_out, full_rebinned = convolve_rebin(
        input_wavelengths=wavelengths,
        input_flux=flux,
        instrument_resolving_power=instrument_resolving_power,
        pixel_sampling=pixel_sampling,
        instrument_wavelength_range=np.array(instrument_wavelength_range)
    )

    # Add noise to the model
    noise_per_pixel = 1 / snr_per_res_element \
        * np.sqrt(
            (1 - full_rebinned) * observing_time * (observing_time - full_rebinned * transit_duration)
            / (transit_duration * (observing_time - transit_duration))
        )

    rng = np.random.default_rng()

    observed_spectrum = full_rebinned + rng.normal(
        loc=0.,
        scale=np.ma.median(noise_per_pixel),
        size=(number, np.size(full_rebinned))
    )

    return observed_spectrum, full_lsf_ed, wavelengths_out, full_rebinned, snr_per_res_element
