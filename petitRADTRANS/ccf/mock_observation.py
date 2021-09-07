"""

"""
import numpy as np
from petitRADTRANS.fort_rebin import fort_rebin as fr
from scipy.ndimage.filters import gaussian_filter1d


def convolve_rebin(input_wavelengths, input_flux,
                   instrument_resolving_power, pixel_sampling, instrument_wavelength_range):
    """
    Function to convolve observation with instrument obs and rebin to pixels of detector.
    Create mock observation for high-res spectrograph.

    Args:
        input_wavelengths: (cm) wavelengths of the input spectrum
        input_flux: flux of the input spectrum
        instrument_resolving_power: resolving power of the instrument
        pixel_sampling: pixel sampling of the instrument
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


def generate_mock_observation(wavelengths, flux, snr_per_res_element, observing_time, transit_duration,
                              instrument_resolving_power, pixel_sampling, instrument_wavelength_range):
    """
    Generate a mock observation from a modelled transmission spectrum.
    The noise of the transmission spectrum is estimated assuming that to retrieve the planetary spectrum, the flux of
    the star with the planet transiting in front of it was subtracted to the flux of the star alone.

    Args:
        wavelengths: (cm) the wavelengths of the model
        flux: the flux of the model
        snr_per_res_element: the signal-to-noise ratio per resolution element of the instrument
        observing_time: (s) the total time passed observing the star (in and out of transit)
        transit_duration: (s) the duration of the planet transit (must be lower than the observing time)
        instrument_resolving_power: the instrument resolving power
        pixel_sampling: the pixel sampling of the instrument
        instrument_wavelength_range: (cm) size-2 array containing the min and max wavelengths of the instrument

    Returns:
        observed_spectrum: the modelled spectrum rebinned, altered, and with a random white noise from the instrument
        full_lsf_ed: the modelled spectrum altered by the instrument's LSF
        freq_out: (Hz) the frequencies of the rebinned spectrum (in descending order)
        full_rebinned: the modelled spectrum, rebinned and altered by the instrument's LSF
    """
    if transit_duration <= 0:
        # There is no transit, so no signal from the planet
        raise ValueError(f"the planet is not transiting (transit duration = {transit_duration})")
    elif transit_duration >= observing_time:
        # It is not possible to extract the planet signal if the signal of the star alone is not taken
        raise ValueError(f"impossible to retrieve the planet transit depth "
                         f"if transit duration is greater then observing time ({transit_duration} >= {observing_time})")

    # Start from the nominal model, and re-bin using the instrument LSF
    full_lsf_ed, wavelengths_out, full_rebinned = convolve_rebin(
        input_wavelengths=wavelengths,
        input_flux=flux,
        instrument_resolving_power=instrument_resolving_power,
        pixel_sampling=pixel_sampling,
        instrument_wavelength_range=np.array(instrument_wavelength_range)
    )

    # Remove 0 SNR  # TODO better way to handle 0 SNR?
    if np.size(snr_per_res_element) > 1:
        wh = np.where(snr_per_res_element[1:-1] != 0)

        snr_per_res_element = snr_per_res_element[1:-1][wh]
        full_rebinned = full_rebinned[wh]
        wavelengths_out = wavelengths_out[wh]
        full_lsf_ed = full_lsf_ed[wh]

    # Add noise to the model
    noise_per_pixel = 1 / snr_per_res_element \
        * np.sqrt(
            (1 - full_rebinned) * observing_time * (observing_time - full_rebinned * transit_duration)
            / (transit_duration * (observing_time - transit_duration))
        )
    observed_spectrum = full_rebinned + np.random.normal(
        loc=0.,
        scale=noise_per_pixel,
        size=len(full_rebinned)
    )

    return observed_spectrum, full_lsf_ed, wavelengths_out, full_rebinned, snr_per_res_element
