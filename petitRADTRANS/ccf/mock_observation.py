"""

"""
import numpy as np
from petitRADTRANS import nat_cst as nc
from petitRADTRANS import rebin_give_width as rgw
from scipy.ndimage.filters import gaussian_filter1d


def convolve_rebin(input_frequency, input_flux,
                   instrument_resolving_power, pixel_sampling, instrument_wavelength_range):
    """
    Function to convolve observation with instrument obs and rebin to pixels of detector.
    Create mock observation for high-res spectrograph.

    Args:
        input_frequency: (Hz) frequencies of the input spectrum
        input_flux: flux of the input spectrum
        instrument_resolving_power: resolving power of the instrument
        pixel_sampling: pixel sampling of the instrument
        instrument_wavelength_range: (um) wavelength range of the instrument

    Returns:
        flux_lsf: flux altered by the instrument's LSF
        freq_out: (Hz) frequencies of the rebinned flux, in descending order
        flux_rebin: the rebinned flux
    """
    # Make sure frequencies are in ascending order
    if input_frequency[0] > input_frequency[-1]:
        input_frequency = input_frequency[::-1]
        input_flux = input_flux[::-1]

    # From talking to Ignas: delta lambda of resolution element is the FWHM of the instrument's LSF (here: a gaussian)
    sigma_lsf = 1. / instrument_resolving_power / (2. * np.sqrt(2. * np.log(2.)))

    # The input resolution of petitCODE is 1e6, but just compute
    # it to be sure, or more versatile in the future.
    # Also, we have a log-spaced grid, so the resolution is constant
    # as a function of wavelength
    model_resolving_power = np.mean((input_frequency[1:] + input_frequency[:-1]) / (2. * np.diff(input_frequency)))

    # Calculate the sigma to be used in the gauss filter in units of input frequency bins
    sigma_lsf_gauss_filter = sigma_lsf * model_resolving_power

    flux_lsf = gaussian_filter1d(
        input=input_flux,
        sigma=sigma_lsf_gauss_filter,
        mode='reflect'
    )

    if np.size(instrument_wavelength_range) == 2:
        freq_out_borders = np.logspace(np.log10(nc.c / instrument_wavelength_range[1]),
                                       np.log10(nc.c / instrument_wavelength_range[0]),
                                       int(pixel_sampling * instrument_resolving_power *
                                           np.log(instrument_wavelength_range[1] / instrument_wavelength_range[0])))
        freq_out = (freq_out_borders[1:] + freq_out_borders[:-1]) / 2.
        bin_width = np.diff(freq_out_borders)
    elif np.size(instrument_wavelength_range) > 2:
        freq_out = nc.c / instrument_wavelength_range[::-1]
        diffs = np.diff(freq_out)
        bin_width = np.concatenate(([diffs[0]], 0.5 * (diffs[:-1] + diffs[1:]), [diffs[-1]]))
    else:
        raise ValueError(f"instrument wavelength must be of size 2 or more, "
                         f"but is of size {np.size(instrument_wavelength_range)}: {instrument_wavelength_range}")

    flux_rebin = rgw.rebin_give_width(input_frequency, flux_lsf, freq_out[1:-1], bin_width[1:-1])

    return flux_lsf[::-1], freq_out[1:-1][::-1], flux_rebin[::-1]


def generate_mock_observation(wavelengths, flux, snr_per_res_element,
                              instrument_resolving_power, pixel_sampling, instrument_wavelength_range):
    """
    Generate a mock observation from a modelled spectrum.

    Args:
        wavelengths: (cm) the wavelengths of the model
        flux: the flux of the model
        snr_per_res_element: the signal-to-noise ratio per resolution element of the instrument
        instrument_resolving_power: the instrument resolving power
        pixel_sampling: the pixel sampling of the instrument
        instrument_wavelength_range: (cm) size-2 array containing the min and max wavelengths of the instrument

    Returns:
        observed_spectrum: the modelled spectrum rebinned, altered, and with a random white noise from the instrument
        full_lsf_ed: the modelled spectrum altered by the instrument's LSF
        freq_out: (Hz) the frequencies of the rebinned spectrum (in descending order)
        full_rebinned: the modelled spectrum, rebinned and altered by the instrument's LSF
    """
    # Start from the nominal model, and re-bin using the instrument LSF
    full_lsf_ed, freq_out, full_rebinned = convolve_rebin(
        input_frequency=nc.c / wavelengths,
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
        freq_out = freq_out[wh]
        full_lsf_ed = full_lsf_ed[wh]

    # Add noise to the model
    noise_per_pixel = 1 / snr_per_res_element
    observed_spectrum = full_rebinned + np.random.normal(
        loc=0.,
        scale=noise_per_pixel,
        size=len(full_rebinned)
    )

    return observed_spectrum, full_lsf_ed, freq_out, full_rebinned, snr_per_res_element
