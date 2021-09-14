from scipy.interpolate import interp1d
from scipy.signal import correlate
from scipy.stats import norm

import numpy as np
import petitRADTRANS.nat_cst as nc


def ccf_analysis(wavelengths, observed_spectrum, modelled_spectrum, velocity_range=2000.):
    """
    Calculate the cross-correlation between an observed spectrum and a modelled spectrum.
    The modelled spectrum can be a spectrum with e.g. the contribution of a single molecule. In that case e.g. log_l_ccf
    gives the log-likelihood of the detection of this molecule.
    The modelled spectrum has to be re-binned to the resolution of the observed spectrum.

    Args:
        wavelengths: (cm) wavelengths of the spectra
        observed_spectrum: observed spectrum
        modelled_spectrum: modelled spectrum
        velocity_range: (km.s-1) velocity range of the cross-correlation

    Returns:
        snr: the signal-to-noise ratio of the CCF
        velocities: the velocities of the CCF
        cross_correlation: the values of the cross-correlation
        log_l: the log-likelihood between the model and the observations
        log_l_ccf: the log-likelihood of the CCF
    """
    corrected_observed_spectrum = remove_large_scale_trends(wavelengths, observed_spectrum)
    corrected_modelled_spectrum = remove_large_scale_trends(wavelengths, modelled_spectrum)

    ccf = correlate(corrected_observed_spectrum, corrected_modelled_spectrum, mode='same', method='fft')

    # Get S/N of detection, the 1e-5 coefficient is to convert from cm.s-1 to km.s-1
    velocities = np.linspace(
        -(np.max(wavelengths) - np.min(wavelengths)) / np.mean(wavelengths) * nc.c * 1e-5,
        (np.max(wavelengths) - np.min(wavelengths)) / np.mean(wavelengths) * nc.c * 1e-5,
        np.size(ccf, axis=1)
    )

    wh = np.where(np.abs(velocities) < velocity_range)
    velocities = velocities[wh]

    snr = np.zeros(np.size(ccf, axis=0))
    mu = np.zeros(np.size(ccf, axis=0))
    std = np.zeros(np.size(ccf, axis=0))

    for i, ccf_ in enumerate(ccf):  # TODO this can be made more efficient
        snr[i], mu[i], std[i] = calculate_ccf_snr(velocities, ccf_[wh])

    log_l = -np.size(corrected_observed_spectrum) / 2. * np.log(
        1. / np.size(corrected_observed_spectrum) * np.sum(
            (corrected_observed_spectrum - corrected_modelled_spectrum) ** 2.,
            axis=1
        )
    )

    log_l_ccf = -np.size(corrected_observed_spectrum, axis=1) / 2. * np.log(
        np.std(corrected_observed_spectrum, axis=1) ** 2.
        - 2. * np.max(ccf[:, wh[0]][:, np.argmin(np.abs(velocities))], axis=0)
        / np.size(corrected_observed_spectrum, axis=1)
        + np.std(corrected_modelled_spectrum, axis=1) ** 2.
    )

    cross_correlation = (np.transpose(ccf[:, wh[0]]) - mu) / std

    return snr, velocities, np.transpose(cross_correlation), log_l, log_l_ccf


def calculate_ccf_snr(xval, signal):
    """
    Calculate the signal-to-noise ratio of a CCF.

    Args:
        xval: (km.s-1) velocities
        signal: a cross-correlation

    Returns:
        snr: the signal-to-noise ratio of the CCF
        mu: the mean value of the CCF's noise
        std: the standard  deviation of the CCF noise
    """
    index = np.where(np.abs(xval) > 50.)  # assumes the peak is within -50, +50
    mu, std = norm.fit(signal[index])  # fit of the CCF noise
    signal_peak = signal[np.argmin(np.abs(xval))]  # assumes the peak is at/near 0 # TODO this method to get the peak is not always accurate!
    snr = (signal_peak - mu) / std

    return snr, mu, std


def remove_large_scale_trends(freq, flux, ran=2 * 0.0015 * 1e-4):  # TODO better function?
    """
    Remove large scale trends from a spectrum.

    Args:
        freq: (cm) wavelengths of the spectrum
        flux: flux of the spectrum
        ran: (um)

    Returns:
        flux_transform: the flux of the spectrum, removed from its large scale trends
    """
    if np.ndim(flux) == 1:
        flux = np.array([flux])

    wavelength_range = np.arange(np.min(freq) - 2. * ran, np.max(freq) + 2. * ran, ran)

    vals = np.zeros((np.size(flux, axis=0), np.size(wavelength_range[:-1]))) * np.nan

    for i in range(1, np.size(wavelength_range) - 2):
        wh = np.where(
                np.logical_and(freq >= wavelength_range[i], freq < wavelength_range[i + 1])
            )[0]

        if np.size(wh) > 0:
            vals[:, i] = np.median(flux[:, wh], axis=1)
        else:
            vals[:, i] = vals[:, i - 1]

    vals[:, 0], vals[:, -1] = vals[:, 1], vals[:, -2]  # expand

    # Remove leading and tailing NaNs
    for i in range(np.size(flux, axis=0)):
        wh = np.where(np.logical_not(np.isnan(vals[i, :])))[0]
        vals[i, :wh[0]] = vals[i, wh[0]]
        vals[i, wh[-1]:] = vals[i, wh[-1]]

    # Interpolate means and divide to the flux
    taut_val = interp1d((wavelength_range[1:] + wavelength_range[:-1]) / 2., vals)
    flux_transform = flux / taut_val(freq) - 1.

    # Remove last remaining NaNs (there should be none)
    flux_transform = np.ma.masked_where(np.isnan(flux_transform), flux_transform)

    return flux_transform
