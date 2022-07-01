"""
Useful functions for data reduction.
"""
import copy
from warnings import warn

import numpy as np
from petitRADTRANS.ccf.utils import median_uncertainties, calculate_uncertainty


def __init_pipeline_outputs(spectrum, reduction_matrix, uncertainties):
    if reduction_matrix is None:
        reduction_matrix = np.ma.ones(spectrum.shape)
        reduction_matrix.mask = np.zeros(spectrum.shape, dtype=bool)

    if isinstance(spectrum, np.ma.core.MaskedArray):
        spectral_data_corrected = np.ma.zeros(spectrum.shape)
        spectral_data_corrected.mask = copy.copy(spectrum.mask)

        if uncertainties is not None:
            pipeline_uncertainties = np.ma.masked_array(copy.copy(uncertainties))
            pipeline_uncertainties.mask = copy.copy(spectrum.mask)
        else:
            pipeline_uncertainties = None
    else:
        spectral_data_corrected = np.zeros(spectrum.shape)

        if uncertainties is not None:
            pipeline_uncertainties = copy.copy(uncertainties)
        else:
            pipeline_uncertainties = None

    return spectral_data_corrected, reduction_matrix, pipeline_uncertainties


def __remove_throughput(spectrum, reduction_matrix, noise=None,
                        throughput_correction_lower_bound=0.70, throughput_correction_upper_bound=None):
    if throughput_correction_upper_bound is None:
        # Ensure that at least the brightest pixel is removed, and that the upper bound is greater than the lower bound
        throughput_correction_upper_bound = np.max(
            (throughput_correction_lower_bound, np.min((0.99, 1 - 1 / np.size(spectrum))))
        )
    elif throughput_correction_upper_bound >= throughput_correction_lower_bound:
        if np.size(spectrum) * (1 - throughput_correction_upper_bound) < 1:
            warn(
                f"data size ({np.size(spectrum)}) is low compared to the throughput correction upper bound "
                f"({throughput_correction_upper_bound}), recommended value is {1 - 1 / np.size(spectrum)}",
                UserWarning
             )
    else:
        raise ValueError(f"Throughput correction upper bound ({throughput_correction_upper_bound}) must be"
                         f"greater or equal to throughput correction lower bound ({throughput_correction_lower_bound})")

    # Look at where the brightest pixels are, in order to avoid telluric lines
    time_averaged_data = np.median(spectrum, axis=0)  # median of the data over time/integrations

    time_averaged_data_lower_bound = np.percentile(
        time_averaged_data,
        throughput_correction_lower_bound * 1e2
    )
    time_averaged_data_upper_bound = np.percentile(
        time_averaged_data,
        throughput_correction_upper_bound * 1e2
    )

    # Exclude the very brightest pixels, that could be not representative
    brightest_pixels = np.where(np.logical_and(
        time_averaged_data >= time_averaged_data_lower_bound,
        time_averaged_data <= time_averaged_data_upper_bound
    ))

    brightest_data_wavelength = spectrum[:, brightest_pixels[0]]
    brightest_data_wavelength = np.median(brightest_data_wavelength, axis=1)

    spectral_data_corrected = np.zeros(spectrum.shape)

    for i, correction_coefficient in enumerate(brightest_data_wavelength):
        spectral_data_corrected[i, :] = spectrum[i, :] / correction_coefficient
        reduction_matrix[i, :] /= correction_coefficient

    pipeline_noise = np.zeros(spectral_data_corrected.shape)

    if noise is not None:
        for i, correction_coefficient in enumerate(brightest_data_wavelength):
            brightest_data_wavelength_noise = median_uncertainties(
                noise[i, brightest_pixels[0]]
            )

            partial_derivatives = np.array([
                spectral_data_corrected[i, :] / spectrum[i, :],  # dS'/dS
                - spectral_data_corrected[i, :] / correction_coefficient  # dS'/dC
            ])

            uncertainties = np.abs(np.array([
                noise[i, :],  # sigma_S
                brightest_data_wavelength_noise * np.ones(noise[i, :].shape)  # sigma_C
            ]))

            for j in range(uncertainties.shape[1]):
                pipeline_noise[i, j] = calculate_uncertainty(partial_derivatives[:, j], uncertainties[:, j])

    return spectral_data_corrected, reduction_matrix, pipeline_noise


def _remove_telluric_lines_old(spectrum, airmass=None, remove_outliers=True, remove_standard_deviation=False):
    """Correct for Earth's atmospheric absorptions.

    Args:
        spectrum: spectral data to correct
        airmass: airmass of the data
        remove_outliers: if True, remove the pixels that are 3 times away from the median of the standard deviation of
            the data over time.
        remove_standard_deviation: if True, remove the standard deviation on time of the data (not recommended)

    Returns:
        Spectral data corrected from the telluric transmittance
    """
    for j, data in enumerate(spectrum):
        # Remove the mean of the telluric lines
        if airmass is not None:
            exp_airmass = np.exp(-airmass[j])

            for i in range(np.size(data, axis=1)):
                # Fit the telluric lines change over time with a 2nd order polynomial
                # The telluric lines opacity depends on airmass (tau = alpha / cos(theta)), so the telluric lines
                # transmittance (T = exp(-tau)) depend on exp(airmass)
                # Using a 1st order polynomial is not enough, as the atm. composition will change slowly over time
                fit_parameters = np.polyfit(x=exp_airmass, y=data[:, i], deg=2)
                fit_function = np.poly1d(fit_parameters)
                fit = fit_function(exp_airmass)  # might be necessary to mask 0 here
                data[:, i] = data[:, i] / fit

        data -= 1
        #spectral_data = np.transpose(np.transpose(spectral_data) - np.ma.mean(spectral_data, axis=1))

        # Get standard deviation over time for each wavelength
        standard_deviation_integration = np.asarray([np.std(data, axis=0)] * np.size(data, axis=0))

        # Remove telluric lines standard deviation
        if remove_standard_deviation:
            # Not recommended
            data /= standard_deviation_integration

        if remove_outliers:
            # TODO this might work when adding telluric transmittance
            #spectral_data = np.ma.masked_where(np.abs(spectral_data) > 3 * np.ma.std(spectral_data), spectral_data)
            # TODO this gives results, but is probably too restrictive, this is a very important step for log_l calc!
            data = np.ma.masked_where(
                np.abs(data) > 3 * np.ma.median(standard_deviation_integration), data
            )

        spectrum[j, :, :] = data

    return spectrum


def __remove_throughput_masked(spectrum, reduction_matrix, noise=None,
                               throughput_correction_lower_bound=0.70, throughput_correction_upper_bound=None):
    if throughput_correction_upper_bound is None:
        # Ensure that at least the brightest pixel is removed, and that the upper bound is greater than the lower bound
        throughput_correction_upper_bound = np.ma.max(
            (throughput_correction_lower_bound, np.ma.min((0.99, 1 - 1 / np.size(spectrum))))
        )
    elif throughput_correction_upper_bound >= throughput_correction_lower_bound:
        if spectrum.size * (1 - throughput_correction_upper_bound) < 1:
            warn(
                f"data size ({spectrum.size}) is low compared to the throughput correction upper bound "
                f"({throughput_correction_upper_bound}), recommended value is {1 - 1 / spectrum.size}",
                UserWarning
             )
    else:
        raise ValueError(f"Throughput correction upper bound ({throughput_correction_upper_bound}) must be"
                         f"greater or equal to throughput correction lower bound ({throughput_correction_lower_bound})")

    # Look at where the brightest pixels are, in order to avoid telluric lines
    time_averaged_data = np.ma.median(spectrum, axis=0)  # median of the data over time/integrations
    time_averaged_data = np.ma.array(time_averaged_data)  # ensure array is masked

    time_averaged_data_lower_bound = np.percentile(
        time_averaged_data[~time_averaged_data.mask],
        throughput_correction_lower_bound * 1e2
    )
    time_averaged_data_upper_bound = np.percentile(
        time_averaged_data[~time_averaged_data.mask],
        throughput_correction_upper_bound * 1e2
    )

    # Exclude the very brightest pixels, that could be not representative
    brightest_pixels = np.ma.where(np.logical_and(
        time_averaged_data >= time_averaged_data_lower_bound,
        time_averaged_data <= time_averaged_data_upper_bound
    ))

    # Time-dependent median of the brightest wavelengths
    brightest_data_wavelength = np.ma.median(spectrum[:, brightest_pixels[0]], axis=1)

    spectral_data_corrected = np.ma.zeros(spectrum.shape)
    spectral_data_corrected.mask = copy.copy(spectrum.mask)

    for i, correction_coefficient in enumerate(brightest_data_wavelength):
        spectral_data_corrected[i, :] = spectrum[i, :] / correction_coefficient
        reduction_matrix[i, :] /= correction_coefficient

    pipeline_noise = np.ma.zeros(spectral_data_corrected.shape)
    pipeline_noise.mask = copy.copy(spectrum.mask)

    if noise is not None:
        for i, correction_coefficient in enumerate(brightest_data_wavelength):
            brightest_data_wavelength_noise = median_uncertainties(
                noise[i, brightest_pixels[0]][~spectrum[i, brightest_pixels[0]].mask]
            )

            partial_derivatives = np.array([
                spectral_data_corrected[i, :] / spectrum[i, :],  # dS'/dS
                - spectral_data_corrected[i, :] / correction_coefficient  # dS'/dC
            ])

            uncertainties = np.abs(np.array([
                noise[i, :],  # sigma_S
                brightest_data_wavelength_noise * np.ones(noise[i, :].shape)  # sigma_C
            ]))

            for j in range(uncertainties.shape[1]):
                pipeline_noise[i, j] = calculate_uncertainty(partial_derivatives[:, j], uncertainties[:, j])
                pipeline_noise.mask[i, j] = spectrum.mask[i, j]

    return spectral_data_corrected, reduction_matrix, pipeline_noise


def _remove_throughput_test(spectrum, reduction_matrix, data_noise=None,
                            throughput_correction_lower_bound=0.70, throughput_correction_upper_bound=None, mean=False):
    """Correct for the variable throughput.

    Args:
        spectrum: spectral data to correct
        reduction_matrix: matrix storing all the operations made to reduce the data
        data_noise: noise of the data
        throughput_correction_lower_bound: [0-1] quantile lower bound on throughput correction
        throughput_correction_upper_bound: [0-1] quantile upper bound on throughput correction
        mean: if True, use mean instead of Brogi et al. methodology

    Returns:
        Spectral data corrected from throughput
    """
    if isinstance(spectrum, np.ma.core.MaskedArray):
        spectral_data_corrected = np.ma.zeros(spectrum.shape)
        spectral_data_corrected.mask = copy.copy(spectrum.mask)

        if data_noise is not None:
            pipeline_noise = np.ma.masked_array(copy.copy(data_noise))
            pipeline_noise.mask = copy.copy(spectrum.mask)
        else:
            pipeline_noise = None
    else:
        spectral_data_corrected = np.zeros(spectrum.shape)

        if data_noise is not None:
            pipeline_noise = copy.copy(data_noise)
        else:
            pipeline_noise = None

    if data_noise is not None:
        weights = 1 / data_noise
    else:
        weights = np.ones(spectrum.shape)

    if mean:
        for i, data in enumerate(spectrum):
            if isinstance(spectrum, np.ma.core.MaskedArray):
                # correction_coefficient = np.ma.mean(data, axis=1)
                # print('Mean!!')
                correction_coefficient = np.ma.average(data, axis=1, weights=weights[i])
            elif isinstance(spectrum, np.ndarray):
                # correction_coefficient = np.mean(data, axis=1)
                # print('Mean!!')
                correction_coefficient = np.average(data, axis=1, weights=weights[i])
            else:
                raise ValueError(f"spectral_data must be a numpy.ndarray or a numpy.ma.core.MaskedArray, "
                                 f"but is of type '{type(spectrum)}'")

            spectral_data_corrected[i, :, :] = np.transpose(
                np.transpose(data) / correction_coefficient
            )
            reduction_matrix[i, :, :] = np.transpose(
                np.transpose(reduction_matrix[i, :, :]) / correction_coefficient
            )

            # TODO check why chi2 < 1 using this and true parameters
            if data_noise is not None:
                pipeline_noise[i, :, :] = np.transpose(
                    np.transpose(pipeline_noise[i, :, :]) / np.abs(correction_coefficient)
                )

                # for j, correction_coefficient_ in enumerate(correction_coefficient):
                #     brightest_data_wavelength_noise = mean_uncertainty(
                #         data_noise[i, j, :][~data[j, :].mask]
                #     )
                #
                #     partial_derivatives = np.array([
                #         spectral_data_corrected[i, j, :] / data[j, :],  # dS'/dS
                #         - spectral_data_corrected[i, j, :] / correction_coefficient_  # dS'/dC
                #     ])
                #
                #     uncertainties = np.abs(np.array([
                #         data_noise[i, j, :],  # sigma_S
                #         brightest_data_wavelength_noise * np.ones(data_noise[i, j, :].shape)  # sigma_C
                #     ]))
                #
                #     for k in range(uncertainties.shape[1]):
                #         pipeline_noise[i, j, k] = calculate_uncertainty(partial_derivatives[:, k], uncertainties[:, k])
                #         pipeline_noise.mask[i, j, k] = data.mask[j, k]
    else:
        print('Not mean!!')
        for i, data in enumerate(spectrum):
            if isinstance(spectrum, np.ma.core.MaskedArray):
                spectral_data_corrected[i, :, :], reduction_matrix[i, :, :], pipeline_noise[i, :, :] = \
                    __remove_throughput_masked(
                        data, reduction_matrix[i, :, :], data_noise[i],
                        throughput_correction_lower_bound, throughput_correction_upper_bound
                    )
            elif isinstance(spectrum, np.ndarray):
                spectral_data_corrected[i, :, :], reduction_matrix[i, :, :], pipeline_noise[i, :, :] = \
                    __remove_throughput(
                        data, reduction_matrix[i, :, :], data_noise[i],
                        throughput_correction_lower_bound, throughput_correction_upper_bound
                    )
            else:
                raise ValueError(f"spectral_data must be a numpy.ndarray or a numpy.ma.core.MaskedArray, "
                                 f"but is of type '{type(spectrum)}'")

            # spectral_data_corrected[i, :, :] = np.transpose(
            #     np.transpose(spectral_data_corrected[i, :, :])
            #     / np.mean(spectral_data_corrected[i, :, :], axis=1)
            # )
            # print(np.mean(spectral_data_corrected[i, :, :], axis=1))
    return spectral_data_corrected, reduction_matrix, pipeline_noise


def _simple_pipeline_test(spectrum, airmass=None, times=None, data_noise=None,
                          throughput_correction_lower_bound=0.70, throughput_correction_upper_bound=None,
                          mean_subtract=False, mean=False):
    """Removes the telluric lines and variable throughput of some data.

    Args:
        spectrum: spectral data to correct
        airmass: airmass of the data
        times: (s) time after first observation: t(i) = dit * (ndit(i) - 1)
        data_noise: noise of the data
        throughput_correction_lower_bound: [0-1] quantile lower bound on throughput correction
        throughput_correction_upper_bound: [0-1] quantile upper bound on throughput correction
        mean_subtract: if True, the data corresponding to each spectrum are mean subtracted

    Returns:
        Spectral data corrected from variable throughput
    """
    reduction_matrix = np.ma.ones(spectrum.shape)
    reduction_matrix.mask = np.zeros(spectrum.shape, dtype=bool)
    spectral_data_corrected = copy.copy(spectrum)

    if isinstance(spectrum, np.ma.core.MaskedArray):
        spectral_data_corrected.mask = copy.copy(spectrum.mask)

        if data_noise is not None:
            pipeline_noise = np.ma.masked_array(copy.copy(data_noise))
            pipeline_noise.mask = copy.copy(spectrum.mask)
        else:
            pipeline_noise = None
    else:
        if data_noise is not None:
            pipeline_noise = copy.copy(data_noise)
        else:
            pipeline_noise = None

    # return spectral_data_corrected, reduction_matrix, pipeline_noise

    # print('rm th')
    if True:
        spectral_data_corrected, reduction_matrix, pipeline_noise = _remove_throughput_test(
            spectrum=spectrum,
            reduction_matrix=reduction_matrix,
            data_noise=pipeline_noise,
            throughput_correction_lower_bound=throughput_correction_lower_bound,
            throughput_correction_upper_bound=throughput_correction_upper_bound,
            mean=mean
        )

    # return spectral_data_corrected, reduction_matrix, pipeline_noise

    # print('rm tl')
    if airmass is None:
        spectral_data_corrected, reduction_matrix, pipeline_noise = remove_telluric_lines_mean(
            spectrum=spectral_data_corrected,
            reduction_matrix=reduction_matrix,  # TODO separate the 2 reduction matrices?
            uncertainties=pipeline_noise,
            mask_threshold=0.2
        )
    else:
        spectral_data_corrected, reduction_matrix, pipeline_noise = remove_telluric_lines_fit(
            spectrum=spectral_data_corrected,
            reduction_matrix=reduction_matrix,
            airmass=airmass,
            uncertainties=pipeline_noise,
            mask_threshold=0.2,
            polynomial_fit_degree=2
        )
        # print('p2', np.mean(pipeline_noise))

    return spectral_data_corrected, reduction_matrix, pipeline_noise

    # print('rm nc')
    # spectral_data_corrected, reduction_matrix = remove_noisy_wavelength_channels(
    #     spectral_data_corrected, reduction_matrix, mean_subtract
    # )
    #
    # return spectral_data_corrected, reduction_matrix, pipeline_noise


def pipeline_validity_test(reduced_true_model, reduced_mock_observations,
                           mock_observations_reduction_matrix=None, mock_noise=None):
    if mock_observations_reduction_matrix is None:
        mock_observations_reduction_matrix = np.ones(reduced_true_model.shape)

    if mock_noise is None:
        mock_noise = np.zeros(reduced_true_model.shape)

    return 1 - (reduced_true_model - mock_noise * mock_observations_reduction_matrix) / reduced_mock_observations


def remove_noisy_wavelength_channels(spectrum, reduction_matrix, mean_subtract=False):
    for i, data in enumerate(spectrum):
        # Get standard deviation over time, for each wavelength channel
        time_standard_deviation = np.asarray([np.ma.std(data, axis=0)] * np.size(data, axis=0))

        # Mask channels where the standard deviation is greater than the total standard deviation
        data = np.ma.masked_where(
            time_standard_deviation > 3 * np.ma.std(data), data
        )

        spectrum[i, :, :] = data

    if mean_subtract:
        mean_spectra = np.mean(spectrum, axis=2)  # mean over wavelengths of each individual spectrum
        spectrum -= mean_spectra
        reduction_matrix -= mean_spectra

    return spectrum, reduction_matrix


def remove_telluric_lines_fit(spectrum, reduction_matrix, airmass, uncertainties=None, mask_threshold=1e-16,
                              polynomial_fit_degree=2):
    """Remove telluric lines with a polynomial function.
    The telluric transmittance can be written as:
        T = exp(-airmass * optical_depth),
    hence the log of the transmittance can be written as a first order polynomial:
        log(T) ~ b * airmass + a.
    Using a 1st order polynomial might be not enough, as the atmospheric composition can change slowly over time. Using
    a second order polynomial, as in:
        log(T) ~ c * airmass ** 2 + b * airmass + a,
    might be safer.

    Args:
        spectrum: spectral data to correct
        reduction_matrix: matrix storing all the operations made to reduce the data
        airmass: airmass of the data
        uncertainties: uncertainties on the data
        mask_threshold: mask wavelengths where the Earth atmospheric transmittance estimate is below this value
        polynomial_fit_degree: degree of the polynomial fit of the Earth atmospheric transmittance

    Returns:
        Corrected spectral data, reduction matrix and uncertainties after correction
    """
    # Initialization
    spectral_data_corrected, reduction_matrix, pipeline_uncertainties = __init_pipeline_outputs(
        spectrum, reduction_matrix, uncertainties
    )

    if uncertainties is not None:
        weights = 1 / uncertainties
        weights[weights.mask] = 0
    else:
        weights = np.ones(spectrum.shape)

    telluric_lines_fits = np.ma.zeros(spectral_data_corrected.shape)

    # Correction
    for i, det in enumerate(spectrum):
        # Mask wavelength columns where at least one value is lower or equal to 0, to avoid invalid log values
        masked_det = np.ma.masked_where(np.ones(det.shape) * np.min(det, axis=0) <= 0, det)
        log_det_t = np.ma.log(np.transpose(masked_det))

        # Fit each wavelength column
        for k, log_wavelength_column in enumerate(log_det_t):
            fit_parameters = np.polynomial.Polynomial.fit(
                x=airmass, y=log_wavelength_column, deg=polynomial_fit_degree, w=weights[i, :, k]
            )
            fit_function = np.polynomial.Polynomial(fit_parameters.convert().coef)
            telluric_lines_fits[i, :, k] = fit_function(airmass)

        # Calculate telluric transmittance estimate
        telluric_lines_fits[i, :, :] = np.exp(telluric_lines_fits[i, :, :])

        # Apply mask where estimate is lower than the threshold, as well as the data mask
        telluric_lines_fits[i, :, :] = np.ma.masked_where(
            np.ones(telluric_lines_fits[i].shape) * np.min(telluric_lines_fits[i, :, :], axis=0) < mask_threshold,
            telluric_lines_fits[i, :, :]
        )
        telluric_lines_fits[i, :, :] = np.ma.masked_where(
            masked_det.mask, telluric_lines_fits[i, :, :]
        )

        # Apply correction
        spectral_data_corrected[i, :, :] = det / telluric_lines_fits[i, :, :]
        reduction_matrix[i, :, :] /= telluric_lines_fits[i, :, :]

    # Propagation of uncertainties
    if uncertainties is not None:
        pipeline_uncertainties /= np.abs(telluric_lines_fits)

    return spectral_data_corrected, reduction_matrix, pipeline_uncertainties


def remove_telluric_lines_mean(spectrum, reduction_matrix, uncertainties=None, mask_threshold=1e-16):
    """Remove the telluric lines using the weighted arithmetic mean over time.

    Args:
        spectrum: spectral data to correct
        reduction_matrix: matrix storing all the operations made to reduce the data
        uncertainties: uncertainties on the data
        mask_threshold: mask wavelengths where the Earth atmospheric transmittance estimate is below this value

    Returns:
        Corrected spectral data, reduction matrix and uncertainties after correction
    """
    # Initialization
    spectral_data_corrected, reduction_matrix, pipeline_uncertainties = __init_pipeline_outputs(
        spectrum, reduction_matrix, uncertainties
    )

    if uncertainties is not None:
        weights = 1 / uncertainties
        weights[weights.mask] = 0
    else:
        weights = np.ones(spectrum.shape)

    mean_spectrum_time = np.ma.average(spectrum, axis=1, weights=weights)
    mean_spectrum_time = np.ma.masked_array(mean_spectrum_time)  # ensure that it is a masked array

    # Correction
    if isinstance(spectral_data_corrected, np.ma.core.MaskedArray):
        for i, data in enumerate(spectrum):
            mean_spectrum_time[i] = np.ma.masked_where(
                mean_spectrum_time[i] < mask_threshold, mean_spectrum_time[i]
            )
            spectral_data_corrected.mask[i, :, :] = mean_spectrum_time.mask[i]
            reduction_matrix.mask[i, :, :] = mean_spectrum_time.mask[i]
            spectral_data_corrected[i, :, :] = data / mean_spectrum_time[i]
            reduction_matrix[i, :, :] /= mean_spectrum_time[i]
    else:
        for i, data in enumerate(spectrum):
            spectral_data_corrected[i, :, :] = data / mean_spectrum_time[i]
            reduction_matrix[i, :, :] /= mean_spectrum_time[i]

    if uncertainties is not None:
        for i, data in enumerate(spectrum):
            pipeline_uncertainties[i, :, :] /= np.abs(mean_spectrum_time[i])

    return spectral_data_corrected, reduction_matrix, pipeline_uncertainties


def remove_throughput_fit(spectrum, reduction_matrix, wavelengths, uncertainties=None, mask_threshold=1e-16,
                          polynomial_fit_degree=2):
    """Remove variable throughput with a polynomial function.

    Args:
        spectrum: spectral data to correct
        reduction_matrix: matrix storing all the operations made to reduce the data
        wavelengths: wavelengths of the data
        uncertainties: uncertainties on the data
        mask_threshold: mask wavelengths where the Earth atmospheric transmittance estimate is below this value
        polynomial_fit_degree: degree of the polynomial fit of the Earth atmospheric transmittance

    Returns:
        Corrected spectral data, reduction matrix and uncertainties after correction
    """
    # Initialization
    spectral_data_corrected, reduction_matrix, pipeline_uncertainties = __init_pipeline_outputs(
        spectrum, reduction_matrix, uncertainties
    )

    if uncertainties is not None:
        weights = 1 / uncertainties
        weights[weights.mask] = 0
    else:
        weights = np.ones(spectrum.shape)

    throughput_fits = np.ma.zeros(spectral_data_corrected.shape)

    if np.ndim(wavelengths) == 3:
        print('Assuming same wavelength solution for each observations, taking wavelengths of observation 0')

    # Correction
    for i, det in enumerate(spectrum):
        if np.ndim(wavelengths) == 1:
            wvl = wavelengths
        elif np.ndim(wavelengths) == 2:
            wvl = wavelengths[i, :]
        elif np.ndim(wavelengths) == 3:
            wvl = wavelengths[i, 0, :]
        else:
            raise ValueError(f"wavelengths must have at most 3 dimensions, but has {np.ndim(wavelengths)}")

        # Fit each observation
        for j, observation in enumerate(det):
            fit_parameters = np.polynomial.Polynomial.fit(
                x=wvl, y=observation, deg=polynomial_fit_degree, w=weights[i, j, :]
            )
            fit_function = np.polynomial.Polynomial(fit_parameters.convert().coef)
            throughput_fits[i, j, :] = fit_function(wvl)

        # Apply mask where estimate is lower than the threshold, as well as the data mask
        throughput_fits[i, :, :] = np.ma.masked_where(
            np.ones(throughput_fits[i].shape) * np.min(throughput_fits[i, :, :], axis=0) < mask_threshold,
            throughput_fits[i, :, :]
        )
        throughput_fits[i, :, :] = np.ma.masked_where(
            det.mask, throughput_fits[i, :, :]
        )

        # Apply correction
        spectral_data_corrected[i, :, :] = det / throughput_fits[i, :, :]
        reduction_matrix[i, :, :] /= throughput_fits[i, :, :]

    # Propagation of uncertainties
    if uncertainties is not None:
        pipeline_uncertainties /= np.abs(throughput_fits)

    return spectral_data_corrected, reduction_matrix, pipeline_uncertainties


def remove_throughput_mean(spectrum, reduction_matrix=None, uncertainties=None):
    """Correct for the variable throughput using the weighted arithmetic mean over wavelength.

    Args:
        spectrum: spectral data to correct
        reduction_matrix: matrix storing all the operations made to reduce the data
        uncertainties: uncertainties on the data

    Returns:
        Corrected spectral data, reduction matrix and uncertainties after correction
    """
    # Initialization
    spectral_data_corrected, reduction_matrix, pipeline_uncertainties = __init_pipeline_outputs(
        spectrum, reduction_matrix, uncertainties
    )

    if uncertainties is not None:
        weights = 1 / uncertainties
        weights[weights.mask] = 0
    else:
        weights = np.ones(spectrum.shape)

    # Correction
    for i, data in enumerate(spectrum):
        if isinstance(spectrum, np.ma.core.MaskedArray):
            correction_coefficient = np.ma.average(data, axis=1, weights=weights[i])
        elif isinstance(spectrum, np.ndarray):
            correction_coefficient = np.average(data, axis=1, weights=weights[i])
        else:
            raise ValueError(f"spectral_data must be a numpy.ndarray or a numpy.ma.core.MaskedArray, "
                             f"but is of type '{type(spectrum)}'")

        spectral_data_corrected[i, :, :] = np.transpose(np.transpose(data) / correction_coefficient)
        reduction_matrix[i, :, :] = np.transpose(np.transpose(reduction_matrix[i, :, :]) / correction_coefficient)

        if uncertainties is not None:
            pipeline_uncertainties[i, :, :] = np.transpose(
                np.transpose(pipeline_uncertainties[i, :, :]) / np.abs(correction_coefficient)
            )

    return spectral_data_corrected, reduction_matrix, pipeline_uncertainties


def simple_pipeline(spectrum, uncertainties=None,
                    wavelengths=None, airmass=None, tellurics_mask_threshold=0.1, polynomial_fit_degree=1,
                    apply_throughput_removal=True, apply_telluric_lines_removal=True, full=False, **kwargs):
    """Removes the telluric lines and variable throughput of some data.
    If airmass is None, the Earth atmospheric transmittance is assumed to be time-independent, so telluric transmittance
    will be fitted using the weighted arithmetic mean. Otherwise, telluric transmittance are fitted with a polynomial.

    Args:
        spectrum: spectral data to correct
        uncertainties: uncertainties on the data
        wavelengths: wavelengths of the data
        airmass: airmass of the data
        tellurics_mask_threshold: mask wavelengths where the Earth atmospheric transmittance estimate is below this value
        polynomial_fit_degree: degree of the polynomial fit of the Earth atmospheric transmittance
        apply_throughput_removal: if True, apply the throughput removal correction
        apply_telluric_lines_removal: if True, apply the telluric lines removal correction

    Returns:
        Reduced spectral data, reduction matrix and uncertainties after reduction
    """
    # Initialize reduction matrix
    reduction_matrix = np.ma.ones(spectrum.shape)
    reduction_matrix.mask = np.zeros(spectrum.shape, dtype=bool)

    # Initialize reduced data and pipeline noise
    reduced_data = copy.copy(spectrum)

    if isinstance(spectrum, np.ma.core.MaskedArray):
        reduced_data.mask = copy.copy(spectrum.mask)

        if uncertainties is not None:
            reduced_data_uncertainties = np.ma.masked_array(copy.copy(uncertainties))
            reduced_data_uncertainties.mask = copy.copy(spectrum.mask)
        else:
            reduced_data_uncertainties = None
    else:
        if uncertainties is not None:
            reduced_data_uncertainties = copy.copy(uncertainties)
        else:
            reduced_data_uncertainties = None

    # Apply corrections
    if apply_throughput_removal:
        if wavelengths is None:
            reduced_data, reduction_matrix, reduced_data_uncertainties = remove_throughput_mean(
                spectrum=spectrum,
                reduction_matrix=reduction_matrix,
                uncertainties=reduced_data_uncertainties
            )
        else:
            reduced_data, reduction_matrix, reduced_data_uncertainties = remove_throughput_fit(
                spectrum=spectrum,
                reduction_matrix=reduction_matrix,
                wavelengths=wavelengths,
                uncertainties=reduced_data_uncertainties,
                mask_threshold=1e-16,
                polynomial_fit_degree=2
            )

    if apply_telluric_lines_removal:
        if airmass is None:
            reduced_data, reduction_matrix, reduced_data_uncertainties = remove_telluric_lines_mean(
                spectrum=reduced_data,
                reduction_matrix=reduction_matrix,
                uncertainties=reduced_data_uncertainties,
                mask_threshold=tellurics_mask_threshold
            )
        else:
            reduced_data, reduction_matrix, reduced_data_uncertainties = remove_telluric_lines_fit(
                spectrum=reduced_data,
                reduction_matrix=reduction_matrix,
                airmass=airmass,
                uncertainties=reduced_data_uncertainties,
                mask_threshold=tellurics_mask_threshold,
                polynomial_fit_degree=polynomial_fit_degree
            )

    if full:
        return reduced_data, reduction_matrix, reduced_data_uncertainties
    else:
        return reduced_data
