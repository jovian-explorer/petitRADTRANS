"""
Useful functions for data reduction.
"""
import copy
from warnings import warn

import numpy as np
from petitRADTRANS.ccf.utils import median_uncertainties, calculate_uncertainty


def __remove_throughput(spectral_data, reduction_matrix, noise=None,
                        throughput_correction_lower_bound=0.70, throughput_correction_upper_bound=None):
    if throughput_correction_upper_bound is None:
        # Ensure that at least the brightest pixel is removed, and that the upper bound is greater than the lower bound
        throughput_correction_upper_bound = np.max(
            (throughput_correction_lower_bound, np.min((0.99, 1 - 1 / np.size(spectral_data))))
        )
    elif throughput_correction_upper_bound >= throughput_correction_lower_bound:
        if np.size(spectral_data) * (1 - throughput_correction_upper_bound) < 1:
            warn(
                f"data size ({np.size(spectral_data)}) is low compared to the throughput correction upper bound "
                f"({throughput_correction_upper_bound}), recommended value is {1 - 1 / np.size(spectral_data)}",
                UserWarning
             )
    else:
        raise ValueError(f"Throughput correction upper bound ({throughput_correction_upper_bound}) must be"
                         f"greater or equal to throughput correction lower bound ({throughput_correction_lower_bound})")

    # Look at where the brightest pixels are, in order to avoid telluric lines
    time_averaged_data = np.median(spectral_data, axis=0)  # median of the data over time/integrations

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

    brightest_data_wavelength = spectral_data[:, brightest_pixels[0]]
    brightest_data_wavelength = np.median(brightest_data_wavelength, axis=1)

    spectral_data_corrected = np.zeros(spectral_data.shape)

    for i, correction_coefficient in enumerate(brightest_data_wavelength):
        spectral_data_corrected[i, :] = spectral_data[i, :] / correction_coefficient
        reduction_matrix[i, :] /= correction_coefficient

    pipeline_noise = np.zeros(spectral_data_corrected.shape)

    if noise is not None:
        for i, correction_coefficient in enumerate(brightest_data_wavelength):
            brightest_data_wavelength_noise = median_uncertainties(
                noise[i, brightest_pixels[0]]
            )

            partial_derivatives = np.array([
                spectral_data_corrected[i, :] / spectral_data[i, :],  # dS'/dS
                - spectral_data_corrected[i, :] / correction_coefficient  # dS'/dC
            ])

            uncertainties = np.abs(np.array([
                noise[i, :],  # sigma_S
                brightest_data_wavelength_noise * np.ones(noise[i, :].shape)  # sigma_C
            ]))

            for j in range(uncertainties.shape[1]):
                pipeline_noise[i, j] = calculate_uncertainty(partial_derivatives[:, j], uncertainties[:, j])

    return spectral_data_corrected, reduction_matrix, pipeline_noise


def __remove_throughput_masked(spectral_data, reduction_matrix, noise=None,
                               throughput_correction_lower_bound=0.70, throughput_correction_upper_bound=None):
    if throughput_correction_upper_bound is None:
        # Ensure that at least the brightest pixel is removed, and that the upper bound is greater than the lower bound
        throughput_correction_upper_bound = np.ma.max(
            (throughput_correction_lower_bound, np.ma.min((0.99, 1 - 1 / np.size(spectral_data))))
        )
    elif throughput_correction_upper_bound >= throughput_correction_lower_bound:
        if spectral_data.size * (1 - throughput_correction_upper_bound) < 1:
            warn(
                f"data size ({spectral_data.size}) is low compared to the throughput correction upper bound "
                f"({throughput_correction_upper_bound}), recommended value is {1 - 1 / spectral_data.size}",
                UserWarning
             )
    else:
        raise ValueError(f"Throughput correction upper bound ({throughput_correction_upper_bound}) must be"
                         f"greater or equal to throughput correction lower bound ({throughput_correction_lower_bound})")

    # Look at where the brightest pixels are, in order to avoid telluric lines
    time_averaged_data = np.ma.median(spectral_data, axis=0)  # median of the data over time/integrations
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
    brightest_data_wavelength = np.ma.median(spectral_data[:, brightest_pixels[0]], axis=1)

    spectral_data_corrected = np.ma.zeros(spectral_data.shape)
    spectral_data_corrected.mask = copy.copy(spectral_data.mask)

    for i, correction_coefficient in enumerate(brightest_data_wavelength):
        spectral_data_corrected[i, :] = spectral_data[i, :] / correction_coefficient
        reduction_matrix[i, :] /= correction_coefficient

    pipeline_noise = np.ma.zeros(spectral_data_corrected.shape)
    pipeline_noise.mask = copy.copy(spectral_data.mask)

    if noise is not None:
        for i, correction_coefficient in enumerate(brightest_data_wavelength):
            brightest_data_wavelength_noise = median_uncertainties(
                noise[i, brightest_pixels[0]][~spectral_data[i, brightest_pixels[0]].mask]
            )

            partial_derivatives = np.array([
                spectral_data_corrected[i, :] / spectral_data[i, :],  # dS'/dS
                - spectral_data_corrected[i, :] / correction_coefficient  # dS'/dC
            ])

            uncertainties = np.abs(np.array([
                noise[i, :],  # sigma_S
                brightest_data_wavelength_noise * np.ones(noise[i, :].shape)  # sigma_C
            ]))

            for j in range(uncertainties.shape[1]):
                pipeline_noise[i, j] = calculate_uncertainty(partial_derivatives[:, j], uncertainties[:, j])
                pipeline_noise.mask[i, j] = spectral_data.mask[i, j]

    return spectral_data_corrected, reduction_matrix, pipeline_noise


def remove_throughput(spectral_data, reduction_matrix, data_noise=None,
                      throughput_correction_lower_bound=0.70, throughput_correction_upper_bound=None, median=False):
    """Correct for the variable throughput.

    Args:
        spectral_data: spectral data to correct
        reduction_matrix: matrix storing all the operations made to reduce the data
        data_noise: noise of the data
        throughput_correction_lower_bound: [0-1] quantile lower bound on throughput correction
        throughput_correction_upper_bound: [0-1] quantile upper bound on throughput correction

    Returns:
        Spectral data corrected from throughput
    """
    if isinstance(spectral_data, np.ma.core.MaskedArray):
        spectral_data_corrected = np.ma.zeros(spectral_data.shape)
        spectral_data_corrected.mask = copy.copy(spectral_data.mask)
        pipeline_noise = np.ma.zeros(spectral_data.shape)
        pipeline_noise.mask = copy.copy(spectral_data.mask)
    else:
        spectral_data_corrected = np.zeros(spectral_data.shape)
        pipeline_noise = np.zeros(spectral_data.shape)

    if data_noise is None:
        data_noise = np.array([None])

    if median:
        print('Median!!')
        for i, data in enumerate(spectral_data):
            if isinstance(spectral_data, np.ma.core.MaskedArray):
                correction_coefficient = np.ma.median(data, axis=1)
            elif isinstance(spectral_data, np.ndarray):
                correction_coefficient = np.median(data, axis=1)
            else:
                raise ValueError(f"spectral_data must be a numpy.ndarray or a numpy.ma.core.MaskedArray, "
                                 f"but is of type '{type(spectral_data)}'")

            spectral_data_corrected[i, :, :] = np.transpose(
                np.transpose(data) / correction_coefficient
            )
            reduction_matrix[i, :, :] = np.transpose(
                np.transpose(reduction_matrix[i, :, :]) / correction_coefficient
            )

            if data_noise is not None:
                for j, correction_coefficient_ in enumerate(correction_coefficient):
                    brightest_data_wavelength_noise = median_uncertainties(
                        data_noise[i, j, :][~data[j, :].mask]
                    )

                    partial_derivatives = np.array([
                        spectral_data_corrected[i, j, :] / data[j, :],  # dS'/dS
                        - spectral_data_corrected[i, j, :] / correction_coefficient_  # dS'/dC
                    ])

                    uncertainties = np.abs(np.array([
                        data_noise[i, j, :],  # sigma_S
                        brightest_data_wavelength_noise * np.ones(data_noise[i, j, :].shape)  # sigma_C
                    ]))

                    for k in range(uncertainties.shape[1]):
                        pipeline_noise[i, j, k] = calculate_uncertainty(partial_derivatives[:, k], uncertainties[:, k])
                        pipeline_noise.mask[i, j, k] = data.mask[j, k]
            # pipeline_noise = copy.copy(data_noise)  # TODO add true pipeline noise
    else:
        print('Not median!!')
        for i, data in enumerate(spectral_data):
            if isinstance(spectral_data, np.ma.core.MaskedArray):
                spectral_data_corrected[i, :, :], reduction_matrix[i, :, :], pipeline_noise[i, :, :] = \
                    __remove_throughput_masked(
                        data, reduction_matrix[i, :, :], data_noise[i],
                        throughput_correction_lower_bound, throughput_correction_upper_bound
                    )
            elif isinstance(spectral_data, np.ndarray):
                spectral_data_corrected[i, :, :], reduction_matrix[i, :, :], pipeline_noise[i, :, :] = \
                    __remove_throughput(
                        data, reduction_matrix[i, :, :], data_noise[i],
                        throughput_correction_lower_bound, throughput_correction_upper_bound
                    )
            else:
                raise ValueError(f"spectral_data must be a numpy.ndarray or a numpy.ma.core.MaskedArray, "
                                 f"but is of type '{type(spectral_data)}'")

            # spectral_data_corrected[i, :, :] = np.transpose(
            #     np.transpose(spectral_data_corrected[i, :, :])
            #     / np.mean(spectral_data_corrected[i, :, :], axis=1)
            # )
            # print(np.mean(spectral_data_corrected[i, :, :], axis=1))
    return spectral_data_corrected, reduction_matrix, pipeline_noise


def remove_telluric_lines(spectral_data, reduction_matrix, airmass=None, times=None):
    """Correct for Earth's atmospheric absorptions.

    Args:
        spectral_data: spectral data to correct
        reduction_matrix: matrix storing all the operations made to reduce the data
        airmass: airmass of the data
        times: (s) time after first observation: t(i) = dit * (ndit(i) - 1)

    Returns:
        Spectral data corrected from the telluric transmittance
    """
    for i, data in enumerate(spectral_data):
        # Remove the mean of the telluric lines
        # TODO better mask management, like in remove_throughput
        if airmass is not None:
            exp_airmass = np.exp(-airmass[i])

            for k in range(np.size(data, axis=1)):
                # Fit the telluric lines change over time with a 2nd order polynomial
                # The telluric lines opacity depends on airmass (tau = alpha / cos(theta)), so the telluric lines
                # transmittance (T = exp(-tau)) depend on exp(airmass)
                # Using a 1st order polynomial is not enough, as the atm. composition will change slowly over time
                fit_parameters = np.polyfit(x=exp_airmass, y=data[:, k], deg=2)
                fit_function = np.poly1d(fit_parameters)
                fit = fit_function(exp_airmass)  # might be necessary to mask 0 here
                data[:, k] = data[:, k] / fit
                reduction_matrix[i, :, k] /= fit
        else:
            for k in range(np.size(data, axis=1)):
                if np.all(data.mask[:, k]):
                    continue

                fit_parameters = np.polyfit(x=times[~data.mask[:, k]], y=data[:, k][~data.mask[:, k]], deg=2)
                fit_function = np.poly1d(fit_parameters)
                fit = fit_function(times)
                data[:, k][~data.mask[:, k]] = data[:, k][~data.mask[:, k]] / fit
                reduction_matrix[i, :, k][~data.mask[:, k]] /= fit

    return spectral_data, reduction_matrix


def remove_telluric_lines_old(spectral_data, airmass=None, remove_outliers=True, remove_standard_deviation=False):
    """Correct for Earth's atmospheric absorptions.

    Args:
        spectral_data: spectral data to correct
        airmass: airmass of the data
        remove_outliers: if True, remove the pixels that are 3 times away from the median of the standard deviation of
            the data over time.
        remove_standard_deviation: if True, remove the standard deviation on time of the data (not recommended)

    Returns:
        Spectral data corrected from the telluric transmittance
    """
    for j, data in enumerate(spectral_data):
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

        spectral_data[j, :, :] = data

    return spectral_data


def remove_noisy_wavelength_channels(spectral_data, reduction_matrix, mean_subtract=False):
    for i, data in enumerate(spectral_data):
        # Get standard deviation over time, for each wavelength channel
        time_standard_deviation = np.asarray([np.std(data, axis=0)] * np.size(data, axis=0))

        # Mask channels where the standard deviation is greater than the total standard deviation
        data = np.ma.masked_where(
            time_standard_deviation > 3 * np.std(data), data
        )

        spectral_data[i, :, :] = data

    if mean_subtract:
        mean_spectra = np.mean(spectral_data, axis=2)  # mean over wavelengths of each individual spectrum
        spectral_data -= mean_spectra
        reduction_matrix -= mean_spectra

    return spectral_data, reduction_matrix


def simple_pipeline(spectral_data, airmass=None, times=None, data_noise=None,
                    throughput_correction_lower_bound=0.70, throughput_correction_upper_bound=None,
                    mean_subtract=False, median=False):
    """Removes the telluric lines and variable throughput of some data.

    Args:
        spectral_data: spectral data to correct
        airmass: airmass of the data
        times: (s) time after first observation: t(i) = dit * (ndit(i) - 1)
        data_noise: noise of the data
        throughput_correction_lower_bound: [0-1] quantile lower bound on throughput correction
        throughput_correction_upper_bound: [0-1] quantile upper bound on throughput correction
        mean_subtract: if True, the data corresponding to each spectrum are mean subtracted

    Returns:
        Spectral data corrected from variable throughput
    """
    reduction_matrix = np.ones(spectral_data.shape)

    spectral_data_corrected, reduction_matrix, pipeline_noise = remove_throughput(
        spectral_data=spectral_data,
        reduction_matrix=reduction_matrix,
        data_noise=data_noise,
        throughput_correction_lower_bound=throughput_correction_lower_bound,
        throughput_correction_upper_bound=throughput_correction_upper_bound,
        median=median
    )

    # for i in range(spectral_data_corrected.shape[0]):  # thomi5 tests
    #     print('mean-subtraction!')
    #     print('m', np.ma.mean(spectral_data_corrected[i], axis=1))
    #     print('md', np.ma.median(spectral_data_corrected[i], axis=1))
    #     spectral_data_corrected[i] = np.transpose(
    #         np.transpose(spectral_data_corrected[i]) - np.ma.median(spectral_data_corrected[i], axis=1)
    #     )
    #     #reduction_matrix[i] -= np.mean(spectral_data_corrected[i], axis=1)

    return spectral_data_corrected, reduction_matrix, pipeline_noise

    spectral_data_corrected, reduction_matrix = remove_telluric_lines(
        spectral_data=spectral_data_corrected,
        reduction_matrix=reduction_matrix,  # TODO separate the 2 reduction matrices?
        airmass=airmass,
        times=times
    )

    return spectral_data_corrected, reduction_matrix, pipeline_noise

    spectral_data_corrected, reduction_matrix = remove_noisy_wavelength_channels(
        spectral_data_corrected, reduction_matrix, mean_subtract
    )

    return spectral_data_corrected, reduction_matrix
