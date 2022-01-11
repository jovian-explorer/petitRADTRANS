"""
Useful functions for data reduction.
"""
from warnings import warn

import numpy as np


def __remove_throughput(spectral_data,
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

    return spectral_data_corrected


def __remove_throughput_masked(spectral_data,
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

    brightest_data_wavelength = np.ma.median(spectral_data[:, brightest_pixels[0]], axis=1)

    spectral_data_corrected = np.zeros(spectral_data.shape)

    for i, correction_coefficient in enumerate(brightest_data_wavelength):
        spectral_data_corrected[i, :] = spectral_data[i, :] / correction_coefficient

    return spectral_data_corrected


def remove_throughput(spectral_data,
                      throughput_correction_lower_bound=0.70, throughput_correction_upper_bound=None):
    """Correct for the variable throughput.

    Args:
        spectral_data: spectral data to correct
        throughput_correction_lower_bound: [0-1] quantile lower bound on throughput correction
        throughput_correction_upper_bound: [0-1] quantile upper bound on throughput correction

    Returns:
        Spectral data corrected from throughput
    """
    spectral_data_corrected = np.zeros(spectral_data.shape)

    for i, data in enumerate(spectral_data):
        print(i)

        if isinstance(spectral_data, np.ndarray):
            spectral_data_corrected[i, :, :] = \
                __remove_throughput(
                    data, throughput_correction_lower_bound, throughput_correction_upper_bound
                )
        elif isinstance(spectral_data, np.ma.core.MaskedArray):
            spectral_data_corrected[i, :, :] = \
                __remove_throughput_masked(
                    data, throughput_correction_lower_bound, throughput_correction_upper_bound
                )
        else:
            raise ValueError(f"spectral_data must be a numpy.ndarray or a numpy.ma.core.MaskedArray, "
                             f"but is of type '{type(spectral_data)}'")

    return spectral_data_corrected


def remove_telluric_lines(spectral_data, airmass=None, remove_standard_deviation=False):
    """Correct for Earth's atmospheric absorptions.

    Args:
        spectral_data: spectral data to correct
        airmass: airmass of the data
        remove_standard_deviation: if True, remove the standard deviation on time of the data (not recommended)

    Returns:
        Spectral data corrected from the telluric transmittance
    """
    for j, data in enumerate(spectral_data):
        print(j)
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

        standard_deviation_integration = np.asarray([np.std(data, axis=0)] * np.size(data, axis=0))

        # Remove telluric lines standard deviation
        if remove_standard_deviation:
            # Not recommended
            data /= standard_deviation_integration
        else:
            # TODO this might work when adding telluric transmittance
            #spectral_data = np.ma.masked_where(np.abs(spectral_data) > 3 * np.ma.std(spectral_data), spectral_data)
            # TODO this gives results, but is probably too restrictive, this is a very important step for log_l calc!
            data = np.ma.masked_where(
                np.abs(data) > 3 * np.ma.median(standard_deviation_integration), data)

        spectral_data[j, :, :] = data

    return spectral_data


def simple_pipeline(spectral_data, airmass=None,
                    throughput_correction_lower_bound=0.70, throughput_correction_upper_bound=None,
                    remove_standard_deviation=False):
    """Removes the telluric lines and variable throughput of some data.

    Args:
        spectral_data: spectral data to correct
        airmass: airmass of the data
        throughput_correction_lower_bound: [0-1] quantile lower bound on throughput correction
        throughput_correction_upper_bound: [0-1] quantile upper bound on throughput correction
        remove_standard_deviation: if True, remove the standard deviation on time of the data (not recommended)

    Returns:
        Spectral data corrected from variable throughput
    """
    spectral_data_corrected = remove_throughput(
        spectral_data, throughput_correction_lower_bound, throughput_correction_upper_bound
    )

    spectral_data_corrected = remove_telluric_lines(spectral_data_corrected, airmass, remove_standard_deviation)

    return spectral_data_corrected
