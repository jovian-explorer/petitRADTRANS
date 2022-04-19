"""Useful functions for spectrum models."""
import copy

from scipy.ndimage import gaussian_filter1d
import numpy as np
import petitRADTRANS.nat_cst as nc
from petitRADTRANS.radtrans import Radtrans
from petitRADTRANS.physics import doppler_shift
from petitRADTRANS.fort_rebin import fort_rebin as fr


def convolve(input_wavelengths, input_spectrum, new_resolving_power):
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


def convolve_rebin(input_wavelengths, input_spectrum, output_wavelengths, new_resolving_power):
    """Convolve and rebin a spectrum to a new wavelength array.
    The spectrum is convolved using a Gaussian filter with a standard deviation ~ R_old / R_new input wavelengths bins.

    Args:
        input_wavelengths: (cm) wavelengths of the input spectrum
        input_spectrum: input spectrum
        output_wavelengths: (cm) wavelengths of the output spectrum
        new_resolving_power: resolving power of output spectrum

    Returns:
        output_spectrum: the convolved and re-binned spectrum at the new resolving power
    """
    convolved_spectrum = convolve(input_wavelengths, input_spectrum, new_resolving_power)
    output_spectrum = fr.rebin_spectrum(input_wavelengths, convolved_spectrum, output_wavelengths)

    return output_spectrum


def convolve_shift_rebin(input_wavelengths, input_spectrum,
                         new_resolving_power, output_wavelengths, relative_velocities=None):
    """Convolve and Doppler-shift a spectrum, then rebin to another wavelength grid.
    The spectrum is convolved using a Gaussian filter with a standard deviation ~ R_old / R_new input wavelengths bins.

    A shifted wavelength grid is calculated from the Doppler shifted input wavelengths and the relative velocities. The
    wavelengths grid is of shape (N_relative_velocities, N_input_wavelengths). A negative relative velocity denotes that
    the target is moving toward the observer. A positive relative velocity denotes that the target is moving away from
    the observer.

    The convolved spectrum is re-binned to the output wavelengths.

    Args:
        input_wavelengths: (cm) wavelengths of the input spectrum
        input_spectrum: input spectrum
        new_resolving_power: resolving power of output spectrum
        output_wavelengths: (cm) wavelengths of the output spectrum
        relative_velocities: (cm.s-1) array containing the relative velocities between the object and the observer

    Returns:
        output_spectra: a 2D-array of shape (N_relative_velocities, N_output_wavelengths) containing the convolved,
        Doppler shifted and re-binned to output_wavelengths spectra for each relative velocity.
    """
    if relative_velocities is None:
        relative_velocities = np.zeros(1)

    convolved_spectrum = convolve(input_wavelengths, input_spectrum, new_resolving_power)

    output_spectra = np.zeros((np.size(relative_velocities), np.size(output_wavelengths)))

    for i, planet_velocity in enumerate(relative_velocities):
        wavelength_shift = doppler_shift(input_wavelengths, planet_velocity)
        output_spectra[i, :] = fr.rebin_spectrum(wavelength_shift, convolved_spectrum, output_wavelengths)

    return output_spectra


def calculate_spectral_radiosity_spectrum(atmosphere: Radtrans, temperatures, mass_mixing_ratios,
                                          planet_surface_gravity, mean_molar_mass, star_effective_temperature,
                                          star_radius, semi_major_axis, cloud_pressure):
    """Wrapper of Radtrans.calc_flux that output wavelengths in um and spectral radiosity in erg.s-1.cm-2.sr-1/cm.
    # TODO move to Radtrans
    Args:
        atmosphere:
        temperatures:
        mass_mixing_ratios:
        planet_surface_gravity:
        mean_molar_mass:
        star_effective_temperature:
        star_radius:
        semi_major_axis:
        cloud_pressure:

    Returns:

    """
    # Calculate the spectrum
    # TODO units in native calc_flux units for more performances?
    atmosphere.calc_flux(
        temp=temperatures,
        abunds=mass_mixing_ratios,
        gravity=planet_surface_gravity,
        mmw=mean_molar_mass,
        Tstar=star_effective_temperature,
        Rstar=star_radius,
        semimajoraxis=semi_major_axis,
        Pcloud=cloud_pressure,
        # stellar_intensity=parameters['star_spectral_radiosity'].value
    )

    # Transform the outputs into the units of our data.
    spectral_radiosity = radiosity_erg_hz2radiosity_erg_cm(atmosphere.flux, atmosphere.freq)
    wavelengths = nc.c / atmosphere.freq * 1e4  # cm to um

    return wavelengths, spectral_radiosity


def calculate_transit_spectrum(atmosphere: Radtrans, temperatures, mass_mixing_ratios, planet_surface_gravity,
                               mean_molar_mass, reference_pressure, planet_radius):
    """Wrapper of Radtrans.calc_transm that output wavelengths in um and transit radius in cm.  # TODO move to Radtrans

    Args:
        atmosphere:
        temperatures:
        mass_mixing_ratios:
        planet_surface_gravity:
        mean_molar_mass:
        reference_pressure:
        planet_radius:

    Returns:

    """
    # Calculate the spectrum
    atmosphere.calc_transm(
        temp=temperatures,
        abunds=mass_mixing_ratios,
        gravity=planet_surface_gravity,
        mmw=mean_molar_mass,
        P0_bar=reference_pressure,
        R_pl=planet_radius
    )

    # Convert into more useful units
    planet_transit_radius = copy.copy(atmosphere.transm_rad)
    wavelengths = nc.c / atmosphere.freq * 1e4  # Hz to um

    return wavelengths, planet_transit_radius


def load_snr_file(file, wavelengths_instrument_boundaries=None, mask_lower=None):
    snr_file_data = np.loadtxt(file)
    wavelengths_instrument = snr_file_data[:, 0]

    # Restrain to wavelength bounds
    if wavelengths_instrument_boundaries is not None:
        wh = np.where(np.logical_and(
            wavelengths_instrument > wavelengths_instrument_boundaries[0],
            wavelengths_instrument < wavelengths_instrument_boundaries[-1]
        ))[0]

        instrument_snr = np.ma.masked_invalid(snr_file_data[wh, 1] / snr_file_data[wh, 2])

        wavelengths_instrument = wavelengths_instrument[wh]
    else:
        instrument_snr = np.ma.masked_invalid(snr_file_data[:, 1] / snr_file_data[:, 2])

    if mask_lower is not None:
        instrument_snr = np.ma.masked_less_equal(instrument_snr, mask_lower)

    return wavelengths_instrument, instrument_snr


def radiosity_erg_hz2radiosity_erg_cm(radiosity_erg_hz, frequency):
    """Convert a radiosity from erg.s-1.cm-2.sr-1/Hz to erg.s-1.cm-2.sr-1/cm at a given frequency.  # TODO move to physics

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


def scale_secondary_eclipse_spectrum(planet_spectral_radiosity, star_spectral_radiosity, planet_radius, star_radius):
    return 1 + (planet_spectral_radiosity * planet_radius ** 2) / (star_spectral_radiosity * star_radius ** 2)


def scale_transit_spectrum(transit_radius, star_radius):
    return 1 - (transit_radius / star_radius) ** 2
