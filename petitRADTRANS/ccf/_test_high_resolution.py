import os

import numpy as np
from scipy.interpolate import interp1d

import petitRADTRANS.nat_cst as nc
from petitRADTRANS.ccf.ccf_utils import radiosity_erg_hz2radiosity_erg_cm
from petitRADTRANS.ccf.mock_observation import convolve_shift_rebin, generate_mock_observations, get_orbital_phases
from petitRADTRANS.ccf.model_containers import Planet
from petitRADTRANS.ccf.model_containers import SpectralModel
from petitRADTRANS.ccf.pipeline import remove_throughput, simple_pipeline
from petitRADTRANS.fort_rebin import fort_rebin as fr
from petitRADTRANS.phoenix import get_PHOENIX_spec
from petitRADTRANS.physics import guillot_global, doppler_shift
from petitRADTRANS.radtrans import Radtrans
from petitRADTRANS.retrieval import RetrievalConfig
from petitRADTRANS.retrieval.util import calc_MMW, uniform_prior

module_dir = os.path.abspath(os.path.dirname(__file__))


class Param:
    def __init__(self, value):
        self.value = value


def init_models(planet, w_bords, p0=1e-2):
    print('Initialization...')
    line_species_str = 'H2O_main_iso'  # 'H2O_Exomol'

    pressures = np.logspace(-6, 2, 100)
    temperature = guillot_global(
        pressure=pressures,
        kappa_ir=0.01,
        gamma=0.4,
        grav=planet.surface_gravity,
        t_int=200,
        t_equ=planet.equilibrium_temperature
    )
    gravity = planet.surface_gravity
    radius = planet.radius
    star_radius = planet.star_radius
    star_effective_temperature = planet.star_effective_temperature
    p_cloud = 1e2
    line_species = [line_species_str]
    rayleigh_species = ['H2', 'He']
    continuum_species = ['H2-H2', 'H2-He']

    mass_fractions = {
        'H2': 0.74,
        'He': 0.24,
        line_species_str: 1e-3
    }

    m_sum = 0.0  # Check that the total mass fraction of all species is <1

    for species in line_species:
        m_sum += mass_fractions[species]

    mass_fractions['H2'] = 0.74 * (1.0 - m_sum)
    mass_fractions['He'] = 0.24 * (1.0 - m_sum)

    for key in mass_fractions:
        mass_fractions[key] *= np.ones_like(pressures)

    mean_molar_mass = calc_MMW(mass_fractions)

    print('Setting up models...')
    atmosphere_grey_cloud = Radtrans(
        line_species=[line_species_str],
        rayleigh_species=['H2', 'He'],
        continuum_opacities=['H2-H2', 'H2-He'],
        wlen_bords_micron=w_bords,
        mode='lbl',
        lbl_opacity_sampling=1,
        do_scat_emis=True
    )
    atmosphere_grey_cloud.setup_opa_structure(pressures)

    models = {
        'grey_cloud': atmosphere_grey_cloud
    }

    return pressures, temperature, gravity, radius, star_radius, star_effective_temperature, p0, p_cloud, \
        mean_molar_mass, mass_fractions, \
        line_species, rayleigh_species, continuum_species, \
        line_species_str, models


def init_run(prt_object, pressures, parameters, rayleigh_species, continuum_species,
             ret_model, wavelength_instrument, observed_spectra, observations_uncertainties):
    run_definition_simple = RetrievalConfig(
        retrieval_name="test",
        run_mode="retrieval",
        AMR=False,
        pressures=pressures,
        scattering=False  # scattering is automatically included for transmission spectra
    )

    retrieved_parameters = [
        'planet_max_radial_orbital_velocity',
        'planet_rest_frame_shift'
    ]

    # Fixed parameters
    for p in parameters:
        if p not in retrieved_parameters:
            run_definition_simple.add_parameter(
                p,
                False,
                value=parameters[p].value
            )

    # Retrieved parameters
    # Prior functions
    def prior_kp(x):
        return uniform_prior(
            cube=x,
            x1=0.5 * parameters['planet_max_radial_orbital_velocity'].value,
            x2=1.5 * parameters['planet_max_radial_orbital_velocity'].value,
        )

    def prior_vr(x):
        return uniform_prior(
            cube=x,
            x1=-100,
            x2=100
        )

    # Add parameters
    run_definition_simple.add_parameter(
        retrieved_parameters[0],
        True,
        transform_prior_cube_coordinate=prior_kp
    )

    run_definition_simple.add_parameter(
        retrieved_parameters[1],
        True,
        transform_prior_cube_coordinate=prior_vr
    )

    # Spectrum parameters
    # Fixed
    run_definition_simple.set_rayleigh_species(rayleigh_species)
    run_definition_simple.set_continuum_opacities(continuum_species)

    # Retrieved
    # run_definition_simple.set_line_species(
    #     line_species,
    #     eq=False,
    #     abund_lim=(
    #         -6,  # min = abund_lim[0]
    #         6  # max = min + abund_lim[1]
    #     )
    # )

    # Load data
    run_definition_simple.add_data(
        name='test',
        path=None,
        model_generating_function=ret_model,
        opacity_mode='lbl',
        pRT_object=prt_object,
        wlen=wavelength_instrument,
        flux=observed_spectra,
        flux_error=observations_uncertainties
    )

    return run_definition_simple


def simple_ccf(wavelength_data, spectral_data_earth_corrected, wavelength_model, spectral_radiosity,
               lsf_fwhm, pixels_per_resolution_element, radial_velocity, kp, error):
    n_detectors, n_integrations, n_spectral_pixels = np.shape(spectral_data_earth_corrected)

    # Calculate star_radial_velocity interval, add extra coefficient just to be sure
    # Effectively, we are moving along the spectral pixels
    radial_velocity_lag_min = (np.min(radial_velocity) - kp)
    radial_velocity_lag_max = (np.max(radial_velocity) + kp)
    radial_velocity_interval = radial_velocity_lag_max - radial_velocity_lag_min
    radial_velocity_lag_min -= 0.25 * radial_velocity_interval
    radial_velocity_lag_max += 0.25 * radial_velocity_interval

    radial_velocity_lag = np.arange(
        radial_velocity_lag_min, radial_velocity_lag_max, lsf_fwhm / pixels_per_resolution_element
    )

    ccf = np.zeros((n_detectors, n_integrations, np.size(radial_velocity_lag)))

    # Shift the wavelengths by
    wavelength_shift = np.zeros((np.size(radial_velocity_lag), np.size(wavelength_model)))
    eclipse_depth_shift = np.zeros((n_detectors, np.size(radial_velocity_lag), n_spectral_pixels))

    for j in range(np.size(radial_velocity_lag)):
        wavelength_shift[j, :] = wavelength_model \
                                 * np.sqrt((1 + radial_velocity_lag[j] / nc.c) / (1 - radial_velocity_lag[j] / nc.c))

    for i in range(n_detectors):
        for k in range(np.size(radial_velocity_lag)):
            eclipse_depth_shift[i, k, :] = \
                fr.rebin_spectrum(wavelength_shift[k, :], spectral_radiosity, wavelength_data[i, :])

    # this is faster than correlate, because we are looking only at the velocity interval we are interested into
    def xcorr(data, model, length):
        # Initialise identity matrix for fast computation
        identity = np.ones(length)
        # The data (data) is already mean-subtracted but the model (model) is not
        model -= (model @ identity) / length  # np.mean is faster for very large arrays
        # Compute variances of model and data (will skip dividing by length because it
        # will cancels out when computing the cross-covariance)
        sf2 = (data @ data)  # np.sum(fvec ** 2) is faster
        sg2 = (model @ model)  # np.sum(gvec ** 2) is faster
        r = (data @ model)  # Data-model cross-covariance, np.sum(fvec * gvec) is faster

        return r / np.sqrt(sf2 * sg2), np.sqrt(sf2), np.sqrt(sg2)  # Data-model cross-correlation

    for i in range(n_detectors):
        for k in range(len(radial_velocity_lag)):
            for j in range(n_integrations):
                ccf[i, j, k], sf, sg = xcorr(
                    spectral_data_earth_corrected[i, j, :], eclipse_depth_shift[i, k, :], n_spectral_pixels
                )

    return ccf


def radiosity_model(prt_object, parameters):
    # Make the P-T profile
    pressures = prt_object.press * 1e-6  # bar to cgs
    temperatures = guillot_global(
        pressure=pressures,
        kappa_ir=0.01,
        gamma=0.4,
        grav=10 ** parameters['log_g'].value,
        t_int=200,
        t_equ=parameters['Temperature'].value
    )

    # Make the abundance profiles
    abundances = {}
    m_sum = 0.0  # Check that the total mass fraction of all species is <1

    for species in prt_object.line_species:
        spec = species.split('_R_')[0]  # deal with the naming scheme for binned down opacities (see below)
        abundances[species] = 10 ** parameters[spec].value * np.ones_like(pressures)
        m_sum += 10 ** parameters[spec].value

    abundances['H2'] = 0.74 * (1.0 - m_sum) * np.ones_like(pressures)
    abundances['He'] = 0.24 * (1.0 - m_sum) * np.ones_like(pressures)

    # Find the mean molecular weight in each layer
    mmw = calc_MMW(abundances)

    # Calculate the spectrum
    prt_object.calc_flux(
        temperatures,
        abundances,
        10 ** parameters['log_g'].value,
        mmw,
        Tstar=parameters['star_effective_temperature'].value,
        Rstar=parameters['Rstar'].value / nc.r_sun,
        semimajoraxis=parameters['semi_major_axis'].value / nc.AU,
        Pcloud=10 ** parameters['log_Pcloud'].value
    )

    # Transform the outputs into the units of our data.
    planet_radiosity = radiosity_erg_hz2radiosity_erg_cm(prt_object.flux, prt_object.freq)
    wlen_model = nc.c / prt_object.freq * 1e4  # wlen in micron

    return wlen_model, planet_radiosity


def retrieval_model_eclipse_grey_cloud(prt_object, parameters, pt_plot_mode=None, AMR=False):
    wlen_model, planet_radiosity = radiosity_model(prt_object, parameters)

    # TODO make these steps as a function common with generate_mock_observations
    planet_velocities = Planet.calculate_planet_radial_velocity(
        parameters['planet_max_radial_orbital_velocity'].value,
        parameters['planet_orbital_inclination'].value,
        parameters['orbital_phases'].value
    )

    planet_radiosity = convolve_shift_rebin(
        wlen_model,
        planet_radiosity,
        parameters['instrument_resolving_power'].value,
        parameters['wavelength_instrument'].value,
        planet_velocities + parameters['planet_rest_frame_shift'].value  # planet + system velocity
    )

    star_spectral_radiosity = convolve_shift_rebin(
        parameters['star_wavelength'].value,
        parameters['star_spectral_radiosity'].value,
        parameters['instrument_resolving_power'].value,
        parameters['wavelength_instrument'].value,
        parameters['planet_rest_frame_shift'].value  # only system velocity
    )

    spectrum_model = 1 + (planet_radiosity * parameters['R_pl'].value ** 2) \
        / (star_spectral_radiosity * parameters['Rstar'].value ** 2)

    # TODO add telluric transmittance (probably)
    # TODO add throughput variations? (maybe take raw data spectrum and multiply by mean(max(flux), axis=time))

    spectrum_model = simple_pipeline(spectrum_model)

    return wlen_model, spectrum_model


def get_radial_velocity_lag(radial_velocity, kp, lsf_fwhm, pixels_per_resolution_element, extra_factor=0.25):
    # Calculate radial velocity lag interval, add extra coefficient just to be sure
    # Effectively, we are moving along the spectral pixels
    radial_velocity_lag_min = (np.min(radial_velocity) - kp)
    radial_velocity_lag_max = (np.max(radial_velocity) + kp)
    radial_velocity_interval = radial_velocity_lag_max - radial_velocity_lag_min

    # Add a bit more to the interval, just to be sure
    radial_velocity_lag_min -= extra_factor * radial_velocity_interval
    radial_velocity_lag_max += extra_factor * radial_velocity_interval

    # Ensure that a lag of 0 km.s-1 is within the lag array, in order to avoid inaccuracies
    # Set interval bounds as multiple of lag step
    lag_step = lsf_fwhm / pixels_per_resolution_element

    radial_velocity_lag_min = np.floor(radial_velocity_lag_min / lag_step) * lag_step
    radial_velocity_lag_max = np.ceil(radial_velocity_lag_max / lag_step) * lag_step

    radial_velocity_lag = np.arange(
        radial_velocity_lag_min,
        radial_velocity_lag_max + lag_step,  # include radial_velocity_lag_max in array
        lag_step
    )

    return radial_velocity_lag


def simple_co_added_ccf(
        ccf, orbital_phases, radial_velocity, kp, planet_orbital_inclination, lsf_fwhm, pixels_per_resolution_element,
        extra_factor=0.25
):
    radial_velocity_lag = get_radial_velocity_lag(
        radial_velocity, kp, lsf_fwhm, pixels_per_resolution_element, extra_factor
    )

    radial_velocity_interval = np.min((np.abs(radial_velocity_lag[0]), np.abs(radial_velocity_lag[-1]))) * 0.5

    v_rest = np.arange(
        0.0, radial_velocity_interval, lsf_fwhm / pixels_per_resolution_element
    )
    v_rest = np.concatenate((-v_rest[:0:-1], v_rest))

    ccf_size = np.size(v_rest)

    kps = np.linspace(
        kp * (1 - 0.3), kp * (1 + 0.3), ccf_size
    )

    # Defining matrix containing the co-added CCFs
    ccf_tot = np.zeros((ccf_size, ccf_size))

    for ikp in range(ccf_size):
        rv_pl = radial_velocity + Planet.calculate_planet_radial_velocity(
            kps[ikp], planet_orbital_inclination, orbital_phases
        )

        for j in range(np.size(radial_velocity)):
            out_rv = v_rest + rv_pl[j]
            ccf_tot[ikp, :] += fr.rebin_spectrum(radial_velocity_lag, ccf[j, :], out_rv)

    return ccf_tot, v_rest, kps


def simple_co_added_ccf_old(
        ccf, orbital_phases, radial_velocity, kp, planet_orbital_inclination, lsf_fwhm, pixels_per_resolution_element
):
    # Calculate star_radial_velocity interval, add extra coefficient just to be sure
    # Effectively, we are moving along the spectral pixels
    radial_velocity_lag_min = (np.min(radial_velocity) - kp)
    radial_velocity_lag_max = (np.max(radial_velocity) + kp)
    radial_velocity_interval = radial_velocity_lag_max - radial_velocity_lag_min
    radial_velocity_lag_min -= 0.25 * radial_velocity_interval
    radial_velocity_lag_max += 0.25 * radial_velocity_interval

    radial_velocity_lag = np.arange(
        radial_velocity_lag_min, radial_velocity_lag_max, lsf_fwhm / pixels_per_resolution_element
    )

    radial_velocity_interval = np.min((np.abs(radial_velocity_lag_min * 0.5), np.abs(radial_velocity_lag_max * 0.5)))

    v_rest = np.arange(
        0.0, radial_velocity_interval, lsf_fwhm / pixels_per_resolution_element
    )
    v_rest = np.concatenate((-v_rest[:0:-1], v_rest))

    ccf_size = np.size(v_rest)

    kps = np.linspace(
        kp * (1 - 0.3), kp * (1 + 0.3), ccf_size
    )

    # Defining matrix containing the co-added CCFs
    ccf_tot = np.zeros((ccf_size, ccf_size))

    for ikp in range(ccf_size):
        rv_pl = radial_velocity + Planet.calculate_planet_radial_velocity(
            kps[ikp], planet_orbital_inclination, orbital_phases
        )

        for j in range(np.size(radial_velocity)):
            out_rv = v_rest + rv_pl[j]
            ccf_tot[ikp, :] += fr.rebin_spectrum(radial_velocity_lag, ccf[j, :], out_rv)

    return ccf_tot, v_rest, kps


def simple_log_l(wavelength_data, spectral_data_earth_corrected, wavelength_model, spectral_radiosity,
                 star_spectral_radiosity, parameters,
                 lsf_fwhm, pixels_per_resolution_element, instrument_resolving_power, radial_velocity, kp, error,
                 extra_factor=0.25):
    n_detectors, n_integrations, n_spectral_pixels = spectral_data_earth_corrected.shape

    radial_velocity_lag = get_radial_velocity_lag(
        radial_velocity, kp, lsf_fwhm, pixels_per_resolution_element, extra_factor
    )

    ccf_ = np.zeros((n_detectors, n_integrations, np.size(radial_velocity_lag)))
    log_l__ = np.zeros((n_detectors, n_integrations, np.size(radial_velocity_lag)))
    log_l__2 = np.zeros((n_detectors, n_integrations, np.size(radial_velocity_lag)))
    sf = np.zeros((n_detectors, n_integrations, np.size(radial_velocity_lag)))
    sg = np.zeros((n_detectors, n_integrations, np.size(radial_velocity_lag)))

    # Shift the wavelengths by
    wavelength_shift = np.zeros((np.size(radial_velocity_lag), np.size(wavelength_model)))
    star_radiosity = np.zeros((n_detectors, n_integrations, n_spectral_pixels))
    eclipse_depth_shift = np.zeros((
        n_detectors, n_integrations, np.size(radial_velocity_lag), n_spectral_pixels
    ))

    for j in range(np.size(radial_velocity_lag)):
        wavelength_shift[j, :] = wavelength_model \
                                 * np.sqrt((1 + radial_velocity_lag[j] / nc.c) / (1 - radial_velocity_lag[j] / nc.c))

    for i in range(n_detectors):
        star_radiosity[i, :, :] = convolve_shift_rebin(
            wavelength_model,
            star_spectral_radiosity,
            instrument_resolving_power,
            wavelength_data[i, :],
            radial_velocity  # only system velocity
        )

    for i in range(n_detectors):
        eclipse_depth_shift[i, :, :, :] = convolve_shift_rebin(
            wavelength_model,
            spectral_radiosity,
            instrument_resolving_power,
            wavelength_data[i, :],
            radial_velocity_lag
        )

    for i in range(n_detectors):
        for k in range(np.size(radial_velocity_lag)):
            eclipse_depth_shift[i, :, k, :] = 1 + (eclipse_depth_shift[i, :, k, :] * parameters['R_pl'].value ** 2) \
                                              / (star_radiosity * parameters['Rstar'].value ** 2)

    for k in range(np.size(radial_velocity_lag)):
        eclipse_depth_shift[:, :, k, :] = remove_throughput(eclipse_depth_shift[:, :, k, :])

    eclipse_depth_shift = np.transpose(
        np.transpose(eclipse_depth_shift) / np.transpose(np.mean(eclipse_depth_shift, axis=3))
    )

    # this is faster than correlate, because we are looking only at the velocity interval we are interested into
    def log_l_(model, data, uncertainties, alpha=1.0, beta=1.0):
        model -= model.mean()
        model = alpha * model
        uncertainties = beta * uncertainties
        chi2 = data - model
        chi2 /= uncertainties
        chi2 *= chi2
        chi2 = chi2.sum()

        return - data.size * np.log(beta) - 0.5 * chi2

    def xcorr2(data, model):
        # The data (data) is already mean-subtracted but the model (model) is not
        model -= np.mean(model)  # np.mean is faster for very large arrays
        # Compute variances of model and data (will skip dividing by N because it
        # will cancels out when computing the cross-covariance)
        sf2 = np.sum(data ** 2)  # np.sum(fvec ** 2) is faster
        sg2 = np.sum(model ** 2)  # np.sum(gvec ** 2) is faster
        r = np.sum(data * model)  # Data-model cross-covariance, np.sum(fvec * gvec) is faster

        return r / np.sqrt(sf2 * sg2), np.sqrt(sf2), np.sqrt(sg2)  # Data-model cross-correlation

    def xcorr2_ma(data, model):
        # The data (data) is already mean-subtracted but the model (model) is not
        model -= np.ma.mean(model)  # np.mean is faster for very large arrays
        # Compute variances of model and data (will skip dividing by N because it
        # will cancels out when computing the cross-covariance)
        sf2 = np.ma.sum(data ** 2)  # np.sum(fvec ** 2) is faster
        sg2 = np.ma.sum(model ** 2)  # np.sum(gvec ** 2) is faster
        r = np.ma.sum(data * model)  # Data-model cross-covariance, np.sum(fvec * gvec) is faster

        return r / np.ma.sqrt(sf2 * sg2), np.sqrt(sf2), np.sqrt(sg2)  # Data-model cross-correlation

    def xcorr(data, model, length):
        # Initialise identity matrix for fast computation
        identity = np.ones(length)
        # The data (data) is already mean-subtracted but the model (model) is not
        model -= (model @ identity) / length  # np.mean is faster for very large arrays
        # Compute variances of model and data (will skip dividing by length because it
        # will cancels out when computing the cross-covariance)
        sf2 = (data @ data)  # np.sum(fvec ** 2) is faster
        sg2 = (model @ model)  # np.sum(gvec ** 2) is faster
        r = (data @ model)  # Data-model cross-covariance, np.sum(fvec * gvec) is faster

        return r / np.sqrt(sf2 * sg2), np.sqrt(sf2), np.sqrt(sg2)  # Data-model cross-correlation

    def xcorr_(data, model):
        # Initialise identity matrix for fast computation
        identity = np.ones(model.size)
        # The data (data) is already mean-subtracted but the model (model) is not
        model -= (model @ identity) / model.size  # np.mean is faster for very large arrays
        # Compute variances of model and data (will skip dividing by length because it
        # will cancels out when computing the cross-covariance)
        sf2 = (data @ data)  # np.sum(fvec ** 2) is faster
        sg2 = (model @ model)  # np.sum(gvec ** 2) is faster
        r = (data @ model)  # Data-model cross-covariance, np.sum(fvec * gvec) is faster

        return r / np.sqrt(sf2 * sg2), np.sqrt(sf2), np.sqrt(sg2)  # Data-model cross-correlation

    def xcorrma(data, model, length):
        # Initialise identity matrix for fast computation
        identity = np.ones(length)
        # The data (data) is already mean-subtracted but the model (model) is not
        model -= (model @ identity) / length  # np.mean is faster for very large arrays
        # Compute variances of model and data (will skip dividing by length because it
        # will cancels out when computing the cross-covariance)
        sf2 = np.ma.sum(data ** 2)  # np.sum(fvec ** 2) is faster
        sg2 = (model @ model)  # np.sum(gvec ** 2) is faster
        r = np.ma.sum(data * model)  # Data-model cross-covariance, np.sum(fvec * gvec) is faster

        return r / np.sqrt(sf2 * sg2), np.sqrt(sf2), np.sqrt(sg2)  # Data-model cross-correlation

    def log_l_b():
        return - 0.5 * n_spectral_pixels \
                    * (
                        np.log(sf[i, j, k] * sg[i, j, k])
                        + np.log(sf[i, j, k] / sg[i, j, k] + sg[i, j, k] / sf[i, j, k] - 2 * ccf_[i, j, k])
                    )

    # Keep only the non-masked values in order to gain time
    # Using lists instead of arrays because spectra won't necessarily be of the same size
    data_ = []
    error_ = []

    for i in range(n_detectors):
        data_.append([])
        error_.append([])

        for j in range(n_integrations):
            data_[i].append(np.array(
                    spectral_data_earth_corrected[i, j, ~spectral_data_earth_corrected.mask[i, j, :]]
            ))
            error_[i].append(np.array(error[~spectral_data_earth_corrected.mask[i, j, :]]))

    for i in range(n_detectors):
        for j in range(n_integrations):
            for k in range(len(radial_velocity_lag)):
                # Convert masked array into array to gain time
                model_ = eclipse_depth_shift[i, j, k, ~spectral_data_earth_corrected.mask[i, j, :]]

                log_l__2[i, j, k] = log_l_(
                    model_, data_[i][j], error_[i][j]
                )
                ccf_[i, j, k], sf[i, j, k], sg[i, j, k] = xcorr_(
                    data_[i][j], model_
                )
                log_l__[i, j, k] = log_l_b()

    return ccf_, log_l__, sf, sg, log_l__2


def true_model(prt_object, parameters):
    wlen_model, planet_radiosity = radiosity_model(prt_object, parameters)

    star_radiosity = fr.rebin_spectrum(
        parameters['star_wavelength'].value,
        parameters['star_spectral_radiosity'].value,
        wlen_model
    )

    spectrum_model = 1 + (planet_radiosity * parameters['R_pl'].value ** 2) \
        / (star_radiosity * parameters['Rstar'].value ** 2)

    spectrum_model = remove_throughput(np.asarray([spectrum_model]))[0]
    spectrum_model /= np.mean(spectrum_model)

    return wlen_model, spectrum_model - 1


def main(apply_noise=True):
    planet_name = 'HD 209458 b'
    planet = Planet.get(planet_name)

    band = 'M'

    star_name = planet.host_name.replace(' ', '_')

    wavelengths_borders = {
        'L': [2.85, 4.20],
        'M': [4.7, 4.8]  # [4.5, 5.5],
    }

    integration_times_ref = {
        'L': 20.83,
        'M': 76.89
    }

    # Load noise
    data = np.loadtxt(os.path.join(module_dir, 'metis', 'SimMETIS', star_name,
                                   f"{star_name}_SNR_{band}-band_calibrated.txt"))
    wavelength_instrument = data[:, 0]

    wh = np.where(np.logical_and(
        wavelength_instrument > wavelengths_borders[band][0],
        wavelength_instrument < wavelengths_borders[band][1]
    ))[0]

    wavelength_instrument = wavelength_instrument[wh]
    instrument_resolving_power = 1e5

    # Number of DITs during the transit, we assume that we had the same number of DITs for the star alone
    ndit_half = int(np.ceil(planet.transit_duration / integration_times_ref[band]))  # actual NDIT is twice this value

    instrument_snr = np.ma.masked_invalid(data[wh, 1] / data[wh, 2])
    instrument_snr = np.ma.masked_less_equal(instrument_snr, 1.0)

    phase_start = 0.507  # just after secondary eclipse
    orbital_phases = get_orbital_phases(phase_start, planet.orbital_period, integration_times_ref[band], ndit_half)
    airmass = None
    telluric_transmittance = None  # TODO get telluric transmittance

    variable_throughput = -(np.linspace(-1, 1, np.size(orbital_phases)) - 0.1) ** 2
    variable_throughput += 0.5 - np.min(variable_throughput)

    # Get models
    kp = planet.calculate_orbital_velocity(planet.star_mass, planet.orbit_semi_major_axis)

    model_wavelengths_border = {
        band: [
            doppler_shift(wavelength_instrument[0], -2 * kp),
            doppler_shift(wavelength_instrument[-1], 2 * kp)
        ]
    }

    star_data = get_PHOENIX_spec(planet.star_effective_temperature)
    star_data[:, 1] = SpectralModel.radiosity_erg_hz2radiosity_erg_cm(
        star_data[:, 1], nc.c / star_data[:, 0]
    )

    star_data[:, 0] *= 1e4  # cm to um

    # Initialization
    pressures, temperature, gravity, radius, star_radius, star_effective_temperature, \
        p0, p_cloud, mean_molar_mass, mass_fractions, \
        line_species, rayleigh_species, continuum_species, \
        line_species_str, models = init_models(planet, model_wavelengths_border[band])

    n_live_points = 1000

    # Initialize true parameters
    true_parameters = {
        'R_pl': Param(radius),
        'Temperature': Param(planet.equilibrium_temperature),
        'log_Pcloud': Param(np.log10(p_cloud)),
        'log_g': Param(np.log10(gravity)),
        'reference_pressure': Param(p0),
        'H2O_main_iso': Param(mass_fractions['H2O_main_iso']),
        'star_effective_temperature': Param(star_effective_temperature),
        'Rstar': Param(star_radius),
        'star_wavelength': Param(star_data[:, 0]),
        'star_spectral_radiosity': Param(star_data[:, 1]),
        'semi_major_axis': Param(planet.orbit_semi_major_axis),
        'planet_max_radial_orbital_velocity': Param(kp),
        'planet_rest_frame_shift': Param(np.zeros_like(orbital_phases)),
        'planet_orbital_inclination': Param(planet.orbital_inclination),
        'orbital_phases': Param(orbital_phases),
        'instrument_resolving_power': Param(instrument_resolving_power),
        'wavelength_instrument': Param(wavelength_instrument)
    }

    for species in line_species:
        true_parameters[species] = Param(np.log10(mass_fractions[species]))

    for model_name in models:
        # Initialize strings
        if apply_noise:
            noise_str = ''
        else:
            noise_str = '_no_noise'

        observation_model_name = model_name
        force_observation_str = ''

        print('----\n', model_name, noise_str, force_observation_str)

        # Select which model to use
        if 'grey_cloud' in model_name:
            retrieval_model = retrieval_model_eclipse_grey_cloud
            observations_model = retrieval_model
        else:
            raise ValueError(f"model {model_name} is not recognized")

        # Generate and save mock observations
        print('True spectrum calculation...')
        true_wavelength, true_spectrum = radiosity_model(models[observation_model_name], true_parameters)

        star_radiosity = fr.rebin_spectrum(
            star_data[:, 0],
            star_data[:, 1],
            true_wavelength
        )

        mock_observations, noise = generate_mock_observations(
            true_wavelength, true_spectrum,
            telluric_transmittance=telluric_transmittance,
            variable_throughput=None,
            integration_time=integration_times_ref[band],
            integration_time_ref=integration_times_ref[band],
            wavelength_instrument=wavelength_instrument,
            instrument_snr=instrument_snr,
            instrument_resolving_power=instrument_resolving_power,
            planet_radius=true_parameters['R_pl'].value,
            star_radius=true_parameters['Rstar'].value,
            star_spectral_radiosity=star_radiosity,
            orbital_phases=orbital_phases,
            system_observer_radial_velocities=np.zeros(ndit_half),
            # TODO set to 0 for now since SNR data from Roy is at 0, but find RV source eventually
            planet_max_radial_orbital_velocity=true_parameters['planet_max_radial_orbital_velocity'].value,
            planet_orbital_inclination=planet.orbital_inclination,
            mode='eclipse',
            add_noise=True,
            number=1
        )

        reduced_mock_observations = simple_pipeline(mock_observations[0], airmass, remove_standard_deviation=False)

        ccf = simple_ccf(
            np.asarray([wavelength_instrument]),
            np.asarray([reduced_mock_observations]),
            true_wavelength,
            true_spectrum,
            lsf_fwhm=3e5,  # cm.s-1
            pixels_per_resolution_element=2,
            radial_velocity=np.zeros_like(ndit_half),
            kp=kp,
            error=1 / instrument_snr
        )

        ccf_tot = simple_co_added_ccf(
            ccf[0], orbital_phases, np.zeros(ndit_half), kp, true_parameters['planet_orbital_inclination'].value, 3e5, 2
        )

        i_peak = np.where(ccf_tot == np.max(ccf_tot))
        i_peak_s = (np.linspace(i_peak[0][0] - 3, i_peak[0][0] + 3, 7, dtype=int),
                    np.linspace(i_peak[1][0] - 3, i_peak[1][0] + 3, 7, dtype=int))
        mask = np.ones_like(ccf_tot, dtype=bool)
        mask[i_peak_s] = False
        noise = np.std(ccf_tot[mask])
        peak = np.max(ccf_tot)

        print(f'S/N: {peak / noise:.2f}')

        # Initialize run
        run_definitions = init_run(
            models[observation_model_name], pressures, true_parameters, rayleigh_species, continuum_species,
            retrieval_model,
            wavelength_instrument, reduced_mock_observations, reduced_mock_observations / instrument_snr
        )
        # TODO actual retrieval


if __name__ == '__main__':
    main()
