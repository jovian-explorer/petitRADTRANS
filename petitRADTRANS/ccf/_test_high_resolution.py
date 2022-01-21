"""
Run with:
    mpiexec -n N --use-hwthread-cpus python3 _test_high_resolution.py
N is the number of processes.
Try:
    sudo mpiexec -n N --allow-run-as-root ...
If for some reason the script crashes.
"""

import copy
import os
import time

import matplotlib.pyplot as plt
import numpy as np

import petitRADTRANS.nat_cst as nc
from petitRADTRANS.ccf.ccf_utils import radiosity_erg_hz2radiosity_erg_cm
from petitRADTRANS.ccf.mock_observation import convolve_shift_rebin, generate_mock_observations, get_orbital_phases, \
    get_mock_secondary_eclipse_spectra
from petitRADTRANS.ccf.model_containers import Planet
from petitRADTRANS.ccf.model_containers import SpectralModel
from petitRADTRANS.ccf.pipeline import remove_throughput, remove_telluric_lines, simple_pipeline
from petitRADTRANS.fort_rebin import fort_rebin as fr
from petitRADTRANS.phoenix import get_PHOENIX_spec
from petitRADTRANS.physics import guillot_global, doppler_shift
from petitRADTRANS.radtrans import Radtrans
from petitRADTRANS.retrieval import RetrievalConfig, Retrieval
from petitRADTRANS.retrieval.data import Data
from petitRADTRANS.retrieval.plotting import contour_corner
from petitRADTRANS.retrieval.util import calc_MMW, uniform_prior

module_dir = os.path.abspath(os.path.dirname(__file__))


class Param:
    def __init__(self, value):
        self.value = value


def co_added_retrieval(wavelength_instrument, reduced_mock_observations, true_wavelength, true_spectrum, star_radiosity,
                       true_parameters, radial_velocity, error, orbital_phases, plot=False, output_dir=None):
    ccfs, log_l_bs, _, _, log_ls = simple_log_l(
        np.asarray([wavelength_instrument] * reduced_mock_observations.shape[0]),
        np.ma.asarray(reduced_mock_observations),
        true_wavelength,
        true_spectrum,
        star_radiosity,
        true_parameters,
        lsf_fwhm=3e5,  # cm.s-1
        pixels_per_resolution_element=2,
        instrument_resolving_power=true_parameters['instrument_resolving_power'].value,
        radial_velocity=radial_velocity,
        kp=true_parameters['planet_max_radial_orbital_velocity'].value,
        error=error[0, 0, :]
    )

    log_l_tot, v_rest, kps = simple_co_added_ccf(
        log_ls, orbital_phases, radial_velocity, true_parameters['planet_max_radial_orbital_velocity'].value,
        true_parameters['planet_orbital_inclination'].value, 3e5, 2
    )

    i_peak = np.where(log_l_tot[0] == np.max(log_l_tot[0]))

    if plot:
        plt.figure()
        plt.imshow(log_l_tot[0], origin='lower', extent=[v_rest[0], v_rest[-1], kps[0], kps[-1]], aspect='auto')
        plt.plot([v_rest[0], v_rest[-1]], [kps[i_peak[0]], kps[i_peak[0]]], color='r')
        plt.vlines([v_rest[i_peak[1]]], ymin=[kps[0]], ymax=[kps[-1]], color='r')
        plt.title(f"Best Kp = {kps[i_peak[0]][0]:.3e} "
                  f"(true = {true_parameters['planet_max_radial_orbital_velocity'].value:.3e}), "
                  f"best V_rest = {v_rest[i_peak[1]][0]:.3e} "
                  f"(true = {np.mean(radial_velocity):.3e})")
        plt.xlabel('V_rest (cm.s-1)')
        plt.ylabel('K_p (cm.s-1)')
        plt.savefig(os.path.join(output_dir, 'co_added_log_l.png'))

    return log_l_tot, v_rest, kps, i_peak


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


def get_retrieval_model(prt_object, parameters, pt_plot_mode=None, AMR=False):
    wlen_model, planet_radiosity = radiosity_model(prt_object, parameters)

    # TODO make these steps as a function common with generate_mock_observations
    planet_velocities = Planet.calculate_planet_radial_velocity(
        parameters['planet_max_radial_orbital_velocity'].value,
        parameters['planet_orbital_inclination'].value,
        parameters['orbital_phases'].value
    )

    spectrum_model = get_mock_secondary_eclipse_spectra(
        wavelength_model=wlen_model,
        spectrum_model=planet_radiosity,
        star_spectral_radiosity=parameters['star_spectral_radiosity'].value,
        planet_radius=parameters['R_pl'].value,
        star_radius=parameters['Rstar'].value,
        wavelength_instrument=parameters['wavelength_instrument'].value,
        instrument_resolving_power=parameters['instrument_resolving_power'].value,
        planet_velocities=planet_velocities,
        system_observer_radial_velocities=parameters['system_observer_radial_velocities'].value,
        planet_rest_frame_shift=parameters['planet_rest_frame_shift'].value
    )

    # TODO add telluric transmittance (probably)
    # TODO add throughput variations? (maybe take raw data spectrum and multiply by mean(max(flux), axis=time))
    # TODO generation of multiple-detector models

    if parameters['apply_pipeline'].value:
        spectrum_model = simple_pipeline(np.array([spectrum_model]))
    else:
        spectrum_model = np.array([spectrum_model])
        spectrum_model -= 1

    return parameters['wavelength_instrument'].value, spectrum_model


def init_model(planet, w_bords, p0=1e-2):
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
    atmosphere = Radtrans(
        line_species=[line_species_str],
        rayleigh_species=['H2', 'He'],
        continuum_opacities=['H2-H2', 'H2-He'],
        wlen_bords_micron=w_bords,
        mode='lbl',
        lbl_opacity_sampling=1,
        do_scat_emis=True
    )
    atmosphere.setup_opa_structure(pressures)

    return pressures, temperature, gravity, radius, star_radius, star_effective_temperature, p0, p_cloud, \
        mean_molar_mass, mass_fractions, \
        line_species, rayleigh_species, continuum_species, \
        line_species_str, atmosphere


def init_parameters(planet, retrieval_name, n_live_points, add_noise, band, wavelengths_borders, integration_times_ref,
                    apply_pipeline=True):
    star_name = planet.host_name.replace(' ', '_')

    retrieval_name += f'_{n_live_points}lp'

    if not apply_pipeline:
        retrieval_name += '_np'

    if not add_noise:
        retrieval_name += '_nn'

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
    v_sys = np.zeros_like(orbital_phases)

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
        line_species_str, model = init_model(planet, model_wavelengths_border[band])

    retrieval_directory = os.path.abspath(os.path.join(module_dir, '..', '__tmp', 'test_retrieval', retrieval_name))

    if not os.path.isdir(retrieval_directory):
        os.mkdir(retrieval_directory)

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
        'semi_major_axis': Param(planet.orbit_semi_major_axis),
        'planet_max_radial_orbital_velocity': Param(kp),
        'system_observer_radial_velocities': Param(v_sys),
        'planet_rest_frame_shift': Param(0.0),
        'planet_orbital_inclination': Param(planet.orbital_inclination),
        'orbital_phases': Param(orbital_phases),
        'instrument_resolving_power': Param(instrument_resolving_power),
        'wavelength_instrument': Param(wavelength_instrument),
        'apply_pipeline': Param(apply_pipeline)
    }

    for species in line_species:
        true_parameters[species] = Param(np.log10(mass_fractions[species]))

    # Initialize strings
    model_name = 'test_model_H2O'

    print('----\n', model_name)

    # Select which model to use
    retrieval_model = get_retrieval_model

    # Generate and save mock observations
    print('True spectrum calculation...')
    true_wavelength, true_spectrum = radiosity_model(model, true_parameters)

    star_radiosity = fr.rebin_spectrum(
        star_data[:, 0],
        star_data[:, 1],
        true_wavelength
    )

    true_parameters['star_spectral_radiosity'] = Param(star_radiosity)

    print('Mock obs...')
    mock_observations, noise, mock_observations_without_noise = generate_mock_observations(
        true_wavelength, true_spectrum,
        telluric_transmittance=telluric_transmittance,
        variable_throughput=None,
        integration_time=integration_times_ref[band],
        integration_time_ref=integration_times_ref[band],
        wavelength_instrument=true_parameters['wavelength_instrument'].value,
        instrument_snr=instrument_snr,
        instrument_resolving_power=true_parameters['instrument_resolving_power'].value,
        planet_radius=true_parameters['R_pl'].value,
        star_radius=true_parameters['Rstar'].value,
        star_spectral_radiosity=true_parameters['star_spectral_radiosity'].value,
        orbital_phases=true_parameters['orbital_phases'].value,
        system_observer_radial_velocities=np.zeros(ndit_half),  # TODO put that in true_parameters
        # TODO set to 0 for now since SNR data from Roy is at 0, but find RV source eventually
        planet_max_radial_orbital_velocity=true_parameters['planet_max_radial_orbital_velocity'].value,
        planet_orbital_inclination=true_parameters['planet_orbital_inclination'].value,
        mode='eclipse',
        add_noise=add_noise,
        apply_snr_mask=True,
        number=1
    )

    if apply_pipeline:
        print('Data reduction...')
        reduced_mock_observations = simple_pipeline(
            mock_observations, airmass,
            remove_standard_deviation=False,
            remove_outliers=add_noise  # no need to remove outliers if no noise
        )

        if add_noise:
            reduced_mock_observations_without_noise = simple_pipeline(
                mock_observations_without_noise, airmass,
                remove_standard_deviation=False,
                remove_outliers=False  # no need to remove outliers if no noise
            )
        else:
            reduced_mock_observations_without_noise = copy.deepcopy(reduced_mock_observations)
    else:
        reduced_mock_observations = copy.deepcopy(mock_observations) - 1

        if add_noise:
            reduced_mock_observations_without_noise = copy.deepcopy(mock_observations_without_noise) - 1
        else:
            reduced_mock_observations_without_noise = copy.deepcopy(reduced_mock_observations)

    w, r = retrieval_model(model, true_parameters)

    if not np.all(w == wavelength_instrument):
        raise ValueError('wrong wvl')

    if not np.all(r == reduced_mock_observations_without_noise):
        raise ValueError('wrong model')

    print("Consistency check OK")

    if not apply_pipeline:
        for i, detector in enumerate(reduced_mock_observations):
            for j, data in enumerate(detector):
                reduced_mock_observations[i, j, :] -= data.mean()

    error = np.ones(reduced_mock_observations.shape) / instrument_snr

    print('Co-addition of log L...')
    log_l_tot, v_rest, kps, i_peak = co_added_retrieval(
        wavelength_instrument, reduced_mock_observations, true_wavelength,
        true_spectrum, star_radiosity, true_parameters, np.zeros(ndit_half), error, orbital_phases,
        plot=True, output_dir=retrieval_directory
    )

    print('Running pseudo-retrieval...')
    kps_pseudo_retrieval = np.linspace(kps[i_peak[0][0] - 5], kps[i_peak[0][0] + 5], 7)
    v_rest_pseudo_retrieval = np.linspace(v_rest[i_peak[1][0] - 5], v_rest[i_peak[1][0] + 5], 7)

    log_l_pseudo_retrieval, wvl_pseudo_retrieval, models_pseudo_retrieval = pseudo_retrieval(
        true_parameters, kps_pseudo_retrieval, v_rest_pseudo_retrieval,
        model, reduced_mock_observations, instrument_snr,
        true_parameters, np.zeros(ndit_half),
        plot=True, output_dir=retrieval_directory
    )

    save_all(retrieval_directory, mock_observations, error, reduced_mock_observations, log_l_tot, v_rest, kps,
             log_l_pseudo_retrieval,
             wvl_pseudo_retrieval, models_pseudo_retrieval, true_parameters, instrument_snr)

    return retrieval_name, retrieval_directory, \
        model, pressures, true_parameters, line_species, rayleigh_species, continuum_species, \
        retrieval_model, \
        wavelength_instrument, reduced_mock_observations, error


def init_run(retrieval_name, prt_object, pressures, parameters, line_species, rayleigh_species, continuum_species,
             retrieval_model, wavelength_instrument, observed_spectra, observations_uncertainties):
    run_definition_simple = RetrievalConfig(
        retrieval_name=retrieval_name,
        run_mode="retrieval",
        AMR=False,
        pressures=pressures,
        scattering=False  # scattering is automatically included for transmission spectra
    )

    # retrieved_parameters = []
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
            x1=0.75 * parameters['planet_max_radial_orbital_velocity'].value,
            x2=1.25 * parameters['planet_max_radial_orbital_velocity'].value,
        )

    def prior_vr(x):
        return uniform_prior(
            cube=x,
            x1=-1e7,
            x2=1e7
        )

    # # Add parameters
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
    run_definition_simple.set_line_species(
        line_species,
        eq=False,
        abund_lim=(
            -6,  # min = abund_lim[0]
            6  # max = min + abund_lim[1]
        )
    )

    # Remove masked values if necessary
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
    run_definition_simple.add_data(
        name='test',
        path=None,
        model_generating_function=retrieval_model,
        opacity_mode='lbl',
        pRT_object=prt_object,
        wlen=wavelength_instrument,
        flux=data_,
        flux_error=error_,
        mask=mask_
    )

    return run_definition_simple


def load_all(directory):
    print('Loading...')

    load_dict = np.load(os.path.join(directory, 'run_parameters.npz'))

    mock_observations = load_dict['mock_observations']
    noise = load_dict['noise']
    reduced_mock_observations = load_dict['reduced_mock_observations']
    log_l_tot = load_dict['log_l_tot']
    v_rest = load_dict['v_rest']
    kps = load_dict['kps']
    log_l_pseudo_retrieval = load_dict['log_l_pseudo_retrieval']
    wvl_pseudo_retrieval = load_dict['wvl_pseudo_retrieval']
    models_pseudo_retrieval = load_dict['mock_observations_mask']
    #true_parameters = load_dict['true_parameters']
    instrument_snr = load_dict['instrument_snr']

    mock_observations = np.ma.asarray(mock_observations)
    mock_observations.mask = load_dict['mock_observations_mask']

    reduced_mock_observations = np.ma.asarray(reduced_mock_observations)
    reduced_mock_observations.mask = load_dict['reduced_mock_observations_mask']

    instrument_snr = np.ma.asarray(instrument_snr)
    instrument_snr.mask = load_dict['instrument_snr_mask']

    return mock_observations, noise, \
        reduced_mock_observations, log_l_tot, v_rest, kps, log_l_pseudo_retrieval, \
        wvl_pseudo_retrieval, models_pseudo_retrieval, \
        instrument_snr


def main():
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    planet_name = 'HD 209458 b'
    planet = Planet.get(planet_name)

    retrieval_name = 't_kp_vr_H2O_79-80'
    n_live_points = 200
    add_noise = True
    apply_pipeline = True

    band = 'M'

    wavelengths_borders = {
        'L': [2.85, 4.20],
        'M': [4.79, 4.80]  # [4.5, 5.5],
    }

    integration_times_ref = {
        'L': 20.83,
        'M': 76.89
    }

    if rank == 0:
        # Initialize parameters
        retrieval_name, retrieval_directory, \
            model, pressures, true_parameters, line_species, rayleigh_species, continuum_species, \
            retrieval_model, \
            wavelength_instrument, reduced_mock_observations, error \
            = init_parameters(
                planet, retrieval_name, n_live_points, add_noise, band, wavelengths_borders, integration_times_ref,
                apply_pipeline=apply_pipeline
            )

        retrieval_parameters = {
            'retrieval_name': retrieval_name,
            'prt_object': model,
            'pressures': pressures,
            'parameters': true_parameters,
            'line_species': line_species,
            'rayleigh_species': rayleigh_species,
            'continuum_species': continuum_species,
            'retrieval_model': retrieval_model,
            'wavelength_instrument': wavelength_instrument,
            'observed_spectra': reduced_mock_observations,
            'observations_uncertainties': error
        }

        retrieval_directory = retrieval_directory
    else:
        print(f"Rank {rank} waiting for main process to finish...")
        retrieval_parameters = None
        retrieval_directory = ''

    retrieval_parameters = comm.bcast(retrieval_parameters, root=0)
    retrieval_directory = comm.bcast(retrieval_directory, root=0)

    # Initialize retrieval
    run_definitions = init_run(**retrieval_parameters)

    retrieval = Retrieval(
        run_definitions,
        output_dir=retrieval_directory,
        sample_spec=False,
        ultranest=False,
        pRT_plot_style=False
    )

    retrieval.run(
        sampling_efficiency=0.8,
        n_live_points=n_live_points,
        const_efficiency_mode=False,
        resume=False
    )

    if rank == 0:
        sample_dict, parameter_dict = retrieval.get_samples(
            output_dir=retrieval_directory + os.path.sep,
            ret_names=[retrieval_name]
        )

        n_param = len(parameter_dict[retrieval_name])
        parameter_plot_indices = {retrieval_name: np.arange(0, n_param)}

        true_values = {retrieval_name: []}

        for p in parameter_dict[retrieval_name]:
            true_values[retrieval_name].append(np.mean(retrieval_parameters['parameters'][p].value))

        fig = contour_corner(sample_dict, parameter_dict, os.path.join(retrieval_directory, 'test_corner.png'),
                             parameter_plot_indices=parameter_plot_indices,
                             true_values=true_values, prt_plot_style=False)

        fig.show()


def pseudo_retrieval(parameters, kps, v_rest, model, reduced_mock_observations, instrument_snr,
                     true_parameters=None, radial_velocity=None, plot=False, output_dir=None):
    ppp = copy.deepcopy(parameters)
    logls = []
    wavelengths = []
    retrieval_models = []
    error = 1 / instrument_snr

    if hasattr(reduced_mock_observations, 'mask'):
        print('Taking care of mask...')
        data_ = []
        error_ = []
        mask_ = copy.copy(reduced_mock_observations.mask)

        for i in range(reduced_mock_observations.shape[0]):
            data_.append([])
            error_.append([])

            for j in range(reduced_mock_observations.shape[1]):
                data_[i].append(np.array(
                        reduced_mock_observations[i, j, ~mask_[i, j, :]]
                ))
                error_[i].append(np.array(error[~mask_[i, j, :]]))

        data_ = np.asarray(data_, dtype=object)
        error_ = np.asarray(error_, dtype=object)
    else:
        data_ = reduced_mock_observations
        error_ = error
        mask_ = np.zeros(reduced_mock_observations.shape, dtype=bool)

    for lag in v_rest:
        ppp['planet_rest_frame_shift'].value = lag
        logls.append([])
        wavelengths.append([])
        retrieval_models.append([])

        for kp_ in kps:
            ppp['planet_max_radial_orbital_velocity'].value = kp_

            w, s = get_retrieval_model(model, ppp)
            wavelengths[-1].append(w)
            retrieval_models[-1].append(s)

            logl = 0

            for i, det in enumerate(data_):
                for j, data in enumerate(det):
                    logl += Data.log_likelihood_gibson(
                        s[i, j, ~mask_[i, j, :]],
                        data,
                        error_[i, j],
                        1.0, 1.0
                    )

            logls[-1].append(logl)

    logls = np.transpose(logls)

    i_peak = np.where(logls == np.max(logls))

    if plot:
        plt.figure()
        plt.imshow(logls, origin='lower', extent=[v_rest[0], v_rest[-1], kps[0], kps[-1]], aspect='auto')
        plt.plot([v_rest[0], v_rest[-1]], [kps[i_peak[0]], kps[i_peak[0]]], color='r')
        plt.vlines([v_rest[i_peak[1]]], ymin=[kps[0]], ymax=[kps[-1]], color='r')
        plt.title(f"Best Kp = {kps[i_peak[0]][0]:.3e} "
                  f"(true = {true_parameters['planet_max_radial_orbital_velocity'].value:.3e}), "
                  f"best V_rest = {v_rest[i_peak[1]][0]:.3e} "
                  f"(true = {np.mean(radial_velocity):.3e})")
        plt.xlabel('V_rest (cm.s-1)')
        plt.ylabel('K_p (cm.s-1)')
        plt.savefig(os.path.join(output_dir, 'pseudo_retrieval.png'))

    return logls, wavelengths, retrieval_models


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
        Pcloud=10 ** parameters['log_Pcloud'].value,
        #stellar_intensity=parameters['star_spectral_radiosity'].value
    )

    # Transform the outputs into the units of our data.
    planet_radiosity = radiosity_erg_hz2radiosity_erg_cm(prt_object.flux, prt_object.freq)
    wlen_model = nc.c / prt_object.freq * 1e4  # wlen in micron

    return wlen_model, planet_radiosity


def retrieval_run(retrieval_name, n_live_points, model, pressures, true_parameters,
                  line_species, rayleigh_species, continuum_species,
                  retrieval_model,
                  wavelength_instrument, reduced_mock_observations, error, plot=False, output_dir=None):
    # Initialize run
    run_definitions = init_run(
        retrieval_name,
        model, pressures, true_parameters, line_species, rayleigh_species, continuum_species,
        retrieval_model,
        wavelength_instrument, reduced_mock_observations, error
    )

    # Retrieval
    retrieval = Retrieval(
        run_definitions,
        output_dir=output_dir,
        sample_spec=False,
        ultranest=False,
        pRT_plot_style=False
    )

    retrieval.run(
        sampling_efficiency=0.8,
        n_live_points=n_live_points,
        const_efficiency_mode=False,
        resume=False
    )

    if plot:
        sample_dict, parameter_dict = retrieval.get_samples(
            output_dir=output_dir + os.path.sep,
            ret_names=[retrieval_name]
        )

        n_param = len(parameter_dict[retrieval_name])
        parameter_plot_indices = {retrieval_name: np.arange(0, n_param)}

        true_values = {retrieval_name: []}

        for p in parameter_dict[retrieval_name]:
            true_values[retrieval_name].append(np.mean(true_parameters[p].value))

        fig = contour_corner(sample_dict, parameter_dict, os.path.join(output_dir, 'test_corner.png'),
                             parameter_plot_indices=parameter_plot_indices,
                             true_values=true_values, prt_plot_style=False)

        fig.show()

    return retrieval


def save_all(directory, mock_observations, noise, reduced_mock_observations, log_l_tot, v_rest, kps,
             log_l_pseudo_retrieval,
             wvl_pseudo_retrieval, models_pseudo_retrieval, true_parameters, instrument_snr):
    print('Saving...')

    fname = os.path.join(directory, 'run_parameters.npz')

    np.savez_compressed(
        file=fname,
        mock_observations=mock_observations,
        mock_observations_mask=mock_observations.mask,
        noise=noise,
        reduced_mock_observations=reduced_mock_observations,
        reduced_mock_observations_mask=reduced_mock_observations.mask,
        log_l_tot=log_l_tot,
        v_rest=v_rest,
        kps=kps,
        log_l_pseudo_retrieval=log_l_pseudo_retrieval,
        wvl_pseudo_retrieval=wvl_pseudo_retrieval,
        models_pseudo_retrieval=models_pseudo_retrieval,
        instrument_snr=instrument_snr,
        instrument_snr_mask=instrument_snr.mask,
        # true_parameters=true_parameters
    )


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


def simple_co_added_ccf(
        evaluation, orbital_phases, radial_velocity, kp, planet_orbital_inclination, lsf_fwhm,
        pixels_per_resolution_element,
        extra_factor=0.25, n_kp=None
):
    radial_velocity_lag = get_radial_velocity_lag(
        radial_velocity, kp, lsf_fwhm, pixels_per_resolution_element, extra_factor
    )

    radial_velocity_interval = np.min((np.abs(radial_velocity_lag[0]), np.abs(radial_velocity_lag[-1]))) * 0.5

    v_rest = np.arange(
        0.0, radial_velocity_interval, lsf_fwhm / pixels_per_resolution_element
    )
    v_rest = np.concatenate((-v_rest[:0:-1], v_rest))

    ccf_size = v_rest.size

    if n_kp is None:
        n_kp = ccf_size

        kps = np.linspace(
            kp * (1 - 0.3), kp * (1 + 0.3), n_kp
        )
    elif n_kp == 1:
        kps = np.asarray([kp])
    else:
        kps = np.linspace(
            kp * (1 - 0.3), kp * (1 + 0.3), n_kp
        )

    # Defining matrix containing the co-added CCFs
    ccf_tot = np.zeros((evaluation.shape[0], ccf_size, ccf_size))

    for i in range(evaluation.shape[0]):
        for ikp in range(n_kp):
            rv_pl = radial_velocity + Planet.calculate_planet_radial_velocity(
                kps[ikp], planet_orbital_inclination, orbital_phases
            )

            for j in range(np.size(radial_velocity)):
                out_rv = v_rest + rv_pl[j]
                ccf_tot[i, ikp, :] += fr.rebin_spectrum(radial_velocity_lag, evaluation[i, j, :], out_rv)

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

    # Get star radiosity assuming the system-observer radial velocity is well-known (e.g. no need to iterate to find it)
    for i in range(n_detectors):
        star_radiosity[i, :, :] = convolve_shift_rebin(
            wavelength_model,
            star_spectral_radiosity,
            instrument_resolving_power,
            wavelength_data[i, :],
            radial_velocity  # only system velocity
        )

    # Shift the model spectrum along the spectral pixels of the detector
    for i in range(n_detectors):
        eclipse_depth_shift[i, :, :, :] = convolve_shift_rebin(
            wavelength_model,
            spectral_radiosity,
            instrument_resolving_power,
            wavelength_data[i, :],
            radial_velocity_lag
        )

    # Calculate eclipse depth for every shifted spectra, at all integrations
    # This prevents the contamination of the log l by the stellar spectrum
    for i in range(n_detectors):
        for k in range(np.size(radial_velocity_lag)):
            eclipse_depth_shift[i, :, k, :] = 1 + (eclipse_depth_shift[i, :, k, :] * parameters['R_pl'].value ** 2) \
                                              / (star_radiosity * parameters['Rstar'].value ** 2)

    # Remove throughput
    for k in range(np.size(radial_velocity_lag)):
        eclipse_depth_shift[:, :, k, :] = remove_throughput(eclipse_depth_shift[:, :, k, :])

    eclipse_depth_shift = np.transpose(
        np.transpose(eclipse_depth_shift) / np.transpose(np.mean(eclipse_depth_shift, axis=3))
    )

    # this is faster than correlate, because we are looking only at the velocity interval we are interested into
    def log_l_(model, data, uncertainties, alpha=1.0, beta=1.0):
        # The stripes along the time axis in log_l are caused by the slightly different average chi2 with time, it is
        # possible to remove these stripes by subtracting log_l with its mean along the lag axis.
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


if __name__ == '__main__':
    t0 = time.time()
    main()
    print(f"Done in {time.time() - t0} s.")
