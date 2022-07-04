"""
Useful functions for high-resolution retrievals.
"""
import copy
import os

import json
import numpy as np
from matplotlib import colors

import petitRADTRANS.nat_cst as nc
from petitRADTRANS.ccf.ccf_utils import radiosity_erg_hz2radiosity_erg_cm
from petitRADTRANS.ccf.mock_observation import add_telluric_lines, add_variable_throughput, \
    generate_mock_observations, get_mock_secondary_eclipse_spectra, get_mock_transit_spectra, get_orbital_phases
from petitRADTRANS.ccf.model_containers import Planet, SpectralModel2
from petitRADTRANS.ccf.pipeline import simple_pipeline, pipeline_validity_test
from petitRADTRANS.ccf.utils import calculate_reduced_chi2
from petitRADTRANS.fort_rebin import fort_rebin as fr
from petitRADTRANS.phoenix import get_PHOENIX_spec
from petitRADTRANS.physics import doppler_shift, guillot_global
from petitRADTRANS.radtrans import Radtrans
from petitRADTRANS.retrieval import RetrievalConfig
from petitRADTRANS.retrieval.util import calc_MMW, uniform_prior

all_species = [
    'CO_main_iso',
    'CO_36',
    'CH4_main_iso',
    'H2S_main_iso',
    'K',
    'NH3_main_iso',
    'Na_allard_new',
    'PH3_main_iso',
    'H2O_main_iso'
]


class Param:
    def __init__(self, value):
        self.value = value
       

# Private functions 
# TODO replace these private functions by a nice object doing everything needed
def _init_model(planet, w_bords, line_species_str, p0=1e-2):
    print('Initialization...')
    #line_species_str = ['H2O_main_iso']  # ['H2O_main_iso', 'CO_all_iso']  # 'H2O_Exomol'

    if not isinstance(planet.mass, float) or planet.mass == 0:
        print("Warning: planet mass undefined, using a made-up one")
        planet.mass = planet.radius ** 3 * 1.5
        planet.surface_gravity = Planet.mass2surface_gravity(planet.mass, planet.radius)[0]

    pressures = np.logspace(-10, 2, 100)
    temperature = np.ones(pressures.shape) * planet.equilibrium_temperature
    gravity = planet.surface_gravity
    radius = planet.radius
    star_radius = planet.star_radius
    star_effective_temperature = planet.star_effective_temperature
    p_cloud = 1e2
    line_species = line_species_str #['CO_main_iso', 'CO_36', 'CH4_main_iso', 'H2O_main_iso']
    rayleigh_species = ['H2', 'He']
    continuum_species = ['H2-H2', 'H2-He', 'H-']

    metallicity = SpectralModel2.calculate_scaled_metallicity(planet.mass, 10**planet.star_metallicity, 1.0)
    mass_fractions = SpectralModel2.calculate_mass_mixing_ratios(
        pressures=pressures,
        line_species=line_species,
        included_line_species='all',
        temperatures=temperature,
        co_ratio=0.55,
        log10_metallicity=np.log10(metallicity),
        carbon_pressure_quench=None,
        use_equilibrium_chemistry=True
    )
    # print('1', np.sum(list(mass_fractions.values()), axis=0))
    # print(list(mass_fractions.keys()))
    #
    # if 'CO_all_iso' not in line_species:
    #     co_mass_mixing_ratio = copy.copy(mass_fractions['CO'])
    #
    #     if 'CO_main_iso' in line_species:
    #         print('main1', np.mean(co_mass_mixing_ratio), co_mass_mixing_ratio - mass_fractions['CO'])
    #         print('main11', np.sum(list(mass_fractions.values()), axis=0))
    #         mass_fractions['CO'] = co_mass_mixing_ratio / (1 + c13c12_ratio)
    #         print('main12', np.sum(list(mass_fractions.values()), axis=0))
    #         mass_fractions['CO_36'] = co_mass_mixing_ratio - mass_fractions['CO']
    #         print('main2',np.mean(co_mass_mixing_ratio), co_mass_mixing_ratio - (mass_fractions['CO_36'] + mass_fractions['CO']))
    #         print('main21',np.sum(list(mass_fractions.values()), axis=0))
    #         print('main22',mass_fractions['CO_36'])
    #     elif 'CO_36' in line_species:
    #         print('36')
    #         mass_fractions['CO_36'] = co_mass_mixing_ratio / (1 + 1 / c13c12_ratio)
    #         mass_fractions['CO'] = \
    #             co_mass_mixing_ratio - mass_fractions['CO_36']
    # print('2', np.sum(list(mass_fractions.values()), axis=0))

    for species in line_species:
        if species == 'CO_36':
            pass

        spec = species.split('_', 1)[0]

        if spec in mass_fractions:
            if species not in mass_fractions and species != 'K':
                mass_fractions[species] = mass_fractions[spec]

            if species != 'K':
                del mass_fractions[spec]

    mean_molar_mass = calc_MMW(mass_fractions)

    print('Setting up models...')
    atmosphere = Radtrans(
        line_species=line_species,
        rayleigh_species=rayleigh_species,
        continuum_opacities=continuum_species,
        wlen_bords_micron=w_bords,
        mode='lbl',
        do_scat_emis=True,
        lbl_opacity_sampling=4
    )
    atmosphere.setup_opa_structure(pressures)

    return pressures, temperature, gravity, radius, star_radius, star_effective_temperature, p0, p_cloud, \
        mean_molar_mass, mass_fractions, metallicity, \
        line_species, rayleigh_species, continuum_species, \
        atmosphere


def _init_retrieval_model(prt_object, parameters):
    temperature = parameters['temperature'].value

    # Make the P-T profile
    pressures = prt_object.press * 1e-6  # bar to cgs
    temperatures = np.ones(pressures.shape) * temperature

    # Make the abundance profiles
    abundances = {}

    for species in prt_object.line_species:
        if species == 'CO_36':
            abundances[species] = 10 ** parameters[species].value * np.ones_like(pressures)
        else:
            spec = species.split('_R_')[0]  # deal with the naming scheme for binned down opacities (see below)
            spec = spec.split('_', 1)[0]
            abundances[spec] = 10 ** parameters[species].value * np.ones_like(pressures)

    abundances = SpectralModel2.calculate_mass_mixing_ratios(
        pressures=pressures,
        line_species=prt_object.line_species,
        included_line_species='all',
        temperatures=temperatures,
        co_ratio=0.55,
        log10_metallicity=np.log10(parameters['planet_metallicity'].value),
        carbon_pressure_quench=None,
        imposed_mass_mixing_ratios=abundances,
        use_equilibrium_chemistry=True
    )

    for species in prt_object.line_species:
        spec = species.split('_', 1)[0]

        if spec in abundances:
            if species not in abundances and species != 'K':
                abundances[species] = abundances[spec]

            if species != 'K':
                del abundances[spec]

    # heh2_ratio = abundances['He'] / abundances['H2']
    #
    # if np.any(m_sum) > 1:
    #     abundances['H2'] = np.zeros(pressures.shape)
    #     abundances['He'] = np.zeros(pressures.shape)
    #
    #     for i, s in enumerate(m_sum):
    #         if s > 1:
    #             abundances = {species: mmr / s for species, mmr in abundances.items()}
    #         else:
    #             abundances['H2'][i] = (1 - s) / (1 + heh2_ratio) * np.ones(pressures.shape)
    #             abundances['He'][i] = abundances['H2'][i] * heh2_ratio
    # else:
    #     abundances['H2'] = (1 - m_sum) / (1 + heh2_ratio) * np.ones(pressures.shape)
    #     abundances['He'] = abundances['H2'] * heh2_ratio

    # Find the mean molecular weight in each layer
    mmw = calc_MMW(abundances)

    return temperatures, abundances, mmw


def _get_deformation_matrix(telluric_transmittance, variable_throughput, shape):
    # TODO put as output of generate_mock_observations, and manage multiple detector case
    if telluric_transmittance is not None:
        if np.ndim(telluric_transmittance) == 1:
            telluric_matrix = telluric_transmittance * np.ones(shape)
        elif np.ndim(telluric_transmittance) == 2:
            telluric_matrix = telluric_transmittance
        else:
            raise ValueError('wrong number of dimensions for telluric matrix')
    else:
        telluric_matrix = np.ones(shape)

    if variable_throughput is not None:
        vt_matrix = add_variable_throughput(
            np.ones(shape), variable_throughput
        )
    else:
        vt_matrix = np.ones(shape)

    return np.ma.masked_array([telluric_matrix * vt_matrix])


# def _get_secondary_eclipse_retrieval_model(prt_object, parameters, pt_plot_mode=None, AMR=False, apply_pipeline=True):
#     wlen_model, planet_radiosity = _radiosity_model(prt_object, parameters)
#
#     planet_velocities = Planet.calculate_planet_radial_velocity(
#         parameters['planet_max_radial_orbital_velocity'].value,
#         parameters['planet_orbital_inclination'].value,
#         np.rad2deg(2 * np.pi * parameters['orbital_phases'].value)
#     )
#
#     spectrum_model = get_mock_secondary_eclipse_spectra(
#         wavelength_model=wlen_model,
#         spectrum_model=planet_radiosity,
#         star_spectral_radiosity=parameters['star_spectral_radiosity'].value,
#         planet_radius=parameters['planet_radius'].value,
#         star_radius=parameters['star_radius'].value,
#         wavelength_instrument=parameters['wavelengths_instrument'].value,
#         instrument_resolving_power=parameters['instrument_resolving_power'].value,
#         planet_velocities=planet_velocities,
#         system_observer_radial_velocities=parameters['system_observer_radial_velocities'].value,
#         planet_rest_frame_shift=parameters['planet_rest_frame_shift'].value
#     )
#
#     # TODO generation of multiple-detector models
#
#     # Add data mask to be as close as possible as the data when performing the pipeline
#     spectrum_model0 = np.ma.masked_array([spectrum_model])
#     spectrum_model0.mask = copy.copy(parameters['data'].value.mask)
#
#     if apply_pipeline:
#         spectrum_model = simple_pipeline(
#             spectral_data=spectrum_model0,
#             airmass=parameters['airmass'].value,
#             data_uncertainties=parameters['data_uncertainties'].value
#         )
#     else:
#         spectrum_model = spectrum_model0
#
#     return parameters['wavelengths_instrument'].value, spectrum_model


def _get_transit_retrieval_model(prt_object, parameters, pt_plot_mode=None, AMR=False, apply_pipeline=True):
    if 'star_radius' not in parameters:
        sr = parameters['Rstar'].value
    else:
        sr = parameters['star_radius'].value

    wlen_model, transit_radius = _transit_radius_model(prt_object, parameters)

    planet_velocities = Planet.calculate_planet_radial_velocity(
        parameters['planet_max_radial_orbital_velocity'].value,
        parameters['planet_orbital_inclination'].value,
        np.rad2deg(2 * np.pi * parameters['orbital_phases'].value)
    )

    spectrum_model = np.zeros((
        parameters['wavelengths_instrument'].value.shape[0],
        planet_velocities.size,
        parameters['wavelengths_instrument'].value.shape[1]
    ))

    for i, wavelengths_detector in enumerate(parameters['wavelengths_instrument'].value):
        spectrum_model[i, :, :] = get_mock_transit_spectra(
            wavelength_model=wlen_model,
            transit_radius_model=transit_radius,
            star_radius=sr,
            wavelength_instrument=wavelengths_detector,
            instrument_resolving_power=parameters['instrument_resolving_power'].value,
            planet_velocities=planet_velocities,
            system_observer_radial_velocities=parameters['system_observer_radial_velocities'].value,
            planet_rest_frame_shift=parameters['planet_rest_frame_shift'].value
        )

    spectrum_model = np.moveaxis(spectrum_model, 0, 1)
    spectrum_model = np.reshape(
        spectrum_model,
        (planet_velocities.size, parameters['wavelengths_instrument'].value.size)
    )
    # TODO generation of multiple-detector model
    # Add data mask to be as close as possible as the data when performing the pipeline
    spectrum_model0 = np.ma.masked_array([spectrum_model])
    spectrum_model0.mask = copy.copy(parameters['data_mask'].value)

    if apply_pipeline:
        spectrum_model = simple_pipeline(
            spectrum=spectrum_model0,
            airmass=parameters['airmass'].value,
            uncertainties=parameters['data_uncertainties'].value,
            apply_throughput_removal=True,
            apply_telluric_lines_removal=True
        )
    else:
        spectrum_model = spectrum_model0

    return parameters['wavelengths_instrument'].value, spectrum_model


def _pseudo_retrieval(parameters, kps, v_rest, model, reduced_mock_observations, error, mode='eclipse'):
    from petitRADTRANS.retrieval.data import Data

    ppp = copy.deepcopy(parameters)
    logls = []
    wavelengths = []
    retrieval_models = []

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
                error_[i].append(np.array(error[i, j, ~mask_[i, j, :]]))

        data_ = np.asarray(data_, dtype=object)
        error_ = np.asarray(error_, dtype=object)
    else:
        data_ = reduced_mock_observations
        error_ = error
        mask_ = np.zeros(reduced_mock_observations.shape, dtype=bool)

    if mode == 'eclipse':
        raise NotImplementedError(f"eclipse mode not yet implemented")  # TODO
        # retrieval_model = _get_secondary_eclipse_retrieval_model
    elif mode == 'transit':
        retrieval_model = _get_transit_retrieval_model
    else:
        raise ValueError(f"mode must be 'eclipse' or 'transit', but is '{mode}'")

    for lag in v_rest:
        ppp['planet_rest_frame_shift'].value = lag
        logls.append([])
        wavelengths.append([])
        retrieval_models.append([])

        for kp_ in kps:
            ppp['planet_max_radial_orbital_velocity'].value = kp_

            w, s = retrieval_model(model, ppp)
            wavelengths[-1].append(w)
            retrieval_models[-1].append(s)

            logl = 0

            for i, det in enumerate(data_):
                for j, data in enumerate(det):
                    logl += Data.log_likelihood_gibson(
                        model=s[i, j, ~mask_[i, j, :]],
                        data=data,
                        uncertainties=error_[i, j],
                        alpha=1.0,
                        beta=1.0
                    )

            logls[-1].append(logl)

    logls = np.transpose(logls)

    return logls, wavelengths, retrieval_models


def _transit_radius_model(prt_object, parameters):
    if 'log10_surface_gravity' not in parameters:
        surface_gravity = 10 ** parameters['log_g'].value
    else:
        surface_gravity = 10 ** parameters['log10_surface_gravity'].value

    if 'planet_radius' not in parameters:
        pr = parameters['R_pl'].value
    else:
        pr = parameters['planet_radius'].value

    temperatures, abundances, mmw = _init_retrieval_model(prt_object, parameters)

    # Calculate the spectrum
    prt_object.calc_transm(
        temp=temperatures,
        abunds=abundances,
        gravity=surface_gravity,
        mmw=mmw,
        P0_bar=parameters['reference_pressure'].value,
        R_pl=pr
    )

    # Transform the outputs into the units of our data.
    planet_transit_radius = prt_object.transm_rad
    wlen_model = nc.c / prt_object.freq * 1e4  # wlen in micron

    return wlen_model, planet_transit_radius


def get_retrieval_name(planet, mode, wavelength_min, wavelength_max, retrieval_species_names, n_live_points,
                       exposure_time):
    return f"{planet.name.lower().replace(' ', '_')}_" \
           f"{mode}_{exposure_time:.3e}s_{wavelength_min:.3f}-{wavelength_max:.3f}um_" \
           f"{'_'.join(retrieval_species_names)}_{n_live_points}lp"


def load_airmassorg_data(file):
    with open(file, 'r') as f:
        jd_times = []
        altitudes = []

        for line in f:
            line = line.strip()
            cols = line.split('\t')

            jd_times.append(float(cols[3]))

            altitude = cols[4]
            altitude = altitude[:4]
            altitudes.append(float(altitude))

    jd_times = np.array(jd_times)
    times = (jd_times - jd_times[0]) * nc.snc.day

    altitudes = np.array(altitudes)
    airmasses = 1 / np.cos(np.deg2rad(90 - altitudes))

    return times, airmasses


# Useful functions
def init_mock_observations(planet, line_species_str, mode,
                           retrieval_directory, retrieval_name,
                           add_noise, wavelengths_borders, integration_times_ref, n_transits=1.0,
                           wavelengths_instrument=None, instrument_snr=None, snr_file=None,
                           telluric_transmittance=None, airmass=None, variable_throughput=None,
                           instrument_resolving_power=1e5,
                           load_from=None, plot=False):
    print('1')
    # Load SNR file
    if snr_file is not None:
        print("Loading SNR...")
        if snr_file.rsplit('.', 1)[-1] == 'npz':
            snr_file_data_ = np.load(snr_file)
            snr_file_data = np.zeros((snr_file_data_['wavelengths'][()].size, 2))
            snr_file_data[:, 0] = snr_file_data_['wavelengths'][()]
            snr_file_data[:, 1] = snr_file_data_['single_pixel_snr'][()]
        else:
            snr_file_data = np.loadtxt(snr_file)

        if wavelengths_instrument is None:
            wavelengths_instrument = snr_file_data[:, 0]

            file_resolving_power = np.mean(wavelengths_instrument[1:] / np.diff(wavelengths_instrument))
            sampling = int(np.round(file_resolving_power / (instrument_resolving_power * 2)))  # 2 pixels per res elem
            wavelengths_instrument = wavelengths_instrument[sampling::sampling]
        else:
            sampling = 1

        instrument_snr_ = snr_file_data[sampling::sampling, 1]
    else:
        snr_file_data = None
        instrument_snr_ = None

    # Restrain to wavelength bounds
    wh = np.where(np.logical_and(
        wavelengths_instrument > wavelengths_borders[0],
        wavelengths_instrument < wavelengths_borders[1]
    ))[0]

    if snr_file_data is not None and instrument_snr is None:
        instrument_snr = np.ma.masked_invalid(instrument_snr_[wh])
    else:
        instrument_snr = instrument_snr[wh]

    # Number of DITs during the transit, we assume that we had the same number of DITs for the star alone
    ndit_half = int(np.ceil(planet.transit_duration / integration_times_ref))  # actual NDIT is twice this value
    # ndit_half = 1  # for bin-by-bin

    if ndit_half > 1:
        print(f"Matching SNR to number of NDIT (SNR is assumed to be given for 2 times the transit duration)")  # for pre and post transit
        instrument_snr /= np.sqrt(ndit_half * 2)

    if n_transits != 1.0:
        print(f"Adjusting SNR for {n_transits} transits (SNR is assumed to be given for 1 transit)")
        instrument_snr *= np.sqrt(n_transits)

    wavelengths_instrument = wavelengths_instrument[wh]
    instrument_snr = np.ma.masked_less_equal(instrument_snr, 1.0)
    wavelengths_instrument = np.array([wavelengths_instrument])
    instrument_snr = np.array([instrument_snr])

    data_shape = (1, ndit_half, wavelengths_instrument.size)

    # Get orbital phases
    if mode == 'eclipse':
        phase_start = 0.507  # just after secondary eclipse
        orbital_phases = \
            get_orbital_phases(phase_start, planet.orbital_period, integration_times_ref, ndit_half)
    elif mode == 'transit':
        orbital_phases = get_orbital_phases(0.0, planet.orbital_period, integration_times_ref, ndit_half)
        orbital_phases -= np.max(orbital_phases) / 2
    else:
        raise ValueError(f"Mode must be 'eclipse' or 'transit', not '{mode}'")

    # Generate deformation arrays
    if telluric_transmittance is not None:
        print('Adding telluric transmittance...')

        if isinstance(telluric_transmittance, str):  # TODO using variable types is quite bad, change that in definitive version
            telluric_data = np.loadtxt(telluric_transmittance)
            telluric_wavelengths = telluric_data[:, 0] * 1e-3  # nm to um
            telluric_transmittance = np.zeros(data_shape)

            for i, detector_wavelengths in enumerate(wavelengths_instrument):
                telluric_transmittance[i, :, :] = np.ones((data_shape[1], data_shape[2])) * \
                                                  fr.rebin_spectrum(telluric_wavelengths, telluric_data[:, 1],
                                                                    detector_wavelengths)
    else:
        print('No telluric transmittance')

    if airmass is not None:
        print('Adding Airmass...')
        
        if isinstance(airmass, str):
            if airmass.rsplit('.', 1)[1] == 'npz':
                airmass = np.load(airmass)
                xp = np.linspace(0, 1, np.size(airmass))
                x = np.linspace(0, 1, np.size(orbital_phases))
            else:
                times, airmass = load_airmassorg_data(airmass)
                ndit_airmass = int(np.max(np.ceil(planet.transit_duration / np.diff(times))))

                if ndit_airmass > airmass.size:
                    raise ValueError(f"airmass file not long enough")

                wh = np.where(airmass == np.min(airmass))[0][0]
                time_mid_transit = times[wh]
                wh = np.where(np.abs(times - time_mid_transit) < planet.transit_duration / 2)
                times = times[wh]
                airmass = airmass[wh]

                orbital_phases_airmass = get_orbital_phases(
                    orbital_phases[int(orbital_phases.size/2)], planet.orbital_period,
                    np.mean(np.diff(times)), times.size
                )
                orbital_phases_airmass -= np.max(orbital_phases_airmass) / 2

                fit_function = np.polynomial.Polynomial.fit(x=orbital_phases_airmass, y=airmass, deg=2)
                fit_function = np.polynomial.Polynomial(fit_function.convert().coef)
                airmass = fit_function(orbital_phases_airmass)

                xp = orbital_phases_airmass
                x = orbital_phases
        else:
            xp = np.linspace(0, 1, np.size(airmass))
            x = np.linspace(0, 1, np.size(orbital_phases))

        # TODO won't work with multi-D wavelengths
        airmass = np.interp(x, xp, airmass)

        for i, detector_wavelengths in enumerate(wavelengths_instrument):
            telluric_transmittance[i] = np.exp(
                np.transpose(np.transpose(
                    np.ones((np.size(orbital_phases), np.size(detector_wavelengths)))
                    * np.log(telluric_transmittance[i])
                ) * airmass)
            )
    else:
        print('No Airmass')

    if variable_throughput is not None:
        print('Adding variable throughput...')
        
        if isinstance(variable_throughput, str):
            variable_throughput = np.load(variable_throughput)
            variable_throughput = np.max(variable_throughput[0], axis=1)
            variable_throughput = variable_throughput / np.max(variable_throughput)
            xp = np.linspace(0, 1, np.size(variable_throughput))
            x = np.linspace(0, 1, np.size(orbital_phases))
            variable_throughput = np.interp(x, xp, variable_throughput)
    else:
        print('No variable throughput')

    # Get models
    kp = planet.calculate_orbital_velocity(planet.star_mass, planet.orbit_semi_major_axis)
    v_sys = np.zeros_like(orbital_phases)

    model_wavelengths_border = [
            doppler_shift(np.min(wavelengths_instrument), -3 * kp),
            doppler_shift(np.max(wavelengths_instrument), 3 * kp)
        ]

    star_data = get_PHOENIX_spec(planet.star_effective_temperature)
    star_data[:, 1] = radiosity_erg_hz2radiosity_erg_cm(
        star_data[:, 1], nc.c / star_data[:, 0]
    )

    star_data[:, 0] *= 1e4  # cm to um

    # "Nice" terminal output
    print('----\n', retrieval_name)

    # Select which model to use
    if mode == 'eclipse':
        raise NotImplementedError(f"eclipse mode not yet implemented")  # TODO (just cp the radiosity model)
        # retrieval_model = _get_secondary_eclipse_retrieval_model
    elif mode == 'transit':
        retrieval_model = _get_transit_retrieval_model
    else:
        raise ValueError(f"Mode must be 'eclipse' or 'transit', not '{mode}'")

    # Initialization
    pressures, temperature, gravity, radius, star_radius, star_effective_temperature, \
        p0, p_cloud, mean_molar_mass, mass_fractions, metallicity, \
        line_species, rayleigh_species, continuum_species, \
        model = _init_model(planet, model_wavelengths_border,
                            line_species_str=all_species)

    assert np.allclose(np.sum(list(mass_fractions.values()), axis=0), 1.0, atol=1e-14, rtol=1e-14)
    print('Mass fractions physicality check OK')

    if not os.path.isdir(retrieval_directory):
        os.mkdir(retrieval_directory)

    # Load existing mock observations or generate a new one
    if load_from is None:
        # TODO all of these could be in an object
        # Initialize true parameters
        true_parameters = {
            'planet_radius': Param(radius),
            'planet_metallicity': Param(metallicity),
            'pressures': Param(pressures),
            'temperature': Param(planet.equilibrium_temperature),
            'log10_cloud_pressure': Param(np.log10(p_cloud)),
            'log10_surface_gravity': Param(np.log10(gravity)),
            'reference_pressure': Param(p0),
            'star_effective_temperature': Param(star_effective_temperature),
            'star_radius': Param(star_radius),
            'semi_major_axis': Param(planet.orbit_semi_major_axis),
            'planet_max_radial_orbital_velocity': Param(kp),
            'system_observer_radial_velocities': Param(v_sys),
            'planet_rest_frame_shift': Param(0.0),
            'planet_orbital_inclination': Param(planet.orbital_inclination),
            'orbital_phases': Param(orbital_phases),
            'integration_time': Param(integration_times_ref),
            'airmass': Param(airmass),
            'instrument_resolving_power': Param(instrument_resolving_power),
            'wavelengths_instrument': Param(wavelengths_instrument),
            'variable_throughput': Param(variable_throughput),
            'telluric_transmittance': Param(telluric_transmittance)
        }

        for species in line_species:
            true_parameters[species] = Param(np.log10(mass_fractions[species]))

        # Generate mock observations
        print('True spectrum calculation...')
        if mode == 'eclipse':
            true_wavelengths, true_spectrum = None, None
        elif mode == 'transit':
            true_wavelengths, true_spectrum = _transit_radius_model(model, true_parameters)
        else:
            raise ValueError(f"Mode must be 'eclipse' or 'transit', not '{mode}'")

        star_radiosity = fr.rebin_spectrum(
            star_data[:, 0],
            star_data[:, 1],
            true_wavelengths
        )

        true_parameters['star_spectral_radiosity'] = Param(star_radiosity)

        print('Generating mock observations...')
        mock_observations, noise, mock_observations_without_noise = generate_mock_observations(
            wavelength_model=true_wavelengths,
            planet_spectrum_model=true_spectrum,
            telluric_transmittance=telluric_transmittance,
            variable_throughput=variable_throughput,
            integration_time=integration_times_ref,
            integration_time_ref=integration_times_ref,
            wavelength_instrument=true_parameters['wavelengths_instrument'].value,
            instrument_snr=instrument_snr,
            instrument_resolving_power=true_parameters['instrument_resolving_power'].value,
            planet_radius=true_parameters['planet_radius'].value,
            star_radius=true_parameters['star_radius'].value,
            star_spectral_radiosity=true_parameters['star_spectral_radiosity'].value,
            orbital_phases=true_parameters['orbital_phases'].value,
            system_observer_radial_velocities=true_parameters['system_observer_radial_velocities'].value,
            # TODO set to 0 for now since SNR data from Roy is at 0, but find RV source eventually
            planet_max_radial_orbital_velocity=true_parameters['planet_max_radial_orbital_velocity'].value,
            planet_orbital_inclination=true_parameters['planet_orbital_inclination'].value,
            mode=mode,
            add_noise=add_noise,
            apply_snr_mask=True,
            number=1
        )

        mock_observations = np.ma.masked_array(mock_observations)
        if mock_observations.mask.size == 1:
            mock_observations.mask = np.zeros(mock_observations.shape, dtype=bool)

        true_parameters['data'] = Param(mock_observations.data)
        true_parameters['data_mask'] = Param(mock_observations.mask)
        true_parameters['noise_matrix'] = Param(np.array(noise))

        uncertainties = np.ones(mock_observations.shape) / instrument_snr.flatten()
        true_parameters['data_uncertainties'] = Param(np.array(copy.copy(uncertainties)))

        deformation_matrix = _get_deformation_matrix(
            telluric_transmittance[0], variable_throughput, shape=mock_observations[0].shape
        )

        true_parameters['deformation_matrix'] = Param(np.array(copy.copy(deformation_matrix)))
    else:
        # Load existing observations
        mock_observations_, noise, mock_observations_without_noise, \
            reduced_mock_observations, reduced_mock_observations_without_noise, \
            log_l_tot, v_rest, kps, log_l_pseudo_retrieval, \
            wvl_pseudo_retrieval, models_pseudo_retrieval, \
            true_parameters, instrument_snr = load_all(load_from)

        # Check noise_matrix consistency
        assert np.allclose(mock_observations_, mock_observations_without_noise + noise, atol=0.0, rtol=1e-15)

        print("Mock observations noise_matrix consistency check OK")

        # Update deformation matrix
        true_parameters['variable_throughput'] = Param(variable_throughput)
        true_parameters['telluric_transmittance'] = Param(telluric_transmittance)
        true_parameters['airmass'] = Param(airmass)
        
        if telluric_transmittance is not None:
            print('Adding telluric lines...')
            
        if variable_throughput is not None:
            print('Adding variable throughput...')
        
        deformation_matrix = _get_deformation_matrix(
            telluric_transmittance, variable_throughput, shape=mock_observations_[0].shape
        )

        true_parameters['deformation_matrix'] = Param(deformation_matrix)
        
        # Update uncertainties
        uncertainties = np.ones(mock_observations_.shape) / instrument_snr
        true_parameters['data_uncertainties'] = Param(copy.copy(uncertainties))
        true_parameters['noise_matrix'] = Param(noise)

        # Update mock observations
        mock_observations = np.ma.asarray(copy.deepcopy(true_parameters['true_spectra'].value))
        mock_observations.mask = copy.deepcopy(mock_observations_.mask)

        for i, data in enumerate(mock_observations):
            mock_observations[i] = data * true_parameters['deformation_matrix'].value[i]

        if add_noise:
            mock_observations += noise

        true_parameters['data'] = Param(mock_observations)

        # Generate and save mock observations
        print('True spectrum calculation...')
        if mode == 'eclipse':
            raise NotImplementedError(f"eclipse mode not yet implemented")  # TODO (just cp the radiosity model)
            # true_wavelengths, true_spectrum = _radiosity_model(model, true_parameters)
        elif mode == 'transit':
            true_wavelengths, true_spectrum = _transit_radius_model(model, true_parameters)
        else:
            raise ValueError(f"Mode must be 'eclipse' or 'transit', not '{mode}'")

        if 'planet_radius' not in true_parameters:
            pr = true_parameters['R_pl'].value
        else:
            pr = true_parameters['planet_radius'].value

        if 'star_radius' not in true_parameters:
            sr = true_parameters['Rstar'].value
        else:
            sr = true_parameters['star_radius'].value

        _, _, mock_observations_without_noise_tmp = generate_mock_observations(
            wavelength_model=true_wavelengths,
            planet_spectrum_model=true_spectrum,
            telluric_transmittance=telluric_transmittance,
            variable_throughput=variable_throughput,
            integration_time=integration_times_ref,
            integration_time_ref=integration_times_ref,
            wavelength_instrument=true_parameters['wavelengths_instrument'].value,
            instrument_snr=instrument_snr,
            instrument_resolving_power=true_parameters['instrument_resolving_power'].value,
            planet_radius=pr,
            star_radius=sr,
            star_spectral_radiosity=true_parameters['star_spectral_radiosity'].value,
            orbital_phases=true_parameters['orbital_phases'].value,
            system_observer_radial_velocities=true_parameters['system_observer_radial_velocities'].value,
            # TODO set to 0 for now since SNR data from Roy is at 0, but find RV source eventually
            planet_max_radial_orbital_velocity=true_parameters['planet_max_radial_orbital_velocity'].value,
            planet_orbital_inclination=true_parameters['planet_orbital_inclination'].value,
            mode=mode,
            add_noise=add_noise,
            apply_snr_mask=True,
            number=1
        )

        # Check loaded data and re-generated data without noise_matrix consistency
        print(np.max(np.abs(1 - mock_observations_without_noise_tmp / (mock_observations - noise))))
        if add_noise:
            assert np.allclose(mock_observations_without_noise_tmp, mock_observations - noise, atol=1e-14, rtol=1e-14)
        else:
            assert np.allclose(mock_observations_without_noise_tmp, mock_observations, atol=1e-14, rtol=1e-14)

        print("Mock observations consistency check OK")

    print('Data reduction...')
    reduced_mock_observations, reduction_matrix, reduced_uncertainties = simple_pipeline(
        spectrum=mock_observations,
        uncertainties=uncertainties,
        wavelengths=wavelengths_instrument,
        airmass=airmass,
        polynomial_fit_degree=2,
        apply_throughput_removal=True,
        apply_telluric_lines_removal=True,
        full=True
    )

    uncertainties *= np.abs(reduction_matrix)

    if add_noise:
        reduced_mock_observations_without_noise = copy.deepcopy(mock_observations_without_noise)

        if telluric_transmittance is not None:
            reduced_mock_observations_without_noise = add_telluric_lines(
                reduced_mock_observations_without_noise, telluric_transmittance
            )

        if variable_throughput is not None:
            for i in range(reduced_mock_observations_without_noise.shape[0]):
                reduced_mock_observations_without_noise[i] = add_variable_throughput(
                    reduced_mock_observations_without_noise[i], variable_throughput
                )

        reduced_mock_observations_without_noise *= reduction_matrix
    else:
        reduced_mock_observations_without_noise = copy.deepcopy(reduced_mock_observations)

    if line_species_str != ['CO_main_iso', 'CO_36', 'CH4_main_iso', 'H2O_main_iso']:
        print("Generating retrieval model...")
        pressures, temperature, gravity, radius, star_radius, star_effective_temperature, \
            p0, p_cloud, mean_molar_mass, mass_fractions, metallicity, \
            line_species, rayleigh_species, continuum_species, \
            model = _init_model(planet, model_wavelengths_border,
                                line_species_str=line_species_str)

    # # Get true values
    print('Consistency checks...')
    _, true_spectra = retrieval_model(model, true_parameters, apply_pipeline=False)

    ts = copy.copy(true_spectra)

    if isinstance(mock_observations, np.ma.core.masked_array):
        ts = np.ma.masked_where(mock_observations.mask, ts)

    fmt, mr0t, _ = simple_pipeline(
        ts, airmass=airmass, uncertainties=true_parameters['data_uncertainties'].value, full=True,
        apply_throughput_removal=True, apply_telluric_lines_removal=True
    )
    w, r = retrieval_model(model, true_parameters)

    if line_species_str != ['CO_main_iso', 'CO_36', 'CH4_main_iso', 'H2O_main_iso']:
        print("Skipping pipeline validity checks")
    else:
        fmtd, mr0td, _ = simple_pipeline(ts * true_parameters['deformation_matrix'].value, airmass=airmass,
                                         uncertainties=true_parameters['data_uncertainties'].value,
                                         apply_throughput_removal=True,
                                         apply_telluric_lines_removal=True,
                                         full=True)
        fs, mr, _ = simple_pipeline(ts * true_parameters['deformation_matrix'].value + noise, airmass=airmass,
                                    apply_throughput_removal=True,
                                    apply_telluric_lines_removal=True,
                                    uncertainties=true_parameters['data_uncertainties'].value, full=True)

        # Check pipeline validity
        assert np.allclose(r, ts * mr0t, atol=1e-12, rtol=1e-12)
        assert np.allclose(reduced_mock_observations, (ts * true_parameters['deformation_matrix'].value + noise) * mr,
                           atol=1e-12, rtol=1e-12)

        print('Pipeline validity check OK')

        # True models checks
        assert np.all(w == wavelengths_instrument)

        if not np.allclose(r, reduced_mock_observations_without_noise, atol=0.0, rtol=1e-14):
            rmown_mean_normalized = copy.deepcopy(reduced_mock_observations_without_noise)

            for i in range(reduced_mock_observations_without_noise.shape[0]):
                rmown_mean_normalized[i, :, :] = np.transpose(
                    np.transpose(
                        reduced_mock_observations_without_noise[i, :, :])
                    / np.mean(reduced_mock_observations_without_noise[i, :, :], axis=1)
                )

            if not np.allclose(r, rmown_mean_normalized, atol=0.0, rtol=1e-14):
                print("Info: model is different from observations")
            else:
                print("True model vs observations / mean consistency check OK")
        else:
            print("True model vs observations consistency check OK")

    # Get true chi2 and true log L
    log_l_tot = None
    v_rest = None
    kps = None
    log_l_pseudo_retrieval = None
    wvl_pseudo_retrieval = None
    models_pseudo_retrieval = None

    print('Calculating true log L...')
    true_log_l, w2, r2 = _pseudo_retrieval(
        parameters=true_parameters,
        kps=[true_parameters['planet_max_radial_orbital_velocity'].value],
        v_rest=[true_parameters['planet_rest_frame_shift'].value],
        model=model, reduced_mock_observations=reduced_mock_observations, error=uncertainties, mode=mode
    )

    # # Check if true spectra are the same
    # assert np.allclose(r2[0][0], r, atol=0.0, rtol=1e-14)

    if isinstance(reduced_mock_observations, np.ma.core.masked_array):
        true_chi2 = -2 * true_log_l[0][0] / np.size(reduced_mock_observations[~reduced_mock_observations.mask])
    else:
        true_chi2 = -2 * true_log_l[0][0] / np.size(reduced_mock_observations)

    # Check Log L and chi2 when using the true set of parameter
    print(f'True log L = {true_log_l[0][0]}')
    print(f'True chi2 = {true_chi2}')

    rm_diff = 1 - 1 / (deformation_matrix[0] * reduction_matrix[0])
    md = np.ma.masked_array(copy.copy(deformation_matrix))
    md.mask = copy.copy(mock_observations.mask)

    true_parameters['true_log_l'] = Param(true_log_l[0][0])
    true_parameters['true_chi2'] = Param(true_chi2)

    # pipeline_test_noiseless = pipeline_validity_test(
    #     reduced_true_model=r,
    #     reduced_mock_observations=fmtd
    # )
    #
    # pipeline_test = pipeline_validity_test(
    #     reduced_true_model=r,
    #     reduced_mock_observations=reduced_mock_observations,
    #     mock_observations_reduction_matrix=reduction_matrix,
    #     mock_noise=noise
    # )

    # Plot figures
    # TODO put that in script instead?
    if plot:
        plot_observations(
            mock_observations[0],
            np.min(wavelengths_instrument), np.max(wavelengths_instrument), true_parameters['orbital_phases'].value[0],
            true_parameters['orbital_phases'].value[-1],
            v_min=np.percentile(mock_observations[0], 16), v_max=np.percentile(mock_observations[0], 84),
            title='Mock observations',
            cbar=True, clabel='Scaled flux',
            file_name=os.path.join(retrieval_directory, 'mock_observation.png')
        )
        plot_observations(
            reduced_mock_observations[0],
            np.min(wavelengths_instrument), np.max(wavelengths_instrument), true_parameters['orbital_phases'].value[0],
            true_parameters['orbital_phases'].value[-1],
            v_min=np.percentile(reduced_mock_observations[0], 16),
            v_max=np.percentile(reduced_mock_observations[0], 84),
            title='Reduced mock observations',
            cbar=True, clabel='Scaled flux',
            file_name=os.path.join(retrieval_directory, 'reduced_mock_observation.png')
        )
        plot_observations(
            reduced_mock_observations_without_noise[0],
            np.min(wavelengths_instrument), np.max(wavelengths_instrument), true_parameters['orbital_phases'].value[0],
            true_parameters['orbital_phases'].value[-1],
            v_min=None, v_max=None, cbar=True, clabel='Scaled flux',
            title='Reduced mock observations without noise_matrix',
            file_name=os.path.join(retrieval_directory, 'reduced_mock_observation_without_noise.png')
        )
        plot_observations(
            true_spectra[0],
            np.min(wavelengths_instrument), np.max(wavelengths_instrument), true_parameters['orbital_phases'].value[0],
            true_parameters['orbital_phases'].value[-1], v_min=None, v_max=None,
            title='True spectra',
            cbar=True, clabel='Scaled flux',
            file_name=os.path.join(retrieval_directory, 'true_spectra.png')
        )
        plot_observations(
            r[0],
            np.min(wavelengths_instrument), np.max(wavelengths_instrument), true_parameters['orbital_phases'].value[0],
            true_parameters['orbital_phases'].value[-1], v_min=None, v_max=None,
            title='True model',
            cbar=True, clabel='Scaled flux',
            file_name=os.path.join(retrieval_directory, 'true_model.png')
        )
        plot_observations(
            reduction_matrix[0],
            np.min(wavelengths_instrument), np.max(wavelengths_instrument), true_parameters['orbital_phases'].value[0],
            true_parameters['orbital_phases'].value[-1],
            title=f'Reduction matrix',
            cbar=True,
            clabel=None,
            file_name=os.path.join(retrieval_directory, 'reduction_matrix.png')
        )
        plot_observations(
            deformation_matrix[0],
            np.min(wavelengths_instrument), np.max(wavelengths_instrument), true_parameters['orbital_phases'].value[0],
            true_parameters['orbital_phases'].value[-1],
            title=f'Deformation matrix',
            cbar=True,
            clabel=None,
            file_name=os.path.join(retrieval_directory, 'deformation_matrix.png')
        )
        plot_observations(
            rm_diff,
            np.min(wavelengths_instrument), np.max(wavelengths_instrument), true_parameters['orbital_phases'].value[0],
            true_parameters['orbital_phases'].value[-1],
            v_min=-np.max(np.abs(rm_diff)), v_max=np.max(np.abs(rm_diff)),
            title=rf'$\chi_\nu^2$ = '
                  rf'{calculate_reduced_chi2(deformation_matrix, reduction_matrix, reduced_uncertainties)}',
            cbar=True,
            cmap='RdBu',
            clabel=r'1 - 1 / ($M_D$ * $M_r$)',
            file_name=os.path.join(retrieval_directory, 'cmp_md_mr.png')
        )
        # plot_observations(
        #     np.log10(np.abs(pipeline_test_noiseless[0])),
        #     np.min(wavelengths_instrument), np.max(wavelengths_instrument), true_parameters['orbital_phases'].value[0],
        #     true_parameters['orbital_phases'].value[-1],
        #     v_min=None, v_max=None,
        #     title=f'Validity = {np.ma.mean(pipeline_test_noiseless)} +/- {np.ma.std(pipeline_test_noiseless)}',
        #     cbar=True,
        #     cmap='RdBu_r',
        #     clabel=r'$\log_{10}$ |validity|',
        #     norm=colors.TwoSlopeNorm(
        #         vmin=None,
        #         vcenter=-2,
        #         vmax=None
        #     ),
        #     file_name=os.path.join(retrieval_directory, 'pipeline_validity_noiseless.png')
        # )
        # plot_observations(
        #     np.log10(np.abs(pipeline_test[0])),
        #     np.min(wavelengths_instrument), np.max(wavelengths_instrument), true_parameters['orbital_phases'].value[0],
        #     true_parameters['orbital_phases'].value[-1],
        #     v_min=None, v_max=None,
        #     title=f'Validity = {np.ma.mean(pipeline_test)} +/- {np.ma.std(pipeline_test)}',
        #     cbar=True,
        #     cmap='RdBu_r',
        #     clabel=r'$\log_{10}$ |validity|',
        #     norm=colors.TwoSlopeNorm(
        #         vmin=None,
        #         vcenter=-2,
        #         vmax=None
        #     ),
        #     file_name=os.path.join(retrieval_directory, 'pipeline_validity.png')
        # )

    save_all(
        directory=retrieval_directory,
        mock_observations=mock_observations,
        mock_observations_without_noise=mock_observations_without_noise,
        noise=noise,
        reduced_mock_observations=reduced_mock_observations,
        reduced_mock_observations_without_noise=reduced_mock_observations_without_noise,
        log_l_tot=log_l_tot,
        v_rest=v_rest,
        kps=kps,
        log_l_pseudo_retrieval=log_l_pseudo_retrieval,
        wvl_pseudo_retrieval=wvl_pseudo_retrieval,
        models_pseudo_retrieval=models_pseudo_retrieval,
        true_parameters=true_parameters,
        instrument_snr=instrument_snr
    )

    # Remove parameters not necessary for retrieval
    del true_parameters['data']
    del true_parameters['noise_matrix']
    del true_parameters['deformation_matrix']
    del true_parameters['pressures']

    return retrieval_name, retrieval_directory, \
        model, pressures, true_parameters, line_species, rayleigh_species, continuum_species, \
        retrieval_model, \
        wavelengths_instrument, reduced_mock_observations, uncertainties


def init_run(retrieval_name, prt_object, pressures, parameters, retrieved_species, rayleigh_species, continuum_species,
             retrieval_model, wavelengths_instrument, observed_spectra, observations_uncertainties):
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
        'planet_rest_frame_shift',
        'temperature'
        # 'variable_throughput_coefficient'
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

    def prior_temperature(x):
        return uniform_prior(
            cube=x,
            x1=100,
            x2=3000
        )

    def prior_vtc(x):
        return uniform_prior(
            cube=x,
            x1=0.995,
            x2=1.005
        )

    def log_prior(cube, abund_lim):
        return abund_lim[0] + abund_lim[1] * cube

    # def prior_lvtc(x):
    #     return log_prior(
    #         cube=x,
    #         abund_lim=(
    #             -15,
    #             15
    #         )
    #     )

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

    run_definition_simple.add_parameter(
        retrieved_parameters[2],
        True,
        transform_prior_cube_coordinate=prior_temperature
    )

    # Spectrum parameters
    # Fixed
    run_definition_simple.set_rayleigh_species(rayleigh_species)
    run_definition_simple.set_continuum_opacities(continuum_species)

    # Retrieved
    run_definition_simple.set_line_species(
        retrieved_species,
        eq=False,
        abund_lim=(
            -12,  # min = abund_lim[0]
            12  # max = min + abund_lim[1]
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
        wlen=wavelengths_instrument,
        flux=data_,
        flux_error=error_,
        mask=mask_
    )

    return run_definition_simple


def load_all(directory):
    print(f'Loading run parameters from {directory}...')

    load_dict = np.load(os.path.join(directory, 'run_parameters.npz'), allow_pickle=True)

    mock_observations = load_dict['mock_observations']
    mock_observations_without_noise = load_dict['mock_observations_without_noise']
    if 'noise_matrix' not in load_dict:
        noise = load_dict['noise']
    else:
        noise = load_dict['noise_matrix']
    reduced_mock_observations = load_dict['reduced_mock_observations']
    reduced_mock_observations_without_noise = load_dict['reduced_mock_observations_without_noise']
    log_l_tot = load_dict['log_l_tot']
    v_rest = load_dict['v_rest']
    kps = load_dict['kps']
    log_l_pseudo_retrieval = load_dict['log_l_pseudo_retrieval']
    wvl_pseudo_retrieval = load_dict['wvl_pseudo_retrieval']
    models_pseudo_retrieval = load_dict['mock_observations_mask']
    true_parameters = load_dict['true_parameters'][()]
    instrument_snr = load_dict['instrument_snr']

    mock_observations = np.ma.asarray(mock_observations)
    mock_observations.mask = load_dict['mock_observations_mask']

    reduced_mock_observations = np.ma.asarray(reduced_mock_observations)
    reduced_mock_observations.mask = load_dict['reduced_mock_observations_mask']

    instrument_snr = np.ma.asarray(instrument_snr)
    instrument_snr.mask = load_dict['instrument_snr_mask']

    return mock_observations, noise, mock_observations_without_noise, \
        reduced_mock_observations, reduced_mock_observations_without_noise, \
        log_l_tot, v_rest, kps, log_l_pseudo_retrieval, \
        wvl_pseudo_retrieval, models_pseudo_retrieval, \
        true_parameters, instrument_snr


def plot_observations(observations, wmin, wmax, phase_min, phase_max, v_min=None, v_max=None, title=None,
                      cbar=False, clabel=None, cmap='viridis', file_name=None, **kwargs):
    import matplotlib.pyplot as plt

    plt.figure()
    plt.imshow(
        observations, origin='lower', extent=[wmin, wmax, phase_min, phase_max], aspect='auto', vmin=v_min, vmax=v_max,
        cmap=cmap, **kwargs
    )
    plt.xlabel(rf'Wavelength ($\mu$m)')
    plt.ylabel(rf'Orbital phases')
    plt.title(title)

    if cbar:
        cbar = plt.colorbar()
        cbar.set_label(clabel)

    if file_name is not None:
        plt.savefig(file_name)


def save_all(directory, mock_observations, mock_observations_without_noise,
             noise, reduced_mock_observations, reduced_mock_observations_without_noise,
             log_l_tot, v_rest, kps,
             log_l_pseudo_retrieval,
             wvl_pseudo_retrieval, models_pseudo_retrieval, true_parameters, instrument_snr):
    print('Saving...')
    # TODO save into HDF5, and better handling of runs (make a class, etc.)

    fname = os.path.join(directory, 'run_parameters.npz')
    instrument_snr = np.ma.masked_array(instrument_snr)
    tp = copy.deepcopy(true_parameters)

    for key, value in tp.items():
        tp[key] = value.value

    # np.savez_compressed(
    #     file=fname,
    #     mock_observations=mock_observations,
    #     mock_observations_mask=mock_observations.mask,
    #     mock_observations_without_noise=mock_observations_without_noise,
    #     noise=noise,
    #     reduced_mock_observations=reduced_mock_observations,
    #     reduced_mock_observations_mask=reduced_mock_observations.mask,
    #     reduced_mock_observations_without_noise=reduced_mock_observations_without_noise,
    #     log_l_tot=log_l_tot,
    #     v_rest=v_rest,
    #     kps=kps,
    #     log_l_pseudo_retrieval=log_l_pseudo_retrieval,
    #     wvl_pseudo_retrieval=wvl_pseudo_retrieval,
    #     models_pseudo_retrieval=models_pseudo_retrieval,
    #     instrument_snr=instrument_snr,
    #     instrument_snr_mask=instrument_snr.mask,
    #     true_parameters=true_parameters
    # )
    #
    fname = os.path.join(directory, 'model_parameters.npz')

    np.savez_compressed(
        file=fname,
        units='pressures: bar, wavelengths: um, species: log10 MMR, rest: cgs',
        **tp
    )


def get_log_l(base_dirname, dir_suffix, wavelengths_borders, max_n):
    wrange_0 = wavelengths_borders * np.array([1.001, 0.999])
    wranges = np.linspace(wrange_0[0], wrange_0[1], int(max_n + 1))
    wavelengths = np.zeros(max_n)
    log_l = np.zeros(max_n)

    for i in range(max_n):
        wavelengths_borders = np.array([wranges[i], wranges[i + 1]])
        dir_name = base_dirname + str(i + 1) + dir_suffix

        data = np.load(os.path.join(dir_name, 'run_parameters2.npz'), allow_pickle=True)

        wavelengths[i] = (wavelengths_borders[0])
        log_l[i] = (data['true_parameters'][()]['true_log_l'])

    return wavelengths, log_l


def load_all_log_l(directory_prefix, directory_suffix, directory_model_suffixes, wavelengths_borders, max_n,
                   main_directory='./petitRADTRANS/__tmp/test_retrieval/'):
    log_ls = {}
    wvl = np.zeros(1)

    for directory_model_suffix in directory_model_suffixes:
        wvl, log_ls[directory_model_suffix] = get_log_l(
            base_dirname=main_directory + directory_prefix,
            dir_suffix=directory_suffix + directory_model_suffix,
            wavelengths_borders=wavelengths_borders,
            max_n=max_n,
        )

    return wvl, log_ls


def save_log_l(file, wavelengths, log_ls):
    np.savez_compressed(
        file,
        wavelengths_units='um',
        wavelengths=wavelengths,
        **log_ls
    )


def plot_log_evidences(file, key='global_log_evidences', label_prefix='', reset_colors=True, delta=True, **kwargs):
    import matplotlib.pyplot as plt

    if reset_colors:
        plt.gca().set_prop_cycle(None)

    data = np.load(file)

    full_model = data['models'][0].replace('[', '').replace(']', '').replace("'", '').split(', ')

    for i, model in enumerate(data['models']):
        if i == 0 and delta:
            continue

        label = 'full'
        model = model.replace('[', '').replace(']', '').replace("'", '').split(', ')

        for species in full_model:
            if species not in model:
                if species == 'CO_36':
                    species = '13CO'
                else:
                    species = species.split('_', 1)[0]

                if label == 'full':
                    label = f"no {species}"
                else:
                    label += f", {species}"

        if delta:
            values = data[key][0] - data[key][i]
        else:
            values = data[key][i]

        plt.semilogy(data['wavelength_bins'][:-1], values,
                     label=label_prefix + label, **kwargs)

    plt.xlabel(r'Wavelength ($\mu$m)')
    plt.ylabel(fr"$\Delta$ {key.replace('_', ' ')}")
    plt.title(f"{data['planet']}")
    plt.legend()


