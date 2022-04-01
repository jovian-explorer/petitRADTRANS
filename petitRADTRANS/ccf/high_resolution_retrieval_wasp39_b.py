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
from petitRADTRANS.ccf.model_containers import Planet, SpectralModel
from petitRADTRANS.ccf.pipeline import simple_pipeline, pipeline_validity_test
from petitRADTRANS.ccf.utils import calculate_reduced_chi2
from petitRADTRANS.fort_rebin import fort_rebin as fr
from petitRADTRANS.phoenix import get_PHOENIX_spec
from petitRADTRANS.physics import doppler_shift, guillot_global
from petitRADTRANS.radtrans import Radtrans
from petitRADTRANS.retrieval import RetrievalConfig
from petitRADTRANS.retrieval.util import calc_MMW, uniform_prior


class Param:
    def __init__(self, value):
        self.value = value
       

# Private functions 
# TODO replace these private functions by a nice object doing everything needed
def _init_model(planet, w_bords, line_species_str, p0=1e-2):
    print('Initialization...')
    #line_species_str = ['H2O_main_iso']  # ['H2O_main_iso', 'CO_all_iso']  # 'H2O_Exomol'

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
    line_species = line_species_str
    rayleigh_species = ['H2', 'He']
    continuum_species = ['H2-H2', 'H2-He']

    # mass_fractions = {
    #     'H2': 0.74,
    #     'He': 0.24,
    #    # line_species_str: 1e-3
    # }
    # for species in line_species_str:
    #     mass_fractions[species] = 1e-3
    #
    # m_sum = 0.0  # Check that the total mass fraction of all species is <1
    #
    # for species in line_species:
    #     m_sum += mass_fractions[species]
    #
    # mass_fractions['H2'] = (1 - m_sum) / (1 + 0.24 / 0.74)
    # mass_fractions['He'] = mass_fractions['H2'] * 0.24 / 0.74

    mass_fractions = {
        # 'H2': 0.381,
        # 'He': 0.132,
        # 'CO_all_iso': 0.3159,
        'CO_main_iso': 0.3159 * 0.99,
        'CO_36': 0.3159 * 0.1,
        'CO2_main_iso': 0.0141,
        'H2O_main_iso': 0.0926
    }

    m_sum = np.sum(list(mass_fractions.values()))
    mass_fractions['H2'] = (1 - m_sum) / (1 + 0.3458)
    mass_fractions['He'] = mass_fractions['H2'] * 0.3458

    for key in mass_fractions:
        mass_fractions[key] *= np.ones_like(pressures)

    mean_molar_mass = calc_MMW(mass_fractions)

    print('Setting up models...')
    atmosphere = Radtrans(
        line_species=line_species_str,
        rayleigh_species=['H2', 'He'],
        continuum_opacities=['H2-H2', 'H2-He'],
        wlen_bords_micron=w_bords,
        mode='lbl',
        do_scat_emis=True,
        lbl_opacity_sampling=1
    )
    atmosphere.setup_opa_structure(pressures)

    return pressures, temperature, gravity, radius, star_radius, star_effective_temperature, p0, p_cloud, \
        mean_molar_mass, mass_fractions, \
        line_species, rayleigh_species, continuum_species, \
        atmosphere


def _init_model_old(planet, w_bords, line_species_str, p0=1e-2):
    print('Initialization...')
    #line_species_str = ['H2O_main_iso']  # ['H2O_main_iso', 'CO_all_iso']  # 'H2O_Exomol'

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
    line_species = line_species_str
    rayleigh_species = ['H2', 'He']
    continuum_species = ['H2-H2', 'H2-He']

    mass_fractions = {
        'H2': 0.74,
        'He': 0.24,
       # line_species_str: 1e-3
    }
    for species in line_species_str:
        mass_fractions[species] = 1e-3

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
        line_species=line_species_str,
        rayleigh_species=['H2', 'He'],
        continuum_opacities=['H2-H2', 'H2-He'],
        wlen_bords_micron=w_bords,
        mode='lbl',
        do_scat_emis=True,
        lbl_opacity_sampling=1
    )
    atmosphere.setup_opa_structure(pressures)

    return pressures, temperature, gravity, radius, star_radius, star_effective_temperature, p0, p_cloud, \
        mean_molar_mass, mass_fractions, \
        line_species, rayleigh_species, continuum_species, \
        atmosphere


def _init_retrieval_model(prt_object, parameters):
    if 'log10_surface_gravity' not in parameters:
        surface_gravity = 10 ** parameters['log_g'].value
    else:
        surface_gravity = 10 ** parameters['log10_surface_gravity'].value

    if 'temperature' not in parameters:
        temperature = parameters['Temperature'].value
    else:
        temperature = parameters['temperature'].value

    # Make the P-T profile
    pressures = prt_object.press * 1e-6  # bar to cgs
    temperatures = guillot_global(
        pressure=pressures,
        kappa_ir=0.01,
        gamma=0.4,
        grav=surface_gravity,
        t_int=200,
        t_equ=temperature
    )

    # Make the abundance profiles
    abundances = {}
    m_sum = np.zeros(pressures.shape)  # Check that the total mass fraction of all species is <1

    for species in prt_object.line_species:
        spec = species.split('_R_')[0]  # deal with the naming scheme for binned down opacities (see below)
        abundances[species] = 10 ** parameters[spec].value * np.ones_like(pressures)
        m_sum += 10 ** parameters[spec].value

    if np.any(m_sum) > 1:
        abundances['H2'] = np.zeros(pressures.shape)
        abundances['He'] = np.zeros(pressures.shape)

        for i, s in enumerate(m_sum):
            if s > 1:
                abundances = {species: mmr / s for species, mmr in abundances.items()}
            else:
                abundances['H2'][i] = (1 - s) / (1 + 0.3458) * np.ones(pressures.shape)
                abundances['He'][i] = abundances['H2'][i] * 0.3458
    else:
        abundances['H2'] = (1 - m_sum) / (1 + 0.3458) * np.ones(pressures.shape)
        abundances['He'] = abundances['H2'] * 0.3458

    # Find the mean molecular weight in each layer
    mmw = calc_MMW(abundances)

    return temperatures, abundances, mmw


def _init_retrieval_model_old(prt_object, parameters):
    if 'log10_surface_gravity' not in parameters:
        surface_gravity = 10 ** parameters['log_g'].value
    else:
        surface_gravity = 10 ** parameters['log10_surface_gravity'].value

    if 'temperature' not in parameters:
        temperature = parameters['Temperature'].value
    else:
        temperature = parameters['temperature'].value

    # Make the P-T profile
    pressures = prt_object.press * 1e-6  # bar to cgs
    temperatures = guillot_global(
        pressure=pressures,
        kappa_ir=0.01,
        gamma=0.4,
        grav=surface_gravity,
        t_int=200,
        t_equ=temperature
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


def _get_secondary_eclipse_retrieval_model(prt_object, parameters, pt_plot_mode=None, AMR=False, apply_pipeline=True):
    wlen_model, planet_radiosity = _radiosity_model(prt_object, parameters)

    planet_velocities = Planet.calculate_planet_radial_velocity(
        parameters['planet_max_radial_orbital_velocity'].value,
        parameters['planet_orbital_inclination'].value,
        np.rad2deg(2 * np.pi * parameters['orbital_phases'].value)
    )

    spectrum_model = get_mock_secondary_eclipse_spectra(
        wavelength_model=wlen_model,
        spectrum_model=planet_radiosity,
        star_spectral_radiosity=parameters['star_spectral_radiosity'].value,
        planet_radius=parameters['planet_radius'].value,
        star_radius=parameters['star_radius'].value,
        wavelength_instrument=parameters['wavelengths_instrument'].value,
        instrument_resolving_power=parameters['instrument_resolving_power'].value,
        planet_velocities=planet_velocities,
        system_observer_radial_velocities=parameters['system_observer_radial_velocities'].value,
        planet_rest_frame_shift=parameters['planet_rest_frame_shift'].value
    )

    # TODO generation of multiple-detector models

    # Add data mask to be as close as possible as the data when performing the pipeline
    spectrum_model0 = np.ma.masked_array([spectrum_model])
    spectrum_model0.mask = copy.copy(parameters['data'].value.mask)

    if apply_pipeline:
        spectrum_model = simple_pipeline(
            spectral_data=spectrum_model0,
            airmass=parameters['airmass'].value,
            data_uncertainties=parameters['data_uncertainties'].value
        )
    else:
        spectrum_model = spectrum_model0

    return parameters['wavelengths_instrument'].value, spectrum_model


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
            spectral_data=spectrum_model0,
            airmass=parameters['airmass'].value,
            data_uncertainties=parameters['data_uncertainties'].value,
            apply_throughput_removal=False
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
        retrieval_model = _get_secondary_eclipse_retrieval_model
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


def _radiosity_model(prt_object, parameters):
    temperatures, abundances, mmw = _init_retrieval_model_old(prt_object, parameters)

    # Calculate the spectrum
    prt_object.calc_flux(
        temperatures,
        abundances,
        10 ** parameters['log10_surface_gravity'].value,
        mmw,
        Tstar=parameters['star_effective_temperature'].value,
        Rstar=parameters['star_radius'].value / nc.r_sun,
        semimajoraxis=parameters['semi_major_axis'].value / nc.AU,
        Pcloud=10 ** parameters['log10_cloud_pressure'].value,
        #stellar_intensity=parameters['star_spectral_radiosity'].value
    )

    # Transform the outputs into the units of our data.
    planet_radiosity = radiosity_erg_hz2radiosity_erg_cm(prt_object.flux, prt_object.freq)
    wlen_model = nc.c / prt_object.freq * 1e4  # wlen in micron

    return wlen_model, planet_radiosity


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


# Useful functions
def init_mock_observations(planet, line_species_str, mode,
                           retrieval_directory, retrieval_name, n_live_points, 
                           add_noise, band, wavelengths_borders, integration_times_ref,
                           wavelengths_instrument=None, instrument_snr=None, snr_file=None,
                           telluric_transmittance=None, airmass=None, variable_throughput=None,
                           instrument_resolving_power=1e5,
                           load_from=None, plot=False):
    retrieval_name += f'_{mode}'
    retrieval_name += f'_{n_live_points}lp'

    # Load SNR file
    if snr_file is not None:
        with open(snr_file, 'r') as f:
            snr_file_data = json.load(f)

            n_orders = len(snr_file_data.keys()) - 3
            n_detectors = len(snr_file_data[list(snr_file_data.keys())[0]].keys())
            n_pixels = len(snr_file_data[list(snr_file_data.keys())[0]][list(
                snr_file_data[list(snr_file_data.keys())[0]].keys()
            )[0]]['wavelength'])
            data_shape = (n_orders * n_detectors, n_pixels)

        if wavelengths_instrument is None:
            wavelengths_instrument = np.zeros(data_shape)
            i = 0

            for order, detectors in snr_file_data.items():
                if int(order) or int(order) == 23 or int(order) == 24:
                    continue
                for detector_wavelengths, values in detectors.items():
                    # if int(detector_wavelengths) != 1:
                    #     continue
                    wavelengths_instrument[i, :] = np.array(values['wavelength'])[:n_pixels] * 1e6
                    i += 1
    else:
        snr_file_data = None

    # Restrain to wavelength bounds
    # wh = np.where(np.logical_and(
    #     wavelengths_instrument > wavelengths_borders[band][0],
    #     wavelengths_instrument < wavelengths_borders[band][1]
    # ))[0]

    if snr_file_data is not None and instrument_snr is None:
        instrument_snr = np.ma.zeros(data_shape)

        i = 0

        for order, detectors in snr_file_data.items():
            if int(order) or int(order) == 23 or int(order) == 24:
                continue
            for detector_wavelengths, values in detectors.items():
                # if int(detector_wavelengths) != 1:
                #     continue
                instrument_snr[i, :] = np.array(values['snr'])[:n_pixels] * 2
                i += 1

        # instrument_snr = instrument_snr[wh]
    else:
        pass
        # instrument_snr = instrument_snr[wh]

    # wavelengths_instrument = wavelengths_instrument[wh]
    instrument_snr = np.ma.masked_less_equal(instrument_snr, 1.0)
    # wavelengths_instrument = np.array(wavelengths_instrument[np.where(~instrument_snr.mask)])
    # instrument_snr = np.array(instrument_snr[np.where(~instrument_snr.mask)])

    # Number of DITs during the transit, we assume that we had the same number of DITs for the star alone
    ndit_half = int(np.ceil(planet.transit_duration / integration_times_ref[band]))  # actual NDIT is twice this value

    # Get orbital phases
    if mode == 'eclipse':
        phase_start = 0.507  # just after secondary eclipse
        orbital_phases = \
            get_orbital_phases(phase_start, planet.orbital_period, integration_times_ref[band], ndit_half)
    elif mode == 'transit':
        orbital_phases = get_orbital_phases(0.0, planet.orbital_period, integration_times_ref[band], ndit_half)
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
                telluric_transmittance[i, :] = fr.rebin_spectrum(telluric_wavelengths, telluric_data[:, 1],
                                                                 detector_wavelengths)
    else:
        print('No telluric transmittance')

    if airmass is not None:
        print('Adding Airmass...')
        
        if isinstance(airmass, str):
            airmass = np.load(airmass)

        # TODO won't work with multi-D wavelengths
        xp = np.linspace(0, 1, np.size(airmass))
        x = np.linspace(0, 1, np.size(orbital_phases))
        airmass = np.interp(x, xp, airmass)
        telluric_transmittance = np.exp(
            np.transpose(np.transpose(
                np.ones((np.size(orbital_phases), np.size(wavelengths_instrument)))
                * np.log(telluric_transmittance)
            ) * airmass)
        )
    else:
        print('No Airmass')

    if variable_throughput is not None:
        print('Adding variable throughput...')
        
        if isinstance(variable_throughput, str):
            data_dir = os.path.abspath(os.path.join(variable_throughput))
            variable_throughput = np.load(os.path.join(data_dir, 'algn.npy'))
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

    model_wavelengths_border = {
        band: [
            doppler_shift(np.min(wavelengths_instrument), -2 * kp),
            doppler_shift(np.max(wavelengths_instrument), 2 * kp)
        ]
    }

    star_data = get_PHOENIX_spec(planet.star_effective_temperature)
    star_data[:, 1] = radiosity_erg_hz2radiosity_erg_cm(
        star_data[:, 1], nc.c / star_data[:, 0]
    )

    star_data[:, 0] *= 1e4  # cm to um

    # "Nice" terminal output
    print('----\n', retrieval_name)

    # Select which model to use
    if mode == 'eclipse':
        retrieval_model = _get_secondary_eclipse_retrieval_model
    elif mode == 'transit':
        retrieval_model = _get_transit_retrieval_model
    else:
        raise ValueError(f"Mode must be 'eclipse' or 'transit', not '{mode}'")

    # Initialization
    pressures, temperature, gravity, radius, star_radius, star_effective_temperature, \
        p0, p_cloud, mean_molar_mass, mass_fractions, \
        line_species, rayleigh_species, continuum_species, \
        model = _init_model(planet, model_wavelengths_border[band], line_species_str)

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
            true_wavelengths, true_spectrum = _radiosity_model(model, true_parameters)
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
            integration_time=integration_times_ref[band],
            integration_time_ref=integration_times_ref[band],
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
        true_parameters['true_noise'] = Param(np.array(noise))

        uncertainties = np.ones(mock_observations.shape) / instrument_snr.flatten()
        true_parameters['data_uncertainties'] = Param(np.array(copy.copy(uncertainties)))

        # Generate deformation matrix
        telluric_transmittance = telluric_transmittance.flatten()
        deformation_matrix = _get_deformation_matrix(
            telluric_transmittance, variable_throughput, shape=mock_observations[0].shape
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
        true_parameters['true_noise'] = Param(noise)

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
            true_wavelengths, true_spectrum = _radiosity_model(model, true_parameters)
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
            integration_time=integration_times_ref[band],
            integration_time_ref=integration_times_ref[band],
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
        spectral_data=mock_observations,
        data_uncertainties=uncertainties,
        airmass=airmass,
        apply_throughput_removal=False,
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

    # # Get true values
    # print('Consistency checks...')
    _, true_spectra = retrieval_model(model, true_parameters, apply_pipeline=False)
    #
    # ts = copy.copy(true_spectra)
    #
    # if isinstance(mock_observations, np.ma.core.masked_array):
    #     ts = np.ma.masked_where(mock_observations.mask, ts)
    #
    # fmt, mr0t, _ = simple_pipeline(
    #     ts, airmass=airmass, data_uncertainties=true_parameters['data_uncertainties'].value, full=True,
    #     apply_throughput_removal=False
    # )
    w, r = retrieval_model(model, true_parameters)
    #
    # fmtd, mr0td, _ = simple_pipeline(ts * true_parameters['deformation_matrix'].value, airmass=airmass,
    #                                  data_uncertainties=true_parameters['data_uncertainties'].value,
    #                                  apply_throughput_removal=False,
    #                                  full=True)
    # fs, mr, _ = simple_pipeline(ts * true_parameters['deformation_matrix'].value + noise, airmass=airmass,
    #                             apply_throughput_removal=False,
    #                             data_uncertainties=true_parameters['data_uncertainties'].value, full=True)
    #
    # # Check pipeline validity
    # assert np.allclose(r, ts * mr0t, atol=1e-14, rtol=1e-14)
    # assert np.allclose(reduced_mock_observations, (ts * true_parameters['deformation_matrix'].value + noise) * mr,
    #                    atol=1e-14, rtol=1e-14)
    #
    # print('Pipeline validity check OK')
    #
    # # True models checks
    # assert np.all(w == wavelengths_instrument)
    #
    # if not np.allclose(r, reduced_mock_observations_without_noise, atol=0.0, rtol=1e-14):
    #     rmown_mean_normalized = copy.deepcopy(reduced_mock_observations_without_noise)
    #
    #     for i in range(reduced_mock_observations_without_noise.shape[0]):
    #         rmown_mean_normalized[i, :, :] = np.transpose(
    #             np.transpose(
    #                 reduced_mock_observations_without_noise[i, :, :])
    #             / np.mean(reduced_mock_observations_without_noise[i, :, :], axis=1)
    #         )
    #
    #     if not np.allclose(r, rmown_mean_normalized, atol=0.0, rtol=1e-14):
    #         print("Info: model is different from observations")
    #     else:
    #         print("True model vs observations / mean consistency check OK")
    # else:
    #     print("True model vs observations consistency check OK")

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
    del true_parameters['true_noise']
    del true_parameters['deformation_matrix']

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
            x1=600,
            x2=2000
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

    np.savez_compressed(
        file=fname,
        mock_observations=mock_observations,
        mock_observations_mask=mock_observations.mask,
        mock_observations_without_noise=mock_observations_without_noise,
        noise=noise,
        reduced_mock_observations=reduced_mock_observations,
        reduced_mock_observations_mask=reduced_mock_observations.mask,
        reduced_mock_observations_without_noise=reduced_mock_observations_without_noise,
        log_l_tot=log_l_tot,
        v_rest=v_rest,
        kps=kps,
        log_l_pseudo_retrieval=log_l_pseudo_retrieval,
        wvl_pseudo_retrieval=wvl_pseudo_retrieval,
        models_pseudo_retrieval=models_pseudo_retrieval,
        instrument_snr=instrument_snr,
        instrument_snr_mask=instrument_snr.mask,
        true_parameters=true_parameters
    )
