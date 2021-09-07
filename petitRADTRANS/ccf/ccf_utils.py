"""
Useful functions for pre/post-processing CCF analysis.
"""
from petitRADTRANS.ccf.ccf import *
from petitRADTRANS.ccf.etc_cli_module import *
from petitRADTRANS.ccf.mock_observation import *
from petitRADTRANS.ccf.model_containers import *


def calculate_star_snr(wavelengths, star_effective_temperature, star_radius, star_distance, exposure_time,
                       telescope_mirror_radius, telescope_throughput, instrument_resolving_power,
                       pixel_per_resolution_element=2):
    stellar_spectral_radiance = nc.get_PHOENIX_spec(star_effective_temperature)
    wavelength_stellar = stellar_spectral_radiance[:, 0]  # in cm

    wh = np.where(np.logical_and(
        wavelength_stellar >= np.min(wavelengths),
        wavelength_stellar <= np.max(wavelengths),
    ))
    wavelength_stellar = wavelength_stellar[wh]

    stellar_spectral_radiance = radiosity_erg_hz2radiosity_erg_cm(
        stellar_spectral_radiance[wh, 1],
        nc.c / wavelength_stellar  # in Hz
    )
    stellar_spectral_radiance = np.mean(stellar_spectral_radiance)
    wavelength_mean = np.mean(wavelengths)

    photon_number = stellar_spectral_radiance * wavelength_mean / (nc.h * nc.c) \
        * (star_radius / star_distance) ** 2 \
        * exposure_time * np.pi * telescope_mirror_radius ** 2 * telescope_throughput \
        * wavelength_mean / instrument_resolving_power / pixel_per_resolution_element

    return np.sqrt(photon_number)


def calculate_star_radiosity(wavelength_boundaries, star_effective_temperature, star_radius, star_distance):
    stellar_spectral_radiance = nc.get_PHOENIX_spec(star_effective_temperature)
    wavelength_stellar = stellar_spectral_radiance[:, 0]  # in cm
    stellar_spectral_radiance = radiosity_erg_hz2radiosity_erg_cm(
        stellar_spectral_radiance[:, 1],
        nc.c / wavelength_stellar
    )

    wh = np.asarray(np.where(np.logical_and(wavelength_stellar > wavelength_boundaries[0],
                             wavelength_stellar <= wavelength_boundaries[1])))[0]

    index_min = np.min(wh)
    index_max = np.max(wh)

    if index_min > 0:
        d_wavelength = wavelength_stellar[index_min:index_max] - wavelength_stellar[index_min - 1:index_max - 1]
    else:
        # Assume that the delta wavelength is constant for wavelengths smaller than the min wavelength
        d_wavelength = wavelength_stellar[index_min + 1:index_max] - wavelength_stellar[index_min:index_max - 1]
        d_wavelength = np.append(wavelength_stellar[index_min + 1] - wavelength_stellar[index_min], d_wavelength)

    stellar_radiance = np.sum(stellar_spectral_radiance[index_min:index_max] * d_wavelength)

    return stellar_radiance * (star_radius / star_distance) ** 2


def calculate_star_apparent_magnitude(wavelength_boundaries, star_effective_temperature, star_radius, star_distance):
    """
    Source for Vega parameters: https://en.wikipedia.org/wiki/Vega

    Args:
        wavelength_boundaries:
        star_effective_temperature:
        star_radius:
        star_distance:

    Returns:

    """
    star_radiosity = calculate_star_radiosity(
        wavelength_boundaries, star_effective_temperature, star_radius, star_distance
    )
    vega_radiosity = calculate_star_radiosity(
        wavelength_boundaries=wavelength_boundaries,
        star_effective_temperature=9602,
        star_radius=np.mean([2.362, 2.818]) * nc.r_sun,
        star_distance=25.04 * nc.c * 3600 * 24 * 365.25
    )

    return -2.5 * np.log10(star_radiosity / vega_radiosity)


def calculate_tsm(wavelength_boundaries, planet_radius, planet_mass, planet_equilibrium_temperature,
                  star_radius, star_effective_temperature, star_distance,
                  scale_factor=1.0, star_apparent_magnitude=None):
    """
    Source: Kempton et al. 2018 (https://iopscience.iop.org/article/10.1088/1538-3873/aadf6f)

    Args:
        wavelength_boundaries: (cm)
        planet_radius: (cm)
        planet_mass: (g)
        planet_equilibrium_temperature: (K)
        star_radius: (cm)
        star_effective_temperature: (K)
        star_distance: (cm)
        scale_factor: see source
        star_apparent_magnitude:

    Returns:

    """
    if star_apparent_magnitude is None:
        star_apparent_magnitude = calculate_star_apparent_magnitude(
            wavelength_boundaries, star_effective_temperature, star_radius, star_distance
        )

    return (planet_radius / nc.r_earth) ** 3 * planet_equilibrium_temperature \
        / ((planet_mass / nc.m_earth) * (star_radius / nc.r_sun) ** 2) * scale_factor \
        * 10 ** (-star_apparent_magnitude / 5)


def calculate_tsm_derivatives(tsm, planet_radius, planet_mass, planet_equilibrium_temperature, star_radius):
    d_tsm_d_planet_radius = 3 * tsm / planet_radius * nc.r_earth
    d_tsm_d_planet_equilibrium_temperature = tsm / planet_equilibrium_temperature
    d_tsm_d_planet_mass = - tsm / planet_mass * nc.m_earth
    d_tsm_d_star_radius = - 2 * tsm / star_radius * nc.r_sun
    d_tsm_d_star_apparent_magnitude = - np.log(10) / 5 * tsm

    return np.array([
        d_tsm_d_planet_radius,
        d_tsm_d_planet_mass,
        d_tsm_d_planet_equilibrium_temperature,
        d_tsm_d_star_radius,
        d_tsm_d_star_apparent_magnitude
    ])  # scale factor is already included in tsm


def generate_phoenix_star_spectrum_file(star_spectrum_file, star_effective_temperature):
    stellar_spectral_radiance = nc.get_PHOENIX_spec(star_effective_temperature)

    # Convert the spectrum to units accepted by the ETC website, don't take the first wavelength to avoid spike in conv.
    wavelength_stellar = stellar_spectral_radiance[1:, 0]  # in cm
    stellar_spectral_radiance = radiosity_erg_hz2radiosity_erg_cm(
        stellar_spectral_radiance[1:, 1],
        nc.c / wavelength_stellar
    )

    wavelength_stellar *= 1e-2  # cm to m
    stellar_spectral_radiance *= 1e-8 / np.pi  # erg.s-1.cm-2.sr-1/cm to erg.s-1.cm-2/A

    np.savetxt(star_spectrum_file, np.transpose((wavelength_stellar, stellar_spectral_radiance)))


def get_ccf_results(band, star_snr, settings, models, instrument_resolving_power, pixel_sampling,
                    species_list, velocity_range,
                    observing_time=1., transit_duration=None,
                    star_snr_reference_apparent_magnitude=None, star_apparent_magnitude=None):
    if transit_duration is None:
        transit_duration = 0.5 * observing_time

    results = {band: {}}
    snr_per_res_element = star_snr[band]

    for setting in settings[band]:
        print(f"Setting '{band}{setting}'...")

        results[band][setting] = {}

        x = {}
        add = {}
        log_l_ccf_all_orders = {}

        for i, order in enumerate(settings[band][setting]):
            if isinstance(star_snr[band], dict):
                detectors = list(star_snr[band][setting][order].keys())
                wrange = 0  # just to be sure it is initialized
            else:
                # Order wavelength range
                detectors = list(settings[band][setting][order].keys())
                wrange = settings[band][setting][order]  # must be in um

            for detector in detectors:
                if isinstance(star_snr[band], dict):
                    snr_per_res_element = np.asarray(star_snr[band][setting][order][detector]['snr'])
                    wrange = np.asarray(star_snr[band][setting][order][detector]['wavelength']) * 1e6  # m to um

                if star_snr_reference_apparent_magnitude is not None:
                    snr_per_res_element *= \
                        10 ** ((star_snr_reference_apparent_magnitude - star_apparent_magnitude) / 5)

                if np.all(snr_per_res_element <= 0):
                    print(f"Setting '{band}{setting}' order {order}: signal-to-noise ratio is 0, skipping...")
                    continue

                # Observed spectrum
                observed_spectrum, full_lsf_ed, wlen_out, full_model_rebinned, snr_obs = \
                    generate_mock_observation(
                        wavelengths=models['all'].wavelengths * 1e-4,
                        flux=models['all'].transit_radius,
                        snr_per_res_element=snr_per_res_element,
                        observing_time=observing_time,
                        transit_duration=transit_duration,
                        instrument_resolving_power=instrument_resolving_power,
                        pixel_sampling=pixel_sampling,
                        instrument_wavelength_range=wrange * 1e-4
                    )

                for species in species_list:
                    if species == 'all':
                        continue  # the all case corresponds to the observed spectrum

                    # Re-bin model spectrum with one species
                    single_lsf_ed, single_out, single_rebinned = \
                        convolve_rebin(
                            models[species].wavelengths * 1e-4,
                            models[species].transit_radius,
                            instrument_resolving_power,
                            pixel_sampling,
                            wlen_out
                        )

                    # CCF analysis
                    snr, velocity, cross_correlation, log_l, log_l_ccf = ccf_analysis(
                        wlen_out, observed_spectrum, single_rebinned
                    )

                    # Add the CCF of each order and each detector to retrieve the CCF of one setting
                    f = interp1d(velocity, cross_correlation)

                    if i == 0:
                        x[species] = np.arange(
                            velocity_range[0], velocity_range[1], np.mean(np.diff(velocity)) / 3.
                        )
                        add[species] = f(x[species])  # upsample the CCF
                        log_l_ccf_all_orders[species] = log_l_ccf
                    else:
                        try:
                            add[species] += f(x[species])
                            log_l_ccf_all_orders[species] += log_l_ccf
                        except ValueError as error_msg:
                            if str(error_msg) == 'A value in x_new is below the interpolation range.':
                                print(f"Got error message: '{error_msg}', probable cause below:")
                                print(f"    Velocity range ({np.min(velocity_range)} -- {np.max(velocity_range[1])}) "
                                      f"was larger than the output velocity range ({velocity[0]} -- {velocity[1]}), "
                                      f"ignoring setting {band}{setting}-{order}-{detector}; "
                                      f"consider reducing velocity range if this happen too often.")
                            else:
                                raise ValueError(str(error_msg))


            for species in species_list:
                if species == 'all':
                    continue

                snr, mu, std = calculate_ccf_snr(x[species], add[species])
                results[band][setting][species] = {
                    'S/N': snr,
                    'velocity': x[species],
                    'CCF': (add[species] - mu) / std,
                    'LogL': log_l_ccf_all_orders[species]
                }

    return results


def get_crires_snr_data(settings, star_apparent_magnitude, star_effective_temperature, exposure_time, integration_time,
                        airmass, star_apparent_magnitude_band='V', star_spectrum_file=None, rewrite=False):
    if star_spectrum_file is None:
        star_spectrum_file = f'{module_dir}/crires/star_spectrum_{star_effective_temperature}K.dat'

    snr_data = {}

    for band in settings:
        snr_data[band] = {}

        for setting_number in settings[band]:
            snr_data[band][setting_number] = {}

            setting_key = f'{band}{setting_number}'
            setting_orders = [int(key) for key in settings[band][setting_number]]

            snr_data_file = get_snr_data_file_name(
                instrument='crires',
                setting=setting_key,
                exposure_time=exposure_time,
                integration_time=integration_time,
                airmass=airmass,
                star_model='PHOENIX',
                star_effective_temperature=star_effective_temperature,
                star_apparent_magnitude_band=star_apparent_magnitude_band,
                star_apparent_magnitude=star_apparent_magnitude
            )

            # Loading the json file is much faster than reading the data from the website
            if not os.path.exists(snr_data_file) or rewrite:
                print(f"file '{snr_data_file}' does not exist, downloading...")

                if not os.path.exists(star_spectrum_file):
                    print(f"file '{star_spectrum_file}' does not exist, generating...")

                    generate_phoenix_star_spectrum_file(star_spectrum_file, star_effective_temperature)

                json_data = download_snr_data(
                    request_file_name='etc-form.json',
                    star_spectrum_file_name=star_spectrum_file,
                    star_apparent_magnitude=star_apparent_magnitude,
                    star_effective_temperature=star_effective_temperature,
                    exposure_time=exposure_time,
                    integration_time=integration_time,
                    airmass=airmass,
                    setting=setting_key,
                    setting_orders=setting_orders,
                    star_apparent_magnitude_band=star_apparent_magnitude_band,
                )

                snr_data[band][setting_number] = get_snr_from_etc_data(json_data, setting_orders)

                output(snr_data[band][setting_number], do_collapse=False, indent=4, outputfile=snr_data_file)
            else:
                print(f"loading file '{snr_data_file}'...")

                with open(snr_data_file, 'r') as f:
                    snr_data[band][setting_number] = json.load(f)

    return snr_data


def get_snr_from_etc_data(json_data, setting_orders):
    snr_data = {}

    for order in setting_orders:
        order = str(order)

        for i in range(np.size(json_data['data']['orders'])):
            if json_data['data']['orders'][i]['order'] == order:
                snr_data[order] = {}

                # ETC's data structure is a bit convoluted
                for j in range(np.size(json_data['data']['orders'][i]['detectors'])):
                    snr_data[order][j] = {
                        'wavelength': [],
                        'snr': [],
                    }

                    snr_data[order][j]['wavelength'] = \
                        json_data['data']['orders'][i]['detectors'][j]['data']['wavelength']['wavelength']['data']

                    snr_data[order][j]['snr'] = \
                        json_data['data']['orders'][i]['detectors'][j]['data']['snr']['snr']['data']

                break

    return snr_data


def get_tsm_snr_pcloud(band, wavelength_boundaries, star_distances, p_clouds, models, species_list, settings, planet,
                       t_int, metallicity, co_ratio,
                       velocity_range,
                       exposure_time, telescope_mirror_radius, telescope_throughput,
                       instrument_resolving_power, pixel_sampling,
                       noise_correction_coefficient=1.0, scale_factor=1.0, star_snr=None,
                       star_apparent_magnitude=None, star_snr_reference_apparent_magnitude=None):
    settings = {band: settings[band]}

    if star_apparent_magnitude is not None:
        tsm_variable = star_apparent_magnitude
    else:
        tsm_variable = star_distances

    # Avoid infinite noise if transit duration is set to the default value of 0
    if planet.transit_duration <= 0:
        planet_transit_duration = None
    else:
        planet_transit_duration = planet.transit_duration

    tsm = np.zeros_like(tsm_variable)

    snrs = {}

    for setting in settings[band]:
        snrs[setting] = {}

        for species in species_list:
            if species != 'all':
                snrs[setting][species] = np.zeros((np.size(tsm_variable), np.size(p_clouds)))

    j_band_boundaries = np.array([1.07, 1.4]) * 1e-4  # cm, used in TSM calculation

    for i, tsm_var in enumerate(tsm_variable):
        print(f"TSM {i + 1}/{len(tsm_variable)}")

        if star_apparent_magnitude is not None:
            tsm[i] = calculate_tsm(
                j_band_boundaries, planet.radius, planet.mass, planet.equilibrium_temperature,
                planet.star_radius, planet.star_effective_temperature, 0, scale_factor, tsm_var
            )  # the TSM definition uses the apparent magnitude of the J-band

            star_mag = tsm_var
        else:
            tsm[i] = calculate_tsm(
                j_band_boundaries, planet.radius, planet.mass, planet.equilibrium_temperature,
                planet.star_radius, planet.star_effective_temperature, tsm_var, scale_factor, None
            )  # the TSM definition uses the apparent magnitude of the J-band

            star_mag = None

        if star_snr is None:
            if star_apparent_magnitude is not None:  # TODO calculate star SNR using apparent magnitude
                raise ValueError('cannot calculate star SNR using apparent magnitude; '
                                 'set star_apparent_magnitude to None or give a value to star_snr')

            print('Calculating star SNR...')

            star_snr = {
                band: calculate_star_snr(
                    wavelength_boundaries, planet.star_temperature, planet.star_radius, tsm_var,
                    exposure_time, telescope_mirror_radius, telescope_throughput,
                    instrument_resolving_power
                ) * noise_correction_coefficient
            }
        else:
            star_snr = {
                band: star_snr[band]
            }

        results = {}

        for j, pc in enumerate(p_clouds):
            target_model = ParametersDict(t_int, metallicity, co_ratio, pc).to_str()
            model_found = False

            for model in models[band]:
                if model == target_model:
                    print(f"Calculating for model '{model}'...")

                    results[model] = get_ccf_results(
                        band=band,
                        star_snr=star_snr,
                        settings=settings,
                        models=models[band][target_model],
                        instrument_resolving_power=instrument_resolving_power,
                        pixel_sampling=pixel_sampling,
                        species_list=species_list,
                        velocity_range=velocity_range,
                        observing_time=exposure_time,
                        transit_duration=planet_transit_duration,
                        star_snr_reference_apparent_magnitude=star_snr_reference_apparent_magnitude,
                        star_apparent_magnitude=star_mag
                    )

                    model_found = True
                    break

            if model_found:
                for setting in settings[band]:
                    for species in species_list:
                        if species != 'all':
                            snrs[setting][species][i, j] = results[target_model][band][setting][species]['S/N']
            else:
                raise KeyError(f"model '{target_model}' was not found in the models dictionary")

    return snrs, tsm


def load_dat(file, **kwargs):
    """
    Load a data file.

    Args:
        file: data file
        **kwargs: keywords arguments for numpy.loadtxt()

    Returns:
        data_dict: a dictionary containing the data
    """
    with open(file, 'r') as f:
        header = f.readline()
        unit_line = f.readline()

    header_keys = header.rsplit('!')[0].split('#')[-1].split()
    units = unit_line.split('#')[-1].split()

    data = np.loadtxt(file, **kwargs)
    data_dict = {}

    for i, key in enumerate(header_keys):
        data_dict[key] = data[:, i]

    data_dict['units'] = units

    return data_dict


def load_wavelength_settings(file):
    """
    Load an instrument settings file into a handy dictionary.
    The dictionary will be organized hierarchically as follows: band > setting > order.

    Args:
        file: file containing the settings

    Returns:
        settings: the settings in a dictionary
    """
    data = load_dat(file, dtype=str)

    # Check wavelengths units
    wavelength_conversion_coefficient = 1

    for i, key in enumerate(data):
        if key == 'starting_wavelength':
            if data['units'][i] == 'nm':
                wavelength_conversion_coefficient = 1e-3
            elif data['units'][i] == 'um':
                wavelength_conversion_coefficient = 1
            else:
                raise ValueError(f"Wavelengths units must be 'nm' or 'um', not in '{data['units'][i]}'")

            break

    settings = {}

    for i, instrument_setting in enumerate(data['instrument_setting']):
        band = instrument_setting[0]
        setting = instrument_setting[1:]
        order = data['order'][i]

        if band not in settings:
            settings[band] = {}

        if setting not in settings[band]:
            settings[band][setting] = {}

        # Adding a detector 0
        settings[band][setting][order] = {0: np.array([
            data['starting_wavelength'][i],
            data['ending_wavelength'][i]
        ], dtype=float) * wavelength_conversion_coefficient}

    return settings


def radiosity_erg_hz2radiosity_erg_cm(radiosity_erg_hz, frequency):
    """
    Convert a radiosity from erg.s-1.cm-2.sr-1/Hz to erg.s-1.cm-2.sr-1/cm at a given frequency.
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
