"""
Script to launch a CCF analysis on multiple models.
"""
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from petitRADTRANS.ccf.ccf import *
from petitRADTRANS.ccf.etc_cli_module import *
from petitRADTRANS.ccf.mock_observation import *
from petitRADTRANS.ccf.model_containers import *


def main():
    # Base parameters
    planet_name = 'WASP-39_b'
    lbl_opacity_sampling = 4
    do_scat_emis = False
    model_suffix = ''
    wlen_modes = {
        # 'Y': np.array([0.92, 1.15]),
        # 'J': np.array([1.07, 1.4]),
        # 'H': np.array([1.4, 1.88]),
        'K': np.array([1.88, 2.55]),
        # 'L': np.array([2.7, 4.25])
    }

    pressures = np.logspace(-10, 2, 130)
    distances = [213.982 * nc.pc]  # np.logspace(1, 3, 7) * nc.c * 3600 * 24 * 365.25
    # star_apparent_magnitude_v = 12.095
    # star_apparent_magnitude_j = 10.663
    star_apparent_magnitude_j = 10
    star_apparent_magnitudes = np.linspace(4, 16, 7)

    # Models to be tested
    t_int = [200]
    metallicity = [1]#[0, np.log10(3), 1]
    co_ratio = [0.55]
    p_cloud = [1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4]
    species_list = ['all', 'H2O', 'CO']

    # Observation parameters
    # Actually matter (used to get the CRIRES SNR data from the ETC website)
    exposure_time = 6 * 3600
    integration_time = 60
    airmass = 1
    velocity_range = [-1990, 1990]
    # Old (don't do anything anymore)
    telescope_mirror_radius = 8.2e2 / 2  # cm
    telescope_throughput = 0.1
    instrument_resolving_power = 8e4
    pixel_sampling = 3

    # Load settings
    settings = load_wavelength_settings(module_dir + '/crires/wavelength_settings.dat')

    # Load planet
    planet = Planet.get(planet_name)

    # Load signal to noise ratios
    star_snr = get_crires_snr_data(settings, star_apparent_magnitude_j,  # TODO apparent mag is really annoying
                                   planet.star_effective_temperature, exposure_time,
                                   integration_time, airmass,  # TODO add FLI and seeing
                                   rewrite=False, star_apparent_magnitude_band='J')

    # Generate parameter dictionaries
    parameter_dicts = get_parameter_dicts(
        t_int, metallicity, co_ratio, p_cloud
    )

    # Load/generate relevant models
    models = {}

    for wlen_mode in wlen_modes:
        print(f"Band {wlen_mode}...")

        # Initialize grid
        models[wlen_mode], all_models_exist = init_model_grid(
            planet_name, lbl_opacity_sampling, do_scat_emis, parameter_dicts, species_list,
            wavelength_boundaries=wlen_modes[wlen_mode],
            model_suffix=model_suffix
        )

        if not all_models_exist:
            # Load or generate atmosphere
            atmosphere, atmosphere_filename = SpectralModel.get_atmosphere_model(
                wlen_bords_micron=wlen_modes[wlen_mode],
                pressures=pressures,
                line_species_list=SpectralModel.default_line_species,
                rayleigh_species=SpectralModel.default_rayleigh_species,
                continuum_opacities=SpectralModel.default_continuum_opacities,
                lbl_opacity_sampling=lbl_opacity_sampling,
                do_scat_emis=do_scat_emis,
                model_suffix=model_suffix
            )

            # Load or generate models
            models[wlen_mode] = generate_model_grid(
                models=models[wlen_mode],
                pressures=pressures,
                line_species_list='default',
                rayleigh_species='default',
                continuum_opacities='default',
                model_suffix=model_suffix,
                atmosphere=atmosphere,
                rewrite=False,
                save=True
            )
        else:
            # Load existing models
            models[wlen_mode] = load_model_grid(models[wlen_mode])

    snrs = {}
    tsm = {}

    for meta in metallicity:
        snrs[meta] = {}
        tsm[meta] = {}

        for band in wlen_modes:
            snrs[meta][band], tsm[meta][band] = get_tsm_snr_pcloud(
                band=band,
                wavelength_boundaries=wlen_modes[band] * 1e-4,
                star_distances=distances,
                p_clouds=p_cloud,
                models=models,
                species_list=species_list,
                settings=settings,
                planet=planet,
                t_int=t_int[0],
                metallicity=metallicity[0],
                co_ratio=co_ratio[0],
                velocity_range=velocity_range,
                exposure_time=exposure_time,
                telescope_mirror_radius=telescope_mirror_radius,
                telescope_throughput=telescope_throughput,
                instrument_resolving_power=instrument_resolving_power,
                pixel_sampling=pixel_sampling,
                noise_correction_coefficient=1.0,
                scale_factor=1.0,
                star_snr=star_snr,
                star_apparent_magnitude=star_apparent_magnitudes,
                star_snr_reference_apparent_magnitude=star_apparent_magnitude_j
            )

    plot_tsm_pcloud_snr(
        p_cloud, tsm, snrs,
        metallicity=metallicity[2],
        band='J',
        setting=list(settings.keys())[0],
        species=species_list[1],
        planet_name=planet_name,
        exposure_time=exposure_time
    )


def plot_spectrum(star_snr, model, instrument_resolving_power, pixel_sampling):
    obs = {}
    mod = {}

    for order in star_snr['K']['2217']:
        obs[order], full_lsf_ed, freq_out, mod[order], snr = generate_mock_observation(
            wavelengths=model.wavelengths * 1e-4,
            flux=model.transit_radius,
            snr_per_res_element=np.asarray(star_snr['K']['2217'][order]['snr']),
            instrument_resolving_power=instrument_resolving_power,
            pixel_sampling=pixel_sampling,
            instrument_wavelength_range=np.asarray(star_snr['K']['2217'][order]['wavelength']) * 1e6 * 1e-4
        )
        plt.errorbar(nc.c / freq_out * 1e4, obs[order], yerr=1 / snr, capsize=2, color='C0', ls='', marker='+')

    plt.plot(model.wavelengths, model.transit_radius, color='C1', label='model')
    plt.errorbar([-1, -2], [-1, -2], yerr=1, color='C0', ls='', marker='+', label='observation')
    plt.ylim([0.01, 0.03])
    plt.xlim([np.min(model.wavelengths), np.max(model.wavelengths)])
    plt.xlabel(r'Wavelength ($\mu$m)')
    plt.ylabel(r'Transit radius')
    plt.legend()


def get_ccf_results(wlen_modes, star_snr, settings, models, instrument_resolving_power, pixel_sampling,
                    species_list, velocity_range,
                    star_snr_reference_apparent_magnitude=None, star_apparent_magnitude=None):
    results = {}

    for band in wlen_modes:
        results[band] = {}

        snr_per_res_element = star_snr[band]

        for setting in settings[band]:
            print(f"Setting '{band}{setting}'...")

            results[band][setting] = {}

            for model in models[band]:
                print(f"Calculating for model '{model}'...")

                results[band][setting][model] = {}
                x = {}
                add = {}
                log_l_ccf_all_orders = {}

                for i, order in enumerate(settings[band][setting]):
                    if isinstance(star_snr[band], dict):
                        snr_per_res_element = np.asarray(star_snr[band][setting][order]['snr'])
                        wrange = np.asarray(star_snr[band][setting][order]['wavelength']) * 1e6  # m to um

                        if star_snr_reference_apparent_magnitude is not None:
                            snr_per_res_element *= \
                                10 ** ((star_snr_reference_apparent_magnitude - star_apparent_magnitude) / 5)
                    else:
                        # Order wavelength range
                        wrange = settings[band][setting][order]  # must be in um

                    # Observed spectrum
                    observed_spectrum, full_lsf_ed, freq_out, full_model_rebinned, snr_obs = generate_mock_observation(
                        wavelengths=models[band][model]['all'].wavelengths * 1e-4,
                        flux=models[band][model]['all'].transit_radius,
                        snr_per_res_element=snr_per_res_element,
                        instrument_resolving_power=instrument_resolving_power,
                        pixel_sampling=pixel_sampling,
                        instrument_wavelength_range=wrange * 1e-4
                    )

                    wlen_out = np.concatenate((
                        [wrange[0] * 1e-4],
                        nc.c / freq_out,
                        [wrange[-1] * 1e-4]
                    ))  # cm

                    for species in species_list:
                        if species == 'all':
                            continue

                        # Re-bin model spectrum with one species
                        single_lsf_ed, single_out, single_rebinned = \
                            convolve_rebin(
                                nc.c / (models[band][model][species].wavelengths * 1e-4),
                                models[band][model][species].transit_radius,
                                instrument_resolving_power,
                                pixel_sampling,
                                wlen_out
                            )

                        snr, velocity, cross_correlation, log_l, log_l_ccf = ccf_analysis(
                            wlen_out[1:-1], observed_spectrum, single_rebinned
                        )

                        f = interp1d(velocity, cross_correlation)

                        if i == 0:
                            x[species] = np.arange(
                                velocity_range[0], velocity_range[1], np.mean(np.diff(velocity)) / 3.
                            )
                            add[species] = f(x[species])  # upsample the CCF
                            log_l_ccf_all_orders[species] = log_l_ccf
                        else:
                            add[species] += f(x[species])
                            log_l_ccf_all_orders[species] += log_l_ccf

                for species in species_list:
                    if species == 'all':
                        continue

                    snr, mu, std = calculate_ccf_snr(x[species], add[species])

                    results[band][setting][model][species] = {
                        'S/N': snr,
                        'velocity': x[species],
                        'CCF': (add[species] - mu) / std,
                        'LogL': log_l_ccf_all_orders[species]
                    }

    return results


def plot_ccf(planet_name, observing_time, band, setting, model, species, results):
    plt.plot(
        results[band][setting][model][species]['velocity'], results[band][setting][model][species]['CCF'],
        label=f"{setting}, S/N = {results[band][setting][model][species]['S/N']}, "
              f"LogL = {results[band][setting][model][species]['LogL']}"
    )

    plt.axvline(0.)
    plt.xlim([-150, 150])
    plt.xlabel('km/s')
    plt.ylabel('S/N')
    plt.legend()
    plt.title(f"{planet_name}, {species} detection, {band}-band, {observing_time} h observing time (in-transit)")


def plot_wavelength_settings():
    settings = load_dat(module_dir + '/crires/wavelength_settings.dat')

    i = 0
    for setting in settings.keys():
        for order in settings[setting].keys():
            plt.plot([settings[setting][order][0] / 1e3, settings[setting][order][1] / 1e3], [i, i], color='C' + str(i))
        i += 1

    plt.xlabel('Wavelength (micron)')
    plt.ylabel('Setting')

    ylabels = []
    for setting in settings.keys():
        ylabels.append(setting)
    plt.yticks(np.linspace(0, len(settings) - 1, len(settings)), ylabels)

    plt.axvline(0.92)
    plt.axvline(1.15, linestyle='--')
    plt.axvline(1.07)
    plt.axvline(1.4)
    plt.axvline(1.88)
    plt.axvline(2.55)
    plt.axvline(2.7)
    plt.axvline(4.25)


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


def tsm_derivatives(tsm, planet_radius, planet_mass, planet_equilibrium_temperature, star_radius):
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

    tsm = np.zeros_like(tsm_variable)

    snrs = {}

    for setting in settings[band]:
        snrs[setting] = {}

        for species in species_list:
            if species != 'all':
                snrs[setting][species] = np.zeros((np.size(tsm_variable), np.size(p_clouds)))

    j_band_boundaries = np.array([1.07, 1.4]) * 1e-4  # cm, used in TSM calculation

    wlen_modes = {band: wavelength_boundaries}

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

        results = get_ccf_results(
            wlen_modes=wlen_modes,
            star_snr=star_snr,
            settings=settings,
            models=models,
            instrument_resolving_power=instrument_resolving_power,
            pixel_sampling=pixel_sampling,
            species_list=species_list,
            velocity_range=velocity_range,
            star_snr_reference_apparent_magnitude=star_snr_reference_apparent_magnitude,
            star_apparent_magnitude=star_mag
        )

        parameter_dict = ParametersDict(t_int, metallicity, co_ratio, 0)

        for j, pc in enumerate(p_clouds):
            parameter_dict['p_cloud'] = pc
            model = parameter_dict.to_str()

            for setting in settings[band]:
                for species in species_list:
                    if species != 'all':
                        snrs[setting][species][i, j] = results[band][setting][model][species]['S/N']

    return snrs, tsm


def plot_tsm_pcloud_snr(p_clouds, tsm, snrs, band, metallicity, setting, species, planet_name='', exposure_time=0.,
                        cmap='RdBu', detection_threshold_snr=5, vmin=0, vmax=None, levels=None):
    if levels is None:
        if vmax is None:
            l_max = np.max(snrs)
        else:
            l_max = vmax

        levels = np.arange(vmin, l_max, 2)

    plt.loglog()

    plt.contour(p_clouds, tsm[metallicity][band], snrs[metallicity][band][setting][species], cmap=cmap, levels=levels,
                norm=TwoSlopeNorm(detection_threshold_snr, vmin=vmin, vmax=vmax))

    plt.colorbar(label='CCF S/N')

    plt.gca().invert_xaxis()
    plt.xlabel('Cloud top pressure (bar)')
    plt.ylabel('TSM')
    plt.title(
        f"{planet_name}, {species} detection, {band}-band, {int(exposure_time / 3600)} h observing time (in-transit)"
    )


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


def get_snr_from_etc_data(json_data, setting_orders):
    snr_data = {}

    for order in setting_orders:
        for i in range(np.size(json_data['data']['orders'])):
            if json_data['data']['orders'][i]['order'] == str(order):
                snr_data[order] = {
                    'wavelength': [],
                    'snr': [],
                }

                # ETC's data structure is a bit convoluted
                for j in range(np.size(json_data['data']['orders'][i]['detectors'])):
                    snr_data[order]['wavelength'] += \
                        json_data['data']['orders'][i]['detectors'][j]['data']['wavelength']['wavelength']['data']

                    snr_data[order]['snr'] += \
                        json_data['data']['orders'][i]['detectors'][j]['data']['snr']['snr']['data']

                break

    return snr_data


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


if __name__ == '__main__':
    main()
