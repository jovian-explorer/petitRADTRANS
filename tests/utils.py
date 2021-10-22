"""
Utility functions for tests.
Regenerate the comparison files only when changing the precision of the models.
"""
import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import json

from .context import petitRADTRANS

version = "2.3.2"  # petitRADTRANS.version.version

tests_data_directory = os.path.join(os.path.dirname(__file__), 'data')
tests_error_directory = os.path.join(os.path.dirname(__file__), 'errors')
reference_filenames = {
    'config_test_radtrans':
        'config_test_radtrans',
    'guillot_2010':
        'radtrans_guillot_2010_temperature_profile_ref',
    'correlated_k_transmission':
        'radtrans_correlated_k_transmission_ref',
    'correlated_k_transmission_cloud_power_law':
        'radtrans_correlated_k_transmission_cloud_power_law_ref',
    'correlated_k_transmission_gray_cloud':
        'radtrans_correlated_k_transmission_gray_cloud_ref',
    'correlated_k_transmission_rayleigh':
        'radtrans_correlated_k_transmission_rayleigh_ref',
    'correlated_k_transmission_cloud_fixed_radius':
        'radtrans_correlated_k_transmission_cloud_fixed_radius_ref',
    'correlated_k_transmission_cloud_calculated_radius':
        'radtrans_correlated_k_transmission_cloud_calculated_radius_ref',
    'correlated_k_transmission_cloud_calculated_radius_scattering':
        'radtrans_correlated_k_transmission_cloud_calculated_radius_scattering_ref',
    'correlated_k_transmission_contribution_cloud_calculated_radius':
        'radtrans_correlated_k_transmission_contribution_cloud_calculated_radius_ref',
    'correlated_k_emission':
        'radtrans_correlated_k_emission_ref',
    'correlated_k_emission_cloud_calculated_radius':
        'radtrans_correlated_k_emission_cloud_calculated_radius_ref',
    'correlated_k_emission_cloud_calculated_radius_scattering':
        'radtrans_correlated_k_emission_cloud_calculated_radius_scattering_ref',
    'correlated_k_emission_cloud_calculated_radius_scattering_planetary_ave':
        'radtrans_correlated_k_emission_cloud_calculated_radius_scattering_average_ref',
    'correlated_k_emission_cloud_calculated_radius_scattering_dayside_ave':
        'radtrans_correlated_k_emission_cloud_calculated_radius_scattering_dayside_ref',
    'correlated_k_emission_cloud_calculated_radius_scattering_non-isotropic':
        'radtrans_correlated_k_emission_cloud_calculated_radius_scattering_non-isotropic_ref',
    'correlated_k_emission_contribution_cloud_calculated_radius':
        'radtrans_correlated_k_emission_contribution_cloud_calculated_radius_ref',
    'correlated_k_emission_cloud_hansen_radius':
        'radtrans_correlated_k_emission_cloud_hansen_radius_ref',
    'line_by_line_transmission':
        'radtrans_line_by_line_transmission_ref',
    'line_by_line_emission':
        'radtrans_line_by_line_emission_ref'
}

# Complete filenames
reference_filenames = {
    key: os.path.join(tests_data_directory, value + '.npz')
    if key != 'config_test_radtrans' else
    os.path.join(tests_data_directory, value + '.json')
    for key, value in reference_filenames.items()
}


# Common parameters
def create_test_radtrans_config_file(filename):
    with open(os.path.join(filename), 'w') as f:
        json.dump(
            obj={
                'header': f'File generated by tests.utils function\n'
                          f'wavelength units: um\n'
                          f'pressure units: log10(bar), generate using numpy.logspace\n'
                          f'planet radius units: R_jup\n'
                          f'star radius units: R_sun\n'
                          f'angle units: deg\n'
                          f'other units: cgs',
                'prt_version': f'{version}',
                'pressures': {
                    'start': -6,
                    'stop': 2,
                    'stop_thin_atmosphere': 0,
                    'num': 27
                },
                'mass_fractions': {
                    'H2': 0.74,
                    'He': 0.24,
                    'H2O_HITEMP': 0.001,
                    'H2O_main_iso': 0.001,
                    'CH4': 0.001,
                    'CO_all_iso': 0.1,
                    'Mg2SiO4(c)': 0.0
                 },
                'mean_molar_mass': 2.33,  # (g.cm-3)
                'temperature_isothermal': 1200,  # (K)
                'temperature_guillot_2010_parameters': {
                    'kappa_ir': 0.01,
                    'gamma': 0.4,
                    'intrinsic_temperature': 200,  # (K)
                    'equilibrium_temperature': 1500  # (K)
                },
                'planetary_parameters': {
                    'reference_pressure': 0.01,  # (bar)
                    'radius': 1.838,  # (R_jup)
                    'surface_gravity': 1e1 ** 2.45,  # (cm.s-2)
                    'eddy_diffusion_coefficient': 10 ** 7.5,
                    'orbit_semi_major_axis': 7.5e11,  # (cm)
                    'surface_reflectance': 0.3
                },
                'stellar_parameters': {
                    'effective_temperature': 5778,  # (K)
                    'radius': 1.0,  # (R_sun)
                    'incidence_angle': 30  # (deg)
                },
                'spectrum_parameters': {
                    'line_species_correlated_k': [
                      'H2O_HITEMP',
                      'CH4'
                    ],
                    'line_species_line_by_line': [
                          'H2O_main_iso',
                          'CO_all_iso'
                    ],
                    'rayleigh_species': ['H2', 'He'],
                    'continuum_opacities': ['H2-H2', 'H2-He'],
                    'wavelength_range_correlated_k': [0.9, 1.2],
                    'wavelength_range_line_by_line': [2.3000, 2.3025],
                },
                'cloud_parameters': {
                   'kappa_zero': 0.01,
                   'gamma_scattering': -4,
                   'cloud_pressure': 0.01,
                   'haze_factor': 10,
                   'cloud_species': {
                       'Mg2SiO4(c)_cd': {
                           'mass_fraction': 5e-7,
                           'radius': 5e-5,  # (cm)
                           'f_sed': 2,
                           'sigma_log_normal': 1.05,
                           'b_hansen': 0.01
                       },
                   }
                }
            },
            fp=f,
            indent=4
        )


def init_radtrans_parameters(recreate_parameter_file=False):
    """
    Initialize various parameters used both to perform the tests and generate the reference files.
    Do not change these parameters when comparing with a previous version.
    """
    if not os.path.isfile(reference_filenames['config_test_radtrans']) or recreate_parameter_file:
        print('Generating Radtrans test parameters file...')
        create_test_radtrans_config_file(filename=reference_filenames['config_test_radtrans'])

    with open(reference_filenames['config_test_radtrans'], 'r') as f:
        parameters = json.load(f)

    parameters['pressures_thin_atmosphere'] = np.logspace(
        parameters['pressures']['start'],
        parameters['pressures']['stop_thin_atmosphere'],
        parameters['pressures']['num']
    )

    parameters['pressures'] = np.logspace(
        parameters['pressures']['start'],
        parameters['pressures']['stop'],
        parameters['pressures']['num']
    )

    for key in parameters['mass_fractions']:
        parameters['mass_fractions'][key] *= np.ones_like(parameters['pressures'])

    parameters['mean_molar_mass'] *= np.ones_like(parameters['pressures'])
    parameters['planetary_parameters']['eddy_diffusion_coefficient'] *= np.ones_like(parameters['pressures'])

    return parameters


radtrans_parameters = init_radtrans_parameters()


# Useful functions
def __save_contribution_function(filename, atmosphere, mode='emission', plot_figure=False, figure_title=None,
                                 prt_version=version):
    wavelength = np.asarray(petitRADTRANS.nat_cst.c / atmosphere.freq * 1e4)

    if mode == 'emission':
        contribution = np.asarray(atmosphere.contr_em)
    elif mode == 'transmission':
        contribution = np.asarray(atmosphere.contr_tr)
    else:
        raise ValueError(f"unknown contribution mode '{mode}', available modes are 'emission' or 'transmission'")

    np.savez_compressed(
        os.path.join(filename),
        wavelength=wavelength,
        contribution=contribution,
        header=f'File generated by tests.utils function\n'
               f'wavelength units: um\n'
               f'spectral radiosity units: erg.cm-2.s-1.Hz-1',
        prt_version=f'{prt_version}'
    )

    if plot_figure:
        plt.figure()
        x, y = np.meshgrid(wavelength, atmosphere.press * 1e-6)
        plt.contourf(x, y, contribution, 30, cmap='bone_r')

        plt.yscale('log')
        plt.xscale('log')
        plt.ylim([1e2, 1e-6])
        plt.xlim([np.min(wavelength), np.max(wavelength)])

        plt.xlabel(r'Wavelength ($\mu$m)')
        plt.ylabel(r'Pressure (bar)')
        plt.title(figure_title)


def __save_emission_spectrum(filename, atmosphere, plot_figure=False, figure_title=None, prt_version=version):
    wavelength = np.asarray(petitRADTRANS.nat_cst.c / atmosphere.freq * 1e4)

    np.savez_compressed(
        os.path.join(filename),
        wavelength=wavelength,
        spectral_radiosity=np.asarray(atmosphere.flux),
        header=f'File generated by tests.utils function\n'
               f'wavelength units: um\n'
               f'spectral radiosity units: erg.cm-2.s-1.Hz-1',
        prt_version=f'{prt_version}'
    )

    if plot_figure:
        plt.figure()
        plt.semilogx(wavelength, atmosphere.flux)
        plt.xlabel(r'Wavelength ($\mu$m)')
        plt.ylabel(r'Spectral radiosity (erg$\cdot$s$^{-1}\cdot$cm$^{-2}\cdot$Hz$^{-1}$)')
        plt.title(figure_title)


def __save_temperature_profile(filename, temperature, plot_figure=False, figure_title=None, prt_version=version):
    np.savez_compressed(
        os.path.join(filename),
        temperature=np.asarray(temperature),
        pressure=np.asarray(radtrans_parameters['pressures']),
        header=f'File generated by tests.utils function\n'
               f'temperature units: K\n'
               f'pressure units: bar',
        prt_version=f'{prt_version}'
    )

    if plot_figure:
        plt.figure()
        plt.semilogy(temperature, radtrans_parameters['pressures'])
        plt.ylim([1e2, 1e-6])
        plt.xlabel('Temperature (K)')
        plt.ylabel('Pressure (bar)')
        plt.title(figure_title)


def __save_transmission_spectrum(filename, atmosphere, plot_figure=False, figure_title=None, prt_version=version):
    wavelength = np.asarray(petitRADTRANS.nat_cst.c / atmosphere.freq * 1e4)
    transit_radius = np.asarray(atmosphere.transm_rad / petitRADTRANS.nat_cst.r_jup_mean)

    np.savez_compressed(
        os.path.join(filename),
        wavelength=wavelength,
        transit_radius=transit_radius,
        header=f'File generated by tests.utils.create_radtrans_correlated_k_transmission_spectrum_ref\n'
               f'wavelength units: um\n'
               f'transit_radius units: R_jup',
        prt_version=f'{prt_version}'
    )

    if plot_figure:
        plt.figure()
        plt.semilogx(wavelength, transit_radius)
        plt.xlabel(r'Wavelength ($\mu$m)')
        plt.ylabel(r'Transit radius (R$_{\rm{Jup}}$)')
        plt.title(figure_title)


# Useful functions
def check_cloud_mass_fractions():
    """
    Check if cloud mass fraction is set to 0 by default.
    This is necessary to correctly assess the effect of the different clear and cloud models.
    """
    for species, mmr in radtrans_parameters['mass_fractions'].items():
        if '(c)' in species or '(l)' in species or '(s)' in species or '(cr)' in species:  # condensed species
            if not np.all(mmr == 0):
                raise ValueError(
                    f"cloud {species} has a default mass fraction different of 0, cannot perform test\n"
                    f"mass fraction was: {mmr}"
                )


def compare_from_reference_file(reference_file, comparison_dict, relative_tolerance, absolute_tolerance=0):
    reference_data = np.load(reference_file)
    print(f"Comparing generated spectrum to result from petitRADTRANS-{reference_data['prt_version']}...")

    for reference_file_key in comparison_dict:
        try:
            assert np.allclose(
                comparison_dict[reference_file_key],
                reference_data[reference_file_key],
                rtol=relative_tolerance,
                atol=absolute_tolerance
            )
        except AssertionError:
            # Save data for diagnostic
            if not os.path.isdir(tests_error_directory):
                os.mkdir(tests_error_directory)

            error_file = os.path.join(
                tests_error_directory,
                f"{os.path.basename(reference_file).rsplit('.', 1)[0]}_error_{reference_file_key}"
            )
            print(f"Saving assertion error data in file '{error_file}' for diagnostic...")

            np.savez_compressed(
                error_file,
                test_result=comparison_dict[reference_file_key],
                data=reference_data[reference_file_key],
                relative_tolerance=relative_tolerance,
                absolute_tolerance=absolute_tolerance
            )

            # Raise the AssertionError
            raise


# Initializations
def init_guillot_2010_temperature_profile():
    temperature_guillot = petitRADTRANS.nat_cst.guillot_global(
        pressure=radtrans_parameters['pressures'],
        kappa_ir=radtrans_parameters['temperature_guillot_2010_parameters']['kappa_ir'],
        gamma=radtrans_parameters['temperature_guillot_2010_parameters']['gamma'],
        grav=radtrans_parameters['planetary_parameters']['surface_gravity'],
        t_int=radtrans_parameters['temperature_guillot_2010_parameters']['intrinsic_temperature'],
        t_equ=radtrans_parameters['temperature_guillot_2010_parameters']['equilibrium_temperature']
    )

    return temperature_guillot


def init_radtrans_test():
    check_cloud_mass_fractions()

    tp_iso = radtrans_parameters['temperature_isothermal'] * np.ones_like(radtrans_parameters['pressures'])
    tp_guillot_2010 = init_guillot_2010_temperature_profile()

    return tp_iso, tp_guillot_2010


temperature_isothermal, temperature_guillot_2010 = init_radtrans_test()


# Data files generation functions
def create_guillot_2010_temperature_profile_ref(plot_figure=False):
    temperature_guillot = petitRADTRANS.nat_cst.guillot_global(
        pressure=radtrans_parameters['pressures'],
        kappa_ir=radtrans_parameters['temperature_guillot_2010_parameters']['kappa_ir'],
        gamma=radtrans_parameters['temperature_guillot_2010_parameters']['gamma'],
        grav=radtrans_parameters['planetary_parameters']['surface_gravity'],
        t_int=radtrans_parameters['temperature_guillot_2010_parameters']['intrinsic_temperature'],
        t_equ=radtrans_parameters['temperature_guillot_2010_parameters']['equilibrium_temperature']
    )

    __save_temperature_profile(
        reference_filenames['guillot_2010'], temperature_guillot, plot_figure, 'Guillot 2010 temperature profile'
    )


def create_radtrans_correlated_k_emission_spectrum_ref(plot_figure=False):
    from .test_radtrans_correlated_k import atmosphere_ck, temperature_guillot_2010

    atmosphere_ck.calc_flux(
        temp=temperature_guillot_2010,
        abunds=radtrans_parameters['mass_fractions'],
        gravity=radtrans_parameters['planetary_parameters']['surface_gravity'],
        mmw=radtrans_parameters['mean_molar_mass'],
    )

    __save_emission_spectrum(
        reference_filenames['correlated_k_emission'], atmosphere_ck, plot_figure, 'Correlated-k emission spectrum'
    )


def create_radtrans_correlated_k_emission_spectrum_cloud_calculated_radius_ref(plot_figure=False):
    from .test_radtrans_correlated_k import atmosphere_ck

    mass_fractions = copy.deepcopy(radtrans_parameters['mass_fractions'])
    mass_fractions['Mg2SiO4(c)'] = \
        radtrans_parameters['cloud_parameters']['cloud_species']['Mg2SiO4(c)_cd']['mass_fraction']

    atmosphere_ck.calc_flux(
        temp=temperature_guillot_2010,
        abunds=mass_fractions,
        gravity=radtrans_parameters['planetary_parameters']['surface_gravity'],
        mmw=radtrans_parameters['mean_molar_mass'],
        Kzz=radtrans_parameters['planetary_parameters']['eddy_diffusion_coefficient'],
        fsed=radtrans_parameters['cloud_parameters']['cloud_species']['Mg2SiO4(c)_cd']['f_sed'],
        sigma_lnorm=radtrans_parameters['cloud_parameters']['cloud_species']['Mg2SiO4(c)_cd']['sigma_log_normal'],
        contribution=True
    )

    __save_emission_spectrum(
        reference_filenames['correlated_k_emission_cloud_calculated_radius'], atmosphere_ck, plot_figure,
        'Correlated-k emission spectrum, with non-gray cloud using Hansen radius',
        prt_version=petitRADTRANS.version.version
    )

    __save_contribution_function(
        reference_filenames['correlated_k_emission_contribution_cloud_calculated_radius'],
        atmosphere_ck,
        mode='emission',
        plot_figure=plot_figure,
        figure_title='Correlated-k emission contribution function, '
                     'with non-gray cloud using calculated radius',
        prt_version=version
    )


def create_radtrans_correlated_k_emission_spectrum_cloud_calculated_radius_stellar_scattering_ref(plot_figure=False):
    from .test_radtrans_correlated_k_scattering import atmosphere_ck_scattering

    mass_fractions = copy.deepcopy(radtrans_parameters['mass_fractions'])
    mass_fractions['Mg2SiO4(c)'] = \
        radtrans_parameters['cloud_parameters']['cloud_species']['Mg2SiO4(c)_cd']['mass_fraction']

    geometries = [
        'planetary_ave',
        'dayside_ave',
        'non-isotropic'
    ]

    for geometry in geometries:
        atmosphere_ck_scattering.calc_flux(
            temp=temperature_guillot_2010,
            abunds=mass_fractions,
            gravity=radtrans_parameters['planetary_parameters']['surface_gravity'],
            mmw=radtrans_parameters['mean_molar_mass'],
            Kzz=radtrans_parameters['planetary_parameters']['eddy_diffusion_coefficient'],
            fsed=radtrans_parameters['cloud_parameters']['cloud_species']['Mg2SiO4(c)_cd']['f_sed'],
            sigma_lnorm=radtrans_parameters['cloud_parameters']['cloud_species']['Mg2SiO4(c)_cd']['sigma_log_normal'],
            geometry=geometry,
            Tstar=radtrans_parameters['stellar_parameters']['effective_temperature'],
            Rstar=radtrans_parameters['stellar_parameters']['radius'] * petitRADTRANS.nat_cst.r_sun,
            semimajoraxis=radtrans_parameters['planetary_parameters']['orbit_semi_major_axis'],
            theta_star=radtrans_parameters['stellar_parameters']['incidence_angle']
        )

        __save_emission_spectrum(
            reference_filenames[f'correlated_k_emission_cloud_calculated_radius_scattering_{geometry}'],
            atmosphere_ck_scattering, plot_figure,
            f'Correlated-k transmission spectrum, '
            f'with non-gray cloud using calculated radius and scattering ({geometry})',
            prt_version=petitRADTRANS.version.version
        )


def create_radtrans_correlated_k_emission_spectrum_cloud_hansen_radius_ref(plot_figure=False):
    from .test_radtrans_correlated_k import atmosphere_ck

    mass_fractions = copy.deepcopy(radtrans_parameters['mass_fractions'])
    mass_fractions['Mg2SiO4(c)'] = \
        radtrans_parameters['cloud_parameters']['cloud_species']['Mg2SiO4(c)_cd']['mass_fraction']

    atmosphere_ck.calc_flux(
        temp=temperature_guillot_2010,
        abunds=mass_fractions,
        gravity=radtrans_parameters['planetary_parameters']['surface_gravity'],
        mmw=radtrans_parameters['mean_molar_mass'],
        Kzz=radtrans_parameters['planetary_parameters']['eddy_diffusion_coefficient'],
        fsed=radtrans_parameters['cloud_parameters']['cloud_species']['Mg2SiO4(c)_cd']['f_sed'],
        b_hans=radtrans_parameters['cloud_parameters']['cloud_species']['Mg2SiO4(c)_cd']['b_hansen'],
        dist='hansen'
    )

    __save_emission_spectrum(
        reference_filenames['correlated_k_emission_cloud_hansen_radius'], atmosphere_ck, plot_figure,
        'Correlated-k emission spectrum, with non-gray cloud using Hansen radius',
        prt_version=petitRADTRANS.version.version
    )


def create_radtrans_correlated_k_emission_spectrum_cloud_calculated_radius_scattering_ref(plot_figure=False):
    from .test_radtrans_correlated_k_scattering import atmosphere_ck_scattering

    mass_fractions = copy.deepcopy(radtrans_parameters['mass_fractions'])
    mass_fractions['Mg2SiO4(c)'] = \
        radtrans_parameters['cloud_parameters']['cloud_species']['Mg2SiO4(c)_cd']['mass_fraction']

    atmosphere_ck_scattering.calc_flux(
        temp=temperature_guillot_2010,
        abunds=mass_fractions,
        gravity=radtrans_parameters['planetary_parameters']['surface_gravity'],
        mmw=radtrans_parameters['mean_molar_mass'],
        Kzz=radtrans_parameters['planetary_parameters']['eddy_diffusion_coefficient'],
        fsed=radtrans_parameters['cloud_parameters']['cloud_species']['Mg2SiO4(c)_cd']['f_sed'],
        sigma_lnorm=radtrans_parameters['cloud_parameters']['cloud_species']['Mg2SiO4(c)_cd']['sigma_log_normal'],
        add_cloud_scat_as_abs=True
    )

    __save_emission_spectrum(
        reference_filenames['correlated_k_emission_cloud_calculated_radius_scattering'],
        atmosphere_ck_scattering,
        plot_figure,
        'Correlated-k emission spectrum, with non-gray cloud using calculated radius and scattering',
        prt_version=version
    )


def create_radtrans_correlated_k_transmission_spectrum_cloud_calculated_radius_scattering_ref(plot_figure=False):
    from .test_radtrans_correlated_k_scattering import atmosphere_ck_scattering

    mass_fractions = copy.deepcopy(radtrans_parameters['mass_fractions'])
    mass_fractions['Mg2SiO4(c)'] = \
        radtrans_parameters['cloud_parameters']['cloud_species']['Mg2SiO4(c)_cd']['mass_fraction']

    atmosphere_ck_scattering.calc_transm(
        temp=radtrans_parameters['temperature_isothermal'] * np.ones_like(radtrans_parameters['pressures']),
        abunds=mass_fractions,
        gravity=radtrans_parameters['planetary_parameters']['surface_gravity'],
        mmw=radtrans_parameters['mean_molar_mass'],
        R_pl=radtrans_parameters['planetary_parameters']['radius'] * petitRADTRANS.nat_cst.r_jup_mean,
        P0_bar=radtrans_parameters['planetary_parameters']['reference_pressure'],
        Kzz=radtrans_parameters['planetary_parameters']['eddy_diffusion_coefficient'],
        fsed=radtrans_parameters['cloud_parameters']['cloud_species']['Mg2SiO4(c)_cd']['f_sed'],
        sigma_lnorm=radtrans_parameters['cloud_parameters']['cloud_species']['Mg2SiO4(c)_cd']['sigma_log_normal']
    )

    __save_transmission_spectrum(
        reference_filenames['correlated_k_transmission_cloud_calculated_radius_scattering'],
        atmosphere_ck_scattering,
        plot_figure,
        'Correlated-k transmission spectrum, with non-gray cloud using calculated radius and scattering',
        prt_version=version
    )


def create_radtrans_correlated_k_transmission_spectrum_ref(plot_figure=False):
    from .test_radtrans_correlated_k import atmosphere_ck

    atmosphere_ck.calc_transm(
        temp=radtrans_parameters['temperature_isothermal'] * np.ones_like(radtrans_parameters['pressures']),
        abunds=radtrans_parameters['mass_fractions'],
        gravity=radtrans_parameters['planetary_parameters']['surface_gravity'],
        mmw=radtrans_parameters['mean_molar_mass'],
        R_pl=radtrans_parameters['planetary_parameters']['radius'] * petitRADTRANS.nat_cst.r_jup_mean,
        P0_bar=radtrans_parameters['planetary_parameters']['reference_pressure']
    )

    __save_transmission_spectrum(
        reference_filenames['correlated_k_transmission'], atmosphere_ck, plot_figure,
        'Correlated-k transmission spectrum'
    )


def create_radtrans_correlated_k_transmission_spectrum_cloud_power_law_ref(plot_figure=False):
    from .test_radtrans_correlated_k import atmosphere_ck

    atmosphere_ck.calc_transm(
        temp=radtrans_parameters['temperature_isothermal'] * np.ones_like(radtrans_parameters['pressures']),
        abunds=radtrans_parameters['mass_fractions'],
        gravity=radtrans_parameters['planetary_parameters']['surface_gravity'],
        mmw=radtrans_parameters['mean_molar_mass'],
        R_pl=radtrans_parameters['planetary_parameters']['radius'] * petitRADTRANS.nat_cst.r_jup_mean,
        P0_bar=radtrans_parameters['planetary_parameters']['reference_pressure'],
        kappa_zero=radtrans_parameters['cloud_parameters']['kappa_zero'],
        gamma_scat=radtrans_parameters['cloud_parameters']['gamma_scattering']
    )

    __save_transmission_spectrum(
        reference_filenames['correlated_k_transmission_cloud_power_law'],
        atmosphere_ck, plot_figure, 'Correlated-k transmission spectrum, with power law cloud'
    )


def create_radtrans_correlated_k_transmission_spectrum_gray_cloud_ref(plot_figure=False):
    from .test_radtrans_correlated_k import atmosphere_ck

    atmosphere_ck.calc_transm(
        temp=radtrans_parameters['temperature_isothermal'] * np.ones_like(radtrans_parameters['pressures']),
        abunds=radtrans_parameters['mass_fractions'],
        gravity=radtrans_parameters['planetary_parameters']['surface_gravity'],
        mmw=radtrans_parameters['mean_molar_mass'],
        R_pl=radtrans_parameters['planetary_parameters']['radius'] * petitRADTRANS.nat_cst.r_jup_mean,
        P0_bar=radtrans_parameters['planetary_parameters']['reference_pressure'],
        Pcloud=radtrans_parameters['cloud_parameters']['cloud_pressure']
    )

    __save_transmission_spectrum(
        reference_filenames['correlated_k_transmission_gray_cloud'],
        atmosphere_ck, plot_figure, 'Correlated-k transmission spectrum, with gray cloud'
    )


def create_radtrans_correlated_k_transmission_spectrum_rayleigh_ref(plot_figure=False):
    from .test_radtrans_correlated_k import atmosphere_ck

    atmosphere_ck.calc_transm(
        temp=radtrans_parameters['temperature_isothermal'] * np.ones_like(radtrans_parameters['pressures']),
        abunds=radtrans_parameters['mass_fractions'],
        gravity=radtrans_parameters['planetary_parameters']['surface_gravity'],
        mmw=radtrans_parameters['mean_molar_mass'],
        R_pl=radtrans_parameters['planetary_parameters']['radius'] * petitRADTRANS.nat_cst.r_jup_mean,
        P0_bar=radtrans_parameters['planetary_parameters']['reference_pressure'],
        haze_factor=radtrans_parameters['cloud_parameters']['haze_factor']
    )

    __save_transmission_spectrum(
        reference_filenames['correlated_k_transmission_rayleigh'],
        atmosphere_ck, plot_figure, 'Correlated-k transmission spectrum, with hazes'
    )


def create_radtrans_correlated_k_transmission_spectrum_cloud_fixed_radius_ref(plot_figure=False):
    from .test_radtrans_correlated_k import atmosphere_ck

    mass_fractions = copy.deepcopy(radtrans_parameters['mass_fractions'])
    mass_fractions['Mg2SiO4(c)'] = \
        radtrans_parameters['cloud_parameters']['cloud_species']['Mg2SiO4(c)_cd']['mass_fraction']

    atmosphere_ck.calc_transm(
        temp=radtrans_parameters['temperature_isothermal'] * np.ones_like(radtrans_parameters['pressures']),
        abunds=mass_fractions,
        gravity=radtrans_parameters['planetary_parameters']['surface_gravity'],
        mmw=radtrans_parameters['mean_molar_mass'],
        R_pl=radtrans_parameters['planetary_parameters']['radius'] * petitRADTRANS.nat_cst.r_jup_mean,
        P0_bar=radtrans_parameters['planetary_parameters']['reference_pressure'],
        radius={'Mg2SiO4(c)': radtrans_parameters['cloud_parameters']['cloud_species']['Mg2SiO4(c)_cd']['radius']},
        sigma_lnorm=radtrans_parameters['cloud_parameters']['cloud_species']['Mg2SiO4(c)_cd']['sigma_log_normal']
    )

    __save_transmission_spectrum(
        reference_filenames['correlated_k_transmission_cloud_fixed_radius'], atmosphere_ck, plot_figure,
        'Correlated-k transmission spectrum, with non-gray cloud using fixed radius'
    )


def create_radtrans_correlated_k_transmission_spectrum_cloud_calculated_radius_ref(plot_figure=False):
    from .test_radtrans_correlated_k import atmosphere_ck

    mass_fractions = copy.deepcopy(radtrans_parameters['mass_fractions'])
    mass_fractions['Mg2SiO4(c)'] = \
        radtrans_parameters['cloud_parameters']['cloud_species']['Mg2SiO4(c)_cd']['mass_fraction']

    atmosphere_ck.calc_transm(
        temp=radtrans_parameters['temperature_isothermal'] * np.ones_like(radtrans_parameters['pressures']),
        abunds=mass_fractions,
        gravity=radtrans_parameters['planetary_parameters']['surface_gravity'],
        mmw=radtrans_parameters['mean_molar_mass'],
        R_pl=radtrans_parameters['planetary_parameters']['radius'] * petitRADTRANS.nat_cst.r_jup_mean,
        P0_bar=radtrans_parameters['planetary_parameters']['reference_pressure'],
        Kzz=radtrans_parameters['planetary_parameters']['eddy_diffusion_coefficient'],
        fsed=radtrans_parameters['cloud_parameters']['cloud_species']['Mg2SiO4(c)_cd']['f_sed'],
        sigma_lnorm=radtrans_parameters['cloud_parameters']['cloud_species']['Mg2SiO4(c)_cd']['sigma_log_normal'],
        contribution=True
    )

    __save_transmission_spectrum(
        reference_filenames['correlated_k_transmission_cloud_calculated_radius'], atmosphere_ck, plot_figure,
        'Correlated-k transmission spectrum, with non-gray cloud using calculated radius'
    )

    __save_contribution_function(
        reference_filenames['correlated_k_transmission_contribution_cloud_calculated_radius'],
        atmosphere_ck,
        mode='transmission',
        plot_figure=plot_figure,
        figure_title='Correlated-k transmission contribution function, '
                     'with non-gray cloud using calculated radius',
        prt_version=version
    )


def create_radtrans_line_by_line_emission_spectrum_ref(plot_figure=False):
    from .test_radtrans_line_by_line import atmosphere_lbl, temperature_guillot_2010

    atmosphere_lbl.calc_flux(
        temp=temperature_guillot_2010,
        abunds=radtrans_parameters['mass_fractions'],
        gravity=radtrans_parameters['planetary_parameters']['surface_gravity'],
        mmw=radtrans_parameters['mean_molar_mass'],
    )

    __save_emission_spectrum(
        reference_filenames['line_by_line_emission'],
        atmosphere_lbl, plot_figure, 'Line-by-line emission spectrum'
    )


def create_radtrans_line_by_line_transmission_spectrum_ref(plot_figure=False):
    from .test_radtrans_line_by_line import atmosphere_lbl

    atmosphere_lbl.calc_transm(
        temp=radtrans_parameters['temperature_isothermal'] * np.ones_like(radtrans_parameters['pressures']),
        abunds=radtrans_parameters['mass_fractions'],
        gravity=radtrans_parameters['planetary_parameters']['surface_gravity'],
        mmw=radtrans_parameters['mean_molar_mass'],
        R_pl=radtrans_parameters['planetary_parameters']['radius'] * petitRADTRANS.nat_cst.r_jup_mean,
        P0_bar=radtrans_parameters['planetary_parameters']['reference_pressure']
    )

    __save_transmission_spectrum(
        reference_filenames['line_by_line_transmission'],
        atmosphere_lbl, plot_figure, 'Line-by-line transmission spectrum'
    )


def create_all_comparison_files(plot_figure=False):
    create_guillot_2010_temperature_profile_ref(plot_figure)
    create_radtrans_correlated_k_emission_spectrum_ref(plot_figure)
    create_radtrans_correlated_k_emission_spectrum_cloud_hansen_radius_ref(plot_figure)
    create_radtrans_correlated_k_transmission_spectrum_ref(plot_figure)
    create_radtrans_correlated_k_transmission_spectrum_cloud_power_law_ref(plot_figure)
    create_radtrans_correlated_k_transmission_spectrum_rayleigh_ref(plot_figure)
    create_radtrans_correlated_k_transmission_spectrum_gray_cloud_ref(plot_figure)
    create_radtrans_correlated_k_transmission_spectrum_cloud_fixed_radius_ref(plot_figure)
    create_radtrans_correlated_k_transmission_spectrum_cloud_calculated_radius_ref(plot_figure)
    create_radtrans_correlated_k_emission_spectrum_cloud_calculated_radius_scattering_ref(plot_figure)
    create_radtrans_correlated_k_transmission_spectrum_cloud_calculated_radius_scattering_ref(plot_figure)
    create_radtrans_line_by_line_emission_spectrum_ref(plot_figure)
    create_radtrans_line_by_line_transmission_spectrum_ref(plot_figure)
