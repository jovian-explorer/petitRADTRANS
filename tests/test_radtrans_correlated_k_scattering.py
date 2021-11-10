"""Test the correlated-k scattering part of the radtrans module.

Essentially go through a simplified version of the tutorial, and compare the results with previous ones.
C.f. (https://petitradtrans.readthedocs.io/en/latest/content/notebooks/getting_started.html).

Do not change the parameters used to generate the comparison files, including input_data files, when running the tests.

Due to the way scattering and correlated-k are calculated in petitRADTRANS, results using the same parameters may have
variations of <~ 1%. To take that into account, an important relative tolerance is set for the tests, and multiple tests
may be performed in order to rule out "unlucky" results.
"""
import copy

import numpy as np

from .context import petitRADTRANS
from .utils import compare_from_reference_file, \
    reference_filenames, radtrans_parameters, temperature_guillot_2010

relative_tolerance = 1e-6  # relative tolerance when comparing with older spectra


# Initializations
def init_radtrans_correlated_k():
    atmosphere = petitRADTRANS.radtrans.Radtrans(
        line_species=radtrans_parameters['spectrum_parameters']['line_species_correlated_k'],
        rayleigh_species=radtrans_parameters['spectrum_parameters']['rayleigh_species'],
        continuum_opacities=radtrans_parameters['spectrum_parameters']['continuum_opacities'],
        cloud_species=list(radtrans_parameters['cloud_parameters']['cloud_species'].keys()),
        wlen_bords_micron=radtrans_parameters['spectrum_parameters']['wavelength_range_correlated_k'],
        mode='c-k',
        do_scat_emis=True
    )

    atmosphere.setup_opa_structure(radtrans_parameters['pressures'])

    return atmosphere


atmosphere_ck_scattering = init_radtrans_correlated_k()


def test_correlated_k_emission_spectrum_cloud_calculated_radius_scattering():
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

    # Comparison
    compare_from_reference_file(
        reference_file=reference_filenames['correlated_k_emission_cloud_calculated_radius_scattering'],
        comparison_dict={
            'wavelength': petitRADTRANS.nat_cst.c / atmosphere_ck_scattering.freq * 1e4,
            'spectral_radiosity': atmosphere_ck_scattering.flux
        },
        relative_tolerance=relative_tolerance
    )


def test_correlated_k_emission_spectrum_cloud_calculated_radius_stellar_scattering_planetary_average():
    mass_fractions = copy.deepcopy(radtrans_parameters['mass_fractions'])
    mass_fractions['Mg2SiO4(c)'] = \
        radtrans_parameters['cloud_parameters']['cloud_species']['Mg2SiO4(c)_cd']['mass_fraction']

    geometry = 'planetary_ave'

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
        semimajoraxis=radtrans_parameters['planetary_parameters']['orbit_semi_major_axis']
    )

    # Comparison
    compare_from_reference_file(
        reference_file=reference_filenames[
            'correlated_k_emission_cloud_calculated_radius_scattering_planetary_ave'
        ],
        comparison_dict={
            'wavelength': petitRADTRANS.nat_cst.c / atmosphere_ck_scattering.freq * 1e4,
            'spectral_radiosity': atmosphere_ck_scattering.flux
        },
        relative_tolerance=relative_tolerance
    )


def test_correlated_k_emission_spectrum_cloud_calculated_radius_stellar_scattering_dayside():
    mass_fractions = copy.deepcopy(radtrans_parameters['mass_fractions'])
    mass_fractions['Mg2SiO4(c)'] = \
        radtrans_parameters['cloud_parameters']['cloud_species']['Mg2SiO4(c)_cd']['mass_fraction']

    geometry = 'dayside_ave'

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
        semimajoraxis=radtrans_parameters['planetary_parameters']['orbit_semi_major_axis']
    )

    # Comparison
    compare_from_reference_file(
        reference_file=reference_filenames['correlated_k_emission_cloud_calculated_radius_scattering_dayside_ave'],
        comparison_dict={
            'wavelength': petitRADTRANS.nat_cst.c / atmosphere_ck_scattering.freq * 1e4,
            'spectral_radiosity': atmosphere_ck_scattering.flux
        },
        relative_tolerance=relative_tolerance
    )


def test_correlated_k_emission_spectrum_cloud_calculated_radius_stellar_scattering_non_isotropic():
    mass_fractions = copy.deepcopy(radtrans_parameters['mass_fractions'])
    mass_fractions['Mg2SiO4(c)'] = \
        radtrans_parameters['cloud_parameters']['cloud_species']['Mg2SiO4(c)_cd']['mass_fraction']

    geometry = 'non-isotropic'

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

    # Comparison
    compare_from_reference_file(
        reference_file=reference_filenames[
            'correlated_k_emission_cloud_calculated_radius_scattering_non-isotropic'
        ],
        comparison_dict={
            'wavelength': petitRADTRANS.nat_cst.c / atmosphere_ck_scattering.freq * 1e4,
            'spectral_radiosity': atmosphere_ck_scattering.flux
        },
        relative_tolerance=relative_tolerance
    )


def test_correlated_k_transmission_spectrum_cloud_calculated_radius_scattering():
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

    # Comparison
    compare_from_reference_file(
        reference_file=reference_filenames['correlated_k_transmission_cloud_calculated_radius_scattering'],
        comparison_dict={
            'wavelength': petitRADTRANS.nat_cst.c / atmosphere_ck_scattering.freq * 1e4,
            'transit_radius': atmosphere_ck_scattering.transm_rad / petitRADTRANS.nat_cst.r_jup_mean
        },
        relative_tolerance=relative_tolerance
    )
