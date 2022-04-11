"""Test the radtrans module.

Essentially go through a simplified version of the tutorial, and compare the results with previous ones.
C.f. (https://petitradtrans.readthedocs.io/en/latest/content/notebooks/getting_started.html).

Do not change the parameters used to generate the comparison files, including input_data files, when running the tests.
"""
import copy

import numpy as np

from .context import petitRADTRANS
from .utils import compare_from_reference_file, \
    reference_filenames, radtrans_parameters, temperature_guillot_2010, temperature_isothermal

relative_tolerance = 1e-6  # relative tolerance when comparing with older results


# Initializations
def init_radtrans_correlated_k():
    atmosphere = petitRADTRANS.radtrans.Radtrans(
        line_species=radtrans_parameters['spectrum_parameters']['line_species_correlated_k'],
        rayleigh_species=radtrans_parameters['spectrum_parameters']['rayleigh_species'],
        continuum_opacities=radtrans_parameters['spectrum_parameters']['continuum_opacities'],
        cloud_species=list(radtrans_parameters['cloud_parameters']['cloud_species'].keys()),
        wlen_bords_micron=radtrans_parameters['spectrum_parameters']['wavelength_range_correlated_k'],
        mode='c-k'
    )

    atmosphere.setup_opa_structure(radtrans_parameters['pressures'])

    return atmosphere


atmosphere_ck = init_radtrans_correlated_k()


# Tests
def test_correlated_k_emission_spectrum():
    # Calculate an emission spectrum
    atmosphere_ck.calc_flux(
        temp=temperature_guillot_2010,
        abunds=radtrans_parameters['mass_fractions'],
        gravity=radtrans_parameters['planetary_parameters']['surface_gravity'],
        mmw=radtrans_parameters['mean_molar_mass']
    )

    # Comparison
    compare_from_reference_file(
        reference_file=reference_filenames['correlated_k_emission'],
        comparison_dict={
            'wavelength': petitRADTRANS.nat_cst.c / atmosphere_ck.freq * 1e4,
            'spectral_radiosity': atmosphere_ck.flux
        },
        relative_tolerance=relative_tolerance
    )


def test_correlated_k_emission_contribution_cloud_calculated_radius():
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

    # Comparison
    compare_from_reference_file(
        reference_file=reference_filenames['correlated_k_emission_contribution_cloud_calculated_radius'],
        comparison_dict={
            'wavelength': petitRADTRANS.nat_cst.c / atmosphere_ck.freq * 1e4,
            'contribution': atmosphere_ck.contr_em
        },
        relative_tolerance=relative_tolerance
    )


def test_correlated_k_emission_cloud_calculated_radius():
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
        sigma_lnorm=radtrans_parameters['cloud_parameters']['cloud_species']['Mg2SiO4(c)_cd']['sigma_log_normal']
    )

    # Comparison
    compare_from_reference_file(
        reference_file=reference_filenames['correlated_k_emission_cloud_calculated_radius'],
        comparison_dict={
            'wavelength': petitRADTRANS.nat_cst.c / atmosphere_ck.freq * 1e4,
            'spectral_radiosity': atmosphere_ck.flux
        },
        relative_tolerance=relative_tolerance
    )


def test_correlated_k_emission_spectrum_cloud_hansen_radius():
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

    # Comparison
    compare_from_reference_file(
        reference_file=reference_filenames['correlated_k_emission_cloud_hansen_radius'],
        comparison_dict={
            'wavelength': petitRADTRANS.nat_cst.c / atmosphere_ck.freq * 1e4,
            'spectral_radiosity': atmosphere_ck.flux
        },
        relative_tolerance=relative_tolerance
    )


def test_correlated_k_transmission_spectrum():
    # Calculate a transmission spectrum
    atmosphere_ck.calc_transm(
        temp=temperature_isothermal,
        abunds=radtrans_parameters['mass_fractions'],
        gravity=radtrans_parameters['planetary_parameters']['surface_gravity'],
        mmw=radtrans_parameters['mean_molar_mass'],
        R_pl=radtrans_parameters['planetary_parameters']['radius'] * petitRADTRANS.nat_cst.r_jup_mean,
        P0_bar=radtrans_parameters['planetary_parameters']['reference_pressure']
    )

    # Comparison
    compare_from_reference_file(
        reference_file=reference_filenames['correlated_k_transmission'],
        comparison_dict={
            'wavelength': petitRADTRANS.nat_cst.c / atmosphere_ck.freq * 1e4,
            'transit_radius': atmosphere_ck.transm_rad / petitRADTRANS.nat_cst.r_jup_mean
        },
        relative_tolerance=relative_tolerance
    )


def test_correlated_k_transmission_spectrum_cloud_power_law():
    # Calculate a transmission spectrum
    atmosphere_ck.calc_transm(
        temp=temperature_isothermal,
        abunds=radtrans_parameters['mass_fractions'],
        gravity=radtrans_parameters['planetary_parameters']['surface_gravity'],
        mmw=radtrans_parameters['mean_molar_mass'],
        R_pl=radtrans_parameters['planetary_parameters']['radius'] * petitRADTRANS.nat_cst.r_jup_mean,
        P0_bar=radtrans_parameters['planetary_parameters']['reference_pressure'],
        kappa_zero=radtrans_parameters['cloud_parameters']['kappa_zero'],
        gamma_scat=radtrans_parameters['cloud_parameters']['gamma_scattering']
    )

    # Comparison
    compare_from_reference_file(
        reference_file=reference_filenames['correlated_k_transmission_cloud_power_law'],
        comparison_dict={
            'wavelength': petitRADTRANS.nat_cst.c / atmosphere_ck.freq * 1e4,
            'transit_radius': atmosphere_ck.transm_rad / petitRADTRANS.nat_cst.r_jup_mean
        },
        relative_tolerance=relative_tolerance
    )


def test_correlated_k_transmission_spectrum_gray_cloud():
    # Calculate a transmission spectrum
    atmosphere_ck.calc_transm(
        temp=temperature_isothermal,
        abunds=radtrans_parameters['mass_fractions'],
        gravity=radtrans_parameters['planetary_parameters']['surface_gravity'],
        mmw=radtrans_parameters['mean_molar_mass'],
        R_pl=radtrans_parameters['planetary_parameters']['radius'] * petitRADTRANS.nat_cst.r_jup_mean,
        P0_bar=radtrans_parameters['planetary_parameters']['reference_pressure'],
        Pcloud=radtrans_parameters['cloud_parameters']['cloud_pressure']
    )

    # Comparison
    compare_from_reference_file(
        reference_file=reference_filenames['correlated_k_transmission_gray_cloud'],
        comparison_dict={
            'wavelength': petitRADTRANS.nat_cst.c / atmosphere_ck.freq * 1e4,
            'transit_radius': atmosphere_ck.transm_rad / petitRADTRANS.nat_cst.r_jup_mean
        },
        relative_tolerance=relative_tolerance
    )


def test_correlated_k_transmission_spectrum_rayleigh():
    # Calculate a transmission spectrum
    atmosphere_ck.calc_transm(
        temp=temperature_isothermal,
        abunds=radtrans_parameters['mass_fractions'],
        gravity=radtrans_parameters['planetary_parameters']['surface_gravity'],
        mmw=radtrans_parameters['mean_molar_mass'],
        R_pl=radtrans_parameters['planetary_parameters']['radius'] * petitRADTRANS.nat_cst.r_jup_mean,
        P0_bar=radtrans_parameters['planetary_parameters']['reference_pressure'],
        haze_factor=radtrans_parameters['cloud_parameters']['haze_factor']
    )

    # Comparison
    compare_from_reference_file(
        reference_file=reference_filenames['correlated_k_transmission_rayleigh'],
        comparison_dict={
            'wavelength': petitRADTRANS.nat_cst.c / atmosphere_ck.freq * 1e4,
            'transit_radius': atmosphere_ck.transm_rad / petitRADTRANS.nat_cst.r_jup_mean
        },
        relative_tolerance=relative_tolerance
    )


def test_correlated_k_transmission_spectrum_cloud_fixed_radius():
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

    # Comparison
    compare_from_reference_file(
        reference_file=reference_filenames['correlated_k_transmission_cloud_fixed_radius'],
        comparison_dict={
            'wavelength': petitRADTRANS.nat_cst.c / atmosphere_ck.freq * 1e4,
            'transit_radius': atmosphere_ck.transm_rad / petitRADTRANS.nat_cst.r_jup_mean
        },
        relative_tolerance=relative_tolerance
    )


def test_correlated_k_transmission_contribution_cloud_calculated_radius():
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

    # Comparison
    compare_from_reference_file(
        reference_file=reference_filenames[
            'correlated_k_transmission_contribution_cloud_calculated_radius'
        ],
        comparison_dict={
            'wavelength': petitRADTRANS.nat_cst.c / atmosphere_ck.freq * 1e4,
            'contribution': atmosphere_ck.contr_tr
        },
        relative_tolerance=relative_tolerance,
        absolute_tolerance=5e-10  # there is a max absolute error of 3.28989835374216e-10 between windows and WSL
                                  # generated files TODO investigate why
    )


def test_correlated_k_transmission_spectrum_cloud_calculated_radius():
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
        sigma_lnorm=radtrans_parameters['cloud_parameters']['cloud_species']['Mg2SiO4(c)_cd']['sigma_log_normal']
    )

    # Comparison
    compare_from_reference_file(
        reference_file=reference_filenames['correlated_k_transmission_cloud_calculated_radius'],
        comparison_dict={
            'wavelength': petitRADTRANS.nat_cst.c / atmosphere_ck.freq * 1e4,
            'transit_radius': atmosphere_ck.transm_rad / petitRADTRANS.nat_cst.r_jup_mean
        },
        relative_tolerance=relative_tolerance
    )


def __test_plot_opacities():
    atmosphere_ck.plot_opas(
        species=[
            'H2O_HITEMP',
            'CH4'
        ],
        temperature=1500,
        pressure_bar=0.1
    )
