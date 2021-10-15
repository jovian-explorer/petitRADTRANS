"""
Test the radtrans module.
Essentially go through a simplified version of the tutorial, and compare the results with previous ones.
C.f. (https://petitradtrans.readthedocs.io/en/latest/content/notebooks/getting_started.html).

Do not change the parameters used to generate the comparison files, including input_data files, when running the tests.
"""
import numpy as np

from .context import petitRADTRANS
from .utils import reference_filenames, pressures, temperature_isothermal, mass_fractions, mean_molar_mass, \
    temperature_guillot_2010_parameters, planetary_parameters, spectrum_parameters, cloud_parameters

relative_tolerance = 1e-6  # relative tolerance when comparing with older results


# Initializations
def init_guillot_2010_temperature_profile():
    temperature_guillot = petitRADTRANS.nat_cst.guillot_global(
        pressure=pressures,
        kappa_ir=temperature_guillot_2010_parameters['kappa_ir'],
        gamma=temperature_guillot_2010_parameters['gamma'],
        grav=planetary_parameters['surface_gravity'],
        t_int=temperature_guillot_2010_parameters['intrinsic_temperature'],
        t_equ=temperature_guillot_2010_parameters['equilibrium_temperature']
    )

    return temperature_guillot


def init_radtrans_correlated_k():
    atmosphere = petitRADTRANS.radtrans.Radtrans(
        line_species=spectrum_parameters['line_species_correlated_k'],
        rayleigh_species=spectrum_parameters['rayleigh_species'],
        continuum_opacities=spectrum_parameters['continuum_opacities'],
        wlen_bords_micron=spectrum_parameters['wavelength_range_correlated_k'],
        mode='c-k'
    )

    atmosphere.setup_opa_structure(pressures)

    return atmosphere


def init_radtrans_line_by_line():
    atmosphere = petitRADTRANS.radtrans.Radtrans(
        line_species=spectrum_parameters['line_species_line_by_line'],
        rayleigh_species=spectrum_parameters['rayleigh_species'],
        continuum_opacities=spectrum_parameters['continuum_opacities'],
        wlen_bords_micron=spectrum_parameters['wavelength_range_line_by_line'],
        mode='lbl'
    )

    atmosphere.setup_opa_structure(pressures)

    return atmosphere


atmosphere_ck = init_radtrans_correlated_k()
atmosphere_lbl = init_radtrans_line_by_line()
temperature_iso = temperature_isothermal * np.ones_like(pressures)
temperature_guillot_2010 = init_guillot_2010_temperature_profile()


# Useful functions
def compare_from_reference_file(reference_file, comparison_dict):
    reference_data = np.load(reference_file)
    print(f"Comparing generated spectrum to result from petitRADTRANS-{reference_data['prt_version']}...")

    for reference_file_key in comparison_dict:
        assert np.allclose(
            comparison_dict[reference_file_key],
            reference_data[reference_file_key],
            rtol=relative_tolerance,
            atol=0
        )


# Tests
def test_guillot_2010_temperature_profile():
    # Load expected results
    reference_data = np.load(reference_filenames['guillot_2010'])
    print(f"Comparing generated spectrum to result from petitRADTRANS-{reference_data['prt_version']}...")
    temperature_ref = reference_data['temperature']
    pressure_ref = reference_data['pressure']

    # Check if temperature is as expected
    assert np.allclose(
        pressures,
        pressure_ref,
        rtol=relative_tolerance,
        atol=0
    )

    assert np.allclose(
        temperature_guillot_2010,
        temperature_ref,
        rtol=relative_tolerance,
        atol=0
    )


def test_correlated_k_emission_spectrum():
    # Calculate an emission spectrum
    atmosphere_ck.calc_flux(
        temp=temperature_guillot_2010,
        abunds=mass_fractions,
        gravity=planetary_parameters['surface_gravity'],
        mmw=mean_molar_mass,
    )

    # Comparison
    compare_from_reference_file(
        reference_file=reference_filenames['correlated_k_emission'],
        comparison_dict={
            'wavelength': petitRADTRANS.nat_cst.c / atmosphere_ck.freq * 1e4,
            'spectral_radiosity': atmosphere_ck.flux
        }
    )


def test_correlated_k_transmission_spectrum():
    # Calculate a transmission spectrum
    atmosphere_ck.calc_transm(
        temp=temperature_iso,
        abunds=mass_fractions,
        gravity=planetary_parameters['surface_gravity'],
        mmw=mean_molar_mass,
        R_pl=planetary_parameters['radius'] * petitRADTRANS.nat_cst.r_jup_mean,
        P0_bar=planetary_parameters['reference_pressure']
    )

    # Comparison
    compare_from_reference_file(
        reference_file=reference_filenames['correlated_k_transmission'],
        comparison_dict={
            'wavelength': petitRADTRANS.nat_cst.c / atmosphere_ck.freq * 1e4,
            'transit_radius': atmosphere_ck.transm_rad / petitRADTRANS.nat_cst.r_jup_mean
        }
    )


def test_correlated_k_transmission_spectrum_cloud_power_law():
    # Calculate a transmission spectrum
    atmosphere_ck.calc_transm(
        temp=temperature_iso,
        abunds=mass_fractions,
        gravity=planetary_parameters['surface_gravity'],
        mmw=mean_molar_mass,
        R_pl=planetary_parameters['radius'] * petitRADTRANS.nat_cst.r_jup_mean,
        P0_bar=planetary_parameters['reference_pressure'],
        kappa_zero=cloud_parameters['kappa_zero'],
        gamma_scat=cloud_parameters['gamma_scattering']
    )

    # Comparison
    compare_from_reference_file(
        reference_file=reference_filenames['correlated_k_transmission_cloud_power_law'],
        comparison_dict={
            'wavelength': petitRADTRANS.nat_cst.c / atmosphere_ck.freq * 1e4,
            'transit_radius': atmosphere_ck.transm_rad / petitRADTRANS.nat_cst.r_jup_mean
        }
    )


def test_correlated_k_transmission_spectrum_gray_cloud():
    # Calculate a transmission spectrum
    atmosphere_ck.calc_transm(
        temp=temperature_iso,
        abunds=mass_fractions,
        gravity=planetary_parameters['surface_gravity'],
        mmw=mean_molar_mass,
        R_pl=planetary_parameters['radius'] * petitRADTRANS.nat_cst.r_jup_mean,
        P0_bar=planetary_parameters['reference_pressure'],
        Pcloud=cloud_parameters['cloud_pressure']
    )

    # Comparison
    compare_from_reference_file(
        reference_file=reference_filenames['correlated_k_transmission_gray_cloud'],
        comparison_dict={
            'wavelength': petitRADTRANS.nat_cst.c / atmosphere_ck.freq * 1e4,
            'transit_radius': atmosphere_ck.transm_rad / petitRADTRANS.nat_cst.r_jup_mean
        }
    )


def test_correlated_k_transmission_spectrum_rayleigh():
    # Calculate a transmission spectrum
    atmosphere_ck.calc_transm(
        temp=temperature_iso,
        abunds=mass_fractions,
        gravity=planetary_parameters['surface_gravity'],
        mmw=mean_molar_mass,
        R_pl=planetary_parameters['radius'] * petitRADTRANS.nat_cst.r_jup_mean,
        P0_bar=planetary_parameters['reference_pressure'],
        haze_factor=cloud_parameters['haze_factor']
    )

    # Comparison
    compare_from_reference_file(
        reference_file=reference_filenames['correlated_k_transmission_rayleigh'],
        comparison_dict={
            'wavelength': petitRADTRANS.nat_cst.c / atmosphere_ck.freq * 1e4,
            'transit_radius': atmosphere_ck.transm_rad / petitRADTRANS.nat_cst.r_jup_mean
        }
    )


def test_line_by_line_emission_spectrum():
    # Calculate an emission spectrum
    atmosphere_lbl.calc_flux(
        temp=temperature_guillot_2010,
        abunds=mass_fractions,
        gravity=planetary_parameters['surface_gravity'],
        mmw=mean_molar_mass,
    )

    # Comparison
    compare_from_reference_file(
        reference_file=reference_filenames['line_by_line_emission'],
        comparison_dict={
            'wavelength': petitRADTRANS.nat_cst.c / atmosphere_lbl.freq * 1e4,
            'spectral_radiosity': atmosphere_lbl.flux
        }
    )


def test_line_by_line_transmission_spectrum():
    # Calculate a transmission spectrum
    atmosphere_lbl.calc_transm(
        temp=temperature_iso,
        abunds=mass_fractions,
        gravity=planetary_parameters['surface_gravity'],
        mmw=mean_molar_mass,
        R_pl=planetary_parameters['radius'] * petitRADTRANS.nat_cst.r_jup_mean,
        P0_bar=planetary_parameters['reference_pressure']
    )

    # Comparison
    compare_from_reference_file(
        reference_file=reference_filenames['line_by_line_transmission'],
        comparison_dict={
            'wavelength': petitRADTRANS.nat_cst.c / atmosphere_lbl.freq * 1e4,
            'transit_radius': atmosphere_lbl.transm_rad / petitRADTRANS.nat_cst.r_jup_mean
        }
    )
