"""
Test the radtrans module.
"""
import os
import numpy as np
from .context import petitRADTRANS


def init_parameters(pressures):
    # Initialize temperature and mass fractions
    temperature = 1200. * np.ones_like(pressures)

    mass_fractions = {
        'H2': 0.74 * np.ones_like(temperature),
        'He': 0.24 * np.ones_like(temperature),
        'H2O_HITEMP': 0.001 * np.ones_like(temperature),
        'CO_all_iso_HITEMP': 0.1 * np.ones_like(temperature)
    }

    mean_molar_mass = 2.33 * np.ones_like(temperature)  # (g.cm-3)

    # Initialize planetary parameters
    planet_reference_pressure = 0.01  # bar
    planet_radius = 1.838 * petitRADTRANS.nat_cst.r_jup_mean  # (cm) radius at reference pressure
    planet_surface_gravity = 1e1 ** 2.45  # (cm.s-2) gravity at reference pressure

    return temperature, mass_fractions, mean_molar_mass, \
        planet_reference_pressure, planet_radius, planet_surface_gravity


def main_radtrans_init_with_correlated_k():
    # Preparing a radiative transfer object
    pressures = np.logspace(-6, 2, 27)
    atmosphere = radtrans_init_correlated_k(pressures)

    # Calculating a transmission spectrum
    temperature, mass_fractions, mean_molar_mass, planet_reference_pressure, planet_radius, planet_surface_gravity = \
        init_parameters(pressures)

    atmosphere.calc_transm(
        temp=temperature,
        abunds=mass_fractions,
        gravity=planet_surface_gravity,
        mmw=mean_molar_mass,
        R_pl=planet_radius,
        P0_bar=planet_reference_pressure
    )

    return atmosphere


def radtrans_init_correlated_k(pressures):
    atmosphere = petitRADTRANS.radtrans.Radtrans(
        line_species=[
            'H2O_HITEMP',
            'CO_all_iso_HITEMP'
        ],
        rayleigh_species=['H2', 'He'],
        continuum_opacities=['H2-H2', 'H2-He'],
        wlen_bords_micron=[4.3, 5.0],
        mode='c-k'
    )

    atmosphere.setup_opa_structure(pressures)

    return atmosphere


def test_radtrans_init_with_correlated_k():
    """
    Go through a simplified version of the tutorial.
    C.f. (https://petitradtrans.readthedocs.io/en/latest/content/notebooks/getting_started.html).
    """
    atmosphere = main_radtrans_init_with_correlated_k()

    # Load expected results
    reference_data = np.loadtxt(os.path.join(
        os.path.dirname(__file__), 'data', 'radtrans_spectrum_correlated_k_ref.dat')
    )
    wavelength_ref = reference_data[:, 0]
    transit_radius_ref = reference_data[:, 1]

    # Check if spectrum is as expected
    assert np.allclose(
        petitRADTRANS.nat_cst.c / atmosphere.freq * 1e4,
        wavelength_ref
    )

    assert np.allclose(
        atmosphere.transm_rad / petitRADTRANS.nat_cst.r_jup_mean,
        transit_radius_ref
    )
