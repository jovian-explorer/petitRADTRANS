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

    atmosphere.setup_opa_structure(radtrans_parameters['pressures_thin_atmosphere'])

    return atmosphere


atmosphere_ck_surface_scattering = init_radtrans_correlated_k()


def test_correlated_k_emission_spectrum_surface_scattering():
    # Copy atmosphere so that change in reflectance is not carried outside the function
    atmosphere = copy.deepcopy(atmosphere_ck_surface_scattering)

    atmosphere.reflectance = radtrans_parameters['planetary_parameters']['surface_reflectance'] * \
        np.ones_like(atmosphere.freq)

    atmosphere.calc_flux(
        temp=temperature_guillot_2010,
        abunds=radtrans_parameters['mass_fractions'],
        gravity=radtrans_parameters['planetary_parameters']['surface_gravity'],
        mmw=radtrans_parameters['mean_molar_mass'],
        geometry='non-isotropic',
        Tstar=radtrans_parameters['stellar_parameters']['effective_temperature'],
        Rstar=radtrans_parameters['stellar_parameters']['radius'] * petitRADTRANS.nat_cst.r_sun,
        semimajoraxis=radtrans_parameters['planetary_parameters']['orbit_semi_major_axis'],
        theta_star=radtrans_parameters['stellar_parameters']['incidence_angle']
    )

    # Comparison
    compare_from_reference_file(
        reference_file=reference_filenames[
            'correlated_k_emission_surface_scattering'
        ],
        comparison_dict={
            'wavelength': petitRADTRANS.nat_cst.c / atmosphere.freq * 1e4,
            'spectral_radiosity': atmosphere.flux
        },
        relative_tolerance=relative_tolerance
    )
