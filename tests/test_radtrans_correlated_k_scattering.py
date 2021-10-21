"""
Test the radtrans module.
Essentially go through a simplified version of the tutorial, and compare the results with previous ones.
C.f. (https://petitradtrans.readthedocs.io/en/latest/content/notebooks/getting_started.html).

Do not change the parameters used to generate the comparison files, including input_data files, when running the tests.
"""
import copy

from .context import petitRADTRANS
from .utils import compare_from_reference_file, \
    reference_filenames, radtrans_parameters, temperature_guillot_2010, temperature_isothermal

relative_tolerance = 7.5e-3  # relative tolerance when comparing with older results


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


def test_correlated_k_emission_spectrum_cloud_calculated_radius_scattering(test_number=0, max_tests=10):
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
    try:
        compare_from_reference_file(
            reference_file=reference_filenames['correlated_k_emission_cloud_calculated_radius_scattering'],
            comparison_dict={
                'wavelength': petitRADTRANS.nat_cst.c / atmosphere_ck_scattering.freq * 1e4,
                'spectral_radiosity': atmosphere_ck_scattering.flux
            },
            relative_tolerance=relative_tolerance
        )
    except AssertionError as error_message:
        if test_number < max_tests:
            test_number += 1
            test_correlated_k_emission_spectrum_cloud_calculated_radius_scattering(test_number)
        else:
            raise AssertionError(
                f"scattering in petitRADTRANS is known to have an important relative error. "
                f"To take that into account, {max_tests} tests were performed, but all failed to reach a relative error"
                f" <= {relative_tolerance} compared to the results of the previous version.\n"
                f"Complete error message was: \n" +
                str(error_message)
            )
