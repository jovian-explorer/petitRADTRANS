"""Test the radtrans module in line-by-line mode with sampling.

Do not change the parameters used to generate the comparison files, including input_data files, when running the tests.
"""
from .context import petitRADTRANS
from .utils import compare_from_reference_file, \
    reference_filenames, radtrans_parameters, temperature_guillot_2010, temperature_isothermal

relative_tolerance = 1e-6  # relative tolerance when comparing with older results


# Initializations
def init_radtrans_downsampled_line_by_line():
    atmosphere = petitRADTRANS.radtrans.Radtrans(
        line_species=radtrans_parameters['spectrum_parameters']['line_species_line_by_line'],
        rayleigh_species=radtrans_parameters['spectrum_parameters']['rayleigh_species'],
        continuum_opacities=radtrans_parameters['spectrum_parameters']['continuum_opacities'],
        wlen_bords_micron=radtrans_parameters['spectrum_parameters']['wavelength_range_line_by_line'],
        lbl_opacity_sampling=radtrans_parameters['spectrum_parameters']['line_by_line_opacity_sampling'],
        mode='lbl'
    )

    atmosphere.setup_opa_structure(radtrans_parameters['pressures'])

    return atmosphere


atmosphere_lbl_downsampled = init_radtrans_downsampled_line_by_line()


def test_line_by_line_downsampled_emission_spectrum():
    # Calculate an emission spectrum
    atmosphere_lbl_downsampled.calc_flux(
        temp=temperature_guillot_2010,
        abunds=radtrans_parameters['mass_fractions'],
        gravity=radtrans_parameters['planetary_parameters']['surface_gravity'],
        mmw=radtrans_parameters['mean_molar_mass'],
    )

    # Comparison
    compare_from_reference_file(
        reference_file=reference_filenames['line_by_line_downsampled_emission'],
        comparison_dict={
            'wavelength': petitRADTRANS.nat_cst.c / atmosphere_lbl_downsampled.freq * 1e4,
            'spectral_radiosity': atmosphere_lbl_downsampled.flux
        },
        relative_tolerance=relative_tolerance
    )


def test_line_by_line_downsampled_transmission_spectrum():
    # Calculate a transmission spectrum
    atmosphere_lbl_downsampled.calc_transm(
        temp=temperature_isothermal,
        abunds=radtrans_parameters['mass_fractions'],
        gravity=radtrans_parameters['planetary_parameters']['surface_gravity'],
        mmw=radtrans_parameters['mean_molar_mass'],
        R_pl=radtrans_parameters['planetary_parameters']['radius'] * petitRADTRANS.nat_cst.r_jup_mean,
        P0_bar=radtrans_parameters['planetary_parameters']['reference_pressure']
    )

    # Comparison
    compare_from_reference_file(
        reference_file=reference_filenames['line_by_line_downsampled_transmission'],
        comparison_dict={
            'wavelength': petitRADTRANS.nat_cst.c / atmosphere_lbl_downsampled.freq * 1e4,
            'transit_radius': atmosphere_lbl_downsampled.transm_rad / petitRADTRANS.nat_cst.r_jup_mean
        },
        relative_tolerance=relative_tolerance
    )
