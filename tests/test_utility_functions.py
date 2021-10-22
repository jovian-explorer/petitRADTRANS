"""
Test petitRADTRANS utility functions.
"""
import numpy as np

from .context import petitRADTRANS

from .utils import compare_from_reference_file, \
    reference_filenames, radtrans_parameters, temperature_guillot_2010


relative_tolerance = 1e-6


def test_guillot_2010_temperature_profile():
    # Load expected results
    reference_data = np.load(reference_filenames['guillot_2010'])
    print(f"Comparing generated spectrum to result from petitRADTRANS-{reference_data['prt_version']}...")
    temperature_ref = reference_data['temperature']
    pressure_ref = reference_data['pressure']

    # Check if temperature is as expected
    assert np.allclose(
        radtrans_parameters['pressures'],
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
