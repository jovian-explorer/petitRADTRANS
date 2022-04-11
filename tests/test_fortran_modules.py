"""Test the fortran modules.
"""
import numpy as np

from .context import petitRADTRANS


def test_rebin_spectrum():
    input_wavelengths = np.linspace(0.85, 2.15, 20)
    input_flux = np.sin(3 * input_wavelengths) + np.sin(10 * input_wavelengths)
    output_wavelengths = np.linspace(1, 2, 5)

    output_flux = petitRADTRANS.fort_rebin.rebin_spectrum(
            input_wavelengths, input_flux, output_wavelengths
    )

    assert np.allclose(output_flux, np.array([-0.26055379, -0.60538758, -0.47629003, -1.54747344,  0.3938092]))
