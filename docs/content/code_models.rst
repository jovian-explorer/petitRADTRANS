Models module
=============

This module contains a set of functions that generate the spectra used 
in the petitRADTRANS retrieval. This includes setting up the
pressure-temperature structure, the chemistry, and the radiative
transfer to compute the emission or transmission spectrum.

All models must take the same set of inputs:

pRT_object : petitRADTRANS.RadTrans
    This is the pRT object that is used to compute the spectrum
    It must be fully initialized prior to be used in the model function
parameters : dict
    A dictionary of Parameter objects. The naming of the parameters
    must be consistent between the Priors and the model function you
    are using.
PT_plot_mode : bool
    If this argument is True, the model function should return the pressure 
    and temperature arrays before computing the flux.
AMR : bool
    If this parameter is True, your model should allow for reshaping of the 
    pressure and temperature arrays based on the position of the clouds or
    the location of the photosphere, increasing the resolution where required.
    For example, using the fixed_length_amr function defined below.

.. automodule:: petitRADTRANS.retrieval.models
    :members: