.. petitRADTRANS documentation master file, created by
   sphinx-quickstart on Tue Jan 15 15:07:15 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

petitRADTRANS documentation
=========================================

Welcome to the **petitRADTRANS** (pRT) documentation. pRT is a
Python package for calculating transmission and emission spectra
of exoplanets for clear and cloudy planets. pRT also incorporates (**new!**)
an easy subpackage for running retrievals with nested sampling.

**To get started with some examples on how to run pRT immediately,
see** `"Getting started" <content/notebooks/getting_started.html>`_. **Otherwise read on for some more general info,
also on how to best run retrievals.**

pRT has two different opacity treatment modes. The low resolution mode runs calculations
at :math:`\lambda/\Delta\lambda\leq 1000` using the so-called correlated-k treatment for opacities.
The high resolution mode runs calculations at :math:`\lambda/\Delta\lambda\leq 10^6`, using a line-by-line opacity treatment.

pRT's low-resolution opacities are initially stored at :math:`\lambda/\Delta\lambda = 1000` but can be rebinned to lower
resolution by the user, for which we make use of the `Exo_k <https://pypi.org/project/exo-k/>`_ package.
This is explained `here <content/notebooks/Rebinning_opacities.html>`_.

The high-resolution opacities are initially stored at :math:`\lambda/\Delta\lambda = 10^6`, and example calculations are shown
`here <content/notebooks/highres.html>`_. Opacities can also be undersampled, to run at a lower resolution, and to speed up
spectral calculations. The user should verify whether this leads to solutions which are identical to the rebinned results of the fiducial
:math:`\lambda/\Delta\lambda = 10^6` resolution. Undersampling is done with the ``lbl_opacity_sampling`` parameter
described in the `API <autoapi/petitRADTRANS/radtrans/index.html#petitRADTRANS.radtrans.Radtrans>`_ here. An example is
given `here <content/notebooks/Rebinning_opacities.html>`_.

pRT can calculate transmission and emission spectra of exoplanets, for clear or cloudy atmospheres. The different cloud treatments
(gray clouds, power law clouds, "real" clouds using optical constants) are described in the tutorial `here <content/notebooks/clouds.html>`_.
Scattering is included in pRT, but must be specifically turned on for emission spectra (note that scattering increases the runtime),
see `Scattering for Emission Spectra <content/notebooks/emis_scat.html>`_. pRT can also calculate the reflection of light
at the surface of rocky planets, for which the user can specify wavelength-dependent albedos and emissivities. This is likewise
explained in `Scattering for Emission Spectra <content/notebooks/emis_scat.html>`_.

The newly added retrieval subpackage is documented `here <content/notebooks/pRT_Retrieval_Example.html>`_.
At the moment pRT retrievals are making use of the `PyMultiNest <https://johannesbuchner.github.io/PyMultiNest/>`_
package for parameter inference. Of course you are free to use pRT spectral synthesis routines with any other inference tool of your liking.
For this you will have to setup your own retrieval framework, however (you can modify / check our source code for inspiration).

petitRADTRANS is available under the MIT License, and documented in
`Mollière et al. (2019) <https://arxiv.org/abs/1904.11504>`_, for the general code, and `Mollière et al. (2020) <https://arxiv.org/abs/2006.09394>`_, Alei et al. (in prep.), for the scattering implementation. Please cite these papers if you make use of petitRADTRANS in your work.

.. _contact: molliere@mpia.de

This documentation webpage contains an `installation guide <content/installation.html>`_, a
`tutorial <content/tutorial.html>`_, an `API documentation <autoapi/index.html>`_.
We also give a `list of easy-to-use resources on how to include
opacities <content/opa_add.html>`_ that may be missing from `our database <content/available_opacities.html>`_.
For the easiest cases this may correspond to simply dropping a file
into the pRT opacity folder.

Developers
___________

- Paul Mollière
- Evert Nasedkin
- Doriann Blain
- Eleonora Alei
- Tomas Stolker

Contributors
________________
- Karan Molaverdikhani
- Mantas Zilinskas

.. toctree::
   :maxdepth: 2
   :caption: Guide:

   content/installation
   content/tutorial
   content/available_opacities
   content/retrieval_examples
   content/opa_add

.. toctree::
   :maxdepth: 2
   :caption: Code documentation:

   content/nat_cst_doc
