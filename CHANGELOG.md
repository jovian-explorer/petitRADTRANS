# Change Log
All notable changes to the CCF module will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com)
and this project adheres to [Semantic Versioning](http://semver.org).

## [2.4.0-a15] - 2021-04-01
### Added
- High-resolution retrievals.
- Possibility to initialize a `retrieval.Data` class without needing a file.
- Possibility to initialize a `retrieval.Data` class with a `Radtrans` object, without the need to re-create one.
- Possibility to initialize a `retrieval.Retrieval` class with a stellar spectrum, without the need to recalculate it.
- Support for 2D and 3D spectral array in retrievals.
- Static function `retrieval.Retrieval._get_samples`, to get retrieval outputs without the need to initialize a `Retrieval`.
- Gibson et al. 2021 log-likelihood calculation.
- Better high-resolution mock observation function.
- Module `phoenix` for PHOENIX stellar models.
- Module `physics` to store useful physical functions.
- Module `utils` to store generic useful functions.
- Module `ccf.pipeline` to reduce ground-based high-resolution data.
- Class `SpectralModel` as a wrapper to `Radtrans` objects.
- Function `get_guillot_2010_temperature_profile`, a more general Guillot 2010 temperature profile.
- Function to calculate the ESM of a planet.
- Function to calculate the orbital phase of a planet.
- Function to calculate the radial velocity of a planet.
- Function to calculate the orbital velocity of a planet.
- Function to calculate the Doppler shift.
- Function to convolve, Doppler shift, and rebin a spectrum.
- Noise estimation for eclipse spectra in addition to transit spectra.
- Method to generate a `Planet` using a NASA Exoplanet Archive tab file.
- Function to calculate the radius of a planet from its surface gravity and mass.
- Option to use the NASA Exoplanet Archive "best mass" when generating a `Planet`.
- Test suite.
- Module `configuration` to manage paths.
- Module `version` to store petitRADTRANS version number.
- Message when loading the `poor_mans_nonequ_chem` chemical equilibrium mass mixing ratios table.

### Changed
- Input data path is now stored in a config file within the folder \<HOME\>/.petitRADTRANS, generated when installing the package or using it for the first time.
- Object `Radtrans` is now imported using `from petitRADTRANS.radtrans import Radtrans` (was `from petitRADTRANS import Radtrans`) for more stable installation.
- Some functions have moved from the module `nat_cst` to another, more specific module.
- Package structure.
- Running mean now uses the faster `scipy.ndimage.filters.uniform_filter1d` implementation.
- Tutorial updated.
- Character limit in retrieval output directory increased from 100 to 200.
- Code clean-up.

### Removed
- Useless make files.

### Fixed
- Hansen cloud particle distribution returning NaN if `b_hansen` set too low.
- Retrieval not converging when using correlated-k.
- Function `contour_corner` not working when not giving optional arguments `parameter_ranges` and `parameter_plot_indices`.
- True values not plotted in function `contour_corner`.
- Function `get_MM` not working with e- and H-.
- Wrong docstrings in function `Radtrans.get_star_spectrum`.

---
No changelog before version 2.4.0.
