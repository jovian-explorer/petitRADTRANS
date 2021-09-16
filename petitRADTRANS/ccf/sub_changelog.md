# Change Log
All notable changes to the CCF module will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com)
and this project adheres to [Semantic Versioning](http://semver.org).

This is a temporary document.

## [0.8.0] - 2021-09-16
### Added
- Option to select interesting regions for species.

### Changed
- Code clean-up.

## [0.7.2] - 2021-09-16
### Removed
- Useless limitation on number of transits.

## [0.7.1] - 2021-09-16
### Changed
- Better CCF analysis of multiple transits calculations.

## [0.7.0] - 2021-09-15
### Added
- CCF analysis of multiple transits.

## [0.6.1] - 2021-09-15
### Fixed
- Argument `include_species` when generating `SpectralModel` ignored when passing argument `mass_fraction`.
- Crash when all the detectors of an order have signal-to-noise ratios lower than 1.

## [0.6.0] - 2021-09-14
### Added
- Option to calculate emission and eclipse depth spectra in `SpectralModel`.

### Changed
- Code clean-up.

### Fixed
- Crash due to too large input velocity range.


## [0.5.1] - 2021-09-13
### Changed
- Mock observed spectra now use the median of the noise.

### Fixed
- Crashes due to empty mock observed spectra.

## [0.5.0] - 2021-09-13
### Changed
- SNR cutoff is now applied by masking instead of removing.
- SNR <= 1 are now masked, instead of SNR <= 0.
- Median is used instead of mean to remove the large scale trends of the spectra.
- When multiple observed spectra are used, the large scale trends are calculated for each one separately instead of using one correction for all spectra.
- Code clean-up.

### Fixed
- Handling of C2H2 in `SpectralModel`.

## [0.4.0] - 2021-09-10
### Changed
- Chemical table is now read in an HDF5 file, increasing loading speed.
- `poor_mans_nonequ_chem` module structure reorganized.

## [0.3.0] - 2021-09-10
### Added
- Option to make models with isothermal or custom temperature profiles.
- Option to make models with custom mass fractions.

### Changed
- Code clean-up.

### Fixed
- Some species names not handled correctly when initializing mass fractions in objects `SpectralModel`.

## [0.2.0] - 2021-09-09
### Added
- Option to make multiple mock observations with the same set of parameters.
- Output of CCF S/N error and of the CCF analysis results in function `get_tsm_snr_pcloud` .

### Changed
- CCF S/N is now based on the distribution (assumed to be gaussian) of multiple mock observations rather than one.

### Fixed
- `Planet` name incorrectly generated.
- `Planet` name loaded from HDF5 file as byte instead of str.


## [0.1.0] - 2021-09-07
### Added
- This file.
- Object `SimplePlanet` to generate planets with fewer attributes than `Planet`.
- Function `make_generic_planet` to easily generate "generic" planets.

### Changed
- Most functions in `_script.py` are moved to the `ccf_utils` module.
- Using `fort_rebin.rebin_spectrum` instead of `rebin_give_width.rebin_give_width` to re-bin the spectra, increasing both performance and accuracy.
- The noise of mock observations is determined more accurately.
- CCF is calculated detector-by-detector instead of order-by-order.
- Code clean-up.

### Fixed
- Object `SpectralModel` include only the default species list, despite explicitely setting its attribute `include_species`.
- Crash when initializing `SpectralModel` due to filename initialization.
- CCF analysis is performed multiple times on one model.

## [0.0.0] - 2021-09-01
First commit.