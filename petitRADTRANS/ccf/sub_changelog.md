# Change Log
All notable changes to the CCF module will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com)
and this project adheres to [Semantic Versioning](http://semver.org).

This is a temporary document.

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