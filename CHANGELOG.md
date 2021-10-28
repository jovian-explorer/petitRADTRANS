# Change Log
All notable changes to the CCF module will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com)
and this project adheres to [Semantic Versioning](http://semver.org).

## [2.4.0-a7] - 2021-10-28
### Added
- Module `phoenix` for PHOENIX stellar models.
- Module `physics` to store useful physical functions.
- Module `utils` to store generic useful functions.
- Function `get_guillot_2010_temperature_profile`, a more general Guillot 2010 temperature profile.
- Test suite.
- Module `configuration` to manage paths.
- Module `version` to store petitRADTRANS version number.

### Changed
- Input data path is now stored in a config file within the folder \<HOME\>/.petitRADTRANS, generated when installing the package or using it for the first time.
- Object `Radtrans` is now imported using `from petitRADTRANS.radtrans import Radtrans` (was `from petitRADTRANS import Radtrans`) for more stable installation.
- Some functions have moved from the module `nat_cst` to another, more specific module.
- Package structure.
- Running mean now uses the faster `scipy.ndimage.filters.uniform_filter1d` implementation.
- Tutorial updated.
- Code clean-up.

### Removed
- Useless make files.

### Fixed
- Hansen cloud particle distribution returning NaN if `b_hansen` set too low.

### Issues
- Retrieval not tested with new organization.

---
No changelog before version 2.4.0.
