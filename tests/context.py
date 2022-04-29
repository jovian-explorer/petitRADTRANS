import os
import sys

import warnings

# Ensure that we are testing the package development files
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Initialize environment variable
path = os.environ.get("pRT_input_data_path")

if path is None:
    warnings.warn("system variable pRT_input_data_path was not defined, "
                  "setting it to the value hardcoded into tests/context.py\n"
                  "Change it if necessary (this will become unnecessary in a future update)")
    os.environ["pRT_input_data_path"] = r"/Users/molliere/Documents/programm_data/petitRADTRANS_public/input_data"

import petitRADTRANS
import petitRADTRANS.fort_rebin
import petitRADTRANS.nat_cst
import petitRADTRANS.poor_mans_nonequ_chem
import petitRADTRANS.radtrans
import petitRADTRANS.retrieval
import petitRADTRANS.version

# Future imports
# import petitRADTRANS.ccf.spectra_utils
# import petitRADTRANS.phoenix
# import petitRADTRANS.physics
