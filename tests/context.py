import os
import sys

# Ensure that we are testing the package development files
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Initialize environment variable
path = os.environ.get("pRT_input_data_path")

if path is None:
    os.environ["pRT_input_data_path"] = r"/home/dblain/petitradtrans/input_data"

import petitRADTRANS
#import petitRADTRANS.ccf.spectra_utils
import petitRADTRANS.fort_rebin
import petitRADTRANS.nat_cst
#import petitRADTRANS.phoenix
#import petitRADTRANS.physics
import petitRADTRANS.poor_mans_nonequ_chem
import petitRADTRANS.radtrans
import petitRADTRANS.retrieval
import petitRADTRANS.version
