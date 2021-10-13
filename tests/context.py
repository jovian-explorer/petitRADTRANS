import os
import sys

# Ensure that we are testing the package development files
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import petitRADTRANS
import petitRADTRANS.radtrans
import petitRADTRANS.version
