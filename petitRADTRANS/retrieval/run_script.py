###########################################
# Input / output, general run definitions
###########################################
import sys, os
# To not have numpy start parallelizing on its own
os.environ["OMP_NUM_THREADS"] = "1"

# Read external packages
import numpy as np
import copy as cp
import matplotlib
matplotlib.use('Agg') # set the backend before importing pyplot

import pymultinest
import json
import argparse as ap
import matplotlib.pyplot as plt
# Read own packages
from petitRADTRANS import Radtrans
from petitRADTRANS import nat_cst as nc
from .run_definition_hst import RunDefinition as rd
from retrieval import Retrieval


retrieval = Retrieval(rd,
                      output_dir = "",
                      sample_spec = True)
retrieval.run()
retrieval.plotAll()


