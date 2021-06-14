import unittest
import numpy as np

import petitRADTRANS.nat_cst as nc
from petitRADTRANS import Radtrans, Parameter
from petitRADTRANS.retrieval.models import emission_model_diseq, guillot_free_emission, guillot_eqchem_transmission, isothermal_eq_transmission, isothermal_free_transmission

from test_data import TestData

class TestModels(unittest.TestCase):
    def test_init(self):
        raise NotImplementedError

        td = TestData()
        self.emtest = TestData.test_fits()
        self.txttest = TestData.test_txt()

        self.pressures = np.logspace(-6,3,10)
        self.em_obj = Radtrans(line_species = ['H2O,CO'],
                               rayleigh_species= ['H2', 'He'],
                               continuum_opacities = ['H2-H2', 'H2-He'],
                               cloud_species = ['MgSiO3(c)_cd'],
                               mode='c-k',
                               wlen_bords_micron = self.emtest.wlen_range_pRT,
                               do_scat_emis = True)
        self.em_obj.setup_opa_structure(self.pressures)
        self.trans_obj = Radtrans()
        return 
        
    def setup_parameters(self):
        self.parameters = {}

    def test_em_diseq(self):
        self.emtest.model_generating_function = emission_model_diseq
        wlen, spec = self.emtest.model_generating_function(self.em_obj, self.parameters, PT_plot_mode= False, AMR=False)

    def test_amr(self):
        pressures = np.logspace(-6,3,127)
        self.parameters['pressure_simple'] = Parameter('pressure_simple',False,value=100)
        self.parameters['pressure_width'] = Parameter('pressure_width',False,value=3)
        self.parameters['pressure_scaling'] = Parameter('pressure_scaling',False,value=10)

        amr_obj = Radtrans(line_species = ['H2O,CO'],
                               rayleigh_species= ['H2', 'He'],
                               continuum_opacities = ['H2-H2', 'H2-He'],
                               cloud_species = ['MgSiO3(c)_cd'],
                               mode='c-k',
                               wlen_bords_micron = self.emtest.wlen_range_pRT,
                               do_scat_emis = True)
        amr_obj.setup_opa_structure(pressures)
        wlen, spec = self.emtest.model_generating_function(amr_obj, self.parameters, PT_plot_mode= False, AMR=True)


