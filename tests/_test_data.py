import unittest
from petitRADTRANS import Data
import petitRADTRANS.nat_cst as nc
from petitRADTRANS.retrieval.models import emission_model_diseq, isothermal_free_transmission

class TestData(unittest.TestCase):
    def test_init(self):
        self.test_fits_file = "../petitRADTRANS/retrieval/examples/emission/HR8799e_spectra.fits"        
        self.test_txt_file = "../petitRADTRANS/retrieval/examples/transmission/hst_example_clear_spec.fits"
        self.test_fits()
        self.test_txt()
        return 

    def test_fits(self):
        fits_test = Data("fits_test",
                         path_to_observations= self.test_fits_file,
                         distance = 40*nc.pc,
                         data_resolution = 500,
                         model_resolution = 500,
                         model_generating_function = emission_model_diseq,
                         scale = False)
        self.assertEqual(fits_test.path_to_observations, self.test_fits_file)
        self.assertEqual(fits_test.data_resolution,500)
        self.assertEqual(fits_test.model_resolution,500)
        self.assertEqual(fits_test.distance,40*nc.pc)
        self.assertEqual(fits_test.calculate_star_radiosity[0], 2.509834381986098e-17)
        self.assertEqual(fits_test.wlen[0],1.9700000286102295)

        fits_test.scale_to_distance(10*nc.pc)
        self.assertEqual(fits_test.distance,10*nc.pc)
        self.assertEqual(fits_test.calculate_star_radiosity[0], 2.509834381986098e-17 * (40. / 10.) ** 2)

        self.fits_test = fits_test
        return self.fits_test

    def test_txt(self):
        txt_test = Data("txt_test",
                         path_to_observations= self.test_txt_file,
                         distance = 40*nc.pc,
                         data_resolution = 50,
                         model_resolution = 50,
                         model_generating_function = isothermal_free_transmission,
                         scale = True)
        self.assertEqual(txt_test.path_to_observations, self.test_fits_file)
        self.assertEqual(txt_test.data_resolution,50)
        self.assertEqual(txt_test.model_resolution,50)
        self.assertEqual(txt_test.distance,40*nc.pc)
        self.assertEqual(txt_test.calculate_star_radiosity[0], 0.001365)
        self.assertEqual(txt_test.wlen[0],1.138000)
        self.assertEqual(txt_test.scale,True)

        txt_test.scale_to_distance(10*nc.pc)
        self.assertEqual(txt_test.distance,10*nc.pc)
        self.assertEqual(txt_test.calculate_star_radiosity[0], 0.001365 * (40. / 10.) ** 2)
        self.txt_test = txt_test
        return self.txt_test
