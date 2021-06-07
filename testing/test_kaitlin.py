import pylab as plt

from petitRADTRANS import Radtrans
from petitRADTRANS import nat_cst as nc

atmosphere = Radtrans(line_species = ['H2O'])
atmosphere.get_star_spectrum(5000, 3, 1)
plt.plot(nc.c/atmosphere.freq/1e-4, atmosphere.stellar_intensity)

atmosphere = Radtrans(line_species = ['H2O_R_10'])
atmosphere.get_star_spectrum(5000, 3, 1)
plt.plot(nc.c/atmosphere.freq/1e-4, atmosphere.stellar_intensity)

plt.yscale('log')
plt.xscale('log')
plt.show()

"""

    def get_star_spectrum(self, Tstar, distance, Rstar):
        '''Method to get the PHOENIX spectrum of the star and rebin it
        to the wavelength points. If Tstar is not explicitly written, the
        spectrum will be 0. If the distance is not explicitly written,
        the code will raise an error and break to urge the user to
        specify the value.

            Args:
                Tstar (float):
                    the stellar temperature in K.
                distance (float):
                    the semi-major axis of the planet in AU.
                Radius (float):
                    if specified, uses this radius in Solar radii
                    to scale the flux, otherwise it uses PHOENIX radius.
        '''

        if Tstar != None:

            spec,rad = nc.get_PHOENIX_spec_rad(Tstar)
            if not Rstar == None:
                 print('Using Rstar value input by user.')
                 rad = Rstar


            add_stellar_flux = np.zeros(100)
            add_wavelengths = np.logspace(np.log10(1.0000002e-02), 2, 100)

            #import pdb
            #pdb.set_trace()

            interpwavelengths = np.append(spec[:,0], add_wavelengths)
            interpfluxes      = np.append(spec[:, 1], add_stellar_flux)

            '''
            self.stellar_intensity = fr.rebin_spectrum(spec[:,0],
                                                       spec[:,1],
                                                       nc.c/self.freq)
            '''

            self.stellar_intensity = fr.rebin_spectrum(interpwavelengths,
                                                       interpfluxes,
                                                       nc.c / self.freq)

            try:
                ###### SCALED INTENSITY (Flux/pi)
                self.stellar_intensity = self.stellar_intensity/ np.pi * \
                (rad/distance)**2
            except TypeError  as e:
                str='********************************'+\
                ' Error! Please set the semi-major axis or turn off the calculation '+\
                'of the stellar spectrum by removing Tstar. ********************************'
                raise Exception(str) from e
        else:

            self.stellar_intensity = np.zeros_like(self.freq)

"""