from __future__ import division, print_function

import copy as cp
import glob

import h5py
import numpy as np

from petitRADTRANS import fort_input as fi
from petitRADTRANS import nat_cst as nc
from petitRADTRANS import pyth_input as pyi


class ReadOpacities:
    def read_line_opacities(self, index, arr_min, arr_max):
        # Reads in the line opacities for spectral calculation

        # First get the P-Ts position where the grid is defined.
        # This here is for the nominal, log-uniform 10 x 13 point
        # P-T grid.
        buffer = np.genfromtxt(self.path+'/opa_input_files/opa_PT_grid.dat')
        self.line_TP_grid = np.zeros_like(buffer)
        self.line_TP_grid[:,0] = buffer[:,1]
        self.line_TP_grid[:,1] = buffer[:,0]
        # Convert from bars to cgs
        self.line_TP_grid[:,1] = 1e6*self.line_TP_grid[:,1]
        self.line_TP_grid = np.array(self.line_TP_grid.reshape( \
                    len(self.line_TP_grid[:,1]),2),dtype='d',order='F')

        # Check if species has custom P-T grid and reads in this grid.
        # Grid must be sorted appropriately, but the pyi.get_custom_grid()
        # will do that for the user in case a randomly ordered PTpaths.ls
        # is specified by the user in the opacity folder of the relevant species.
        # Only condition: it needs to be rectangular.
        # Because it is easier, a custom grid is saved for every species,
        # also the ones that use the nominal P-T grid of petitRADTRANS

        self.custom_grid = {}

        if len(self.line_species) > 0:
            self.custom_line_TP_grid = {}
            self.custom_line_paths = {}
            self.custom_diffTs, self.custom_diffPs = {}, {}

            for i_spec in range(len(self.line_species)):
                # Check if it is an Exomol hdf5 file that needs to be read:
                Chubb = False
                if self.mode == 'c-k':
                     path_opa = self.path+'/opacities/lines/corr_k/'+self.line_species[i_spec]
                     if glob.glob(path_opa+'/*.h5') != []:
                         Chubb = True

                 # If not Exomol k-table made by Katy Chubb
                if not Chubb:


                    # Check and sort custom grid for species, if defined.
                    custom_grid_data = \
                      pyi.get_custom_PT_grid(self.path, \
                                             self.mode, \
                                             self.line_species[i_spec])

                    # If no custom grid was specified (no PTpaths.ls found):
                    # take nominal grid. This assumes that the files indeed
                    # are following the nominal grid and naming convention.
                    # Otherwise it will take the info provided in PTpaths.ls
                    # which was filled into custom_grid_data.
                    if custom_grid_data == None:
                        self.custom_line_TP_grid[self.line_species[i_spec]] = \
                          self.line_TP_grid
                        self.custom_line_paths[self.line_species[i_spec]] = None
                        self.custom_diffTs[self.line_species[i_spec]], \
                          self.custom_diffPs[self.line_species[i_spec]] = 13, 10
                        self.custom_grid[self.line_species[i_spec]] = False
                    else:
                        self.custom_line_TP_grid[self.line_species[i_spec]] = \
                          custom_grid_data[0]
                        self.custom_line_paths[self.line_species[i_spec]] = \
                          custom_grid_data[1]
                        self.custom_diffTs[self.line_species[i_spec]], \
                          self.custom_diffPs[self.line_species[i_spec]] = \
                          custom_grid_data[2], \
                          custom_grid_data[3]
                        self.custom_grid[self.line_species[i_spec]] = True
                else:
                    # If the user wants to make use of an Exomol k-table.
                    # In this case the custom grid is defined by reading
                    # the grid coordinates from the Exomol hdf5 file.
                    path_opa = self.path+'/opacities/lines/corr_k/'+self.line_species[i_spec]
                    file_path_hdf5 = glob.glob(path_opa+'/*.h5')[0]
                    f = h5py.File(file_path_hdf5,'r')

                    lent = len(f['t'][:])
                    lenp = len(f['p'][:])
                    retVal = np.zeros(lent*lenp*2).reshape(lent*lenp,2)
                    for i_t in range(lent):
                        for i_p in range(lenp):
                                                     # convert from bar to cgs
                            retVal[i_t*lenp+i_p, 1] = f['p'][i_p]*1e6
                            retVal[i_t*lenp+i_p, 0] = f['t'][i_t]
                    self.custom_line_TP_grid[self.line_species[i_spec]] = retVal
                    self.custom_diffTs[self.line_species[i_spec]], \
                      self.custom_diffPs[self.line_species[i_spec]] = \
                          lent, \
                          lenp
                    self.custom_grid[self.line_species[i_spec]] = True
                    f.close()

        # Read actual opacities....
        # The nominal petitRADTRANS opacity grid "line_grid_kappas"
        # has the shape g_len,freq_len,len(line_species),len(line_TP_grid[:,0])
        # line_grid_kappas_custom_PT's entries have the shape
        # g_len,freq_len,len(self.custom_line_TP_grid[self.line_species[i_spec]])
        # From now on also the nominal grid opacities are read into
        # line_grid_kappas_custom_PT, because this makes things easier.
        self.line_grid_kappas_custom_PT = {}

        if len(self.line_species) > 0:

            tot_str = ''
            for sstring in self.line_species:
                tot_str = tot_str + sstring + ':'

            custom_file_names = ''

            for i_spec in range(len(self.line_species)):

                # Read in opacities in the petitRADTRANS format, either
                # in pRT P-T grid spacing or custom P-T grid spacing.

                # Check if it is an Exomol hdf5 file that needs to be read:
                Chubb = False
                if self.mode == 'c-k':
                    path_opa = self.path+'/opacities/lines/corr_k/'+self.line_species[i_spec]
                    if glob.glob(path_opa+'/*.h5') != []:
                        Chubb = True

                if not Chubb:

                    if not self.custom_grid[self.line_species[i_spec]]:
                        len_TP = len(self.line_TP_grid[:,0])
                    else:
                        len_TP = len(self.custom_line_TP_grid[ \
                                self.line_species[i_spec]][:,0])

                    custom_file_names = ''
                    if self.custom_grid[self.line_species[i_spec]]:
                        for i_TP in range(len_TP):
                            custom_file_names = custom_file_names + \
                                self.custom_line_paths[self.line_species[i_spec]][i_TP] \
                                + ':'

                    #######
                    ####### TODO PAUL EXO_K Project: do index_fill treatment from below!
                    ####### Also needs to read "custom" freq_len and freq here again!
                    ####### FOR ALL TABLES, HERE AND CHUBB: TEST THAT THE GRID IS INDEED THE SAME AS REQUIRED IN THE REGIONS
                    ####### WITH OPACITY. NEXT STEPS AFTER THIS: (i) MAKE ISOLATED EXO_K rebinning test, (ii) Do external bin down script, save as pRT method
                    ####### (iii) enable on the fly down-binning, (iv) look into outsourcing methods from class to separate files, this here is getting too long!
                    #######

                    if self.mode == 'c-k':
                        local_freq_len, local_g_len = fi.get_freq_len(self.path, self.line_species[i_spec])
                        local_freq_len_full = cp.copy(local_freq_len)
                        # Read in the frequency range of the opcity data
                        local_freq, local_border_freqs = fi.get_freq(self.path, \
                                                                   self.line_species[i_spec], \
                                                                   local_freq_len)
                    else:
                        if self.lbl_opacity_sampling is None:
                            local_freq_len_full = self.freq_len
                        else:
                            local_freq_len_full = self.freq_len * self.lbl_opacity_sampling

                        local_g_len = self.g_len

                    self.line_grid_kappas_custom_PT[self.line_species[i_spec]] = \
                      fi.read_in_molecular_opacities( \
                        self.path, \
                        self.line_species[i_spec]+':', \
                        local_freq_len_full, \
                        local_g_len, \
                        1, \
                        len_TP, \
                        self.mode, \
                        arr_min, \
                        arr_max, \
                        self.custom_grid[self.line_species[i_spec]], \
                        custom_file_names)

                    if self.mode == 'c-k':
                        # Initialize an empty array that has the same spectral entries as
                        # pRT object has nominally. Only fill those values where the k-tables
                        # have entries.
                        retVal = np.zeros(self.g_len* self.freq_len_full*len_TP).reshape( \
                                              self.g_len, self.freq_len_full, 1, \
                                              len_TP)

                        # Indices in retVal to be filled with read-in opacities
                        index_fill = (self.freq_full <= local_freq[0]*(1.+1e-10)) & \
                          (self.freq_full >= local_freq[-1]*(1.-1e-10))
                        # Indices of read-in opacities to be filled into retVal
                        index_use = (local_freq <= self.freq_full[0]*(1.+1e-10)) & \
                                    (local_freq >= self.freq_full[-1]*(1.-1e-10))

                        #print(np.shape(retVal[:, index_fill, 0, :]))
                        #print(np.shape(self.line_grid_kappas_custom_PT[self.line_species[i_spec]][:,:,0,:]))
                        #print(np.shape(self.line_grid_kappas_custom_PT[self.line_species[i_spec]][:, index_use, 0, :]))
                        #import pdb
                        #pdb.set_trace()

                        retVal[:, index_fill, 0, :] = \
                            self.line_grid_kappas_custom_PT[self.line_species[i_spec]][:,index_use,0,:]
                        self.line_grid_kappas_custom_PT[self.line_species[i_spec]] = retVal

                    # Down-sample opacities in lbl mode if requested
                    if self.mode == 'lbl' and self.lbl_opacity_sampling is not None:
                        self.line_grid_kappas_custom_PT[self.line_species[i_spec]] = \
                            self.line_grid_kappas_custom_PT[
                                self.line_species[i_spec]][:, ::self.lbl_opacity_sampling, :]

                # Read in the Exomol k-table by Katy Chubb if requested by the user
                else:
                    print('  Read line opacities of '+self.line_species[i_spec]+'...')

                    path_opa = self.path+'/opacities/lines/corr_k/'+self.line_species[i_spec]
                    file_path_hdf5 = glob.glob(path_opa+'/*.h5')[0]
                    f = h5py.File(file_path_hdf5,'r')


                    lenf = len(f['bin_centers'][:])
                    freqs_chubb = nc.c*f['bin_centers'][:][::-1]
                    lent = len(f['t'][:])
                    lenp = len(f['p'][:])

                    # Some swapaxes magic is required because the tables are sorted
                    # differently when coming from the Exomol website.
                    k_table = np.array(f['kcoeff'])
                    k_table = np.swapaxes(k_table, 0, 1)
                    k_table2 = k_table.reshape(lenp*lent, lenf, 16)
                    k_table2 = np.swapaxes(k_table2, 0, 2)
                    k_table2 = k_table2[:,::-1,:]

                    # Initialize an empty array that has the same spectral entries as
                    # pRT object has nominally. Only fill those values where the Exomol tables
                    # have entries.
                    retVal = np.zeros(self.g_len* self.freq_len_full* \
                       len(self.custom_line_TP_grid[self.line_species[i_spec]])).reshape( \
                                          self.g_len, self.freq_len_full, 1, \
                                          len(self.custom_line_TP_grid[self.line_species[i_spec]]))
                    index_fill = (self.freq_full <= freqs_chubb[0]*(1.+1e-10)) & \
                      (self.freq_full >= freqs_chubb[-1]*(1.-1e-10))
                    index_use = (freqs_chubb <= self.freq_full[0] * (1. + 1e-10)) & \
                                (freqs_chubb >= self.freq_full[-1] * (1. - 1e-10))
                    retVal[:, index_fill, 0, :] = k_table2[:, index_use, :]

                    # Divide by mass to go from cross-sections to opacities, the latter
                    # is what pRT requires.
                    exomol_mass = float(f['mol_mass'][0])
                    self.line_grid_kappas_custom_PT[self.line_species[i_spec]] = retVal/exomol_mass/nc.amu
                    print(' Done.')

                    f.close()

                # Cut the wavelength range of the just-read species to the wavelength range
                # requested by the user
                if self.mode == 'c-k':
                    self.line_grid_kappas_custom_PT[self.line_species[i_spec]] = \
                      np.array(self.line_grid_kappas_custom_PT[ \
                        self.line_species[i_spec]][:,index,0,:], \
                                 dtype='d',order='F')
                else:
                    self.line_grid_kappas_custom_PT[self.line_species[i_spec]] = \
                    np.array(self.line_grid_kappas_custom_PT[ \
                        self.line_species[i_spec]][:,:,0,:], \
                                 dtype='d',order='F')

            print()

        # Read in g grid for correlated-k
        if self.mode == 'c-k':
            buffer = np.genfromtxt(self.path+'/opa_input_files/g_comb_grid.dat')
            self.g_gauss, self.w_gauss = buffer[:,0], buffer[:,1]
            self.g_gauss,self.w_gauss = np.array(self.g_gauss,dtype='d', \
                order='F'),np.array(self.w_gauss, \
                dtype='d',order='F')
        elif self.mode == 'lbl':
            self.g_gauss, self.w_gauss = np.ones(1), np.ones(1)
            self.g_gauss,self.w_gauss = np.array(self.g_gauss,dtype='d', \
                order='F'),np.array(self.w_gauss, \
                dtype='d',order='F')

    def read_cloud_opas(self):
        # Function to read cloud opacities
        self.cloud_species_mode = []
        for i in range(int(len(self.cloud_species))):
            splitstr = self.cloud_species[i].split('_')
            self.cloud_species_mode.append(splitstr[1])
            self.cloud_species[i] = splitstr[0]

        # Prepare single strings delimited by ':' which are then
        # put into F routines
        tot_str_names = ''
        for sstring in self.cloud_species:
            tot_str_names = tot_str_names + sstring + ':'

        tot_str_modes = ''
        for sstring in self.cloud_species_mode:
            tot_str_modes = tot_str_modes + sstring + ':'

        self.N_cloud_lambda_bins = int(len(np.genfromtxt(self.path + \
            '/opacities/continuum/clouds/MgSiO3_c/amorphous/mie/opa_0001.dat' \
                                                             )[:,0]))

        # Actual reading of opacities
        rho_cloud_particles, cloud_specs_abs_opa, cloud_specs_scat_opa, \
          cloud_aniso, cloud_lambdas, cloud_rad_bins, cloud_radii \
          = fi.read_in_cloud_opacities(self.path,tot_str_names,tot_str_modes, \
                            len(self.cloud_species),self.N_cloud_lambda_bins)

        self.rho_cloud_particles = \
          np.array(rho_cloud_particles,dtype='d',order='F')
        self.cloud_specs_abs_opa = \
          np.array(cloud_specs_abs_opa,dtype='d',order='F')
        self.cloud_specs_scat_opa = \
          np.array(cloud_specs_scat_opa,dtype='d',order='F')
        self.cloud_aniso = np.array(cloud_aniso,dtype='d',order='F')
        self.cloud_lambdas = np.array(cloud_lambdas,dtype='d',order='F')
        self.cloud_rad_bins = np.array(cloud_rad_bins,dtype='d',order='F')
        self.cloud_radii = np.array(cloud_radii,dtype='d',order='F')
