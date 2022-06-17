"""
Run with:
    mpiexec -n N --use-hwthread-cpus python3 _test_high_resolution.py
N is the number of processes.
Try:
    sudo mpiexec -n N --allow-run-as-root ...
If for some reason the script crashes.
"""
import argparse
import copy
import os.path
import sys
import time

import numpy as np

# from petitRADTRANS.ccf.high_resolution_retrieval_wasp39_b import *
# from petitRADTRANS.ccf.high_resolution_retrieval_toi270_c import *
#from petitRADTRANS.ccf.high_resolution_retrieval_HD_189733_b import *
from petitRADTRANS.ccf.high_resolution_retrieval_HD_189733_b2 import *
# from petitRADTRANS.ccf._high_resolution_retrieval2 import *
from petitRADTRANS.ccf.model_containers import RetrievalSpectralModel
from petitRADTRANS.ccf.spectra_utils import load_snr_file
from petitRADTRANS.retrieval import Retrieval
from petitRADTRANS.retrieval.plotting import contour_corner
import matplotlib.pyplot as plt


# Arguments definition
parser = argparse.ArgumentParser(
    description='Launch HR retrieval script'
)

parser.add_argument(
    '--planet',
    default='HD 189733 b',
    help='planet name '
)


def main(sim_id=0):
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    module_dir = os.path.abspath(os.path.dirname(__file__))

    planet_name = 'WASP-39 b'
    planet = Planet.get(planet_name)
    planet.radius *= 6

    line_species_str = ['CO_main_iso', 'CO_36', 'CO2_main_iso', 'H2O_main_iso']
    # line_species_str = ['CO_all_iso', 'H2O_main_iso']

    retrieval_name = f't{planet_name}{sim_id}_tt_p_kp_vr_CO_13CO_CO2_H2O_t_79-80'
    retrieval_directory = os.path.abspath(os.path.join(module_dir, '..', '__tmp', 'test_retrieval', retrieval_name))

    mode = 'transit'
    n_live_points = 100
    add_noise = False

    retrieval_directories = os.path.abspath(os.path.join(module_dir, '..', '__tmp', 'test_retrieval'))

    load_from = None
    # load_from = os.path.join(retrieval_directories, f't0_kp_vr_CO_H2O_79-80_{mode}_200lp_np')
    # load_from = os.path.join(retrieval_directories, f't{sim_id}_vttt_p_kp_vr_CO_H2O_79-80_{mode}_30lp')
    # load_from = os.path.join(retrieval_directories, f't1_tt2_p_mr_kp_vr_CO_H2O_79-80_{mode}_100lp')

    band = 'K'

    wavelengths_borders = {
        'K': [1.945, 2.502],
        # 'M': [4.79, 4.80]  # [4.5, 5.5],
    }

    integration_times_ref = {
        'K': 60,
        # 'M': 76.89
    }

    wavelengths_borders = wavelengths_borders[band]
    integration_times_ref = integration_times_ref[band]

    setting = '2192'

    snr_file = os.path.join(module_dir, 'crires',
                            f"snr_crires_K2192_exp60s_dit60s_airmass1.2_PHOENIX-5485.0K_mJ10.663.json")
    telluric_transmittance = \
        os.path.join(module_dir, 'crires', 'transmission_2640m_1945-2502nm_R160k_FWHM2_default.dat')
    airmass = None #os.path.join(module_dir, 'metis', 'brogi_crires_test', 'air.npy')
    variable_throughput = None #os.path.join(module_dir, 'metis', 'brogi_crires_test')

    wavelengths_instrument = None
    instrument_snr = None
    plot = True
    instrument_resolving_power = 8e4

    if rank == 0:
        # Initialize parameters
        retrieval_name, retrieval_directory, \
            model, pressures, true_parameters, line_species, rayleigh_species, continuum_species, \
            retrieval_model, \
            wavelength_instrument, reduced_mock_observations, error \
            = init_mock_observations(
                planet=planet,
                line_species_str=line_species_str,
                mode=mode,
                retrieval_directory=retrieval_directory,
                retrieval_name=retrieval_name,
                add_noise=add_noise,
                wavelengths_borders=wavelengths_borders,
                integration_times_ref=integration_times_ref,
                n_transits=1,
                wavelengths_instrument=wavelengths_instrument,
                instrument_snr=instrument_snr,
                snr_file=snr_file,
                telluric_transmittance=telluric_transmittance,
                airmass=airmass,
                variable_throughput=variable_throughput,
                instrument_resolving_power=instrument_resolving_power,
                load_from=load_from,
                plot=plot
            )

        retrieval_parameters = {
            'retrieval_name': retrieval_name,
            'pressures': pressures,
            'retrieved_species': line_species,
            'rayleigh_species': rayleigh_species,
            'continuum_species': continuum_species,
            'retrieval_model': retrieval_model,
            'wavelengths_instrument': wavelength_instrument,
            'observed_spectra': reduced_mock_observations,
            'observations_uncertainties': error,
            'prt_object': model,
            'parameters': true_parameters
        }

        retrieval_directory = retrieval_directory
        for p, it in retrieval_parameters.items():
            if p == 'parameters':
                for pp, it2 in it.items():
                    print(f'{p} {pp}: {it2.value}')
            else:
                print(f'size of {p}: {it}')
        for p, it in retrieval_parameters.items():
            if p == 'parameters':
                for pp, it2 in it.items():
                    print(f'{p} {pp}: {float(sys.getsizeof(it2.value)) / 8 * 1e-6} MB')
                    print(f'{p} {pp}: {type(it2.value)}')
                    if hasattr(it2.value, '__iter__'):
                        print(np.shape(it2.value))
            else:
                print(f'size of {p}: {float(sys.getsizeof(it)) / 8 * 1e-6} MB')
    else:
        print(f"Rank {rank} waiting for main process to finish...")
        retrieval_parameters = {  # needed for broadcasting
            'retrieval_name': None,
            'pressures': None,
            'retrieved_species': None,
            'rayleigh_species': None,
            'continuum_species': None,
            'retrieval_model': None,
            'wavelengths_instrument': None,
            'observed_spectra': None,
            'observations_uncertainties': None,
            'prt_object': Radtrans(
                line_species=['H2O_main_iso'], mode='lbl', pressures=np.array([1e2]), wlen_bords_micron=[1.0, 1.00001]
            ),  # initializing here because Radtrans cannot be an empty instance
            'parameters': {}
        }
        retrieval_directory = ''

    # return 0

    for key in retrieval_parameters:
        # print(f'rank {rank}:', key)
        # if rank == 0:
        #     print(f'Broadcasting key {key}...')
        #
        # if key == 'parameters':
        #     print(f'rank {rank}:', key)
        #
        #     if rank == 0:
        #         p_list = list(retrieval_parameters[key].keys())
        #     else:
        #         p_list = None
        #
        #     p_list = comm.bcast(p_list, root=0)
        #     p_dict = {key: Param(None) for key in p_list}
        #
        #     if rank != 0:
        #         retrieval_parameters[key] = p_dict
        #
        #     print(f'rank {rank}:', list(retrieval_parameters[key].keys()))
        #     comm.barrier()
        #
        #     for pkey in retrieval_parameters[key]:
        #         print(f'rank {rank}:', pkey)
        #         comm.barrier()
        #         arr_shape = None
        #         arr_dtype = None
        #         if rank == 0:
        #             print(f'Broadcasting parameter key {pkey}...')
        #             if isinstance(retrieval_parameters[key][pkey].value, np.ndarray) \
        #                     and not isinstance(retrieval_parameters[key][pkey].value, np.ma.core.MaskedArray):
        #                 arr_shape = retrieval_parameters[key][pkey].value.shape
        #                 arr_dtype = retrieval_parameters[key][pkey].value.dtype
        #
        #         arr_shape = comm.bcast(arr_shape, root=0)
        #         arr_dtype = comm.bcast(arr_dtype, root=0)
        #
        #         print(f'rank {rank}:', arr_shape, arr_dtype)
        #         comm.barrier()
        #
        #         if arr_shape is not None:
        #             print(f'Arr init...')
        #             if rank == 0:
        #                 arr_tmp = retrieval_parameters[key][pkey].value.flatten()
        #                 print(arr_tmp)
        #             else:
        #                 arr_tmp = np.empty(np.prod(arr_shape), dtype=arr_dtype)
        #             print(f'rank {rank}: {np.shape(arr_tmp)}')
        #             print(f'Arr bcast...')
        #             comm.Bcast([arr_tmp, MPI.DOUBLE_PRECISION], root=0)
        #             print(f'Parametrization...')
        #             if rank != 0:
        #                 retrieval_parameters[key][pkey] = Param(arr_tmp.reshape(arr_shape))
        #             print(f'rank {rank}: {np.shape(retrieval_parameters[key][pkey].value)}')
        #         else:
        #             retrieval_parameters[key][pkey] = Param(comm.bcast(retrieval_parameters[key][pkey].value, root=0))
        #         print('Done')
        # elif key == 'prt_object':
        if key == 'prt_object':
            init_dict = {
                'line_species': None,
                'rayleigh_species': None,
                'continuum_opacities': None,
                'wlen_bords_micron': None,
                'mode': None,
                'do_scat_emis': None,
                'lbl_opacity_sampling': None,
                'press': None
            }
            # prt_keys = comm.bcast(list(retrieval_parameters[key].__dict__.keys()), root=0)

            for pkey in init_dict:
                if rank == 0:
                    print(f'Broadcasting Radtrans init key {pkey}...')

                if pkey == 'continuum_opacities':
                    init_dict[pkey] = ['H2-H2', 'H2-He']
                else:
                    init_dict[pkey] = comm.bcast(retrieval_parameters[key].__dict__[pkey], root=0)

            if rank != 0:
                retrieval_parameters[key] = Radtrans(
                    line_species=init_dict['line_species'],
                    rayleigh_species=init_dict['rayleigh_species'],
                    continuum_opacities=init_dict['continuum_opacities'],
                    wlen_bords_micron=init_dict['wlen_bords_micron'],
                    mode=init_dict['mode'],
                    do_scat_emis=init_dict['do_scat_emis'],
                    lbl_opacity_sampling=init_dict['lbl_opacity_sampling']
                )
                retrieval_parameters[key].setup_opa_structure(init_dict['press'] * 1e-6)

            print(f'rank {rank} waiting...')
            comm.barrier()
        else:
            retrieval_parameters[key] = comm.bcast(retrieval_parameters[key], root=0)

    retrieval_directory = comm.bcast(retrieval_directory, root=0)

    if rank == 0:
        print("Bcasting done !!!")

    # # Check if all observations are the same
    # obs_tmp = comm.allgather(retrieval_parameters['observed_spectra'])
    #
    # for obs_tmp_proc in obs_tmp[1:]:
    #     assert np.allclose(obs_tmp_proc, obs_tmp[0], atol=0, rtol=1e-15)

    # print(f"Shared observations consistency check OK")

    # Initialize retrieval
    run_definitions = init_run(**retrieval_parameters)

    retrieval = Retrieval(
        run_definitions,
        output_dir=retrieval_directory,
        sample_spec=False,
        ultranest=False,
        pRT_plot_style=False
    )

    retrieval.run(
        sampling_efficiency=0.8,
        n_live_points=n_live_points,
        const_efficiency_mode=False,
        resume=False
    )

    if rank == 0:
        sample_dict, parameter_dict = retrieval.get_samples(
            output_dir=retrieval_directory + os.path.sep,
            ret_names=[retrieval_name]
        )

        n_param = len(parameter_dict[retrieval_name])
        parameter_plot_indices = {retrieval_name: np.arange(0, n_param)}

        true_values = {retrieval_name: []}

        for p in parameter_dict[retrieval_name]:
            true_values[retrieval_name].append(np.mean(retrieval_parameters['parameters'][p].value))

        fig = contour_corner(
            sample_dict, parameter_dict, os.path.join(retrieval_directory, f'corner_{retrieval_name}.png'),
            parameter_plot_indices=parameter_plot_indices,
            true_values=true_values, prt_plot_style=False
        )

        fig.show()


def maintoi(sim_id=0):
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    module_dir = os.path.abspath(os.path.dirname(__file__))

    planet_name = 'TOI-270 c'
    planet = Planet.get(planet_name)

    planet.orbit_semi_major_axis /= 3

    line_species_str = ['CH4_main_iso', 'CO_all_iso', 'H2O_main_iso', 'NH3_main_iso']
    # line_species_str = ['CO_all_iso', 'H2O_main_iso', 'NH3_main_iso']
    # line_species_str = ['CO_all_iso', 'H2O_main_iso']

    retrieval_name = f't{planet_name}{sim_id}_p_kp_vr_CH4_CO_CO2_H2O_t_K2192-25-26-27-28_4transits_nn'
    retrieval_directory = os.path.abspath(os.path.join(module_dir, '..', '__tmp', 'test_retrieval', retrieval_name))

    mode = 'transit'
    n_live_points = 100
    add_noise = False

    retrieval_directories = os.path.abspath(os.path.join(module_dir, '..', '__tmp', 'test_retrieval'))

    load_from = None
    # load_from = os.path.join(retrieval_directories, f't0_kp_vr_CO_H2O_79-80_{mode}_200lp_np')
    # load_from = os.path.join(retrieval_directories, f't{sim_id}_vttt_p_kp_vr_CO_H2O_79-80_{mode}_30lp')
    # load_from = os.path.join(retrieval_directories, f't1_tt2_p_mr_kp_vr_CO_H2O_79-80_{mode}_100lp')

    band = 'K'

    wavelengths_borders = {
        'K': [1.945, 2.502],
        # 'M': [4.79, 4.80]  # [4.5, 5.5],
    }

    integration_times_ref = {
        'K': 60,
        # 'M': 76.89
    }

    setting = '2192'

    snr_file = os.path.join(module_dir, 'crires',
                            f"snr_crires_K2192_exp60s_dit60s_airmass1.33_PHOENIX-3506.0K_mJ9.099.json")
    telluric_transmittance = None#\
        # os.path.join(module_dir, 'crires', 'transmission_2640m_1945-2502nm_R160k_FWHM2_default.dat')
    airmass = None #os.path.join(module_dir, 'metis', 'brogi_crires_test', 'air.npy')
    variable_throughput = None #os.path.join(module_dir, 'metis', 'brogi_crires_test')

    wavelengths_instrument = None
    instrument_snr = None
    plot = True
    instrument_resolving_power = 8e4

    if rank == 0:
        # Initialize parameters
        retrieval_name, retrieval_directory, \
            model, pressures, true_parameters, line_species, rayleigh_species, continuum_species, \
            retrieval_model, \
            wavelength_instrument, reduced_mock_observations, error \
            = init_mock_observations(
                planet, line_species_str, mode,
                retrieval_directory, retrieval_name, n_live_points,
                add_noise, band, wavelengths_borders, integration_times_ref,
                wavelengths_instrument=wavelengths_instrument, instrument_snr=instrument_snr, snr_file=snr_file,
                telluric_transmittance=telluric_transmittance, airmass=airmass,
                variable_throughput=variable_throughput,
                instrument_resolving_power=instrument_resolving_power,
                same_data_model_radtrans=True,
                load_from=load_from, plot=plot
            )

        retrieval_parameters = {
            'retrieval_name': retrieval_name,
            'pressures': pressures,
            'retrieved_species': line_species_str,
            'rayleigh_species': rayleigh_species,
            'continuum_species': continuum_species,
            'retrieval_model': retrieval_model,
            'wavelengths_instrument': wavelength_instrument,
            'observed_spectra': reduced_mock_observations,
            'observations_uncertainties': error,
            'prt_object': model,
            'parameters': true_parameters
        }

        retrieval_directory = retrieval_directory
    else:
        print(f"Rank {rank} waiting for main process to finish...")
        retrieval_parameters = {  # needed for broadcasting
            'retrieval_name': None,
            'pressures': None,
            'retrieved_species': None,
            'rayleigh_species': None,
            'continuum_species': None,
            'retrieval_model': None,
            'wavelengths_instrument': None,
            'observed_spectra': None,
            'observations_uncertainties': None,
            'prt_object': Radtrans(
                line_species=['H2O_main_iso'], mode='lbl', pressures=np.array([1e2]), wlen_bords_micron=[1.0, 1.00001]
            ),  # initializing here because Radtrans cannot be an empty instance
            'parameters': {}
        }
        retrieval_directory = ''

    # return 0

    for key in retrieval_parameters:
        # print(f'rank {rank}:', key)
        # if rank == 0:
        #     print(f'Broadcasting key {key}...')
        #
        # if key == 'parameters':
        #     print(f'rank {rank}:', key)
        #
        #     if rank == 0:
        #         p_list = list(retrieval_parameters[key].keys())
        #     else:
        #         p_list = None
        #
        #     p_list = comm.bcast(p_list, root=0)
        #     p_dict = {key: Param(None) for key in p_list}
        #
        #     if rank != 0:
        #         retrieval_parameters[key] = p_dict
        #
        #     print(f'rank {rank}:', list(retrieval_parameters[key].keys()))
        #     comm.barrier()
        #
        #     for pkey in retrieval_parameters[key]:
        #         print(f'rank {rank}:', pkey)
        #         comm.barrier()
        #         arr_shape = None
        #         arr_dtype = None
        #         if rank == 0:
        #             print(f'Broadcasting parameter key {pkey}...')
        #             if isinstance(retrieval_parameters[key][pkey].value, np.ndarray) \
        #                     and not isinstance(retrieval_parameters[key][pkey].value, np.ma.core.MaskedArray):
        #                 arr_shape = retrieval_parameters[key][pkey].value.shape
        #                 arr_dtype = retrieval_parameters[key][pkey].value.dtype
        #
        #         arr_shape = comm.bcast(arr_shape, root=0)
        #         arr_dtype = comm.bcast(arr_dtype, root=0)
        #
        #         print(f'rank {rank}:', arr_shape, arr_dtype)
        #         comm.barrier()
        #
        #         if arr_shape is not None:
        #             print(f'Arr init...')
        #             if rank == 0:
        #                 arr_tmp = retrieval_parameters[key][pkey].value.flatten()
        #                 print(arr_tmp)
        #             else:
        #                 arr_tmp = np.empty(np.prod(arr_shape), dtype=arr_dtype)
        #             print(f'rank {rank}: {np.shape(arr_tmp)}')
        #             print(f'Arr bcast...')
        #             comm.Bcast([arr_tmp, MPI.DOUBLE_PRECISION], root=0)
        #             print(f'Parametrization...')
        #             if rank != 0:
        #                 retrieval_parameters[key][pkey] = Param(arr_tmp.reshape(arr_shape))
        #             print(f'rank {rank}: {np.shape(retrieval_parameters[key][pkey].value)}')
        #         else:
        #             retrieval_parameters[key][pkey] = Param(comm.bcast(retrieval_parameters[key][pkey].value, root=0))
        #         print('Done')
        # elif key == 'prt_object':
        if key == 'prt_object':
            init_dict = {
                'line_species': None,
                'rayleigh_species': None,
                'continuum_opacities': None,
                'wlen_bords_micron': None,
                'mode': None,
                'do_scat_emis': None,
                'lbl_opacity_sampling': None,
                'press': None
            }
            # prt_keys = comm.bcast(list(retrieval_parameters[key].__dict__.keys()), root=0)

            for pkey in init_dict:
                if rank == 0:
                    print(f'Broadcasting Radtrans init key {pkey}...')

                if pkey == 'continuum_opacities':
                    init_dict[pkey] = ['H2-H2', 'H2-He']
                else:
                    init_dict[pkey] = comm.bcast(retrieval_parameters[key].__dict__[pkey], root=0)

            if rank != 0:
                retrieval_parameters[key] = Radtrans(
                    line_species=init_dict['line_species'],
                    rayleigh_species=init_dict['rayleigh_species'],
                    continuum_opacities=init_dict['continuum_opacities'],
                    wlen_bords_micron=init_dict['wlen_bords_micron'],
                    mode=init_dict['mode'],
                    do_scat_emis=init_dict['do_scat_emis'],
                    lbl_opacity_sampling=init_dict['lbl_opacity_sampling']
                )
                retrieval_parameters[key].setup_opa_structure(init_dict['press'] * 1e-6)

            print(f'rank {rank} waiting...')
            comm.barrier()
        else:
            retrieval_parameters[key] = comm.bcast(retrieval_parameters[key], root=0)

    retrieval_directory = comm.bcast(retrieval_directory, root=0)

    if rank == 0:
        print("Bcasting done !!!")

    # # Check if all observations are the same
    # obs_tmp = comm.allgather(retrieval_parameters['observed_spectra'])
    #
    # for obs_tmp_proc in obs_tmp[1:]:
    #     assert np.allclose(obs_tmp_proc, obs_tmp[0], atol=0, rtol=1e-15)

    # print(f"Shared observations consistency check OK")

    # Initialize retrieval
    run_definitions = init_run(**retrieval_parameters)

    retrieval = Retrieval(
        run_definitions,
        output_dir=retrieval_directory,
        sample_spec=False,
        ultranest=False,
        pRT_plot_style=False
    )

    retrieval.run(
        sampling_efficiency=0.8,
        n_live_points=n_live_points,
        const_efficiency_mode=False,
        resume=False
    )

    if rank == 0:
        sample_dict, parameter_dict = retrieval.get_samples(
            output_dir=retrieval_directory + os.path.sep,
            ret_names=[retrieval_name]
        )

        n_param = len(parameter_dict[retrieval_name])
        parameter_plot_indices = {retrieval_name: np.arange(0, n_param)}

        true_values = {retrieval_name: []}

        for p in parameter_dict[retrieval_name]:
            true_values[retrieval_name].append(np.mean(retrieval_parameters['parameters'][p].value))

        fig = contour_corner(
            sample_dict, parameter_dict, os.path.join(retrieval_directory, f'corner_{retrieval_name}.png'),
            parameter_plot_indices=parameter_plot_indices,
            true_values=true_values, prt_plot_style=False
        )

        fig.show()


def maintic(sim_id=0):
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    module_dir = os.path.abspath(os.path.dirname(__file__))

    planet_name = 'TIC 237913194 b'
    planet = Planet.get(planet_name)

    line_species_str = ['CH4_main_iso', 'CO_all_iso', 'H2O_main_iso', 'NH3_main_iso']
    # line_species_str = ['CO_all_iso', 'H2O_main_iso', 'NH3_main_iso']
    # line_species_str = ['CO_all_iso', 'H2O_main_iso']

    retrieval_name = f't{planet_name}{sim_id}_tt_p_kp_vr_CO_CO2_H2O_t_79-80_nn'
    retrieval_directory = os.path.abspath(os.path.join(module_dir, '..', '__tmp', 'test_retrieval', retrieval_name))

    mode = 'transit'
    n_live_points = 100
    add_noise = False

    retrieval_directories = os.path.abspath(os.path.join(module_dir, '..', '__tmp', 'test_retrieval'))

    load_from = None
    # load_from = os.path.join(retrieval_directories, f't0_kp_vr_CO_H2O_79-80_{mode}_200lp_np')
    # load_from = os.path.join(retrieval_directories, f't{sim_id}_vttt_p_kp_vr_CO_H2O_79-80_{mode}_30lp')
    # load_from = os.path.join(retrieval_directories, f't1_tt2_p_mr_kp_vr_CO_H2O_79-80_{mode}_100lp')

    band = 'K'

    wavelengths_borders = {
        'K': [1.945, 2.502],
        # 'M': [4.79, 4.80]  # [4.5, 5.5],
    }

    integration_times_ref = {
        'K': 60,
        # 'M': 76.89
    }

    setting = '2192'

    snr_file = os.path.join(module_dir, 'crires',
                            f"snr_crires_K2192_exp60s_dit60s_airmass1.2_PHOENIX-5485.0K_mJ10.663.json")
    telluric_transmittance = \
        os.path.join(module_dir, 'crires', 'transmission_2640m_1945-2502nm_R160k_FWHM2_default.dat')
    airmass = None #os.path.join(module_dir, 'metis', 'brogi_crires_test', 'air.npy')
    variable_throughput = None #os.path.join(module_dir, 'metis', 'brogi_crires_test')

    wavelengths_instrument = None
    instrument_snr = None
    plot = True
    instrument_resolving_power = 8e4

    if rank == 0:
        # Initialize parameters
        retrieval_name, retrieval_directory, \
            model, pressures, true_parameters, line_species, rayleigh_species, continuum_species, \
            retrieval_model, \
            wavelength_instrument, reduced_mock_observations, error \
            = init_mock_observations(
                planet, line_species_str, mode,
                retrieval_directory, retrieval_name, n_live_points,
                add_noise, band, wavelengths_borders, integration_times_ref,
                wavelengths_instrument=wavelengths_instrument, instrument_snr=instrument_snr, snr_file=snr_file,
                telluric_transmittance=telluric_transmittance, airmass=airmass,
                variable_throughput=variable_throughput,
                instrument_resolving_power=instrument_resolving_power,
                same_data_model_radtrans=True,
                load_from=load_from, plot=plot
            )

        retrieval_parameters = {
            'retrieval_name': retrieval_name,
            'pressures': pressures,
            'retrieved_species': line_species_str,
            'rayleigh_species': rayleigh_species,
            'continuum_species': continuum_species,
            'retrieval_model': retrieval_model,
            'wavelengths_instrument': wavelength_instrument,
            'observed_spectra': reduced_mock_observations,
            'observations_uncertainties': error,
            'prt_object': model,
            'parameters': true_parameters
        }

        retrieval_directory = retrieval_directory
    else:
        print(f"Rank {rank} waiting for main process to finish...")
        retrieval_parameters = {  # needed for broadcasting
            'retrieval_name': None,
            'pressures': None,
            'retrieved_species': None,
            'rayleigh_species': None,
            'continuum_species': None,
            'retrieval_model': None,
            'wavelengths_instrument': None,
            'observed_spectra': None,
            'observations_uncertainties': None,
            'prt_object': Radtrans(
                line_species=['H2O_main_iso'], mode='lbl', pressures=np.array([1e2]), wlen_bords_micron=[1.0, 1.00001]
            ),  # initializing here because Radtrans cannot be an empty instance
            'parameters': {}
        }
        retrieval_directory = ''

    # return 0

    for key in retrieval_parameters:
        # print(f'rank {rank}:', key)
        # if rank == 0:
        #     print(f'Broadcasting key {key}...')
        #
        # if key == 'parameters':
        #     print(f'rank {rank}:', key)
        #
        #     if rank == 0:
        #         p_list = list(retrieval_parameters[key].keys())
        #     else:
        #         p_list = None
        #
        #     p_list = comm.bcast(p_list, root=0)
        #     p_dict = {key: Param(None) for key in p_list}
        #
        #     if rank != 0:
        #         retrieval_parameters[key] = p_dict
        #
        #     print(f'rank {rank}:', list(retrieval_parameters[key].keys()))
        #     comm.barrier()
        #
        #     for pkey in retrieval_parameters[key]:
        #         print(f'rank {rank}:', pkey)
        #         comm.barrier()
        #         arr_shape = None
        #         arr_dtype = None
        #         if rank == 0:
        #             print(f'Broadcasting parameter key {pkey}...')
        #             if isinstance(retrieval_parameters[key][pkey].value, np.ndarray) \
        #                     and not isinstance(retrieval_parameters[key][pkey].value, np.ma.core.MaskedArray):
        #                 arr_shape = retrieval_parameters[key][pkey].value.shape
        #                 arr_dtype = retrieval_parameters[key][pkey].value.dtype
        #
        #         arr_shape = comm.bcast(arr_shape, root=0)
        #         arr_dtype = comm.bcast(arr_dtype, root=0)
        #
        #         print(f'rank {rank}:', arr_shape, arr_dtype)
        #         comm.barrier()
        #
        #         if arr_shape is not None:
        #             print(f'Arr init...')
        #             if rank == 0:
        #                 arr_tmp = retrieval_parameters[key][pkey].value.flatten()
        #                 print(arr_tmp)
        #             else:
        #                 arr_tmp = np.empty(np.prod(arr_shape), dtype=arr_dtype)
        #             print(f'rank {rank}: {np.shape(arr_tmp)}')
        #             print(f'Arr bcast...')
        #             comm.Bcast([arr_tmp, MPI.DOUBLE_PRECISION], root=0)
        #             print(f'Parametrization...')
        #             if rank != 0:
        #                 retrieval_parameters[key][pkey] = Param(arr_tmp.reshape(arr_shape))
        #             print(f'rank {rank}: {np.shape(retrieval_parameters[key][pkey].value)}')
        #         else:
        #             retrieval_parameters[key][pkey] = Param(comm.bcast(retrieval_parameters[key][pkey].value, root=0))
        #         print('Done')
        # elif key == 'prt_object':
        if key == 'prt_object':
            init_dict = {
                'line_species': None,
                'rayleigh_species': None,
                'continuum_opacities': None,
                'wlen_bords_micron': None,
                'mode': None,
                'do_scat_emis': None,
                'lbl_opacity_sampling': None,
                'press': None
            }
            # prt_keys = comm.bcast(list(retrieval_parameters[key].__dict__.keys()), root=0)

            for pkey in init_dict:
                if rank == 0:
                    print(f'Broadcasting Radtrans init key {pkey}...')

                if pkey == 'continuum_opacities':
                    init_dict[pkey] = ['H2-H2', 'H2-He']
                else:
                    init_dict[pkey] = comm.bcast(retrieval_parameters[key].__dict__[pkey], root=0)

            if rank != 0:
                retrieval_parameters[key] = Radtrans(
                    line_species=init_dict['line_species'],
                    rayleigh_species=init_dict['rayleigh_species'],
                    continuum_opacities=init_dict['continuum_opacities'],
                    wlen_bords_micron=init_dict['wlen_bords_micron'],
                    mode=init_dict['mode'],
                    do_scat_emis=init_dict['do_scat_emis'],
                    lbl_opacity_sampling=init_dict['lbl_opacity_sampling']
                )
                retrieval_parameters[key].setup_opa_structure(init_dict['press'] * 1e-6)

            print(f'rank {rank} waiting...')
            comm.barrier()
        else:
            retrieval_parameters[key] = comm.bcast(retrieval_parameters[key], root=0)

    retrieval_directory = comm.bcast(retrieval_directory, root=0)

    if rank == 0:
        print("Bcasting done !!!")

    # # Check if all observations are the same
    # obs_tmp = comm.allgather(retrieval_parameters['observed_spectra'])
    #
    # for obs_tmp_proc in obs_tmp[1:]:
    #     assert np.allclose(obs_tmp_proc, obs_tmp[0], atol=0, rtol=1e-15)

    # print(f"Shared observations consistency check OK")

    # Initialize retrieval
    run_definitions = init_run(**retrieval_parameters)

    retrieval = Retrieval(
        run_definitions,
        output_dir=retrieval_directory,
        sample_spec=False,
        ultranest=False,
        pRT_plot_style=False
    )

    retrieval.run(
        sampling_efficiency=0.8,
        n_live_points=n_live_points,
        const_efficiency_mode=False,
        resume=False
    )

    if rank == 0:
        sample_dict, parameter_dict = retrieval.get_samples(
            output_dir=retrieval_directory + os.path.sep,
            ret_names=[retrieval_name]
        )

        n_param = len(parameter_dict[retrieval_name])
        parameter_plot_indices = {retrieval_name: np.arange(0, n_param)}

        true_values = {retrieval_name: []}

        for p in parameter_dict[retrieval_name]:
            true_values[retrieval_name].append(np.mean(retrieval_parameters['parameters'][p].value))

        fig = contour_corner(
            sample_dict, parameter_dict, os.path.join(retrieval_directory, f'corner_{retrieval_name}.png'),
            parameter_plot_indices=parameter_plot_indices,
            true_values=true_values, prt_plot_style=False
        )

        fig.show()


def main_hd(sim_id=0):
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    module_dir = os.path.abspath(os.path.dirname(__file__))

    star_name = 'WASP-82'
    planet_name = star_name + ' b'
    planet = Planet.get(planet_name)

    # Paul's setup for HD 189733 b
    # planet.equilibrium_temperature = 1209

    # line_species_str = ['CO_main_iso', 'CO_36', 'CH4_main_iso', 'H2O_main_iso']
    line_species_str = ['CO_main_iso', 'CH4_main_iso', 'H2O_main_iso']
    # line_species_str = ['CO_36', 'CH4_main_iso', 'H2O_main_iso']
    # line_species_str = ['CO_main_iso', 'CO_36', 'H2O_main_iso']
    # line_species_str = ['CO_main_iso', 'CO_36', 'CH4_main_iso']

    retrieval_name = f't{planet_name}{sim_id}_12CO_13CO_CH4_H2O_t_18-28_noCO_36'

    if not os.path.isdir(
            os.path.join(module_dir, '..', '__tmp', 'test_retrieval', 'bins_' + planet_name.lower().replace(' ', '_'))
    ):
        os.mkdir(
            os.path.join(module_dir, '..', '__tmp', 'test_retrieval', 'bins_' + planet_name.lower().replace(' ', '_'))
        )

    retrieval_directory = os.path.abspath(
        os.path.join(module_dir, '..', '__tmp', 'test_retrieval', 'bins_' + planet_name.lower().replace(' ', '_'),
                     retrieval_name)
    )

    mode = 'transit'
    n_live_points = 100
    add_noise = False

    retrieval_directories = os.path.abspath(os.path.join(module_dir, '..', '__tmp', 'test_retrieval'))

    load_from = None
    # load_from = os.path.join(retrieval_directories, f't0_kp_vr_CO_H2O_79-80_{mode}_200lp_np')
    # load_from = os.path.join(retrieval_directories, f't{sim_id}_vttt_p_kp_vr_CO_H2O_79-80_{mode}_30lp')
    # load_from = os.path.join(retrieval_directories, f't1_tt2_p_mr_kp_vr_CO_H2O_79-80_{mode}_100lp')

    band = 'K'

    wavelengths_borders = {
        'K': [1.8, 2.8],
        # 'K': [2.28, 2.42],
        # 'K': [2.15, 2.4],
        # 'M': [4.79, 4.80]  # [4.5, 5.5],
    }

    wrange_0 = wavelengths_borders[band] * np.array([1.001, 0.999])
    n_wranges = 100
    wranges = np.linspace(wrange_0[0], wrange_0[1], int(n_wranges + 1))
    wavelengths_borders[band] = np.array([wranges[sim_id], wranges[sim_id + 1]])

    if wavelengths_borders[band][-1] > 2.5:
        print("End of transmission file")
        return 0

    integration_times_ref = {
        'K': 60,
        # 'M': 76.89
    }

    wavelengths_borders = wavelengths_borders[band]
    integration_times_ref = integration_times_ref[band]

    snr_file = os.path.join(module_dir, 'andes', star_name.replace(' ', '_'), f"ANDES_snrs.npz")
    telluric_transmittance = os.path.join(module_dir, 'andes', 'sky', 'transmission',
                                          f"transmission_1500_2500.dat")
    airmass = os.path.join(module_dir, 'andes', star_name.replace(' ', '_'), 'airmass_optimal.txt')
    variable_throughput = os.path.join(module_dir, 'metis', 'brogi_crires_test', 'algn.npy')

    wavelengths_instrument = None
    instrument_snr = None
    plot = True
    instrument_resolving_power = 1e5

    if rank == 0:
        # Initialize parameters
        retrieval_name, retrieval_directory, \
            model, pressures, true_parameters, line_species, rayleigh_species, continuum_species, \
            retrieval_model, \
            wavelength_instrument, reduced_mock_observations, error \
            = init_mock_observations(
                planet, line_species_str, mode,
                retrieval_directory, retrieval_name,
                add_noise, wavelengths_borders[band], integration_times_ref[band],
                wavelengths_instrument=wavelengths_instrument, instrument_snr=instrument_snr, snr_file=snr_file,
                telluric_transmittance=telluric_transmittance, airmass=airmass,
                variable_throughput=variable_throughput,
                instrument_resolving_power=instrument_resolving_power,
                load_from=load_from, plot=plot
            )

        retrieval_parameters = {
            'retrieval_name': retrieval_name,
            'pressures': pressures,
            'retrieved_species': line_species,
            'rayleigh_species': rayleigh_species,
            'continuum_species': continuum_species,
            'retrieval_model': retrieval_model,
            'wavelengths_instrument': wavelength_instrument,
            'observed_spectra': reduced_mock_observations,
            'observations_uncertainties': error,
            'prt_object': model,
            'parameters': true_parameters
        }

        retrieval_directory = retrieval_directory
    else:
        print(f"Rank {rank} waiting for main process to finish...")
        retrieval_parameters = {  # needed for broadcasting
            'retrieval_name': None,
            'pressures': None,
            'retrieved_species': None,
            'rayleigh_species': None,
            'continuum_species': None,
            'retrieval_model': None,
            'wavelengths_instrument': None,
            'observed_spectra': None,
            'observations_uncertainties': None,
            'prt_object': Radtrans(
                line_species=['H2O_main_iso'], mode='lbl', pressures=np.array([1e2]), wlen_bords_micron=[1.0, 1.00001]
            ),  # initializing here because Radtrans cannot be an empty instance
            'parameters': {}
        }
        retrieval_directory = ''

    # return 0

    for key in retrieval_parameters:
        if key == 'prt_object':
            init_dict = {
                'line_species': None,
                'rayleigh_species': None,
                'continuum_opacities': None,
                'wlen_bords_micron': None,
                'mode': None,
                'do_scat_emis': None,
                'lbl_opacity_sampling': None,
                'press': None
            }
            # prt_keys = comm.bcast(list(retrieval_parameters[key].__dict__.keys()), root=0)

            for pkey in init_dict:
                if rank == 0:
                    print(f'Broadcasting Radtrans init key {pkey}...')

                if pkey == 'continuum_opacities':
                    init_dict[pkey] = ['H2-H2', 'H2-He']
                else:
                    init_dict[pkey] = comm.bcast(retrieval_parameters[key].__dict__[pkey], root=0)

            if rank != 0:
                retrieval_parameters[key] = Radtrans(
                    line_species=init_dict['line_species'],
                    rayleigh_species=init_dict['rayleigh_species'],
                    continuum_opacities=init_dict['continuum_opacities'],
                    wlen_bords_micron=init_dict['wlen_bords_micron'],
                    mode=init_dict['mode'],
                    do_scat_emis=init_dict['do_scat_emis'],
                    lbl_opacity_sampling=init_dict['lbl_opacity_sampling']
                )
                retrieval_parameters[key].setup_opa_structure(init_dict['press'] * 1e-6)

            print(f'rank {rank} waiting...')
            comm.barrier()
        else:
            retrieval_parameters[key] = comm.bcast(retrieval_parameters[key], root=0)

    retrieval_directory = comm.bcast(retrieval_directory, root=0)

    if rank == 0:
        print("Bcasting done !!!")

    # # Check if all observations are the same
    # obs_tmp = comm.allgather(retrieval_parameters['observed_spectra'])
    #
    # for obs_tmp_proc in obs_tmp[1:]:
    #     assert np.allclose(obs_tmp_proc, obs_tmp[0], atol=0, rtol=1e-15)

    # print(f"Shared observations consistency check OK")

    # Initialize retrieval
    run_definitions = init_run(**retrieval_parameters)

    retrieval = Retrieval(
        run_definitions,
        output_dir=retrieval_directory,
        sample_spec=False,
        ultranest=False,
        pRT_plot_style=False
    )

    retrieval.run(
        sampling_efficiency=0.8,
        n_live_points=n_live_points,
        const_efficiency_mode=False,
        resume=False
    )

    if rank == 0:
        sample_dict, parameter_dict = retrieval.get_samples(
            output_dir=retrieval_directory + os.path.sep,
            ret_names=[retrieval_name]
        )

        n_param = len(parameter_dict[retrieval_name])
        parameter_plot_indices = {retrieval_name: np.arange(0, n_param)}

        true_values = {retrieval_name: []}

        for p in parameter_dict[retrieval_name]:
            true_values[retrieval_name].append(np.mean(retrieval_parameters['parameters'][p].value))

        fig = contour_corner(
            sample_dict, parameter_dict, os.path.join(retrieval_directory, f'corner_{retrieval_name}.png'),
            parameter_plot_indices=parameter_plot_indices,
            true_values=true_values, prt_plot_style=False
        )

        fig.show()


def main_hd2(sim_id=0):
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    module_dir = os.path.abspath(os.path.dirname(__file__))

    star_name = 'WASP-82'
    planet_name = star_name + ' b'
    planet = Planet.get(planet_name)

    # Paul's setup for HD 189733 b
    # planet.equilibrium_temperature = 1209

    # line_species_str = ['CO_main_iso', 'CO_36', 'CH4_main_iso', 'H2O_main_iso']
    line_species_str = ['CO_main_iso', 'CH4_main_iso', 'H2O_main_iso']
    # line_species_str = ['CO_36', 'CH4_main_iso', 'H2O_main_iso']
    # line_species_str = ['CO_main_iso', 'CO_36', 'H2O_main_iso']
    # line_species_str = ['CO_main_iso', 'CO_36', 'CH4_main_iso']

    retrieval_name = f't{planet_name}{sim_id}_12CO_13CO_CH4_H2O_t_18-28_noCO_36'

    if not os.path.isdir(
            os.path.join(module_dir, '..', '__tmp', 'test_retrieval', 'bins_' + planet_name.lower().replace(' ', '_'))
    ):
        os.mkdir(
            os.path.join(module_dir, '..', '__tmp', 'test_retrieval', 'bins_' + planet_name.lower().replace(' ', '_'))
        )

    retrieval_directory = os.path.abspath(
        os.path.join(module_dir, '..', '__tmp', 'test_retrieval', 'bins_' + planet_name.lower().replace(' ', '_'),
                     retrieval_name)
    )

    mode = 'transit'
    n_live_points = 100
    add_noise = False

    retrieval_directories = os.path.abspath(os.path.join(module_dir, '..', '__tmp', 'test_retrieval'))

    load_from = None
    # load_from = os.path.join(retrieval_directories, f't0_kp_vr_CO_H2O_79-80_{mode}_200lp_np')
    # load_from = os.path.join(retrieval_directories, f't{sim_id}_vttt_p_kp_vr_CO_H2O_79-80_{mode}_30lp')
    # load_from = os.path.join(retrieval_directories, f't1_tt2_p_mr_kp_vr_CO_H2O_79-80_{mode}_100lp')

    band = 'K'

    wavelengths_borders = {
        'K': [1.8, 2.8],
        # 'K': [2.28, 2.42],
        # 'K': [2.15, 2.4],
        # 'M': [4.79, 4.80]  # [4.5, 5.5],
    }

    wrange_0 = wavelengths_borders[band] * np.array([1.001, 0.999])
    n_wranges = 100
    wranges = np.linspace(wrange_0[0], wrange_0[1], int(n_wranges + 1))
    wavelengths_borders[band] = np.array([wranges[sim_id], wranges[sim_id + 1]])

    if wavelengths_borders[band][-1] > 2.5:
        print("End of transmission file")
        return 0

    integration_times_ref = {
        'K': 60,
        # 'M': 76.89
    }

    wavelengths_borders = wavelengths_borders[band]
    integration_times_ref = integration_times_ref[band]

    snr_file = os.path.join(module_dir, 'andes', star_name.replace(' ', '_'), f"ANDES_snrs.npz")
    telluric_transmittance = os.path.join(module_dir, 'andes', 'sky', 'transmission',
                                          f"transmission_1500_2500.dat")
    airmass = os.path.join(module_dir, 'andes', star_name.replace(' ', '_'), 'airmass_optimal.txt')
    variable_throughput = os.path.join(module_dir, 'metis', 'brogi_crires_test', 'algn.npy')

    wavelengths_instrument = None
    instrument_snr = None
    plot = True
    instrument_resolving_power = 1e5

    if rank == 0:
        # Initialize parameters
        retrieval_name, retrieval_directory, \
            model, prt_object, reduced_mock_observations, \
            retrieval_model \
            = init_mock_observations(
                planet, line_species_str, mode,
                retrieval_directory, retrieval_name,
                add_noise, wavelengths_borders, integration_times_ref,
                wavelengths_instrument=wavelengths_instrument, instrument_snr=instrument_snr, snr_file=snr_file,
                telluric_transmittance=telluric_transmittance, airmass=airmass,
                variable_throughput=variable_throughput,
                instrument_resolving_power=instrument_resolving_power,
                load_from=load_from, plot=plot
            )

        retrieval_parameters = {
            'retrieval_name': retrieval_name,
            'pressures': model.pressures,
            'retrieved_species': model.line_species,
            'rayleigh_species': model.rayleigh_species,
            'continuum_species': model.continuum_opacities,
            'retrieval_model': retrieval_model,
            'wavelengths_instrument': model.model_parameters['output_wavelengths'],
            'observed_spectra': reduced_mock_observations,
            'observations_uncertainties': model.model_parameters['uncertainties'],
            'prt_object': prt_object,
            'parameters': copy.deepcopy(model.model_parameters)
        }

        retrieval_directory = retrieval_directory
    else:
        print(f"Rank {rank} waiting for main process to finish...")
        retrieval_parameters = {  # needed for broadcasting
            'retrieval_name': None,
            'pressures': None,
            'retrieved_species': None,
            'rayleigh_species': None,
            'continuum_species': None,
            'retrieval_model': None,
            'wavelengths_instrument': None,
            'observed_spectra': None,
            'observations_uncertainties': None,
            'prt_object': Radtrans(
                line_species=['H2O_main_iso'], mode='lbl', pressures=np.array([1e2]), wlen_bords_micron=[1.0, 1.00001]
            ),  # initializing here because Radtrans cannot be an empty instance
            'parameters': {}
        }
        retrieval_directory = ''

    #return 0

    for key in retrieval_parameters:
        if key == 'prt_object':
            init_dict = {
                'line_species': None,
                'rayleigh_species': None,
                'continuum_opacities': None,
                'wlen_bords_micron': None,
                'mode': None,
                'do_scat_emis': None,
                'lbl_opacity_sampling': None,
                'press': None
            }
            # prt_keys = comm.bcast(list(retrieval_parameters[key].__dict__.keys()), root=0)

            for pkey in init_dict:
                if rank == 0:
                    print(f'Broadcasting Radtrans init key {pkey}...')

                if pkey == 'continuum_opacities':
                    init_dict[pkey] = ['H2-H2', 'H2-He']
                else:
                    init_dict[pkey] = comm.bcast(retrieval_parameters[key].__dict__[pkey], root=0)

            if rank != 0:
                retrieval_parameters[key] = Radtrans(
                    line_species=init_dict['line_species'],
                    rayleigh_species=init_dict['rayleigh_species'],
                    continuum_opacities=init_dict['continuum_opacities'],
                    wlen_bords_micron=init_dict['wlen_bords_micron'],
                    mode=init_dict['mode'],
                    do_scat_emis=init_dict['do_scat_emis'],
                    lbl_opacity_sampling=init_dict['lbl_opacity_sampling']
                )
                retrieval_parameters[key].setup_opa_structure(init_dict['press'] * 1e-6)

            print(f'rank {rank} waiting...')
            comm.barrier()
        else:
            retrieval_parameters[key] = comm.bcast(retrieval_parameters[key], root=0)

    retrieval_directory = comm.bcast(retrieval_directory, root=0)

    if rank == 0:
        print("Bcasting done !!!")

    # # Check if all observations are the same
    # obs_tmp = comm.allgather(retrieval_parameters['observed_spectra'])
    #
    # for obs_tmp_proc in obs_tmp[1:]:
    #     assert np.allclose(obs_tmp_proc, obs_tmp[0], atol=0, rtol=1e-15)

    # print(f"Shared observations consistency check OK")

    # Initialize retrieval
    run_definitions = init_run(**retrieval_parameters)

    retrieval = Retrieval(
        run_definitions,
        output_dir=retrieval_directory,
        sample_spec=False,
        ultranest=False,
        pRT_plot_style=False
    )

    retrieval.run(
        sampling_efficiency=0.8,
        n_live_points=n_live_points,
        const_efficiency_mode=False,
        resume=False
    )

    if rank == 0:
        sample_dict, parameter_dict = retrieval.get_samples(
            output_dir=retrieval_directory + os.path.sep,
            ret_names=[retrieval_name]
        )

        n_param = len(parameter_dict[retrieval_name])
        parameter_plot_indices = {retrieval_name: np.arange(0, n_param)}

        true_values = {retrieval_name: []}

        for p in parameter_dict[retrieval_name]:
            true_values[retrieval_name].append(np.mean(retrieval_parameters['parameters'][p]))

        fig = contour_corner(
            sample_dict, parameter_dict, os.path.join(retrieval_directory, f'corner_{retrieval_name}.png'),
            parameter_plot_indices=parameter_plot_indices,
            true_values=true_values, prt_plot_style=False
        )

        fig.show()


def _main_hd2(sim_id=0):
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    module_dir = os.path.abspath(os.path.dirname(__file__))

    star_name = 'WASP-82'
    planet_name = star_name + ' b'
    planet = Planet.get(planet_name)

    # Paul's setup for HD 189733
    # planet.equilibrium_temperature = 1209

    line_species_str = [
        ['CO_main_iso', 'CO_36', 'CH4_main_iso', 'H2O_main_iso'],
        ['CO_main_iso', 'CH4_main_iso', 'H2O_main_iso'],
        ['CO_36', 'CH4_main_iso', 'H2O_main_iso'],
        ['CO_main_iso', 'CO_36', 'H2O_main_iso'],
        ['CO_main_iso', 'CO_36', 'CH4_main_iso']
    ]

    retrieval_names = [
        f't{planet_name}{sim_id}_12CO_13CO_CH4_H2O_t_18-28_full',
        f't{planet_name}{sim_id}_12CO_13CO_CH4_H2O_t_18-28_noCO_36',
        f't{planet_name}{sim_id}_12CO_13CO_CH4_H2O_t_18-28_noCO_26',
        f't{planet_name}{sim_id}_12CO_13CO_CH4_H2O_t_18-28_noCH4',
        f't{planet_name}{sim_id}_12CO_13CO_CH4_H2O_t_18-28_noH2O',
    ]

    if not os.path.isdir(
            os.path.join(module_dir, '..', '__tmp', 'test_retrieval', 'bins_' + planet_name.lower().replace(' ', '_'))
    ):
        os.mkdir(
            os.path.join(module_dir, '..', '__tmp', 'test_retrieval', 'bins_' + planet_name.lower().replace(' ', '_'))
        )

    mode = 'transit'
    n_live_points = 100
    add_noise = False

    retrieval_directories = os.path.abspath(os.path.join(module_dir, '..', '__tmp', 'test_retrieval'))

    load_from = None
    # load_from = os.path.join(retrieval_directories, f't0_kp_vr_CO_H2O_79-80_{mode}_200lp_np')
    # load_from = os.path.join(retrieval_directories, f't{sim_id}_vttt_p_kp_vr_CO_H2O_79-80_{mode}_30lp')
    # load_from = os.path.join(retrieval_directories, f't1_tt2_p_mr_kp_vr_CO_H2O_79-80_{mode}_100lp')

    band = 'K'

    wavelengths_borders = {
        'K': [1.8, 2.8],
        # 'K': [2.28, 2.42],
        # 'K': [2.15, 2.4],
        # 'M': [4.79, 4.80]  # [4.5, 5.5],
    }

    wrange_0 = wavelengths_borders[band] * np.array([1.001, 0.999])
    Nwranges = 100
    wranges = np.linspace(wrange_0[0], wrange_0[1], int(Nwranges + 1))
    wavelengths_borders[band] = np.array([wranges[sim_id], wranges[sim_id + 1]])

    if wavelengths_borders[band][-1] > 2.5:
        print("End of transmission file")
        return 0

    integration_times_ref = {
        'K': 60,
        # 'M': 76.89
    }

    snr_file = os.path.join(module_dir, 'andes', star_name.replace(' ', '_'), f"ANDES_snrs.npz")
    telluric_transmittance = os.path.join(module_dir, 'andes', 'sky', 'sky', 'transmission',
                                          f"transmission_1500_2500.dat")
    # airmass = os.path.join(module_dir, 'carmenes', 'hd_189733_b', 'air.npy')
    airmass = os.path.join(module_dir, 'andes', star_name.replace(' ', '_'), 'airmass_optimal.txt')
    variable_throughput = os.path.join(module_dir, 'metis', 'brogi_crires_test')

    wavelengths_instrument = None
    instrument_snr = None
    plot = True
    instrument_resolving_power = 1e5

    for j, ls_str in enumerate(line_species_str):
        if rank == j:
            print(f"rank {j}, line list {ls_str}, wrange = {wavelengths_borders[band]}")
            # Initialize parameters
            retrieval_directory = os.path.abspath(
                os.path.join(module_dir, '..', '__tmp', 'test_retrieval',
                             'bins_' + planet_name.lower().replace(' ', '_'),
                             retrieval_names[j])
            )

            retrieval_name, retrieval_directory, \
                model, pressures, true_parameters, line_species, rayleigh_species, continuum_species, \
                retrieval_model, \
                wavelength_instrument, reduced_mock_observations, error \
                = init_mock_observations(
                    planet, ls_str, mode,
                    retrieval_directory, retrieval_names[j], n_live_points,
                    add_noise, band, wavelengths_borders, integration_times_ref,
                    wavelengths_instrument=wavelengths_instrument, instrument_snr=instrument_snr, snr_file=snr_file,
                    telluric_transmittance=telluric_transmittance, airmass=airmass,
                    variable_throughput=variable_throughput,
                    instrument_resolving_power=instrument_resolving_power,
                    load_from=load_from, plot=plot
                )


def main_hd3(planet, sim_id):
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_proc = comm.Get_size()

    planet = a

    module_dir = os.path.abspath(os.path.dirname(__file__))

    mode = 'transit'
    n_live_points = 100

    star_names = [
        'HD 189733',
        'WASP-82',
        'HAT-P-46',
        'TOI-1130',
        'LHS 1140'
    ]

    planet_names = []
    planets = []

    for star_name in star_names:
        planet_names.append(star_name + ' b')
        planets.append(Planet.get(planet_names[-1]))

        if planets[-1].name == 'HD 189733 b':  # Paul's setup for HD 189733
            planets[-1].equilibrium_temperature = 1209

    line_species_strs = [
        ['CO_main_iso', 'CO_36', 'CH4_main_iso', 'H2O_main_iso'],
        ['CO_main_iso', 'CH4_main_iso', 'H2O_main_iso'],
        ['CO_36', 'CH4_main_iso', 'H2O_main_iso'],
        ['CO_main_iso', 'CO_36', 'H2O_main_iso'],
        ['CO_main_iso', 'CO_36', 'CH4_main_iso']
    ]

    retrieval_base_names = []

    for planet in planets:
        for line_species_str in line_species_strs:
            retrieval_species_names = []

            for species in line_species_str:
                species.replace('main_iso', '')
                retrieval_species_names.append(species)

            retrieval_base_names.append(
                f"{planet.name.lower().replace(' ', '_')}_"
                f"t_1.8-2.5um_"
                f"{'_'.join(retrieval_species_names)}_"
            )

    if not os.path.isdir(
            os.path.join(module_dir, '..', '__tmp', 'test_retrieval', 'bins_' + planet_name.lower().replace(' ', '_'))
    ):
        os.mkdir(
            os.path.join(module_dir, '..', '__tmp', 'test_retrieval', 'bins_' + planet_name.lower().replace(' ', '_'))
        )

    add_noise = False

    retrieval_directories = os.path.abspath(os.path.join(module_dir, '..', '__tmp', 'test_retrieval'))

    load_from = None
    # load_from = os.path.join(retrieval_directories, f't0_kp_vr_CO_H2O_79-80_{mode}_200lp_np')
    # load_from = os.path.join(retrieval_directories, f't{sim_id}_vttt_p_kp_vr_CO_H2O_79-80_{mode}_30lp')
    # load_from = os.path.join(retrieval_directories, f't1_tt2_p_mr_kp_vr_CO_H2O_79-80_{mode}_100lp')

    band = 'K'

    wavelengths_borders = {
        'K': [1.8, 2.8],
        # 'K': [2.28, 2.42],
        # 'K': [2.15, 2.4],
        # 'M': [4.79, 4.80]  # [4.5, 5.5],
    }

    wrange_0 = wavelengths_borders[band] * np.array([1.001, 0.999])
    Nwranges = 100
    wranges = np.linspace(wrange_0[0], wrange_0[1], int(Nwranges + 1))
    wavelengths_borders[band] = np.array([wranges[sim_id], wranges[sim_id + 1]])

    if wavelengths_borders[band][-1] > 2.5:
        print("End of transmission file")
        return 0

    integration_times_ref = {
        'K': 60,
        # 'M': 76.89
    }

    snr_file = os.path.join(module_dir, 'andes', star_name.replace(' ', '_'), f"ANDES_snrs.npz")
    telluric_transmittance = os.path.join(module_dir, 'andes', 'sky', 'sky', 'transmission',
                                          f"transmission_1500_2500.dat")
    # airmass = os.path.join(module_dir, 'carmenes', 'hd_189733_b', 'air.npy')
    airmass = os.path.join(module_dir, 'andes', star_name.replace(' ', '_'), 'airmass_optimal.txt')
    variable_throughput = os.path.join(module_dir, 'metis', 'brogi_crires_test')

    wavelengths_instrument = None
    instrument_snr = None
    plot = True
    instrument_resolving_power = 1e5

    for j, ls_str in enumerate(line_species_str):
        if rank == j:
            print(f"rank {j}, line list {ls_str}, wrange = {wavelengths_borders[band]}")
            # Initialize parameters
            retrieval_directory = os.path.abspath(
                os.path.join(module_dir, '..', '__tmp', 'test_retrieval',
                             'bins_' + planet_name.lower().replace(' ', '_'),
                             retrieval_names[j])
            )

            retrieval_name, retrieval_directory, \
                model, pressures, true_parameters, line_species, rayleigh_species, continuum_species, \
                retrieval_model, \
                wavelength_instrument, reduced_mock_observations, error \
                = init_mock_observations(
                    planet, ls_str, mode,
                    retrieval_directory, retrieval_names[j], n_live_points,
                    add_noise, band, wavelengths_borders, integration_times_ref,
                    wavelengths_instrument=wavelengths_instrument, instrument_snr=instrument_snr, snr_file=snr_file,
                    telluric_transmittance=telluric_transmittance, airmass=airmass,
                    variable_throughput=variable_throughput,
                    instrument_resolving_power=instrument_resolving_power,
                    load_from=load_from, plot=plot
                )


if __name__ == '__main__':
    t0 = time.time()

    args = parser.parse_args()

    for i in range(1):
        print(f'====\n sim {i + 1}')
        #main_hd3(planet=args.planet, sim_id=i + 1)
        main_hd2(sim_id=i + 1)
        print(f'====\n')
        plt.close('all')
    # main(sim_id=16)
    print(f"Done in {time.time() - t0} s.")
