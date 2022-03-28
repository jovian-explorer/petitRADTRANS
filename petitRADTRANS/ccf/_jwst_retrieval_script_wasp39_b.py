"""
Run with:
    mpiexec -n N --use-hwthread-cpus python3 _test_high_resolution.py
N is the number of processes.
Try:
    sudo mpiexec -n N --allow-run-as-root ...
If for some reason the script crashes.
"""
import sys
import time

import numpy as np

from petitRADTRANS.ccf.jwst_retrieval_wasp39_b import *
from petitRADTRANS.ccf.model_containers import RetrievalSpectralModel
from petitRADTRANS.ccf.spectra_utils import load_snr_file
from petitRADTRANS.retrieval import Retrieval
from petitRADTRANS.retrieval.plotting import contour_corner
import matplotlib.pyplot as plt


def main(sim_id=0):
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    module_dir = os.path.abspath(os.path.dirname(__file__))

    planet_name = 'WASP-39 b'
    planet = Planet.get(planet_name)

    line_species_str = ['CO_all_iso_HITEMP_R_120', 'CO2_R_120', 'H2O_Exomol_R_120']#['CO_all_iso', 'CO2_main_iso', 'H2O_main_iso']
    # line_species_str = ['CO_all_iso', 'H2O_main_iso']

    retrieval_name = f't{planet_name}{sim_id}_jwst_p_kp_vr_CO_CO2_H2O_t_NIRCam_F322W2'
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

    snr_file = {
        # 'NIRSpec_PRISM': os.path.join(module_dir, 'JWST', f"W39b_NIRSpec_PRISM.p"),
        # 'NIRISS_SOSS': os.path.join(module_dir, 'JWST', f"W39b_NIRISS_SOSS.p"),
        'NIRCam_F322W2': os.path.join(module_dir, 'JWST', f"W39b_NIRCam_F322W2.p"),
        # 'NIRSpec_G395H': os.path.join(module_dir, 'JWST', f"W39b_NIRSpec_G395H.p"),
    }
    telluric_transmittance = None
    airmass = None
    variable_throughput = None #os.path.join(module_dir, 'metis', 'brogi_crires_test')

    wavelengths_instrument = None
    instrument_snr = None
    plot = True
    instrument_resolving_power = 8e4
    target_instrument_resolving_power = 200

    if rank == 0:
        # Initialize parameters
        retrieval_model_ = []
        wavelength_instrument_ = []
        reduced_mock_observations_ = []
        error_ = []
        model_ = []
        true_parameters_ = {}
        pressures = None
        line_species = None
        rayleigh_species = None
        continuum_species = None

        for instrument_name, file in snr_file.items():
            retrieval_name, retrieval_directory, \
                model, pressures, true_parameters, line_species, rayleigh_species, continuum_species, \
                retrieval_model, \
                wavelength_instrument, reduced_mock_observations, error \
                = init_mock_observations(
                    planet, line_species_str, mode,
                    retrieval_directory, retrieval_name, n_live_points,
                    add_noise, band, wavelengths_borders, integration_times_ref,
                    wavelengths_instrument=wavelengths_instrument, instrument_snr=instrument_snr, snr_file=file,
                    telluric_transmittance=telluric_transmittance, airmass=airmass,
                    variable_throughput=variable_throughput,
                    instrument_resolving_power=instrument_resolving_power, instrument_name=instrument_name,
                    target_instrument_resolving_power=target_instrument_resolving_power,
                    load_from=load_from, plot=plot
                )

            retrieval_model_.append(retrieval_model)
            print(retrieval_model_[-1].__name__)
            wavelength_instrument_.append(wavelength_instrument)
            reduced_mock_observations_.append(reduced_mock_observations)
            error_.append(error)
            model_.append(model)

            for p, v in true_parameters.items():
                if p not in true_parameters_:
                    true_parameters_[p] = v

        retrieval_parameters = {
            'retrieval_name': retrieval_name,
            'pressures': pressures,
            'retrieved_species': line_species,
            'rayleigh_species': rayleigh_species,
            'continuum_species': continuum_species,
            'retrieval_model': retrieval_model_,
            'wavelengths_instrument': wavelength_instrument_,
            'observed_spectra': reduced_mock_observations_,
            'observations_uncertainties': error_,
            'prt_object': model_,
            'parameters': true_parameters_
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

            if rank == 0:
                number_models = np.size(retrieval_parameters[key])
            else:
                number_models = None

            number_models = comm.bcast(number_models, root=0)

            if rank != 0:
                retrieval_parameters[key] = [retrieval_parameters[key]] * number_models

            print(f'rank {rank} has {len(retrieval_parameters[key])} models')
            comm.barrier()

            for m in range(number_models):
                for pkey in init_dict:
                    if rank == 0:
                        print(f'Broadcasting Radtrans init key {pkey} for model {m}...')

                    if pkey == 'continuum_opacities':
                        init_dict[pkey] = ['H2-H2', 'H2-He']
                    else:
                        init_dict[pkey] = comm.bcast(retrieval_parameters[key][m].__dict__[pkey], root=0)

                if rank != 0:
                    retrieval_parameters[key][m] = Radtrans(
                        line_species=init_dict['line_species'],
                        rayleigh_species=init_dict['rayleigh_species'],
                        continuum_opacities=init_dict['continuum_opacities'],
                        wlen_bords_micron=init_dict['wlen_bords_micron'],
                        mode=init_dict['mode'],
                        do_scat_emis=init_dict['do_scat_emis'],
                        lbl_opacity_sampling=init_dict['lbl_opacity_sampling']
                    )
                    retrieval_parameters[key][-1].setup_opa_structure(init_dict['press'] * 1e-6)

                print(f'rank {rank} waiting...')
                comm.barrier()
        else:
            retrieval_parameters[key] = comm.bcast(retrieval_parameters[key], root=0)
            print(f'rank {rank} has {len(retrieval_parameters[key])} {key}')

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


if __name__ == '__main__':
    t0 = time.time()
    for i in [1]:
        print(f'====\n sim {i + 1}')
        main(sim_id=i + 1)
        print(f'====\n')
        plt.close('all')
    # main(sim_id=16)
    print(f"Done in {time.time() - t0} s.")
