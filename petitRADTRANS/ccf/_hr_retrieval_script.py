"""
Run with:
    mpiexec -n N --use-hwthread-cpus python3 _test_high_resolution.py
N is the number of processes.
Try:
    sudo mpiexec -n N --allow-run-as-root ...
If for some reason the script crashes.
"""
import time

from petitRADTRANS.ccf.high_resolution_retrieval import *
from petitRADTRANS.retrieval import Retrieval
from petitRADTRANS.retrieval.plotting import contour_corner
import matplotlib.pyplot as plt


def main(sim_id=0):
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    module_dir = os.path.abspath(os.path.dirname(__file__))

    planet_name = 'HD 209458 b'
    planet = Planet.get(planet_name)

    line_species_str = ['CO_all_iso', 'H2O_main_iso']

    retrieval_name = f't{sim_id}_vttt_mbrogimd_kp_vr_CO_H2O_79-80'
    retrieval_directory = os.path.abspath(os.path.join(module_dir, '..', '__tmp', 'test_retrieval', retrieval_name))

    mode = 'transit'
    n_live_points = 100
    add_noise = True

    retrieval_directories = os.path.abspath(os.path.join(module_dir, '..', '__tmp', 'test_retrieval'))

    # load_from = None
    # load_from = os.path.join(retrieval_directories, f't0_kp_vr_CO_H2O_79-80_{mode}_200lp_np')
    load_from = os.path.join(retrieval_directories, f't{sim_id}_vttt_p_kp_vr_CO_H2O_79-80_{mode}_30lp')
    # load_from = os.path.join(retrieval_directories, f't1_tt2_p_mr_kp_vr_CO_H2O_79-80_{mode}_100lp')

    band = 'M'

    wavelengths_borders = {
        'L': [2.85, 4.20],
        'M': [4.79, 4.80]  # [4.5, 5.5],
    }

    integration_times_ref = {
        'L': 20.83,
        'M': 76.89
    }

    star_name = planet.host_name.replace(' ', '_')
    snr_file = os.path.join(module_dir, 'metis', 'SimMETIS', star_name,
                            f"{star_name}_SNR_{band}-band_calibrated.txt")
    telluric_transmittance = \
        os.path.join(module_dir, 'metis', 'skycalc', 'transmission_3060m_4750-4850nm_R150k_FWHM1.5_default.dat')
    airmass = None #os.path.join(module_dir, 'metis', 'brogi_crires_test', 'air.npy')
    variable_throughput = os.path.join(module_dir, 'metis', 'brogi_crires_test')

    wavelengths_instrument = None
    instrument_snr = None
    plot = True
    instrument_resolving_power = 1e5

    if rank == 0:
        # Initialize parameters
        '''
        For retrievals: the pipeline must be exactly the same, step-by-step, for both the data and the model.
        It is probable that any perturbation (telluric lines, variable throughput) must be mimicked in the model as 
        well, or very well removed by the pipeline.
        '''
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
                load_from=load_from, plot=plot
            )

        retrieval_parameters = {
            'retrieval_name': retrieval_name,
            'prt_object': model,
            'pressures': pressures,
            'parameters': true_parameters,
            'line_species': line_species,
            'rayleigh_species': rayleigh_species,
            'continuum_species': continuum_species,
            'retrieval_model': retrieval_model,
            'wavelengths_instrument': wavelength_instrument,
            'observed_spectra': reduced_mock_observations,
            'observations_uncertainties': error
        }

        retrieval_directory = retrieval_directory
    else:
        print(f"Rank {rank} waiting for main process to finish...")
        retrieval_parameters = None
        retrieval_directory = ''

    print('New')
    # return 0

    retrieval_parameters = comm.bcast(retrieval_parameters, root=0)
    retrieval_directory = comm.bcast(retrieval_directory, root=0)

    # Check if all observations are the same
    obs_tmp = comm.allgather(retrieval_parameters['observed_spectra'])

    for obs_tmp_proc in obs_tmp[1:]:
        assert np.allclose(obs_tmp_proc, obs_tmp[0], atol=0, rtol=1e-15)

    print(f"Shared observations consistency check OK")

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
    for i in [96]:
        print(f'====\n sim {i + 1}')
        main(sim_id=i + 1)
        print(f'====\n')
        plt.close('all')
    # main(sim_id=16)
    print(f"Done in {time.time() - t0} s.")
