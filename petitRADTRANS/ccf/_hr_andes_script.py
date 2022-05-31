"""
Run with:
    mpiexec -n N --use-hwthread-cpus python3 _hr_andes_script.py --planet <planet_name>
Try:
    sudo mpiexec -n N --allow-run-as-root ...
If for some reason the script crashes.

N: number of cores in parallel run for PyMultiNest/MPI (tasks for slurm).
planet_name: planet name (e.g. 'HD 189733 b')
"""
import argparse
import json
import os.path
import time
from pathlib import Path

import numpy as np

from petitRADTRANS.ccf.high_resolution_retrieval_HD_189733_b import all_species, init_mock_observations, init_run, \
    get_retrieval_name
from petitRADTRANS.ccf.high_resolution_retrieval_TRAPPIST_1_b import init_mock_observations_co2
from petitRADTRANS.ccf.model_containers import Planet
from petitRADTRANS.radtrans import Radtrans
from petitRADTRANS.retrieval import Retrieval

# Arguments definition
parser = argparse.ArgumentParser(
    description='Launch HR retrieval script'
)

parser.add_argument(
    '--planet',
    default='HD 189733 b',
    help='planet name '
)

parser.add_argument(
    '--output-directory',
    default=Path.home(),
    help='directory where to save the results'
)

parser.add_argument(
    '--additional-data-directory',
    default=Path.home(),
    help='directory where the additional data are stored'
)

parser.add_argument(
    '--wavelength-min',
    type=float,
    default=1.8,
    help='(um) minimum wavelength'
)

parser.add_argument(
    '--wavelength-max',
    type=float,
    default=2.5,
    help='(um) maximum wavelength'
)

parser.add_argument(
    '--mode',
    default='transit',
    help='spectral model mode, eclipse or transit'
)

parser.add_argument(
    '--n-live-points',
    type=int,
    default=100,
    help='number of live points to use in the retrieval'
)

parser.add_argument(
    '--n-transits',
    type=float,
    default=1.0,
    help='number of planetary transits'
)

parser.add_argument(
    '--co2',
    action='store_true',
    help='if activated, switch to the full CO2 atmosphere mode'
)

parser.add_argument(
    '--no-rewrite',
    action='store_false',
    help='if activated, do not re-run retrievals already performed'
)

parser.add_argument(
    '--resume',
    action='store_true',
    help='if activated, resume retrievals'
)

parser.add_argument(
    '--get-results',
    action='store_true',
    help='if activated, do not run the retrievals, get the results of multiple retrievals and store them in a file'
)

parser.add_argument(
    '--jobs-config-filename',
    help='filename of the jobs config file, used only with --get-results to get the results'
)

line_species_strs = [  # TODO put that into a file
    all_species,
    ['CO_main_iso', 'CH4_main_iso', 'H2S_main_iso', 'K', 'NH3_main_iso', 'Na_allard_new', 'PH3_main_iso', 'H2O_main_iso'],
    ['CO_36', 'CH4_main_iso', 'H2S_main_iso', 'K', 'NH3_main_iso', 'Na_allard_new', 'PH3_main_iso', 'H2O_main_iso'],
    ['CO_main_iso', 'CO_36', 'H2S_main_iso', 'K', 'NH3_main_iso', 'Na_allard_new', 'PH3_main_iso', 'H2O_main_iso'],
    ['CO_main_iso', 'CO_36', 'CH4_main_iso', 'H2S_main_iso', 'K', 'NH3_main_iso', 'Na_allard_new', 'PH3_main_iso'],
    ['CO_main_iso', 'CO_36', 'CH4_main_iso', 'K', 'NH3_main_iso', 'Na_allard_new', 'PH3_main_iso', 'H2O_main_iso'],
    ['CO_main_iso', 'CO_36', 'CH4_main_iso', 'H2S_main_iso', 'NH3_main_iso', 'Na_allard_new', 'PH3_main_iso', 'H2O_main_iso'],
    ['CO_main_iso', 'CO_36', 'CH4_main_iso', 'H2S_main_iso', 'K', 'Na_allard_new', 'PH3_main_iso', 'H2O_main_iso'],
    ['CO_main_iso', 'CO_36', 'CH4_main_iso', 'H2S_main_iso', 'K', 'NH3_main_iso', 'PH3_main_iso', 'H2O_main_iso'],
    ['CO_main_iso', 'CO_36', 'CH4_main_iso', 'H2S_main_iso', 'K', 'NH3_main_iso', 'Na_allard_new', 'H2O_main_iso']
]

line_species_strs_co2 = [
    ['CO2_main_iso'],
    []
]


def get_log_evidences(planet_name, output_directory, mode, n_live_points, n_transits,
                      jobs_config_filename, line_species):
    planet = Planet.get(planet_name)

    retrieval_output_directory = get_retrievals_directory(
        planet_name=planet_name,
        output_directory=output_directory
    )

    jobs_configuration = np.load(jobs_config_filename)
    wavelength_bins = jobs_configuration['wavelength_bins']

    # Get the log Ls and log Zs of the retrievals
    model_shape = (len(line_species), wavelength_bins.size - 1)

    log_ls = np.zeros(model_shape)
    chi2s = np.zeros(model_shape)

    log_zs = np.zeros(model_shape)
    log_zs_err = np.zeros(model_shape)
    nested_sampling_log_zs = np.zeros(model_shape)
    nested_sampling_log_zs_err = np.zeros(model_shape)
    nested_importance_sampling_log_zs = np.zeros(model_shape)
    nested_importance_sampling_log_zs_err = np.zeros(model_shape)

    run_times = np.zeros(model_shape)

    # Empty arrays because number of modes in each retrieval is unknown and can vary
    modes_strictly_local_log_zs = np.empty(model_shape, dtype=object)
    modes_strictly_local_log_zs_err = np.empty(model_shape, dtype=object)
    modes_local_log_zs = np.empty(model_shape, dtype=object)
    modes_local_log_zs_err = np.empty(model_shape, dtype=object)

    for i in range(wavelength_bins.size - 1):
        print(f"Fetching bin {i + 1}/{wavelength_bins.size - 1}: {wavelength_bins[i]}-{wavelength_bins[i + 1]} um...")

        model_retrievals_names = get_retrieval_base_names(
            planet=planet,
            mode=mode,
            wavelength_min=wavelength_bins[i],
            wavelength_max=wavelength_bins[i + 1],
            n_live_points=n_live_points,
            line_species=line_species,
            exposure_time=planet.transit_duration * n_transits
        )

        for j, model_retrievals_name in enumerate(model_retrievals_names):
            retrieval_base_name = os.path.join(
                retrieval_output_directory,
                model_retrievals_name
            )

            model_parameters = np.load(os.path.join(retrieval_base_name, 'model_parameters.npz'))
            log_ls[j, i] = model_parameters['true_log_l']
            chi2s[j, i] = model_parameters['true_chi2']

            run_time = np.load(os.path.join(retrieval_base_name, 'run_time.npz'))
            run_times[j, i] = run_time['run_time']

            with open(os.path.join(retrieval_base_name, 'out_PMN', model_retrievals_name + '_stats.json'), 'r') as f:
                stats = json.load(f)

            modes = np.empty(len(stats['modes']), dtype=object)

            for m in stats['modes']:
                modes[m['index']] = m

            modes_strictly_local_log_z = np.zeros(modes.size)
            modes_strictly_local_log_z_err = np.zeros(modes.size)
            modes_local_log_z = np.zeros(modes.size)
            modes_local_log_z_err = np.zeros(modes.size)

            for k, m in enumerate(modes):
                modes_strictly_local_log_z[k] = m['strictly local log-evidence']
                modes_strictly_local_log_z_err[k] = m['strictly local log-evidence error']
                modes_local_log_z[k] = m['local log-evidence']
                modes_local_log_z_err[k] = m['local log-evidence error']

            modes_strictly_local_log_zs[j, i] = modes_strictly_local_log_z
            modes_strictly_local_log_zs_err[j, i] = modes_strictly_local_log_z_err
            modes_local_log_zs[j, i] = modes_local_log_z
            modes_local_log_zs_err[j, i] = modes_local_log_z_err

            log_zs[j, i] = stats['global evidence']
            log_zs_err[j, i] = stats['global evidence error']
            nested_sampling_log_zs[j, i] = stats['nested sampling global log-evidence']
            nested_sampling_log_zs_err[j, i] = stats['nested sampling global log-evidence error']
            nested_importance_sampling_log_zs[j, i] = stats['nested importance sampling global log-evidence']
            nested_importance_sampling_log_zs_err[j, i] = stats['nested importance sampling global log-evidence error']

    file_basename = get_retrieval_base_names(
        planet=planet,
        mode=mode,
        wavelength_min=wavelength_bins[0],
        wavelength_max=wavelength_bins[-1],
        n_live_points=n_live_points,
        exposure_time=planet.transit_duration * n_transits,
        line_species=line_species
    )[0] + f"_{len(line_species) - 1}bins_"

    result_filename = os.path.join(retrieval_output_directory, file_basename + "log_evidences.npz")

    print(f"Saving log-evidences in file '{result_filename}'")

    np.savez_compressed(
        file=result_filename,
        planet=planet_name,
        mode=mode,
        n_live_points=n_live_points,
        n_transits=n_transits,
        models=[str(species_included) for species_included in line_species],
        wavelength_bins=wavelength_bins,
        true_log_likelihoods=log_ls,
        true_chi2s=chi2s,
        global_log_evidences=log_zs,
        global_log_evidences_error=log_zs_err,
        nested_sampling_global_log_evidences=nested_sampling_log_zs,
        nested_sampling_global_log_evidences_error=nested_sampling_log_zs_err,
        nested_importance_sampling_global_log_evidences=nested_importance_sampling_log_zs,
        nested_importance_sampling_global_log_evidences_error=nested_importance_sampling_log_zs_err,
        run_times=run_times
    )

    # Saving modes log Zs into a separate file to avoid pickle usage in the global log Zs file
    result_filename = os.path.join(retrieval_output_directory, file_basename + 'mode_log_evidences.json')
    print(f"Saving modes log-evidences in file '{result_filename}'")

    save_dict = {}

    for i, model in enumerate(line_species):
        model = str(model)
        save_dict[model] = {}

        for j, wavelength in enumerate(wavelength_bins[:-1]):
            save_dict[model][wavelength] = {}

            for k in range(len(modes_strictly_local_log_zs[i, j])):
                save_dict[model][wavelength][k] = {
                    'strictly local log-evidence': modes_strictly_local_log_zs[i, j][k],
                    'strictly local log-evidence error': modes_strictly_local_log_zs_err[i, j][k],
                    'local log-evidence': modes_local_log_zs[i, j][k],
                    'local log-evidence error': modes_local_log_zs_err[i, j][k],
                }

    with open(result_filename, 'w') as f:
        json.dump(
            obj=save_dict,
            fp=f
        )


def get_retrieval_base_names(planet, mode, wavelength_min, wavelength_max, n_live_points, exposure_time,
                             line_species=None):
    if line_species is None:
        line_species = line_species_strs

    retrieval_base_names = []

    for line_species_str in line_species:
        retrieval_species_names = []

        for species in line_species_str:
            species = species.replace('_main_iso', '')

            if species == 'CO_36':
                species = '13CO'

            retrieval_species_names.append(species)

        retrieval_base_names.append(
            get_retrieval_name(
                planet=planet,
                mode=mode,
                wavelength_min=wavelength_min,
                wavelength_max=wavelength_max,
                retrieval_species_names=retrieval_species_names,
                n_live_points=n_live_points,
                exposure_time=exposure_time
            )
        )

    return retrieval_base_names


def get_retrievals_directory(planet_name, output_directory):
    return os.path.join(output_directory, 'bins_' + planet_name.lower().replace(' ', '_'))


def init_and_run_retrieval(comm, rank, planet, line_species_str, mode, retrieval_directory, retrieval_name,
                           n_live_points, add_noise, wavelengths_borders, integration_times_ref, n_transits,
                           wavelengths_instrument,
                           instrument_snr, snr_file, telluric_transmittance, airmass, variable_throughput,
                           instrument_resolving_power, load_from, plot, retrieved_species=None, co2_mode=False,
                           rewrite=True, resume=False):
    run_time_file = os.path.join(retrieval_directory, 'run_time.npz')

    if not rewrite and os.path.isfile(run_time_file):
        if rank == 0:
            print(f"Retrieval '{retrieval_name}' already performed, skipping")

        comm.barrier()

        return
    elif os.path.isfile(run_time_file):
        if rank == 0:
            print(f"Re-running already performed retrieval '{retrieval_name}'...")
    else:
        if rank == 0:
            print(f"Running retrieval '{retrieval_name}'")

    if co2_mode:
        init_mo = init_mock_observations_co2
    else:
        init_mo = init_mock_observations

    t_start = time.time()

    # Initialize models
    if rank == 0:
        retrieval_name, retrieval_directory, \
            model, pressures, true_parameters, line_species, rayleigh_species, continuum_species, \
            retrieval_model, \
            wavelength_instrument, reduced_mock_observations, error \
            = init_mo(
                planet=planet,
                line_species_str=line_species_str,
                mode=mode,
                retrieval_directory=retrieval_directory,
                retrieval_name=retrieval_name,
                add_noise=add_noise,
                wavelengths_borders=wavelengths_borders,
                integration_times_ref=integration_times_ref,
                n_transits=n_transits,
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

        if retrieved_species is None:
            retrieved_species = line_species

        retrieval_parameters = {
            'retrieval_name': retrieval_name,
            'pressures': pressures,
            'retrieved_species': retrieved_species,
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

    for key in retrieval_parameters:
        if key == 'prt_object':
            # Create Radtrans initialization dict
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

            # Broadcast key values form main process
            for pkey in init_dict:
                if rank == 0:
                    print(f'Broadcasting Radtrans init key {pkey}...')

                if pkey == 'continuum_opacities':
                    if not co2_mode:
                        init_dict[pkey] = ['H2-H2', 'H2-He']
                    else:
                        init_dict[pkey] = ['CO2-CO2']
                else:
                    init_dict[pkey] = comm.bcast(retrieval_parameters[key].__dict__[pkey], root=0)

            # Initialize the Radtrans object in each processes
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
        print("Bcasting done!")

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
        resume=resume
    )

    # Save the time needed to perform the retrieval, can also be used to test if the retrieval ended correctly
    if rank == 0:
        np.savez_compressed(
            run_time_file,
            run_time_units='s',
            run_time=time.time() - t_start
        )


def main(planet_name, output_directory, additional_data_directory, wavelength_min, wavelength_max,
         mode, n_live_points, n_transits=1.0, co2_mode=False,
         rewrite=True, resume=False):
    from mpi4py import MPI

    print('Initializing...')
    # MPI initialization
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Manual initialization
    instrument_resolving_power = 1e5
    integration_times_ref = 60  # (s) for ANDES, this is (until more thoughts are put into this) an arbitrary value

    add_noise = False
    wavelengths_instrument = None
    instrument_snr = None
    plot = False

    load_from = None

    # Initialization
    wavelengths_borders = [wavelength_min, wavelength_max]
    star_name = planet_name.rsplit(' ', 1)[0]
    planet = Planet.get(planet_name)

    if planet.name == 'HD 189733 b':  # Paul's setup for HD 189733
        planet.equilibrium_temperature = 1209

    if co2_mode:
        line_species = line_species_strs_co2
    else:
        line_species = line_species_strs

    retrieval_base_names = []

    for line_species_str in line_species:
        retrieval_species_names = []

        for species in line_species_str:
            species = species.replace('_main_iso', '')

            if species == 'CO_36':
                species = '13CO'

            retrieval_species_names.append(species)

        retrieval_base_names.append(
            get_retrieval_name(
                planet=planet,
                mode=mode,
                wavelength_min=wavelength_min,
                wavelength_max=wavelength_max,
                retrieval_species_names=retrieval_species_names,
                n_live_points=n_live_points,
                exposure_time=planet.transit_duration * n_transits
            )
        )

    retrieval_output_directory = get_retrievals_directory(
        planet_name=planet_name,
        output_directory=output_directory
    )

    if rank == 0:
        if not os.path.isdir(retrieval_output_directory):
            os.mkdir(retrieval_output_directory)

    snr_file = os.path.join(additional_data_directory, star_name.replace(' ', '_'), f"ANDES_snrs.npz")
    telluric_transmittance = os.path.join(additional_data_directory, 'sky', 'transmission',
                                          f"transmission_1500_2500.dat")
    airmass = os.path.join(additional_data_directory, star_name.replace(' ', '_'), 'airmass_optimal.txt')
    variable_throughput = os.path.join(additional_data_directory, 'brogi_crires', 'algn.npy')

    # Retrievals
    for j, line_species_str in enumerate(line_species):
        if co2_mode:
            retrieved_species = line_species_str
        else:
            retrieved_species = None  # default

        retrieval_directory = os.path.abspath(
            os.path.join(retrieval_output_directory, retrieval_base_names[j])
        )

        init_and_run_retrieval(
            comm, rank, planet, line_species_str, mode, retrieval_directory, retrieval_base_names[j],
            n_live_points, add_noise, wavelengths_borders, integration_times_ref, n_transits, wavelengths_instrument,
            instrument_snr, snr_file, telluric_transmittance, airmass, variable_throughput,
            instrument_resolving_power, load_from, plot, retrieved_species, co2_mode, rewrite, resume
        )


if __name__ == '__main__':
    t0 = time.time()

    args = parser.parse_args()

    if not args.get_results:
        main(
            planet_name=args.planet,
            output_directory=args.output_directory,
            additional_data_directory=args.additional_data_directory,
            wavelength_min=args.wavelength_min,
            wavelength_max=args.wavelength_max,
            mode=args.mode,
            n_live_points=args.n_live_points,
            n_transits=args.n_transits,
            co2_mode=args.co2,
            rewrite=args.no_rewrite,
            resume=args.resume
        )
    else:
        if args.co2:
            line_strs = line_species_strs_co2
        else:
            line_strs = line_species_strs

        get_log_evidences(
            planet_name=args.planet,
            output_directory=args.output_directory,
            mode=args.mode,
            n_live_points=args.n_live_points,
            n_transits=args.n_transits,
            jobs_config_filename=args.jobs_config_filename,
            line_species=line_strs
        )

    print(f"Done in {time.time() - t0} s.")
