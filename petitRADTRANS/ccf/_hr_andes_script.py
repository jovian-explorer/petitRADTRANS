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
import os.path
import time
from pathlib import Path

import numpy as np
from mpi4py import MPI

from petitRADTRANS.ccf.high_resolution_retrieval_HD_189733_b import init_mock_observations, init_run
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
    '--no-rewrite',
    action='store_false',
    help='if activated, do not re-run retrievals already performed'
)

parser.add_argument(
    '--resume',
    action='store_true',
    help='if activated, resume retrievals'
)


def init_and_run_retrieval(comm, rank, planet, line_species_str, mode, retrieval_directory, retrieval_name,
                           n_live_points, add_noise, wavelengths_borders, integration_times_ref, wavelengths_instrument,
                           instrument_snr, snr_file, telluric_transmittance, airmass, variable_throughput,
                           instrument_resolving_power, load_from, plot, rewrite=True, resume=False):
    run_time_file = os.path.join(retrieval_directory, 'run_time.npz')

    if not rewrite and os.path.isfile(run_time_file):
        if rank == 0:
            print(f"Retrieval '{retrieval_name}' already performed, skipping")

        comm.barrier()

        return
    elif os.path.isfile(run_time_file):
        if rank == 0:
            print(f"Re-running already performed retrieval '{retrieval_name}'...")

    t_start = time.time()

    # Initialize models
    if rank == 0:
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
                    init_dict[pkey] = ['H2-H2', 'H2-He']
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
         rewrite=True, resume=False):
    # MPI initialization
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Manual initialization
    mode = 'transit'
    n_live_points = 15
    add_noise = False

    wavelengths_instrument = None
    instrument_snr = None
    plot = False
    instrument_resolving_power = 1e5
    integration_times_ref = 60

    load_from = None

    line_species_strs = [
        ['CO_main_iso', 'CO_36', 'CH4_main_iso', 'H2O_main_iso'],
        ['CO_main_iso', 'CH4_main_iso', 'H2O_main_iso'],
        ['CO_36', 'CH4_main_iso', 'H2O_main_iso'],
        ['CO_main_iso', 'CO_36', 'H2O_main_iso'],
        ['CO_main_iso', 'CO_36', 'CH4_main_iso']
    ]

    # Initialization
    wavelengths_borders = [wavelength_min, wavelength_max]
    star_name = planet_name.rsplit(' ', 1)[0]
    planet = Planet.get(planet_name)

    if planet.name == 'HD 189733 b':  # Paul's setup for HD 189733
        planet.equilibrium_temperature = 1209

    retrieval_base_names = []

    for line_species_str in line_species_strs:
        retrieval_species_names = []

        for species in line_species_str:
            species = species.replace('_main_iso', '')

            if species == 'CO_36':
                species = '13CO'

            retrieval_species_names.append(species)

        retrieval_base_names.append(
            f"{planet.name.lower().replace(' ', '_')}_"
            f"{mode}_{wavelength_min:.3f}-{wavelength_max:.3f}um_"
            f"{'_'.join(retrieval_species_names)}_{n_live_points}lp"
        )

    retrieval_output_directory = os.path.join(output_directory, 'bins_' + planet_name.lower().replace(' ', '_'))

    if not os.path.isdir(retrieval_output_directory):
        os.mkdir(retrieval_output_directory)

    snr_file = os.path.join(additional_data_directory, star_name.replace(' ', '_'), f"ANDES_snrs.npz")
    telluric_transmittance = os.path.join(additional_data_directory, 'sky', 'transmission',
                                          f"transmission_1500_2500.dat")
    airmass = os.path.join(additional_data_directory, star_name.replace(' ', '_'), 'airmass_optimal.txt')
    variable_throughput = os.path.join(additional_data_directory, 'brogi_crires', 'algn.npy')

    # Retrievals
    for j, line_species_str in enumerate(line_species_strs):
        retrieval_directory = os.path.abspath(
            os.path.join(retrieval_output_directory, retrieval_base_names[j])
        )

        init_and_run_retrieval(
            comm, rank, planet, line_species_str, mode, retrieval_directory, retrieval_base_names[j],
            n_live_points, add_noise, wavelengths_borders, integration_times_ref, wavelengths_instrument,
            instrument_snr, snr_file, telluric_transmittance, airmass, variable_throughput,
            instrument_resolving_power, load_from, plot, rewrite, resume
        )


if __name__ == '__main__':
    t0 = time.time()

    args = parser.parse_args()

    main(
        planet_name=args.planet,
        output_directory=args.output_directory,
        additional_data_directory=args.additional_data_directory,
        wavelength_min=args.wavelength_min,
        wavelength_max=args.wavelength_max,
        rewrite=args.no_rewrite,
        resume=args.resume
    )

    print(f"Done in {time.time() - t0} s.")
