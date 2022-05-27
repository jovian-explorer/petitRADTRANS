"""Script to generate SLURM run scripts.
This script will try to use one CPU per run if there is not enough nodes to launch all the runs, while maximising the
number of nodes used.

Glossary:
    - Cores are the number of hardware processing units (not threads) of a CPU. Cores communicate fast with each other.
    - Runs corresponds to the execution of a single srun command on one or multiple cores <=> tasks for MPI runs.
    - CPUs are the hardware processors installed in one node. CPUs can communicate relatively fast within one node.
    - Nodes are "separate" instances containing one or multiple CPUs. Nodes can communicate with each other, but slowly.
    - Clusters are ensembles of nodes.
    - Jobs are allocations of a number of nodes to execute one or several runs.
    - Calls are the request of one or multiple jobs at once.

Run with (one line):
    python _script_andes_slurm.py <path/to/template/file/script.bash> <path/to/python/script.py> \
        --output-directory <path> \
        --additional-data-directory <path> --planets <planets> --nwavelength-bins <N>
Example (one line):
    python _script_andes_slurm.py './my_slurm_script_template.sbatch' './my_prt_script.py' \
        --output-directory './my/output/dir' \
        --additional-data-directory './my/data/dir' --planets 'HD 189733 b' 'K2-18 b' --nwavelength-bins 100
"""

import argparse
import os.path
from pathlib import Path
import subprocess

import numpy as np

from petitRADTRANS.slurm_runs.sbatch import make_srun_script_from_template

# Arguments definition
parser = argparse.ArgumentParser(
    description='Launch HR retrieval script'
)

parser.add_argument(
    'template_filename',
    help='template file to use to generate the slurm batch scripts'
)

parser.add_argument(
    'python_script',
    help='python script to launch'
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
    '--planets',
    nargs='+',
    default='HD 189733 b',
    help='planet name '
)

parser.add_argument(
    '--nodes',
    type=int,
    default=1,
    help='maximum number of nodes to allocate for each job'
)

parser.add_argument(
    '--ncores-per-cpus',
    type=int,
    default=36,
    help='number of hardware cores per processors'
)

parser.add_argument(
    '--ncpus-per-nodes',
    type=int,
    default=2,
    help='number of processors per node'
)

parser.add_argument(
    '--njobs-per-calls',
    type=int,
    default=1,
    help='number of jobs to launch at the same time, use 0 or a negative value to put no limit'
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
    '--nwavelength-bins',
    type=int,
    default=70,
    help='number of wavelength bins within the wavelength range'
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
    '--co2',
    action='store_true',
    help='if activated, switch to the full CO2 atmosphere mode'
)

parser.add_argument(
    '--job-base-name',
    default='pRT_ANDES',
    help='number of wavelength bins within the wavelength range'
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
    '--no-mpi-control',
    action='store_false',
    help='if activated, assume that the cluster crashes when trying to call mpi'
)


# Functions
def fair_share(array, n_entities, append_within_existing=True):
    """Split the elements of an array into a given number of entities.

    Examples:
        fair_share([1, 2, 3], 2) => [[1, 3], [2]]
        fair_share([1, 2, 3], 3) => [[1], [2], [3]]
        fair_share([1, 2, 3, 4], 2) => [[1, 2], [3, 4]]
        fair_share([1, 2, 3, 4, 5], 2) => [[1, 2, 5], [3, 4]]
        fair_share([1, 2, 3, 4, 5], 2, False) => [[1, 2], [3, 4], [5]]

    Args:
        array: a numpy array
        n_entities: the number of entities
        append_within_existing: if True, leftover elements will be shared within the entities; otherwise, leftover
            elements will be added as an extra entity

    Returns:
        list with n_entities elements if append_within_existing is True, n_entities + 1 elements otherwise, each
        sharing elements of the original array.
    """
    elements_per_entities = int(np.floor(array.size / n_entities))
    n_leftover_elements = (array.size - elements_per_entities * n_entities)

    if array.size > n_leftover_elements:
        shared_array = list(
            array[:array.size - n_leftover_elements].reshape(
                n_entities, elements_per_entities
            )
        )
        leftover_elements = array[array.size - n_leftover_elements:]
    else:
        shared_array = [array]
        leftover_elements = np.array([])

    if leftover_elements.size > 0:
        if append_within_existing:
            for i, leftover_element in enumerate(leftover_elements):
                shared_array[np.mod(i, n_entities)] = np.append(shared_array[i], leftover_element)
        else:
            if array.size - n_leftover_elements <= n_entities:
                shared_array[0] = np.append(shared_array[0], leftover_elements)
            else:
                shared_array.append(leftover_elements)

    return shared_array


def main(python_script, template_filename, output_directory, additional_data_directory, planets, n_nodes,
         n_cores_per_cpus, n_cpus_per_nodes, n_jobs_per_calls, wavelength_min, wavelength_max, n_wavelength_bins,
         mode, n_live_points, co2_mode,
         job_basename='pRT_ANDES', rewrite=True, resume=False, mpi_control=True):
    # Generate bins array
    wavelengths_borders = [wavelength_min, wavelength_max]

    wavelength_bins = wavelengths_borders #* np.array([1.05, 0.95])
    wavelength_bins = np.linspace(wavelength_bins[0], wavelength_bins[1], int(n_wavelength_bins + 1))

    # Split bins/retrievals/runs by nodes, will use the same configuration for each planet
    run_ids = np.linspace(1, n_wavelength_bins, n_wavelength_bins, dtype=int)

    nodes_runs = fair_share(run_ids, n_nodes, append_within_existing=True)

    # Split bins by processors within each node, to get the list of jobs per node
    # Communication-wise it is more efficient to keep each run within a unique CPU, at the cost of slower individual run
    nodes_jobs_runs = []

    if not mpi_control:
        print("No MPI control, forcing the use of all the node CPUs")
        n_cores_per_cpus *= n_cpus_per_nodes
        n_cpus_per_nodes = 1

    for i, node_runs in enumerate(nodes_runs):
        n_jobs_per_node_min = np.max((int(np.floor(node_runs.size / n_cpus_per_nodes)), n_cpus_per_nodes))
        nodes_jobs_runs.append(fair_share(node_runs, n_jobs_per_node_min, append_within_existing=False))

    # Make run script files
    n_jobs_node = [len(node) for node in nodes_jobs_runs]  # number of jobs in each node
    n_jobs_node_max = np.max(n_jobs_node)

    if n_jobs_per_calls <= 0:
        print("All the jobs will be called at once.")
        n_jobs_per_calls = n_jobs_node_max

    for planet in planets:
        job_basename_planet = f"{job_basename}_{planet.lower().replace(' ', '_')}"
        job_names = []
        tasks_per_node = n_cpus_per_nodes * n_cores_per_cpus
        n_nodes_current_job = n_nodes

        # Loop over the number of jobs
        for i_job in range(n_jobs_node_max):
            srun_lines = []

            # Look for the corresponding job in each node
            for i, node_jobs in enumerate(nodes_jobs_runs):
                if i_job >= n_jobs_node[i]:  # node has fewer jobs than the maximum number of jobs
                    n_nodes_current_job -= 1  # one less node is required for this job

                    continue

                node_job = node_jobs[i_job]

                if mpi_control:
                    run_command = f"mpiexec -n {int(tasks_per_node / node_job.size)}"
                else:
                    run_command = f"srun -N 1 -n {tasks_per_node}"

                # Have one run per CPU instead of one run per node
                for cpu_run in node_job:
                    srun_lines.append(
                        f"{run_command} python3 {python_script} "
                        f"--planet '{planet}' "
                        f"--output-directory '{output_directory}' "
                        f"--additional-data-directory '{additional_data_directory}' "
                        f"--wavelength-min {wavelength_bins[cpu_run - 1]} "
                        f"--wavelength-max {wavelength_bins[cpu_run]} "
                        f"--mode '{mode}' "
                        f"--n-live-points {n_live_points}"
                    )

                    if co2_mode:
                        srun_lines[-1] += f" --co2"

                    if not rewrite:
                        srun_lines[-1] += f" --no-rewrite"

                    if resume:
                        srun_lines[-1] += f" --resume"

                    srun_lines[-1] += ' &\n'

            srun_lines.append('\nwait\n\nexit 0\n')

            # Make the script with all the runs
            job_names.append(f"{job_basename_planet}_job{i_job}")
            i_job += 1

            make_srun_script_from_template(
                filename=job_names[-1] + '.sbatch',
                template_filename=template_filename,
                job_name=job_names[-1],
                nodes=n_nodes_current_job,  # nodes per job
                tasks_per_node=tasks_per_node,
                cpus_per_task=1,
                time='0-12:00:00',
                srun_lines=srun_lines
            )

        # Save jobs configuration
        # This file can be used to easily collect the results after the call
        jobs_config_file = os.path.join(output_directory, job_basename_planet + '.npz')

        print(f"Saving jobs configuration for planet {planet} in file '{jobs_config_file}'...")
        np.savez_compressed(
            file=jobs_config_file,
            planet=planet,
            python_script=python_script,
            wavelength_bins=wavelength_bins
        )

        # Submit jobs
        print(f"Submitting jobs for planet {planet}...")
        command_line = ''
        i_job = 0

        while i_job < n_jobs_node_max:
            for i in range(n_jobs_per_calls):
                command_line += f"sbatch {job_names[i_job] + '.sbatch'}"

                if i == n_jobs_per_calls - 1:
                    # subprocess.run(command_line, shell=True)
                    subprocess.run(f"echo '{command_line}'", shell=True)
                    command_line = ''
                else:
                    command_line += ' && '  # assume that the script are launched from linux

                i_job += 1


if __name__ == '__main__':
    args = parser.parse_args()

    main(
        python_script=args.python_script,
        template_filename=args.template_filename,
        output_directory=args.output_directory,
        additional_data_directory=args.additional_data_directory,
        planets=args.planets,
        n_nodes=args.nodes,
        n_cores_per_cpus=args.ncores_per_cpus,
        n_cpus_per_nodes=args.ncpus_per_nodes,
        n_jobs_per_calls=args.njobs_per_calls,
        wavelength_min=args.wavelength_min,
        wavelength_max=args.wavelength_max,
        n_wavelength_bins=args.nwavelength_bins,
        mode=args.mode,
        n_live_points=args.n_live_points,
        co2_mode=args.co2,
        job_basename=args.job_base_name,
        rewrite=args.no_rewrite,
        resume=args.resume,
        mpi_control=args.no_mpi_control
    )
