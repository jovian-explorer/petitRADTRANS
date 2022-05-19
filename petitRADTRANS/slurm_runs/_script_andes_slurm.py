"""Script to generate SLURM run scripts.

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
from pathlib import Path

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
    help='maximum number of nodes to allocate for the run'
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
    default=100,
    help='number of wavelength bins within the wavelength range'
)


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

    shared_array = list(
        array[:array.size - n_leftover_elements].reshape(
            n_entities, elements_per_entities
        )
    )
    leftover_elements = array[array.size - n_leftover_elements:]

    if leftover_elements.size > 0:
        if append_within_existing:
            for i, leftover_element in enumerate(leftover_elements):
                shared_array[np.mod(i, n_entities)] = np.append(shared_array[i], leftover_element)
        else:
            if array.size - n_leftover_elements <= n_entities:
                shared_array[-1] = np.append(shared_array[-1], leftover_elements)
            else:
                shared_array.append(leftover_elements)

    return shared_array


def main(python_script, template_filename, output_directory, additional_data_directory, planets, n_nodes,
         n_cores_per_cpus, n_cpus_per_nodes, wavelength_min, wavelength_max, n_wavelength_bins,
         job_basename='pRT_ANDES'):
    # Generate bins array
    wavelengths_borders = [wavelength_min, wavelength_max]

    wavelength_bins = wavelengths_borders * np.array([1.001, 0.999])
    wavelength_bins = np.linspace(wavelength_bins[0], wavelength_bins[1], int(n_wavelength_bins + 1))

    # Split bins by nodes, will use the same configuration for each planet
    sim_ids = np.linspace(1, n_wavelength_bins, n_wavelength_bins, dtype=int)

    node_sims = fair_share(sim_ids, n_nodes, append_within_existing=True)

    # Split bins by processors within each node, to get the list of jobs per node
    # Communication-wise it is more efficient to keep each run within a unique CPU, at the cost of slower individual run
    node_jobs_sims = []

    for i, node_sim in enumerate(node_sims):
        n_jobs_per_node = np.max((int(np.floor(node_sim.size / n_cpus_per_nodes)), n_cpus_per_nodes))
        print(node_sim.size, node_sim, n_cpus_per_nodes, n_jobs_per_node)
        node_jobs_sims.append(fair_share(node_sim, n_jobs_per_node, append_within_existing=False))

    print(node_jobs_sims)

    for planet in planets:
        for i, node_jobs in enumerate(node_jobs_sims):
            for j, jobs in enumerate(node_jobs):
                tasks_per_node = n_cpus_per_nodes * n_cores_per_cpus  # using one node only
                srun_lines = []

                for small_job in jobs:
                    print(wavelength_bins)
                    srun_lines.append(
                        f"srun mpiexec -n {int(tasks_per_node / jobs.size)} python3 {python_script} "
                        f"--planet '{planet}' "
                        f"--output-directory '{output_directory}' "
                        f"--additional-data-directory '{additional_data_directory}' "
                        f"--wavelength-min {wavelength_bins[small_job - 1]} "
                        f"--wavelength-max {wavelength_bins[small_job]}\n"
                    )

                print(planet,i,j, jobs, srun_lines)

                make_srun_script_from_template(
                    filename=f"{job_basename}_{planet.lower().replace(' ', '_')}_job{(i + 1) * j}.batch",
                    template_filename=template_filename,
                    job_name=f"{job_basename}_{planet.lower().replace(' ', '_')}_job{(i + 1) * j}",
                    nodes=1,  # nodes per job
                    tasks_per_node=tasks_per_node,
                    cpus_per_task=1,
                    time='0-12:00:00',
                    srun_lines=srun_lines
                )


if __name__ == '__main__':
    args = parser.parse_args()
    print(args.__dict__)

    main(
        python_script=args.python_script,
        template_filename=args.template_filename,
        output_directory=args.output_directory,
        additional_data_directory=args.additional_data_directory,
        planets=args.planets,
        n_nodes=args.nodes,
        n_cores_per_cpus=args.ncores_per_cpus,
        n_cpus_per_nodes=args.ncpus_per_nodes,
        wavelength_min=args.wavelength_min,
        wavelength_max=args.wavelength_max,
        n_wavelength_bins=args.nwavelength_bins,
        job_basename='pRT_ANDES'
    )
