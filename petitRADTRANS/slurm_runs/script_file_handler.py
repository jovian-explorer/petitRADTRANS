"""Handle slurm sbatch files
"""


def get_line_separator(line):
    """Find the line separator of a line.
    Assume that the line separator is at the end of the line.
    Possible line separators are \r (CR), \n (LF) or \r\n (CRLF).

    Args:
        line: any string ending with a line separator

    Returns:
        linesep: the line separator as a string
    """
    linesep = line.encode('unicode_escape').rsplit(b'\\', 2)[-2:]

    if linesep[0] == b'r':  # CRLF
        linesep = '\r\n'
    elif len(linesep) == 1:
        raise ValueError(f"line {line} has no line separator")
    else:  # CR or LF
        if linesep[1] != b'r' and linesep[1] != b'n':
            raise ValueError(f"'\\{linesep[1].decode('unicode_escape')}' is not a valid line separator "
                             f"(must be '\\r', '\\n' or '\\r\\n', i.e. CR, LF or CRLF)")

        linesep = b'\\' + linesep[1]
        linesep = linesep.decode('unicode_escape')

    return linesep


def make_srun_script_from_template(filename, template_filename, job_name='petitRADTRANS_job', nodes=1, tasks_per_node=1,
                                   cpus_per_task=1,
                                   time='0-00:01:00'):
    """Make a new srun script file from a template file, updating some options.
    The template script has to be a working slurm script. All the #SBATCH option lines must be at the beginning of the
    template script.

    Args:
        filename: name of the new file to generate
        template_filename: any working sbatch script
        job_name: name of the slurm job allocation
        nodes: number of nodes to allocate to the slurm job
        tasks_per_node: number of tasks to be invoked on each node, equivalent to the number of processes for MPI
        cpus_per_task: number of processors per task, relevant when launching application with a required number of CPUs
        time: (days-hours:minutes:seconds) total job run time limit
    """
    # Read template
    with open(template_filename, 'r') as f:
        lines = f.readlines()

    # Initialization
    i_line = 0
    n_lines = len(lines)

    sbatch_options_found = {
        f"#SBATCH --job-name={job_name}\n": False,
        f"#SBATCH --nodes={nodes}\n": False,
        f"#SBATCH --tasks-per-node={tasks_per_node}\n": False,
        f"#SBATCH --cpus-per-task={cpus_per_task}\n": False,
        f"#SBATCH --time={time}\n": False
    }

    file_linesep = get_line_separator(lines[0])

    # Check template first lines
    if lines[0][:11] != '#!/bin/bash':
        lines.insert(0, '#!/bin/bash' + file_linesep)

    if lines[1] != file_linesep:
        lines.insert(1, file_linesep)

    # Find the SBATCH options first line
    for i_line in range(n_lines):
        if lines[i_line][:8] == '#SBATCH ':
            break

    # Change relevant options to new values
    if i_line == n_lines:  # no option found
        print(f"no #SBATCH option found in template file, adding required ones")
        for sbatch_option in list(sbatch_options_found.keys())[::-1]:  # invert to insert in the same order as the dict
            lines.insert(2, sbatch_option)
    else:  # some options found, replace with new values
        # Initialization
        option_found = []
        i_line_option = i_line

        for i in range(len(sbatch_options_found)):
            option_found.append(False)

        for i_line in range(i_line_option, n_lines):
            # Breaking conditions
            if i_line > n_lines:
                break

            if lines[i_line][:8] != '#SBATCH ' and lines[i_line] != file_linesep:
                break

            # Search options in lines
            for sbatch_option, option_found in sbatch_options_found.items():
                sbatch_option_base = sbatch_option.rsplit('=', 1)[0]

                if lines[i_line][:len(sbatch_option_base)] == sbatch_option_base:
                    # Handle duplicates
                    if option_found:
                        print(f"option '{sbatch_option_base}' duplicated in template file, "
                              f"removing duplicate")
                        lines.pop(i_line)
                        n_lines -= 1

                        if i_line + 1 == n_lines:
                            break
                        elif i_line > n_lines:
                            i_line = n_lines
                            break
                        else:
                            continue

                    # Update line
                    lines[i_line] = sbatch_option
                    sbatch_options_found[sbatch_option] = True

                    break

        if i_line == n_lines:
            raise ValueError(f"template file '{template_filename}' must have more than just SBATCH options to run "
                             f"anything!")

        # Add lacking options
        if not all(list(sbatch_options_found.values())):
            print('Adding lacking options')

            for sbatch_option, option_found in sbatch_options_found.items():
                if not option_found:
                    lines.insert(i_line - 1, sbatch_option)

    # Write new file
    with open(filename, 'w') as f:
        f.writelines(lines)
