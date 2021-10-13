"""
Manages the configuration files.
"""


def _get_petitradtrans_config_directory():
    """
    Get the petitRADTRANS configuration directory, where the configuration file is stored.

    Returns:
        The configuration directory.
    """
    import os
    from pathlib import Path

    config_directory = os.path.join(str(Path.home()), '.petitradtrans')

    if not os.path.isdir(config_directory):
        os.mkdir(config_directory)

    return config_directory


def _get_petitradtrans_config_file():
    """
    Get the full path to the petitRADTRANS configuration configuration file.

    Returns:
        The configuration filename.
    """
    import os

    config_directory = _get_petitradtrans_config_directory()

    return config_directory + os.path.sep + 'petitradtrans_config_file.ini'


def _make_petitradtrans_config_file():
    """
    Make the petitRADTRANS configuration file.
    """
    import os
    import configparser
    from pathlib import Path

    print('Generating configuration file...')

    config = configparser.ConfigParser()

    # Default path to the input data and to the outputs
    config['Paths'] = {
        'pRT_input_data_path': os.path.join(str(Path.home()), 'petitRADTRANS', 'input_data'),
        'pRT_outputs_path': os.path.join(str(Path.home()), 'petitRADTRANS', 'outputs')
    }

    with open(petitradtrans_config_file, 'w') as configfile:
        config.write(configfile)


def load_petitradtrans_config_file():
    """
    Load the petitRADTRANS configuration file. Generate it if necessary.

    Returns:
        A dictionary containing the petitRADTRANS configuration.
    """
    import configparser
    import os

    if not os.path.isfile(petitradtrans_config_file):
        _make_petitradtrans_config_file()  # TODO find a better, safer way to do generate the configuration file?

    config = configparser.ConfigParser()
    config.read(petitradtrans_config_file)

    return config


def update_petitradtrans_config_file(configuration_dict: dict):
    """
    Update the configuration file of petitRADTRANS.

    Args:
        configuration_dict: dictionary containing the updated configuration of petitRADTRANS
    """
    import configparser

    config = configparser.ConfigParser()
    config.update(configuration_dict)


petitradtrans_config_file = _get_petitradtrans_config_file()
petitradtrans_config = load_petitradtrans_config_file()
