import os
import pickle

import h5py
import numpy as np
import pyvo
from astropy.table.table import Table
from petitRADTRANS.fort_rebin import fort_rebin as fr

from petitRADTRANS import nat_cst as nc
from petitRADTRANS.ccf.utils import calculate_uncertainty, module_dir
from petitRADTRANS.phoenix import get_PHOENIX_spec
from petitRADTRANS.physics import guillot_global
from petitRADTRANS.radtrans import Radtrans

# from petitRADTRANS.config import petitradtrans_config

# planet_models_directory = os.path_input_data.abspath(Path.home()) + os.path_input_data.sep + 'Downloads' + os.path_input_data.sep + 'tmp' #os.path_input_data.abspath(os.path_input_data.dirname(__file__) + os.path_input_data.sep + 'planet_models')
planet_models_directory = os.path.abspath(os.path.dirname(__file__) + os.path.sep + 'planet_models')  # TODO change that


# planet_models_directory = petitradtrans_config['Paths']['pRT_outputs_path']


class Planet:
    def __init__(
            self,
            name,
            mass=0.,
            mass_error_upper=0.,
            mass_error_lower=0.,
            radius=0.,
            radius_error_upper=0.,
            radius_error_lower=0.,
            orbit_semi_major_axis=0.,
            orbit_semi_major_axis_error_upper=0.,
            orbit_semi_major_axis_error_lower=0.,
            orbital_eccentricity=0.,
            orbital_eccentricity_error_upper=0.,
            orbital_eccentricity_error_lower=0.,
            orbital_inclination=0.,
            orbital_inclination_error_upper=0.,
            orbital_inclination_error_lower=0.,
            orbital_period=0.,
            orbital_period_error_upper=0.,
            orbital_period_error_lower=0.,
            argument_of_periastron=0.,
            argument_of_periastron_error_upper=0.,
            argument_of_periastron_error_lower=0.,
            epoch_of_periastron=0.,
            epoch_of_periastron_error_upper=0.,
            epoch_of_periastron_error_lower=0.,
            ra=0.,
            dec=0.,
            x=0.,
            y=0.,
            z=0.,
            reference_pressure=0.01,
            density=0.,
            density_error_upper=0.,
            density_error_lower=0.,
            surface_gravity=0.,
            surface_gravity_error_upper=0.,
            surface_gravity_error_lower=0.,
            equilibrium_temperature=0.,
            equilibrium_temperature_error_upper=0.,
            equilibrium_temperature_error_lower=0.,
            insolation_flux=0.,
            insolation_flux_error_upper=0.,
            insolation_flux_error_lower=0.,
            bond_albedo=0.,
            bond_albedo_error_upper=0.,
            bond_albedo_error_lower=0.,
            transit_depth=0.,
            transit_depth_error_upper=0.,
            transit_depth_error_lower=0.,
            transit_midpoint_time=0.,
            transit_midpoint_time_error_upper=0.,
            transit_midpoint_time_error_lower=0.,
            transit_duration=0.,
            transit_duration_error_upper=0.,
            transit_duration_error_lower=0.,
            projected_obliquity=0.,
            projected_obliquity_error_upper=0.,
            projected_obliquity_error_lower=0.,
            true_obliquity=0.,
            true_obliquity_error_upper=0.,
            true_obliquity_error_lower=0.,
            radial_velocity_amplitude=0.,
            radial_velocity_amplitude_error_upper=0.,
            radial_velocity_amplitude_error_lower=0.,
            planet_stellar_radius_ratio=0.,
            planet_stellar_radius_ratio_error_upper=0.,
            planet_stellar_radius_ratio_error_lower=0.,
            semi_major_axis_stellar_radius_ratio=0.,
            semi_major_axis_stellar_radius_ratio_error_upper=0.,
            semi_major_axis_stellar_radius_ratio_error_lower=0.,
            reference='',
            discovery_year=0,
            discovery_method='',
            discovery_reference='',
            confirmation_status='',
            host_name='',
            star_spectral_type='',
            star_mass=0.,
            star_mass_error_upper=0.,
            star_mass_error_lower=0.,
            star_radius=0.,
            star_radius_error_upper=0.,
            star_radius_error_lower=0.,
            star_age=0.,
            star_age_error_upper=0.,
            star_age_error_lower=0.,
            star_metallicity=0.,
            star_metallicity_error_upper=0.,
            star_metallicity_error_lower=0.,
            star_effective_temperature=0.,
            star_effective_temperature_error_upper=0.,
            star_effective_temperature_error_lower=0.,
            star_luminosity=0.,
            star_luminosity_error_upper=0.,
            star_luminosity_error_lower=0.,
            star_rotational_period=0.,
            star_rotational_period_error_upper=0.,
            star_rotational_period_error_lower=0.,
            star_radial_velocity=0.,
            star_radial_velocity_error_upper=0.,
            star_radial_velocity_error_lower=0.,
            star_rotational_velocity=0.,
            star_rotational_velocity_error_upper=0.,
            star_rotational_velocity_error_lower=0.,
            star_density=0.,
            star_density_error_upper=0.,
            star_density_error_lower=0.,
            star_surface_gravity=0.,
            star_surface_gravity_error_upper=0.,
            star_surface_gravity_error_lower=0.,
            star_reference='',
            system_star_number=0,
            system_planet_number=0,
            system_moon_number=0,
            system_distance=0.,
            system_distance_error_upper=0.,
            system_distance_error_lower=0.,
            system_apparent_magnitude_v=0.,
            system_apparent_magnitude_v_error_upper=0.,
            system_apparent_magnitude_v_error_lower=0.,
            system_apparent_magnitude_j=0.,
            system_apparent_magnitude_j_error_upper=0.,
            system_apparent_magnitude_j_error_lower=0.,
            system_proper_motion=0.,
            system_proper_motion_error_upper=0.,
            system_proper_motion_error_lower=0.,
            system_proper_motion_ra=0.,
            system_proper_motion_ra_error_upper=0.,
            system_proper_motion_ra_error_lower=0.,
            system_proper_motion_dec=0.,
            system_proper_motion_dec_error_upper=0.,
            system_proper_motion_dec_error_lower=0.,
            units=None
    ):
        self.name = name
        self.mass = mass
        self.mass_error_upper = mass_error_upper
        self.mass_error_lower = mass_error_lower
        self.radius = radius
        self.radius_error_upper = radius_error_upper
        self.radius_error_lower = radius_error_lower
        self.orbit_semi_major_axis = orbit_semi_major_axis
        self.orbit_semi_major_axis_error_upper = orbit_semi_major_axis_error_upper
        self.orbit_semi_major_axis_error_lower = orbit_semi_major_axis_error_lower
        self.orbital_eccentricity = orbital_eccentricity
        self.orbital_eccentricity_error_upper = orbital_eccentricity_error_upper
        self.orbital_eccentricity_error_lower = orbital_eccentricity_error_lower
        self.orbital_inclination = orbital_inclination
        self.orbital_inclination_error_upper = orbital_inclination_error_upper
        self.orbital_inclination_error_lower = orbital_inclination_error_lower
        self.orbital_period = orbital_period
        self.orbital_period_error_upper = orbital_period_error_upper
        self.orbital_period_error_lower = orbital_period_error_lower
        self.argument_of_periastron = argument_of_periastron
        self.argument_of_periastron_error_upper = argument_of_periastron_error_upper
        self.argument_of_periastron_error_lower = argument_of_periastron_error_lower
        self.epoch_of_periastron = epoch_of_periastron
        self.epoch_of_periastron_error_upper = epoch_of_periastron_error_upper
        self.epoch_of_periastron_error_lower = epoch_of_periastron_error_lower
        self.ra = ra
        self.dec = dec
        self.x = x
        self.y = y
        self.z = z
        self.reference_pressure = reference_pressure
        self.density = density
        self.density_error_upper = density_error_upper
        self.density_error_lower = density_error_lower
        self.surface_gravity = surface_gravity
        self.surface_gravity_error_upper = surface_gravity_error_upper
        self.surface_gravity_error_lower = surface_gravity_error_lower
        self.equilibrium_temperature = equilibrium_temperature
        self.equilibrium_temperature_error_upper = equilibrium_temperature_error_upper
        self.equilibrium_temperature_error_lower = equilibrium_temperature_error_lower
        self.insolation_flux = insolation_flux
        self.insolation_flux_error_upper = insolation_flux_error_upper
        self.insolation_flux_error_lower = insolation_flux_error_lower
        self.bond_albedo = bond_albedo
        self.bond_albedo_error_upper = bond_albedo_error_upper
        self.bond_albedo_error_lower = bond_albedo_error_lower
        self.transit_depth = transit_depth
        self.transit_depth_error_upper = transit_depth_error_upper
        self.transit_depth_error_lower = transit_depth_error_lower
        self.transit_midpoint_time = transit_midpoint_time
        self.transit_midpoint_time_error_upper = transit_midpoint_time_error_upper
        self.transit_midpoint_time_error_lower = transit_midpoint_time_error_lower
        self.transit_duration = transit_duration
        self.transit_duration_error_upper = transit_duration_error_upper
        self.transit_duration_error_lower = transit_duration_error_lower
        self.projected_obliquity = projected_obliquity
        self.projected_obliquity_error_upper = projected_obliquity_error_upper
        self.projected_obliquity_error_lower = projected_obliquity_error_lower
        self.true_obliquity = true_obliquity
        self.true_obliquity_error_upper = true_obliquity_error_upper
        self.true_obliquity_error_lower = true_obliquity_error_lower
        self.radial_velocity_amplitude = radial_velocity_amplitude
        self.radial_velocity_amplitude_error_upper = radial_velocity_amplitude_error_upper
        self.radial_velocity_amplitude_error_lower = radial_velocity_amplitude_error_lower
        self.planet_stellar_radius_ratio = planet_stellar_radius_ratio
        self.planet_stellar_radius_ratio_error_upper = planet_stellar_radius_ratio_error_upper
        self.planet_stellar_radius_ratio_error_lower = planet_stellar_radius_ratio_error_lower
        self.semi_major_axis_stellar_radius_ratio = semi_major_axis_stellar_radius_ratio
        self.semi_major_axis_stellar_radius_ratio_error_upper = semi_major_axis_stellar_radius_ratio_error_upper
        self.semi_major_axis_stellar_radius_ratio_error_lower = semi_major_axis_stellar_radius_ratio_error_lower
        self.reference = reference
        self.discovery_year = discovery_year
        self.discovery_method = discovery_method
        self.discovery_reference = discovery_reference
        self.confirmation_status = confirmation_status
        self.host_name = host_name
        self.star_spectral_type = star_spectral_type
        self.star_mass = star_mass
        self.star_mass_error_upper = star_mass_error_upper
        self.star_mass_error_lower = star_mass_error_lower
        self.star_radius = star_radius
        self.star_radius_error_upper = star_radius_error_upper
        self.star_radius_error_lower = star_radius_error_lower
        self.star_age = star_age
        self.star_age_error_upper = star_age_error_upper
        self.star_age_error_lower = star_age_error_lower
        self.star_metallicity = star_metallicity
        self.star_metallicity_error_upper = star_metallicity_error_upper
        self.star_metallicity_error_lower = star_metallicity_error_lower
        self.star_effective_temperature = star_effective_temperature
        self.star_effective_temperature_error_upper = star_effective_temperature_error_upper
        self.star_effective_temperature_error_lower = star_effective_temperature_error_lower
        self.star_luminosity = star_luminosity
        self.star_luminosity_error_upper = star_luminosity_error_upper
        self.star_luminosity_error_lower = star_luminosity_error_lower
        self.star_rotational_period = star_rotational_period
        self.star_rotational_period_error_upper = star_rotational_period_error_upper
        self.star_rotational_period_error_lower = star_rotational_period_error_lower
        self.star_radial_velocity = star_radial_velocity
        self.star_radial_velocity_error_upper = star_radial_velocity_error_upper
        self.star_radial_velocity_error_lower = star_radial_velocity_error_lower
        self.star_rotational_velocity = star_rotational_velocity
        self.star_rotational_velocity_error_upper = star_rotational_velocity_error_upper
        self.star_rotational_velocity_error_lower = star_rotational_velocity_error_lower
        self.star_density = star_density
        self.star_density_error_upper = star_density_error_upper
        self.star_density_error_lower = star_density_error_lower
        self.star_surface_gravity = star_surface_gravity
        self.star_surface_gravity_error_upper = star_surface_gravity_error_upper
        self.star_surface_gravity_error_lower = star_surface_gravity_error_lower
        self.star_reference = star_reference
        self.system_star_number = system_star_number
        self.system_planet_number = system_planet_number
        self.system_moon_number = system_moon_number
        self.system_distance = system_distance
        self.system_distance_error_upper = system_distance_error_upper
        self.system_distance_error_lower = system_distance_error_lower
        self.system_apparent_magnitude_v = system_apparent_magnitude_v
        self.system_apparent_magnitude_v_error_upper = system_apparent_magnitude_v_error_upper
        self.system_apparent_magnitude_v_error_lower = system_apparent_magnitude_v_error_lower
        self.system_apparent_magnitude_j = system_apparent_magnitude_j
        self.system_apparent_magnitude_j_error_upper = system_apparent_magnitude_j_error_upper
        self.system_apparent_magnitude_j_error_lower = system_apparent_magnitude_j_error_lower
        self.system_proper_motion = system_proper_motion
        self.system_proper_motion_error_upper = system_proper_motion_error_upper
        self.system_proper_motion_error_lower = system_proper_motion_error_lower
        self.system_proper_motion_ra = system_proper_motion_ra
        self.system_proper_motion_ra_error_upper = system_proper_motion_ra_error_upper
        self.system_proper_motion_ra_error_lower = system_proper_motion_ra_error_lower
        self.system_proper_motion_dec = system_proper_motion_dec
        self.system_proper_motion_dec_error_upper = system_proper_motion_dec_error_upper
        self.system_proper_motion_dec_error_lower = system_proper_motion_dec_error_lower

        if units is None:
            self.units = {
                'name': 'N/A',
                'mass': 'g',
                'mass_error_upper': 'g',
                'mass_error_lower': 'g',
                'radius': 'cm',
                'radius_error_upper': 'cm',
                'radius_error_lower': 'cm',
                'orbit_semi_major_axis': 'cm',
                'orbit_semi_major_axis_error_upper': 'cm',
                'orbit_semi_major_axis_error_lower': 'cm',
                'orbital_eccentricity': 'None',
                'orbital_eccentricity_error_upper': 'None',
                'orbital_eccentricity_error_lower': 'None',
                'orbital_inclination': 'deg',
                'orbital_inclination_error_upper': 'deg',
                'orbital_inclination_error_lower': 'deg',
                'orbital_period': 's',
                'orbital_period_error_upper': 's',
                'orbital_period_error_lower': 's',
                'argument_of_periastron': 'deg',
                'argument_of_periastron_error_upper': 'deg',
                'argument_of_periastron_error_lower': 'deg',
                'epoch_of_periastron': 's',
                'epoch_of_periastron_error_upper': 's',
                'epoch_of_periastron_error_lower': 's',
                'ra': 'deg',
                'dec': 'deg',
                'x': 'cm',
                'y': 'cm',
                'z': 'cm',
                'reference_pressure': 'bar',
                'density': 'g/cm^3',
                'density_error_upper': 'g/cm^3',
                'density_error_lower': 'g/cm^3',
                'surface_gravity': 'cm/s^2',
                'surface_gravity_error_upper': 'cm/s^2',
                'surface_gravity_error_lower': 'cm/s^2',
                'equilibrium_temperature': 'K',
                'equilibrium_temperature_error_upper': 'K',
                'equilibrium_temperature_error_lower': 'K',
                'insolation_flux': 'erg/s/cm^2',
                'insolation_flux_error_upper': 'erg/s/cm^2',
                'insolation_flux_error_lower': 'erg/s/cm^2',
                'bond_albedo': 'None',
                'bond_albedo_error_upper': 'None',
                'bond_albedo_error_lower': 'None',
                'transit_depth': 'None',
                'transit_depth_error_upper': 'None',
                'transit_depth_error_lower': 'None',
                'transit_midpoint_time': 's',
                'transit_midpoint_time_error_upper': 's',
                'transit_midpoint_time_error_lower': 's',
                'transit_duration': 's',
                'transit_duration_error_upper': 's',
                'transit_duration_error_lower': 's',
                'projected_obliquity': 'deg',
                'projected_obliquity_error_upper': 'deg',
                'projected_obliquity_error_lower': 'deg',
                'true_obliquity': 'deg',
                'true_obliquity_error_upper': 'deg',
                'true_obliquity_error_lower': 'deg',
                'radial_velocity_amplitude': 'cm/s',
                'radial_velocity_amplitude_error_upper': 'cm/s',
                'radial_velocity_amplitude_error_lower': 'cm/s',
                'planet_stellar_radius_ratio': 'None',
                'planet_stellar_radius_ratio_error_upper': 'None',
                'planet_stellar_radius_ratio_error_lower': 'None',
                'semi_major_axis_stellar_radius_ratio': 'None',
                'semi_major_axis_stellar_radius_ratio_error_upper': 'None',
                'semi_major_axis_stellar_radius_ratio_error_lower': 'None',
                'reference': 'N/A',
                'discovery_year': 'year',
                'discovery_method': 'N/A',
                'discovery_reference': 'N/A',
                'confirmation_status': 'N/A',
                'host_name': 'N/A',
                'star_spectral_type': 'N/A',
                'star_mass': 'g',
                'star_mass_error_upper': 'g',
                'star_mass_error_lower': 'g',
                'star_radius': 'cm',
                'star_radius_error_upper': 'cm',
                'star_radius_error_lower': 'cm',
                'star_age': 's',
                'star_age_error_upper': 's',
                'star_age_error_lower': 's',
                'star_metallicity': 'dex',
                'star_metallicity_error_upper': 'dex',
                'star_metallicity_error_lower': 'dex',
                'star_effective_temperature': 'K',
                'star_effective_temperature_error_upper': 'K',
                'star_effective_temperature_error_lower': 'K',
                'star_luminosity': 'erg/s',
                'star_luminosity_error_upper': 'erg/s',
                'star_luminosity_error_lower': 'erg/s',
                'star_rotational_period': 's',
                'star_rotational_period_error_upper': 's',
                'star_rotational_period_error_lower': 's',
                'star_radial_velocity': 'cm/s',
                'star_radial_velocity_error_upper': 'cm/s',
                'star_radial_velocity_error_lower': 'cm/s',
                'star_rotational_velocity': 'cm/s',
                'star_rotational_velocity_error_upper': 'cm/s',
                'star_rotational_velocity_error_lower': 'cm/s',
                'star_density': 'g/cm^3',
                'star_density_error_upper': 'g/cm^3',
                'star_density_error_lower': 'g/cm^3',
                'star_surface_gravity': 'cm/s^2',
                'star_surface_gravity_error_upper': 'cm/s^2',
                'star_surface_gravity_error_lower': 'cm/s^2',
                'star_reference': 'N/A',
                'system_star_number': 'None',
                'system_planet_number': 'None',
                'system_moon_number': 'None',
                'system_distance': 'cm',
                'system_distance_error_upper': 'cm',
                'system_distance_error_lower': 'cm',
                'system_apparent_magnitude_v': 'None',
                'system_apparent_magnitude_v_error_upper': 'None',
                'system_apparent_magnitude_v_error_lower': 'None',
                'system_apparent_magnitude_j': 'None',
                'system_apparent_magnitude_j_error_upper': 'None',
                'system_apparent_magnitude_j_error_lower': 'None',
                'system_proper_motion': 'deg/s',
                'system_proper_motion_error_upper': 'deg/s',
                'system_proper_motion_error_lower': 'deg/s',
                'system_proper_motion_ra': 'deg/s',
                'system_proper_motion_ra_error_upper': 'deg/s',
                'system_proper_motion_ra_error_lower': 'deg/s',
                'system_proper_motion_dec': 'deg/s',
                'system_proper_motion_dec_error_upper': 'deg/s',
                'system_proper_motion_dec_error_lower': 'deg/s',
                'units': 'N/A'
            }
        else:
            self.units = units

    def calculate_planetary_equilibrium_temperature(self):
        """
        Calculate the equilibrium temperature of a planet.
        """
        equilibrium_temperature = \
            self.star_effective_temperature * np.sqrt(self.star_radius / (2 * self.orbit_semi_major_axis)) \
            * (1 - self.bond_albedo) ** 0.25

        partial_derivatives = np.array([
            equilibrium_temperature / self.star_effective_temperature,  # dt_eq/dt_eff
            0.5 * equilibrium_temperature / self.star_radius,  # dt_eq/dr*
            - 0.5 * equilibrium_temperature / self.orbit_semi_major_axis  # dt_eq/dd
        ])
        uncertainties = np.abs(np.array([
            [self.star_effective_temperature_error_lower, self.star_effective_temperature_error_upper],
            [self.star_radius_error_lower, self.star_radius_error_upper],
            [self.orbit_semi_major_axis_error_lower, self.orbit_semi_major_axis_error_upper]
        ]))

        errors = calculate_uncertainty(partial_derivatives, uncertainties)  # lower and upper errors

        return equilibrium_temperature, errors[1], -errors[0]

    def get_filename(self):
        return self.generate_filename(self.name)

    def save(self, filename=None):
        if filename is None:
            filename = self.get_filename()

        with h5py.File(filename, 'w') as f:
            for key in self.__dict__:
                if key == 'units':
                    continue

                data_set = f.create_dataset(
                    name=key,
                    data=self.__dict__[key]
                )

                if self.units[key] != 'N/A':
                    data_set.attrs['units'] = self.units[key]

    @classmethod
    def from_votable(cls, votable):
        new_planet = cls('new_planet')
        parameter_dict = {}

        for key in votable.keys():
            # Clearer keynames
            value, key = Planet.__convert_nasa_exoplanet_archive(votable[key], key)
            parameter_dict[key] = value

        parameter_dict = new_planet.select_best_in_column(parameter_dict)

        for key in parameter_dict:
            if key in new_planet.__dict__:
                new_planet.__dict__[key] = parameter_dict[key]

        if 'surface_gravity' not in parameter_dict:
            new_planet.surface_gravity, \
                new_planet.surface_gravity_error_upper, new_planet.surface_gravity_error_lower = \
                new_planet.mass2surface_gravity(
                    new_planet.mass,
                    new_planet.radius,
                    mass_error_upper=new_planet.mass_error_upper,
                    mass_error_lower=new_planet.mass_error_lower,
                    radius_error_upper=new_planet.radius_error_upper,
                    radius_error_lower=new_planet.radius_error_lower
                )

        if 'equilibrium_temperature' not in parameter_dict:
            new_planet.equilibrium_temperature, \
                new_planet.equilibrium_temperature_error_upper, new_planet.equilibrium_temperature_error_lower = \
                new_planet.calculate_planetary_equilibrium_temperature()

        return new_planet

    @classmethod
    def from_votable_file(cls, filename):
        astro_table = Table.read(filename)

        return cls.from_votable(astro_table)

    @classmethod
    def get(cls, name):
        filename = cls.generate_filename(name)

        if not os.path.exists(filename):
            filename_vot = filename.rsplit('.', 1)[0] + '.vot'  # search for votable

            if not os.path.exists(filename_vot):
                print(f"file '{filename_vot}' not found, downloading...")
                cls.download_from_nasa_exoplanet_archive(name)

            # Save into HDF5 and remove the VO table
            new_planet = cls.from_votable_file(filename_vot)
            new_planet.save()
            os.remove(filename_vot)

            return new_planet
        else:
            return cls.load(name, filename)

    @classmethod
    def load(cls, name, filename=None):
        new_planet = cls(name)

        if filename is None:
            filename = new_planet.get_filename()

        with h5py.File(filename, 'r') as f:
            for key in f:
                if isinstance(f[key][()], bytes):
                    value = str(f[key][()], 'utf-8')
                else:
                    value = f[key][()]

                new_planet.__dict__[key] = value

                if 'units' in f[key].attrs:
                    if key in new_planet.units:
                        if f[key].attrs['units'] != new_planet.units[key]:
                            raise ValueError(f"units of key '{key}' must be '{new_planet.units[key]}', "
                                             f"but is '{f[key].attrs['units']}'")
                    else:
                        new_planet.units[key] = f[key].attrs['units'][()]
                else:
                    new_planet.units[key] = 'N/A'

        return new_planet

    @staticmethod
    def __convert_nasa_exoplanet_archive(value, key, verbose=False):
        skip_unit_conversion = False

        # Heads
        if key[:3] == 'sy_':
            key = 'system_' + key[3:]
        elif key[:3] == 'st_':
            key = 'star_' + key[3:]
        elif key[:5] == 'disc_':
            key = 'discovery_' + key[5:]

        # Tails
        if key[-4:] == 'err1':
            key = key[:-4] + '_error_upper'
        elif key[-4:] == 'err2':
            key = key[:-4] + '_error_lower'
        elif key[-3:] == 'lim':
            key = key[:-3] + '_limit_flag'
            skip_unit_conversion = True
        elif key[-3:] == 'str':
            key = key[:-3] + '_str'
            skip_unit_conversion = True

        # Parameters of interest
        if '_orbper' in key:
            key = key.replace('_orbper', '_orbital_period')

            if not skip_unit_conversion:
                value *= nc.snc.day
        elif '_orblper' in key:
            key = key.replace('_orblper', '_argument_of_periastron')
        elif '_orbsmax' in key:
            key = key.replace('_orbsmax', '_orbit_semi_major_axis')

            if not skip_unit_conversion:
                value *= nc.AU
        elif '_orbincl' in key:
            key = key.replace('_orbincl', '_orbital_inclination')
        elif '_orbtper' in key:
            key = key.replace('_orbtper', '_epoch_of_periastron')

            if not skip_unit_conversion:
                value *= nc.snc.day
        elif '_orbeccen' in key:
            key = key.replace('_orbeccen', '_orbital_eccentricity')
        elif '_eqt' in key:
            key = key.replace('_eqt', '_equilibrium_temperature')
        elif '_occdep' in key:
            key = key.replace('_occdep', '_occultation_depth')
        elif '_insol' in key:
            key = key.replace('_insol', '_insolation_flux')

            if not skip_unit_conversion:
                value *= nc.s_earth
        elif '_dens' in key:
            key = key.replace('_dens', '_density')
        elif '_trandep' in key:
            key = key.replace('_trandep', '_transit_depth')

            if not skip_unit_conversion:
                value *= 1e2
        elif '_tranmid' in key:
            key = key.replace('_tranmid', '_transit_midpoint_time')

            if not skip_unit_conversion:
                value *= nc.snc.hour
        elif '_trandur' in key:
            key = key.replace('_trandur', '_transit_duration')

            if not skip_unit_conversion:
                value *= nc.snc.hour
        elif '_spectype' in key:
            key = key.replace('_spectype', '_spectral_type')
        elif '_rotp' in key:
            key = key.replace('_rotp', '_rotational_period')

            if not skip_unit_conversion:
                value *= nc.snc.day
        elif '_projobliq' in key:
            key = key.replace('_projobliq', '_projected_obliquity')
        elif '_rvamp' in key:
            key = key.replace('_rvamp', '_radial_velocity_amplitude')

            if not skip_unit_conversion:
                value *= 1e2
        elif '_radj' in key:
            key = key.replace('_radj', '_radius')

            if not skip_unit_conversion:
                value *= nc.r_jup
        elif '_ratror' in key:
            key = key.replace('_ratror', '_planet_stellar_radius_ratio')
        elif '_trueobliq' in key:
            key = key.replace('_trueobliq', '_true_obliquity')
        elif '_ratdor' in key:
            key = key.replace('_ratdor', '_semi_major_axis_stellar_radius_ratio')
        elif '_imppar' in key:
            key = key.replace('_imppar', '_impact_parameter')
        elif '_msinij' in key:
            key = key.replace('_msinij', '_mass_sini')

            if not skip_unit_conversion:
                value *= nc.m_jup
        elif '_massj' in key:
            key = key.replace('_massj', '_mass')

            if not skip_unit_conversion:
                value *= nc.m_jup
        elif '_teff' in key:
            key = key.replace('_teff', '_effective_temperature')
        elif '_met' in key:
            key = key.replace('_met', '_metallicity')
        elif '_radv' in key:
            key = key.replace('_radv', '_radial_velocity')

            if not skip_unit_conversion:
                value *= 1e5
        elif '_vsin' in key:
            key = key.replace('_vsin', '_rotational_velocity')

            if not skip_unit_conversion:
                value *= 1e5
        elif '_lum' in key:
            key = key.replace('_lum', '_luminosity')

            if not skip_unit_conversion:
                value = 10 ** value * nc.l_sun
        elif '_logg' in key:
            key = key.replace('_logg', '_surface_gravity')

            if not skip_unit_conversion:
                value = 10 ** value
        elif '_age' in key:
            if not skip_unit_conversion:
                value *= 1e9 * nc.snc.year
        elif 'star_mass' in key:
            if not skip_unit_conversion:
                value *= nc.m_sun
        elif 'star_rad' in key:
            key = key.replace('star_rad', 'star_radius')

            if not skip_unit_conversion:
                value *= nc.r_sun
        elif '_dist' in key:
            key = key.replace('_dist', '_distance')

            if not skip_unit_conversion:
                value *= nc.pc
        elif '_plx' in key:
            key = key.replace('_plx', '_parallax')

            if not skip_unit_conversion:
                value *= 3.6e-6
        elif '_pm' in key:
            if key[-3:] == '_pm':
                key = key.replace('_pm', '_proper_motion')
            else:
                i = key.find('_pm')
                key = key[:i] + '_proper_motion_' + key[i + len('_pm'):]

            if not skip_unit_conversion:
                value *= 3.6e-6 / nc.snc.year

        elif key == 'hostname':
            key = 'host_name'
        elif key == 'discoverymethod':
            key = 'discovery_method'
        elif key == 'discovery_refname':
            key = 'discovery_reference'
        elif 'controv_flag' in key:
            key = 'controversy_flag'
        elif key == 'star_refname':
            key = 'star_reference'
        elif key == 'soltype':
            key = 'confirmation_status'
        elif key == 'system_snum':
            key = 'system_star_number'
        elif key == 'system_pnum':
            key = 'system_planet_number'
        elif key == 'system_mnum':
            key = 'system_moon_number'
        elif 'mag' in key:
            i = key.find('mag')

            if i + len('mag') == len(key):
                tail = ''
            else:
                tail = key[i + 3:]

                if tail[0] != '_':  # should not be necessary
                    tail = '_' + tail

            # Move magnitude band to the end
            if key[i - 2] == '_':  # one-character band
                letter = key[i - 1]
                key = key[:i - 1] + 'apparent_magnitude_' + letter + tail
            elif key[i - 3] == '_':  # two-characters band
                letters = key[i - 2:i]
                key = key[:i - 2] + 'apparent_magnitude_' + letters + tail
            elif 'kepmag' in key:
                key = key[:i - 3] + 'apparent_magnitude_' + 'kepler' + tail
            elif 'gaiamag' in key:
                key = key[:i - 4] + 'apparent_magnitude_' + 'gaia' + tail
            else:
                raise ValueError(f"unidentified apparent magnitude key '{key}'")
        elif verbose:
            print(f"unchanged key '{key}' with value {value}")

        if key[:3] == 'pl_':
            key = key[3:]

        return value, key

    @staticmethod
    def generate_filename(name):
        return f"{planet_models_directory}{os.path.sep}planet_{name.replace(' ', '_')}.h5"

    @staticmethod
    def download_from_nasa_exoplanet_archive(name):
        service = pyvo.dal.TAPService("https://exoplanetarchive.ipac.caltech.edu/TAP")
        result_set = service.search(f"select * from ps where pl_name = '{name}'")

        astro_table = result_set.to_table()
        filename = Planet.generate_filename(name).rsplit('.', 1)[0] + '.vot'

        astro_table.write(filename, format='votable')

        return astro_table

    @staticmethod
    def select_best_in_column(dictionary):
        parameter_dict = {}
        tails = ['_error_upper', '_error_lower', '_limit_flag', '_str']

        for key in dictionary.keys():
            if tails[0] in key or tails[1] in key or tails[2] in key or tails[3] in key:
                continue  # skip every tailed parameters
            elif dictionary[key].dtype == object or not (key + tails[0] in dictionary and key + tails[1] in dictionary):
                # if object or no error tailed parameters, get the first value that is not masked
                parameter_dict[key] = dictionary[key][0]

                for value in dictionary[key][1:]:
                    if not hasattr(value, 'mask'):
                        parameter_dict[key] = value

                        break
            else:
                value_error_upper = dictionary[key + tails[0]]
                value_error_lower = dictionary[key + tails[1]]
                error_interval = np.abs(value_error_upper) + np.abs(value_error_lower)

                wh = np.where(error_interval == np.min(error_interval))[0]

                parameter_dict[key] = dictionary[key][wh][0]

                for tail in tails:
                    if key + tail in dictionary:
                        parameter_dict[key + tail] = dictionary[key + tail][wh][0]

        return parameter_dict

    @staticmethod
    def mass2surface_gravity(mass, radius,
                             mass_error_upper=0., mass_error_lower=0., radius_error_upper=0., radius_error_lower=0.):
        """
        Convert the mass of a planet to its surface gravity.
        Args:
            mass: (g) mass of the planet
            radius: (cm) radius of the planet
            mass_error_upper: (g) upper error on the planet mass
            mass_error_lower: (g) lower error on the planet mass
            radius_error_upper: (cm) upper error on the planet radius
            radius_error_lower: (cm) lower error on the planet radius

        Returns:
            (cm.s-2) the surface gravity of the planet, and its upper and lower error
        """
        surface_gravity = nc.G * mass / radius ** 2

        partial_derivatives = np.array([
            surface_gravity / mass,  # dg/dm
            - 2 * surface_gravity / radius  # dg/dr
        ])
        uncertainties = np.abs(np.array([
            [mass_error_lower, mass_error_upper],
            [radius_error_lower, radius_error_upper]
        ]))

        errors = calculate_uncertainty(partial_derivatives, uncertainties)  # lower and upper errors

        return surface_gravity, errors[1], -errors[0]


class SimplePlanet(Planet):
    def __init__(self, name, radius, surface_gravity, star_effective_temperature, star_radius, orbit_semi_major_axis,
                 reference_pressure=0.01, bond_albedo=0, equilibrium_temperature=None, mass=None):
        """

        Args:
            name: name of the planet
            radius: (cm) radius of the planet
            surface_gravity: (cm.s-2) gravity of the planet
            star_effective_temperature: (K) surface effective temperature of the star
            star_radius: (cm) mean radius of the star
            orbit_semi_major_axis: (cm) distance between the planet and the star
            reference_pressure: (bar) reference pressure for the radius and the gravity of the planet
            bond_albedo: bond albedo of the planet
        """
        super().__init__(
            name=name,
            mass=mass,
            radius=radius,
            surface_gravity=surface_gravity,
            orbit_semi_major_axis=orbit_semi_major_axis,
            reference_pressure=reference_pressure,
            equilibrium_temperature=equilibrium_temperature,
            bond_albedo=bond_albedo,
            star_radius=star_radius,
            star_effective_temperature=star_effective_temperature
        )

        if equilibrium_temperature is None:
            self.equilibrium_temperature = self.calculate_planetary_equilibrium_temperature()[0]
        else:
            self.equilibrium_temperature = equilibrium_temperature

        if mass is None:
            self.mass = self.surface_gravity2mass(self.surface_gravity, self.radius)
        else:
            self.mass = mass

    @staticmethod
    def surface_gravity2mass(surface_gravity, radius):
        return surface_gravity * radius ** 2 / nc.G


class SpectralModel:
    default_line_species = [
        'CH4_main_iso',
        'CO_all_iso',
        'CO2_main_iso',
        'H2O_main_iso',
        'HCN_main_iso',
        'K',
        'Na_allard',
        'NH3_main_iso',
        'TiO_all_iso_exo',
        'VO'
    ]
    default_rayleigh_species = [
        'H2',
        'He'
    ]
    default_continuum_opacities = [
        'H2-H2',
        'H2-He'
    ]

    def __init__(self, planet_name, wavelength_boundaries, lbl_opacity_sampling, do_scat_emis,
                 t_int, metallicity, co_ratio, p_cloud, kappa_ir_z0=0.01, gamma=0.4, p_quench_c=None, haze_factor=1,
                 atmosphere_file=None, wavelengths=None, transit_radius=None, eclipse_depth=None,
                 spectral_radiosity=None, star_spectral_radiosity=None,
                 temperature=None, mass_fractions=None, planet_model_file=None, model_suffix='', filename=None):
        self.planet_name = planet_name
        self.wavelength_boundaries = wavelength_boundaries
        self.lbl_opacity_sampling = lbl_opacity_sampling
        self.do_scat_emis = do_scat_emis
        self.t_int = t_int
        self.metallicity = metallicity
        self.co_ratio = co_ratio
        self.p_cloud = p_cloud

        self.kappa_ir_z0 = kappa_ir_z0
        self.gamma = gamma
        self.p_quench_c = p_quench_c
        self.haze_factor = haze_factor

        self.atmosphere_file = atmosphere_file

        self.temperature = temperature
        self.mass_fractions = mass_fractions

        self.wavelengths = wavelengths
        self.transit_radius = transit_radius
        self.eclipse_depth = eclipse_depth
        self.spectral_radiosity = spectral_radiosity
        self.star_spectral_radiosity = star_spectral_radiosity

        self.name_suffix = model_suffix

        if planet_model_file is None:
            self.planet_model_file = Planet(planet_name).get_filename()
        else:
            self.planet_model_file = planet_model_file

        if filename is None:
            self.filename = self.get_filename()

    def calculate_eclipse_depth(self, atmosphere: Radtrans, planet: Planet):
        star_radiosity_filename = self.get_star_radiosity_filename(planet.star_effective_temperature, path=module_dir)

        if not os.path.isfile(star_radiosity_filename):
            self.generate_phoenix_star_spectrum_file(star_radiosity_filename, planet.star_effective_temperature)

        data = np.loadtxt(star_radiosity_filename)
        star_wavelength = data[:, 0] * 1e6  # m to um
        star_radiosity = data[:, 1] * 1e8 * np.pi  # erg.s-1.cm-2.sr-1/A to erg.s-1.cm-2/cm

        print('Calculating eclipse depth...')
        # TODO fix stellar flux calculated multiple time if do_scat_emis is True
        wavelengths, planet_radiosity = self.calculate_emission_spectrum(atmosphere, planet)
        star_radiosity = fr.rebin_spectrum(star_wavelength, star_radiosity, wavelengths)

        eclipse_depth = (planet_radiosity * planet.radius ** 2) / (star_radiosity * planet.star_radius ** 2)

        return wavelengths, eclipse_depth, planet_radiosity

    def calculate_emission_spectrum(self, atmosphere: Radtrans, planet: Planet):
        print('Calculating emission spectrum...')

        atmosphere.calc_flux(
            self.temperature,
            self.mass_fractions,
            planet.surface_gravity,
            self.mass_fractions['MMW'],
            Tstar=planet.star_effective_temperature,
            Rstar=planet.star_radius / nc.r_sun,
            semimajoraxis=planet.orbit_semi_major_axis / nc.AU,
            Pcloud=self.p_cloud
        )

        flux = self.radiosity_erg_hz2radiosity_erg_cm(atmosphere.flux, atmosphere.freq)
        wavelengths = nc.c / atmosphere.freq * 1e4  # m to um

        return wavelengths, flux

    def calculate_transmission_spectrum(self, atmosphere: Radtrans, planet: Planet):
        print('Calculating transmission spectrum...')

        atmosphere.calc_transm(
            self.temperature,
            self.mass_fractions,
            planet.surface_gravity,
            self.mass_fractions['MMW'],
            R_pl=planet.radius,
            P0_bar=planet.reference_pressure,
            Pcloud=self.p_cloud,
            haze_factor=self.haze_factor,
        )

        transit_radius = (atmosphere.transm_rad / planet.star_radius) ** 2
        wavelengths = nc.c / atmosphere.freq * 1e4  # m to um

        return wavelengths, transit_radius

    @staticmethod
    def generate_phoenix_star_spectrum_file(star_spectrum_file, star_effective_temperature):
        stellar_spectral_radiance = get_PHOENIX_spec(star_effective_temperature)

        # Convert the spectrum to units accepted by the ETC website
        # Don't take the first wavelength to avoid spike in convolution
        wavelength_stellar = \
            stellar_spectral_radiance[1:, 0]  # in cm
        stellar_spectral_radiance = SpectralModel.radiosity_erg_hz2radiosity_erg_cm(
            stellar_spectral_radiance[1:, 1],
            nc.c / wavelength_stellar
        )

        wavelength_stellar *= 1e-2  # cm to m
        stellar_spectral_radiance *= 1e-8 / np.pi  # erg.s-1.cm-2/cm to erg.s-1.cm-2.sr-1/A

        np.savetxt(star_spectrum_file, np.transpose((wavelength_stellar, stellar_spectral_radiance)))

    def get_filename(self):
        name = self.get_name()

        return planet_models_directory + os.path.sep + name + '.pkl'

    def get_name(self):
        name = 'spectral_model_'
        name += f"{self.planet_name.replace(' ', '_')}_" \
                f"Tint{self.t_int}K_Z{self.metallicity}_co{self.co_ratio}_pc{self.p_cloud}bar_" \
                f"{self.wavelength_boundaries[0]}-{self.wavelength_boundaries[1]}um_ds{self.lbl_opacity_sampling}"

        if self.do_scat_emis:
            name += '_scat'
        else:
            name += '_noscat'

        if self.name_suffix != '':
            name += f'_{self.name_suffix}'

        return name

    @staticmethod
    def get_star_radiosity_filename(star_effective_temperature, path='.'):
        return f'{path}/crires/star_spectrum_{star_effective_temperature}K.dat'

    def init_mass_fractions(self, atmosphere, temperature, include_species, mass_fractions=None):
        from petitRADTRANS.poor_mans_nonequ_chem import poor_mans_nonequ_chem as pm  # import is here because it's long!

        if mass_fractions is None:
            mass_fractions = {}
        elif not isinstance(mass_fractions, dict):
            raise ValueError(
                f"mass fractions must be in a dict, but the input was of type '{type(mass_fractions)}'")

        pressures = atmosphere.press * 1e-6  # cgs to bar

        if np.size(self.co_ratio) == 1:
            co_ratios = np.ones_like(pressures) * self.co_ratio
        else:
            co_ratios = self.co_ratio

        if np.size(self.metallicity) == 1:
            metallicity = np.ones_like(pressures) * self.metallicity
        else:
            metallicity = self.metallicity

        abundances = pm.interpol_abundances(
            COs_goal_in=co_ratios,
            FEHs_goal_in=metallicity,
            temps_goal_in=temperature,
            pressures_goal_in=pressures,
            Pquench_carbon=self.p_quench_c
        )

        # Check mass_fractions keys
        for key in mass_fractions:
            if key not in atmosphere.line_species and key not in abundances:
                raise KeyError(f"key '{key}' not in line species list or "
                               f"standard petitRADTRANS mass fractions dict")

        # Get the right keys for the mass fractions dictionary
        mass_fractions_dict = {}

        for key in abundances:
            found = False

            for line_species_name in atmosphere.line_species:
                line_species = line_species_name.split('_', 1)[0]

                if line_species == 'C2H2':   # C2H2 special case
                    line_species += ',acetylene'

                if key == line_species:
                    if key not in include_species:
                        mass_fractions_dict[line_species_name] = np.zeros_like(temperature)
                    elif line_species_name in mass_fractions:
                        mass_fractions_dict[line_species_name] = mass_fractions[line_species_name]
                    else:
                        mass_fractions_dict[line_species_name] = abundances[line_species]

                    found = True

                    break

            if not found:
                if key in mass_fractions:
                    mass_fractions_dict[key] = mass_fractions[key]
                else:
                    mass_fractions_dict[key] = abundances[key]

        return mass_fractions_dict

    def init_temperature_guillot(self, planet: Planet, atmosphere: Radtrans):
        kappa_ir = self.kappa_ir_z0 * 10 ** self.metallicity
        pressures = atmosphere.press * 1e-6  # cgs to bar

        temperature = guillot_global(
            pressures, kappa_ir, self.gamma, planet.surface_gravity, self.t_int, planet.equilibrium_temperature
        )

        return temperature

    def save(self):
        with open(self.get_filename(), 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, file):
        with open(file, 'rb') as f:
            return pickle.load(f)

    @classmethod
    def get(cls, planet_name, wavelength_boundaries, lbl_opacity_sampling, pressures, do_scat_emis, t_int,
            metallicity, co_ratio, p_cloud, kappa_ir_z0=0.01, gamma=0.4, p_quench_c=None, haze_factor=1,
            line_species_list='default', rayleigh_species='default', continuum_opacities='default',
            include_species='all', model_suffix='', atmosphere=None, calculate_transmission_spectrum=False,
            calculate_emission_spectrum=False, calculate_eclipse_depth=False,
            rewrite=True):
        # Initialize model
        model = cls.species_init(
            include_species=include_species,
            planet_name=planet_name,
            wavelength_boundaries=wavelength_boundaries,
            lbl_opacity_sampling=lbl_opacity_sampling,
            do_scat_emis=do_scat_emis,
            t_int=t_int,
            metallicity=metallicity,
            co_ratio=co_ratio,
            p_cloud=p_cloud,
            kappa_ir_z0=kappa_ir_z0,
            gamma=gamma,
            p_quench_c=p_quench_c,
            haze_factor=haze_factor,
            model_suffix=model_suffix
        )

        # Generate or load model
        return cls.generate_from(
            model=model,
            pressures=pressures,
            line_species_list=line_species_list,
            rayleigh_species=rayleigh_species,
            continuum_opacities=continuum_opacities,
            include_species=include_species,
            model_suffix=model_suffix,
            atmosphere=atmosphere,
            calculate_transmission_spectrum=calculate_transmission_spectrum,
            calculate_emission_spectrum=calculate_emission_spectrum,
            calculate_eclipse_depth=calculate_eclipse_depth,
            rewrite=rewrite
        )

    @classmethod
    def generate_from(cls, model, pressures,
                      line_species_list='default', rayleigh_species='default', continuum_opacities='default',
                      include_species=None, model_suffix='',
                      atmosphere=None, temperature_profile=None, mass_fractions=None,
                      calculate_transmission_spectrum=False, calculate_emission_spectrum=False,
                      calculate_eclipse_depth=False,
                      rewrite=False):
        if not hasattr(include_species, '__iter__') or isinstance(include_species, str):
            include_species = [include_species]
        elif include_species is None:
            include_species = ['all']

        if len(include_species) > 1:
            raise ValueError("Please include either only one species or all of them using keyword 'all'")

        # Check if model already exists
        if os.path.isfile(model.filename) and not rewrite:
            print(f"Model '{model.filename}' already exists, loading from file...")
            return model.load(model.filename)
        else:
            if os.path.isfile(model.filename) and rewrite:
                print(f"Rewriting already existing model '{model.filename}'...")

            print(f"Generating model '{model.filename}'...")

            # Initialize species
            if line_species_list == 'default':
                line_species_list = cls.default_line_species

            if rayleigh_species == 'default':
                rayleigh_species = cls.default_rayleigh_species

            if continuum_opacities == 'default':
                continuum_opacities = cls.default_continuum_opacities

            if include_species == ['all']:
                include_species = [species_name.split('_', 1)[0] for species_name in line_species_list]

            # Generate the model
            return cls._generate(
                model, pressures, line_species_list, rayleigh_species, continuum_opacities, include_species,
                model_suffix, atmosphere, temperature_profile, mass_fractions, calculate_transmission_spectrum,
                calculate_emission_spectrum, calculate_eclipse_depth
            )

    @staticmethod
    def radiosity_erg_hz2radiosity_erg_cm(radiosity_erg_hz, frequency):
        """
        Convert a radiosity from erg.s-1.cm-2.sr-1/Hz to erg.s-1.cm-2.sr-1/cm at a given frequency.
        Steps:
            [cm] = c[cm.s-1] / [Hz]
            => d[cm]/d[Hz] = d(c / [Hz])/d[Hz]
            => d[cm]/d[Hz] = c / [Hz]**2
            => d[Hz]/d[cm] = [Hz]**2 / c
            integral of flux must be conserved: radiosity_erg_cm * d[cm] = radiosity_erg_hz * d[Hz]
            radiosity_erg_cm = radiosity_erg_hz * d[Hz]/d[cm]
            => radiosity_erg_cm = radiosity_erg_hz * frequency**2 / c

        Args:
            radiosity_erg_hz: (erg.s-1.cm-2.sr-1/Hz)
            frequency: (Hz)

        Returns:
            (erg.s-1.cm-2.sr-1/cm) the radiosity in converted units
        """
        return radiosity_erg_hz * frequency ** 2 / nc.c

    @classmethod
    def species_init(cls, include_species, planet_name, wavelength_boundaries, lbl_opacity_sampling, do_scat_emis,
                     t_int, metallicity, co_ratio, p_cloud, kappa_ir_z0=0.01, gamma=0.4, p_quench_c=None, haze_factor=1,
                     atmosphere_file=None, wavelengths=None, transit_radius=None, temperature=None,
                     mass_fractions=None, planet_model_file=None, model_suffix='', filename=None):
        # Initialize include_species
        if not hasattr(include_species, '__iter__') or isinstance(include_species, str):
            include_species = [include_species]

        if len(include_species) > 1:
            raise ValueError("Please include either only one species or all of them using keyword 'all'")
        else:
            if model_suffix == '':
                species_suffix = f'{include_species[0]}'
            else:
                species_suffix = f'_{include_species[0]}'

        # Initialize model
        return cls(
            planet_name=planet_name,
            wavelength_boundaries=wavelength_boundaries,
            lbl_opacity_sampling=lbl_opacity_sampling,
            do_scat_emis=do_scat_emis,
            t_int=t_int,
            metallicity=metallicity,
            co_ratio=co_ratio,
            p_cloud=p_cloud,
            kappa_ir_z0=kappa_ir_z0,
            gamma=gamma,
            p_quench_c=p_quench_c,
            haze_factor=haze_factor,
            atmosphere_file=atmosphere_file,
            wavelengths=wavelengths,
            transit_radius=transit_radius,
            temperature=temperature,
            mass_fractions=mass_fractions,
            planet_model_file=planet_model_file,
            model_suffix=model_suffix + species_suffix,
            filename=filename
        )

    @staticmethod
    def _generate(model, pressures, line_species_list, rayleigh_species, continuum_opacities, include_species,
                  model_suffix, atmosphere=None, temperature_profile=None, mass_fractions=None,
                  calculate_transmission_spectrum=False, calculate_emission_spectrum=False,
                  calculate_eclipse_depth=False):
        if atmosphere is None:
            atmosphere, model.atmosphere_file = model.get_atmosphere_model(
                model.wavelength_boundaries, pressures, line_species_list, rayleigh_species, continuum_opacities,
                model_suffix
            )
        else:
            model.atmosphere_file = SpectralModel._get_hires_atmosphere_filename(
                pressures, model.wavelength_boundaries, model.lbl_opacity_sampling, model_suffix
            )

        # A Planet needs to be generated and saved first
        model.planet_model_file = Planet.generate_filename(model.planet_name)
        planet = Planet.load(model.planet_name, model.planet_model_file)

        if temperature_profile is None:
            model.temperature = model.init_temperature_guillot(
                planet=planet,
                atmosphere=atmosphere
            )
        elif isinstance(temperature_profile, (float, int)):
            model.temperature = np.ones_like(atmosphere.press) * temperature_profile
        elif np.size(temperature_profile) == np.size(atmosphere.press):
            model.temperature = np.asarray(temperature_profile)
        else:
            raise ValueError(f"could not initialize temperature profile using input {temperature_profile}; "
                             f"possible inputs are None, float, int, "
                             f"or a 1-D array of the same size of argument 'pressures' ({np.size(atmosphere.press)})")

        # Generate mass fractions from equilibrium chemistry first to have all the keys
        # TODO generate the mass fractions dict without calling equilibrium chemistry
        model.mass_fractions = model.init_mass_fractions(
            atmosphere=atmosphere,
            temperature=model.temperature,
            include_species=include_species,
            mass_fractions=mass_fractions
        )

        if not calculate_transmission_spectrum and not calculate_emission_spectrum and not calculate_eclipse_depth:
            print(f"No spectrum will be calculated")

            return model

        if calculate_transmission_spectrum:
            model.wavelengths, model.transit_radius = model.calculate_transmission_spectrum(
                atmosphere=atmosphere,
                planet=planet
            )

        if calculate_emission_spectrum and not calculate_eclipse_depth:
            model.wavelengths, model.spectral_radiosity = model.calculate_emission_spectrum(
                atmosphere=atmosphere,
                planet=planet
            )
        elif calculate_eclipse_depth:
            model.wavelengths, model.eclipse_depth, model.spectral_radiosity = model.calculate_eclipse_depth(
                atmosphere=atmosphere,
                planet=planet
            )

        return model

    @staticmethod
    def _get_hires_atmosphere_filename(pressures, wlen_bords_micron, lbl_opacity_sampling, do_scat_emis,
                                       model_suffix=''):
        filename = planet_models_directory + os.path.sep \
                   + f"atmosphere_{np.max(pressures)}-{np.min(pressures)}bar_" \
                     f"{wlen_bords_micron[0]}-{wlen_bords_micron[1]}um_ds{lbl_opacity_sampling}"

        if do_scat_emis:
            filename += '_scat'

        if model_suffix != '':
            filename += f"_{model_suffix}"

        filename += '.pkl'

        return filename

    @staticmethod
    def get_atmosphere_model(wlen_bords_micron, pressures,
                             line_species_list=None, rayleigh_species=None, continuum_opacities=None,
                             lbl_opacity_sampling=1, do_scat_emis=False,
                             model_suffix=''):
        atmosphere_filename = SpectralModel._get_hires_atmosphere_filename(
            pressures, wlen_bords_micron, lbl_opacity_sampling, do_scat_emis, model_suffix
        )

        if os.path.isfile(atmosphere_filename):
            print('Loading atmosphere model...')
            with open(atmosphere_filename, 'rb') as f:
                atmosphere = pickle.load(f)
        else:
            atmosphere = SpectralModel.init_atmosphere(
                pressures, wlen_bords_micron, line_species_list, rayleigh_species, continuum_opacities,
                lbl_opacity_sampling, do_scat_emis
            )

            print('Saving atmosphere model...')
            with open(atmosphere_filename, 'wb') as f:
                pickle.dump(atmosphere, f)

        return atmosphere, atmosphere_filename

    @staticmethod
    def init_atmosphere(pressures, wlen_bords_micron, line_species_list, rayleigh_species, continuum_opacities,
                        lbl_opacity_sampling, do_scat_emis):
        print('Generating atmosphere...')

        atmosphere = Radtrans(
            wlen_bords_micron=wlen_bords_micron,
            mode='lbl',
            lbl_opacity_sampling=lbl_opacity_sampling,
            do_scat_emis=do_scat_emis,
            rayleigh_species=rayleigh_species,
            continuum_opacities=continuum_opacities,
            line_species=line_species_list
        )

        atmosphere.setup_opa_structure(pressures)

        return atmosphere


class ParametersDict(dict):
    def __init__(self, t_int, metallicity, co_ratio, p_cloud):
        super().__init__()

        self['intrinsic_temperature'] = t_int
        self['metallicity'] = metallicity
        self['co_ratio'] = co_ratio
        self['p_cloud'] = p_cloud

    def to_str(self):
        return f"T_int = {self['intrinsic_temperature']}, [Fe/H] = {self['metallicity']}, C/O = {self['co_ratio']}, " \
               f"P_cloud = {self['p_cloud']}"


def _get_generic_planet_name(radius, surface_gravity, equilibrium_temperature):
    return f"generic_{radius / nc.r_jup:.2f}Rjup_logg{np.log10(surface_gravity):.2f}_teq{equilibrium_temperature:.2f}K"


def generate_model_grid(models, pressures,
                        line_species_list='default', rayleigh_species='default', continuum_opacities='default',
                        model_suffix='', atmosphere=None, temperature_profile=None, mass_fractions=None,
                        calculate_transmission_spectrum=False, calculate_emission_spectrum=False,
                        calculate_eclipse_depth=False,
                        rewrite=False, save=False):
    """
    Get a grid of models, generate it if needed.
    Models are generated using petitRADTRANS in its line-by-line mode. Clouds are modelled as a gray deck.
    Output will be organized hierarchically as follows: model string > included species

    Args:
        models: dictionary of models
        pressures: (bar) 1D-array containing the pressure grid of the models
        line_species_list: list containing all the line species to include in the models
        rayleigh_species: list containing all the rayleigh species to include in the models
        continuum_opacities: list containing all the continua to include in the models
        model_suffix: suffix of the model
        atmosphere: pre-loaded Radtrans object
        temperature_profile: if None, a Guillot temperature profile is generated, if int or float, an isothermal
            temperature profile is generated, if 1-D array of the same size of pressures, the temperature profile is
            directly used
        mass_fractions: if None, equilibrium chemistry is used, if dict, the values from the dict are used
        calculate_transmission_spectrum: if True, calculate the transmission spectrum of the model
        calculate_emission_spectrum: if True, calculate the emission spectrum of the model
        calculate_eclipse_depth: if True, calculate the eclipse depth, and the emission spectrum, of the model
        rewrite: if True, rewrite all the models, even if they already exists
        save: if True, save the models once generated

    Returns:
        models: a dictionary containing all the requested models
    """
    i = 0

    for model in models:
        for species in models[model]:
            i += 1
            print(f"Model {i}/{len(models) * len(models[model])}...")

            models[model][species] = SpectralModel.generate_from(
                model=models[model][species],
                pressures=pressures,
                include_species=species,
                line_species_list=line_species_list,
                rayleigh_species=rayleigh_species,
                continuum_opacities=continuum_opacities,
                model_suffix=model_suffix,
                atmosphere=atmosphere,
                temperature_profile=temperature_profile,
                mass_fractions=mass_fractions,
                calculate_transmission_spectrum=calculate_transmission_spectrum,
                calculate_emission_spectrum=calculate_emission_spectrum,
                calculate_eclipse_depth=calculate_eclipse_depth,
                rewrite=rewrite
            )

            if save:
                models[model][species].save()

    return models


def get_model_grid(planet_name, lbl_opacity_sampling, do_scat_emis, parameter_dicts, species_list, pressures,
                   wavelength_boundaries, line_species_list='default', rayleigh_species='default',
                   continuum_opacities='default', model_suffix='', atmosphere=None,
                   calculate_transmission_spectrum=False, calculate_emission_spectrum=False,
                   calculate_eclipse_depth=False,
                   rewrite=False, save=False):
    """
    Get a grid of models, generate it if needed.
    Models are generated using petitRADTRANS in its line-by-line mode. Clouds are modelled as a gray deck.
    Output will be organized hierarchically as follows: model string > included species

    Args:
        planet_name: name of the planet modelled
        lbl_opacity_sampling: downsampling coefficient of
        do_scat_emis: if True, include the scattering for emission spectra
        parameter_dicts: list of dictionaries containing the models parameters
        species_list: list of lists of species to include in the models (e.g. ['all', 'H2O', 'CH4'])
        pressures: (bar) 1D-array containing the pressure grid of the models
        wavelength_boundaries: (um) size-2 array containing the min and max wavelengths
        line_species_list: list containing all the line species to include in the models
        rayleigh_species: list containing all the rayleigh species to include in the models
        continuum_opacities: list containing all the continua to include in the models
        model_suffix: suffix of the model
        atmosphere: pre-loaded Radtrans object
        calculate_transmission_spectrum: if True, calculate the transmission spectrum of the model
        calculate_emission_spectrum: if True, calculate the emission spectrum of the model
        calculate_eclipse_depth: if True, calculate the eclipse depth, and the emission spectrum, of the model
        rewrite: if True, rewrite all the models, even if they already exists
        save: if True, save the models once generated

    Returns:
        models: a dictionary containing all the requested models
    """
    models = {}
    i = 0

    for parameter_dict in parameter_dicts:
        models[parameter_dict.to_str()] = {}

        for species in species_list:
            i += 1
            print(f"Model {i}/{len(parameter_dicts) * len(species_list)}...")

            models[parameter_dict.to_str()][species] = SpectralModel.get(
                planet_name=planet_name,
                wavelength_boundaries=wavelength_boundaries,
                lbl_opacity_sampling=lbl_opacity_sampling,
                do_scat_emis=do_scat_emis,
                t_int=parameter_dict['intrinsic_temperature'],
                metallicity=parameter_dict['metallicity'],
                co_ratio=parameter_dict['co_ratio'],
                p_cloud=parameter_dict['p_cloud'],
                pressures=pressures,
                include_species=species,
                kappa_ir_z0=0.01,
                gamma=0.4,
                p_quench_c=None,
                haze_factor=1,
                line_species_list=line_species_list,
                rayleigh_species=rayleigh_species,
                continuum_opacities=continuum_opacities,
                model_suffix=model_suffix,
                atmosphere=atmosphere,
                calculate_transmission_spectrum=calculate_transmission_spectrum,
                calculate_emission_spectrum=calculate_emission_spectrum,
                calculate_eclipse_depth=calculate_eclipse_depth,
                rewrite=rewrite
            )

            if save:
                models[parameter_dict.to_str()][species].save()

    return models


def get_parameter_dicts(t_int: list, metallicity: list, co_ratio: list, p_cloud: list):
    """
    Generate a parameter dictionary from parameters.
    To be used in get_model_grid()

    Args:
        t_int: (K) intrinsic temperature of the planet
        metallicity: metallicity of the planet
        co_ratio: C/O ratio of the planet
        p_cloud: (bar) cloud top pressure of the planet

    Returns:
        parameter_dict: a ParameterDict
    """

    parameter_dicts = []

    for t in t_int:
        for z in metallicity:
            for co in co_ratio:
                for pc in p_cloud:
                    parameter_dicts.append(
                        ParametersDict(
                            t_int=t,
                            metallicity=z,
                            co_ratio=co,
                            p_cloud=pc
                        )
                    )

    return parameter_dicts


def init_model_grid(planet_name, lbl_opacity_sampling, do_scat_emis, parameter_dicts, species_list,
                    wavelength_boundaries, model_suffix=''):
    # Initialize models
    models = {}
    all_models_exist = True

    for parameter_dict in parameter_dicts:
        models[parameter_dict.to_str()] = {}

        for species in species_list:
            models[parameter_dict.to_str()][species] = SpectralModel.species_init(
                include_species=species,
                planet_name=planet_name,
                wavelength_boundaries=wavelength_boundaries,
                lbl_opacity_sampling=lbl_opacity_sampling,
                do_scat_emis=do_scat_emis,
                t_int=parameter_dict['intrinsic_temperature'],
                metallicity=parameter_dict['metallicity'],
                co_ratio=parameter_dict['co_ratio'],
                p_cloud=parameter_dict['p_cloud'],
                kappa_ir_z0=0.01,
                gamma=0.4,
                p_quench_c=None,
                haze_factor=1,
                model_suffix=model_suffix
            )

            if not os.path.isfile(models[parameter_dict.to_str()][species].filename) and all_models_exist:
                all_models_exist = False

    return models, all_models_exist


def load_model_grid(models):
    i = 0

    for model in models:
        for species in models[model]:
            i += 1
            print(f"Loading model {i}/{len(models) * len(models[model])} from '{models[model][species].filename}'...")

            models[model][species] = models[model][species].load(models[model][species].filename)

    return models


def make_generic_planet(radius, surface_gravity, equilibrium_temperature,
                        star_effective_temperature=5500, star_radius=nc.r_sun, orbit_semi_major_axis=nc.AU):
    name = _get_generic_planet_name(radius, surface_gravity, equilibrium_temperature)

    return SimplePlanet(
        name,
        radius,
        surface_gravity,
        star_effective_temperature,
        star_radius,
        orbit_semi_major_axis,
        equilibrium_temperature=equilibrium_temperature
    )
