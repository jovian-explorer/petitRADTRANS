"""Stores useful physical functions.
"""
import numpy as np
from scipy.ndimage.filters import uniform_filter1d

import petitRADTRANS.nat_cst as nc


def b(temperature, nu):
    """Returns the Planck function :math:`B_{\\nu}(T)` in units of
    :math:`\\rm erg/s/cm^2/Hz/steradian`.

    Args:
        temperature (float):
            Temperature in K.
        nu:
            Array containing the frequency in Hz.
    """

    planck_function = 2. * nc.h * nu ** 3. / nc.c ** 2. / (np.exp(nc.h * nu / nc.kB / temperature) - 1.)

    return planck_function


def d_b_d_temperature(temperature, nu):
    """Returns the derivative of the Planck function with respect to the temperature in units of
    :math:`\\rm erg/s/cm^2/Hz/steradian`.

    Args:
        temperature:
            Temperature in K.
        nu:
            Array containing the frequency in Hz.
    Returns:

    """
    planck_function = b(temperature, nu)
    planck_function /= np.exp(nc.h * nu / nc.kB / temperature) - 1.
    planck_function *= np.exp(nc.h * nu / nc.kB / temperature) * nc.h * nu / nc.kB / temperature ** 2.

    return planck_function


def get_dist(t_irr, dist, t_star, r_star, mode, mode_what):
    # TODO rework/replace this function
    mu_star = 0.
    angle_use = False

    if (mode != 'p') & (mode != 'd'):
        mu_star = float(mode)
        angle_use = True

    if mode_what == 'temp':
        if angle_use:
            t_irr = ((r_star * nc.r_sun / (dist * nc.AU)) ** 2. * t_star ** 4. * mu_star) ** 0.25
        elif mode == 'p':
            t_irr = ((r_star * nc.r_sun / (dist * nc.AU)) ** 2. * t_star ** 4. / 4.) ** 0.25
        else:
            t_irr = ((r_star * nc.r_sun / (dist * nc.AU)) ** 2. * t_star ** 4. / 2.) ** 0.25
        return t_irr
    elif mode_what == 'dist':
        if angle_use:
            dist = np.sqrt((r_star * nc.r_sun) ** 2. * (t_star / t_irr) ** 4. * mu_star) / nc.AU
        elif mode == 'p':
            dist = np.sqrt((r_star * nc.r_sun) ** 2. * (t_star / t_irr) ** 4. / 4.) / nc.AU
        else:
            dist = np.sqrt((r_star * nc.r_sun) ** 2. * (t_star / t_irr) ** 4. / 2.) / nc.AU
        return dist


def get_guillot_2010_temperature_profile(pressure, infrared_mean_opacity, gamma, gravity, intrinsic_temperature,
                                         equilibrium_temperature, redistribution_coefficient=0.25):
    """ Returns a temperature array, in units of K,
    of the same dimensions as the pressure P
    (in bar).

    For this the temperature model of Guillot (2010) is used (his Equation 29).
    Source: https://doi.org/10.1051/0004-6361/200913396

    Args:
        pressure:
            numpy array of floats, containing the input pressure in bars.
        infrared_mean_opacity:
            The infrared mean opacity in units of :math:`\\rm cm^2/s`.
        gamma:
            The ratio between the visual and infrared mean opacities.
        gravity:
            The planetary gravity at the given pressures in units of :math:`\\rm cm/s^2`.
        intrinsic_temperature:
            The planetary intrinsic temperature (in units of K).
        equilibrium_temperature:
            The planetary equilibrium temperature (in units of K).
        redistribution_coefficient:
            The redistribution coefficient of the irradiance. A value of 1 corresponds to the substellar point, 1/2 for
            the day-side average and 1/4 for the global average.
    """
    # Estimate tau from eq. 24: m is the column mass, dm = rho * dz, dP / dz = -g * rho, so m = P / g
    tau = infrared_mean_opacity * pressure * 1e6 / gravity
    t_irr = equilibrium_temperature * 2.0 ** 0.5  # from eqs. 1 and 2

    temperature = (
        0.75 * intrinsic_temperature ** 4. * (2. / 3. + tau)
        + 0.75 * t_irr ** 4. * redistribution_coefficient
        * (
            2. / 3.
            + 1. / gamma / 3. ** 0.5
            + (gamma / 3. ** 0.5 - 1. / 3. ** 0.5 / gamma) * np.exp(-gamma * tau * 3. ** 0.5)
        )
    ) ** 0.25

    return temperature


# TODO remove deprecated functions
def guillot_day(pressure, kappa_ir, gamma, grav, t_int, t_equ):
    """ Returns a temperature array, in units of K,
    of the same dimensions as the pressure P
    (in bar). For this the temperature model of Guillot (2010)
    is used (his Equation 29), in the case of averaging the flux over the day side of the planet.

    Args:
        pressure:
            numpy array of floats, containing the input pressure in bars.
        kappa_ir (float):
            The infrared opacity in units of :math:`\\rm cm^2/s`.
        gamma (float):
            The ratio between the visual and infrated opacity.
        grav (float):
            The planetary surface gravity in units of :math:`\\rm cm/s^2`.
        t_int (float):
            The planetary internal temperature (in units of K).
        t_equ (float):
            The planetary equilibrium temperature (in units of K).
    """
    return get_guillot_2010_temperature_profile(
        pressure=pressure,
        infrared_mean_opacity=kappa_ir,
        gamma=gamma,
        gravity=grav,
        intrinsic_temperature=t_int,
        equilibrium_temperature=t_equ,
        redistribution_coefficient=0.5
    )


def guillot_global(pressure, kappa_ir, gamma, grav, t_int, t_equ):
    """ Returns a temperature array, in units of K,
    of the same dimensions as the pressure P
    (in bar). For this the temperature model of Guillot (2010)
    is used (his Equation 29), in the case of averaging the flux over the whole planetary surface.

    Args:
        pressure:
            numpy array of floats, containing the input pressure in bars.
        kappa_ir (float):
            The infrared opacity in units of :math:`\\rm cm^2/s`.
        gamma (float):
            The ratio between the visual and infrated opacity.
        grav (float):
            The planetary surface gravity in units of :math:`\\rm cm/s^2`.
        t_int (float):
            The planetary internal temperature (in units of K).
        t_equ (float):
            The planetary equilibrium temperature (in units of K).
    """
    return get_guillot_2010_temperature_profile(
        pressure=pressure,
        infrared_mean_opacity=kappa_ir,
        gamma=gamma,
        gravity=grav,
        intrinsic_temperature=t_int,
        equilibrium_temperature=t_equ,
        redistribution_coefficient=0.25
    )


def guillot_global_ret(pressure, delta, gamma, t_int, t_equ):
    """Global Guillot P-T formula with kappa/gravity replaced by delta."""
    delta = np.abs(delta)
    gamma = np.abs(gamma)
    t_int = np.abs(t_int)
    t_equ = np.abs(t_equ)
    tau = pressure * 1e6 * delta
    t_irr = t_equ * np.sqrt(2.)
    temperature = (0.75 * t_int ** 4. * (2. / 3. + tau)
                   + 0.75 * t_irr ** 4. / 4.
                   * (2. / 3. + 1. / gamma / 3. ** 0.5
                      + (gamma / 3. ** 0.5 - 1. / 3. ** 0.5 / gamma)
                      * np.exp(-gamma * tau * 3. ** 0.5))) ** 0.25
    return temperature


def guillot_modif(pressure, delta, gamma, t_int, t_equ, ptrans, alpha):
    """Modified Guillot P-T formula"""
    return guillot_global_ret(
        pressure,
        np.abs(delta),
        np.abs(gamma),
        np.abs(t_int), np.abs(t_equ)
    ) * (1. - alpha * (1. / (1. + np.exp((np.log(pressure / ptrans))))))


def make_press_temp(rad_trans_params):  # TODO pressure grid in input?
    """Function to make temp."""
    press_many = np.logspace(-8, 5, 260)
    t_no_ave = guillot_modif(press_many,
                             1e1 ** rad_trans_params['log_delta'], 1e1 ** rad_trans_params['log_gamma'],
                             rad_trans_params['intrinsic_temperature'], rad_trans_params['equilibrium_temperature'],
                             1e1 ** rad_trans_params['log_p_trans'], rad_trans_params['alpha'])

    # new
    press_many_new = 1e1 ** uniform_filter1d(np.log10(press_many), 25)
    t_new = uniform_filter1d(t_no_ave, 25)
    index_new = (press_many_new <= 1e3) & (press_many_new >= 1e-6)
    temp_new = t_new[index_new][::2]
    press_new = press_many_new[index_new][::2]

    return press_new, temp_new


def make_press_temp_iso(rad_trans_params):
    """Function to make temp."""
    press_many = np.logspace(-8, 5, 260)
    t_no_ave = rad_trans_params['equilibrium_temperature'] * np.ones_like(press_many)

    # new
    press_many_new = 1e1 ** uniform_filter1d(np.log10(press_many), 25)
    t_new = uniform_filter1d(t_no_ave, 25)
    index_new = (press_many_new <= 1e3) & (press_many_new >= 1e-6)
    temp_new = t_new[index_new][::2]
    press_new = press_many_new[index_new][::2]

    return press_new, temp_new
