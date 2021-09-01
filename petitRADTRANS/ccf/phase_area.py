import numpy as np
from petitRADTRANS import nat_cst as nc
from petitRADTRANS import rebin_give_width as rgw
import time

'''
############################################################
# Inclination zero
############################################################

def good(phi_phase, phi):
    return ((np.cos(phi_phase)*np.cos(phi) + np.sin(phi_phase)*np.sin(phi)) > 0.)

phi_phase = np.linspace(0.,2*np.pi,100)

def calc_visible_disc_fraction(phi_phase):
    phis = np.linspace(-np.pi/2.,np.pi/2.,1000)
    mean_phis = (phis[1:]+phis[:-1])/2.
    diff_phis = np.diff(phis)
    index = good(phi_phase,mean_phis)
    integral_terms = np.cos(mean_phis)*diff_phis
    return np.sum(integral_terms[index])/2.

#print(calc_visible_disc_fraction(np.pi/2.))

vis_disc_fraction = np.zeros_like(phi_phase)
for i in range(int(len(phi_phase))):
    vis_disc_fraction[i] = calc_visible_disc_fraction(phi_phase[i])

plt.plot(phi_phase/np.pi, vis_disc_fraction)

phi = np.linspace(0.,2*np.pi,1000)
ret = 0.5*(1.+np.cos(phi))
plt.plot(phi/np.pi,ret,'--')

'''


class Rtfl:
    """ Class to read precalculated spectra in. """

    def __init__(self, path=None, freq=None, flux=None):
        if path is not None:
            # Read in spectra
            dat = np.genfromtxt(path)
            self.freq = nc.c / (dat[:, 0] * 1e-4)
            self.flux = dat[:, 1]
        else:
            self.freq = freq
            self.flux = flux

    def make_observation(self, instrument_res, pixel_sampling, wlen_range,
                         snr_star=None, contrast=None, rel_velocity=0., phase=0., inclination=90.,
                         detector_bg=0.,
                         sky_transmission_use=None, contrast_better=1.):

        # Calculate fraction of dayside visible on planetary disk
        phase_reduction = calc_visible_disc_fraction_inc(phase, inclination)
        print('phase_reduction', phase_reduction)

        # Apply a doppler shift, using the relative radial velocity
        freq_use = self.freq * (1. - rel_velocity / nc.c)

        wlen_nm = nc.c / freq_use * 1e7
        # Calculate noise-less mock observation of planet in
        # isolation
        flux_lsf_iso, freq_instrument, flux_rebin_iso = \
            rgw.convolve_rebin(freq_use, self.flux * phase_reduction, instrument_res,
                               pixel_sampling, wlen_range)

        spectral_data = {
            'freq': freq_use, 'freq instrument': freq_instrument,
            'Isolated planet flux': self.flux * phase_reduction, 'Isolated planet flux, LSF': flux_lsf_iso,
            'Isolated planet flux, LSFed, rebinned to instrument': flux_rebin_iso
        }

        # Calculate noise affected and noise free mock observation
        # (planet plus star)
        if contrast is not None:
            # Paul in
            np.random.seed(int(time.time() * 100) % 8388607)
            # Paul out
            # np.random.seed(0)

            # Model actual observation
            stellar_flux = np.ones_like(self.flux) * np.mean(self.flux) / contrast

            # Do the following steps (line-by-line)
            # F_pl + F_s
            # Normalize to get correct SNR
            # Add telluric transmission
            # Add sky background etc
            observation = (self.flux * phase_reduction + stellar_flux / contrast_better) \
                / np.mean(stellar_flux) * snr_star ** 2. \
                * sky_transmission_use(wlen_nm)

            # print(np.sum(self.flux),phase_reduction,np.sum(stellar_flux), \
            #          np.sum(1./np.mean(stellar_flux)*SNR_star**2. *sky_transmission_use(wlen_nm)))
            # print('transmission_sum',np.sum(sky_transmission_use(wlen_nm)))

            # Convolve with LSF, rebin to instrument pixels
            buffer1, buffer2, observation = \
                rgw.convolve_rebin(freq_use, observation, instrument_res,
                                   pixel_sampling, wlen_range)
            observation[observation < 0.] = 0.

            # Add errorbars
            err_scale = np.sqrt(observation)
            # print('errrr', observation[np.isnan(err_scale)])
            err_scale[np.isnan(err_scale)] = 0.
            err_scale[err_scale < 0.] = 0.
            spectral_data['(F_s+F_pl)*T+D'] = observation \
                + np.random.normal(loc=0., scale=err_scale, size=int(len(observation))) \
                + detector_bg * snr_star ** 2

            spectral_data['(F_s+F_pl)*T+D, no err.'] = observation \
                + detector_bg * snr_star ** 2

            print('Error estimate', np.mean(
                (self.flux * phase_reduction)
                / np.mean(stellar_flux) * snr_star ** 2. * sky_transmission_use(wlen_nm)
            ) / np.mean(err_scale))

            # Model retrieved transmission + stellar profile

            # Do the following steps (line-by-line)
            # F_s
            # Normalize to get correct SNR
            # Add telluric transmission
            # Add sky background etc
            transmission = stellar_flux / contrast_better \
                / np.mean(stellar_flux) * snr_star ** 2. \
                * sky_transmission_use(wlen_nm)

            # Convolve with LSF, rebin to instrument pixels
            buffer1, buffer2, transmission = \
                rgw.convolve_rebin(freq_use, transmission, instrument_res, pixel_sampling, wlen_range)
            transmission[transmission < 0.] = 0.

            # Add errorbars
            err_scale = np.sqrt(transmission)
            # print('errrr2', transmission[np.isnan(err_scale)])
            err_scale[np.isnan(err_scale)] = 0.
            err_scale[err_scale < 0.] = 0.
            spectral_data['F_s*T+D'] = transmission \
                + np.random.normal(loc=0., scale=err_scale, size=int(len(transmission))) \
                + detector_bg * snr_star ** 2

        return spectral_data


def planet_vel(phases=None, times=None, k_p=None,
               inclination=None, m_star=None, sma=None,
               v_kepler=None, p_orb=None):
    """
    Calculate the relative velocity of a planet.

    Args:
        phases:
        times:
        k_p:
        inclination:
        m_star:
        sma:
        v_kepler:
        p_orb:

    Returns:

    """
    if (k_p is not None) or (v_kepler is not None):
        ret_val = None

        if times is not None:
            phases = times / p_orb * 2. * np.pi

        if k_p is not None:
            ret_val = k_p * np.cos(phases + np.pi / 2.)
        elif v_kepler is not None:
            k_p = v_kepler * np.sin(inclination * np.pi / 180.)
            ret_val = k_p * np.cos(phases + np.pi / 2.)
    else:
        p_orb = 2. * np.pi * np.sqrt(sma ** 3. / nc.G / m_star)

        if times is not None:
            phases = times / p_orb * 2. * np.pi

        v_kepler = np.sqrt(nc.G * m_star / sma)
        k_p = v_kepler * np.sin(inclination * np.pi / 180.)
        ret_val = k_p * np.cos(phases + np.pi / 2.)

    return ret_val


############################################################
# Non-zero inclination
############################################################
def good_phi_matrix(phi_phase, inclination, phi_grid, thetas_grid):
    # Calculate the position of the substellar point, rotate it about the y-axis for the inclination...
    turn = (90. - inclination) / 90. * np.pi / 2.
    subs_vec = np.array([np.cos(turn) * np.cos(phi_phase), np.sin(phi_phase), -np.sin(turn) * np.cos(phi_phase)])

    # Prepare position vectors from arrays, tp calculate normal vectors at positions where we calculate the area
    x = np.cos(phi_grid) * np.sin(thetas_grid)
    y = np.sin(phi_grid) * np.sin(thetas_grid)
    z = np.cos(thetas_grid)

    vecs = np.zeros(int(len(phi_grid[:, 0])) * int(len(phi_grid[0, :])) * 3).reshape(
        int(len(phi_grid[:, 0])),
        int(len(thetas_grid[0, :]))  # , 3  # TODO what does the 3 was for?
    )
    vecs[:, :, 0] = x
    vecs[:, :, 1] = y
    vecs[:, :, 2] = z

    # On the dayside if angular distance to substellar point is smaller than pi/2.
    scalar = np.sum(vecs * subs_vec, axis=2)
    return scalar > 0.


def calc_visible_disc_fraction_inc(phi_phase, inclination):
    # Inclination == 90. for perfectly transiting planet
    # Phase goes from 0 to 2 pi for full phase
    # Define coordinates, and coordinate distances, over which to take the area integral
    phis = np.linspace(-np.pi / 2., np.pi / 2., 300)
    mean_phis = (phis[1:] + phis[:-1]) / 2.
    diff_phis = np.diff(phis)

    thetas = np.linspace(0., np.pi, 300)
    mean_thetas = (thetas[1:] + thetas[:-1]) / 2.
    diff_thetas = np.diff(thetas)

    thetas_grid, phi_grid = np.meshgrid(mean_thetas, mean_phis)

    # Which of these are on the planet's dayside?
    index = good_phi_matrix(phi_phase, inclination, phi_grid, thetas_grid)
    # Calc integral
    integral_terms = np.outer(np.cos(mean_phis) * diff_phis, np.sin(mean_thetas) ** 2. * diff_thetas)
    return np.sum(integral_terms[index]) / np.pi


'''
#print('p',calc_visible_disc_fraction_inc(0.,90.))

inclination = np.linspace(0,90,10)

for inc in inclination:
    print(inc)
    vis_disc_fraction_i = np.zeros_like(phi_phase)
    for i in range(int(len(phi_phase))):
        vis_disc_fraction_i[i] = calc_visible_disc_fraction_inc(phi_phase[i], inc)

    plt.plot(phi_phase/np.pi, vis_disc_fraction_i,':',color='green')

plt.show()

############################################################
# Plot planetary disk as fct of phase and inclination
############################################################

phi_phase = np.linspace(-np.pi/4.,np.pi/4.,10)

def good_phi_matrix(phi_phase, inclination, phi_grid, thetas_grid):

    # Calculate the position of the substellar point, rotate it about the y-axis for the inclination...
    turn = (90.-inclination)/90.*np.pi/2.
    subs_vec = np.array([np.cos(turn)*np.cos(phi_phase),np.sin(phi_phase),-np.sin(turn)*np.cos(phi_phase)])

    # Prepare position vectors from arrays, tp calculate normal vectors at positions where we calculate the area
    x = np.cos(phi_grid)*np.sin(thetas_grid)
    y = np.sin(phi_grid)*np.sin(thetas_grid)
    z = np.cos(thetas_grid)

    vecs = np.zeros(int(len(phi_grid[:,0]))*int(len(phi_grid[0,:]))*3).reshape(int(len(phi_grid[:,0])),
    int(len(thetas_grid[0,:])),3)
    vecs[:,:,0] = x
    vecs[:,:,1] = y
    vecs[:,:,2] = z

    # On the dayside if angular distance to substellar point is smaller than pi/2.
    scalar = np.sum(vecs*subs_vec,axis=2)
    index = scalar > 0.
    return index, y, z

def plot_visible_disc_inc(phi_phase, inclination):

    # Define coordinates, and coordinate distances, over which to take the area integral
    phis = np.linspace(-np.pi/2.,np.pi/2.,300)
    mean_phis = (phis[1:]+phis[:-1])/2.
    diff_phis = np.diff(phis)
    
    thetas = np.linspace(0.,np.pi,300)
    mean_thetas = (thetas[1:]+thetas[:-1])/2.
    diff_thetas = np.diff(thetas)

    thetas_grid, phi_grid = np.meshgrid(mean_thetas,mean_phis)

    # Which of these are on the planet's dayside?
    index, x , y = good_phi_matrix(phi_phase, inclination, phi_grid, thetas_grid)

    plt.plot(x[index].flatten(), y[index].flatten(), '.', color='white',rasterized=True)
    plt.plot(x[~index].flatten(), y[~index].flatten(), '.', color='black',rasterized=True)

for p in phi_phase:

    print(p/2./np.pi)
    plot_visible_disc_inc(p, 67.7)
    plt.xlim([-1.1,1.1])
    plt.ylim([-1.1,1.1])

    circ_corr = np.linspace(0., 2.*np.pi, 300)
    x = np.cos(circ_corr)
    y = np.sin(circ_corr)
    plt.plot(x,y)

    plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    #plt.gca().get_xaxis().set_visible(False)
    #plt.gca().get_xaxis().set_visible(False)
    #plt.axes(frameon=False)
    plt.axis('off')
    plt.gcf().set_size_inches(4.6, 4.6)
    plt.savefig('phase_plots/phase_{0: 4.2F}.pdf'.format(p),transparent=True)
    plt.show()

'''
