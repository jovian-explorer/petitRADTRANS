import numpy as np
import glob
import seaborn as sns
import corner
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.stats import binned_statistic
from petitRADTRANS import nat_cst as nc
from .data import Data

def plot_specs(fig, ax, path, name, nsample, color1, color2, zorder, rebin_val = None):
    # TODO write generic plotting functions rather than copy pasting code.
    specs = sorted([f for f in glob.glob(path+'/' + name + '*.dat')])
    wlen = np.genfromtxt(specs[0])[:,0]
    if rebin_val != None:
        wlen = nc.running_mean(wlen, rebin_val)[::rebin_val]
    npoints = int(len(wlen))
    spectra= np.zeros((nsample,npoints))
    for i_s in range(nsample):
        if rebin_val != None:
            npoints = int(len(wlen))
            spectra[i_s, :]= nc.running_mean(np.genfromtxt(specs[i_s])[:,1], \
                                                rebin_val)[::rebin_val]
        else:
            wlen = np.genfromtxt(specs[i_s])[:,0]
            npoints = int(len(wlen))
            for i_s in range(nsample):
                spectra[i_s, :] = np.genfromtxt(specs[i_s])[:,1]

    sort_spec = np.sort(spectra, axis = 0)
    # 3 sigma
    if int(nsample*0.02275) > 1:
        ax.fill_between(wlen, \
                        y1 = sort_spec[int(nsample*0.02275), :], \
                        y2 = sort_spec[int(nsample*(1.-0.02275)), :], \
                        color = color1, zorder = zorder*2)
    # 1 sigma
    ax.fill_between(wlen, \
                      y1 = sort_spec[int(nsample*0.16), :], \
                      y2 = sort_spec[int(nsample*0.84), :], \
                      color = color2, zorder = zorder*2+1)
    return fig,ax

def plot_data(fig,ax,data,resolution = None, scaling = 1.0):
    scale = 1.0
    if data.scale:
        scale = data.scale_factor
    if not data.photometry:
        try:
            # Sometimes this fails, I'm not super sure why.
            resolution_data = np.mean(data.wlen[1:]/np.diff(data.wlen))
            ratio = resolution_data / resolution
            if int(ratio) > 1:
                flux,edges,_ = binned_statistic(data.wlen,data.flux,'mean',data.wlen.shape[0]/ratio)
                error,_,_ = binned_statistic(data.wlen,data.flux_error,\
                                            'mean',data.wlen.shape[0]/ratio)/np.sqrt(ratio)
                wlen = np.array([(edges[i]+edges[i+1])/2.0 for i in range(edges.shape[0]-1)])
            else:
                wlen = data.wlen
                error = data.flux_error
                flux = data.flux
        except:
            wlen = data.wlen
            error = data.flux_error
            flux = data.flux
    else:
        wlen = np.mean(data.width_photometry)
        flux = data.flux
        error = data.flux_error
        wlen_bins = data.wlen_bins
    marker = 'o'
    if data.photometry:
        marker = 's'
    if not data.photometry:
        ax.errorbar(wlen, \
            flux * scaling * scale, \
            yerr = error * scaling *scale, \
            marker=marker, markeredgecolor='k', linewidth = 0, elinewidth = 2, \
            label = data.name, zorder =10, alpha = 0.9,)
    else:
        ax.errorbar(wlen, \
                    flux * scaling * scale, \
                    yerr = error * scaling *scale, \
                    xerr = data.wlen_bins/2., linewidth = 0, elinewidth = 2, \
                    marker=marker, markeredgecolor='k', color = 'grey', zorder = 10, \
                    label = None, alpha = 0.6)
    return fig,ax

def contour_corner(sampledict, \
                parameter_names, \
                output_file, \
                parameter_ranges = None, \
                parameter_plot_indices = None, \
                true_values = None, \
                short_name = None,
                legend = False,
                prt_plot_style = True,
                plot_best_fit = False,
                **kwargs):
    """
    Use the corner package to plot the posterior distributions produced by pymultinest.

    Args:
        sampledict : dict
            A dictionary of samples, each sample has shape (N_Samples,N_params). The keys of the
            dictionary correspond to the names of each retrieval, and are the prefixes to the
            post_equal_weights.dat files. These are passed as arguments to retrieve.py.
            By default, this is only the current retrieval, and plots the posteriors for a single
            retrieval. If multiple names are passed, they are overplotted on the same figure.
        parameter_names : dict
            A dictionary with keys for each retrieval name, as in sampledict. Each value of the
            dictionary is the names of the parameters to beplotted, as set in the
            run_definition file.
        output_file : str
            Output file name
        parameter_ranges : dict
            A dictionary with keys for each retrieval name as in sampledict. Each value
            contains the ranges of parameters that have a range set with corner_range in the
            parameter class. Otherwise the range is +/- 4 sigma
        parameter_plot_indicies : dict
            A dictionary with keys for each retrieval name as in sampledict. Each value
            contains the indices of the sample to plot, as set by the plot_in_corner
            parameter of the parameter class
        true_values : dict
            A dictionary with keys for each retrieval name as in sampledict. Each value
            contains the known values of the parameters.
        short_name : dict
            A dictionary with keys for each retrieval name as in sampledict. Each value
            contains the names to be plotted in the corner plot legend. If non, uses the
            retrieval names used as keys for sampledict
        legend : bool
            Turn the legend on or off
        prt_plot_style : bool
            Use the prt plot style, changes the colour scheme and fonts to match the rest of
            the prt plots.
        kwargs : dict
            Each kwarg can be one of the kwargs used in corner.corner. These can be used to adjust
            the title_kwargs,label_kwargs,hist_kwargs, hist2d_kawargs or the contour kwargs. Each
            kwarg must be a dictionary with the arguments as keys and values as the values.
    """
    import matplotlib as mpl
    if prt_plot_style:
        mpl.rcParams.update(mpl.rcParamsDefault)
        font = {'family' : 'serif'}
        xtick = {'top' : True,
                 'bottom' : True,
                 'direction' : 'in'}

        ytick = {'left' : True,
                 'right' : True,
                 'direction' : 'in'}
        xmin = {'visible' : True}
        ymin = {'visible' : True}
        mpl.rc('xtick',**xtick)
        mpl.rc('xtick.minor',**xmin)
        mpl.rc('ytick',**ytick)
        mpl.rc('ytick.minor',**ymin)
        mpl.rc('font', **font)

        color_list = ['#009FB8','#FF695C', '#70FF92',  '#FFBB33', '#6171FF', "#FF1F69", "#52AC25", '#E574FF', "#FF261D", "#B429FF" ]
    else:
        color_list = [None,None,None,None,None] #TODO get the default cmap
        mpl.rcParams.update(mpl.rcParamsDefault)

        #from .plot_style import prt_colours
    #color_list = prt_colours
    N_samples = []
    range_list = []
    handles = []
    count = 0
    for key,samples in sampledict.items():
        if prt_plot_style and count > len(color_list):
            print("Not enough colors to continue plotting. Please add to the list.")
            print("Outputting first " + str(count) + " retrievals.")
            break
        dimensions = len(parameter_plot_indices[key])
        N_samples = len(samples)
        S = N_samples
        try:
            if parameter_plot_indices[key] == None:
                parameter_plot_indices = {}
                parameter_plot_indices[key] = np.linspace(0, len(parameter_names[key])-1, \
                                            len(parameter_names[key])-1).astype('int')
        except:
            pass

        data_list = []
        labels_list = []
        best_fit = None
        if plot_best_fit:
            best_fit = []
            best_fit_ind = np.argmax(samples[:,-1])
            for i in parameter_plot_indices[key]:
                best_fit.append(samples[best_fit_ind][i])
        for i in parameter_plot_indices[key]:
            data_list.append(samples[len(samples)-S:,i])
            labels_list.append(parameter_names[key][i])
            if parameter_ranges[key][i] == None:
                range_mean = np.mean(samples[len(samples)-S:,i])
                range_std = np.std(samples[len(samples)-S:,i])
                low = range_mean-4*range_std
                high = range_mean+4*range_std
                if count > 0:
                    if low > range_list[i][0]:
                        low = range_list[i][0]
                    if high < range_list[i][1]:
                        high = range_list[i][1]
                    range_take = (low,high)
                    range_list[i] = range_take
                else:
                    range_list.append((low,high))
            else:
                range_take = (parameter_ranges[key][i][0],parameter_ranges[key][i][1])
                range_list.append(range_take)
        try:
            truths_list = []
            for i in parameter_plot_indices[key]:
                truths_list.append(true_values[key][i])
        except:
            pass
        #fig = plt.figure(figsize = (60,60),dpi=80)
        label_kwargs = None
        title_kwargs = None
        hist_kwargs = None
        hist2d_kwargs = None
        contour_kwargs = None
        if "label_kwargs" in kwargs.keys():
            label_kwargs = kwargs["label_kwargs"]
        if "title_kwargs" in kwargs.keys():
            title_kwargs = kwargs["title_kwargs"]
        if "hist_kwargs" in kwargs.keys():
            hist_kwargs = kwargs["hist_kwargs"]
        if "hist2d_kwargs" in kwargs.keys():
            hist2d_kwargs = kwargs["hist2d_kwargs"]
        if "contour_kwargs" in kwargs.keys():
            contour_kwargs = kwargs["contour_kwargs"]

        if count == 0:
            print(len(data_list))
            fig = corner.corner(np.array(data_list).T,
                                #fig = fig,
                                smooth=True,
                                title_fmt = ".2f",
                                show_titles = True,
                                title_kwargs = title_kwargs,
                                labels = labels_list,
                                label_kwargs = label_kwargs,
                                range = range_list,
                                color = color_list[count],
                                quantiles=[0.16, 0.5, 0.84],
                                hist2d_kwargs = hist2d_kwargs,
                                plot_contours = True,
                                contour_kwargs = contour_kwargs,
                                hist_kwargs = hist_kwargs,
                                levels=[1-np.exp(-0.5),1-np.exp(-2),1-np.exp(-4.5)],
                                truths=best_fit,
                                truth_color='k'
                                )
            count +=1
        else:
            corner.corner(np.array(data_list).T,
                          fig = fig,
                          smooth=True,
                          title_fmt = ".2f",
                          title_kwargs = title_kwargs,
                          show_titles = True,
                          range = range_list,
                          color = color_list[count],
                          labels = labels_list,
                          label_kwargs = label_kwargs,
                          hist2d_kwargs = hist2d_kwargs,
                          plot_contours = True,
                          contour_kwargs = contour_kwargs,
                          hist_kwargs = hist_kwargs,
                          levels=[1-np.exp(-0.5),1-np.exp(-2),1-np.exp(-4.5)],
                          truths=best_fit,
                          truth_color='red'
                          )
            count += 1
        #if dimensions == 1:
        #    plt.tight_layout(h_pad=0, w_pad=0)
        if short_name is None:
            label = key
        else:
            label = short_name[key]
        handles.append(Line2D([0], [0], marker = 'o',color=color_list[count], label = label,markersize = 15))
    #fig.subplots_adjust( wspace=0.005, hspace=0.005)
    if legend:
        fig.get_axes()[2].legend(handles = handles,
                                 loc = 'upper right' )
    plt.savefig(output_file,dpi=300, bbox_inches='tight')
    if prt_plot_style:
        import petitRADTRANS.retrieval.plot_style
    return fig




def nice_corner(samples, \
                parameter_names, \
                output_file, \
                N_samples = None, \
                parameter_ranges = None, \
                parameter_plot_indices = None, \
                true_values = None, \
                max_val_ratio = None):
    """
    Paul's custom hex grid corner plots.
    Won't work with sampledict setup in retrieve.py!
    """
    font = {'family' : 'serif',
        'weight' : 'normal',
            'size'   : int(23*5./len(parameter_plot_indices))}

    plt.rc('font', **font)
    plt.rc('text', usetex=True)


    if N_samples != None:
        S = N_samples
    else:
        S = len(samples)

    try:
        if parameter_plot_indices == None:
            parameter_plot_indices = np.linspace(0, len(parameter_names)-1, \
                                        len(parameter_names)-1).astype('int')
    except:
        pass

    if max_val_ratio == None:
        max_val_ratio = 5.

    data_list = []
    labels_list = []
    range_list = []

    for i in parameter_plot_indices:

        data_list.append(samples[len(samples)-S:,i])
        labels_list.append(parameter_names[i])

        try:
            if parameter_ranges[i] == None:
                range_mean = np.mean(samples[len(samples)-S:,i])
                range_std = np.std(samples[len(samples)-S:,i])
                range_take = (range_mean-4*range_std, range_mean+4*range_std)
                range_list.append(range_take)
            else:
                range_list.append(parameter_ranges[i])
        except:
            range_mean = np.mean(samples[len(samples)-S:,i])
            range_std = np.std(samples[len(samples)-S:,i])
            range_take = (range_mean-4*range_std, range_mean+4*range_std)
            range_list.append(range_take)

    try:
        truths_list = []
        for i in parameter_plot_indices:
            truths_list.append(true_values[i])
    except:
        pass

    dimensions = len(parameter_plot_indices)

    # Set up the matplotlib figure
    f, axes = plt.subplots(dimensions, dimensions, figsize=(13, 13), \
                           sharex=False, sharey=False)
    i_col = 0
    i_lin = 0

    try:
        ax_array = axes.flat
    except:
        ax_array = [plt.gca()]
    #print(len(axes).flat)
    for ax in ax_array:

        #print(i_col, i_lin, dimensions, len(axes.flat))
        if i_col < i_lin:
            x = cp.copy(data_list[i_col])
            y = cp.copy(data_list[i_lin])
            indexx = (x >= range_list[i_col][0]) & \
                (x <= range_list[i_col][1])
            indexy = (y >= range_list[i_lin][0]) & \
                (y <= range_list[i_lin][1])

            x = x[indexx & indexy]
            y = y[indexx & indexy]

            gridsize = 22
            ax.hexbin(x, y, cmap='bone_r', gridsize=gridsize, \
                          vmax = int(max_val_ratio*S/gridsize**2.), \
                          rasterized=True)

            ax.set_xlim([range_list[i_col][0], range_list[i_col][1]])
            ax.set_ylim([range_list[i_lin][0], range_list[i_lin][1]])

            try:
                ax.axhline(truths_list[i_lin], color = 'red', \
                               linestyle = '--', linewidth = 2.5)
                ax.axvline(truths_list[i_col], color = 'red', \
                               linestyle = '--', linewidth = 2.5)
            except:
                pass

            if i_col > 0:
                ax.get_yaxis().set_visible(False)

        elif i_col == i_lin:

            med = np.median(data_list[i_col])
            up = str(np.round(np.percentile(data_list[i_col], 84) - med,2))
            do = str(np.round(med - np.percentile(data_list[i_col], 16),2))
            med = str(np.round(med,2))
            med = med.split('.')[0]+'.'+med.split('.')[1].ljust(2, '0')
            up = up.split('.')[0]+'.'+up.split('.')[1].ljust(2, '0')
            do = do.split('.')[0]+'.'+do.split('.')[1].ljust(2, '0')

            ax.set_title(med+r'$^{+'+up+'}_{-'+do+'}$', \
                             fontdict=font, fontsize = \
                             int(22*5./len(parameter_plot_indices)))

            import copy as cp
            use_data = cp.copy(data_list[i_col])
            index = (use_data >= range_list[i_col][0]) & \
                (use_data <= range_list[i_col][1])
            use_data = use_data[index]
            sns.distplot(use_data, bins=22, kde=False, \
                             rug=False, ax=ax, color = 'gray')

            ax.set_xlim([range_list[i_col][0], range_list[i_col][1]])
            ax.get_yaxis().set_visible(False)
            try:
                ax.axvline(truths_list[i_col], color = 'red', \
                                   linestyle = '--', linewidth = 2.5)
            except:
                pass

            ax.axvline(float(med)+float(up), color = 'black', \
                           linestyle = ':', linewidth=1.0)
            ax.axvline(float(med), color = 'black', \
                           linestyle = ':', linewidth=1.0)
            ax.axvline(float(med)-float(do), color = 'black', \
                           linestyle = ':', linewidth=1.0)

        else:
            ax.axis('off')

        labels = ax.yaxis.get_ticklabels(which='both')
        #print(labels)
        #labels[-1].set_visible(False)

        i_col += 1
        if i_col%dimensions == 0:
            i_col = 0
            i_lin += 1

    for i_col in range(dimensions):
        for i_lin in range(dimensions):
            try:
                plt.sca(axes[i_lin, i_col])
            except:
                pass
            range_use = np.linspace(range_list[i_col][0], \
                                        range_list[i_col][1], 5)[1:-1]

            labels_use = []
            for r in range_use:
                labels_use.append(str(np.round(r,1)))
            plt.xticks(range_use[::2], labels_use[::2])
            plt.xlabel(labels_list[i_col])

    for i_lin in range(dimensions):
        try:
            plt.sca(axes[i_lin,0])
        except:
            pass
        plt.ylabel(labels_list[i_lin])

    plt.subplots_adjust(wspace = 0, hspace = 0)
    if dimensions == 1:
        plt.tight_layout(h_pad=0, w_pad=0)
    plt.savefig(output_file)
    #plt.show()
    #plt.clf()
